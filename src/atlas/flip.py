"""Flip the fetch cache into the series store and write a coverage report.

    python -m src.atlas.flip --all
    python -m src.atlas.flip --theme themes/housing_col.json
    python -m src.atlas.flip --datasets d_056a886a7c7f7fc87f0cd875f426a653,d_01966fdb498fa7fa2863635e25539693
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from .readers import Matched, Table, Unmatched, flip
from .store import write_store

DEFAULT_OUT = Path("atlas")


def dataset_names(catalog: Path) -> dict[str, str]:
    if not catalog.exists():
        return {}
    with sqlite3.connect(catalog) as conn:
        return dict(conn.execute("SELECT dataset_id, name FROM datasets"))


def cache_files(cache_dir: Path) -> dict[str, list[tuple[int | None, Path]]]:
    """Every parquet in the cache by dataset id, best first: the uncapped `_all` file, then
    the largest row cap. `_<cap>` is the `max_rows` the file was fetched with."""
    found: dict[str, list[tuple[int | None, Path]]] = {}
    for path in cache_dir.glob("d_*_*.parquet"):
        dataset_id, _, key = path.stem.rpartition("_")
        found.setdefault(dataset_id, []).append((None if key == "all" else int(key), path))
    return {d: sorted(files, key=lambda f: (f[0] is not None, -(f[0] or 0))) for d, files in sorted(found.items())}


def flip_cache(
    ids: list[str], files: dict[str, list[tuple[int | None, Path]]], names: dict[str, str]
) -> Iterator[tuple[Matched | Unmatched, bool]]:
    """Yield `(result, truncated)` per dataset. `truncated` marks a cache file that hit its
    row cap, so long tables in it may be missing their later rows."""
    for dataset_id in ids:
        if dataset_id not in files:
            yield Unmatched(dataset_id, "not_in_cache"), False
            continue
        cap, path = files[dataset_id][0]
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            yield Unmatched(dataset_id, "unreadable_cache_file", f"{type(exc).__name__}: {exc}"), False
            continue
        table = Table(dataset_id, names.get(dataset_id, dataset_id), frame)
        try:
            result = flip(table)
        except Exception as exc:
            result = Unmatched(dataset_id, "reader_error", f"{type(exc).__name__}: {exc}")
        yield result, cap is not None and len(frame) >= cap


def build_report(
    results: list[tuple[Matched | Unmatched, bool]], names: dict[str, str], shadowed: list[str]
) -> dict:
    matched = [(r, t) for r, t in results if isinstance(r, Matched)]
    unmatched = [r for r, _ in results if isinstance(r, Unmatched)]
    series = [s for r, _ in matched for s in r.series]
    skipped: Counter[str] = Counter()
    for r, _ in matched:
        skipped.update(r.skipped)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tables": len(results),
        "matched_tables": len(matched),
        "unmatched_tables": len(unmatched),
        "matched_by_reader": dict(Counter(r.reader for r, _ in matched)),
        "unmatched_by_reason": dict(Counter(u.reason for u in unmatched).most_common()),
        "series": len(series),
        "series_by_freq": dict(Counter(s.freq for s in series)),
        "series_by_reader": dict(Counter(r.reader for r, _ in matched for _ in r.series)),
        "series_with_unit": sum(s.unit is not None for s in series),
        "points": sum(len(s.points) for s in series),
        "candidate_series_skipped": dict(skipped),
        "matched_tables_possibly_truncated": sum(t for _, t in matched),
        "cache_files_not_flipped": shadowed,
        "unmatched": [
            {"dataset_id": u.dataset_id, "name": names.get(u.dataset_id, ""), "reason": u.reason, "detail": u.detail}
            for u in unmatched
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Atlas flip coverage", "",
        f"{report['matched_tables']} of {report['tables']} tables became series "
        f"({report['series']:,} series, {report['points']:,} points). {report['unmatched_tables']} are unmatched.", "",
        "## Matched tables by reader", "",
        *(f"- {k}: {v}" for k, v in report["matched_by_reader"].items()), "",
        "## Unmatched tables by reason", "",
        *(f"- {k}: {v}" for k, v in report["unmatched_by_reason"].items()), "",
        "## Series by frequency", "",
        *(f"- {k}: {v:,}" for k, v in sorted(report["series_by_freq"].items())), "",
        f"{report['series_with_unit']:,} series carry a unit.", "",
        "## Candidate series skipped inside matched tables", "",
        *(f"- {k}: {v:,}" for k, v in report["candidate_series_skipped"].items()), "",
        f"{report['matched_tables_possibly_truncated']} matched tables hit their cache row cap "
        "and may be missing later rows.",
        f"{len(report['cache_files_not_flipped'])} cache files were not flipped because a better file of the "
        "same dataset was (listed in coverage.json).", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="flip every table in the cache")
    which.add_argument("--datasets", help="comma-separated dataset ids")
    which.add_argument("--theme", type=Path, help="json object keyed by dataset id")
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--catalog", type=Path, default=Path("catalog.db"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    files = cache_files(args.cache_dir)
    if args.all:
        ids = list(files)
    elif args.theme:
        ids = sorted(json.loads(args.theme.read_text(encoding="utf-8")))
    else:
        ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    shadowed = sorted(p.name for d in ids for _, p in files.get(d, [])[1:])
    names = dataset_names(args.catalog)
    results = list(flip_cache(ids, files, names))
    n_series, n_points = write_store((s for r, _ in results if isinstance(r, Matched) for s in r.series), args.out)
    report = build_report(results, names, shadowed)
    (args.out / "coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "coverage.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote {n_series:,} series and {n_points:,} points to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
