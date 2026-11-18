"""Flip the fetch cache into the series store and write a coverage report.

    python -m src.atlas.flip --all
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

CACHE_ROWS = 10_000
DEFAULT_OUT = Path("atlas")


def dataset_names(catalog: Path) -> dict[str, str]:
    if not catalog.exists():
        return {}
    with sqlite3.connect(catalog) as conn:
        return dict(conn.execute("SELECT dataset_id, name FROM datasets"))


def cached_ids(cache_dir: Path) -> list[str]:
    suffix = f"_{CACHE_ROWS}.parquet"
    return sorted(p.name[: -len(suffix)] for p in cache_dir.glob(f"*{suffix}"))


def flip_cache(ids: list[str], cache_dir: Path, names: dict[str, str]) -> Iterator[tuple[Matched | Unmatched, bool]]:
    """Yield `(result, truncated)` per dataset. `truncated` marks a cache file that hit the
    row cap, so long tables in it may be missing their later rows."""
    for dataset_id in ids:
        path = cache_dir / f"{dataset_id}_{CACHE_ROWS}.parquet"
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
        yield result, len(frame) >= CACHE_ROWS


def build_report(results: list[tuple[Matched | Unmatched, bool]], names: dict[str, str]) -> dict:
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
        f"{report['matched_tables_possibly_truncated']} matched tables hit the {CACHE_ROWS:,}-row cache cap "
        "and may be missing later rows.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="flip every table in the cache")
    which.add_argument("--datasets", help="comma-separated dataset ids")
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--catalog", type=Path, default=Path("catalog.db"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    ids = cached_ids(args.cache_dir) if args.all else [d.strip() for d in args.datasets.split(",") if d.strip()]
    names = dataset_names(args.catalog)
    results = list(flip_cache(ids, args.cache_dir, names))
    n_series, n_points = write_store((s for r, _ in results if isinstance(r, Matched) for s in r.series), args.out)
    report = build_report(results, names)
    (args.out / "coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "coverage.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote {n_series:,} series and {n_points:,} points to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
