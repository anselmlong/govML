"""Render a findings report in Markdown from one or more reviewed stores.

    python -m src.atlas.report --store atlas/housing_col --store atlas/all --intro docs/findings-intro.md \
        --out docs/findings-v0.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _short(label: str, n: int = 90) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _stats(c: dict) -> str:
    lag = f"x leads by {c['lag']} {c['freq']}" if c["kind"] == "lead" else "same period"
    return (f"{lag}; r train {c['r_train']:+.2f}, r test {c['r_test']:+.2f} (n {c['n_test']}), "
            f"ex-2020-22 {c['r_ex_covid']:+.2f}, BH p {c['p_test_adj']:.2g}; span {c['span']}")


def render_store(store: Path, top_breaks: int) -> list[str]:
    coverage = json.loads((store / "coverage.json").read_text(encoding="utf-8"))
    candidates = _load(store / "candidates.json")
    kept = _load(store / "findings.json")
    dropped = _load(store / "dropped.json")
    breaks = _load(store / "breaks.json")
    notes = {(n["series_id"], n["period"]): n for n in _load(store / "break_notes.json")}
    reviewed = len(kept) + len(dropped)
    out = [
        f"## Store `{store}`", "",
        f"{coverage['matched_tables']} of {coverage['tables']} tables became {coverage['series']:,} series. "
        f"The statistical gate passed {len(candidates)} candidates; the top {reviewed} were reviewed and "
        f"{len(kept)} kept. {len(breaks)} series have a break above the 1% sup-F critical value.", "",
        "### Kept findings", "",
    ]
    if not kept:
        out += ["None.", ""]
    for f in kept:
        out += [
            f"**{f['headline']}**", "",
            f"- x: {_short(f['x_label'], 140)}",
            f"- y: {_short(f['y_label'], 140)}",
            f"- evidence: {_stats(f)}",
            f"- explanation: {f['explanation']}",
            *(f"- caveat: {c}" for c in f["caveats"]),
            *(f"- idea ({i['kind']}): {i['title']}. {i['description']}" for i in f["ideas"]),
            f"- reviewer: {f.get('rationale', '')}", "",
        ]
    out += ["### Reviewed and dropped", "",
            "| x | y | evidence | reason | reviewer's rationale |", "|---|---|---|---|---|"]
    for d in dropped:
        out.append(f"| {_cell(_short(d['x_label'], 70))} | {_cell(_short(d['y_label'], 70))} | {_cell(_stats(d))} "
                   f"| {d['drop_reason']} | {_cell(d.get('rationale') or d.get('review_error', ''))} |")
    out += ["", f"### Strongest breaks (top {top_breaks} by sup-F)", "",
            "| series | period | sup-F | shift (sd) | note |", "|---|---|---|---|---|"]
    for b in sorted(breaks, key=lambda b: -b["supF"])[:top_breaks]:
        n = notes.get((b["series_id"], b["period"]))
        note = f"{n['confidence']}: {n['note']}" if n else ""
        out.append(f"| {_cell(_short(b['label'], 80))} | {b['period']} | {b['supF']:.1f} | {b['delta']:+.2f} | {_cell(note)} |")
    return out + [""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, action="append", required=True)
    ap.add_argument("--intro", type=Path)
    ap.add_argument("--top-breaks", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    lines = [args.intro.read_text(encoding="utf-8").rstrip(), ""] if args.intro else []
    for store in args.store:
        lines += render_store(store, args.top_breaks)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
