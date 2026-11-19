"""Findings engine: co-movement and lead/lag candidates between series of different tables.
Breaks within one series live in `breaks.py`.

    python -m src.atlas.findings --store atlas/housing_col --catalog catalog.db --out atlas/housing_col

Rules and thresholds are documented in docs/atlas-plan.md (phase 3).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .flip import dataset_names
from .series import Series
from .store import read_store

PER_YEAR = {"A": 1, "H": 2, "Q": 4, "M": 12}
COARSENESS = {"A": 0, "H": 1, "Q": 2, "M": 3}


@dataclass(frozen=True)
class Config:
    k: int = 3
    pair_freqs: tuple[str, ...] = ("M", "Q")
    max_lag: dict[str, int] = field(default_factory=lambda: {"A": 2, "Q": 8, "M": 12})
    min_overlap: dict[str, int] = field(default_factory=lambda: {"Q": 32, "M": 60})
    min_window: int = 10
    train_share: float = 0.7
    q: float = 0.05
    min_abs_r: float = 0.3
    dup_corr: float = 0.98
    covid_years: tuple[int, int] = (2020, 2022)
    min_stability: float = 0.5


def period_index(freq: str, period: str) -> int:
    year = int(period[:4])
    if freq == "A":
        return year
    n = int(period[5:]) if freq == "M" else int(period[5])
    return year * PER_YEAR[freq] + n - 1


def index_period(freq: str, i: int) -> str:
    if freq == "A":
        return f"{i:04d}"
    year, n = divmod(i, PER_YEAR[freq])
    return f"{year:04d}-{n + 1:02d}" if freq == "M" else f"{year:04d}{freq}{n + 1}"


@dataclass(frozen=True)
class Grid:
    """Values on consecutive periods of one frequency; `start` is the period index of values[0]."""
    freq: str
    start: int
    values: np.ndarray

    @property
    def end(self) -> int:
        return self.start + self.values.size - 1


def to_grid(s: Series) -> Grid:
    idx = np.array([period_index(s.freq, p) for p in s.points])
    values = np.full(idx[-1] - idx[0] + 1, np.nan)
    values[idx - idx[0]] = list(s.points.values())
    return Grid(s.freq, int(idx[0]), values)


def coarsen(g: Grid, freq: str) -> Grid:
    """Levels to means over a coarser frequency (M to Q, Q to A, ...). A coarse period missing
    any sub-period stays empty, so a partial quarter or year can't bias the mean."""
    k = PER_YEAR[g.freq] // PER_YEAR[freq]
    q0, q1 = g.start // k, g.end // k
    padded = np.full((q1 - q0 + 1) * k, np.nan)
    padded[g.start - q0 * k: g.start - q0 * k + g.values.size] = g.values
    return Grid(freq, q0, padded.reshape(-1, k).mean(axis=1))


HEADLINE = re.compile(r"^(total|overall|all|whole)\b|\boverall items\b", re.I)
EXCLUDED_MEASURES = {"median_block"}


def _measure_rank(measure: str | None) -> int:
    """Records tables: a median price or rent first, then the transaction count."""
    m = measure or ""
    if m.startswith("median_"):
        return 0 if ("price" in m or "rent" in m) else 2
    return 1 if m == "count" else 0


def representatives(series: list[Series], k: int) -> list[Series]:
    """At most k series per (dataset, frequency): headline rows first, then shallower dims,
    then table order."""
    groups: dict[tuple[str, str], list[tuple[tuple, Series]]] = {}
    for order, s in enumerate(series):
        if s.measure in EXCLUDED_MEASURES:
            continue
        key = (not _is_headline(s), len(s.dims), _measure_rank(s.measure), order)
        groups.setdefault((s.dataset_id, s.freq), []).append((key, s))
    reps = []
    for members in groups.values():
        reps.extend(s for _, s in sorted(members, key=lambda m: m[0])[:k])
    return reps


_SAME_TABLE_STRIP = [
    r"\([^)]*(=|based on)[^)]*\)",
    r"\bbase (year|quarter|period)\b[\s\w-]*?=\s*100",
    r"\b\d{4}\s+as base year\b",
    r"\b[\w-]+\s*=\s*100\b",
    r"\b(in chained volume terms|at current prices|at constant (\d{4} )?(market )?prices)\b",
    r"\b(annual|quarterly|monthly|half-yearly|half yearly|yearly)\b",
]
_COMMODITY = re.compile(r"\(\d-digit level\)")
_BY_CLAUSE = r"\bby\b[^,()]*"
_BREAKDOWN_STRIP = [
    _COMMODITY.pattern,
    r"\bby (commodity (section|division|group)|type of property|type|market segment|locality"
    r"|geographical distribution|(age and )?sex)\b",
    r"\bin (the )?(whole of singapore|core central region|rest of central region|outside central region)\b",
]


def table_key(name: str) -> str:
    """A dataset name with frequency words, base-year phrases, '(Based on ... Date)', the price
    basis and a breakdown ('By Commodity Group (3-Digit Level)', 'by Market Segment',
    'in Core Central Region') stripped. Breakdowns are listed, not matched as any 'by ...':
    in 'Resident Households by Monthly Household Employment Income' the clause is the subject. Tables that share a key are one publication sliced
    or based differently, so pairs between them are a series against its own copy or part."""
    s = name.lower()
    for pattern in _SAME_TABLE_STRIP + _BREAKDOWN_STRIP:
        s = re.sub(pattern, " ", s)
    return re.sub(r"^(average|total) ", "", _norm(s))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_headline(s: Series) -> bool:
    return not s.dims or any(HEADLINE.search(v) for v in s.dims.values())


def same_row(a: Series, b: Series, names: dict[str, str]) -> bool:
    """Rule 2, by row. Two SITC commodity price indices ('(1-Digit Level)' tables: import,
    export, domestic supply, manufactured) price the same baskets, and domestic supply is built
    from the others. Otherwise, the same non-headline row ('Core Central Region') in two
    tables broken down the same way ('By Locality') is one slice measured twice."""
    na, nb = names.get(a.dataset_id, "").lower(), names.get(b.dataset_id, "").lower()
    if _COMMODITY.search(na) and _COMMODITY.search(nb):
        return True
    if _is_headline(a) or _is_headline(b):
        return False
    row = lambda s: _norm(" ".join(v for v in s.dims.values() if v != "na"))
    clause = lambda n: _norm(" ".join(m.group(0) for m in re.finditer(_BY_CLAUSE, n)))
    return bool(clause(na)) and clause(na) == clause(nb) and row(a) == row(b)


def one_dataset_per_table(reps: list[Series], names: dict[str, str]) -> tuple[list[Series], int]:
    """Rule 2, by name: keep one dataset per table key: the finest frequency, then one with a
    headline row, then the most points. Returns the kept reps and how many were dropped."""
    by_dataset: dict[str, list[Series]] = {}
    for s in reps:
        by_dataset.setdefault(s.dataset_id, []).append(s)
    families: dict[str, list[str]] = {}
    for d in by_dataset:
        families.setdefault(table_key(names.get(d, d)), []).append(d)

    def rank(d: str) -> tuple:
        members = by_dataset[d]
        finest = max(COARSENESS[s.freq] for s in members)
        return (-finest, not any(_is_headline(s) for s in members), -sum(len(s.points) for s in members), d)

    keep = {min(ds, key=rank) for ds in families.values()}
    kept = [s for s in reps if s.dataset_id in keep]
    return kept, len(reps) - len(kept)


def name_overlap(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


@dataclass(frozen=True)
class View:
    """A transformed series at one frequency: values[i] is the change into period start + i."""
    series: Series
    freq: str
    start: int
    values: np.ndarray
    transform: str


def make_view(s: Series, freq: str) -> View:
    g = to_grid(s)
    if g.freq != freq:
        g = coarsen(g, freq)
    changes, kind = stats.transform(g.values)
    return View(s, freq, g.start + 1, changes, kind)


def _on(v: View, lo: int, size: int) -> np.ndarray:
    out = np.full(size, np.nan)
    out[v.start - lo: v.start - lo + v.values.size] = v.values
    return out


def deseason(x: np.ndarray, season: np.ndarray, fit: np.ndarray, per_year: int) -> np.ndarray:
    """Subtract each month's or quarter's mean, fitted on the positions in `fit` only."""
    out = x.copy()
    for k in range(per_year):
        sel = fit & (season == k) & ~np.isnan(x)
        if sel.any():
            out[season == k] -= x[sel].mean()
    return out


def _lagged(X: np.ndarray, Y: np.ndarray, k: int):
    """Pairs (X[t], Y[t + k]) where both exist, with their positions."""
    n = X.size
    tx = np.arange(0, n - k) if k >= 0 else np.arange(-k, n)
    ty = tx + k
    xs, ys = X[tx], Y[ty]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    return xs[ok], ys[ok], tx[ok], ty[ok]


def _align(vx: View, vy: View) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Both views on one grid from the earlier start to the later end, and the positions
    where both have a value."""
    lo = min(vx.start, vy.start)
    size = max(vx.start + vx.values.size, vy.start + vy.values.size) - lo
    X, Y = _on(vx, lo, size), _on(vy, lo, size)
    return lo, X, Y, np.flatnonzero(~np.isnan(X) & ~np.isnan(Y))


def full_span_r(vx: View, vy: View, min_overlap: int) -> float:
    """Lag-0 correlation of the changes over the whole overlap (rule 2), deseasoned on the
    overlap: season means from a longer history would make two copies of one series differ."""
    lo, X, Y, both = _align(vx, vy)
    if both.size < min_overlap:
        return float("nan")
    if vx.freq in ("M", "Q"):
        pos = np.arange(X.size)
        season, overlap = (lo + pos) % PER_YEAR[vx.freq], (pos >= both[0]) & (pos <= both[-1])
        X = deseason(X, season, overlap, PER_YEAR[vx.freq])
        Y = deseason(Y, season, overlap, PER_YEAR[vx.freq])
    return stats.pearson(X[both], Y[both])


def evaluate_pair(vx: View, vy: View, cfg: Config) -> dict | str:
    """Lag chosen on the first 70% of the overlap, measured once on the rest. Returns the
    measurements, or the name of the rule that dropped the pair before testing."""
    freq = vx.freq
    lo, X, Y, both = _align(vx, vy)
    size = X.size
    if both.size < cfg.min_overlap[freq]:
        return "rule3_overlap"
    split = both[math.ceil(cfg.train_share * both.size) - 1]
    pos = np.arange(size)
    fit = (pos >= both[0]) & (pos <= split)
    if freq in ("M", "Q"):
        season = (lo + pos) % PER_YEAR[freq]
        X = deseason(X, season, fit, PER_YEAR[freq])
        Y = deseason(Y, season, fit, PER_YEAR[freq])

    L = cfg.max_lag[freq]
    best = None
    for k in range(-L, L + 1):
        xs, ys, tx, ty = _lagged(X, Y, k)
        train = np.maximum(tx, ty) <= split
        if train.sum() < cfg.min_window:
            continue
        r = stats.pearson(xs[train], ys[train])
        if np.isfinite(r) and (best is None or (-abs(r), abs(k), -k) < (-abs(best[1]), abs(best[0]), -best[0])):
            best = (k, r)
    if best is None:
        return "rule3_overlap"
    k, r_train = best
    xs, ys, tx, ty = _lagged(X, Y, k)
    train, test = np.maximum(tx, ty) <= split, np.minimum(tx, ty) > split
    n_test = int(test.sum())
    if n_test < cfg.min_window:
        return "rule3_overlap"

    m = max(1, int(fit.sum()) // 4)
    acf_x, acf_y = stats.acf(np.where(fit, X, np.nan), m), stats.acf(np.where(fit, Y, np.nan), m)
    n_eff = stats.bartlett_neff(n_test, acf_x, acf_y)
    r_test = stats.pearson(xs[test], ys[test])
    r_spear = stats.spearman(xs[test], ys[test])

    y0, y1 = cfg.covid_years
    c_lo, c_hi = y0 * PER_YEAR[freq] - lo, (y1 + 1) * PER_YEAR[freq] - 1 - lo
    clean = ~((tx >= c_lo) & (tx <= c_hi)) & ~((ty >= c_lo) & (ty <= c_hi))
    n_ex = int((test & clean).sum())
    r_train_ex = stats.pearson(xs[train & clean], ys[train & clean])
    r_ex = stats.pearson(xs[test & clean], ys[test & clean]) if n_ex >= cfg.min_window else float("nan")
    n_eff_ex = stats.bartlett_neff(n_ex, acf_x, acf_y) if n_ex >= 3 else 3.0

    x, y = (vx, vy) if k >= 0 else (vy, vx)
    return {
        "x": x.series.series_id, "y": y.series.series_id, "freq": freq, "lag": abs(k),
        "kind": "lead" if k else "comove",
        "n_train": int(train.sum()), "n_test": n_test, "n_eff": n_eff,
        "r_train": r_train, "r_test": r_test, "p_test": stats.p_value(r_test, n_eff),
        "r_test_spearman": r_spear, "p_spearman": stats.p_value(r_spear, n_eff),
        "r_train_ex_covid": r_train_ex, "r_ex_covid": r_ex, "n_ex_covid": n_ex,
        "p_ex_covid": stats.p_value(r_ex, n_eff_ex),
        "span": f"{index_period(freq, lo + both[0])}..{index_period(freq, lo + both[-1])}",
        "transform": f"{x.transform}/{y.transform}" if x.transform != y.transform else x.transform,
    }


@dataclass
class Findings:
    candidates: pd.DataFrame
    tested: pd.DataFrame
    report: dict


def _passes(r: float, r_train: float, p: float, cut: float, cfg: Config) -> bool:
    return bool(np.isfinite(r) and np.sign(r) == np.sign(r_train) and abs(r) >= cfg.min_abs_r and p <= cut)


def drop_duplicates(views: dict[tuple[str, str], View], cfg: Config) -> set[str]:
    """Rule 2, by data: series in different datasets whose full-span changes correlate above
    cfg.dup_corr are one quantity. Keep the one with the most points per connected group."""
    parent: dict[str, str] = {}

    def root(a: str) -> str:
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    series = {v.series.series_id: v.series for v in views.values()}
    for (a, fa), (b, fb) in combinations(sorted(views), 2):
        sa, sb = series[a], series[b]
        if sa.dataset_id == sb.dataset_id or a == b:
            continue
        f = min(sa.freq, sb.freq, key=COARSENESS.get)
        if fa != f or fb != f:
            continue
        if abs(full_span_r(views[(a, f)], views[(b, f)], cfg.min_overlap[f])) > cfg.dup_corr:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for sid in parent:
        groups.setdefault(root(sid), []).append(sid)
    for r, members in groups.items():
        members.append(r)
    dropped = set()
    for members in groups.values():
        keep = min(set(members), key=lambda i: (-len(series[i].points), i))
        dropped |= set(members) - {keep}
    return dropped


def find(series: list[Series], names: dict[str, str], cfg: Config = Config()) -> Findings:
    all_reps = representatives(series, cfg.k)
    reps, family_dropped = one_dataset_per_table(all_reps, names)
    killed: Counter[str] = Counter()

    pair_reps = [s for s in reps if s.freq in cfg.pair_freqs]
    views: dict[tuple[str, str], View] = {}
    for s in pair_reps:
        for f in cfg.pair_freqs:
            if COARSENESS[f] <= COARSENESS[s.freq]:
                views[(s.series_id, f)] = make_view(s, f)
    flat = {key for key, v in views.items() if stats.is_flat(v.values)}
    duplicates = drop_duplicates({k: v for k, v in views.items() if k not in flat}, cfg)

    def cross_pairs(members: list[Series]) -> list[tuple[Series, Series]]:
        return [(a, b) for a, b in combinations(sorted(members, key=lambda s: s.series_id), 2)
                if a.dataset_id != b.dataset_id]

    before_rule2 = len(cross_pairs([s for s in all_reps if s.freq in cfg.pair_freqs]))
    pair_reps = [s for s in pair_reps if s.series_id not in duplicates]
    pairs = cross_pairs(pair_reps)
    killed["rule2_same_table_or_duplicate"] = before_rule2 - len(pairs)
    rows = []
    for a, b in pairs:
        if same_row(a, b, names):
            killed["rule2_same_row"] += 1
            continue
        f = min(a.freq, b.freq, key=COARSENESS.get)
        if (a.series_id, f) in flat or (b.series_id, f) in flat:
            killed["rule4_flat"] += 1
            continue
        result = evaluate_pair(views[(a.series_id, f)], views[(b.series_id, f)], cfg)
        if isinstance(result, str):
            killed[result] += 1
            continue
        rows.append(result)

    tested = pd.DataFrame(rows)
    F = len(tested)
    report: dict = {
        "representatives": dict(sorted(Counter(s.freq for s in reps).items())),
        "rule2_series_dropped": {"same_table": family_dropped, "duplicate": len(duplicates)},
        "flat_series": len(flat), "cross_dataset_pairs": {"before_rule2": before_rule2, "after": len(pairs)},
        "family_size": F,
    }
    if F:
        tested["p_test_adj"] = stats.bh_adjust(tested["p_test"].to_numpy())
        discoveries = int((tested["p_test_adj"] < cfg.q).sum())
        cut = cfg.q * max(discoveries, 1) / F
        status = []
        for r in tested.itertuples(index=False):
            if r.p_test_adj >= cfg.q:
                status.append("rule4_bh")
            elif np.sign(r.r_train) != np.sign(r.r_test):
                status.append("rule4_sign")
            elif abs(r.r_test) < cfg.min_abs_r:
                status.append("rule4_min_r")
            elif abs(r.r_train) < cfg.min_stability * abs(r.r_test):
                status.append("rule4_episode")
            elif not (_passes(r.r_test_spearman, r.r_train, r.p_spearman, cut, cfg)
                      and np.sign(r.r_train_ex_covid) == np.sign(r.r_train)
                      and _passes(r.r_ex_covid, r.r_train, r.p_ex_covid, cut, cfg)):
                status.append("rule5_covid")
            else:
                status.append("survived")
        tested["status"] = status
        killed.update(s for s in status if s != "survived")
        report.update({
            "median_n_test": float(tested["n_test"].median()),
            "median_n_eff": float(tested["n_eff"].median()),
            "bh_discoveries": discoveries, "bh_p_cutoff": cut,
            "detectable_r_at_median_n_test": stats.detectable_r(cut, float(tested["n_test"].median())),
            "detectable_r_at_median_n_eff": stats.detectable_r(cut, float(tested["n_eff"].median())),
        })
    else:
        tested = pd.DataFrame(columns=["status"])

    by_id = {s.series_id: s for s in reps}
    cands = tested[tested["status"] == "survived"].copy() if F else tested.iloc[0:0].copy()
    if len(cands):
        cands["x_label"] = [label(by_id[i], names) for i in cands["x"]]
        cands["y_label"] = [label(by_id[i], names) for i in cands["y"]]
        cands["name_overlap"] = [name_overlap(names.get(by_id[x].dataset_id, x), names.get(by_id[y].dataset_id, y)) for x, y in zip(cands["x"], cands["y"])]
        cands["score"] = cands["r_test"].abs() * (1 - cands["name_overlap"]) * np.minimum(1, cands["n_test"] / 20)
        cands = cands.sort_values(["score", "x", "y", "lag"], ascending=[False, True, True, True], kind="stable")
        pair_key = [tuple(sorted((by_id[x].dataset_id, by_id[y].dataset_id))) for x, y in zip(cands["x"], cands["y"])]
        repeat = pd.Series(pair_key, index=cands.index).duplicated()
        killed["rule6_same_dataset_pair"] += int(repeat.sum())
        cands = cands[~repeat]
    cands = cands.reindex(columns=CANDIDATE_COLUMNS).reset_index(drop=True)

    report["killed"] = dict(sorted(killed.items()))
    report["surviving"] = len(cands)
    report["by_kind"] = dict(sorted(Counter(cands["kind"]).items()))
    return Findings(cands, tested, report)


CANDIDATE_COLUMNS = [
    "kind", "x", "y", "x_label", "y_label", "freq", "lag", "n_train", "n_test", "n_eff",
    "r_train", "r_test", "p_test", "p_test_adj", "r_test_spearman", "r_ex_covid", "n_ex_covid",
    "span", "transform", "name_overlap", "score",
]


def label(s: Series, names: dict[str, str]) -> str:
    parts = [v for v in s.dims.values() if v and v != "na"]
    if s.measure:
        parts.append(s.measure)
    head = names.get(s.dataset_id, s.dataset_id)
    return f"{head}: {' / '.join(parts)}" if parts else head


def json_records(df: pd.DataFrame) -> str:
    records = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
               for r in df.to_dict(orient="records")]
    return json.dumps(records, ensure_ascii=False, sort_keys=True, indent=1, default=float) + "\n"


def write(findings: Findings, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    findings.candidates.to_parquet(out / "candidates.parquet", index=False)
    (out / "candidates.json").write_text(json_records(findings.candidates))


def print_report(f: Findings) -> None:
    r = f.report
    pairs = r["cross_dataset_pairs"]
    print(f"representatives by freq: {r['representatives']}  dropped by rule 2: {r['rule2_series_dropped']}  "
          f"flat views: {r['flat_series']}")
    print(f"cross-dataset pairs: {pairs['before_rule2']} before rule 2, {pairs['after']} after  "
          f"family size F (pairs tested): {r['family_size']}")
    if r["family_size"]:
        print(f"median n_test {r['median_n_test']:.0f}  median n_eff {r['median_n_eff']:.1f}  "
              f"BH discoveries {r['bh_discoveries']}  p cutoff {r['bh_p_cutoff']:.2e}")
        print(f"smallest detectable |r| at the cutoff: {r['detectable_r_at_median_n_test']:.3f} at median n_test, "
              f"{r['detectable_r_at_median_n_eff']:.3f} at median n_eff")
    print("killed:", ", ".join(f"{k} {v}" for k, v in r["killed"].items()) or "none")
    print(f"surviving: {r['surviving']}  by kind: {r['by_kind']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=Path("catalog.db"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=Config.k)
    args = ap.parse_args(argv)
    series = read_store(args.store)
    names = {s.dataset_id: s.label.split(":")[0] for s in series} | dataset_names(args.catalog)
    findings = find(series, names, Config(k=args.k))
    write(findings, args.out)
    print_report(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
