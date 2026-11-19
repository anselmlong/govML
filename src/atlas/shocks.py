"""Shocks: quarters (and, for annual tables, years) in which more table families than chance
jumped at once. These lead the year pages; phase 6 turns in the same year follow as secondary.

    python -m src.atlas.shocks --store atlas/all --catalog catalog.db --out atlas/all

A series shocks in a period when its change is at least 4 robust sd away from its local level.
A family shocks when at least half its series covering the period shock. The count of shocked
families per period is tested against a null that circularly shifts each family's shock
indicator, family-wise over all periods (the null's maximum count). Rules, the size check and the
real-store numbers are in docs/atlas-plan.md (phase 7).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import binomtest

from . import stats
from .breaks import longest_run
from .findings import PER_YEAR, index_period, make_view, representatives, table_key
from .flip import dataset_names
from .periods import quarters_dated_by_month
from .series import Series
from .store import read_store

Direction = Literal["up", "down"]
Level = Literal["quarter", "year"]
LEVEL_FREQS: dict[str, tuple[str, ...]] = {"quarter": ("M", "Q"), "year": ("A",)}


@dataclass(frozen=True)
class Config:
    z: float = 4.0
    share: float = 0.5
    min_changes: int = 40
    local_years: dict[str, int] = field(default_factory=lambda: {"M": 2, "Q": 2, "A": 0})
    copy_corr: float = 0.9
    copy_overlap: int = 40
    copy_clip: float = 3.0
    copy_reps: int = 10
    covid_years: tuple[int, int] = (2020, 2022)
    reps: int = 10_000
    seed: int = 0
    alpha: float = 0.01
    min_reversion: float = 0.5
    agency_p: float = 0.01


# --- data shape -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shock:
    """One series' jump. `period` is native; `slot` is the quarter or year it counts in. `reversion`
    is the share of the jump undone over the next year of changes: about 1 for a spike that comes
    back (a shock), about 0 for a step that stays (a revision, a rebasing, a lasting fall). None
    when the series ends within a year."""
    series_id: str
    dataset_id: str
    family: str
    freq: str
    period: str
    slot: str
    z: float
    direction: Direction
    reversion: float | None


@dataclass(frozen=True)
class Member:
    """A family shocked in the event's period: `shocked` of its `covering` series jumped."""
    family: str
    agency: str | None
    datasets: list[str]
    shocked: int
    covering: int
    reversion: float | None
    series: list[dict]


@dataclass(frozen=True)
class Turn:
    series_id: str
    label: str
    family: str
    period: str
    direction: str


@dataclass(frozen=True)
class Event:
    """`period` is a quarter ("2020Q2") or, for annual tables, a year ("1998"). `flags` name artefact
    signatures: `no_reversion` (the median member stepped and stayed: a revision looks like this,
    and so does a lasting fall) and `one_agency` (members more concentrated in one agency than the
    families exposed that period would give)."""
    period: str
    level: Level
    families: int
    expected: float
    p: float
    p_familywise: float
    reversion: float | None
    top_agency: str | None
    top_agency_share: float
    flags: list[str]
    members: list[Member]
    turns: list[Turn]


# --- families ---------------------------------------------------------------------------------

_POPULATION_SPLIT = re.compile(r"\b(citizens?|residents?|overall|total|males?|females?|persons?)\b")
_LABOUR_BREAKDOWN = re.compile(r"\bby (industry|occupation|occupational group)\b.*$")


def family_key(name: str) -> str:
    """The phase 3 table key with population splits removed too: 'Number of Unemployed Citizens',
    '... Residents' and 'Number of Overall Unemployed' are one publication cut three ways, and
    phase 6 found table_key leaves male, female and total tables apart. Manpower tables also come
    as a total and 'by Industry', 'by Occupation' or 'by Industry (SSIC 2020) and Occupational
    Group' cuts, which phase 3's breakdown list does not name."""
    key = _LABOUR_BREAKDOWN.sub(" ", _POPULATION_SPLIT.sub(" ", table_key(name)))
    return re.sub(r"\s+", " ", key).strip() or table_key(name)


def dataset_agencies(catalog: Path) -> dict[str, str]:
    if not catalog.exists():
        return {}
    with sqlite3.connect(catalog) as conn:
        return {d: a for d, a in conn.execute("SELECT dataset_id, agency FROM datasets") if a}


# --- series to robust z -----------------------------------------------------------------------

@dataclass(frozen=True)
class Track:
    """One series' changes on its longest gap-free run. `slots` maps each change to its quarter or
    year index; `resid` is the deseasoned change minus its local median; `z` is resid over the
    residuals' robust sd."""
    series: Series
    family: str
    start: int
    slots: np.ndarray
    resid: np.ndarray
    z: np.ndarray


def robust_z(x: np.ndarray, season: np.ndarray, per_year: int, local_years: int) -> tuple[np.ndarray, np.ndarray]:
    """Season means out, then a centred rolling median of the changes (local_years each side, 0 for
    none). Without it, a fast-growth stretch (mobile subscriptions in 2000) sits at z 4 to 6 for
    months on end against a MAD set by the later, flat years: a regime, not a shock. Annual series
    get none: a 5-point median soaks up a two-year episode and leaves a MAD of one-year wiggles,
    which put 8.3% of annual changes past z 4 against 2.7% without it."""
    x = x.copy()
    for k in range(per_year):
        sel = season == k
        if sel.any():
            x[sel] -= x[sel].mean()
    resid = x - median_filter(x, size=2 * local_years * per_year + 1, mode="nearest") if local_years else x
    med = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - med))
    return resid, (resid - med) / mad if mad > 1e-12 else np.zeros_like(resid)


def slot_of(freq: str, index: np.ndarray, level: Level) -> np.ndarray:
    if level == "year":
        return index // PER_YEAR[freq]
    return index // 3 if freq == "M" else index


def slot_label(level: Level, slot: int) -> str:
    return str(slot) if level == "year" else index_period("Q", slot)


def tracks(series: list[Series], families: dict[str, str], level: Level, cfg: Config) -> list[Track]:
    out = []
    for s in sorted(series, key=lambda s: s.series_id):
        if s.freq not in LEVEL_FREQS[level]:
            continue
        run = longest_run(s)
        if run.values.size < cfg.min_changes or stats.is_flat(run.values):
            continue
        per_year = PER_YEAR[s.freq]
        idx = run.start + np.arange(run.values.size)
        resid, z = robust_z(run.values, idx % per_year, per_year, cfg.local_years[s.freq])
        if not z.any():
            continue
        out.append(Track(s, families[s.dataset_id], run.start, slot_of(s.freq, idx, level), resid, z))
    return out


def common_changes(series: list[Series], covid_years: tuple[int, int], clip: float) -> np.ndarray:
    """Rows of deseasoned changes on one grid at the coarsest frequency present (monthly series
    averaged to quarters), clipped at `clip` robust sd, NaN outside each series and in
    covid_years."""
    freq = "A" if any(s.freq == "A" for s in series) else "Q"
    per_year = PER_YEAR[freq]
    views = [make_view(s, freq) for s in series]
    lo = min(v.start for v in views)
    width = max(v.start + v.values.size for v in views) - lo
    X = np.full((len(views), width), np.nan)
    for i, v in enumerate(views):
        x = v.values.copy()
        season = (v.start + np.arange(x.size)) % per_year
        for k in range(per_year if per_year > 1 else 0):
            sel = (season == k) & ~np.isnan(x)
            if sel.any():
                x[season == k] -= x[sel].mean()
        med = np.nanmedian(x)
        mad = 1.4826 * np.nanmedian(np.abs(x - med))
        X[i, v.start - lo:v.start - lo + x.size] = np.clip(x, med - clip * mad, med + clip * mad)
    years = (lo + np.arange(width)) // per_year
    X[:, (years >= covid_years[0]) & (years <= covid_years[1])] = np.nan
    return X


def pairwise_r(X: np.ndarray, min_overlap: int) -> np.ndarray:
    """Pearson r of every row pair over the columns both have, NaN below min_overlap."""
    M = (~np.isnan(X)).astype(float)
    Z = np.nan_to_num(X)
    n = M @ M.T
    sx = Z @ M.T
    sxx = (Z * Z) @ M.T
    sxy = Z @ Z.T
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * sxy - sx * sx.T) / np.sqrt((n * sxx - sx * sx) * (n * sxx.T - sx.T * sx.T))
    return np.where(n >= min_overlap, r, np.nan)


def merge_copies(ts: list[Track], cfg: Config) -> dict[str, str]:
    """Families with a pair of series whose changes correlate above cfg.copy_corr are one quantity
    published twice under names no key can match: 'HDB Resale Price Index' and 'Housing and
    Development Board (HDB) Resale Price Index', arrivals and air arrivals by country. Rule 2's
    duplicate check over the phase 3 representatives (cfg.copy_reps per dataset and frequency) at
    a looser bar, because copies' top rows are often different cuts; one headline per family put
    the two arrivals tables at r 0.85, though their country rows match at 0.999. Changes at
    the level's coarsest frequency, with those in cfg.covid_years left out: with them, unrelated series that all fell in
    2020 (air passengers and visitor arrivals, r 0.985) look like copies, and an event would merge
    its own members. Returns family -> the connected group's name: its member with the most
    series, then the smallest name."""
    by_family: dict[str, list[Track]] = defaultdict(list)
    for t in ts:
        by_family[t.family].append(t)
    names = sorted(by_family)
    rows = [(i, s) for i, f in enumerate(names)
            for s in sorted(representatives([t.series for t in by_family[f]], cfg.copy_reps), key=lambda s: s.series_id)]
    owner = np.array([i for i, _ in rows])
    r = pairwise_r(common_changes([s for _, s in rows], cfg.covid_years, cfg.copy_clip), cfg.copy_overlap)
    parent = list(range(len(names)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in zip(*np.nonzero(np.abs(np.nan_to_num(r)) > cfg.copy_corr)):
        ra, rb = root(int(owner[a])), root(int(owner[b]))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[str]] = defaultdict(list)
    for i, f in enumerate(names):
        groups[root(i)].append(f)
    out = {}
    for members in groups.values():
        name = min(members, key=lambda f: (-len(by_family[f]), f))
        out |= {f: name for f in members}
    return out


def reversion(resid: np.ndarray, i: int, per_year: int) -> float | None:
    if i + per_year >= resid.size or resid[i] == 0:
        return None
    return float(-resid[i + 1:i + 1 + per_year].sum() / resid[i])


def find_shocks(ts: list[Track], level: Level, cfg: Config) -> list[Shock]:
    out = []
    for t in ts:
        for i in np.flatnonzero(np.abs(t.z) >= cfg.z):
            s = t.series
            out.append(Shock(s.series_id, s.dataset_id, t.family, s.freq, index_period(s.freq, t.start + int(i)),
                             slot_label(level, int(t.slots[i])), float(t.z[i]), "up" if t.z[i] > 0 else "down",
                             reversion(t.resid, int(i), PER_YEAR[s.freq])))
    return out


# --- family indicators and the null -----------------------------------------------------------

@dataclass(frozen=True)
class Panel:
    """Per family, the slots any of its series covers, and whether at least `share` of the series
    covering each slot shocked there. Rows are families in sorted order; `lo` is the first slot."""
    families: list[str]
    lo: int
    width: int
    covered: list[np.ndarray]
    hit: list[np.ndarray]
    shocked: list[np.ndarray]
    covering: list[np.ndarray]


def panel(ts: list[Track], cfg: Config) -> Panel:
    lo = min(int(t.slots.min()) for t in ts)
    width = max(int(t.slots.max()) for t in ts) - lo + 1
    by_family: dict[str, list[Track]] = defaultdict(list)
    for t in ts:
        by_family[t.family].append(t)
    names = sorted(by_family)
    covered, hit, shocked_out, covering_out = [], [], [], []
    for f in names:
        shocked, covering = np.zeros(width, int), np.zeros(width, int)
        for t in by_family[f]:
            slots = np.unique(t.slots) - lo
            covering[slots] += 1
            shocked[np.unique(t.slots[np.abs(t.z) >= cfg.z]) - lo] += 1
        cov = np.flatnonzero(covering)
        covered.append(cov)
        hit.append(shocked[cov] >= cfg.share * covering[cov])
        shocked_out.append(shocked)
        covering_out.append(covering)
    return Panel(names, lo, width, covered, hit, shocked_out, covering_out)


def observed(p: Panel) -> np.ndarray:
    counts = np.zeros(p.width, int)
    for cov, h in zip(p.covered, p.hit):
        counts[cov[h]] += 1
    return counts


def null_counts(p: Panel, reps: int, seed: int, batch: int = 500) -> np.ndarray:
    """reps x width shocked-family counts with each family's hit sequence rotated by a random
    offset over the slots it covers. Rotating whole families, not series: the series of one table
    jump together, and rotating them apart made "half the family at once" nearly impossible under
    the null, so the null total came out 30% under the observed total and noise quarters looked
    significant."""
    rng = np.random.default_rng(seed)
    sizes = np.array([c.size for c in p.covered])
    seg = np.repeat(np.cumsum(sizes) - sizes, sizes)
    length = np.repeat(sizes, sizes)
    within = np.arange(sizes.sum()) - seg
    slots = np.concatenate(p.covered)
    hits = np.concatenate(p.hit).astype(float)
    out = np.empty((reps, p.width), np.int32)
    for b0 in range(0, reps, batch):
        b = min(batch, reps - b0)
        off = np.repeat(rng.integers(0, sizes, size=(b, sizes.size)), sizes, axis=1)
        moved = slots[seg + (within + off) % length]
        rows = np.repeat(np.arange(b)[:, None] * p.width, moved.shape[1], axis=1)
        out[b0:b0 + b] = np.bincount((rows + moved).ravel(), weights=np.broadcast_to(hits, moved.shape).ravel(),
                                     minlength=b * p.width).reshape(b, p.width)
    return out


@dataclass(frozen=True)
class Significance:
    observed: np.ndarray
    expected: np.ndarray
    p: np.ndarray
    p_familywise: np.ndarray
    bar: int
    reps: int
    seconds: float


def significance(p: Panel, cfg: Config) -> Significance:
    """Per-period p against that period's null counts, and the family-wise p against the null's
    maximum over all periods. `bar` is the smallest count with family-wise p <= alpha."""
    t0 = time.time()
    null = null_counts(p, cfg.reps, cfg.seed)
    obs = observed(p)
    n = cfg.reps + 1
    top = np.sort(null.max(axis=1))
    beaten = lambda c: cfg.reps - np.searchsorted(top, c, side="left")
    grid = np.arange(top[-1] + 2)
    bar = int(grid[np.argmax((1 + beaten(grid)) / n <= cfg.alpha)])
    return Significance(obs, null.mean(axis=0), (1 + (null >= obs).sum(axis=0)) / n, (1 + beaten(obs)) / n, bar,
                        cfg.reps, time.time() - t0)


# --- events -----------------------------------------------------------------------------------

def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def build_events(level: Level, p: Panel, t: Significance, shocks: list[Shock], agencies: dict[str, str | None],
                 family_names: dict[str, list[str]], turns: list[Turn], cfg: Config) -> list[Event]:
    by_slot_family: dict[tuple[str, str], list[Shock]] = defaultdict(list)
    for s in shocks:
        by_slot_family[(s.slot, s.family)].append(s)
    events = []
    for j in np.flatnonzero((t.p_familywise <= cfg.alpha) & (t.observed > 0)):
        label = slot_label(level, p.lo + int(j))
        members = []
        for i, f in enumerate(p.families):
            k = np.searchsorted(p.covered[i], j)
            if k >= p.covered[i].size or p.covered[i][k] != j or not p.hit[i][k]:
                continue
            fs = sorted(by_slot_family[(label, f)], key=lambda s: (s.series_id, s.period))
            revs = [s.reversion for s in fs if s.reversion is not None]
            members.append(Member(f, agencies.get(f), family_names[f], int(p.shocked[i][j]), int(p.covering[i][j]),
                                  _median(revs), [{"series_id": s.series_id, "period": s.period, "z": round(s.z, 2),
                                                   "direction": s.direction} for s in fs]))
        exposed = [agencies.get(f) for i, f in enumerate(p.families) if p.covering[i][j]]
        top, n_top = Counter(agencies.get(m.family) for m in members).most_common(1)[0]
        base = sum(a == top for a in exposed) / len(exposed)
        rev = _median([m.reversion for m in members if m.reversion is not None])
        flags = []
        if rev is not None and rev < cfg.min_reversion:
            flags.append("no_reversion")
        if top and binomtest(n_top, len(members), base, alternative="greater").pvalue < cfg.agency_p:
            flags.append("one_agency")
        year = label[:4]
        events.append(Event(label, level, int(t.observed[j]), round(float(t.expected[j]), 2),
                            float(t.p[j]), float(t.p_familywise[j]), rev, top, round(n_top / len(members), 2), flags,
                            members, [x for x in turns if x.period[:4] == year]))
    return sorted(events, key=lambda e: (-e.families, e.p_familywise, e.period))


def load_turns(path: Path | None, families: dict[str, str]) -> list[Turn]:
    """De-duplicated phase 6 breaks. A breaks file from the phase 3 code has other columns and
    is refused, not read as no turns."""
    if path is None:
        return []
    df = pd.read_parquet(path)
    need = {"series_id", "dataset_id", "label", "period", "direction", "duplicate_of"}
    if not need <= set(df.columns):
        raise SystemExit(f"{path} lacks {sorted(need - set(df.columns))}: rerun `python -m src.atlas.breaks` there")
    df = df[df["duplicate_of"].isna()].sort_values(["period", "series_id"], kind="stable")
    return [Turn(r.series_id, r.label, families.get(r.dataset_id, r.dataset_id), r.period, r.direction)
            for r in df.itertuples(index=False)]


@dataclass
class Result:
    events: list[Event]
    shocks: list[Shock]
    report: dict


def run(series: list[Series], names: dict[str, str], agencies_by_dataset: dict[str, str], turns_path: Path | None,
        cfg: Config = Config(), levels: tuple[Level, ...] = ("quarter", "year")) -> Result:
    series = [quarters_dated_by_month(s) for s in series]
    keys = {s.dataset_id: family_key(names.get(s.dataset_id, s.dataset_id)) for s in series}
    turns = load_turns(turns_path, keys)
    events, shocks, report = [], [], {}
    for level in levels:
        ts = tracks(series, keys, level, cfg)
        if not ts:
            continue
        merged = merge_copies(ts, cfg)
        ts = [replace(t, family=merged[t.family]) for t in ts]
        datasets: dict[str, set[str]] = defaultdict(set)
        for t in ts:
            datasets[t.family].add(t.series.dataset_id)
        family_names = {f: sorted(names.get(d, d) for d in ds) for f, ds in datasets.items()}
        agencies = {f: Counter(agencies_by_dataset.get(d) for d in sorted(ds)).most_common(1)[0][0]
                    for f, ds in datasets.items()}
        p = panel(ts, cfg)
        t = significance(p, cfg)
        found = find_shocks(ts, level, cfg)
        level_events = build_events(level, p, t, found, agencies, family_names, turns, cfg)
        events += level_events
        shocks += found
        report[level] = {
            "series": len(ts), "families": len(p.families), "family_keys": len(merged), "shocks": len(found),
            "shock_rate": round(len(found) / sum(x.z.size for x in ts), 4),
            "bar": t.bar, "reps": t.reps, "null_seconds": round(t.seconds, 1),
            "observed_total": int(t.observed.sum()), "expected_total": round(float(t.expected.sum()), 1),
            "events": len(level_events),
            "top": [{"period": slot_label(level, p.lo + int(j)), "families": int(t.observed[j]),
                     "expected": round(float(t.expected[j]), 2), "p": float(t.p[j]),
                     "p_familywise": float(t.p_familywise[j])}
                    for j in np.argsort(-(t.observed - t.expected), kind="stable")[:40]],
        }
    return Result(events, shocks, report)


# --- output -----------------------------------------------------------------------------------

def _clean(v):
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def write(result: Result, out: Path, cfg: Config) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shocks = pd.DataFrame([asdict(s) for s in result.shocks], columns=list(Shock.__dataclass_fields__))
    shocks.to_parquet(out / "shocks.parquet", index=False)
    report = {level: {k: v for k, v in r.items() if k != "null_seconds"} for level, r in result.report.items()}
    body = {"config": asdict(cfg), "report": report, "events": [asdict(e) for e in result.events]}
    (out / "events.json").write_text(json.dumps(_clean(body), ensure_ascii=False, sort_keys=True, indent=1) + "\n")


def print_report(result: Result) -> None:
    for level, r in result.report.items():
        print(f"[{level}] series {r['series']}  families {r['families']} (from {r['family_keys']} table keys)  shocks {r['shocks']} "
              f"(rate {r['shock_rate']:.2%})  observed family-periods {r['observed_total']} vs null "
              f"{r['expected_total']}  family-wise bar {r['bar']} families  "
              f"{r['reps']} reps in {r['null_seconds']}s  events {r['events']}")
    for e in result.events:
        flags = f"  [{', '.join(e.flags)}]" if e.flags else ""
        rev = "n/a" if e.reversion is None else f"{e.reversion:.2f}"
        print(f"{e.period:7s} families {e.families:3d}  expected {e.expected:5.2f}  p {e.p:.4f}  p_fw "
              f"{e.p_familywise:.4f}  reversion {rev}  top agency {e.top_agency} {e.top_agency_share:.0%}{flags}")
        for m in e.members:
            dirs = Counter(s["direction"] for s in m.series)
            print(f"    {m.family[:70]:70s} {m.shocked}/{m.covering}  {dict(dirs)}")
        if e.turns:
            print(f"    + {len(e.turns)} turns in {e.period[:4]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=Path("catalog.db"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--breaks", type=Path, help="phase 6 breaks.parquet (default: <out>/breaks.parquet if present)")
    ap.add_argument("--reps", type=int, default=Config.reps)
    ap.add_argument("--seed", type=int, default=Config.seed)
    args = ap.parse_args(argv)
    breaks = args.breaks or (args.out / "breaks.parquet" if (args.out / "breaks.parquet").exists() else None)
    if breaks is None:
        print("no breaks.parquet: events carry no turns")
    series = read_store(args.store)
    names = {s.dataset_id: s.label.split(":")[0] for s in series} | dataset_names(args.catalog)
    cfg = Config(reps=args.reps, seed=args.seed)
    result = run(series, names, dataset_agencies(args.catalog), breaks, cfg)
    write(result, args.out, cfg)
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
