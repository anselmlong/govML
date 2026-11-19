"""Breaks: the period where a series changed course, tested on every series in a store.

    python -m src.atlas.breaks --store atlas/all --catalog catalog.db --out atlas/all
    python -m src.atlas.breaks --simulate

A break is one shift in the mean of a series' period-on-period changes: growth sped up or slowed
down. The test is a sup-Wald statistic whose variance is an AR(1)-prewhitened long-run variance,
calibrated by a wild AR sieve bootstrap fitted under the null. Rules and the size/power table are in
docs/atlas-plan.md (phase 6).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import stats
from .findings import (COARSENESS, PER_YEAR, full_span_r, index_period, json_records, label, make_view,
                       period_index, table_key, View)
from .flip import dataset_names
from .series import Series
from .store import read_store

Direction = Literal["accelerated", "slowed"]
Status = Literal["tested", "too_short", "flat"]


@dataclass(frozen=True)
class Config:
    alpha: float = 0.01
    trim: float = 0.15
    min_segment: int = 5
    min_changes: int = 30
    boot: int = 999
    max_ar: dict[str, int] = field(default_factory=lambda: {"A": 1, "H": 2, "Q": 4, "M": 12})
    dup_corr: float = 0.98
    dup_min_overlap: int = 10
    dup_window_years: float = 1.0


@dataclass(frozen=True)
class Break:
    """`period` is the first period of the new regime. `before`/`after` are mean changes per
    period in `transform` units (dlog: about the growth rate), from the deseasoned regime means
    plus the overall mean, so a season mix in a short regime can't flip their order. `size` is
    |after - before| over the within-regime sd of the deseasoned changes; `direction` carries
    the sign."""
    series_id: str
    dataset_id: str
    label: str
    family: str
    freq: str
    period: str
    direction: Direction
    size: float
    before: float
    after: float
    transform: str
    statistic: float
    p: float
    n: int
    duplicate_of: str | None


@dataclass(frozen=True)
class Test:
    """One row per series in the store, broken or not, so a series page can say 'tested, no
    break' apart from 'too short to test'."""
    series_id: str
    status: Status
    n: int
    statistic: float | None = None
    p: float | None = None
    period: str | None = None


# --- numerics ---------------------------------------------------------------------------------

def segment_bounds(n: int, trim: float, min_segment: int) -> tuple[int, int]:
    """Candidate first indices of the new regime: both regimes keep max(trim * n, min_segment)."""
    lo = max(math.ceil(trim * n), min_segment)
    return lo, n - lo


def wald_curve(X: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row of X (B, n) and each break index tau in [lo, hi]: the mean-shift Wald
    statistic with an AR(1)-prewhitened long-run variance of the residuals under that break.
    Studentising by the long-run variance makes the statistic close to pivotal in phi, so the
    bootstrap needs only a rough phi."""
    B, n = X.shape
    tau = np.arange(lo, hi + 1)
    cs = np.cumsum(X, axis=1)
    tot = cs[:, -1:]
    s1 = cs[:, tau - 1]
    n1, n2 = tau.astype(float), (n - tau).astype(float)
    m1, m2 = s1 / n1, (tot - s1) / n2
    gain = s1 * s1 / n1 + (tot - s1) ** 2 / n2 - tot * tot / n
    ssr1 = ((X - X.mean(1, keepdims=True)) ** 2).sum(1, keepdims=True) - gain
    # lag-1 autocovariance of the residuals x_t - m(t), from cumulative sums
    xx = np.cumsum(X[:, 1:] * X[:, :-1], axis=1)
    a = np.cumsum(X[:, 1:], axis=1)
    b = np.cumsum(X[:, :-1], axis=1)
    k1, k2 = tau - 2, tau - 1
    first = xx[:, k1] - m1 * (a[:, k1] + b[:, k1]) + (tau - 1) * m1 * m1
    cross = (X[:, tau] - m2) * (X[:, tau - 1] - m1)
    last = ((xx[:, -1:] - xx[:, k2]) - m2 * ((a[:, -1:] - a[:, k2]) + (b[:, -1:] - b[:, k2]))
            + (n - 1 - tau) * m2 * m2)
    gamma0 = ssr1 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.clip((first + cross + last) / n / gamma0, -0.97, 0.97)
        lrv = gamma0 / (1 - rho) ** 2 * n / (n - 2)
        w = np.where(lrv > 0, gain / lrv, 0.0)
    return w, tau


def _ar_fit(e: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    n = e.size
    Z = np.column_stack([np.ones(n - p)] + [e[p - j:n - j] for j in range(1, p + 1)])
    coef, *_ = np.linalg.lstsq(Z, e[p:], rcond=None)
    return coef[1:], e[p:] - Z @ coef


def _stationary(phi: np.ndarray, max_root: float = 0.97) -> np.ndarray:
    for _ in range(200):
        if np.abs(np.roots(np.r_[1.0, -phi])).max() < max_root:
            break
        phi = phi * 0.97
    return phi


def sieve_draws(x: np.ndarray, max_ar: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B series of x's length from an AR(p) fitted to x minus its mean, i.e. under the null of
    no shift. Fitting under the null keeps size: a sup-chosen break absorbs persistence and the
    alternative's phi comes out too low. p >= 1 is picked by AIC on a common sample; AR(1) gets
    Kendall's small-sample bias correction. Innovations are wild (each residual stays at its
    period with a random sign), so a series that was noisier early on keeps that in every draw:
    resampling residuals iid rejected 5-8% at a nominal 1% when the first third had 3x the sd."""
    n = x.size
    e = x - x.mean()
    pmax = max(1, min(max_ar, n // 10))
    fits = [_ar_fit(e, p) for p in range(1, pmax + 1)]
    aic = [(n - pmax) * np.log(np.mean(res[-(n - pmax):] ** 2) + 1e-300) + 2 * (i + 1) for i, (_, res) in enumerate(fits)]
    phi, res = fits[int(np.argmin(aic))]
    if phi.size == 1:
        phi = phi + (1 + 3 * phi) / n
    phi = _stationary(phi)
    res = res - res.mean()
    res = np.r_[res[:phi.size], res]
    burn = 50
    head = rng.choice(res[:max(10, n // 10)], size=(B, burn))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, n))
    draws = np.concatenate([head, res[None] * signs], axis=1)
    return lfilter([1.0], np.r_[1.0, -phi], draws, axis=1)[:, burn:]


def _deseason_rows(X: np.ndarray, season: np.ndarray, per_year: int) -> np.ndarray:
    if per_year == 1:
        return X
    X = X.copy()
    for k in range(per_year):
        cols = season == k
        if cols.any():
            X[:, cols] -= X[:, cols].mean(axis=1, keepdims=True)
    return X


@dataclass(frozen=True)
class Shift:
    statistic: float
    p: float
    tau: int
    delta: float
    before: float
    after: float


def detect_shift(x: np.ndarray, season: np.ndarray, per_year: int, max_ar: int, cfg: Config,
               rng: np.random.Generator) -> Shift:
    """One mean shift in the changes x (no NaNs). Season means come out of x and out of every
    bootstrap draw alike, so the draws see the same lost degrees of freedom."""
    n = x.size
    lo, hi = segment_bounds(n, cfg.trim, cfg.min_segment)
    xd = _deseason_rows(x[None], season, per_year)
    w, tau = wald_curve(xd, lo, hi)
    j = int(np.argmax(w[0]))
    stat, t = float(w[0, j]), int(tau[j])
    draws = _deseason_rows(sieve_draws(xd[0], max_ar, cfg.boot, rng), season, per_year)
    null = wald_curve(draws, lo, hi)[0].max(axis=1)
    p = (1 + int((null >= stat).sum())) / (cfg.boot + 1)
    d = xd[0]
    m1, m2 = d[:t].mean(), d[t:].mean()
    resid = np.r_[d[:t] - m1, d[t:] - m2]
    sd = np.sqrt((resid ** 2).sum() / (n - 2))
    level = x.mean() - d.mean()
    return Shift(stat, p, t, float((m2 - m1) / sd) if sd > 0 else 0.0, float(level + m1), float(level + m2))


# --- series to tasks --------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    """The longest gap-free stretch of a series' changes. `start` is the period index of the
    change into values[0]. A gap is never bridged and the AR model needs consecutive periods."""
    start: int
    values: np.ndarray
    transform: str


def longest_run(s: Series) -> Run:
    v = make_view(s, s.freq)
    edges = np.flatnonzero(np.diff(np.r_[0, (~np.isnan(v.values)).astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    if not starts.size:
        return Run(v.start, v.values[:0], v.transform)
    k = int(np.argmax(ends - starts))
    return Run(v.start + int(starts[k]), v.values[starts[k]:ends[k]], v.transform)


def seed_for(series_id: str) -> int:
    """Stable across processes and runs, unlike hash(), so each p depends only on its series."""
    return zlib.crc32(series_id.encode())


def _test_task(task: tuple) -> Shift:
    series_id, x, season, per_year, max_ar, cfg = task
    return detect_shift(x, season, per_year, max_ar, cfg, np.random.default_rng(seed_for(series_id)))


def _map(fn, tasks: list, workers: int) -> list:
    if workers <= 1 or len(tasks) < 2:
        return [fn(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 16)))


# --- de-duplication -----------------------------------------------------------------------------

def _year(freq: str, period: str) -> float:
    return period_index(freq, period) / PER_YEAR[freq]


def mark_duplicates(breaks: list[Break], by_id: dict[str, Series], cfg: Config) -> dict[str, str]:
    """Two broken series whose breaks fall within cfg.dup_window_years and whose changes, at the
    coarser of their frequencies, correlate above cfg.dup_corr are one quantity: the same table
    in two units, a monthly and quarterly twin, a total that is one dominant row. Each connected
    group keeps the finest frequency, then the most points, then the smallest id. Returns
    duplicate -> keeper. Only broken series are compared, so the cost follows the break count,
    not the store's 10^9 pairs."""
    parent: dict[str, str] = {}

    def root(a: str) -> str:
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    views: dict[tuple[str, str], View] = {}

    def view(sid: str, freq: str) -> View:
        if (sid, freq) not in views:
            views[(sid, freq)] = make_view(by_id[sid], freq)
        return views[(sid, freq)]

    dated = sorted(((_year(b.freq, b.period), b) for b in breaks), key=lambda t: (t[0], t[1].series_id))
    for i, (yi, a) in enumerate(dated):
        for yj, b in dated[i + 1:]:
            if yj - yi > cfg.dup_window_years:
                break
            f = min(a.freq, b.freq, key=COARSENESS.get)
            r = full_span_r(view(a.series_id, f), view(b.series_id, f), cfg.dup_min_overlap)
            if abs(r) > cfg.dup_corr:
                ra, rb = root(a.series_id), root(b.series_id)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, set[str]] = {}
    for sid in parent:
        groups.setdefault(root(sid), {root(sid)}).add(sid)
    out = {}
    for members in groups.values():
        keep = min(members, key=lambda i: (-COARSENESS[by_id[i].freq], -len(by_id[i].points), i))
        out |= {m: keep for m in members - {keep}}
    return out


# --- pipeline ---------------------------------------------------------------------------------

@dataclass
class Result:
    breaks: list[Break]
    tests: list[Test]
    report: dict


def find_breaks(series: list[Series], names: dict[str, str], cfg: Config = Config(), workers: int = 1) -> Result:
    t0 = time.time()
    tests: dict[str, Test] = {}
    runs: dict[str, Run] = {}
    tasks = []
    for s in sorted(series, key=lambda s: s.series_id):
        run = longest_run(s)
        if run.values.size < cfg.min_changes:
            tests[s.series_id] = Test(s.series_id, "too_short", int(run.values.size))
        elif stats.is_flat(run.values):
            tests[s.series_id] = Test(s.series_id, "flat", int(run.values.size))
        else:
            per_year = PER_YEAR[s.freq]
            season = (run.start + np.arange(run.values.size)) % per_year
            runs[s.series_id] = run
            tasks.append((s.series_id, run.values, season, per_year, cfg.max_ar[s.freq], cfg))
    shifts = _map(_test_task, tasks, workers)

    by_id = {s.series_id: s for s in series}
    found: list[Break] = []
    for (sid, *_), sh in zip(tasks, shifts):
        s, run = by_id[sid], runs[sid]
        period = index_period(s.freq, run.start + sh.tau)
        tests[sid] = Test(sid, "tested", int(run.values.size), sh.statistic, sh.p, period)
        if sh.p > cfg.alpha:
            continue
        name = names.get(s.dataset_id, s.dataset_id)
        found.append(Break(sid, s.dataset_id, label(s, names), table_key(name), s.freq, period,
                           "accelerated" if sh.delta > 0 else "slowed", abs(sh.delta), sh.before, sh.after, run.transform,
                           sh.statistic, sh.p, int(run.values.size), None))
    dup = mark_duplicates(found, by_id, cfg)
    found = [Break(**{**asdict(b), "duplicate_of": dup.get(b.series_id)}) for b in found]
    found.sort(key=lambda b: (b.period[:4], b.series_id))

    status = Counter(t.status for t in tests.values())
    n_tested = status["tested"]
    edge = sum(sh.p <= cfg.alpha and sh.tau in segment_bounds(runs[sid].values.size, cfg.trim, cfg.min_segment)
               for (sid, *_), sh in zip(tasks, shifts))
    report = {
        "series": len(series), "tested": n_tested, "skipped": {k: status[k] for k in ("too_short", "flat")},
        "breaks": len(found), "breaks_deduped": sum(b.duplicate_of is None for b in found),
        "rate": len(found) / n_tested if n_tested else 0.0,
        "expected_false_positives": cfg.alpha * n_tested,
        "breaks_at": {str(a): sum(sh.p <= a for sh in shifts) for a in (0.01, 0.005, 0.002)},
        "edge_share": edge / len(found) if found else 0.0,
        "tested_by_freq": dict(sorted(Counter(by_id[t].freq for t in runs).items())),
        "breaks_by_freq": dict(sorted(Counter(b.freq for b in found).items())),
        "by_year": by_year(found, runs, by_id, cfg),
        "alpha": cfg.alpha, "boot": cfg.boot, "workers": workers, "seconds": round(time.time() - t0, 1),
    }
    return Result(found, [tests[k] for k in sorted(tests)], report)


def by_year(found: list[Break], runs: dict[str, Run], by_id: dict[str, Series], cfg: Config) -> dict[str, dict]:
    """Per calendar year: breaks, de-duplicated breaks, distinct datasets and families among
    them, the tested series whose candidate break window covers the year (exposed), and the
    breaks expected with no break anywhere: alpha spread evenly over each exposed window. A year
    near its expected_null count is noise."""
    exposed: Counter[int] = Counter()
    expected: Counter[int] = Counter()
    for sid, run in runs.items():
        k = PER_YEAR[by_id[sid].freq]
        lo, hi = segment_bounds(run.values.size, cfg.trim, cfg.min_segment)
        years = range((run.start + lo) // k, (run.start + hi) // k + 1)
        exposed.update(years)
        for y in years:
            expected[y] += cfg.alpha / len(years)
    rows: dict[int, dict] = {y: {"breaks": 0, "deduped": 0, "datasets": set(), "families": set()} for y in exposed}
    for b in found:
        r = rows[int(b.period[:4])]
        r["breaks"] += 1
        if b.duplicate_of is None:
            r["deduped"] += 1
            r["datasets"].add(b.dataset_id)
            r["families"].add(b.family)
    return {str(y): {"breaks": r["breaks"], "deduped": r["deduped"], "datasets": len(r["datasets"]),
                     "families": len(r["families"]), "exposed": exposed[y], "expected_null": round(expected[y], 2)}
            for y, r in sorted(rows.items())}


def write(result: Result, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    breaks = pd.DataFrame([asdict(b) for b in result.breaks], columns=list(Break.__dataclass_fields__))
    breaks.to_parquet(out / "breaks.parquet", index=False)
    (out / "breaks.json").write_text(json_records(breaks))
    tests = pd.DataFrame([asdict(t) for t in result.tests], columns=list(Test.__dataclass_fields__))
    tests.to_parquet(out / "break_tests.parquet", index=False)
    report = {k: v for k, v in result.report.items() if k not in ("seconds", "workers")}
    (out / "breaks_report.json").write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")


def print_report(r: dict) -> None:
    sk = r["skipped"]
    print(f"series {r['series']}  tested {r['tested']} {r['tested_by_freq']}  "
          f"skipped {sk['too_short'] + sk['flat']} (too short {sk['too_short']}, flat {sk['flat']})")
    print(f"breaks {r['breaks']} at p <= {r['alpha']} {r['breaks_by_freq']}  rate {r['rate']:.1%}  "
          f"after de-duplication {r['breaks_deduped']}  expected false positives {r['expected_false_positives']:.0f}")
    print(f"breaks at stricter p: {r['breaks_at']}  share dated at a trim edge {r['edge_share']:.1%}")
    print(f"runtime {r['seconds']}s on {r['workers']} workers, {r['boot']} bootstrap draws per series")
    print("year  breaks  deduped  datasets  families  exposed  expected_null")
    for y, row in r["by_year"].items():
        if row["breaks"]:
            print(f"{y}  {row['breaks']:6d}  {row['deduped']:7d}  {row['datasets']:8d}  {row['families']:8d}  "
                  f"{row['exposed']:7d}  {row['expected_null']:13.1f}")


# --- simulation -------------------------------------------------------------------------------

def ar1(n: int, phi: float, rng: np.random.Generator, burn: int = 100) -> np.ndarray:
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n + burn))[burn:]


def _sim_task(task: tuple) -> tuple[float, int]:
    n, phi, shift, at, per_year, cfg, seed = task
    rng = np.random.default_rng(seed)
    x = ar1(n, phi, rng)
    t_true = int(at * n)
    x[t_true:] += shift / np.sqrt(1 - phi * phi)
    season = np.arange(n) % per_year
    sh = detect_shift(x, season, per_year, cfg.max_ar["M" if per_year == 12 else "A"], cfg, rng)
    return sh.p, sh.tau - t_true


def simulate(ns=(15, 20, 25, 30, 60, 150, 300), phis=(0.0, 0.5, 0.8), shift: float = 0.0, at: float = 0.5,
             reps: int = 1000, per_year: int = 1, cfg: Config = Config(), workers: int = 1) -> list[dict]:
    """Rejection rate at cfg.alpha on AR(1) series of unit innovations, with `shift` marginal sd
    added from `at` of the way through. With shift 0 it is the size, otherwise the power plus the
    median |date error| of rejections and their share within 2 periods."""
    rows = []
    for n in ns:
        for phi in phis:
            tasks = [(n, phi, shift, at, per_year, cfg, 10_000 * n + int(100 * phi) + r * 7919) for r in range(reps)]
            out = _map(_sim_task, tasks, workers)
            hits = [err for p, err in out if p <= cfg.alpha]
            rows.append({"n": n, "phi": phi, "rate": len(hits) / reps,
                         "median_date_error": float(np.median(np.abs(hits))) if hits else None,
                         "within_2": float(np.mean(np.abs(hits) <= 2)) if hits else None})
    return rows


# --- CLI --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path)
    ap.add_argument("--catalog", type=Path, default=Path("catalog.db"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--simulate", action="store_true", help="print the size and power table instead")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.simulate:
        for title, kw in [("size, annual-style (AR order 1, no seasons)", {}),
                          ("size, monthly-style (12 season means, AR order up to 12)", {"ns": (60, 150, 300), "per_year": 12}),
                          ("power, 1 sd shift at the middle", {"shift": 1.0}),
                          ("power, 2 sd shift at the middle", {"shift": 2.0}),
                          ("power, 2 sd shift a quarter of the way in", {"shift": 2.0, "at": 0.25})]:
            print(title)
            for r in simulate(reps=args.reps, workers=args.workers, **kw):
                extra = (f"  median |date error| {r['median_date_error']:.0f}  within 2: {r['within_2']:.2f}"
                         if r["median_date_error"] is not None and "shift" in kw else "")
                print(f"  n {r['n']:3d}  phi {r['phi']:.1f}  rate {r['rate']:.3f}{extra}")
        return 0
    if not (args.store and args.out):
        ap.error("--store and --out are required unless --simulate")
    series = read_store(args.store)
    names = {s.dataset_id: s.label.split(":")[0] for s in series} | dataset_names(args.catalog)
    result = find_breaks(series, names, Config(), args.workers)
    write(result, args.out)
    print_report(result.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
