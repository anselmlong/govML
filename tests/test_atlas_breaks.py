import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.atlas.breaks import Config, ar1, find_breaks, main, segment_bounds, detect_shift
from src.atlas.series import Series, format_period
from src.atlas.store import read_store, write_store

REAL_STORE = Path("/Users/anselm/src/govml-flipper/atlas/all")
FAST = Config(boot=199)


def series_from_changes(dataset_id: str, changes: np.ndarray, freq: str = "M", label: str = "Total",
                        scale: float = 100.0) -> Series:
    levels = scale * np.exp(np.concatenate([[0.0], np.cumsum(changes)]))
    if freq == "A":
        periods = [format_period("A", 1960 + i) for i in range(levels.size)]
    else:
        periods = [format_period("M", 1990 + i // 12, i % 12 + 1) for i in range(levels.size)]
    return Series(f"{dataset_id}:{label.lower()}", dataset_id, f"{dataset_id}: {label}", {"series": label}, None, None,
                  freq, dict(zip(periods, levels.tolist())))


def iid_sup_f(x: np.ndarray, cfg: Config) -> float:
    """The phase 3 statistic: sup-F with iid variance, for the negative control."""
    n = x.size
    lo, hi = segment_bounds(n, cfg.trim, cfg.min_segment)
    best = 0.0
    for t in range(lo, hi + 1):
        ssr1 = ((x[:t] - x[:t].mean()) ** 2).sum() + ((x[t:] - x[t:].mean()) ** 2).sum()
        best = max(best, (((x - x.mean()) ** 2).sum() - ssr1) / (ssr1 / (n - 2)))
    return best


def rejections(n: int, phi: float, reps: int, shift: float = 0.0, at: float = 0.5) -> tuple[list[float], list[int], int]:
    rng = np.random.default_rng(1000 * n + int(10 * phi))
    ps, errors, iid = [], [], 0
    for _ in range(reps):
        x = ar1(n, phi, rng)
        x[int(at * n):] += shift / math.sqrt(1 - phi * phi)
        sh = detect_shift(x, np.zeros(n, int), 1, 1, FAST, rng)
        ps.append(sh.p)
        errors.append(sh.tau - int(at * n))
        iid += iid_sup_f(x, FAST) >= 12.16
    return ps, errors, iid


def test_no_break_ar1_series_reject_near_one_percent_where_the_iid_sup_f_does_not():
    reps = 400
    ps, _, iid = rejections(60, 0.5, reps)
    assert np.mean(np.array(ps) <= 0.01) <= 0.02
    assert iid / reps > 0.10, "the phase 3 test over-rejects on the same series"


def test_a_series_noisier_in_its_first_third_still_rejects_near_one_percent():
    rng = np.random.default_rng(8)
    n, reps, hits = 150, 300, 0
    sd = np.where(np.arange(n) < n // 3, 3.0, 1.0)
    for _ in range(reps):
        x = ar1(n, 0.3, rng) * sd
        hits += detect_shift(x, np.zeros(n, int), 1, 1, FAST, rng).p <= 0.01
    assert hits / reps <= 0.02


def test_a_two_sd_shift_is_found_and_dated_within_two_periods():
    ps, errors, _ = rejections(120, 0.0, 100, shift=2.0, at=0.3)
    hit = np.array(ps) <= 0.01
    assert hit.mean() >= 0.9
    assert np.median(np.abs(np.array(errors)[hit])) <= 2


def test_direction_follows_the_shift_and_before_after_agree_with_it():
    rng = np.random.default_rng(3)
    up = rng.normal(0.001, 0.003, 200)
    up[120:] += 0.02
    down = rng.normal(0.02, 0.01, 200)
    down[80:] -= 0.02
    res = find_breaks([series_from_changes("d_up", up), series_from_changes("d_down", down)], {}, FAST)
    by = {b.dataset_id: b for b in res.breaks}
    assert by["d_up"].direction == "accelerated" and by["d_up"].after > by["d_up"].before
    assert by["d_down"].direction == "slowed" and by["d_down"].after < by["d_down"].before
    assert abs(by["d_up"].after - by["d_up"].before - 0.02) < 0.006
    assert by["d_up"].period == format_period("M", 1990 + 121 // 12, 121 % 12 + 1)


def test_short_and_flat_series_are_skipped_with_their_reason():
    rng = np.random.default_rng(4)
    short = series_from_changes("d_short", rng.normal(0, 0.01, 20))
    steps = np.zeros(100)
    steps[[10, 50]] = 0.1
    flat = series_from_changes("d_flat", steps)
    fine = series_from_changes("d_fine", rng.normal(0, 0.01, 100))
    res = find_breaks([short, flat, fine], {}, FAST)
    assert {t.series_id: t.status for t in res.tests} == {
        short.series_id: "too_short", flat.series_id: "flat", fine.series_id: "tested"}
    assert res.report["skipped"] == {"too_short": 1, "flat": 1} and res.report["tested"] == 1


def test_one_quantity_published_twice_counts_once():
    rng = np.random.default_rng(5)
    x = rng.normal(0.001, 0.01, 200)
    x[100:] += 0.02
    other = rng.normal(0.001, 0.01, 200)
    other[100:] += 0.02
    copies = [series_from_changes("d_a", x), series_from_changes("d_b", x, scale=1e6),
              series_from_changes("d_c", other)]
    res = find_breaks(copies, {}, FAST)
    dup = {b.dataset_id: b.duplicate_of for b in res.breaks}
    assert dup == {"d_a": None, "d_b": "d_a:total", "d_c": None}
    assert res.report["breaks"] == 3 and res.report["breaks_deduped"] == 2


def test_output_is_byte_identical_across_runs_and_worker_counts(tmp_path, capsys):
    rng = np.random.default_rng(6)
    series = []
    for k in range(6):
        x = rng.normal(0.001, 0.01, 90)
        x[45:] += 0.02 * (k % 2)
        series.append(series_from_changes(f"d_{k}", x))
    series.append(series_from_changes("d_annual", rng.normal(0.02, 0.01, 40), freq="A"))
    write_store(series, tmp_path / "store")
    digests = []
    for run, workers in (("a", "1"), ("b", "2"), ("c", "2")):
        assert main(["--store", str(tmp_path / "store"), "--catalog", str(tmp_path / "none.db"),
                     "--out", str(tmp_path / run), "--workers", workers]) == 0
        digests.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((tmp_path / run).iterdir())})
    assert digests[0] == digests[1] == digests[2]
    assert set(digests[0]) == {"breaks.parquet", "breaks.json", "break_tests.parquet", "breaks_report.json"}
    printed = capsys.readouterr().out
    assert "tested 7" in printed and "expected false positives" in printed


def test_real_store_subset_keeps_every_invariant():
    if not (REAL_STORE / "series_meta.parquet").exists():
        pytest.skip("whole-catalog store not present")
    meta = pd.read_parquet(REAL_STORE / "series_meta.parquet")
    keep = set(meta[meta["n_points"] >= 60]["dataset_id"].drop_duplicates().head(15))
    series = [s for s in read_store(REAL_STORE) if s.dataset_id in keep]
    cfg = Config()
    res = find_breaks(series, {}, cfg, workers=4)
    by_id = {s.series_id: s for s in series}
    tested = {t.series_id for t in res.tests if t.status == "tested"}
    assert res.report["tested"] == len(tested) > 0
    kept = {b.series_id for b in res.breaks if b.duplicate_of is None}
    for b in res.breaks:
        s = by_id[b.series_id]
        assert b.series_id in tested and b.p <= cfg.alpha and b.direction in ("accelerated", "slowed")
        assert (b.after > b.before) == (b.direction == "accelerated")
        assert min(s.points) < b.period <= max(s.points) and b.n >= cfg.min_changes
        assert b.duplicate_of is None or b.duplicate_of in kept
    assert len({b.series_id for b in res.breaks}) == len(res.breaks)
