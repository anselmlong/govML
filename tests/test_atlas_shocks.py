import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.atlas.series import Series, format_period
from src.atlas.shocks import Config, family_key, find_shocks, main, merge_copies, run, tracks

FAST = Config(reps=2000)
REAL_STORE = Path(os.environ.get("GOVML_TEST_STORE", "/Users/anselm/src/govml-flipper/atlas/all"))
REAL_CATALOG = REAL_STORE.parent.parent / "catalog.db"
N = 120


def quarterly(dataset_id: str, label: str, changes: np.ndarray) -> Series:
    levels = 100 * np.exp(np.concatenate([[0.0], np.cumsum(changes)]))
    periods = [format_period("Q", 1990 + i // 4, i % 4 + 1) for i in range(levels.size)]
    return Series(f"{dataset_id}:{label.lower()}", dataset_id, f"{dataset_id}: {label}", {"series": label}, None, None,
                  "Q", dict(zip(periods, levels.tolist())))


def families(rng: np.random.Generator, n_families: int, per_family: int = 3, common: float = 0.8,
             plant: dict[int, np.ndarray] | None = None) -> tuple[list[Series], dict[str, str]]:
    """Each family is one dataset whose series share a heavy-tailed factor (t with 3 df), so its
    series jump together, and families are independent. `plant` adds a change path to every series
    of the given families."""
    series, names = [], {}
    for f in range(n_families):
        d = f"d_{f:03d}"
        names[d] = f"Synthetic Table {f:03d}, Quarterly"
        factor = rng.standard_t(3, N)
        for k in range(per_family):
            x = 0.01 * (np.sqrt(common) * factor + np.sqrt(1 - common) * rng.standard_t(3, N))
            if plant and f in plant:
                x = x + plant[f]
            series.append(quarterly(d, f"row{k}", x))
    return series, names


def spike(at: int, size: float = 0.15, revert: bool = True) -> np.ndarray:
    x = np.zeros(N)
    x[at] = -size
    if revert:
        x[at + 1] = size
    return x


def test_a_planted_common_shock_is_found_with_its_families():
    rng = np.random.default_rng(1)
    hit = range(8)
    series, names = families(rng, 25, plant={f: spike(60) for f in hit})
    result = run(series, names, {}, None, FAST, levels=("quarter",))
    assert sorted(e.period for e in result.events) == ["2005Q2", "2005Q3"], "the drop into 2005Q2, the rebound after"
    e = next(e for e in result.events if e.period == "2005Q2")
    assert {m.family for m in e.members} >= {family_key(names[f"d_{f:03d}"]) for f in hit}
    assert e.p_familywise <= 0.01 and e.families >= result.report["quarter"]["bar"] and e.expected < 2
    assert "no_reversion" not in e.flags and e.reversion > 0.7


def test_independent_heavy_tailed_families_give_no_event_in_most_seeds():
    """Size at the 99% family-wise bar, with series inside a family correlated as real table cuts
    are. Rotating each series alone fails this: it splits the family's joint jumps apart."""
    with_event = 0
    for seed in range(20):
        series, names = families(np.random.default_rng(100 + seed), 30)
        with_event += bool(run(series, names, {}, None, FAST, levels=("quarter",)).events)
    assert with_event <= 2


def test_copies_of_one_table_count_once():
    keys = {family_key(n) for n in [
        "Private Residential Property Price Index (Base Quarter 2009-Q1 = 100), Quarterly",
        "Private Residential Property Price Index by Market Segment (Base Quarter 2009-Q1 = 100), Quarterly",
        "Private Residential Property Price Index By Type Of Property (Base Quarter 2009-Q1 = 100), Quarterly",
        "Private Residential Property Price Index in Core Central Region (Base Quarter 2009-Q1 = 100), Quarterly",
    ]}
    assert len(keys) == 1
    assert family_key("Number of Unemployed Citizens, Quarterly") == family_key("Number of Unemployed Residents, Quarterly")
    assert family_key("Number of Overall Unemployed, Quarterly") == family_key("Number of Unemployed Residents, Quarterly")
    assert family_key("Employees on Short Work-Week or Temporary Layoff by Occupation - Quarterly") == \
        family_key("Employees on Short Work-Week or Temporary Layoff - Quarterly")

    rng = np.random.default_rng(2)
    series, names = families(rng, 25, plant={f: spike(60) for f in range(6)})
    copy = [Series(s.series_id.replace("d_000", "d_900"), "d_900", s.label, s.dims, None, None, "Q", s.points)
            for s in series if s.dataset_id == "d_000"]
    names["d_900"] = "Housing and Development Board (HDB) Republished Table"
    keyed = {s.dataset_id: family_key(names[s.dataset_id]) for s in series + copy}
    merged = merge_copies(tracks(series + copy, keyed, "quarter", FAST), FAST)
    assert merged[keyed["d_900"]] == merged[keyed["d_000"]]
    alone = run(series, names, {}, None, FAST, levels=("quarter",)).events[0]
    with_copy = run(series + copy, names, {}, None, FAST, levels=("quarter",)).events[0]
    assert with_copy.families == alone.families


def test_related_series_pushed_over_the_copy_bar_by_covid_are_not_merged():
    """r about 0.85 outside 2020-2022, above 0.9 with a shared COVID swing."""
    rng = np.random.default_rng(6)
    common = rng.standard_normal(140)
    swing = np.zeros(140)
    swing[119:131] = np.tile([-8, 6, 4, -5, 7, -3], 2)
    series = [quarterly(d, "row", 0.01 * (0.92 * common + 0.39 * rng.standard_normal(140) + swing)) for d in ("d_a", "d_b")]
    ts = tracks(series, {"d_a": "a", "d_b": "b"}, "quarter", FAST)
    assert merge_copies(ts, FAST) == {"a": "a", "b": "b"}
    assert merge_copies(ts, Config(reps=2000, covid_years=(1900, 1900))) == {"a": "a", "b": "a"}


def test_reversion_separates_a_spike_that_comes_back_from_a_step_that_stays():
    rng = np.random.default_rng(3)
    revs = {True: [], False: []}
    for trial in range(40):
        for revert in (True, False):
            x = 0.01 * rng.standard_t(3, N) + spike(60, 0.12, revert)
            s = quarterly(f"d_{trial}", "row", x)
            ts = tracks([s], {s.dataset_id: "f"}, "quarter", FAST)
            revs[revert] += [sh.reversion for sh in find_shocks(ts, "quarter", FAST) if sh.period == "2005Q2"]
    assert min(len(revs[True]), len(revs[False])) >= 38, "a t(3) draw can cancel the planted jump now and then"
    assert np.median(revs[True]) > 0.8 and np.median(revs[False]) < 0.2
    assert np.mean(np.array(revs[True]) > FAST.min_reversion) > 0.9
    assert np.mean(np.array(revs[False]) < FAST.min_reversion) > 0.9


def test_an_event_of_steps_from_one_agency_is_flagged():
    rng = np.random.default_rng(4)
    series, names = families(rng, 25, plant={f: spike(60, revert=False) for f in range(8)})
    agencies = {f"d_{f:03d}": ("Agency A" if f < 8 else f"Agency {f}") for f in range(25)}
    [e] = run(series, names, agencies, None, FAST, levels=("quarter",)).events
    assert e.flags == ["no_reversion", "one_agency"] and e.top_agency == "Agency A"


def test_a_fast_growth_stretch_is_not_a_run_of_shocks():
    rng = np.random.default_rng(5)
    x = 0.01 * rng.standard_normal(N)
    x[20:36] += 0.08
    s = quarterly("d_g", "row", x)
    ts = tracks([s], {"d_g": "f"}, "quarter", FAST)
    assert len(find_shocks(ts, "quarter", FAST)) <= 3
    seasonal_only = Config(reps=2000, local_years={"Q": 0, "M": 0, "A": 0})
    assert len(find_shocks(tracks([s], {"d_g": "f"}, "quarter", seasonal_only), "quarter", seasonal_only)) >= 12


def test_turns_attach_by_year_and_an_old_breaks_file_is_refused(tmp_path):
    series, names = families(np.random.default_rng(1), 25, plant={f: spike(60) for f in range(8)})
    breaks = tmp_path / "breaks.parquet"
    pd.DataFrame([{"series_id": "d_020:row0", "dataset_id": "d_020", "label": "x", "period": "2005Q3",
                   "direction": "slowed", "duplicate_of": None},
                  {"series_id": "d_021:row0", "dataset_id": "d_021", "label": "y", "period": "2005Q3",
                   "direction": "slowed", "duplicate_of": "d_020:row0"}]).to_parquet(breaks)
    e = next(e for e in run(series, names, {}, breaks, FAST, levels=("quarter",)).events if e.period == "2005Q2")
    assert [t.series_id for t in e.turns] == ["d_020:row0"]
    pd.DataFrame([{"series_id": "a", "supF": 1.0}]).to_parquet(breaks)
    with pytest.raises(SystemExit, match="rerun"):
        run(series, names, {}, breaks, FAST, levels=("quarter",))


def test_two_runs_write_identical_files(tmp_path):
    from src.atlas.store import write_store
    series, _ = families(np.random.default_rng(1), 25, plant={f: spike(60) for f in range(8)})
    write_store(series, tmp_path / "store")
    digests = []
    for k in range(2):
        out = tmp_path / f"out{k}"
        main(["--store", str(tmp_path / "store"), "--catalog", str(tmp_path / "none.db"), "--out", str(out),
              "--reps", "2000"])
        digests.append([hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("events.json", "shocks.parquet")])
    assert digests[0] == digests[1]


@pytest.mark.skipif(not (REAL_STORE / "series_meta.parquet").exists(), reason="real store not on this machine")
def test_real_store_finds_covid_and_counts_families_not_series():
    from src.atlas.flip import dataset_names
    from src.atlas.shocks import dataset_agencies
    from src.atlas.store import read_store
    series = read_store(REAL_STORE)
    names = {s.dataset_id: s.label.split(":")[0] for s in series} | dataset_names(REAL_CATALOG)
    result = run(series, names, dataset_agencies(REAL_CATALOG), None, FAST, levels=("quarter",))
    by_period = {e.period: e for e in result.events}
    assert "2020Q2" in by_period and by_period["2020Q2"].families >= 12
    for e in result.events:
        assert len({m.family for m in e.members}) == e.families
    assert result.report["quarter"]["families"] < result.report["quarter"]["family_keys"]
