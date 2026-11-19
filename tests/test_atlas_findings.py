import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.atlas.findings import Config, find, main
from src.atlas.flip import dataset_names
from src.atlas.series import Series, format_period
from src.atlas.store import read_store, write_store

STORE = Path("atlas/housing_col")


def monthly(dataset_id: str, changes: np.ndarray, start: tuple[int, int] = (2005, 1), label: str = "Total") -> Series:
    """A positive level series whose dlog is `changes`."""
    levels = 100 * np.exp(np.concatenate([[0.0], np.cumsum(changes)]))
    y, m = start
    periods = []
    for i in range(levels.size):
        yy, mm = divmod((m - 1) + i, 12)
        periods.append(format_period("M", y + yy, mm + 1))
    return Series(f"{dataset_id}:{label.lower()}", dataset_id, f"{dataset_id} Table: {label}", {"series": label}, None, None, "M",
                  dict(zip(periods, levels.tolist())))


def ar1(rng: np.random.Generator, n: int, phi: float, sd: float = 0.01) -> np.ndarray:
    e = rng.normal(0, sd, n)
    for t in range(1, n):
        e[t] += phi * e[t - 1]
    return e


def names_for(series: list[Series]) -> dict[str, str]:
    return {s.dataset_id: f"Table {s.dataset_id[2:]} Index, Monthly" for s in series}


def test_planted_lag_three_is_found_as_x_leading_y_by_three():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 243)
    y = 0.8 * np.roll(x, 3)[3:] + rng.normal(0, 0.006, 240)
    series = [monthly("d_alpha", x[3:]), monthly("d_beta", y)]
    f = find(series, names_for(series))
    (c,) = f.candidates.itertuples()
    assert (c.kind, c.lag, c.x, c.y) == ("lead", 3, "d_alpha:total", "d_beta:total")
    assert c.r_test > 0.6 and c.p_test_adj < 0.05


def test_independent_autocorrelated_random_walks_almost_never_survive():
    rng = np.random.default_rng(2)
    series = [monthly(f"d_{i:02d}", ar1(rng, 200, phi=0.6)) for i in range(21)]
    f = find(series, names_for(series))
    assert f.report["family_size"] == 210
    assert len(f.candidates) <= 0.05 * 210
    assert (f.tested["p_test"] < 0.05).mean() < 0.10, "Bartlett n_eff keeps raw p near nominal"


def test_a_republished_table_and_a_copied_series_are_dropped_by_rule_two():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 200)
    series = [
        monthly("d_cpi_m", base),
        monthly("d_cpi_q", base),
        monthly("d_copy", base + rng.normal(0, 0.0005, 200)),
        monthly("d_other", rng.normal(0, 0.01, 200)),
    ]
    names = {"d_cpi_m": "Consumer Price Index (CPI), 2024 As Base Year, Monthly",
             "d_cpi_q": "Consumer Price Index (CPI), Base Year 2019 = 100, Quarterly",
             "d_copy": "Household Prices Rebased", "d_other": "Unrelated Volume"}
    f = find(series, names)
    assert f.report["rule2_series_dropped"] == {"same_table": 1, "duplicate": 1}
    assert f.report["family_size"] == 1 and f.candidates.empty


def test_a_pair_linked_only_by_a_shared_2020_shock_is_killed_by_the_covid_gate():
    rng = np.random.default_rng(4)
    shock = np.zeros(159)
    first = (2020 - 2011) * 12 + 3 - 9
    shock[first], shock[first + 14] = -0.15, 0.15
    series = [monthly(f"d_{k}", rng.normal(0, 0.01, 159) + shock, start=(2011, 10)) for k in "ab"]
    f = find(series, names_for(series))
    (t,) = f.tested.itertuples()
    assert t.p_test_adj < 0.05 and t.r_test > 0.5, "the shock alone passes BH"
    assert t.status == "rule5_covid" and f.candidates.empty


def test_cli_output_is_byte_identical_across_runs(tmp_path, capsys):
    rng = np.random.default_rng(6)
    x = rng.normal(0, 0.01, 203)
    series = [monthly("d_alpha", x[3:]), monthly("d_beta", 0.8 * x[:200] + rng.normal(0, 0.006, 200)),
              monthly("d_gamma", ar1(rng, 200, 0.5))]
    write_store(series, tmp_path / "store")
    digests = []
    for run in ("a", "b"):
        assert main(["--store", str(tmp_path / "store"), "--catalog", str(tmp_path / "none.db"),
                     "--out", str(tmp_path / run)]) == 0
        digests.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((tmp_path / run).iterdir())})
    assert digests[0] == digests[1]
    assert set(digests[0]) == {"candidates.parquet", "candidates.json"}
    printed = capsys.readouterr().out
    assert "family size F" in printed and "smallest detectable |r|" in printed
    assert json.loads((tmp_path / "a" / "candidates.json").read_text())[0]["kind"] == "lead"


HDB_RPI = {"d_14f63e595975691e7c24a27ae4c07c79", "d_913fd3cff8b7f462cf70cf415001e02b",
           "d_f0c768860912d66940efdac9435dc046"}
SMOKE = HDB_RPI | {
    "d_97f8a2e995022d311c6c68cfda6d034c", "d_bdaff844e3ef89d39fceb962ff8f0791",
    "d_42cb05dd61cddf473a8ba82cd10c7e34", "d_6b78d625911483860e162288a4000a0c",
    "d_8bd9b9872c7fcb456b7d2891a350be34", "d_d7933d23e5fab92a086585cfb9224ba2",
    "d_37e46b203bb77387328d25882247d000", "d_dce35e6697eadb9884f2e02af764e468",
}


def test_real_theme_subset_obeys_every_rule():
    if not (STORE / "series_meta.parquet").exists():
        pytest.skip("theme store not built")
    series = [s for s in read_store(STORE) if s.dataset_id in SMOKE]
    names = {s.dataset_id: s.label.split(":")[0] for s in series} | dataset_names(Path("catalog.db"))
    cfg = Config()
    f = find(series, names, cfg)
    dataset = {s.series_id: s.dataset_id for s in series}

    tested = f.tested
    assert len(tested) and (tested["x"].map(dataset) != tested["y"].map(dataset)).all()
    hdb = tested["x"].map(dataset).isin(HDB_RPI) & tested["y"].map(dataset).isin(HDB_RPI)
    assert not hdb.any(), "the three HDB resale price index tables are one quantity"
    retail = {"d_6b78d625911483860e162288a4000a0c", "d_8bd9b9872c7fcb456b7d2891a350be34"}
    assert not (tested["x"].map(dataset).isin(retail) & tested["y"].map(dataset).isin(retail)).any()

    c = f.candidates
    assert (c["p_test_adj"] < cfg.q).all() and (c["r_test"].abs() >= cfg.min_abs_r).all()
    assert (np.sign(c["r_train"]) == np.sign(c["r_test"])).all() and (c["lag"] >= 0).all()
    assert list(c["score"]) == sorted(c["score"], reverse=True)


def test_a_link_that_only_appears_in_the_test_window_is_killed_as_an_episode():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, 240)
    coupling = np.where(np.arange(240) < 168, 0.25, 1.0)
    y = coupling * x + rng.normal(0, 0.006, 240)
    series = [monthly("d_alpha", x), monthly("d_beta", y)]
    f = find(series, names_for(series))
    (t,) = f.tested.itertuples()
    assert t.p_test_adj < 0.05 and abs(t.r_train) < 0.5 * abs(t.r_test)
    assert t.status == "rule4_episode" and f.candidates.empty


def _month_index(year: int, month: int, start: tuple[int, int] = (2011, 10)) -> int:
    """Position in a `monthly(..., start=start)` changes array of the change into (year, month)."""
    return (year - start[0]) * 12 + (month - start[1]) - 1


def test_a_link_confined_to_2020_through_2022_is_killed_because_the_window_covers_2022():
    rng = np.random.default_rng(11)
    common = np.zeros(159)
    for year in (2020, 2021, 2022):
        for month in range(1, 13):
            common[_month_index(year, month)] = rng.normal(0, 0.03)
    series = [monthly(f"d_{k}", rng.normal(0, 0.01, 159) + common, start=(2011, 10)) for k in "ab"]
    (t,) = find(series, names_for(series)).tested.itertuples()
    assert t.p_test_adj < 0.05 and t.p_spearman < 0.05, "rank check alone would pass it"
    assert t.status == "rule5_covid"
    (old,) = find(series, names_for(series), Config(covid_years=(2020, 2021))).tested.itertuples()
    assert old.status == "survived", "a 2020-2021 window lets the 2022 stretch through"


def test_a_link_carried_by_one_outlier_fails_the_rank_check_even_though_ex_covid_passes():
    rng = np.random.default_rng(12)
    outliers = np.zeros(159)
    for year, month, v in [(2013, 3, .2), (2015, 3, -.2), (2016, 6, .2), (2018, 6, -.2), (2024, 5, .4)]:
        outliers[_month_index(year, month)] = v
    series = [monthly(f"d_{k}", rng.normal(0, 0.01, 159) + outliers, start=(2011, 10)) for k in "ab"]
    (t,) = find(series, names_for(series)).tested.itertuples()
    assert t.p_test_adj < 0.05 and t.r_ex_covid > 0.9, "ex-COVID check alone would pass it"
    assert t.p_spearman > 0.05 and t.status == "rule5_covid"


def test_two_rows_of_one_table_linked_to_another_table_give_one_candidate():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 0.01, 240)
    a1 = monthly("d_alpha", x + rng.normal(0, 0.004, 240), label="Total")
    a2 = monthly("d_alpha", x + rng.normal(0, 0.004, 240), label="Imports")
    b = monthly("d_beta", x + rng.normal(0, 0.004, 240))
    f = find([a1, a2, b], names_for([a1, b]))
    assert len(f.candidates) == 1
    assert f.report["killed"].get("rule6_same_dataset_pair") == 1
