"""Behaviour on real cached tables. Skipped where the fetch cache is absent (CI)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.atlas.flip import main
from src.atlas.readers import Matched, Table, Unmatched, flip
from src.atlas.store import read_store

CACHE = Path(__file__).resolve().parent.parent / "cache"

EXPORTS = "d_056a886a7c7f7fc87f0cd875f426a653"
JOB_VACANCY = "d_01966fdb498fa7fa2863635e25539693"
PRICE_INDEX_Q = "d_2692545ef4aefe779b8b0bcdf910dbfd"
INPUT_OUTPUT = "d_da2c259cb79497c6b681e308515796dd"
DAILY_FX = "d_046ff8d521a218d9178178cfbfc45c2c"
HDB_RESALE_2017 = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

NAMES = {
    EXPORTS: "Merchandise Exports By Region/ Market, Monthly",
    JOB_VACANCY: "Number of Job Vacancy by Industry - Annual",
    PRICE_INDEX_Q: "Domestic Supply Price Index, By Commodity Division (2-Digit Level), Base Year 2023 = 100, Quarterly",
    INPUT_OUTPUT: "Import Content of Domestic Exports (Supply, Use and Input-Output Tables 2015)",
}


def real(dataset_id: str) -> Table:
    path = CACHE / f"{dataset_id}_10000.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not in the local cache")
    return Table(dataset_id, NAMES.get(dataset_id, dataset_id), pd.read_parquet(path))


def test_monthly_wide_exports_table_flips_to_one_monthly_series_per_row():
    table = real(EXPORTS)
    result = flip(table)
    assert isinstance(result, Matched) and result.reader == "wide_time"
    assert len(result.series) == len(table.frame) == 91
    asia = next(s for s in result.series if s.dims == {"series": "Asia"})
    assert asia.series_id == f"{EXPORTS}:asia" and asia.freq == "M" and asia.measure is None
    assert asia.points["2024-09"] == 39661.9 and asia.points["2024-08"] == 40890.9
    assert "1976-01" not in asia.points, "'na' cells are not data"
    assert list(asia.points) == sorted(asia.points)
    assert asia.label == f"{NAMES[EXPORTS]}: Asia"


def test_annual_long_job_vacancy_table_flips_to_one_series_per_industry():
    result = flip(real(JOB_VACANCY))
    assert isinstance(result, Matched) and result.reader == "long_time"
    manufacturing = next(s for s in result.series if s.dims == {"industry": "manufacturing"})
    assert manufacturing.series_id == f"{JOB_VACANCY}:manufacturing:job-vacancy"
    assert manufacturing.freq == "A" and manufacturing.measure == "job_vacancy"
    assert manufacturing.points["1998"] == 5200.0 and manufacturing.points["1999"] == 6000.0
    assert len({s.series_id for s in result.series}) == len(result.series)


def test_quarterly_wide_table_reads_compact_quarter_headers_and_the_index_base_unit():
    result = flip(real(PRICE_INDEX_Q))
    assert isinstance(result, Matched)
    meat = next(s for s in result.series if s.dims["series"] == "Meat & Meat Preparations")
    assert meat.freq == "Q" and meat.unit == "Index (2023=100)"
    assert meat.points["2026Q1"] == 104.471 and meat.points["2026Q2"] == 105.502


def test_input_output_snapshot_is_unmatched_with_a_reason():
    result = flip(real(INPUT_OUTPUT))
    assert isinstance(result, Unmatched)
    assert result.reason == "snapshot" and result.dataset_id == INPUT_OUTPUT and result.detail


def test_daily_table_is_unmatched_as_sub_monthly():
    result = flip(real(DAILY_FX))
    assert isinstance(result, Unmatched) and result.reason == "sub_monthly_time"


def test_transaction_level_resale_table_aggregates_to_monthly_medians_and_counts():
    path = CACHE / f"{HDB_RESALE_2017}_all.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not in the local cache")
    frame = pd.read_parquet(path)
    result = flip(Table(HDB_RESALE_2017, "Resale flat prices", frame))
    assert isinstance(result, Matched) and result.reader == "records"
    by_id = {s.series_id: s for s in result.series}
    count = by_id[f"{HDB_RESALE_2017}:all:count"]
    assert sum(count.points.values()) == len(frame)
    price = by_id[f"{HDB_RESALE_2017}:all:median-resale-price"]
    jan = frame[frame["month"] == "2017-01"]["resale_price"].astype(float).median()
    assert price.freq == "M" and price.points["2017-01"] == jan
    assert any(s.dims == {"flat_type": "4 ROOM"} for s in result.series)
    assert not any("block" in s.dims or "street_name" in s.dims for s in result.series)


def test_cli_writes_a_store_that_reads_back_and_a_coverage_report(tmp_path):
    for dataset_id in (EXPORTS, JOB_VACANCY, INPUT_OUTPUT):
        real(dataset_id)
    out = tmp_path / "atlas"
    ids = ",".join((EXPORTS, JOB_VACANCY, INPUT_OUTPUT))
    assert main(["--datasets", ids, "--cache-dir", str(CACHE), "--catalog", str(tmp_path / "none.db"), "--out", str(out)]) == 0

    series = {s.series_id: s for s in read_store(out)}
    assert series[f"{EXPORTS}:asia"].points["2024-09"] == 39661.9
    assert series[f"{JOB_VACANCY}:manufacturing:job-vacancy"].freq == "A"

    meta = pd.read_parquet(out / "series_meta.parquet")
    points = pd.read_parquet(out / "series_points.parquet")
    assert list(points.columns) == ["series_id", "period", "value"]
    assert meta.series_id.is_unique and not points.duplicated(["series_id", "period"]).any()
    assert meta.n_points.sum() == len(points)

    report = json.loads((out / "coverage.json").read_text())
    assert report["matched_by_reader"] == {"wide_time": 1, "long_time": 1}
    assert report["unmatched_by_reason"] == {"snapshot": 1}
    assert report["series_by_freq"] == {"M": 91, "A": report["series_by_freq"]["A"]}
    assert (out / "coverage.md").read_text().startswith("# Atlas flip coverage")
    assert [u["dataset_id"] for u in report["unmatched"]] == [INPUT_OUTPUT]
