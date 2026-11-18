import pandas as pd
import pytest

from src.atlas.readers import Matched, Table, Unmatched, flip


def table(data: dict, name="Test Table") -> Table:
    return Table("d_test", name, pd.DataFrame(data).astype("string"))


def only(result: Matched, **dims) -> object:
    (s,) = [s for s in result.series if all(s.dims.get(k) == v for k, v in dims.items())]
    return s


def test_wide_table_rows_become_series_and_junk_cells_are_dropped():
    r = flip(table({
        "vault_id": ["1", "2"],
        "DataSeries": ["Total", "  Asia"],
        "2024Mar": ["30", "1,200"], "2024Feb": ["na", "11"], "2024Jan": ["-", "10"], "2023Dec": ["8", "9.5"],
    }))
    assert isinstance(r, Matched) and r.reader == "wide_time"
    asia = only(r, series="Asia")
    assert asia.points == {"2023-12": 9.5, "2024-01": 10.0, "2024-02": 11.0, "2024-03": 1200.0}
    assert [s.dims["series"] for s in r.series] == ["Asia"] and r.skipped == {"too_few_points": 1}
    assert asia.freq == "M" and asia.measure is None and asia.series_id == "d_test:asia"


def test_wide_row_with_too_few_values_is_counted_not_silently_dropped():
    r = flip(table({"DataSeries": ["A", "B"], "2022": ["1", "na"], "2023": ["2", "na"], "2024": ["3", "5"]}))
    assert [s.dims["series"] for s in r.series] == ["A"] and r.skipped == {"too_few_points": 1}


def test_repeated_wide_labels_are_told_apart_by_their_indented_parents():
    r = flip(table({
        "vault_id": ["1", "2", "3", "4", "5", "6"],
        "DataSeries": ["Manufacturing", "  Asia", "    Japan", "Construction", "  Asia", "    Japan"],
        "2022": ["1", "2", "3", "4", "5", "6"], "2023": ["1", "2", "3", "4", "5", "6"], "2024": ["1", "2", "3", "4", "5", "6"],
    }))
    ids = [s.series_id for s in r.series]
    assert len(set(ids)) == 6
    assert only(r, section="Manufacturing", series="Asia").points["2022"] == 2.0
    assert only(r, section="Construction / Asia", series="Japan").points["2022"] == 6.0


def test_wide_rows_are_ordered_by_vault_id_before_reading_hierarchy():
    r = flip(table({
        "vault_id": ["3", "1", "2"], "DataSeries": ["  Asia", "Total", "  Asia"],
        "2022": ["1", "2", "3"], "2023": ["1", "2", "3"], "2024": ["1", "2", "3"],
    }))
    assert {s.dims["section"] for s in r.series if "section" in s.dims} == {"Total"}


def test_wide_unit_comes_from_the_row_label_and_other_parentheses_stay():
    r = flip(table({
        "DataSeries": ["Exports (Million Dollars)", "Youths (7 To 19 Years Old)"],
        "2022": ["1", "1"], "2023": ["2", "2"], "2024": ["3", "3"],
    }))
    assert only(r, series="Exports").unit == "Million Dollars"
    youths = only(r, series="Youths (7 To 19 Years Old)")
    assert youths.unit is None


def test_index_base_in_the_dataset_name_is_the_unit():
    r = flip(table({"DataSeries": ["All Items"], "2022": ["1"], "2023": ["2"], "2024": ["3"]},
                   name="Retail Sales Index, (2017 = 100), Monthly"))
    assert r.series[0].unit == "Index (2017=100)"


def test_mixed_frequency_headers_are_unmatched_with_a_reason():
    r = flip(table({"DataSeries": ["A"], "2022": ["1"], "2023": ["2"], "2024 1Q": ["3"]}))
    assert isinstance(r, Unmatched) and r.reason == "mixed_freq"


LONG = {
    "year": ["2020", "2021", "2022", "2020", "2021", "2022"],
    "industry": ["a", "a", "a", "b", "b", "b"],
    "sex": ["f", "f", "f", "f", "f", "f"],
    "value": ["1", "2", "3", "10", "na", "30"],
    "other": ["5", "6", "7", "8", "9", "10"],
}


def test_long_table_gets_one_series_per_dimension_combination_and_measure():
    r = flip(table(LONG))
    assert r.reader == "long_time" and len(r.series) == 3
    assert {tuple(s.dims.items()) for s in r.series} == {(("industry", "a"), ("sex", "f")), (("industry", "b"), ("sex", "f"))}
    pts = {s.measure: s.points for s in r.series if s.dims["industry"] == "a"}
    assert pts["value"] == {"2020": 1.0, "2021": 2.0, "2022": 3.0}
    assert r.skipped == {"too_few_points": 1}
    assert {s.series_id for s in r.series} >= {"d_test:a.f:value", "d_test:a.f:other"}


def test_long_unit_comes_from_the_measure_column_name():
    r = flip(table({"year": ["2020", "2021", "2022"], "MillionDollars": ["1", "2", "3"], "MillionDollars_Ratio": ["1", "2", "3"]}))
    assert {s.measure: s.unit for s in r.series} == {"MillionDollars": "Million Dollars", "MillionDollars_Ratio": None}


def test_exact_duplicate_rows_collapse_but_conflicting_duplicates_skip_the_series():
    clean = ["2024-Q1", "2024-Q2", "2024-Q3"]
    r = flip(table({
        "quarter": ["2024-Q1", "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q1", "2024-Q1", "2024-Q2", "2024-Q3"] + clean * 3,
        "town": ["a"] * 4 + ["b"] * 4 + ["c"] * 3 + ["d"] * 3 + ["e"] * 3,
        "rent": ["1", "1", "2", "3", "1", "9", "2", "3"] + ["5"] * 9,
    }))
    assert sorted(s.dims["town"] for s in r.series) == ["a", "c", "d", "e"]
    assert only(r, town="a").points == {"2024Q1": 1.0, "2024Q2": 2.0, "2024Q3": 3.0}
    assert r.skipped == {"duplicate_periods": 1}


def test_identifier_like_numeric_columns_are_dimensions_not_measures():
    r = flip(table({
        "year": ["2020", "2021", "2022"] * 2, "station_id": ["7"] * 3 + ["8"] * 3, "reading": ["1", "2", "3", "4", "5", "6"],
    }))
    assert sorted(s.dims["station_id"] for s in r.series) == ["7", "8"]


def test_year_with_month_number_is_monthly_and_the_finest_time_column_wins():
    r = flip(table({"year": ["2005"] * 3, "mth": ["1", "2", "3"], "mw": ["5", "6", "7"]}))
    assert r.series[0].freq == "M" and list(r.series[0].points) == ["2005-01", "2005-02", "2005-03"]


def test_unit_records_are_unmatched_not_half_read():
    n = 60
    r = flip(table({
        "month": [f"2024-{(i % 12) + 1:02d}" for i in range(n)], "block": [str(i) for i in range(n)],
        "price": [str(100 + i) for i in range(n)],
    }))
    assert isinstance(r, Unmatched) and r.reason == "no_series_survived"
    mixed = flip(table({
        "month": [f"2024-{(i % 12) + 1:02d}" for i in range(n)] + ["2024-01", "2024-02", "2024-03"] * 2,
        "block": [str(i) for i in range(n)] + ["x"] * 3 + ["y"] * 3, "price": ["1"] * (n + 6),
    }))
    assert isinstance(mixed, Unmatched) and mixed.reason == "low_row_coverage"


@pytest.mark.parametrize("data, reason", [
    ({"a": ["x", "y"], "b": ["1", "2"]}, "snapshot"),
    ({"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "v": ["1", "2", "3"]}, "sub_monthly_time"),
    ({"year": ["FY24", "FY25", "FY26"], "v": ["1", "2", "3"]}, "time_format_unrecognised"),
    ({"year": ["2020", "2021", "2022"], "name": ["a", "b", "c"]}, "no_measure"),
    ({"a": []}, "empty_table"),
])
def test_tables_no_reader_can_use_say_why(data, reason):
    r = flip(table(data))
    assert isinstance(r, Unmatched) and r.reason == reason and r.dataset_id == "d_test"
