import pandas as pd
import pytest

from src.atlas.periods import combine_year_month, parse_header, parse_time_values
from src.atlas.series import Series, make_series_id, slug


@pytest.mark.parametrize("header, expected", [
    ("2024Sep", ("M", "2024-09")),
    (" 1976Jan ", ("M", "1976-01")),
    ("2024 1Q", ("Q", "2024Q1")),
    ("20262Q", ("Q", "2026Q2")),
    ("2024 2H", ("H", "2024H2")),
    ("2024", ("A", "2024")),
])
def test_wide_headers_become_canonical_periods(header, expected):
    assert parse_header(header) == expected


@pytest.mark.parametrize("header", ["DataSeries", "vault_id", "2024 5Q", "2024 3H", "2024Foo", "1650", "Total 2024", ""])
def test_non_period_headers_are_rejected(header):
    assert parse_header(header) is None


def parse(values):
    return parse_time_values(pd.Series(values, dtype="string"))


def test_long_time_formats():
    assert parse(["2024", "2025"])[0] == "A"
    freq, periods = parse(["2024-01", "2024-12", None])
    assert freq == "M" and periods.dropna().tolist() == ["2024-01", "2024-12"]
    assert parse(["2024-Q1", "2024q4", "2024Q2"])[1].tolist() == ["2024Q1", "2024Q4", "2024Q2"]
    assert parse(["2024-H1", "2024-H2"])[0] == "H"


def test_first_of_month_dates_are_monthly_but_daily_dates_are_not():
    freq, periods = parse(["2024-01-01", "2024-02-01T00:00:00+08:00"])
    assert freq == "M" and periods.tolist() == ["2024-01", "2024-02"]
    assert parse(["2024-01-01", "2024-01-02", "2024-01-03"]) is None


def test_a_truncated_daily_table_with_a_single_date_is_not_mistaken_for_monthly():
    assert parse(["2018-01-01"] * 50) is None


def test_invalid_months_quarters_and_years_do_not_parse():
    assert parse(["2024-13", "2024-14"]) is None
    assert parse(["2024-Q5"]) is None
    assert parse(["0999", "1000"]) is None
    assert parse(["FY2024", "FY2025"]) is None


def test_year_and_month_number_columns_combine_to_monthly():
    p = combine_year_month(pd.Series(["2005", "2005"]), pd.Series(["1", "12"]))
    assert p.tolist() == ["2005-01", "2005-12"]
    assert combine_year_month(pd.Series(["2005"]), pd.Series(["13"])) is None


def test_series_rejects_unsorted_points_and_periods_of_the_wrong_frequency():
    kw = dict(series_id="x:y", dataset_id="x", label="l", dims={}, measure=None, unit=None)
    with pytest.raises(ValueError, match="ascending"):
        Series(**kw, freq="M", points={"2024-02": 1.0, "2024-01": 2.0})
    with pytest.raises(ValueError, match="not Q periods"):
        Series(**kw, freq="Q", points={"2024-01": 1.0})
    with pytest.raises(ValueError, match="freq"):
        Series(**kw, freq="D", points={})


def test_series_ids_are_stable_slugs():
    assert slug("Food, Beverages & Tobacco") == "food-beverages-tobacco"
    assert make_series_id("d_1", {"region": "Asia"}) == "d_1:asia"
    assert make_series_id("d_1", {"a": "X Y", "b": "Z"}, "job_vacancy") == "d_1:x-y.z:job-vacancy"
    assert make_series_id("d_1", {}, "n") == "d_1:all:n"
    long_a, long_b = "word " * 40 + "alpha", "word " * 40 + "beta"
    assert make_series_id("d", {"k": long_a}) != make_series_id("d", {"k": long_b})


def test_mon_yyyy_values_parse_as_monthly():
    freq, periods = parse_time_values(pd.Series(["OCT-2017", "MAR-2018", "OCT-2017"]))
    assert freq == "M" and list(periods) == ["2017-10", "2018-03", "2017-10"]
