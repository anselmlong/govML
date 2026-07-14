"""Monthly series extraction and detrending helpers."""

from __future__ import annotations

import pandas as pd

from .preprocess import find_time_column


def extract_monthly_series(df: pd.DataFrame, value_col: str) -> pd.Series | None:
    if value_col not in df.columns:
        return None
    time_col = find_time_column(df)
    if not time_col:
        return None
    dates = pd.to_datetime(df[time_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    frame = pd.DataFrame({"date": dates, "value": values}).dropna()
    if len(frame) < 6:
        return None
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    return frame.groupby("month")["value"].mean().sort_index()


def first_difference(series: pd.Series) -> pd.Series:
    return series.sort_index().diff().dropna()

