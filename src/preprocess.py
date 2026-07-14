"""EDA and generic preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import re


@dataclass
class EDAReport:
    row_count: int
    column_count: int
    missingness: dict[str, float]
    target_stats: dict[str, Any] = field(default_factory=dict)
    time_range: dict[str, str] = field(default_factory=dict)
    summaries: dict[str, Any] = field(default_factory=dict)


def find_time_column(df: pd.DataFrame) -> str | None:
    best_col: str | None = None
    best_ratio = 0.0
    for col in df.columns:
        name = col.lower()
        name_hint = any(x in name for x in ["date", "time", "month", "year", "quarter"])
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not name_hint:
            continue
        if pd.api.types.is_numeric_dtype(series) and "year" in name:
            numeric = pd.to_numeric(series, errors="coerce")
            if not numeric.dropna().between(1900, 2100).mean() >= 0.85:
                continue
            parsed = pd.to_datetime(numeric.dropna().astype(int).astype(str), format="%Y", errors="coerce")
            ratio = float(parsed.notna().sum() / max(1, int(series.notna().sum())))
            if ratio >= 0.85 and ratio > best_ratio:
                best_col = col
                best_ratio = ratio
            continue
        samples = series.dropna().astype(str).head(30).tolist()
        sample_hint = any(
            re.search(r"\d{4}[-/]\d{1,2}", value) or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", value)
            for value in samples
        )
        if not name_hint and not sample_hint:
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        denom = max(1, int(series.notna().sum()))
        ratio = float(parsed.notna().sum() / denom)
        if ratio >= 0.85 and ratio > best_ratio:
            best_col = col
            best_ratio = ratio
    return best_col


def build_eda_report(df: pd.DataFrame, target: str | None = None) -> EDAReport:
    missingness = {col: float(df[col].isna().mean()) for col in df.columns}
    target_stats: dict[str, Any] = {}
    if target and target in df.columns:
        s = df[target].dropna()
        target_stats["n_unique"] = int(s.nunique())
        if pd.api.types.is_numeric_dtype(s):
            target_stats.update(
                {
                    "mean": float(s.mean()) if len(s) else None,
                    "median": float(s.median()) if len(s) else None,
                    "min": float(s.min()) if len(s) else None,
                    "max": float(s.max()) if len(s) else None,
                }
            )

    time_range: dict[str, str] = {}
    time_col = find_time_column(df)
    if time_col:
        parsed = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if len(parsed):
            time_range = {
                "column": time_col,
                "min": parsed.min().date().isoformat(),
                "max": parsed.max().date().isoformat(),
            }

    summaries = {
        "numeric_columns": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "categorical_columns": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
    }
    return EDAReport(len(df), len(df.columns), missingness, target_stats, time_range, summaries)


def sample_values(df: pd.DataFrame, limit: int = 5) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for col in df.columns:
        vals = df[col].dropna().astype(str).unique().tolist()[:limit]
        out[col] = vals
    return out
