"""Cross-dataset correlation against related datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .fetch import fetch_resource
from .timeseries import extract_monthly_series, first_difference


@dataclass
class CorrelationResult:
    dataset_id: str
    name: str
    proxy_column: str
    overlap_months: int
    r: float


def _proxy_numeric_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= 6 and s.nunique(dropna=True) > 3:
            candidates.append((s.notna().sum(), col))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None


def correlate_related(
    current_df: pd.DataFrame,
    target: str,
    related: list[dict],
    *,
    max_related: int = 6,
) -> list[CorrelationResult]:
    base = extract_monthly_series(current_df, target)
    if base is None:
        return []
    base_d = first_difference(base)
    results: list[CorrelationResult] = []
    for item in related[:max_related]:
        dataset_id = item.get("dataset_id") or item.get("resource_id")
        if not dataset_id:
            continue
        try:
            other = fetch_resource(str(dataset_id), max_rows=10_000)
            proxy = _proxy_numeric_column(other)
            if not proxy:
                continue
            series = extract_monthly_series(other, proxy)
            if series is None:
                continue
            aligned = pd.concat([base_d, first_difference(series)], axis=1, join="inner").dropna()
            if len(aligned) < 6:
                continue
            r = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if pd.notna(r):
                results.append(
                    CorrelationResult(
                        dataset_id=str(dataset_id),
                        name=str(item.get("name") or dataset_id),
                        proxy_column=proxy,
                        overlap_months=int(len(aligned)),
                        r=r,
                    )
                )
        except Exception:
            continue
    results.sort(key=lambda x: abs(x.r), reverse=True)
    return results

