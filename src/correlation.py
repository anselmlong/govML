"""Cross-dataset correlation against related datasets.

Connections are derived two ways:
  1. Semantic neighbours — the most embedding-similar datasets to the current
     one (this is the primary signal: it finds *topically related* tables even
     when they can't be numerically correlated).
  2. Numeric correlation — when both datasets expose a monthly time series, the
     first-differenced series are correlated over their overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .catalog import connect, DB_PATH
from .fetch import fetch_resource
from .insights import _matrix
from .timeseries import extract_monthly_series, first_difference


@dataclass
class CorrelationResult:
    dataset_id: str
    name: str
    proxy_column: str
    overlap_months: int
    r: float
    similarity: float | None = None
    correlation_note: str = ""


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


def _semantic_neighbours(current_dataset_id: str, k: int = 8) -> list[dict]:
    """Top-k embedding-similar datasets, excluding self. Returns [] if no
    embeddings are stored yet (callers fall back to a name/text search)."""
    try:
        rows, mat = _matrix()
    except Exception:
        return []
    if mat is None or len(mat) == 0:
        return []
    idx = None
    for i, row in enumerate(rows):
        if row["dataset_id"] == current_dataset_id:
            idx = i
            break
    if idx is None or mat.shape[0] <= 1:
        return []
    vec = mat[idx]
    sims = mat @ vec  # rows are L2-normalised already
    order = np.argsort(-sims)
    out = []
    for j in order:
        if j == idx:
            continue
        out.append(
            {
                "dataset_id": rows[j]["dataset_id"],
                "name": rows[j].get("dataset_name") or rows[j]["dataset_id"],
                "similarity": float(sims[j]),
            }
        )
        if len(out) >= k:
            break
    return out


def correlate_related(
    current_df: pd.DataFrame,
    target: str,
    current_dataset_id: str,
    related: list[dict] | None = None,
    *,
    max_related: int = 6,
) -> list[CorrelationResult]:
    base = extract_monthly_series(current_df, target)

    # 1) prefer semantic neighbours; fall back to whatever the caller supplied
    neighbours = _semantic_neighbours(current_dataset_id, k=max_related)
    if not neighbours and related:
        neighbours = [{"dataset_id": r.get("dataset_id") or r.get("resource_id"),
                       "name": r.get("name") or r.get("dataset_id"),
                       "similarity": None} for r in related]

    results: list[CorrelationResult] = []
    for item in neighbours[:max_related]:
        dataset_id = item.get("dataset_id") or item.get("resource_id")
        if not dataset_id or dataset_id == current_dataset_id:
            continue
        name = str(item.get("name") or dataset_id)
        similarity = item.get("similarity")
        try:
            other = fetch_resource(str(dataset_id), max_rows=10_000)
            proxy = _proxy_numeric_column(other)
        except Exception:
            proxy = None
        row = CorrelationResult(
            dataset_id=str(dataset_id),
            name=name,
            proxy_column=proxy or "",
            overlap_months=0,
            r=0.0,
            similarity=similarity,
        )
        # numeric correlation only possible when both sides are monthly series
        if base is not None and proxy:
            try:
                series = extract_monthly_series(other, proxy)
                if series is not None:
                    aligned = pd.concat([base, first_difference(series)], axis=1, join="inner").dropna()
                    if len(aligned) >= 6:
                        r = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                        if pd.notna(r):
                            row.proxy_column = proxy
                            row.overlap_months = int(len(aligned))
                            row.r = r
                            row.correlation_note = "numeric-timeseries"
            except Exception:
                row.correlation_note = "numeric-correlation-error"
        results.append(row)

    # rank: numeric corr pairs first by |r|, then semantic by similarity
    results.sort(
        key=lambda x: (
            x.correlation_note == "numeric-timeseries",
            abs(x.r),
            x.similarity or 0.0,
        ),
        reverse=True,
    )
    return results
