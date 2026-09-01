"""EDA and generic preprocessing helpers."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
    # rich-EDA additions (all optional; reports render what exists)
    target_histogram: str = ""          # data URI of target distribution
    target_trend: str = ""              # data URI of target vs time (if time column)
    feature_correlations: list[dict[str, Any]] = field(default_factory=list)  # top |r| with target
    outliers: dict[str, list[Any]] = field(default_factory=dict)  # col -> IQR-outlier row values
    cardinality_notes: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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

    # ---------- rich EDA (all best-effort; failures never block the run) ----------
    target_histogram = ""
    target_trend = ""
    feature_correlations: list[dict[str, Any]] = []
    outliers: dict[str, list[Any]] = {}
    cardinality_notes: list[str] = []

    try:
        if target and target in df.columns:
            s_num = pd.to_numeric(df[target], errors="coerce").dropna()
            if len(s_num) >= 3:
                # distribution of the target
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.hist(s_num, bins=min(20, max(5, len(s_num) // 2)), color="#3f7d52", edgecolor="white")
                ax.set_title(f"Distribution of {target}")
                ax.set_xlabel(target)
                ax.set_ylabel("count")
                target_histogram = fig_to_data_uri(fig)

                # numeric feature correlations with the target
                corrs: list[tuple[str, float]] = []
                for col in df.columns:
                    if col == target:
                        continue
                    col_num = pd.to_numeric(df[col], errors="coerce")
                    if col_num.notna().sum() < 3:
                        continue
                    joined = pd.concat([s_num, col_num], axis=1, join="inner").dropna()
                    if len(joined) >= 3:
                        r = float(joined.corr().iloc[0, 1])
                        if np.isfinite(r):
                            corrs.append((col, r))
                corrs.sort(key=lambda t: abs(t[1]), reverse=True)
                feature_correlations = [{"feature": c, "r": round(r, 3)} for c, r in corrs[:8]]

            # trend over time when a time axis exists
            if time_range.get("column"):
                tcol = time_range["column"]
                parsed_t = pd.to_datetime(df[tcol], errors="coerce")
                frame = pd.DataFrame({"t": parsed_t, "y": s_num}).dropna().sort_values("t")
                if len(frame) >= 3:
                    fig, ax = plt.subplots(figsize=(7, 3.2))
                    ax.plot(frame["t"], frame["y"], marker="o", ms=3.5, lw=1.4, color="#2f6e7f")
                    ax.set_title(f"{target} over {tcol}")
                    ax.set_ylabel(target)
                    target_trend = fig_to_data_uri(fig)

        # IQR outliers for numeric columns (top 5 by spread)
        num_cols = [c for c in summaries["numeric_columns"]][:8]
        for col in num_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) < 8:
                continue
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
            if mask.any():
                outliers[col] = {
                    "count": int(mask.sum()),
                    "pct": round(float(mask.mean() * 100), 1),
                    "examples": [round(float(v), 3) for v in vals[mask].head(5)],
                }

        # cardinality warnings — high-cardinality categoricals often leak IDs
        for col in summaries["categorical_columns"][:20]:
            uniq = int(df[col].nunique(dropna=True))
            if uniq > 0.9 * len(df) and len(df) > 20:
                cardinality_notes.append(f"{col}: {uniq} unique values on {len(df)} rows — likely an ID, excluded from modelling")

        sample_rows = df.head(5).to_dict(orient="records")
    except Exception:
        pass

    return EDAReport(
        len(df),
        len(df.columns),
        missingness,
        target_stats,
        time_range,
        summaries,
        target_histogram=target_histogram,
        target_trend=target_trend,
        feature_correlations=feature_correlations,
        outliers=outliers,
        cardinality_notes=cardinality_notes,
        sample_rows=sample_rows,
    )


def sample_values(df: pd.DataFrame, limit: int = 5) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for col in df.columns:
        vals = df[col].dropna().astype(str).unique().tolist()[:limit]
        out[col] = vals
    return out
