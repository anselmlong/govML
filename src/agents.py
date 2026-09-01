"""Target proposal agents and fallback heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .llm import chat_json
from .preprocess import sample_values


@dataclass
class TargetProposal:
    target: str
    task_type: str
    confidence: float
    rationale: str


@dataclass
class TargetReport:
    provenance: str
    proposals: list[TargetProposal]
    reasoning: str
    model: str | None = None


def _schema_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    samples = sample_values(df, 5)
    return [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "samples": samples.get(col, []),
        }
        for col in df.columns
    ]


def _reject_name(name: str) -> bool:
    lower = name.lower().strip()
    if lower == "id" or bool(re.search(r"(^|[\s_-])id([\s_-]|$)|index", lower)):
        return True
    # trivial / leaky prediction targets: geographic coordinates, aggregate sums,
    # and bare year columns are not meaningful targets to model.
    if lower in {"latitude", "longitude", "location_latitude", "location_longitude", "lat", "lng", "lon"}:
        return True
    if re.fullmatch(r"(total|tot|sum|count|overall|grand_total)[\s_]?.*", lower):
        return True
    # bare 4-digit year (e.g. "2013", "2023", "year", "fy2013")
    if re.fullmatch(r"(20|19)\d{2}|fy[\s_]?(20|19)\d{2}|year", lower):
        return True
    return False


def heuristic_proposals(df: pd.DataFrame) -> TargetReport:
    scored: list[tuple[float, TargetProposal]] = []
    n_rows = len(df)
    # small datasets (annual tables, tiny aggregates) need a scaled bar:
    # requiring >30 uniques kills every column of a 26-row table.
    min_unique_reg = 2 if n_rows < 60 else 30
    for col in df.columns:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100)
        unique = int(s.nunique(dropna=True))
        if _reject_name(col) or missing_pct > 30 or unique <= 1:
            continue
        dtype = str(s.dtype)
        score = 0.0
        task = ""
        rationale = ""
        if pd.api.types.is_numeric_dtype(s) and unique >= min_unique_reg and not _reject_name(col):
            spread = float((s.max(skipna=True) or 0) - (s.min(skipna=True) or 0))
            mean = abs(float(s.mean(skipna=True) or 1))
            spread_bonus = min(5.0, spread / (mean + 1e-9))
            score = 3.0 + spread_bonus
            task = "regression"
            rationale = (
                "continuous numeric with wide spread"
                if n_rows >= 60
                else "numeric series in a small table — predict value from row context"
            )
        elif unique == 2:
            score = 2.0
            task = "binary_classification"
            rationale = "two observed classes"
        elif 3 <= unique <= 12 and not pd.api.types.is_numeric_dtype(s):
            score = 1.5
            task = "multiclass_classification"
            rationale = "bounded categorical outcome"
        if score > 0:
            scored.append((score, TargetProposal(col, task, min(0.9, score / 5.0), rationale)))
    # last-resort fallback: still nothing? propose the most-informative numeric
    # column anyway so tiny datasets produce a model instead of an error.
    if not scored:
        best_col, best_unique = None, -1
        for col in df.columns:
            s = df[col]
            if _reject_name(col):
                continue
            if pd.api.types.is_numeric_dtype(s) and int(s.nunique(dropna=True)) > best_unique:
                best_col, best_unique = col, int(s.nunique(dropna=True))
        if best_col is not None and best_unique > 1:
            scored.append(
                (
                    1.0,
                    TargetProposal(
                        best_col,
                        "regression",
                        0.3,
                        "fallback: most-varied numeric column in this small dataset",
                    ),
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return TargetReport("heuristic", [p for _, p in scored[:3]], "Ranked columns by target suitability.")


def _safe_confidence(value: Any) -> float:
    """LLMs sometimes emit 'high'/'medium'/'low' instead of numbers."""
    try:
        return float(value)
    except (TypeError, ValueError):
        word = str(value or "").lower()
        return {"high": 0.85, "medium": 0.6, "low": 0.35}.get(word, 0.5)


def propose_targets(
    df: pd.DataFrame,
    *,
    research_text: str = "",
    use_llm: bool = True,
    forced_target: str | None = None,
    forced_task_type: str | None = None,
) -> TargetReport:
    if forced_target and forced_target in df.columns:
        task = forced_task_type or ("regression" if pd.api.types.is_numeric_dtype(df[forced_target]) else "multiclass_classification")
        return TargetReport(
            "user",
            [TargetProposal(forced_target, task, 1.0, "User-provided target override.")],
            "User target overrides agent proposals.",
        )

    if use_llm:
        prompt = (
            "You are a data-science agent proposing prediction targets for tabular data. "
            "Inputs are schema details, sample values, and optional recent domain context. "
            "Output valid JSON only with fields reasoning and proposals. Each proposal includes "
            "target, task_type, rationale, difficulty_hint, and confidence. Prefer continuous or "
            "bounded-cardinality targets. Avoid IDs, timestamps unless meaningful, and free text. "
            "Allowed task types: regression, binary_classification, multiclass_classification, "
            f"time_series_forecast. Schema: {_schema_summary(df)} Context: {research_text[:4000]}"
        )
        result = chat_json(prompt, max_tokens=1600)
        data = result.value
        if isinstance(data, dict) and isinstance(data.get("proposals"), list):
            proposals: list[TargetProposal] = []
            for raw in data["proposals"][:3]:
                if not isinstance(raw, dict):
                    continue
                target = str(raw.get("target") or "")
                if target not in df.columns:
                    continue
                task = str(raw.get("task_type") or "regression")
                if task == "time_series_forecast":
                    task = "regression"
                proposals.append(
                    TargetProposal(
                        target,
                        task,
                        _safe_confidence(raw.get("confidence")),
                        str(raw.get("rationale") or raw.get("difficulty_hint") or ""),
                    )
                )
            if proposals:
                return TargetReport(result.provenance, proposals, str(data.get("reasoning") or ""), model=None)

    return heuristic_proposals(df)
