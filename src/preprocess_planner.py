"""Preprocess plan generation and execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .llm import chat_json
from .preprocess import sample_values

YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*TO\s*(-?\d+(?:\.\d+)?)", re.I)
DURATION_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*years?)?\s*(?:(\d+(?:\.\d+)?)\s*months?)?",
    re.I,
)
NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
ID_HINTS = ["_id", "block", "street", "address", "name", "code", "ref", "postal"]


@dataclass
class PlanStep:
    op: str
    column: str | None = None
    columns: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessPlan:
    provenance: str
    steps: list[PlanStep]
    reasoning: str = ""


def _looks_id_like(name: str) -> bool:
    lower = name.lower()
    return lower == "id" or "index" in lower or any(h in lower for h in ID_HINTS)


def _ratio(values: list[str], predicate) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if predicate(v.strip())) / len(values)


def _schema_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    samples = sample_values(df, 8)
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


def llm_plan(df: pd.DataFrame, target: str | None = None) -> PreprocessPlan | None:
    schema = _schema_payload(df)
    prompt = (
        "You are a data-cleaning agent. Given schema and sample values, return valid JSON only. "
        "Fields: reasoning and steps. Steps may only use these operations: coerce_numeric, "
        "parse_year_month, range_midpoint, duration_years, derive_difference, drop, "
        "drop_null_target. Ensure target numeric coercion when needed. Drop almost-unique "
        "identifiers unless informative. Drop original string columns after parsing dates or "
        f"durations. Target: {target}. Schema: {schema}"
    )
    result = chat_json(prompt, max_tokens=1600)
    data = result.value
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    steps: list[PlanStep] = []
    for raw in data["steps"]:
        if not isinstance(raw, dict):
            continue
        columns = raw.get("columns")
        if isinstance(columns, str):
            columns = [columns]
        steps.append(
            PlanStep(
                op=str(raw.get("op") or raw.get("operation") or ""),
                column=raw.get("column"),
                columns=list(columns or []),
                params=dict(raw.get("params") or {}),
            )
        )
    return PreprocessPlan(result.provenance, steps, str(data.get("reasoning") or ""))


def heuristic_plan(df: pd.DataFrame, target: str | None = None) -> PreprocessPlan:
    steps: list[PlanStep] = []
    samples = sample_values(df, 80)
    for col in df.columns:
        vals = samples.get(col, [])
        n_unique = int(df[col].nunique(dropna=True))
        if target and col == target:
            if not pd.api.types.is_numeric_dtype(df[col]) and _ratio(vals, lambda v: bool(NUMERIC_RE.match(v.replace(",", "")))) > 0.7:
                steps.append(PlanStep("coerce_numeric", column=col))
            continue

        numeric_ratio = _ratio(vals, lambda v: bool(NUMERIC_RE.match(v.replace(",", ""))))
        ym_ratio = _ratio(vals, lambda v: bool(YEAR_MONTH_RE.match(v)))
        range_ratio = _ratio(vals, lambda v: bool(RANGE_RE.search(v)))
        duration_ratio = _ratio(vals, lambda v: ("year" in v.lower() or "month" in v.lower()))

        if _looks_id_like(col) and n_unique > 100:
            steps.append(PlanStep("drop", columns=[col]))
            continue

        if ym_ratio > 0.7:
            steps.append(PlanStep("parse_year_month", column=col))
            steps.append(PlanStep("drop", columns=[col]))
            continue
        elif range_ratio > 0.5:
            steps.append(PlanStep("range_midpoint", column=col))
        elif duration_ratio > 0.5:
            steps.append(PlanStep("duration_years", column=col))
            steps.append(PlanStep("drop", columns=[col]))
            continue
        elif numeric_ratio > 0.7 and not pd.api.types.is_numeric_dtype(df[col]):
            steps.append(PlanStep("coerce_numeric", column=col))

        if not pd.api.types.is_numeric_dtype(df[col]):
            if numeric_ratio <= 0.7 and (n_unique > 60 or _looks_id_like(col)):
                steps.append(PlanStep("drop", columns=[col]))
        elif _looks_id_like(col) and n_unique > 100:
            steps.append(PlanStep("drop", columns=[col]))

    if target:
        steps.append(PlanStep("coerce_numeric", column=target))
        steps.append(PlanStep("drop_null_target", column=target))
    return PreprocessPlan("heuristic", steps, "Schema heuristic plan.")


def build_plan(df: pd.DataFrame, target: str | None = None, use_llm: bool = True) -> PreprocessPlan:
    if use_llm:
        plan = llm_plan(df, target)
        if plan and plan.steps:
            return plan
    return heuristic_plan(df, target)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _duration_to_years(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if NUMERIC_RE.match(text):
        return float(text)
    years = re.search(r"(\d+(?:\.\d+)?)\s*years?", text)
    months = re.search(r"(\d+(?:\.\d+)?)\s*months?", text)
    total = 0.0
    found = False
    if years:
        total += float(years.group(1))
        found = True
    if months:
        total += float(months.group(1)) / 12.0
        found = True
    return total if found else None


def execute_plan(df: pd.DataFrame, plan: PreprocessPlan, target: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    notes: list[str] = []
    for step in plan.steps:
        try:
            op = step.op
            col = step.column
            if op == "coerce_numeric" and col in out.columns:
                out[col] = _coerce_numeric(out[col])
                notes.append(f"coerced {col} to numeric")
            elif op == "parse_year_month" and col in out.columns:
                parsed = pd.to_datetime(out[col].astype(str), format="%Y-%m", errors="coerce")
                year_col = step.params.get("year_col", "sale_year")
                month_col = step.params.get("month_col", "sale_month")
                out[year_col] = parsed.dt.year
                out[month_col] = parsed.dt.month
                notes.append(f"parsed {col} into {year_col}/{month_col}")
            elif op == "range_midpoint" and col in out.columns:
                def midpoint(v: Any) -> float | None:
                    m = RANGE_RE.search(str(v))
                    return (float(m.group(1)) + float(m.group(2))) / 2.0 if m else None

                new_col = step.params.get("output_col", f"{col}_midpoint")
                out[new_col] = out[col].map(midpoint)
                notes.append(f"derived {new_col} from {col}")
            elif op == "duration_years" and col in out.columns:
                new_col = step.params.get("output_col", f"{col}_years")
                out[new_col] = out[col].map(_duration_to_years)
                notes.append(f"derived {new_col} from {col}")
            elif op == "derive_difference":
                cols = step.columns or []
                if len(cols) >= 2 and cols[0] in out.columns and cols[1] in out.columns:
                    new_col = step.params.get("output_col", f"{cols[0]}_minus_{cols[1]}")
                    out[new_col] = _coerce_numeric(out[cols[0]]) - _coerce_numeric(out[cols[1]])
                    notes.append(f"derived {new_col}")
            elif op == "drop":
                drops = [c for c in (step.columns or ([col] if col else [])) if c in out.columns and c != target]
                if drops:
                    out = out.drop(columns=drops)
                    notes.append(f"dropped {', '.join(drops)}")
            elif op == "drop_null_target":
                target_col = col or target
                if target_col and target_col in out.columns:
                    before = len(out)
                    out = out.dropna(subset=[target_col])
                    notes.append(f"dropped {before - len(out)} rows with null {target_col}")
            else:
                notes.append(f"skipped unknown op {op}")
        except Exception as exc:
            notes.append(f"error in {step.op}: {exc}")
    return out, notes
