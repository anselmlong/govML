"""Context fact records and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .preprocess import find_time_column


@dataclass
class ContextFact:
    kind: str
    date: str
    label: str
    description: str


@dataclass
class ContextPack:
    facts: list[ContextFact] = field(default_factory=list)
    provenance: str = "none"


def add_context_features(df: pd.DataFrame, pack: ContextPack) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    notes: list[str] = []
    time_col = find_time_column(out)
    if not time_col or not pack.facts:
        return out, notes

    dates = pd.to_datetime(out[time_col], errors="coerce")
    regime_dates = sorted(
        pd.to_datetime([f.date for f in pack.facts if f.kind == "regime_boundary"], errors="coerce").dropna()
    )
    if regime_dates:
        out["policy_regime"] = 0
        for idx, boundary in enumerate(regime_dates, start=1):
            out.loc[dates >= boundary, "policy_regime"] = idx
        notes.append("policy_regime")
    else:
        out["policy_regime"] = 0

    out["mop_wave_active"] = 0
    for fact in pack.facts:
        if fact.kind != "supply_wave":
            continue
        start = pd.to_datetime(fact.date, errors="coerce")
        if pd.isna(start):
            continue
        end = start + pd.DateOffset(months=12)
        out.loc[(dates >= start) & (dates <= end), "mop_wave_active"] = 1
    if any(f.kind == "supply_wave" for f in pack.facts):
        notes.append("mop_wave_active")

    return out, notes

