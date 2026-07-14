"""Lightweight ML suitability scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

IDENTIFIER_HINTS = ["_id", "code", "block", "postal"]


@dataclass
class SuitabilityReason:
    kind: str
    text: str


@dataclass
class SuitabilityResult:
    score: int
    tone: str
    verdict: str
    reasons: list[SuitabilityReason] = field(default_factory=list)
    recommended_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "tone": self.tone,
            "verdict": self.verdict,
            "reasons": [r.__dict__ for r in self.reasons],
            "recommended_targets": self.recommended_targets,
        }


def _is_identifier(name: str) -> bool:
    n = f" {name.lower()} "
    return (
        name.lower() == "id"
        or bool(re.search(r"(^|[_\s-])id([_\s-]|$)", name.lower()))
        or any(h in n for h in IDENTIFIER_HINTS)
    )


def _column_kind(col: dict[str, Any]) -> str:
    kind = (col.get("kind") or "").lower()
    dtype = (col.get("dtype") or col.get("type") or "").lower()
    if kind in {"numeric", "date", "text"}:
        return kind
    if any(x in dtype for x in ["int", "float", "double", "numeric", "decimal"]):
        return "numeric"
    if any(x in dtype for x in ["date", "time", "timestamp"]):
        return "date"
    return "text"


def assess_suitability(
    columns: list[dict[str, Any]] | None,
    row_count: int | None = None,
    *,
    has_datastore: bool = True,
) -> SuitabilityResult:
    if not columns:
        return SuitabilityResult(
            15,
            "marginal",
            "This looks like a download-only dataset; ML suitability is unknown until rows are fetched.",
            [SuitabilityReason("schema", "No datastore schema was available.")],
            [],
        )

    score = 0
    reasons: list[SuitabilityReason] = []
    blocks: list[SuitabilityReason] = []

    rows = int(row_count or 0)
    if rows == 0:
        reasons.append(SuitabilityReason("rows", "Row count is unknown until fetch."))
    elif rows < 30:
        blocks.append(SuitabilityReason("rows", "Fewer than 30 rows blocks useful ML."))
    elif rows < 200:
        score += 12
    elif rows < 2000:
        score += 22
    else:
        score += 30

    usable_features = 0
    numeric_targets: list[str] = []
    categorical_targets: list[str] = []
    has_time = False
    for col in columns:
        name = str(col.get("name") or col.get("title") or "")
        if not name:
            continue
        kind = _column_kind(col)
        unique = col.get("unique")
        try:
            unique_n = int(unique) if unique is not None else None
        except Exception:
            unique_n = None
        if kind == "date":
            has_time = True
        if _is_identifier(name):
            continue
        if kind in {"numeric", "date", "text"}:
            usable_features += 1
        if kind == "numeric" and (unique_n is None or unique_n > 20):
            numeric_targets.append(name)
        elif unique_n is not None and 2 <= unique_n <= 15:
            categorical_targets.append(name)

    if numeric_targets:
        score += 30
        targets = numeric_targets[:5]
    elif categorical_targets:
        score += 25
        targets = categorical_targets[:5]
    else:
        targets = []
        blocks.append(SuitabilityReason("target", "No usable numeric or bounded categorical target was found."))

    if usable_features >= 5:
        score += 20
    elif usable_features >= 2:
        score += 10
    else:
        reasons.append(SuitabilityReason("features", "Too few usable features were detected."))

    if has_time:
        score += 10
    if not has_datastore:
        score = min(score, 40)
        reasons.append(SuitabilityReason("datastore", "No CKAN datastore preview is available."))

    score = max(0, min(100, int(score)))
    if blocks:
        tone = "blocked"
    elif score >= 70:
        tone = "good"
    elif score >= 45:
        tone = "okay"
    else:
        tone = "marginal"

    verdicts = {
        "good": "Strong candidate for a quick ML run.",
        "okay": "Usable for ML with some caveats.",
        "marginal": "Possible, but inspect the schema before running.",
        "blocked": "Blocked for ML until the data shape improves.",
    }
    return SuitabilityResult(score, tone, verdicts[tone], blocks + reasons, targets)

