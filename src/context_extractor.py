"""Extract ML context facts from research markdown."""

from __future__ import annotations

from .context import ContextFact, ContextPack
from .llm import chat_json


def extract_context(research_markdown: str, *, use_llm: bool = True) -> ContextPack:
    if not research_markdown.strip() or not use_llm:
        return ContextPack([], "empty")
    prompt = (
        "You are a fact-extraction agent for predictive ML context. Input is a research markdown "
        "briefing. Output valid JSON only with fields reasoning and facts. Fact categories: "
        "regime_boundary and supply_wave. Include only facts with a specific date. Use first day "
        "of month when only year-month is known. Do not invent dates. Return at most 6 facts "
        f"ordered by date. Ignore speculation. Research: {research_markdown[:8000]}"
    )
    result = chat_json(prompt, max_tokens=1200)
    data = result.value
    facts: list[ContextFact] = []
    if isinstance(data, dict) and isinstance(data.get("facts"), list):
        for raw in data["facts"][:6]:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "")
            date = str(raw.get("date") or "")
            if kind not in {"regime_boundary", "supply_wave"} or not date:
                continue
            facts.append(
                ContextFact(
                    kind=kind,
                    date=date,
                    label=str(raw.get("label") or kind),
                    description=str(raw.get("description") or ""),
                )
            )
    return ContextPack(facts, result.provenance if facts else "empty")

