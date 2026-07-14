"""Research artifact generation with graceful fallbacks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .llm import chat_text


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "research"


def collect_research(topic: str, output_dir: str | Path = "output", *, engine_fresh: bool = False, use_llm: bool = True) -> tuple[str, str]:
    research_dir = Path(output_dir) / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    path = research_dir / f"{slugify(topic)}.md"

    if path.exists() and not engine_fresh:
        return path.read_text(encoding="utf-8"), "cache"

    if engine_fresh:
        try:
            proc = subprocess.run(
                ["last30days", topic],
                text=True,
                capture_output=True,
                timeout=90,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                text = proc.stdout.strip()
                path.write_text(text, encoding="utf-8")
                return text, "last30days"
        except Exception:
            pass

    if use_llm:
        result = chat_text(
            "Write a concise recent research briefing for predictive modeling. "
            f"Topic: {topic}. Include dated policy, demand, supply, and macro context when known.",
            max_tokens=1200,
        )
        if result.value:
            path.write_text(result.value, encoding="utf-8")
            return result.value, result.provenance

    if path.exists():
        return path.read_text(encoding="utf-8"), "cache"
    return "", "none"

