"""OpenAI-backed LLM helpers with safe local fallbacks.

Configuration is read from the environment (or a local `.env`):

- ``OPENAI_API_KEY``     required for remote chat + embeddings
- ``OPENAI_BASE_URL``    optional, any OpenAI-compatible endpoint
- ``OPENAI_CHAT_MODEL``  optional, defaults to ``gpt-4o-mini``
- ``OPENAI_EMBED_MODEL`` optional, defaults to ``text-embedding-3-small``

Without a key the module degrades to deterministic local behavior: chat
helpers return ``None`` values and embeddings fall back to a hash-based
local encoder, so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import env as _env  # noqa: F401

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass
class LLMResult:
    value: Any
    provenance: str


def _api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def _chat_model() -> str:
    return os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def _embed_model() -> str:
    return os.environ.get("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def is_available() -> bool:
    return bool(_api_key())


def _client() -> tuple[Any | None, str]:
    key = _api_key()
    if not key:
        return None, "unavailable: OPENAI_API_KEY not set"
    try:
        from openai import OpenAI
    except Exception as exc:
        return None, f"unavailable: openai import failed: {exc}"
    try:
        kwargs: dict[str, Any] = {"api_key": key}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs), "openai"
    except Exception as exc:
        return None, f"unavailable: client init failed: {exc}"


def chat_text(prompt: str, *, system: str | None = None, max_tokens: int = 1200) -> LLMResult:
    client, provenance = _client()
    if client is None:
        return LLMResult(None, provenance)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=_chat_model(),
            max_tokens=max_tokens,
            messages=messages,
        )
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return LLMResult(text or None, provenance)
    except Exception as exc:
        return LLMResult(None, f"{provenance}: {exc}")


def _first_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    match = re.search(r"\{.*\}", text, flags=re.S)
    return match.group(0) if match else None


def chat_json(prompt: str, *, system: str | None = None, max_tokens: int = 1600) -> LLMResult:
    result = chat_text(prompt, system=system, max_tokens=max_tokens)
    if result.value is None:
        return result
    try:
        obj = _first_json_object(result.value)
        if obj is None:
            return LLMResult(None, f"{result.provenance}: no json object")
        return LLMResult(json.loads(obj), result.provenance)
    except Exception as exc:
        return LLMResult(None, f"{result.provenance}: json parse failed: {exc}")


def _deterministic_embedding(text: str, dim: int = 384) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    for token in tokens or ["empty"]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm:
        vec /= norm
    return vec.astype(np.float32).tolist()


def embed_texts(texts: list[str], *, model: str | None = None) -> LLMResult:
    """Return embeddings. Falls back to deterministic local embeddings."""

    model_name = model or _embed_model()
    client, provenance = _client()
    if client is not None:
        try:
            resp = client.embeddings.create(model=model_name, input=texts)
            vectors = [item.embedding for item in resp.data]
            if len(vectors) == len(texts):
                return LLMResult(vectors, provenance)
            return LLMResult(
                [_deterministic_embedding(t) for t in texts],
                "local-hash: incomplete provider response",
            )
        except Exception as exc:
            return LLMResult([_deterministic_embedding(t) for t in texts], f"local-hash: {exc}")
    return LLMResult([_deterministic_embedding(t) for t in texts], "local-hash")
