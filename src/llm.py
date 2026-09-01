"""Provider-configurable LLM helpers with safe fallbacks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from . import env as _env  # noqa: F401

DEFAULT_CHAT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass


@dataclass
class LLMResult:
    value: Any
    provenance: str


def _provider_token() -> str | None:
    return os.environ.get("GENAI_API_KEY") or os.environ.get("VISA_GENAI_TOKEN")


def _openai_token() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def _chat_model() -> str:
    return os.environ.get("VISA_GENAI_MODEL", DEFAULT_CHAT_MODEL)


def _embed_model() -> str:
    return os.environ.get("VISA_GENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def is_available() -> bool:
    return bool(_provider_token() or os.environ.get("ANTHROPIC_API_KEY"))


def _client() -> tuple[Any | None, str]:
    token = _provider_token()
    public_token = os.environ.get("ANTHROPIC_API_KEY")
    try:
        import anthropic
    except Exception as exc:
        return None, f"unavailable: anthropic import failed: {exc}"

    try:
        if token:
            base_url = os.environ.get("VISA_GENAI_BASE_URL")
            kwargs: dict[str, Any] = {"api_key": token}
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.Anthropic(**kwargs), "provider"
        if public_token:
            return anthropic.Anthropic(api_key=public_token), "anthropic"
    except Exception as exc:
        return None, f"unavailable: client init failed: {exc}"
    return None, "unavailable: no token"


def chat_text(prompt: str, *, system: str | None = None, max_tokens: int = 1200) -> LLMResult:
    # OpenAI first when a key exists — same provider as the embeddings.
    openai_key = _openai_token()
    if openai_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": os.environ.get("GOVML_CHAT_MODEL", "gpt-4o-mini"),
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system or ""},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=90,
            )
            resp.raise_for_status()
            payload = resp.json()
            content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
            if content and content.strip():
                return LLMResult(content.strip(), "openai")
        except Exception:
            # fall through to the anthropic path rather than failing the request
            pass

    client, provenance = _client()
    if client is None:
        return LLMResult(None, provenance)
    try:
        msg = client.messages.create(
            model=_chat_model(),
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return LLMResult("\n".join(parts).strip() or None, provenance)
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def compute_idf(texts: list[str]) -> dict[str, float]:
    """Corpus-wide inverse document frequency, used to down-weight boilerplate.

    Singapore open-data descriptions repeat heavy agency/legal boilerplate
    ("Singapore", "please contact", "has not undergone quality control"...).
    Without down-weighting, the hashed bag-of-words fallback embedding below
    is dominated by these shared words and every dataset collapses toward the
    same point. IDF lets distinctive topic words drive the vector instead.
    """
    doc_count = max(1, len(texts))
    doc_freq: dict[str, int] = {}
    for text in texts:
        for token in set(_tokenize(text)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {token: math.log((doc_count + 1) / (freq + 1)) + 1.0 for token, freq in doc_freq.items()}


def _deterministic_embedding(text: str, dim: int = 384, idf: dict[str, float] | None = None) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    for token in tokens or ["empty"]:
        weight = (idf.get(token, 1.0) if idf else 1.0)
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm:
        vec /= norm
    return vec.astype(np.float32).tolist()


def embed_texts(texts: list[str], *, model: str | None = None, idf: dict[str, float] | None = None) -> LLMResult:
    """Return embeddings. Falls back to deterministic local embeddings.

    `idf` only affects the local-hash fallback path; real provider embeddings
    ignore it.
    """

    model_name = model or _embed_model()

    openai_key = _openai_token()
    if openai_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={"model": model_name, "input": texts},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in data if "embedding" in item]
            if len(vectors) == len(texts):
                return LLMResult(vectors, "openai")
            return LLMResult(
                [_deterministic_embedding(t, idf=idf) for t in texts], "local-hash: incomplete openai response"
            )
        except Exception as exc:
            return LLMResult([_deterministic_embedding(t, idf=idf) for t in texts], f"local-hash: openai error: {exc}")

    token = _provider_token()
    base_url = os.environ.get("VISA_GENAI_BASE_URL")
    if token and base_url:
        try:
            url = base_url.rstrip("/") + "/embeddings"
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model_name": model_name,
                    "application_name": "datagov_ml",
                    "query": texts,
                },
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("full_model_response", {}).get("data") or payload.get("data") or []
            vectors = [item["embedding"] for item in data if "embedding" in item]
            if len(vectors) == len(texts):
                return LLMResult(vectors, "provider")
            return LLMResult(
                [_deterministic_embedding(t, idf=idf) for t in texts], "local-hash: incomplete provider response"
            )
        except Exception as exc:
            return LLMResult([_deterministic_embedding(t, idf=idf) for t in texts], f"local-hash: {exc}")
    return LLMResult([_deterministic_embedding(t, idf=idf) for t in texts], "local-hash")

