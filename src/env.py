"""Minimal .env loader.

Existing environment variables win. The loader runs once when imported.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env(path: str | Path = ".env") -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()

