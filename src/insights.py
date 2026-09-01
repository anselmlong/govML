"""Semantic search over stored ML run insights.

Insights are embedded alongside catalog datasets so that a natural-language
query can surface not just raw datasets but also what the ML pipeline learned
from them — model quality, predicted targets, and cross-dataset correlations.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.catalog import DB_PATH, connect  # noqa: E402
from src.llm import embed_texts  # noqa: E402

_CACHE: dict[str, Any] = {"fingerprint": None, "rows": None, "mat": None}


def _all_insight_embeddings(conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], np.ndarray]:
    rows = list(
        conn.execute(
            """SELECT i.*, d.name AS dataset_name, d.agency FROM run_insights i
               JOIN datasets d ON d.dataset_id = i.dataset_id
               WHERE i.embedding IS NOT NULL"""
        )
    )
    vectors = []
    for row in rows:
        try:
            vec = np.asarray(json.loads(bytes(row["embedding"]).decode()), dtype=np.float32)
        except Exception:
            vec = None
        if vec is not None:
            n = np.linalg.norm(vec)
            vectors.append(vec / n if n else vec)
    mat = np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
    return rows, mat


def _fingerprint(conn: sqlite3.Connection):
    return conn.execute("SELECT COUNT(*), MAX(updated_at) FROM run_insights WHERE embedding IS NOT NULL").fetchone()


def _matrix() -> tuple[list[sqlite3.Row], np.ndarray]:
    conn = connect(DB_PATH)
    fp = _fingerprint(conn)
    conn.close()
    if _CACHE["fingerprint"] == fp and _CACHE["mat"] is not None:
        return _CACHE["rows"], _CACHE["mat"]
    conn = connect(DB_PATH)
    rows, mat = _all_insight_embeddings(conn)
    conn.close()
    _CACHE.update({"fingerprint": fp, "rows": rows, "mat": mat})
    return rows, mat


def search_insights(query: str, top_k: int = 5, db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    """Return top-k insights whose embedded summaries best match the query."""
    query = (query or "").strip()
    if not query:
        return []
    rows, mat = _matrix()
    if not len(rows):
        return []
    emb = embed_texts([query])
    vec = (emb.value or [None])[0]
    if vec is None:
        # keyword fallback over insight text
        tokens = [t for t in query.lower().split() if len(t) > 2]
        scored = []
        for r in rows:
            text = r["insight_text"].lower()
            score = sum(text.count(t) for t in tokens)
            if score:
                scored.append((score, r))
        scored.sort(key=lambda p: -p[0])
        return [_insight_dict(r) for _, r in scored[:top_k]]
    q = np.asarray(vec, dtype=np.float32)[: mat.shape[1]]
    n = np.linalg.norm(q)
    if n:
        q = q / n
    sims = mat @ q
    order = np.argsort(-sims)[:top_k]
    return [_insight_dict(rows[i], float(sims[i])) for i in order]


def insights_for(dataset_id: str, db_path: str | Path = DB_PATH) -> dict[str, Any] | None:
    conn = connect(db_path)
    row = conn.execute(
        """SELECT i.*, d.name AS dataset_name, d.agency FROM run_insights i
           JOIN datasets d ON d.dataset_id = i.dataset_id WHERE i.dataset_id=?""",
        (dataset_id,),
    ).fetchone()
    conn.close()
    return _insight_dict(row) if row else None


def _insight_dict(row: sqlite3.Row, score: float | None = None) -> dict[str, Any]:
    try:
        metrics = json.loads(row["metrics_json"] or "{}")
    except Exception:
        metrics = {}
    try:
        correlations = json.loads(row["correlations_json"] or "[]")
    except Exception:
        correlations = []
    out = {
        "dataset_id": row["dataset_id"],
        "dataset_name": row["dataset_name"],
        "agency": row["agency"],
        "run_id": row["run_id"],
        "target": row["target"],
        "task_type": row["task_type"],
        "best_model": row["best_model"],
        "metrics": metrics,
        "verdict_tone": row["verdict_tone"],
        "verdict_text": row["verdict_text"],
        "insight_text": row["insight_text"],
        "correlations": correlations,
    }
    if score is not None:
        out["score"] = round(score, 4)
    return out
