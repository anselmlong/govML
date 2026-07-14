"""SQLite semantic catalog for data.gov.sg datasets."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .llm import DEFAULT_EMBED_MODEL, embed_texts
from .suitability import assess_suitability

DB_PATH = Path("catalog.db")
MAP_PATH = Path("map_coords.json")
DATASETS_URL = "https://api-production.data.gov.sg/v2/public/api/datasets"
METADATA_URL = "https://api-production.data.gov.sg/v2/public/api/datasets/{dataset_id}/metadata"
LIST_ROWS_URL = "https://api-production.data.gov.sg/v2/public/api/datasets/{dataset_id}/list-rows"


@dataclass
class Dataset:
    dataset_id: str
    name: str
    description: str = ""
    agency: str = ""
    format: str = ""
    status: str = ""
    created_at: str = ""
    last_updated_at: str = ""
    coverage_start: str = ""
    coverage_end: str = ""


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            agency TEXT,
            format TEXT,
            status TEXT,
            created_at TEXT,
            last_updated_at TEXT,
            coverage_start TEXT,
            coverage_end TEXT,
            ingested_at TEXT,
            embedding BLOB,
            embedding_model TEXT,
            suitability_score INTEGER,
            suitability_tone TEXT,
            suitability_json TEXT,
            suitability_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_agency ON datasets(agency)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_last_updated ON datasets(last_updated_at)")
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
    migrations = {
        "suitability_score": "INTEGER",
        "suitability_tone": "TEXT",
        "suitability_json": "TEXT",
        "suitability_at": "TEXT",
    }
    for name, dtype in migrations.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE datasets ADD COLUMN {name} {dtype}")
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_from_raw(raw: dict[str, Any]) -> Dataset:
    return Dataset(
        dataset_id=str(raw.get("datasetId") or raw.get("dataset_id") or raw.get("id") or ""),
        name=str(raw.get("name") or "Untitled dataset"),
        description=str(raw.get("description") or ""),
        agency=str(raw.get("managedByAgencyName") or raw.get("managedBy") or raw.get("agency") or ""),
        format=str(raw.get("format") or ""),
        status=str(raw.get("status") or ""),
        created_at=str(raw.get("createdAt") or raw.get("created_at") or ""),
        last_updated_at=str(raw.get("lastUpdatedAt") or raw.get("last_updated_at") or ""),
        coverage_start=str(raw.get("coverageStart") or ""),
        coverage_end=str(raw.get("coverageEnd") or ""),
    )


def upsert_datasets(conn: sqlite3.Connection, datasets: Iterable[Dataset]) -> int:
    rows = [d for d in datasets if d.dataset_id and d.name]
    conn.executemany(
        """
        INSERT INTO datasets (
            dataset_id, name, description, agency, format, status, created_at,
            last_updated_at, coverage_start, coverage_end, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            name=excluded.name,
            description=excluded.description,
            agency=excluded.agency,
            format=excluded.format,
            status=excluded.status,
            created_at=excluded.created_at,
            last_updated_at=excluded.last_updated_at,
            coverage_start=excluded.coverage_start,
            coverage_end=excluded.coverage_end,
            ingested_at=excluded.ingested_at
        """,
        [
            (
                d.dataset_id,
                d.name,
                d.description,
                d.agency,
                d.format,
                d.status,
                d.created_at,
                d.last_updated_at,
                d.coverage_start,
                d.coverage_end,
                _now(),
            )
            for d in rows
        ],
    )
    conn.commit()
    return len(rows)


def ingest(refresh: bool = False, max_pages: int | None = None, db_path: str | Path = DB_PATH) -> int:
    conn = connect(db_path)
    if refresh:
        conn.execute("DELETE FROM datasets")
        conn.commit()

    total = 0
    page = 1
    pages = None
    while True:
        resp = requests.get(DATASETS_URL, params={"page": page}, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or {}
        raw_datasets = data.get("datasets") or []
        pages = int(data.get("pages") or pages or page)
        batch = [_dataset_from_raw(raw) for raw in raw_datasets]
        total += upsert_datasets(conn, batch)
        print(f"ingested page {page}/{pages}: {len(batch)} datasets")
        if page >= pages:
            break
        page += 1
        if max_pages and page > max_pages:
            break
    conn.close()
    print(f"ingested {total} datasets")
    return total


def _rows_to_embed(conn: sqlite3.Connection, model: str, limit: int | None, force: bool) -> list[sqlite3.Row]:
    sql = "SELECT * FROM datasets"
    params: list[Any] = []
    if not force:
        sql += " WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?"
        params.append(model)
    sql += " ORDER BY name"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(sql, params))


def embed(model: str = DEFAULT_EMBED_MODEL, batch_size: int = 64, limit: int | None = None, force: bool = False, db_path: str | Path = DB_PATH) -> int:
    conn = connect(db_path)
    rows = _rows_to_embed(conn, model, limit, force)
    done = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [f"{row['name']}. {row['description'] or ''}" for row in batch]
        result = embed_texts(texts, model=model)
        vectors = result.value or []
        for row, vector in zip(batch, vectors):
            arr = np.asarray(vector, dtype=np.float32)
            conn.execute(
                "UPDATE datasets SET embedding=?, embedding_model=? WHERE dataset_id=?",
                (arr.tobytes(), model, row["dataset_id"]),
            )
            done += 1
        conn.commit()
        print(f"embedded {done}/{len(rows)} via {result.provenance}")
    conn.close()
    return done


def _embedding_from_blob(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _all_embeddings(conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], np.ndarray]:
    rows = list(conn.execute("SELECT * FROM datasets WHERE embedding IS NOT NULL"))
    vectors = [_embedding_from_blob(row["embedding"]) for row in rows]
    rows2 = [row for row, vec in zip(rows, vectors) if vec is not None]
    vectors2 = [vec for vec in vectors if vec is not None]
    if not vectors2:
        return [], np.empty((0, 0), dtype=np.float32)
    dim = min(vec.shape[0] for vec in vectors2)
    mat = np.vstack([vec[:dim] for vec in vectors2]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return rows2, mat / norms


def _row_dict(row: sqlite3.Row, score: float | None = None) -> dict[str, Any]:
    data = {
        "dataset_id": row["dataset_id"],
        "name": row["name"],
        "description": row["description"] or "",
        "agency": row["agency"] or "",
        "format": row["format"] or "",
        "status": row["status"] or "",
        "created_at": row["created_at"] or "",
        "last_updated_at": row["last_updated_at"] or "",
        "coverage_start": row["coverage_start"] or "",
        "coverage_end": row["coverage_end"] or "",
        "suitability_score": row["suitability_score"],
        "suitability_tone": row["suitability_tone"],
    }
    if score is not None:
        data["score"] = float(score)
    return data


def _keyword_search(conn: sqlite3.Connection, query: str, top_k: int) -> list[dict[str, Any]]:
    like = f"%{query}%"
    rows = list(
        conn.execute(
            """
            SELECT * FROM datasets
            WHERE name LIKE ? OR description LIKE ? OR agency LIKE ?
            ORDER BY CASE WHEN name LIKE ? THEN 0 ELSE 1 END, last_updated_at DESC
            LIMIT ?
            """,
            (like, like, like, like, top_k),
        )
    )
    return [_row_dict(row, 0.5) for row in rows]


def search(query: str, top_k: int = 10, db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    conn = connect(db_path)
    rows, mat = _all_embeddings(conn)
    if mat.size == 0:
        out = _keyword_search(conn, query, top_k)
        conn.close()
        return out
    q = np.asarray(embed_texts([query]).value[0], dtype=np.float32)[: mat.shape[1]]
    q_norm = np.linalg.norm(q)
    if q_norm:
        q = q / q_norm
    sims = mat @ q
    order = np.argsort(-sims)[:top_k]
    out = [_row_dict(rows[i], float(sims[i])) for i in order]
    conn.close()
    return out


def related(dataset_id: str, top_k: int = 6, db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    conn = connect(db_path)
    source = conn.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
    if not source or not source["embedding"]:
        conn.close()
        return []
    rows, mat = _all_embeddings(conn)
    src_vec = _embedding_from_blob(source["embedding"])
    if src_vec is None or mat.size == 0:
        conn.close()
        return []
    q = src_vec[: mat.shape[1]].astype(np.float32)
    norm = np.linalg.norm(q)
    if norm:
        q = q / norm
    sims = mat @ q
    order = [i for i in np.argsort(-sims) if rows[i]["dataset_id"] != dataset_id][:top_k]
    out = [_row_dict(rows[i], float(sims[i])) for i in order]
    conn.close()
    return out


def stats(db_path: str | Path = DB_PATH) -> dict[str, Any]:
    conn = connect(db_path)
    total = conn.execute("SELECT COUNT(*) n FROM datasets").fetchone()["n"]
    embedded = conn.execute("SELECT COUNT(*) n FROM datasets WHERE embedding IS NOT NULL").fetchone()["n"]
    agencies = [
        {"agency": row["agency"] or "Unknown", "count": row["n"]}
        for row in conn.execute(
            "SELECT agency, COUNT(*) n FROM datasets GROUP BY agency ORDER BY n DESC LIMIT 12"
        )
    ]
    conn.close()
    return {"total": int(total), "embedded": int(embedded), "top_agencies": agencies}


def sample(limit: int = 250, db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    conn = connect(db_path)
    agencies = [
        row["agency"]
        for row in conn.execute(
            "SELECT agency, COUNT(*) n FROM datasets GROUP BY agency ORDER BY n DESC LIMIT 20"
        )
    ]
    out: list[dict[str, Any]] = []
    per = max(1, limit // max(1, len(agencies)))
    for agency in agencies:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE agency IS ? OR agency=? ORDER BY last_updated_at DESC LIMIT ?",
            (agency, agency, per),
        )
        out.extend(_row_dict(row) for row in rows)
        if len(out) >= limit:
            break
    if len(out) < limit:
        seen = {r["dataset_id"] for r in out}
        for row in conn.execute("SELECT * FROM datasets ORDER BY last_updated_at DESC LIMIT ?", (limit,)):
            if row["dataset_id"] not in seen:
                out.append(_row_dict(row))
            if len(out) >= limit:
                break
    conn.close()
    return out[:limit]


def _normalize_xy(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where(maxs - mins == 0, 1, maxs - mins)
    return (coords - mins) / span


def compute_map_coords(force: bool = False, db_path: str | Path = DB_PATH, map_path: str | Path = MAP_PATH) -> list[dict[str, Any]]:
    path = Path(map_path)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    conn = connect(db_path)
    rows, mat = _all_embeddings(conn)
    if mat.shape[0] == 0:
        data = sample(250, db_path)
        for idx, item in enumerate(data):
            item["x"] = (idx % 25) / 24 if len(data) > 1 else 0.5
            item["y"] = (idx // 25) / max(1, (len(data) // 25))
        path.write_text(json.dumps(data), encoding="utf-8")
        conn.close()
        return data
    if mat.shape[0] >= 3:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42)
            coords = reducer.fit_transform(mat)
        except Exception:
            centered = mat - mat.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ vt[:2].T
    else:
        coords = np.column_stack([np.linspace(0, 1, mat.shape[0]), np.zeros(mat.shape[0])])
    coords = _normalize_xy(np.asarray(coords, dtype=np.float32))
    data = []
    for row, xy in zip(rows, coords):
        data.append(
            {
                "dataset_id": row["dataset_id"],
                "name": row["name"],
                "description": row["description"] or "",
                "agency": row["agency"] or "",
                "suitability_score": row["suitability_score"],
                "suitability_tone": row["suitability_tone"],
                "x": float(xy[0]),
                "y": float(xy[1]),
            }
        )
    path.write_text(json.dumps(data), encoding="utf-8")
    conn.close()
    return data


def metadata(dataset_id: str) -> dict[str, Any] | None:
    try:
        resp = requests.get(METADATA_URL.format(dataset_id=dataset_id), timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data") or None
    except Exception:
        return None


def preview_rows(dataset_id: str, limit: int = 500) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        resp = requests.get(LIST_ROWS_URL.format(dataset_id=dataset_id), params={"limit": limit}, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or {}
        return list(data.get("rows") or []), data
    except Exception:
        return [], {}


def infer_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        dates = pd.to_datetime(s, errors="coerce")
        non_null = int(s.notna().sum())
        if non_null and numeric.notna().sum() / non_null >= 0.8:
            kind = "numeric"
        elif non_null and dates.notna().sum() / non_null >= 0.85:
            kind = "date"
        else:
            kind = "text"
        columns.append(
            {
                "name": col,
                "title": col,
                "dtype": str(s.dtype),
                "kind": kind,
                "non_null": non_null,
                "null_pct": float(s.isna().mean() * 100),
                "is_categorical": int(s.nunique(dropna=True)) <= 20,
                "unique": int(s.nunique(dropna=True)),
                "sample": s.dropna().astype(str).unique().tolist()[:6],
            }
        )
    return columns


def score_dataset(dataset_id: str, db_path: str | Path = DB_PATH) -> dict[str, Any]:
    meta = metadata(dataset_id) or {}
    rows, data = preview_rows(dataset_id, 500)
    columns = infer_columns(rows)
    row_count = data.get("total") or data.get("rowCount") or len(rows)
    result = assess_suitability(columns, row_count, has_datastore=bool(rows or columns))
    conn = connect(db_path)
    conn.execute(
        """
        UPDATE datasets SET suitability_score=?, suitability_tone=?, suitability_json=?, suitability_at=?
        WHERE dataset_id=?
        """,
        (result.score, result.tone, json.dumps(result.to_dict()), _now(), dataset_id),
    )
    conn.commit()
    conn.close()
    return result.to_dict() | {"dataset_id": dataset_id, "name": meta.get("name")}


def suitability_lookup(db_path: str | Path = DB_PATH) -> dict[str, Any]:
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT dataset_id, suitability_score, suitability_tone, suitability_json FROM datasets WHERE suitability_score IS NOT NULL"
    )
    out = {}
    for row in rows:
        try:
            payload = json.loads(row["suitability_json"] or "{}")
        except Exception:
            payload = {}
        payload.setdefault("score", row["suitability_score"])
        payload.setdefault("tone", row["suitability_tone"])
        out[row["dataset_id"]] = payload
    conn.close()
    return out


def score_status(db_path: str | Path = DB_PATH) -> dict[str, Any]:
    conn = connect(db_path)
    total = conn.execute("SELECT COUNT(*) n FROM datasets").fetchone()["n"]
    scored = conn.execute("SELECT COUNT(*) n FROM datasets WHERE suitability_score IS NOT NULL").fetchone()["n"]
    conn.close()
    return {"total": int(total), "scored": int(scored), "running": False}


def score_all(limit: int | None = None, resume: bool = True, rescore_below: int | None = None, db_path: str | Path = DB_PATH) -> dict[str, Any]:
    conn = connect(db_path)
    sql = "SELECT dataset_id FROM datasets"
    params: list[Any] = []
    clauses = []
    if resume:
        clauses.append("suitability_score IS NULL")
    if rescore_below is not None:
        clauses.append("(suitability_score IS NULL OR suitability_score < ?)")
        params.append(rescore_below)
    if clauses:
        sql += " WHERE " + " OR ".join(clauses)
    sql += " ORDER BY name"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    ids = [row["dataset_id"] for row in conn.execute(sql, params)]
    conn.close()
    done = 0
    for dataset_id in ids:
        try:
            score_dataset(dataset_id, db_path)
            done += 1
        except Exception:
            continue
    return {"requested": len(ids), "scored": done}


def get_dataset(dataset_id: str, db_path: str | Path = DB_PATH) -> dict[str, Any] | None:
    conn = connect(db_path)
    row = conn.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
    conn.close()
    return _row_dict(row) if row else None

