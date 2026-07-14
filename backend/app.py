"""FastAPI gallery backend for govML."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import catalog  # noqa: E402
from src.llm import chat_text, is_available  # noqa: E402
from src.suitability import assess_suitability  # noqa: E402

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
RUNS_FILE = ROOT / "backend" / "runs.json"
active_procs: dict[str, subprocess.Popen] = {}
score_lock = threading.Lock()
score_state: dict[str, Any] = {"running": False, "started_at": None, "completed_at": None, "last": None}

app = FastAPI(title="govML")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class RunRequest(BaseModel):
    resource_id: str
    name: str | None = None
    max_rows: int = 30_000
    no_research: bool = True
    no_correlation: bool = False
    force: bool = False


def _load_runs() -> list[dict[str, Any]]:
    if not RUNS_FILE.exists():
        RUNS_FILE.write_text("[]", encoding="utf-8")
    try:
        data = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_runs(runs: list[dict[str, Any]]) -> None:
    RUNS_FILE.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def _get_run(run_id: str) -> dict[str, Any] | None:
    for run in _load_runs():
        if run.get("run_id") == run_id:
            return run
    return None


def _update_run(run_id: str, **fields: Any) -> dict[str, Any] | None:
    runs = _load_runs()
    updated = None
    for run in runs:
        if run.get("run_id") == run_id:
            run.update(fields)
            updated = run
            break
    _save_runs(runs)
    return updated


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_done(line: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"done\. best:\s*(.*?)\s{2,}", line)
    if m:
        out["best_model"] = m.group(1).strip()
    mae = re.search(r"MAE=([-\d.eE]+)", line)
    r2 = re.search(r"R2=([-\d.eE]+)", line)
    if mae:
        out["mae"] = float(mae.group(1))
    if r2:
        out["r2"] = float(r2.group(1))
    acc = re.search(r"acc=([-\d.eE]+)", line)
    f1 = re.search(r"f1=([-\d.eE]+)", line)
    if acc:
        out["accuracy"] = float(acc.group(1))
    if f1:
        out["f1"] = float(f1.group(1))
    return out


def _drain_process(run_id: str, proc: subprocess.Popen, log_path: Path) -> None:
    final_fields: dict[str, Any] = {}
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if line.startswith("done. best:"):
                final_fields.update(_parse_done(line))
    code = proc.wait()
    active_procs.pop(run_id, None)

    report_path = OUTPUT / f"report_{run_id}.html"
    latest = OUTPUT / "report.html"
    if latest.exists():
        shutil.copyfile(latest, report_path)

    status = "completed" if code == 0 else "failed"
    _update_run(
        run_id,
        status=status,
        completed_at=_utc_now(),
        report_path=str(report_path),
        **final_fields,
    )


def _column_info_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        numeric = pd.to_numeric(s, errors="coerce")
        dates = pd.to_datetime(s, errors="coerce")
        if non_null and numeric.notna().sum() / non_null >= 0.8:
            kind = "numeric"
            info = {
                "min": float(numeric.min()) if numeric.notna().any() else None,
                "max": float(numeric.max()) if numeric.notna().any() else None,
                "mean": float(numeric.mean()) if numeric.notna().any() else None,
            }
        elif non_null and dates.notna().sum() / non_null >= 0.85:
            kind = "date"
            info = {
                "min": dates.min().date().isoformat() if dates.notna().any() else None,
                "max": dates.max().date().isoformat() if dates.notna().any() else None,
            }
        else:
            kind = "text"
            info = {"sample": s.dropna().astype(str).unique().tolist()[:8]}
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
                **info,
            }
        )
    return columns


def _ckan_preview(resource_id: str, sample_rows: int) -> tuple[list[dict[str, Any]], int | None, list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        resp = requests.get(
            "https://data.gov.sg/api/action/datastore_search",
            params={"resource_id": resource_id, "limit": sample_rows, "offset": 0},
            timeout=45,
        )
        if resp.status_code == 404:
            return [], None, [], [f"CKAN resource not found: {resource_id}"]
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("success") is not True:
            errors.append(str(payload.get("error") or payload))
            return [], None, [], errors
        result = payload.get("result") or {}
        records = result.get("records") or []
        fields = result.get("fields") or []
        return records, result.get("total"), fields, errors
    except Exception as exc:
        return [], None, [], [str(exc)]


def _size_human(size: Any) -> str:
    try:
        n = float(size)
    except Exception:
        return ""
    units = ["B", "KB", "MB", "GB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.1f} {units[i]}"


@app.get("/api/catalog/search")
def api_catalog_search(q: str = "", top_k: int = 10) -> list[dict[str, Any]]:
    if not q.strip():
        return []
    return catalog.search(q, top_k=top_k)


@app.get("/api/catalog/stats")
def api_catalog_stats() -> dict[str, Any]:
    return catalog.stats()


@app.get("/api/catalog/sample")
def api_catalog_sample(limit: int = 250) -> list[dict[str, Any]]:
    return catalog.sample(limit)


@app.get("/api/catalog/map")
def api_catalog_map(force: bool = False) -> list[dict[str, Any]]:
    return catalog.compute_map_coords(force=force)


@app.get("/api/catalog/suitability")
def api_catalog_suitability() -> dict[str, Any]:
    return catalog.suitability_lookup()


@app.get("/api/catalog/score-status")
def api_score_status() -> dict[str, Any]:
    status = catalog.score_status()
    status.update(score_state)
    return status


def _score_all_job(limit: int | None, resume: bool, rescore_below: int | None) -> None:
    with score_lock:
        score_state.update({"running": True, "started_at": _utc_now(), "completed_at": None, "last": None})
    try:
        result = catalog.score_all(limit=limit, resume=resume, rescore_below=rescore_below)
        score_state["last"] = result
    finally:
        score_state.update({"running": False, "completed_at": _utc_now()})


@app.post("/api/catalog/score-all", status_code=202)
def api_score_all(background: BackgroundTasks, limit: int | None = None, resume: bool = True, rescore_below: int | None = None) -> dict[str, Any]:
    if score_state.get("running"):
        raise HTTPException(409, "catalog scoring is already in progress")
    background.add_task(_score_all_job, limit, resume, rescore_below)
    return {"started": True}


@app.get("/api/datasets/{resource_id}/info")
def api_dataset_info(resource_id: str, sample_rows: int = 500) -> dict[str, Any]:
    meta = catalog.metadata(resource_id) if resource_id.startswith("d_") else None
    local = catalog.get_dataset(resource_id)
    records, total, fields, errors = _ckan_preview(resource_id, sample_rows)
    has_datastore = bool(records or fields)
    if not records and resource_id.startswith("d_"):
        records, data = catalog.preview_rows(resource_id, sample_rows)
        total = data.get("total") or data.get("rowCount") or len(records)
        has_datastore = bool(records)

    df = pd.DataFrame(records)
    columns = _column_info_from_df(df) if not df.empty else []
    if not columns and meta and isinstance(meta.get("columnMetadata"), dict):
        mapping = meta["columnMetadata"].get("metaMapping") or {}
        for raw in mapping.values():
            columns.append(
                {
                    "name": raw.get("name") or raw.get("columnTitle") or "",
                    "title": raw.get("columnTitle") or raw.get("name") or "",
                    "dtype": str(raw.get("dataType") or ""),
                    "kind": "text",
                    "non_null": 0,
                    "null_pct": 0,
                    "is_categorical": False,
                    "unique": None,
                    "sample": [],
                }
            )

    suitability = assess_suitability(columns, total or len(records), has_datastore=has_datastore).to_dict()
    name = (meta or {}).get("name") or (local or {}).get("name") or resource_id
    description = (meta or {}).get("description") or (local or {}).get("description") or ""
    agency = (meta or {}).get("managedByAgencyName") or (meta or {}).get("managedBy") or (local or {}).get("agency") or ""
    return {
        "resource_id": resource_id,
        "name": name,
        "description": description,
        "agency": agency,
        "url": f"https://data.gov.sg/datasets/{resource_id}/view" if resource_id.startswith("d_") else "",
        "format": (meta or {}).get("format") or (local or {}).get("format") or "",
        "coverage_start": (meta or {}).get("coverageStart") or (local or {}).get("coverage_start") or "",
        "coverage_end": (meta or {}).get("coverageEnd") or (local or {}).get("coverage_end") or "",
        "last_updated": (meta or {}).get("lastUpdatedAt") or (local or {}).get("last_updated_at") or "",
        "created_at": (meta or {}).get("createdAt") or (local or {}).get("created_at") or "",
        "frequency": (meta or {}).get("frequency") or "",
        "contact_emails": (meta or {}).get("contactEmails") or [],
        "size_bytes": (meta or {}).get("size") or None,
        "size_human": _size_human((meta or {}).get("size")),
        "row_count_total": total,
        "sample_rows": records[:sample_rows],
        "column_count": len(columns),
        "columns": columns,
        "has_datastore": has_datastore,
        "errors": errors,
        "suitability": suitability,
    }


@app.get("/api/datasets/{resource_id}/preview")
def api_dataset_preview(resource_id: str, sample_rows: int = 500) -> dict[str, Any]:
    return api_dataset_info(resource_id, sample_rows)


@app.post("/api/ask")
def api_ask(req: AskRequest) -> dict[str, Any]:
    datasets = catalog.search(req.query, top_k=req.top_k)
    runs = [r for r in _load_runs() if r.get("status") == "completed"]
    metrics = []
    for ds in datasets:
        prior = [r for r in runs if r.get("resource_id") == ds.get("dataset_id")]
        if prior:
            metrics.append({"dataset_id": ds["dataset_id"], "runs": prior[:3]})

    if is_available():
        context = "\n".join(
            f"{i+1}. {d['name']} ({d.get('agency','')}): {d.get('description','')[:500]}" for i, d in enumerate(datasets)
        )
        prompt = (
            "You are a senior data analyst familiar with Singapore government open data. "
            "Answer using only retrieved datasets. Refer to datasets by full name in bold. "
            "Do not refer to bracket indices. First paragraph gives the direct answer. "
            "Second paragraph gives correlations and analytical angles. Final line starts with "
            "Start with the selected dataset name in bold. Mention prior run metrics when available. "
            f"Keep under roughly 220 words.\nQuestion: {req.query}\nDatasets:\n{context}\nPrior metrics: {metrics}"
        )
        answer = chat_text(prompt, max_tokens=900)
        if answer.value:
            return {"answer": answer.value, "datasets": datasets, "provenance": answer.provenance}

    if datasets:
        first = datasets[0]
        text = (
            f"Start with **{first['name']}**. I found {len(datasets)} related catalog matches, "
            "but no LLM is configured, so this is a retrieval-only answer."
        )
    else:
        text = "No matching datasets were found in the local catalog."
    return {"answer": text, "datasets": datasets, "provenance": "fallback"}


@app.get("/api/runs")
def api_runs() -> list[dict[str, Any]]:
    return sorted(_load_runs(), key=lambda r: r.get("started_at", ""), reverse=True)


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict[str, Any]:
    run = _get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.post("/api/runs", status_code=202)
def api_start_run(req: RunRequest) -> dict[str, Any]:
    runs = _load_runs()
    if not req.force:
        for run in runs:
            if (
                run.get("resource_id") == req.resource_id
                and run.get("max_rows") == req.max_rows
                and run.get("no_research") == req.no_research
                and run.get("no_correlation") == req.no_correlation
                and run.get("status") in {"completed", "running"}
            ):
                return {"run_id": run["run_id"], "reused": True}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
    log_path = OUTPUT / f"log_{run_id}.txt"
    report_path = OUTPUT / f"report_{run_id}.html"
    record = {
        "run_id": run_id,
        "resource_id": req.resource_id,
        "name": req.name or req.resource_id,
        "max_rows": req.max_rows,
        "no_research": req.no_research,
        "no_correlation": req.no_correlation,
        "started_at": _utc_now(),
        "completed_at": None,
        "status": "running",
        "report_path": str(report_path),
        "log_path": str(log_path),
        "best_model": None,
        "mae": None,
        "r2": None,
    }
    runs.append(record)
    _save_runs(runs)

    cmd = [
        sys.executable,
        str(ROOT / "pipeline.py"),
        "--resource-id",
        req.resource_id,
        "--name",
        req.name or req.resource_id,
        "--topic",
        req.name or req.resource_id,
        "--max-rows",
        str(req.max_rows),
    ]
    if req.no_research:
        cmd.append("--no-research")
    if req.no_correlation:
        cmd.append("--no-correlation")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    active_procs[run_id] = proc
    thread = threading.Thread(target=_drain_process, args=(run_id, proc, log_path), daemon=True)
    thread.start()
    return {"run_id": run_id, "reused": False}


@app.get("/api/runs/{run_id}/stream")
async def api_stream_run(run_id: str) -> StreamingResponse:
    run = _get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    log_path = Path(run["log_path"])

    async def events():
        pos = 0
        while True:
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as f:
                    f.seek(pos)
                    lines = f.readlines()
                    pos = f.tell()
                for line in lines:
                    yield f"data: {line.rstrip()}\n\n"
            current = _get_run(run_id)
            if current and current.get("status") in {"completed", "failed"}:
                yield "data: " + json.dumps({"event": "status", "run": current}) + "\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/reports/{run_id}")
def api_report(run_id: str) -> FileResponse:
    path = OUTPUT / f"report_{run_id}.html"
    if not path.exists():
        raise HTTPException(404, "report not found")
    return FileResponse(path)


dist = ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

