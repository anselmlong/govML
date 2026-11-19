"""Fetch whole tables through data.gov.sg's bulk download endpoint into the parquet cache.

    python -m src.atlas.bulk --missing          every CSV dataset in the catalog with no cache file
    python -m src.atlas.bulk --datasets d_1,d_2

The paged list-rows API answers deep pages with 404 while it throttles, which reads as "dataset
gone". The bulk endpoint returns the full CSV in one file. Runs are resumable: a dataset with an
`_all` cache file is skipped, and failures are written to `<cache>/bulk_failures.json`.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

BASE = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/"
RATE_LIMITED = 24
OK = 0


def missing_ids(catalog: Path, cache_dir: Path) -> list[str]:
    cached = {p.name.rsplit("_", 1)[0] for p in cache_dir.glob("d_*_*.parquet")}
    with sqlite3.connect(catalog) as conn:
        rows = conn.execute("SELECT dataset_id FROM datasets WHERE upper(format) = 'CSV' ORDER BY dataset_id")
        return [d for (d,) in rows if d not in cached]


def fetch_one(dataset_id: str, cache_dir: Path, get: Callable[..., requests.Response],
              sleep: Callable[[float], None], spacing: float, attempts: int = 6) -> int:
    """Download one table to `<cache>/<id>_all.parquet` and return its row count."""
    last = "no attempt"
    for _ in range(attempts):
        sleep(spacing)
        reply = get(BASE.format(dataset_id) + "poll-download", timeout=30).json()
        code, url = reply.get("code"), (reply.get("data") or {}).get("url")
        if code == OK and url:
            body = get(url, timeout=600).content
            frame = pd.read_csv(io.BytesIO(body), low_memory=False, dtype=str)
            frame.to_parquet(cache_dir / f"{dataset_id}_all.parquet", index=False)
            (cache_dir / f"{dataset_id}_all.meta.json").write_text(json.dumps({
                "fetched_at": datetime.now(timezone.utc).isoformat(), "row_count": len(frame),
                "resource_id": dataset_id, "max_rows": None, "source": "poll-download",
            }, indent=2), encoding="utf-8")
            return len(frame)
        if code != RATE_LIMITED:
            sleep(spacing)
            get(BASE.format(dataset_id) + "initiate-download", timeout=30)
        last = f"code {code}: {reply.get('errorMsg') or reply.get('name') or ''}".strip()
    raise RuntimeError(f"no download after {attempts} polls ({last})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--missing", action="store_true")
    which.add_argument("--datasets")
    ap.add_argument("--catalog", type=Path, default=Path("catalog.db"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--spacing", type=float, default=12.0, help="seconds between API calls")
    args = ap.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ids = missing_ids(args.catalog, args.cache_dir) if args.missing else [d.strip() for d in args.datasets.split(",") if d.strip()]
    failures_path = args.cache_dir / "bulk_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}
    session = requests.Session()
    for i, dataset_id in enumerate(ids, 1):
        if (args.cache_dir / f"{dataset_id}_all.parquet").exists():
            continue
        try:
            rows = fetch_one(dataset_id, args.cache_dir, session.get, time.sleep, args.spacing)
            failures.pop(dataset_id, None)
            print(f"OK {i}/{len(ids)} {dataset_id} rows={rows}", flush=True)
        except Exception as exc:
            failures[dataset_id] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"FAIL {i}/{len(ids)} {dataset_id} {failures[dataset_id]}", flush=True)
        failures_path.write_text(json.dumps(failures, indent=1, sort_keys=True), encoding="utf-8")
    print(f"done: {len(ids)} attempted, {len(failures)} failing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
