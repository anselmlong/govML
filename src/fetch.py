"""CKAN datastore fetcher with parquet caching and retry behavior."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

CKAN_DATASTORE_SEARCH = "https://data.gov.sg/api/action/datastore_search"
V2_LIST_ROWS = "https://api-production.data.gov.sg/v2/public/api/datasets/{dataset_id}/list-rows"
INTERNAL_MAX_ROWS = 200_000
CACHE_DIR = Path("cache")


def _cache_paths(resource_id: str, max_rows: int | None, cache_dir: Path = CACHE_DIR) -> tuple[Path, Path]:
    key = max_rows if max_rows and max_rows > 0 else "all"
    safe = resource_id.replace("/", "_")
    parquet = cache_dir / f"{safe}_{key}.parquet"
    meta = cache_dir / f"{safe}_{key}.meta.json"
    return parquet, meta


def _cache_age_days(path: Path) -> float:
    return max(0.0, (time.time() - path.stat().st_mtime) / 86400)


def _read_cache(path: Path) -> pd.DataFrame:
    print(f"[cache hit] {path} (age {_cache_age_days(path):.1f}d)")
    return pd.read_parquet(path)


MAX_RETRY_ATTEMPTS = 5


class GoneError(ValueError):
    """Dataset definitively retired / removed from data.gov.sg (404). Permanent."""


class ThrottledError(RuntimeError):
    """data.gov.sg is transiently throttling us (silent empty-200 / 429 / 5xx),
    but the dataset is alive. Caller may retry later."""


def _request_page(resource_id: str, limit: int, offset: int) -> dict[str, Any]:
    attempts = 0
    while True:
        try:
            resp = requests.get(
                CKAN_DATASTORE_SEARCH,
                params={"resource_id": resource_id, "limit": limit, "offset": offset},
                timeout=30,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            attempts += 1
            if attempts >= 3:
                raise RuntimeError(f"network error fetching {resource_id}: {exc}") from exc
            time.sleep(2 ** attempts)
            continue

        if resp.status_code == 404:
            # Ambiguous: data.gov.sg occasionally answers a *live* id with 404
            # ("No table found for dataset ID") during a silent-throttle window —
            # we proved such ids flap back to 200 within seconds. So a first-seen
            # 404 is NOT proof the dataset is retired. Back off and retry; only
            # raise GoneError if the 404 persists across the retry window.
            attempts += 1
            if attempts >= MAX_RETRY_ATTEMPTS:
                raise GoneError(f"missing dataset resource: {resource_id}")
            time.sleep(min(5, 2 ** attempts))
            continue
        if resp.status_code == 413:
            return {"__status__": 413}
        if resp.status_code in {429, 502, 503, 504}:
            # Bounded: an upstream stuck in sustained rate-limiting or an
            # outage used to retry every 5s forever, permanently pinning a
            # worker thread and, for background jobs looping over many
            # datasets, eventually starving the whole API of threads.
            attempts += 1
            if attempts >= MAX_RETRY_ATTEMPTS:
                raise RuntimeError(f"data.gov.sg kept returning {resp.status_code} for {resource_id} after {attempts} attempts")
            time.sleep(5)
            continue
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("success") is not True:
            raise RuntimeError(f"CKAN datastore_search failed for {resource_id}: {payload}")
        return payload


def _request_v2_page(resource_id: str, limit: int, offset: int, next_url: str | None = None) -> dict[str, Any]:
    attempts = 0
    eff_limit = limit
    while True:
        try:
            url = next_url or V2_LIST_ROWS.format(dataset_id=resource_id)
            params = None if next_url else {"limit": eff_limit, "offset": offset}
            resp = requests.get(url, params=params, timeout=30)
        except (requests.Timeout, requests.ConnectionError) as exc:
            attempts += 1
            if attempts >= 3:
                raise RuntimeError(f"network error fetching {resource_id}: {exc}") from exc
            time.sleep(2 ** attempts)
            continue
        if resp.status_code == 404:
            # Ambiguous: data.gov.sg occasionally answers a *live* v2 id with
            # 404 ("No table found for dataset ID") mid-throttle; it flaps back
            # to 200 within seconds. A first-seen 404 is NOT proof of retirement.
            # Back off and retry; only raise GoneError if it persists.
            attempts += 1
            if attempts >= MAX_RETRY_ATTEMPTS:
                raise GoneError(f"missing dataset resource: {resource_id}")
            time.sleep(min(5, 2 ** attempts))
            continue
        if resp.status_code == 413:
            # payload too large: halve the page and retry instead of giving up
            eff_limit = max(100, eff_limit // 2)
            if eff_limit < 200:
                raise RuntimeError(f"payload too large for {resource_id} even at page={eff_limit}")
            attempts += 1
            if attempts >= 6:
                raise RuntimeError(f"payload too large for {resource_id} after shrinking pages")
            time.sleep(1)
            continue
        if resp.status_code in {429, 502, 503, 504}:
            attempts += 1
            if attempts >= MAX_RETRY_ATTEMPTS:
                raise RuntimeError(f"data.gov.sg kept returning {resp.status_code} for {resource_id} after {attempts} attempts")
            time.sleep(5)
            continue
        # data.gov.sg silently rate-limits by returning HTTP 200 with an empty/
        # HTML body (content-length: 0) instead of a clean 429. The dataset is alive;
        # this is a throttle, not a dead id. Treat it as retriable (ThrottledError)
        # rather than a permanent failure so callers can back off and retry later.
        try:
            payload = resp.json()
        except ValueError:
            attempts += 1
            if attempts >= MAX_RETRY_ATTEMPTS:
                raise ThrottledError(
                    f"data.gov.sg returned non-JSON body ({resp.status_code}) for {resource_id} "
                    f"after {attempts} attempts (throttled, dataset alive)"
                )
            time.sleep(min(60, 10 * (2 ** attempts)))
            continue
        resp.raise_for_status()
        if payload.get("code") not in {None, 0}:
            raise RuntimeError(f"data.gov.sg list-rows failed for {resource_id}: {payload}")
        return payload


def _fetch_v2_records(resource_id: str, requested: int, page_size: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    next_url: str | None = None
    printed_total = False
    while len(records) < requested:
        payload = _request_v2_page(resource_id, min(page_size, requested - len(records)), offset, next_url)
        data = payload.get("data") or {}
        page_records = data.get("rows") or []
        total = data.get("total") or data.get("rowCount")
        if total is not None and not printed_total:
            print(f"total rows available: {total}.")
            printed_total = True
        records.extend(page_records)
        offset += len(page_records)
        link = (data.get("links") or {}).get("next")
        next_url = urljoin("https://api-production.data.gov.sg", link) if link else None
        if not page_records or not next_url:
            break
    return records[:requested]


def fetch_resource(
    resource_id: str,
    max_rows: int | None = None,
    *,
    page_size: int = 10_000,
    refresh: bool = False,
    cache_ttl_days: int | None = None,
    cache_dir: str | Path = CACHE_DIR,
) -> pd.DataFrame:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    parquet, meta = _cache_paths(resource_id, max_rows, cache_root)

    if parquet.exists() and not refresh:
        stale = False
        if cache_ttl_days is not None:
            stale = _cache_age_days(parquet) > cache_ttl_days
        if not stale:
            return _read_cache(parquet)

    print(f"[cache miss] {parquet}")
    requested = INTERNAL_MAX_ROWS if not max_rows or max_rows <= 0 else min(max_rows, INTERNAL_MAX_ROWS)
    print(f"fetching resource {resource_id} (max_rows={max_rows or 0})...")

    if resource_id.startswith("d_"):
        records = _fetch_v2_records(resource_id, requested, max(100, int(page_size)))
    else:
        records = []
        offset = 0
        current_page_size = max(100, int(page_size))
        total: int | None = None
        printed_total = False

        while len(records) < requested:
            limit = min(current_page_size, requested - len(records))
            payload = _request_page(resource_id, limit, offset)
            if payload.get("__status__") == 413:
                if current_page_size <= 100:
                    raise ValueError(f"CKAN 413 for {resource_id} even at minimum page_size=100")
                current_page_size = max(100, current_page_size // 2)
                continue

            result = payload.get("result") or {}
            page_records = result.get("records") or []
            total = result.get("total", total)
            if total is not None and not printed_total:
                print(f"total rows available: {total}.")
                printed_total = True
            records.extend(page_records)
            offset += len(page_records)
            if not page_records:
                break
            if total is not None and offset >= total:
                break

    if max_rows and max_rows > 0:
        records = records[:max_rows]
    df = pd.DataFrame.from_records(records)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    df.to_parquet(parquet, index=False)
    meta.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "row_count": int(len(df)),
                "resource_id": resource_id,
                "max_rows": max_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"fetched {len(df)} rows -> cached to {parquet}.")
    return df
