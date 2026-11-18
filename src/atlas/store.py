"""Parquet store: `series_meta.parquet` (one row per series) and `series_points.parquet`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .series import Series

META_FILE = "series_meta.parquet"
POINTS_FILE = "series_points.parquet"


def write_store(series: Iterable[Series], out_dir: str | Path) -> tuple[int, int]:
    """Replace the store in `out_dir`. Returns (series written, points written)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta, ids, periods, values = [], [], [], []
    for s in series:
        meta.append({
            "series_id": s.series_id, "dataset_id": s.dataset_id, "label": s.label,
            "dims": json.dumps(s.dims, ensure_ascii=False), "measure": s.measure, "unit": s.unit,
            "freq": s.freq, "n_points": len(s.points), "start": next(iter(s.points)),
            "end": next(reversed(s.points)),
        })
        ids.extend([s.series_id] * len(s.points))
        periods.extend(s.points)
        values.extend(s.points.values())
    columns = ["series_id", "dataset_id", "label", "dims", "measure", "unit", "freq", "n_points", "start", "end"]
    pd.DataFrame(meta, columns=columns).to_parquet(out / META_FILE, index=False)
    pd.DataFrame({"series_id": ids, "period": periods, "value": pd.array(values, dtype="float64")}).to_parquet(
        out / POINTS_FILE, index=False)
    return len(meta), len(ids)


def read_store(out_dir: str | Path) -> list[Series]:
    out = Path(out_dir)
    meta = pd.read_parquet(out / META_FILE)
    points = pd.read_parquet(out / POINTS_FILE)
    by_id = {sid: g.sort_values("period") for sid, g in points.groupby("series_id", sort=False)}
    out_series = []
    for r in meta.itertuples(index=False):
        g = by_id[r.series_id]
        out_series.append(Series(
            series_id=r.series_id, dataset_id=r.dataset_id, label=r.label, dims=json.loads(r.dims),
            measure=r.measure, unit=r.unit, freq=r.freq,
            points={p: float(v) for p, v in zip(g["period"], g["value"])},
        ))
    return out_series
