"""Reader registry: one reader per table shape, first match wins, never a silent drop.

A reader is a `(detect, read)` pair. `detect` is a cheap yes/no on the frame. `read`
returns series, or raises `Reject` when it recognised the shape but cannot produce
series. `flip` turns either failure, or no reader matching at all, into an `Unmatched`
record carrying the reason.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .periods import FINENESS, combine_year_month, parse_header, parse_time_values
from .series import Freq, Series, make_series_id
from .units import split_label_unit, unit_from_dataset_name, unit_from_text

MIN_POINTS = 3
ID_COLUMNS = {"vault_id", "_id"}
MIN_PERIOD_COLUMNS = 3
MIN_PERIOD_SHARE = 0.5
TIME_NAMES = {
    "year", "financial_year", "fy", "yr", "month", "quarter", "qtr", "period", "date", "half",
    "calendar_year", "academic_year", "year_of_assessment", "year_of_balance", "shs_year",
}
MONTH_NUMBER_NAMES = {"mth", "month", "mon"}
TIMESTAMP_NAMES = {"timestamp", "update_timestamp", "datetime", "date_time"}
NOT_A_MEASURE = re.compile(r"(^|_)(id|code|postal|lat|lng|lon|long|latitude|longitude|age|rank)($|_)")
NUMERIC_SHARE = 0.9
MIN_ROW_COVERAGE = 0.5

SKIP_TOO_FEW_POINTS = "too_few_points"
SKIP_DUPLICATE_PERIODS = "duplicate_periods"


@dataclass(frozen=True)
class Table:
    dataset_id: str
    name: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class Matched:
    dataset_id: str
    reader: str
    series: tuple[Series, ...]
    skipped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Unmatched:
    dataset_id: str
    reason: str
    detail: str = ""


class Reject(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class Reader:
    name: str
    detect: Callable[[Table], bool]
    read: Callable[[Table], tuple[list[Series], Counter[str]]]


def to_number(col: pd.Series) -> pd.Series:
    """Parse a string column to floats; `na`, `-`, blanks and other junk become NaN."""
    s = col.astype("string").str.strip().str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce").astype("float64")
    return out.where(np.isfinite(out))


def _data_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS]


def _row_order(df: pd.DataFrame) -> pd.DataFrame:
    if "vault_id" in df.columns:
        order = pd.to_numeric(df["vault_id"], errors="coerce")
        return df.loc[order.sort_values(kind="stable").index]
    return df


def _unique_ids(series: list[Series]) -> list[Series]:
    """Two distinct dim combinations can slug to one id; later ones get `-2`, `-3`."""
    seen: Counter[str] = Counter()
    out = []
    for s in series:
        seen[s.series_id] += 1
        if seen[s.series_id] > 1:
            s = replace(s, series_id=f"{s.series_id}-{seen[s.series_id]}")
        out.append(s)
    return out


def _period_columns(df: pd.DataFrame) -> dict[str, tuple[Freq, str]]:
    return {c: p for c in _data_columns(df) if (p := parse_header(c))}


# ---------------------------------------------------------------- wide_time

def detect_wide_time(table: Table) -> bool:
    cols = _data_columns(table.frame)
    periods = _period_columns(table.frame)
    return len(periods) >= MIN_PERIOD_COLUMNS and len(periods) >= MIN_PERIOD_SHARE * len(cols)


def _indent(raw: object) -> int:
    text = str(raw)
    return len(text) - len(text.lstrip())


def _ancestors(df: pd.DataFrame, label_col: str) -> list[str]:
    """SingStat encodes hierarchy by indenting `DataSeries` labels. For each row, the
    chain of shallower rows above it, as `A / B`; empty for top-level rows."""
    stack: list[tuple[int, str]] = []
    out = []
    for raw in df[label_col]:
        depth, label = _indent(raw), str(raw).strip()
        while stack and stack[-1][0] >= depth:
            stack.pop()
        out.append(" / ".join(name for _, name in stack))
        stack.append((depth, label))
    return out


def _wide_dims(df: pd.DataFrame, label_cols: list[str]) -> list[dict[str, str]]:
    """Dims per row. SingStat's single `DataSeries` column becomes the dim `series`.
    Labels repeat inside one table under different parents ("Males" under each cause),
    so a repeated label also gets `section`, its ancestor chain. Anything still colliding
    gets an occurrence number `n`."""
    names = ["series" if c == "DataSeries" else c for c in label_cols]
    keys = [tuple(str(v).strip() for v in row) for row in df[label_cols].itertuples(index=False)]
    counts = Counter(keys)
    parents = _ancestors(df, "DataSeries") if "DataSeries" in label_cols else [""] * len(df)
    dims = []
    for key, parent in zip(keys, parents):
        d = dict(zip(names, key))
        dims.append({"section": parent, **d} if counts[key] > 1 and parent else d)
    seen: Counter[tuple] = Counter()
    for d in dims:
        k = tuple(d.items())
        seen[k] += 1
        if seen[k] > 1:
            d["n"] = str(seen[k])
    return dims


def read_wide_time(table: Table) -> tuple[list[Series], Counter[str]]:
    df = _row_order(table.frame)
    periods = _period_columns(df)
    freqs = {f for f, _ in periods.values()}
    if len(freqs) > 1:
        raise Reject("mixed_freq", f"period headers mix frequencies {sorted(freqs)}")
    freq = freqs.pop()
    label_cols = [c for c in _data_columns(df) if c not in periods]
    if not label_cols:
        raise Reject("no_row_labels", "period columns but no column naming the series")
    values = df[list(periods)].apply(to_number)
    canon = [periods[c][1] for c in periods]
    order = np.argsort(canon, kind="stable")
    canon = [canon[i] for i in order]
    matrix = values.to_numpy()[:, order]
    dims_per_row = _wide_dims(df, label_cols)
    table_unit = unit_from_dataset_name(table.name)
    out: list[Series] = []
    skipped: Counter[str] = Counter()
    for dims, row in zip(dims_per_row, matrix):
        key = "series" if "series" in dims else next(iter(dims))
        stripped, unit = split_label_unit(dims[key])
        dims = {**dims, key: stripped}
        mask = ~np.isnan(row)
        if mask.sum() < MIN_POINTS:
            skipped[SKIP_TOO_FEW_POINTS] += 1
            continue
        points = {p: float(v) for p, v, ok in zip(canon, row, mask) if ok}
        out.append(Series(
            series_id=make_series_id(table.dataset_id, dims),
            dataset_id=table.dataset_id,
            label=f"{table.name}: {' / '.join(dims.values())}",
            dims=dims, measure=None, unit=unit or table_unit, freq=freq, points=points,
        ))
    return out, skipped


# ---------------------------------------------------------------- long_time

def find_time(df: pd.DataFrame) -> tuple[Freq, pd.Series, list[str]] | None:
    """Best time axis of a long table: the finest frequency any time-named column parses
    to. Also returns every time-named column, all of which are kept out of the dims."""
    named = [c for c in _data_columns(df) if c.lower() in TIME_NAMES | MONTH_NUMBER_NAMES]
    found: list[tuple[Freq, pd.Series, list[str]]] = []
    for c in named:
        if c.lower() in TIME_NAMES and (parsed := parse_time_values(df[c])):
            found.append((parsed[0], parsed[1], [c]))
    years = [c for c in named if c.lower() in {"year", "yr"}]
    months = [c for c in named if c.lower() in MONTH_NUMBER_NAMES]
    for y in years:
        for m in months:
            if (p := combine_year_month(df[y], df[m])) is not None:
                found.append(("M", p, [y, m]))
    if not found:
        return None
    freq, periods, _ = max(found, key=lambda f: FINENESS[f[0]])
    return freq, periods, named


def detect_long_time(table: Table) -> bool:
    return find_time(table.frame) is not None


def _measure_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    out = []
    for c in candidates:
        if NOT_A_MEASURE.search(c.lower()):
            continue
        s = df[c].astype("string").str.strip()
        considered = s[s.notna() & ~s.str.lower().isin(["", "na", "n.a.", "n.a", "-", "--", "nan", "null"])]
        if len(considered) and to_number(considered).notna().mean() >= NUMERIC_SHARE:
            out.append(c)
    return out


def read_long_time(table: Table) -> tuple[list[Series], Counter[str]]:
    df = _row_order(table.frame).reset_index(drop=True)
    found = find_time(df)
    assert found is not None
    freq, periods, time_cols = found
    rest = [c for c in _data_columns(df) if c not in time_cols]
    measures = _measure_columns(df, rest)
    if not measures:
        raise Reject("no_measure", f"no numeric column among {rest[:6]}")
    dim_cols = [c for c in rest if c not in measures]
    dims_df = df[dim_cols].astype("string").fillna("").apply(lambda c: c.str.strip())
    group_key = dims_df.agg("\x1f".join, axis=1) if dim_cols else pd.Series("", index=df.index)
    out: list[Series] = []
    skipped: Counter[str] = Counter()
    dataset_unit = unit_from_dataset_name(table.name)
    candidate_rows = 0
    for m in measures:
        values = to_number(df[m])
        keep = periods.notna() & values.notna()
        frame = dims_df[keep].assign(g=group_key[keep], period=periods[keep], value=values[keep])
        frame = frame.drop_duplicates(["g", "period", "value"]).sort_values("period", kind="stable")
        candidate_rows += len(frame)
        unit = unit_from_text(m) or dataset_unit
        for _, rows in frame.groupby("g", sort=False):
            if rows["period"].duplicated().any():
                skipped[SKIP_DUPLICATE_PERIODS] += 1
                continue
            if len(rows) < MIN_POINTS:
                skipped[SKIP_TOO_FEW_POINTS] += 1
                continue
            dims = {c: v for c in dim_cols if (v := rows.iloc[0][c])}
            parts = list(dims.values()) + ([m] if len(measures) > 1 else [])
            out.append(Series(
                series_id=make_series_id(table.dataset_id, dims, m),
                dataset_id=table.dataset_id,
                label=f"{table.name}: {' / '.join(parts)}" if parts else table.name,
                dims=dims, measure=m, unit=unit, freq=freq,
                points=dict(zip(rows["period"], rows["value"].astype(float))),
            ))
    kept_rows = sum(len(s.points) for s in out)
    if out and kept_rows < MIN_ROW_COVERAGE * candidate_rows:
        raise Reject("low_row_coverage", f"series hold {kept_rows} of {candidate_rows} values ("
                     + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
                     + "); rows are probably individual records that need aggregating, or a dimension is missing")
    return out, skipped


READERS: tuple[Reader, ...] = (
    Reader("wide_time", detect_wide_time, read_wide_time),
    Reader("long_time", detect_long_time, read_long_time),
)


# ---------------------------------------------------------------- registry

def explain_unmatched(table: Table) -> Unmatched:
    df, did = table.frame, table.dataset_id
    cols = _data_columns(df)
    if df.empty or not cols:
        return Unmatched(did, "empty_table")
    n_periods = len(_period_columns(df))
    if n_periods:
        return Unmatched(did, "period_columns_too_few", f"{n_periods} period-like of {len(cols)} columns")
    lowered = {c.lower() for c in cols}
    if lowered & TIMESTAMP_NAMES or (lowered & TIME_NAMES):
        named = sorted(lowered & (TIMESTAMP_NAMES | TIME_NAMES))
        sub_monthly = bool(lowered & TIMESTAMP_NAMES) or "date" in lowered
        reason = "sub_monthly_time" if sub_monthly else "time_format_unrecognised"
        return Unmatched(did, reason, f"time-like columns {named}")
    return Unmatched(did, "snapshot", "no period columns and no time column")


def flip(table: Table, readers: tuple[Reader, ...] = READERS) -> Matched | Unmatched:
    """Run the registry over one table. Always returns a record."""
    for reader in readers:
        if not reader.detect(table):
            continue
        try:
            series, skipped = reader.read(table)
        except Reject as r:
            return Unmatched(table.dataset_id, r.reason, r.detail)
        if not series:
            return Unmatched(table.dataset_id, "no_series_survived",
                             ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())) or "no rows with values")
        return Matched(table.dataset_id, reader.name, tuple(_unique_ids(series)), dict(skipped))
    return explain_unmatched(table)
