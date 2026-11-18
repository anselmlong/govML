"""Turn the time strings found in data.gov.sg tables into canonical periods."""

from __future__ import annotations

import re

import pandas as pd

from .series import Freq, format_period

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_HEADER = re.compile(r"^(\d{4})\s*(?:(\d)([QH])|([A-Za-z]{3,9}))?$")
MIN_YEAR, MAX_YEAR = 1800, 2100
FINENESS = {"A": 0, "H": 1, "Q": 2, "M": 3}


def parse_header(label: object) -> tuple[Freq, str] | None:
    """Wide-table column header: `2024Sep`, `2024 1Q`, `2024 2H`, `2024`."""
    m = _HEADER.match(str(label).strip())
    if not m:
        return None
    year = int(m.group(1))
    if not MIN_YEAR <= year <= MAX_YEAR:
        return None
    if m.group(2):
        n, unit = int(m.group(2)), m.group(3)
        if unit == "Q" and 1 <= n <= 4:
            return "Q", format_period("Q", year, n)
        if unit == "H" and 1 <= n <= 2:
            return "H", format_period("H", year, n)
        return None
    if m.group(4):
        month = MONTHS.get(m.group(4)[:3].lower())
        return ("M", format_period("M", year, month)) if month else None
    return "A", format_period("A", year)


_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-01(?:[T ]00:00(?::00)?(?:\.0+)?(?:Z|[+-]\d\d:?\d\d)?)?$")
_VALUE_PATTERNS: tuple[tuple[Freq, re.Pattern[str]], ...] = (
    ("A", re.compile(r"^(\d{4})$")),
    ("M", re.compile(r"^(\d{4})-(\d{2})$")),
    ("M", _DATE_PATTERN),
    ("Q", re.compile(r"^(\d{4})\s*-?\s*[Qq]([1-4])$")),
    ("H", re.compile(r"^(\d{4})\s*-?\s*[Hh]([12])$")),
)
MATCH_SHARE = 0.95


def parse_time_values(values: pd.Series) -> tuple[Freq, pd.Series] | None:
    """Parse a long-table time column. Returns the frequency and one canonical period per
    row (NA where a row does not parse), or None unless a single frequency explains at
    least 95% of the non-null values. Daily dates (any day but the 1st) never match."""
    s = values.dropna().astype(str).str.strip()
    if s.empty:
        return None
    full = values.astype("string").str.strip()
    for freq, pat in _VALUE_PATTERNS:
        parts = full.str.extract(pat)
        ok = parts.notna().all(axis=1)
        if ok.sum() < MATCH_SHARE * len(s):
            continue
        year = pd.to_numeric(parts[0], errors="coerce")
        n = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else pd.Series(1, index=parts.index)
        valid = ok & year.between(MIN_YEAR, MAX_YEAR)
        if freq == "M":
            valid &= n.between(1, 12)
        if valid.sum() < MATCH_SHARE * len(s):
            continue
        periods = pd.Series(pd.NA, index=values.index, dtype="object")
        periods[valid] = [format_period(freq, int(y), int(k)) for y, k in zip(year[valid], n[valid])]
        if pat is _DATE_PATTERN and periods.nunique() < 2:
            continue
        return freq, periods
    return None


def combine_year_month(year: pd.Series, month: pd.Series) -> pd.Series | None:
    """`year` plus a numeric `mth` column (1-12) becomes monthly periods."""
    y = pd.to_numeric(year, errors="coerce")
    m = pd.to_numeric(month, errors="coerce")
    ok = y.between(MIN_YEAR, MAX_YEAR) & m.between(1, 12)
    if ok.sum() < MATCH_SHARE * year.notna().sum() or not ok.any():
        return None
    periods = pd.Series(pd.NA, index=year.index, dtype="object")
    periods[ok] = [format_period("M", int(a), int(b)) for a, b in zip(y[ok], m[ok])]
    return periods
