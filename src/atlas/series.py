"""The Series model every atlas stage reads.

Period format (canonical, lexicographically sortable within one frequency, and
self-describing so the frequency can be read back from the string):

    M  "2024-09"     A  "2024"
    Q  "2024Q3"      H  "2024H2"
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, get_args

Freq = Literal["M", "Q", "H", "A"]
FREQS: tuple[str, ...] = get_args(Freq)

_PERIOD_RE: dict[str, re.Pattern[str]] = {
    "M": re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    "Q": re.compile(r"^\d{4}Q[1-4]$"),
    "H": re.compile(r"^\d{4}H[12]$"),
    "A": re.compile(r"^\d{4}$"),
}

MAX_SLUG = 80


def format_period(freq: Freq, year: int, n: int = 1) -> str:
    """`n` is the month, quarter or half; ignored for annual."""
    if freq == "M":
        return f"{year:04d}-{n:02d}"
    if freq == "Q":
        return f"{year:04d}Q{n}"
    if freq == "H":
        return f"{year:04d}H{n}"
    return f"{year:04d}"


def slug(text: str) -> str:
    """Lowercase alphanumerics joined by single hyphens; long values keep a hash so
    two long labels that share a prefix stay distinct."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(s) > MAX_SLUG:
        s = s[: MAX_SLUG - 9].rstrip("-") + "-" + hashlib.sha1(text.encode()).hexdigest()[:8]
    return s


def make_series_id(dataset_id: str, dims: dict[str, str], measure: str | None = None) -> str:
    """`<dataset_id>:<dim slugs joined by '.'>[:<measure slug>]`. Dim slugs never contain
    '.', ':' or whitespace, so the id parses back unambiguously."""
    sid = f"{dataset_id}:{'.'.join(s for s in (slug(v) for v in dims.values()) if s) or 'all'}"
    return f"{sid}:{slug(measure)}" if measure else sid


@dataclass(frozen=True)
class Series:
    series_id: str
    dataset_id: str
    label: str
    dims: dict[str, str]
    measure: str | None
    unit: str | None
    freq: Freq
    points: dict[str, float]

    def __post_init__(self) -> None:
        if self.freq not in FREQS:
            raise ValueError(f"{self.series_id}: freq {self.freq!r} not in {FREQS}")
        periods = list(self.points)
        bad = [p for p in periods if not _PERIOD_RE[self.freq].match(p)]
        if bad:
            raise ValueError(f"{self.series_id}: periods {bad[:3]} are not {self.freq} periods")
        if periods != sorted(periods):
            raise ValueError(f"{self.series_id}: points are not ascending")
