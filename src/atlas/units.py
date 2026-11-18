"""Find a series' unit from the text around it. Returns None rather than guess."""

from __future__ import annotations

import re

_SCALE = r"(million|thousand|billion)"
_UNITS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"{_SCALE} (?:us|usd|u s) dollars?"), "{0} US Dollars"),
    (re.compile(rf"{_SCALE} dollars?"), "{0} Dollars"),
    (re.compile(r"dollars?"), "Dollars"),
    (re.compile(r"per ?cent|percent|percentage|pct|%"), "Per Cent"),
    (re.compile(r"thousands?"), "Thousand"),
)
_INDEX_BASE = re.compile(r"(?:\(|base year )\s*(\d{4})\s*=\s*100", re.I)


def _normalise(text: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def unit_from_text(text: str) -> str | None:
    """Unit named by a column name or a trailing `(...)` of a row label. The whole text
    must be the unit phrase: `MillionDollars` is a unit, `MillionDollars_Ratio` is not."""
    t = _normalise(text)
    for pat, canon in _UNITS:
        m = pat.fullmatch(t)
        if m:
            return canon.format(*(g.title() for g in m.groups()))
    return None


_TRAILING_PAREN = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def split_label_unit(label: str) -> tuple[str, str | None]:
    """`"Total Exports (Million Dollars)"` -> `("Total Exports", "Million Dollars")`.
    A trailing parenthesis that is not a unit stays in the label."""
    m = _TRAILING_PAREN.match(label)
    if m and m.group(1):
        unit = unit_from_text(m.group(2))
        if unit:
            return m.group(1), unit
    return label, None


def unit_from_dataset_name(name: str) -> str | None:
    """Price and volume indices are named with their base: `(2017 = 100)`."""
    m = _INDEX_BASE.search(name)
    if m and re.search(r"\bindex\b", name, re.I):
        return f"Index ({m.group(1)}=100)"
    return None
