"""Numerics for the findings engine. Arrays are float64 on a regular period grid, NaN where a
period has no value."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def transform(levels: np.ndarray) -> tuple[np.ndarray, str]:
    """dlog when every observed level is positive, else the first difference. A change is
    defined only between two adjacent grid periods, so a gap never gets bridged."""
    observed = levels[~np.isnan(levels)]
    if observed.size and (observed > 0).all():
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.diff(np.log(levels)), "dlog"
    return np.diff(levels), "diff"


def is_flat(changes: np.ndarray, max_zero_share: float = 0.5) -> bool:
    """Near-zero variance, or a step series that holds still in at least half its periods
    (tariffs, quotas): its correlations would rest on a handful of steps."""
    c = changes[~np.isnan(changes)]
    if c.size < 3 or np.std(c) < 1e-9:
        return True
    return float(np.mean(np.abs(c) < 1e-12)) >= max_zero_share


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(sps.rankdata(a), sps.rankdata(b))


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelations at lags 1..max_lag over the non-NaN pairs of a grid array."""
    out = np.zeros(max_lag)
    valid = ~np.isnan(x)
    if valid.sum() < 3:
        return out
    mu, var = x[valid].mean(), x[valid].var()
    if var <= 0:
        return out
    d = np.where(valid, x - mu, 0.0)
    n = valid.sum()
    for j in range(1, max_lag + 1):
        if j >= x.size:
            break
        out[j - 1] = (d[:-j] * d[j:]).sum() / n / var
    return out


def bartlett_neff(n: int, acf_x: np.ndarray, acf_y: np.ndarray) -> float:
    """n / (1 + 2 sum rho_x(j) rho_y(j)), clamped to [3, n]: the sum can go negative."""
    inflation = 1.0 + 2.0 * float((acf_x * acf_y).sum())
    return float(np.clip(n / inflation if inflation > 0 else n, 3.0, n))


def p_value(r: float, n_eff: float) -> float:
    """Two-sided p for a correlation r with n_eff effective observations."""
    if not np.isfinite(r):
        return 1.0
    df = n_eff - 2.0
    if abs(r) >= 1.0:
        return 0.0
    t = abs(r) * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * sps.t.sf(t, df))


def detectable_r(p_cut: float, n_eff: float) -> float:
    """Smallest |r| whose p at n_eff effective observations is below p_cut."""
    df = n_eff - 2.0
    t = sps.t.isf(p_cut / 2.0, df)
    return float(t / np.sqrt(df + t * t))


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out
