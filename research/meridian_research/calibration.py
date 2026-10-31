"""Calibration of the quoting model from simulated (or real) quote/fill logs.

Fill intensity: the probe strategy quotes at random depths; every quote contributes its resting
time to the bucket of its placement distance and its fill count to the same bucket, so
lambda(delta) = fills / exposure. A weighted log-linear fit gives lambda(delta) = A exp(-kappa delta).

Volatility: realised volatility of the lead price on a 1-second grid (jumps included, since the
market maker carries inventory through them).
"""

from __future__ import annotations

import math

import numpy as np

from .market import SAMPLE_COLS


def fill_intensity(quote_logs: list[np.ndarray], edges: np.ndarray) -> dict:
    exposure = np.zeros(len(edges) - 1)
    fills = np.zeros(len(edges) - 1)
    for log in quote_logs:
        if len(log) == 0:
            continue
        done = ~np.isnan(log[:, 2])
        dist, life, n = log[done, 3], log[done, 2] - log[done, 1], log[done, 4]
        b = np.digitize(dist, edges) - 1
        ok = (b >= 0) & (b < len(exposure))
        np.add.at(exposure, b[ok], life[ok])
        np.add.at(fills, b[ok], n[ok])
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(exposure > 0, fills / exposure, np.nan)
    return {"depth": centers, "intensity": lam, "exposure_s": exposure, "fills": fills}


def fit_exponential(depth: np.ndarray, intensity: np.ndarray, fills: np.ndarray, min_fills: int = 20) -> dict:
    """Weighted least squares of log(lambda) on depth; weights ~ fill counts (Poisson variance of log)."""
    ok = (fills >= min_fills) & np.isfinite(intensity) & (intensity > 0)
    if ok.sum() < 3:
        raise ValueError("not enough populated buckets to fit lambda(delta)")
    x, y, w = depth[ok], np.log(intensity[ok]), np.sqrt(fills[ok])
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum(fills[ok] * (y - pred) ** 2))
    ss_tot = float(np.sum(fills[ok] * (y - np.average(y, weights=fills[ok])) ** 2))
    return {"A": float(math.exp(beta[0])), "kappa": float(-beta[1]), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0}


def realized_sigma(samples: np.ndarray, dt: float = 1.0) -> float:
    """Realised volatility of the fair/lead price in ticks per sqrt(second)."""
    t = samples[:, SAMPLE_COLS.index("t")]
    v = samples[:, SAMPLE_COLS.index("fair")]
    grid = np.arange(t[0], t[-1], dt)
    idx = np.searchsorted(t, grid, side="right") - 1
    r = np.diff(v[idx])
    return float(np.std(r) / math.sqrt(dt))
