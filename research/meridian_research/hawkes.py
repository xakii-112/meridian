"""Multivariate Hawkes process with exponential kernels.

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha[i][j] * exp(-beta * (t - t_k^j))

Used for uninformed taker flow: buy and sell market orders cluster in time (self-excitation) and
trigger each other (cross-excitation), as in real order flow. Simulation is Ogata thinning with
the O(1) recursive kernel update, so cost is linear in the number of events.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np


def simulate(mu: Sequence[float], alpha: Sequence[Sequence[float]], beta: float, horizon: float,
             rng: random.Random) -> list[tuple[float, int]]:
    """Returns time-ordered events ``(t, dimension)`` on ``[0, horizon)``."""
    d = len(mu)
    if spectral_radius(alpha, beta) >= 1.0:
        raise ValueError("Hawkes process is not stationary (branching ratio >= 1)")
    excite = [0.0] * d  # sum_j sum_k alpha[i][j] exp(-beta (t - t_k^j)) for each target dimension i
    t = 0.0
    events: list[tuple[float, int]] = []
    mu = list(mu)
    while True:
        # Between events intensities only decay, so the current total is a valid upper bound.
        bound = sum(mu) + sum(excite)
        w = rng.expovariate(bound)
        t += w
        if t >= horizon:
            return events
        decay = math.exp(-beta * w)
        excite = [e * decay for e in excite]
        lam = [m + e for m, e in zip(mu, excite)]
        u = rng.random() * bound
        total = sum(lam)
        if u > total:
            continue  # thinned
        k, acc = 0, lam[0]
        while u > acc:
            k += 1
            acc += lam[k]
        events.append((t, k))
        for i in range(d):
            excite[i] += alpha[i][k]


def spectral_radius(alpha: Sequence[Sequence[float]], beta: float) -> float:
    """Branching ratio: spectral radius of the kernel L1-norm matrix alpha / beta."""
    return float(max(abs(np.linalg.eigvals(np.asarray(alpha, dtype=float) / beta))))


def stationary_intensity(mu: Sequence[float], alpha: Sequence[Sequence[float]], beta: float) -> np.ndarray:
    """E[lambda] = (I - alpha/beta)^-1 mu for a stationary process."""
    a = np.asarray(alpha, dtype=float) / beta
    return np.linalg.solve(np.eye(len(mu)) - a, np.asarray(mu, dtype=float))
