"""Market-making quoting policies.

All prices are in ticks, time in seconds, inventory in lots. Fill intensity at distance delta from
the reference price is modelled as lambda(delta) = A * exp(-kappa * delta) (calibrated in
``calibration.py``). Inventory enters the closed forms in units of the quote size.

* ``Symmetric``        fixed half-spread around the reference, inventory limit only.
* ``AvellanedaStoikov`` Avellaneda & Stoikov (2008) with a rolling horizon tau instead of T - t
                        (the finite-horizon form blows the spread up early in a long session).
* ``GLFT``             Gueant, Lehalle & Fernandez-Tapia (2013) closed-form asymptotic quotes:
                           delta_b(q) = c1 + (2q + 1)/2 * c2,  delta_a(q) = c1 - (2q - 1)/2 * c2
                           c1 = ln(1 + gamma/kappa) / gamma
                           c2 = sqrt(sigma^2 gamma / (2 kappa A) * (1 + gamma/kappa)^(1 + kappa/gamma))
* ``Probe``            random fixed depth around the lead price, used only for calibration.

The reference price is where quotes are centred: ``lead`` (fair value seen on a faster correlated
market, as a futures-led equity desk would), ``mid`` or ``micro`` (size-weighted microprice).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(slots=True)
class MarketView:
    t: float
    lead: float
    best_bid: int | None
    best_ask: int | None
    bid_qty: int
    ask_qty: int
    inventory: int


@dataclass(slots=True, frozen=True)
class Quote:
    bid: int | None
    ask: int | None
    size: int


def reference_price(view: MarketView, mode: str) -> float | None:
    if mode == "lead":
        return view.lead
    bb, ba = view.best_bid, view.best_ask
    if bb is None and ba is None:
        return None
    if bb is None:
        return ba - 0.5
    if ba is None:
        return bb + 0.5
    if mode == "mid":
        return 0.5 * (bb + ba)
    if mode == "micro":
        total = view.bid_qty + view.ask_qty
        return 0.5 * (bb + ba) if total == 0 else (bb * view.ask_qty + ba * view.bid_qty) / total
    raise ValueError(f"unknown reference mode {mode!r}")


def to_ticks(view: MarketView, bid: float | None, ask: float | None, size: int, max_inventory: int) -> Quote:
    """Rounds continuous quotes onto the tick grid (away from the reference), keeps them passive
    (post-only would reject a crossing quote) and enforces the inventory limit."""
    b = math.floor(bid) if bid is not None and view.inventory < max_inventory else None
    a = math.ceil(ask) if ask is not None and view.inventory > -max_inventory else None
    if b is not None and view.best_ask is not None:
        b = min(b, view.best_ask - 1)
    if a is not None and view.best_bid is not None:
        a = max(a, view.best_bid + 1)
    if b is not None and a is not None and a <= b:
        a = b + 1
    return Quote(b, a, size)


class Strategy:
    name = "base"
    reference = "lead"

    def __init__(self, size: int = 2, max_inventory: int = 50) -> None:
        self.size = size
        self.max_inventory = max_inventory

    def quote(self, view: MarketView) -> Quote | None:
        s = reference_price(view, self.reference)
        if s is None:
            return None
        bid, ask = self.levels(s, view)
        return to_ticks(view, bid, ask, self.size, self.max_inventory)

    def levels(self, s: float, view: MarketView) -> tuple[float, float]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "reference": self.reference, "size": self.size, "max_inventory": self.max_inventory}


class Symmetric(Strategy):
    name = "Symmetric"

    def __init__(self, half_spread: float = 1.0, reference: str = "lead", **kw) -> None:
        super().__init__(**kw)
        self.half_spread = half_spread
        self.reference = reference

    def levels(self, s, view):
        return s - self.half_spread, s + self.half_spread

    def describe(self):
        return {**super().describe(), "half_spread": self.half_spread}


class AvellanedaStoikov(Strategy):
    name = "Avellaneda-Stoikov"

    def __init__(self, gamma: float, kappa: float, sigma: float, horizon_s: float = 30.0,
                 reference: str = "lead", **kw) -> None:
        super().__init__(**kw)
        self.gamma, self.kappa, self.sigma, self.horizon_s = gamma, kappa, sigma, horizon_s
        self.reference = reference
        g, k = gamma, kappa
        self._skew = g * sigma * sigma * horizon_s
        self._half = 0.5 * (self._skew + (2.0 / g) * math.log1p(g / k))

    def levels(self, s, view):
        q = view.inventory / self.size
        r = s - q * self._skew
        return r - self._half, r + self._half

    def describe(self):
        return {**super().describe(), "gamma": self.gamma, "kappa": self.kappa, "sigma": self.sigma,
                "horizon_s": self.horizon_s}


class GLFT(Strategy):
    name = "GLFT"

    def __init__(self, gamma: float, kappa: float, A: float, sigma: float, reference: str = "lead", **kw) -> None:
        super().__init__(**kw)
        self.gamma, self.kappa, self.A, self.sigma = gamma, kappa, A, sigma
        self.reference = reference
        g, k = gamma, kappa
        self.c1 = math.log1p(g / k) / g
        self.c2 = math.sqrt(sigma * sigma * g / (2.0 * k * A) * (1.0 + g / k) ** (1.0 + k / g))

    def half_spread(self, q: float = 0.0) -> float:
        return 0.5 * (self.c1 + (2 * q + 1) / 2 * self.c2 + self.c1 - (2 * q - 1) / 2 * self.c2)

    def levels(self, s, view):
        q = view.inventory / self.size
        return s - (self.c1 + (2 * q + 1) / 2 * self.c2), s + (self.c1 - (2 * q - 1) / 2 * self.c2)

    def describe(self):
        return {**super().describe(), "gamma": self.gamma, "kappa": self.kappa, "A": self.A, "sigma": self.sigma,
                "c1": self.c1, "c2": self.c2}


class Probe(Strategy):
    """Quotes at a random depth in [lo, hi] ticks from the lead price on every decision."""

    name = "Probe"

    def __init__(self, lo: float = 0.0, hi: float = 4.0, seed: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.lo, self.hi = lo, hi
        self._rng = random.Random(seed)

    def levels(self, s, view):
        return s - self._rng.uniform(self.lo, self.hi), s + self._rng.uniform(self.lo, self.hi)
