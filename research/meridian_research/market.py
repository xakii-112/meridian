"""Event-driven, agent-based market simulator for market-making research.

Agents (owner ids are carried onto the MWP tape)
  MM     (1)  the strategy under test: post-only quotes, decisions reach the book after a latency
  LP     (2)  background liquidity providers: limit orders placed around the fair value at arrival,
              resting for an exponential lifetime (they go stale when the fair value moves)
  NOISE  (3)  uninformed takers: market orders driven by a bivariate Hawkes process
  ARB    (4)  zero-latency arbitrageurs: take any resting quote that is mispriced against the
              current fair value by more than ``arb_edge`` ticks (latency-driven adverse selection)
  INFORMED (5) takers with private short-horizon information: they trade when the fair value
              ``informed_horizon_s`` ahead differs from the touch by more than ``informed_edge``
              (adverse selection that no amount of speed removes)

Fair value V follows a Brownian motion plus compound-Poisson jumps (news), pre-simulated on a
``path_dt`` grid. The MM observes V on a faster "lead" market (think: index futures leading a cash
equity) but every action it takes lands on the book ``latency_s`` later, so after a jump its stale
quotes race the arbitrageurs.

Everything is driven by one seeded ``random.Random``: a (config, seed) pair is fully reproducible,
and with ``record_tape=True`` the complete order flow is written as an MWP tape for replay
through the C++ engine.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import asdict, dataclass, replace

import numpy as np

from . import hawkes
from .lob import BUY, SELL, LimitOrderBook
from .strategies import MarketView, Quote, Strategy, reference_price
from .wire import TapeWriter

MM, LP, NOISE, ARB, INFORMED = 1, 2, 3, 4, 5
_LP, _LP_CANCEL, _NOISE, _ARB, _INFORMED, _DECIDE, _APPLY, _SAMPLE = range(8)


@dataclass
class MarketConfig:
    duration_s: float = 3600.0
    warmup_s: float = 60.0
    v0: float = 10_000.0           # $100.00 with a $0.01 tick
    sigma: float = 0.8             # fair-value diffusion, ticks / sqrt(s)  (~1.2% daily vol at $100)
    jump_rate: float = 1 / 90      # news jumps per second
    jump_mean_abs: float = 5.0     # mean |jump| in ticks (exponential magnitude, random sign)
    path_dt: float = 0.01          # fair-value grid
    lp_rate: float = 10.0          # background limit orders per second (both sides together)
    lp_depth_p: float = 0.50       # depth beyond the inside tick ~ Geometric(p)
    lp_size_p: float = 0.45        # lots = 1 + Geometric(p)
    lp_lifetime_s: float = 4.0     # mean resting lifetime
    noise_mu: float = 0.40         # Hawkes baseline intensity per side (orders/s)
    noise_alpha_self: float = 0.8
    noise_alpha_cross: float = 0.3
    noise_beta: float = 2.0
    noise_size_p: float = 0.40
    arb_rate: float = 4.0          # arbitrageur arrivals per second
    arb_edge: float = 0.5          # minimum mispricing (ticks) an arbitrageur needs
    arb_max_lots: int = 10
    informed_rate: float = 0.8     # informed-taker arrivals per second
    informed_horizon_s: float = 10.0
    informed_edge: float = 0.5
    informed_size_p: float = 0.4
    maker_rebate: float = 0.20     # ticks per lot  (= $0.002/share at 100 shares/lot)
    tick_value: float = 1.0        # USD per tick per lot ($0.01 x 100 shares)
    sample_dt: float = 0.1
    # Day-to-day regime variation: volatility, informed-flow and noise-flow intensities are each
    # scaled per session by an independent mean-one lognormal with this log-dispersion.
    regime_dispersion: float = 0.35


@dataclass
class MakerConfig:
    latency_s: float = 0.005       # decision -> order at the matching engine
    requote_interval_s: float = 1.0
    requote_threshold: float = 0.35  # reference move (ticks) that triggers an immediate requote


@dataclass
class SessionResult:
    seed: int
    market: dict
    maker: dict
    strategy: dict
    fills: np.ndarray       # columns: t, side(+1 buy/-1 sell), price, qty, mid_pre, fair, inventory_after
    samples: np.ndarray     # columns: t, fair, mid, micro, inventory, cash, rebates, best_bid, best_ask, mm_bid, mm_ask
    quote_log: np.ndarray   # columns: side, t_place, t_end, distance, fills
    market_volume: int
    trade_count: int
    tape_hash: int
    decisions: int
    tape: TapeWriter | None = None
    regime: dict | None = None


FILL_COLS = ("t", "side", "price", "qty", "mid_pre", "fair", "inventory")
SAMPLE_COLS = ("t", "fair", "mid", "micro", "inventory", "cash", "rebates", "best_bid", "best_ask", "mm_bid", "mm_ask")


def _geometric(rng: random.Random, p: float) -> int:
    """Failures before the first success, P(k) = (1-p)^k p."""
    return int(math.log(1.0 - rng.random()) / math.log1p(-p))


class Simulator:
    def __init__(self, market: MarketConfig, maker: MakerConfig, strategy: Strategy, seed: int,
                 record_tape: bool = False) -> None:
        self.mk = maker
        self.strategy = strategy
        self.seed = seed
        self.rng = random.Random(seed)
        self.regime = self._draw_regime(market)
        self.m = replace(market, sigma=market.sigma * self.regime["volatility"],
                         jump_mean_abs=market.jump_mean_abs * self.regime["volatility"],
                         informed_rate=market.informed_rate * self.regime["informed"],
                         noise_mu=market.noise_mu * self.regime["noise"])
        self.tape = TapeWriter() if record_tape else None
        self.book = LimitOrderBook(tape=self.tape)
        self._events: list = []
        self._seq = 0
        self._cl = 0
        self.V = market.v0  # fair value at the current event
        # market maker state
        self.q = 0
        self.cash = 0.0
        self.rebates = 0.0
        self._mm = [None, None]  # per side: [cl, price, quote_log_index]
        self._pending = False
        self._last_ref = None
        self.decisions = 0
        self.fills: list[tuple] = []
        self.samples: list[tuple] = []
        self.quote_log: list[list] = []

    # ---- helpers -----------------------------------------------------------------------------

    def _draw_regime(self, market: MarketConfig) -> dict:
        d = market.regime_dispersion
        draw = lambda: math.exp(self.rng.gauss(0.0, d) - 0.5 * d * d) if d > 0 else 1.0
        return {"volatility": draw(), "informed": draw(), "noise": draw()}

    def _push(self, t: float, kind: int, payload=None) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, payload))

    def _next_cl(self) -> int:
        self._cl += 1
        return self._cl

    def _mid(self) -> float | None:
        bb, ba = self.book.best_bid(), self.book.best_ask()
        return None if bb is None or ba is None else 0.5 * (bb + ba)

    def _view(self, t: float) -> MarketView:
        """The book as the strategy should see it: the market maker's own quotes are excluded,
        otherwise a mid-referenced quote chases its own inventory skew."""
        own_b, own_a = self._mm
        bb, bq = self.book.top_excluding(BUY, MM, own_b[0] if own_b is not None else None)
        ba, aq = self.book.top_excluding(SELL, MM, own_a[0] if own_a is not None else None)
        return MarketView(t, self.V, bb, ba, bq, aq, self.q)

    def _take(self, owner: int, side: int, price: int, qty: int, t: float, *, market: bool) -> None:
        mid_pre = self._mid()
        _, fills, _ = self.book.submit(owner, self._next_cl(), side, price, qty, ioc=True, market=market,
                                       ts=int(t * 1e9))
        if fills:
            self._on_fills(fills, 1 - side, t, mid_pre)

    def _on_fills(self, fills: list, maker_side: int, t: float, mid_pre: float | None) -> None:
        mm_hit = False
        for owner, cl, price, qty, leaves in fills:
            if owner != MM:
                continue
            mm_hit = True
            sgn = 1 if maker_side == BUY else -1
            self.q += sgn * qty
            self.cash -= sgn * price * qty
            self.rebates += self.m.maker_rebate * qty
            self.fills.append((t, sgn, price, qty, mid_pre if mid_pre is not None else price, self.V, self.q))
            cur = self._mm[maker_side]
            if cur is not None and cur[0] == cl:
                self.quote_log[cur[2]][4] += 1
                if leaves == 0:
                    self.quote_log[cur[2]][2] = t
                    self._mm[maker_side] = None
        if mm_hit and not self._pending and t >= self.m.warmup_s:
            self._pending = True
            self._push(t, _DECIDE)

    # ---- market maker ------------------------------------------------------------------------

    def _decide(self, t: float) -> None:
        self._pending = False
        view = self._view(t)
        self._last_ref = reference_price(view, self.strategy.reference)
        quote = self.strategy.quote(view)
        self.decisions += 1
        self._push(t + self.mk.latency_s, _APPLY, quote)

    def _apply(self, t: float, quote: Quote | None) -> None:
        ts = int(t * 1e9)
        book = self.book
        targets = (None, None) if quote is None else (quote.bid, quote.ask)
        place = []
        # Phase 1: pull every quote that changes (so a new bid can never hit our own stale ask).
        for side in (BUY, SELL):
            target = targets[side]
            cur = self._mm[side]
            if cur is not None and book.leaves(MM, cur[0]) == 0:  # filled while the decision was in flight
                self.quote_log[cur[2]][2] = t
                self._mm[side] = cur = None
            if cur is not None and target == cur[1]:
                continue  # same price: keep queue priority
            if cur is not None:
                book.cancel(MM, cur[0], ts)
                self.quote_log[cur[2]][2] = t
                self._mm[side] = None
            place.append((side, target))
        # Phase 2: post the new quotes.
        for side, target in place:
            if target is not None:
                cl = self._next_cl()
                accepted, _, _ = book.submit(MM, cl, side, target, quote.size, post_only=True, ts=ts)
                if accepted:
                    dist = self.V - target if side == BUY else target - self.V
                    self.quote_log.append([side, t, math.nan, dist, 0])
                    self._mm[side] = [cl, target, len(self.quote_log) - 1]

    # ---- main loop ---------------------------------------------------------------------------

    def run(self) -> SessionResult:
        m, rng, book = self.m, self.rng, self.book
        T = m.duration_s

        # Seed a two-sided book so the session does not start empty.
        for depth in range(6):
            for side in (BUY, SELL):
                for _ in range(2):
                    price = math.ceil(self.V) - 1 - depth if side == BUY else math.floor(self.V) + 1 + depth
                    cl = self._next_cl()
                    book.submit(LP, cl, side, price, 1 + _geometric(rng, m.lp_size_p))
                    self._push(rng.expovariate(1 / m.lp_lifetime_s), _LP_CANCEL, cl)

        # Fair value on a fine grid: Brownian increments plus compound-Poisson jumps.
        prng = np.random.default_rng(rng.getrandbits(63))
        n = int(T / m.path_dt) + 2
        steps = m.sigma * math.sqrt(m.path_dt) * prng.standard_normal(n)
        steps[0] = 0.0
        n_jumps = int(prng.poisson(m.jump_rate * T))
        np.add.at(steps, prng.integers(1, n, size=n_jumps),
                  prng.exponential(m.jump_mean_abs, size=n_jumps) * prng.choice([-1.0, 1.0], size=n_jumps))
        path = (m.v0 + np.cumsum(steps)).tolist()  # list indexing beats numpy scalar indexing
        inv_dt, last_idx = 1.0 / m.path_dt, n - 1

        alpha = [[m.noise_alpha_self, m.noise_alpha_cross], [m.noise_alpha_cross, m.noise_alpha_self]]
        for t, dim in hawkes.simulate([m.noise_mu, m.noise_mu], alpha, m.noise_beta, T, rng):
            self._push(t, _NOISE, BUY if dim == 0 else SELL)
        self._push(rng.expovariate(m.lp_rate), _LP)
        self._push(rng.expovariate(m.arb_rate), _ARB)
        self._push(rng.expovariate(m.informed_rate), _INFORMED)
        self._push(m.warmup_s, _SAMPLE)
        self._push(m.warmup_s, _DECIDE)
        self._pending = True
        next_timer = m.warmup_s + self.mk.requote_interval_s
        ref_mode = self.strategy.reference
        threshold = self.mk.requote_threshold

        events = self._events
        while events:
            t, _, kind, payload = heapq.heappop(events)
            if t >= T:
                break
            self.V = path[int(t * inv_dt)]

            if kind == _LP:
                side = BUY if rng.random() < 0.5 else SELL
                d = _geometric(rng, m.lp_depth_p)
                price = math.ceil(self.V) - 1 - d if side == BUY else math.floor(self.V) + 1 + d
                cl = self._next_cl()
                mid_pre = self._mid()
                _, fills, leaves = book.submit(LP, cl, side, price, 1 + _geometric(rng, m.lp_size_p),
                                               ts=int(t * 1e9))
                if fills:
                    self._on_fills(fills, 1 - side, t, mid_pre)
                if leaves > 0:
                    self._push(t + rng.expovariate(1 / m.lp_lifetime_s), _LP_CANCEL, cl)
                self._push(t + rng.expovariate(m.lp_rate), _LP)
            elif kind == _LP_CANCEL:
                book.cancel(LP, payload, int(t * 1e9))
            elif kind == _NOISE:
                self._take(NOISE, payload, 0, 1 + _geometric(rng, m.noise_size_p), t, market=True)
            elif kind == _ARB:
                ask = book.best_ask()
                if ask is not None and ask <= self.V - m.arb_edge:
                    self._take(ARB, BUY, math.floor(self.V - m.arb_edge), m.arb_max_lots, t, market=False)
                else:
                    bid = book.best_bid()
                    if bid is not None and bid >= self.V + m.arb_edge:
                        self._take(ARB, SELL, math.ceil(self.V + m.arb_edge), m.arb_max_lots, t, market=False)
                self._push(t + rng.expovariate(m.arb_rate), _ARB)
            elif kind == _INFORMED:
                future = path[min(int((t + m.informed_horizon_s) * inv_dt), last_idx)]
                size = 1 + _geometric(rng, m.informed_size_p)
                ask = book.best_ask()
                if ask is not None and future - ask >= m.informed_edge:
                    self._take(INFORMED, BUY, 0, size, t, market=True)
                else:
                    bid = book.best_bid()
                    if bid is not None and bid - future >= m.informed_edge:
                        self._take(INFORMED, SELL, 0, size, t, market=True)
                self._push(t + rng.expovariate(m.informed_rate), _INFORMED)
            elif kind == _DECIDE:
                self._decide(t)
                next_timer = t + self.mk.requote_interval_s
            elif kind == _APPLY:
                self._apply(t, payload)
            elif kind == _SAMPLE:
                self._sample(t)
                self._push(t + m.sample_dt, _SAMPLE)

            # Event-driven requotes: periodic timer, or the reference moved by >= threshold.
            if t >= m.warmup_s and not self._pending:
                if t >= next_timer:
                    self._pending = True
                    self._push(t, _DECIDE)
                else:
                    ref = self.V if ref_mode == "lead" else reference_price(self._view(t), ref_mode)
                    if ref is not None and (self._last_ref is None or abs(ref - self._last_ref) >= threshold):
                        self._pending = True
                        self._push(t, _DECIDE)

        for side in (BUY, SELL):
            if self._mm[side] is not None:
                self.quote_log[self._mm[side][2]][2] = T
        return SessionResult(
            seed=self.seed, market=asdict(self.m), maker=asdict(self.mk), strategy=self.strategy.describe(),
            fills=np.asarray(self.fills, dtype=float).reshape(-1, len(FILL_COLS)),
            samples=np.asarray(self.samples, dtype=float).reshape(-1, len(SAMPLE_COLS)),
            quote_log=np.asarray(self.quote_log, dtype=float).reshape(-1, 5),
            market_volume=book.volume, trade_count=book.trade_count, tape_hash=book.tape_hash,
            decisions=self.decisions, tape=self.tape, regime=self.regime)

    def _sample(self, t: float) -> None:
        b = self.book
        bb, ba = b.best_bid(), b.best_ask()
        nan = math.nan
        if bb is not None and ba is not None:
            mid = 0.5 * (bb + ba)
            bq, aq = b.qty_at(BUY, bb), b.qty_at(SELL, ba)
            micro = (bb * aq + ba * bq) / (bq + aq)
        else:
            mid = micro = nan
        mmb = self._mm[BUY][1] if self._mm[BUY] is not None else nan
        mma = self._mm[SELL][1] if self._mm[SELL] is not None else nan
        self.samples.append((t, self.V, mid, micro, self.q, self.cash, self.rebates,
                             bb if bb is not None else nan, ba if ba is not None else nan, mmb, mma))


def run_session(market: MarketConfig, maker: MakerConfig, strategy: Strategy, seed: int,
                record_tape: bool = False) -> SessionResult:
    return Simulator(market, maker, strategy, seed, record_tape).run()
