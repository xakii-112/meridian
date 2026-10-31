"""Queue-aware market-making backtest on recorded Binance L2 data.

The recorded spot book, trades and best quotes are replayed in arrival order; the market maker's
orders are virtual (no market impact) and are matched against the real flow:

* Placement: an order joins the back of its price level; the quantity ahead of it is the level's
  size at that moment (the live book from the depth stream, or the fresher bookTicker size at the
  touch). Post-only: an order that would cross is not placed.
* Queue progress: real trades at our price consume the queue ahead first. A drop in the level size
  that trades do not explain is a cancellation; under the ``proportional`` model it removes queue
  ahead of us in proportion to our position, under ``conservative`` it removes none.
* Fills: trades at our price beyond the queue ahead fill us; a trade through our price, or the
  opposite touch moving onto our price, fills the remainder at our price.
* Latency: a decision made at receive-time t reaches the book at t + ``latency_s``. Receive times
  already include the feed delay, so ``latency_s`` is the full round trip seen from this machine.

Prices are integer ticks ($0.01), quantities BTC, cash in tick x BTC; the USD conversion happens in
the metrics. Fees are applied afterwards from the traded notional so one run serves every fee tier.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .data import DEPTH, FBBO, FTRADE, GAP, SBBO, STRADE, SYNC

HORIZONS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
FEE_BPS = (0.0, 0.5, 1.0, 2.0, 7.5, 10.0)


@dataclass
class Config:
    strategy: str = "glft"          # glft | symmetric | probe
    reference: str = "mid"          # mid | micro | lead (perp mid + EWMA basis)
    gamma: float = 0.05
    kappa: float = 0.05             # 1 / ticks
    A: float = 1.0                  # fills / s at zero depth
    sigma: float = 700.0            # ticks / sqrt(s)
    half_spread: float = 50.0       # ticks, symmetric strategy
    skew: float = 0.0               # ticks per quote unit of inventory, symmetric strategy
    probe_lo: float = 0.0           # ticks, probe strategy
    probe_hi: float = 300.0
    quote_size: float = 0.001       # BTC per quote
    max_units: int = 10             # inventory limit in quote units
    latency_s: float = 0.2
    requote_interval_s: float = 1.0
    requote_threshold: float = 5.0  # ticks of reference move that triggers a requote
    min_decision_gap_s: float = 0.05
    basis_halflife_s: float = 30.0
    lead_weight: float = 1.0        # lead = mid + w * (perp mid + basis - mid); w = regression beta is E[mid in 1 s]
    queue: str = "proportional"     # proportional | conservative
    start_s: float = 0.0            # trading window, seconds from the first event
    end_s: float = 1e18
    seed: int = 0
    sample_dt: float = 0.1


def prepare(ev: dict) -> dict:
    """Adds the merged event-time column (seconds from the first event) and Python lists for the
    columns the inner loop touches per event (numpy scalar indexing is slow)."""
    kinds, idx = ev["seq_kind"], ev["seq_idx"]
    t = np.zeros(len(kinds))
    sources = [(SBBO, ev["sbbo"][:, 0]), (FBBO, ev["fbbo"][:, 0]), (STRADE, ev["strd"][:, 0]),
               (FTRADE, ev["ftrd"][:, 0]), (DEPTH, ev["dt"]), (SYNC, ev["sync_t"])]
    if "gap_t" in ev:
        sources.append((GAP, ev["gap_t"]))
    for k, arr in sources:
        m = kinds == k
        t[m] = arr[idx[m]]
    gap = kinds == GAP  # files written before gaps were timestamped: inherit the previous event's time
    if gap.any() and "gap_t" not in ev:
        pos = np.where(~gap, np.arange(len(t)), 0)
        np.maximum.accumulate(pos, out=pos)
        t[gap] = t[pos[gap]]
    t0 = t[0]
    # The loop never reads perp trades, and perp quote updates that leave its mid unchanged.
    fb = ev["fbbo"]
    fm = 0.5 * (fb[:, 1] + fb[:, 2])
    moved = np.ones(len(fm), dtype=bool)
    moved[1:] = fm[1:] != fm[:-1]
    keep = kinds != FTRADE
    is_f = kinds == FBBO
    keep[is_f] &= moved[idx[is_f]]
    p = dict(ev)
    p["seq_t"] = ((t[keep] - t0) / 1e9).tolist()
    p["t0_ns"] = int(t0)
    p["kinds"] = kinds[keep].tolist()
    p["idx"] = idx[keep].tolist()
    sb = ev["sbbo"]
    p["sb"] = [sb[:, c].tolist() for c in range(1, 5)]
    p["fmid"] = fm.tolist()
    st = ev["strd"]
    p["st"] = [st[:, 1].astype(np.int64).tolist(), st[:, 2].tolist(), st[:, 3].astype(np.int64).tolist()]
    p["doff_l"] = ev["doff"].tolist()
    p["soff_l"] = ev["sync_off"].tolist()
    p["duration_s"] = p["seq_t"][-1]
    return p


class MarketMaker:
    """The virtual market maker and its view of the market.

    Feed events in arrival order through ``on_event(t, kind, payload)``; ``run`` does this for a
    recording and ``paper.py`` for the live Binance streams, so both share every rule below.
    Payloads: SBBO (bid, ask, bid_qty, ask_qty), FBBO perp mid, STRADE (price, qty, taker_side),
    DEPTH / SYNC (sides, prices, qtys), GAP None.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.book = ({}, {})
        self.book_ok = False
        self.bb = self.ba = None
        self.bq = self.aq = 0.0
        self.f_mid = None
        self.basis = None
        self.last_basis_t = 0.0
        self.tau = cfg.basis_halflife_s / math.log(2)
        self.orders = [None, None]  # [price, remaining, ahead, traded_at_price_since_depth, log_index]
        self.pending: deque = deque()
        self.q = self.cash = self.notional = 0.0
        self.msgs = 0
        self.fills: list = []
        self.samples: list = []
        self.qlog: list = []  # side, t_place, t_end, distance_from_mid, fill_events
        self.last_ref = None
        self.next_timer = cfg.start_s
        self.last_decision = -1e18
        self.next_sample = cfg.start_s
        self.filled_now = False
        self.proportional = cfg.queue == "proportional"
        self.size = cfg.quote_size
        self.max_q = cfg.max_units * cfg.quote_size
        self.t = None  # time of the last event processed
        if cfg.strategy == "glft":
            g, k = cfg.gamma, cfg.kappa
            self.c1 = math.log1p(g / k) / g
            self.c2 = math.sqrt(cfg.sigma ** 2 * g / (2 * k * cfg.A) * (1 + g / k) ** (1 + k / g))
        else:
            self.c1 = self.c2 = 0.0

    # ---- views -------------------------------------------------------------------------------

    def mid(self):
        return None if self.bb is None or self.ba is None else 0.5 * (self.bb + self.ba)

    def lead(self):
        m = self.mid()
        if self.f_mid is None or self.basis is None or m is None:
            return m
        return m + self.cfg.lead_weight * (self.f_mid + self.basis - m)

    def reference(self):
        r = self.cfg.reference
        if r == "lead":
            return self.lead()
        bb, ba, bq, aq = self.bb, self.ba, self.bq, self.aq
        if bb is None or ba is None:
            return None
        if r == "micro" and bq + aq > 0:
            return (bb * aq + ba * bq) / (bq + aq)
        return 0.5 * (bb + ba)

    # ---- order handling ----------------------------------------------------------------------

    def _close(self, side, t):
        o = self.orders[side]
        if o is not None:
            self.qlog[o[4]][2] = t
            self.orders[side] = None

    def _fill(self, side, x, t):
        o = self.orders[side]
        px = o[0]
        sgn = 1 if side == 0 else -1
        self.q += sgn * x
        self.cash -= sgn * px * x
        self.notional += px * x
        m = self.mid()
        self.fills.append((t, sgn, px, x, m if m is not None else px, self.lead() or px))
        o[1] -= x
        self.qlog[o[4]][4] += 1
        self.filled_now = True
        if o[1] <= 1e-12:
            self._close(side, t)

    def _decide(self, t):
        cfg = self.cfg
        self.last_decision = t
        self.next_timer = t + cfg.requote_interval_s
        ref = self.reference() if self.book_ok else None
        self.last_ref = ref
        if ref is None:
            self.pending.append((t + cfg.latency_s, None, None))
            return
        q, size = self.q, self.size
        if cfg.strategy == "glft":
            qn = q / size
            b, a = ref - (self.c1 + (2 * qn + 1) / 2 * self.c2), ref + (self.c1 - (2 * qn - 1) / 2 * self.c2)
        elif cfg.strategy == "symmetric":
            shift = cfg.skew * q / size
            b, a = ref - cfg.half_spread - shift, ref + cfg.half_spread - shift
        else:
            b, a = ref - self.rng.uniform(cfg.probe_lo, cfg.probe_hi), ref + self.rng.uniform(cfg.probe_lo, cfg.probe_hi)
        bid = math.floor(b) if q < self.max_q - 1e-12 else None
        ask = math.ceil(a) if q > -self.max_q + 1e-12 else None
        if bid is not None and self.ba is not None:
            bid = min(bid, self.ba - 1)
        if ask is not None and self.bb is not None:
            ask = max(ask, self.bb + 1)
        if bid is not None and ask is not None and ask <= bid:
            ask = bid + 1
        self.pending.append((t + cfg.latency_s, bid, ask))

    def _apply(self, t, bid, ask):
        m = self.mid()
        bb, ba = self.bb, self.ba
        for side, target in ((0, bid), (1, ask)):
            o = self.orders[side]
            if o is not None and o[0] == target:
                continue
            if o is not None:
                self._close(side, t)
                self.msgs += 1
            if target is None or not self.book_ok:
                continue
            if side == 0:
                if ba is not None and target >= ba:
                    continue  # post-only reject
                ahead = self.bq if target == bb else self.book[0].get(target, 0.0)
                dist = (m - target) if m is not None else 0.0
            else:
                if bb is not None and target <= bb:
                    continue
                ahead = self.aq if target == ba else self.book[1].get(target, 0.0)
                dist = (target - m) if m is not None else 0.0
            self.qlog.append([side, t, math.nan, dist, 0])
            self.orders[side] = [target, self.size, ahead, 0.0, len(self.qlog) - 1]
            self.msgs += 1

    def _sample(self, ts):
        m = self.mid()
        o = self.orders
        self.samples.append((ts, m if m is not None else math.nan, self.lead() or math.nan, self.q, self.cash,
                             o[0][0] if o[0] else math.nan, o[1][0] if o[1] else math.nan))

    # ---- event entry point -------------------------------------------------------------------

    def on_event(self, t, kind, payload):
        cfg = self.cfg
        self.t = t
        active = t >= cfg.start_s
        pending = self.pending
        while pending and pending[0][0] <= t:
            _, b_, a_ = pending.popleft()
            if active:
                self._apply(t, b_, a_)
        if active:
            while self.next_sample <= t:
                self._sample(self.next_sample)
                self.next_sample += cfg.sample_dt

        orders = self.orders
        if kind == SBBO:
            bb, ba, bq, aq = payload
            self.bb, self.ba, self.bq, self.aq = bb, ba, bq, aq
            f_mid = self.f_mid
            if f_mid is not None:
                sm = 0.5 * (bb + ba)
                if self.basis is None:
                    self.basis = sm - f_mid
                else:
                    self.basis += (1.0 - math.exp(-(t - self.last_basis_t) / self.tau)) * (sm - f_mid - self.basis)
                self.last_basis_t = t
            # The opposite touch moving onto our price would have traded with us.
            if orders[0] is not None and ba <= orders[0][0]:
                self._fill(0, orders[0][1], t)
            if orders[1] is not None and bb >= orders[1][0]:
                self._fill(1, orders[1][1], t)
        elif kind == FBBO:
            self.f_mid = payload
        elif kind == STRADE:
            px, v, taker = payload
            side = 0 if taker < 0 else 1  # a taker sell hits bids
            o = orders[side]
            if o is not None:
                through = px < o[0] if side == 0 else px > o[0]
                if through:
                    self._fill(side, o[1], t)
                elif px == o[0]:
                    o[3] += v
                    o[2] -= v
                    if o[2] < 0.0:
                        x = min(o[1], -o[2])
                        o[2] = 0.0
                        self._fill(side, x, t)
        elif kind == DEPTH:
            book, proportional = self.book, self.proportional
            for s, px, qty in zip(*payload):
                lvl = book[s]
                old = lvl.get(px, 0.0)
                if qty == 0.0:
                    lvl.pop(px, None)
                else:
                    lvl[px] = qty
                o = orders[s]
                if o is not None and o[0] == px:
                    traded, o[3] = o[3], 0.0
                    cancelled = (old - qty) - traded
                    base = old - traded
                    if proportional and cancelled > 0 and base > 0:
                        o[2] -= cancelled * o[2] / base
                    o[2] = max(0.0, min(o[2], qty))
        elif kind == SYNC:
            book = ({}, {})
            for s, px, qty in zip(*payload):
                book[s][px] = qty
            self.book = book
            self.book_ok = True
            for s in (0, 1):
                if orders[s] is not None:
                    orders[s][2] = min(orders[s][2], book[s].get(orders[s][0], 0.0))
        elif kind == GAP:
            self.book_ok = False
            if active:
                self._close(0, t)
                self._close(1, t)

        if active and self.book_ok and self.bb is not None and t - self.last_decision >= cfg.min_decision_gap_s:
            go = self.filled_now or t >= self.next_timer
            if not go:
                ref = self.lead() if cfg.reference == "lead" else self.reference()
                go = ref is not None and (self.last_ref is None or abs(ref - self.last_ref) >= cfg.requote_threshold)
            if go:
                self.filled_now = False
                self._decide(t)

    # ---- results -----------------------------------------------------------------------------

    def status(self, tick: float = 0.01) -> dict:
        """Point-in-time view for a live display (USD, gross of fees)."""
        m = self.mid()
        mark = m if m is not None else 0.0
        vol = sum(f[3] for f in self.fills)
        return {"t": self.t, "mid": m, "lead": self.lead(), "book_ok": self.book_ok,
                "bid": None if self.orders[0] is None else (self.orders[0][0], self.orders[0][2]),
                "ask": None if self.orders[1] is None else (self.orders[1][0], self.orders[1][2]),
                "q": self.q, "fills": len(self.fills), "volume_btc": vol, "notional_usd": self.notional * tick,
                "gross_usd": (self.cash + self.q * mark) * tick}

    def finish(self, t_end: float | None = None, record: bool = False) -> dict:
        t_end = (self.t if self.t is not None else self.cfg.start_s) if t_end is None else t_end
        self._close(0, t_end)
        self._close(1, t_end)
        m = self.mid()
        self.samples.append((t_end, m if m is not None else math.nan, self.lead() or math.nan, self.q, self.cash,
                             math.nan, math.nan))
        return summarize(self.cfg, self.fills, self.samples, self.qlog, self.q, self.cash, self.notional, self.msgs,
                         record)


def run(p: dict, cfg: Config, record: bool = False) -> dict:
    """Replays a prepared recording through a MarketMaker."""
    mm = MarketMaker(cfg)
    kinds, idxs, seq_t = p["kinds"], p["idx"], p["seq_t"]
    sb_b, sb_a, sb_bq, sb_aq = p["sb"]
    fmid = p["fmid"]
    st_px, st_q, st_side = p["st"]
    doff, dside, dpx, dqty = p["doff_l"], p["dside"], p["dpx"], p["dqty"]
    soff, sside, spx, sqty = p["soff_l"], p["sync_side"], p["sync_px"], p["sync_qty"]
    end_s = min(cfg.end_s, p["duration_s"])
    on_event = mm.on_event
    for i in range(len(kinds)):
        t = seq_t[i]
        if t > end_s:
            break
        kind, j = kinds[i], idxs[i]
        if kind == SBBO:
            payload = (int(sb_b[j]), int(sb_a[j]), sb_bq[j], sb_aq[j])
        elif kind == FBBO:
            payload = fmid[j]
        elif kind == STRADE:
            payload = (st_px[j], st_q[j], st_side[j])
        elif kind == DEPTH:
            a0, a1 = doff[j], doff[j + 1]
            payload = (dside[a0:a1].tolist(), dpx[a0:a1].tolist(), dqty[a0:a1].tolist())
        elif kind == SYNC:
            a0, a1 = soff[j], soff[j + 1]
            payload = (sside[a0:a1].tolist(), spx[a0:a1].tolist(), sqty[a0:a1].tolist())
        else:
            payload = None
        on_event(t, kind, payload)
    return mm.finish(mm.t if mm.t is not None else end_s, record)


def _markouts(fills: np.ndarray, st: np.ndarray, sp: np.ndarray, tick: float) -> dict:
    out = {}
    if len(fills) == 0:
        return {str(h): math.nan for h in HORIZONS}
    t, sgn, px, x = fills[:, 0], fills[:, 1], fills[:, 2], fills[:, 3]
    for h in HORIZONS:
        keep = t + h <= st[-1]
        if not keep.any():
            out[str(h)] = math.nan
            continue
        k = np.clip(np.searchsorted(st, t[keep] + h, side="right") - 1, 0, len(sp) - 1)
        out[str(h)] = float(np.sum(sgn[keep] * (sp[k] - px[keep]) * x[keep]) / np.sum(x[keep]) * tick)
    return out


def summarize(cfg, fills, samples, qlog, q, cash, notional, msgs, record, tick: float = 0.01) -> dict:
    f = np.asarray(fills, dtype=float).reshape(-1, 6)
    s = np.asarray(samples, dtype=float).reshape(-1, 7)
    ql = np.asarray(qlog, dtype=float).reshape(-1, 5)
    if len(s) == 0:
        return {"config": asdict(cfg), "hours": 0.0}
    st = s[:, 0]
    sm = s[:, 1].copy()
    good = ~np.isnan(sm)
    sm[~good] = np.interp(st[~good], st[good], sm[good]) if good.any() else 0.0
    mtm = (s[:, 4] + s[:, 3] * sm) * tick          # USD, gross of fees
    hours = (st[-1] - st[0]) / 3600.0
    gross = float(mtm[-1])
    vol = float(f[:, 3].sum()) if len(f) else 0.0
    notional_usd = notional * tick
    spread = float(np.sum(f[:, 1] * (f[:, 4] - f[:, 2]) * f[:, 3]) * tick) if len(f) else 0.0
    minute = np.diff(mtm[:: max(1, int(round(60 / cfg.sample_dt)))])
    both = ~np.isnan(s[:, 5]) & ~np.isnan(s[:, 6])
    out = {
        "config": asdict(cfg),
        "hours": hours,
        "fills": int(len(f)),
        "volume_btc": vol,
        "notional_usd": notional_usd,
        "gross_pnl_usd": gross,
        "gross_pnl_per_hour": gross / hours if hours else 0.0,
        # standard error from per-minute PnL increments (treated as independent)
        "gross_pnl_per_hour_se": float(np.std(minute, ddof=1) / math.sqrt(len(minute)) * 60) if len(minute) > 2 else math.nan,
        "spread_usd": spread,
        "inventory_usd": gross - spread,
        "gross_edge_per_btc": gross / vol if vol else math.nan,
        "spread_per_btc": spread / vol if vol else math.nan,
        "breakeven_fee_bps": gross / notional_usd * 1e4 if notional_usd else math.nan,
        "net_pnl_per_hour": {str(b): (gross - notional_usd * b / 1e4) / hours if hours else 0.0 for b in FEE_BPS},
        "markout_usd_per_btc": _markouts(f, st, sm, tick),
        "inventory_abs_mean_btc": float(np.mean(np.abs(s[:, 3]))),
        "inventory_abs_max_btc": float(np.max(np.abs(s[:, 3]))),
        "quote_uptime": float(np.mean(both)),
        "quoted_spread_ticks": float(np.mean(s[both, 6] - s[both, 5])) if both.any() else math.nan,
        "msgs_per_sec": msgs / (hours * 3600) if hours else 0.0,
        "max_drawdown_usd": float(np.max(np.maximum.accumulate(mtm) - mtm)),
        "minute_pnl_ratio": float(np.mean(minute) / np.std(minute)) if len(minute) > 2 and np.std(minute) > 0 else math.nan,
    }
    if record:
        out["fills_arr"] = f
        out["samples_arr"] = s
        out["quote_log"] = ql
    else:
        out["quote_log"] = ql if cfg.strategy == "probe" else None
    return out
