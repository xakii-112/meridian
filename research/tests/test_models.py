import math
import random

import numpy as np
import pytest

from meridian_research import hawkes
from meridian_research.analytics import aggregate, markouts, max_drawdown
from meridian_research.calibration import fill_intensity, fit_exponential
from meridian_research.strategies import GLFT, AvellanedaStoikov, MarketView, Symmetric, reference_price, to_ticks
from meridian_research.wire import FRAME, HEADER, TapeWriter, read_tape


# ---- Hawkes ------------------------------------------------------------------------------

def test_hawkes_empirical_rate_matches_stationary_intensity():
    mu, alpha, beta = [0.5, 0.5], [[0.8, 0.3], [0.3, 0.8]], 2.0
    events = hawkes.simulate(mu, alpha, beta, 20_000.0, random.Random(7))
    counts = np.bincount([d for _, d in events], minlength=2) / 20_000.0
    expected = hawkes.stationary_intensity(mu, alpha, beta)
    assert np.allclose(counts, expected, rtol=0.06)
    assert all(a[0] < b[0] for a, b in zip(events, events[1:]))


def test_hawkes_clusters_more_than_poisson():
    events = hawkes.simulate([1.0], [[1.5]], 2.0, 5_000.0, random.Random(3))
    counts = np.histogram([t for t, _ in events], bins=np.arange(0, 5_001, 10.0))[0]
    assert counts.var() / counts.mean() > 2.0  # Poisson would give a dispersion index of ~1


def test_hawkes_rejects_explosive_kernel():
    with pytest.raises(ValueError):
        hawkes.simulate([1.0], [[2.5]], 2.0, 10.0, random.Random(0))


# ---- strategies --------------------------------------------------------------------------

def view(inventory=0, lead=100.3, bb=99, ba=101, bq=5, aq=5):
    return MarketView(0.0, lead, bb, ba, bq, aq, inventory)


def test_glft_is_symmetric_flat_and_skews_with_inventory():
    s = GLFT(gamma=0.1, kappa=0.7, A=0.7, sigma=1.0)
    b0, a0 = s.levels(100.0, view(0))
    assert math.isclose(100.0 - b0, a0 - 100.0)
    assert math.isclose(100.0 - b0, s.half_spread())
    b1, a1 = s.levels(100.0, view(10))  # long: lean both quotes down to sell
    assert b1 < b0 and a1 < a0
    assert math.isclose((b0 - b1), s.c2 * 10 / s.size)


def test_glft_spread_widens_with_volatility_and_narrows_with_kappa():
    base = GLFT(gamma=0.1, kappa=0.7, A=0.7, sigma=1.0).half_spread()
    assert GLFT(gamma=0.1, kappa=0.7, A=0.7, sigma=2.0).half_spread() > base
    assert GLFT(gamma=0.1, kappa=1.5, A=0.7, sigma=1.0).half_spread() < base


def test_avellaneda_stoikov_reservation_price():
    s = AvellanedaStoikov(gamma=0.1, kappa=1.0, sigma=1.0, horizon_s=10.0)
    b, a = s.levels(100.0, view(4))
    reservation = 100.0 - (4 / s.size) * 0.1 * 1.0 * 10.0
    assert math.isclose((a + b) / 2, reservation)


def test_to_ticks_rounds_away_keeps_passive_and_respects_limits():
    q = to_ticks(view(), 99.6, 100.4, 2, 50)
    assert (q.bid, q.ask) == (99, 101)
    q = to_ticks(view(bb=100, ba=101), 101.2, 101.9, 2, 50)  # would cross the ask
    assert q.bid == 100 and q.ask == 102
    q = to_ticks(view(inventory=50), 99.0, 101.0, 2, 50)
    assert q.bid is None and q.ask == 101


def test_reference_modes():
    v = view(bb=99, ba=101, bq=1, aq=3)
    assert reference_price(v, "lead") == 100.3
    assert reference_price(v, "mid") == 100.0
    assert reference_price(v, "micro") == pytest.approx((99 * 3 + 101 * 1) / 4)
    assert reference_price(view(bb=None, ba=101), "mid") == 100.5


def test_symmetric_quotes():
    s = Symmetric(half_spread=1.0)
    q = s.quote(view(lead=100.0))
    assert (q.bid, q.ask) == (99, 101)


# ---- calibration -------------------------------------------------------------------------

def test_fit_exponential_recovers_parameters():
    rng = np.random.default_rng(1)
    A, kappa = 0.8, 0.9
    rows = []
    for d in rng.uniform(0, 4, 40_000):
        life = rng.exponential(2.0)
        rows.append([0, 0.0, life, d, rng.poisson(A * math.exp(-kappa * d) * life)])
    fi = fill_intensity([np.array(rows)], np.arange(0, 4.01, 0.25))
    fit = fit_exponential(fi["depth"], fi["intensity"], fi["fills"])
    assert fit["A"] == pytest.approx(A, rel=0.1)
    assert fit["kappa"] == pytest.approx(kappa, rel=0.05)
    assert fit["r2"] > 0.95


# ---- analytics ---------------------------------------------------------------------------

def test_markouts_and_drawdown():
    times = np.arange(0, 10.0, 0.1)
    mid = np.where(times < 5.0, 100.0, 102.0)
    fills = np.array([[1.0, 1, 99.0, 2, 100.0, 100.0, 2],     # bought 2 @ 99
                      [4.0, -1, 101.0, 1, 100.0, 100.0, 1]])  # sold 1 @ 101
    m = markouts(fills, times, mid, horizons=(0.5, 2.0, 8.5, 9.5))
    assert m[0.5] == pytest.approx((2 * 1 + 1 * 1) / 3)
    assert m[2.0] == pytest.approx((2 * 1 + 1 * -1) / 3)  # second fill marked after the jump
    assert m[8.5] == pytest.approx((2 * 3) / 2)            # only the first fill has an 8.5s horizon left
    assert np.isnan(m[9.5])                                # no fill has a full 9.5s horizon
    assert max_drawdown(np.array([0, 5, 3, 7, 1, 4])) == 6


def test_aggregate_sharpe():
    sessions = [{"pnl_usd": p, "hours": 1.0, "markout_mid": {1.0: 0.0}, "markout_fair": {1.0: 0.0},
                 **{k: 0.0 for k in ("spread_usd", "rebates_usd", "inventory_usd", "max_drawdown_usd", "fills", "lots",
                                     "volume_share", "pnl_per_lot_ticks", "edge_per_lot_ticks", "inventory_abs_mean",
                                     "inventory_std", "time_at_limit", "quote_uptime", "quoted_spread_ticks")}}
                for p in (10.0, 20.0, 30.0)]
    a = aggregate(sessions)
    assert a["pnl_mean_usd"] == 20.0
    assert a["sharpe_annualized"] == pytest.approx(20.0 / 10.0 * math.sqrt(252 * 6.5))


# ---- wire --------------------------------------------------------------------------------

def test_wire_layout_matches_engine():
    w = TapeWriter(symbol=3, min_price=1, max_price=1 << 20)
    w.enter(7, 42, 1, 10_050, 9, ioc=True, market=False, post_only=False, ts_ns=123)
    w.cancel(7, 42, ts_ns=456)
    data = w.to_bytes()
    assert len(data) == HEADER.size + 2 * FRAME.size == 40 + 96
    assert data[:4] == b"MWP1"
    header, frames = read_tape(data)
    assert header == {"symbol": 3, "count": 2, "min_price": 1, "max_price": 1 << 20}
    assert frames[0] == (ord("O"), 1, 0, 1, 0, 0, 3, 7, 0, 42, 10_050, 9, 123)
    assert frames[1][0] == ord("X") and frames[1][9] == 42
