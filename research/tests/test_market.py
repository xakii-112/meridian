import numpy as np
import pytest

from meridian_research.analytics import S, session_metrics
from meridian_research.market import FILL_COLS, MakerConfig, MarketConfig, run_session
from meridian_research.strategies import GLFT, Symmetric

F = {c: i for i, c in enumerate(FILL_COLS)}
MARKET = MarketConfig(duration_s=600.0)


def glft(**kw):
    return GLFT(gamma=0.1, kappa=0.7, A=0.7, sigma=1.0, **kw)


def test_sessions_are_deterministic_per_seed():
    a = run_session(MARKET, MakerConfig(), glft(), seed=11)
    b = run_session(MARKET, MakerConfig(), glft(), seed=11)
    c = run_session(MARKET, MakerConfig(), glft(), seed=12)
    assert a.tape_hash == b.tape_hash and np.array_equal(a.fills, b.fills)
    assert a.tape_hash != c.tape_hash


def test_inventory_and_cash_reconcile_with_fills():
    res = run_session(MARKET, MakerConfig(), glft(), seed=3)
    f, s = res.fills, res.samples
    assert len(f) > 50
    signed = f[:, F["side"]] * f[:, F["qty"]]
    assert np.array_equal(np.cumsum(signed), f[:, F["inventory"]])
    last_t = s[-1, S["t"]]
    done = f[:, F["t"]] <= last_t
    assert s[-1, S["inventory"]] == signed[done].sum()
    assert s[-1, S["cash"]] == pytest.approx(-(signed[done] * f[done, F["price"]]).sum())


def test_inventory_limit_is_respected():
    res = run_session(MARKET, MakerConfig(), Symmetric(half_spread=0.5, max_inventory=6, size=2), seed=5)
    # A quote is only sent while |q| < limit, so a single fill can overshoot by at most one quote size.
    assert np.abs(res.fills[:, F["inventory"]]).max() <= 6 + 2


def test_pnl_attribution_adds_up():
    m = session_metrics(run_session(MARKET, MakerConfig(), glft(), seed=4))
    assert m["pnl_usd"] == pytest.approx(m["spread_usd"] + m["rebates_usd"] + m["inventory_usd"])
    assert 0 < m["quote_uptime"] <= 1


def test_lead_reference_beats_mid_reference_against_informed_flow():
    lead = np.mean([session_metrics(run_session(MARKET, MakerConfig(), glft(), s))["pnl_usd"] for s in (1, 2, 3)])
    mid = np.mean([session_metrics(run_session(MARKET, MakerConfig(), glft(reference="mid"), s))["pnl_usd"]
                   for s in (1, 2, 3)])
    assert lead > mid


def test_tape_records_every_order():
    res = run_session(MarketConfig(duration_s=120.0), MakerConfig(), glft(), seed=2, record_tape=True)
    assert len(res.tape) > res.trade_count > 0
