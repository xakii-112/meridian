"""Queue-model and book-reconstruction tests on hand-built event streams."""

import math

import numpy as np
import pytest

from meridian_research.binance.backtest import Config, prepare, run
from meridian_research.binance.calibrate import _market_orders, fit_hawkes
from meridian_research.binance.data import DEPTH, FBBO, GAP, SBBO, STRADE, SYNC, DepthBook

NS = 1_000_000_000


class Tape:
    """Builds the arrays ``data.preprocess`` would produce, one event at a time."""

    def __init__(self, bids=((100, 2.0),), asks=((101, 2.0),)):
        self.sbbo, self.fbbo, self.strd, self.dt, self.doff = [], [], [], [], []
        self.dside, self.dpx, self.dqty = [], [], []
        self.kind, self.idx = [], []
        levels = [(0, p, q) for p, q in bids] + [(1, p, q) for p, q in asks]
        self.sync = levels
        self.kind.append(SYNC)
        self.idx.append(0)
        self.bbo(0.0, bids[0][0], asks[0][0], bids[0][1], asks[0][1])

    def bbo(self, t, b, a, bq, aq):
        self.sbbo.append((t * NS, b, a, bq, aq))
        self.kind.append(SBBO)
        self.idx.append(len(self.sbbo) - 1)
        return self

    def trade(self, t, px, qty, taker):
        self.strd.append((t * NS, px, qty, taker, t * 1e3))
        self.kind.append(STRADE)
        self.idx.append(len(self.strd) - 1)
        return self

    def depth(self, t, changes):
        self.dt.append(int(t * NS))
        self.doff.append(len(self.dpx))
        for s, p, q in changes:
            self.dside.append(s)
            self.dpx.append(p)
            self.dqty.append(q)
        self.kind.append(DEPTH)
        self.idx.append(len(self.dt) - 1)
        return self

    def gap(self):
        self.kind.append(GAP)
        self.idx.append(-1)
        return self

    def fut(self, t, b, a):
        self.fbbo.append((t * NS, b, a, 1.0, 1.0))
        self.kind.append(FBBO)
        self.idx.append(len(self.fbbo) - 1)
        return self

    def build(self):
        ev = {
            "sbbo": np.array(self.sbbo, dtype=float).reshape(-1, 5),
            "fbbo": np.array(self.fbbo or [(0.0, 100.0, 101.0, 1.0, 1.0)], dtype=float).reshape(-1, 5),
            "strd": np.array(self.strd or [(0, 0, 0, 0, 0)], dtype=float).reshape(-1, 5),
            "ftrd": np.zeros((1, 5)),
            "dt": np.array(self.dt, dtype=np.int64), "doff": np.array(self.doff + [len(self.dpx)], dtype=np.int64),
            "dside": np.array(self.dside, dtype=np.int8), "dpx": np.array(self.dpx, dtype=np.int64),
            "dqty": np.array(self.dqty, dtype=float),
            "sync_t": np.array([0], dtype=np.int64), "sync_off": np.array([0, len(self.sync)], dtype=np.int64),
            "sync_side": np.array([s for s, _, _ in self.sync], dtype=np.int8),
            "sync_px": np.array([p for _, p, _ in self.sync], dtype=np.int64),
            "sync_qty": np.array([q for _, _, q in self.sync], dtype=float),
            "seq_kind": np.array(self.kind, dtype=np.int8), "seq_idx": np.array(self.idx, dtype=np.int64),
        }
        return prepare(ev)


def touch_quoter(**kw):
    # Quotes the touch once (bid 100 / ask 101 for a 100/101 book) and never requotes on its own.
    base = dict(strategy="symmetric", half_spread=0.5, quote_size=1.0, max_units=5, latency_s=0.01,
                requote_interval_s=1e9, requote_threshold=1e9, sample_dt=0.05)
    base.update(kw)
    return Config(**base)


def fills_of(res):
    return [(round(f[0], 3), int(f[1]), int(f[2]), round(f[3], 6)) for f in res["fills_arr"]]


def test_queue_ahead_is_consumed_by_trades_before_we_fill():
    tape = Tape().bbo(0.1, 100, 101, 2.0, 2.0)
    tape.trade(0.2, 100, 1.5, -1).trade(0.3, 100, 1.0, -1).trade(0.4, 100, 1.0, -1).bbo(0.5, 100, 101, 1.0, 2.0)
    res = run(tape.build(), touch_quoter(), record=True)
    assert fills_of(res) == [(0.3, 1, 100, 0.5), (0.4, 1, 100, 0.5)]


def test_trades_before_the_order_is_live_do_not_fill_it():
    tape = Tape().trade(0.005, 100, 5.0, -1).bbo(0.1, 100, 101, 2.0, 2.0)
    res = run(tape.build(), touch_quoter(), record=True)
    assert res["fills"] == 0


def test_proportional_cancellation_moves_us_up_the_queue():
    def build():
        tape = Tape().bbo(0.1, 100, 101, 2.0, 2.0)
        tape.depth(0.2, [(0, 100, 4.0)])          # 2 lots join behind us
        tape.depth(0.3, [(0, 100, 3.0)])          # 1 lot cancelled, no trades
        tape.trade(0.4, 100, 1.6, -1).bbo(0.5, 100, 101, 1.4, 2.0)
        return tape.build()
    # ahead 2.0 -> 2.0 - 1.0 * 2/4 = 1.5; a 1.6 trade then fills 0.1
    assert fills_of(run(build(), touch_quoter(), record=True)) == [(0.4, 1, 100, 0.1)]
    assert run(build(), touch_quoter(queue="conservative"), record=True)["fills"] == 0


def test_trade_through_and_crossing_quotes_fill_the_remainder():
    tape = Tape().bbo(0.1, 100, 101, 2.0, 2.0).trade(0.2, 99, 0.3, -1)  # a sell printed below our bid
    tape.bbo(0.3, 99, 100, 1.0, 1.0)                                        # ask now at our old... (bid gone)
    res = run(tape.build(), touch_quoter(), record=True)
    assert fills_of(res)[0] == (0.2, 1, 100, 1.0)
    tape = Tape().bbo(0.1, 100, 101, 2.0, 2.0).bbo(0.2, 101, 102, 1.0, 1.0)  # bid moved onto our ask
    assert fills_of(run(tape.build(), touch_quoter(), record=True)) == [(0.2, -1, 101, 1.0)]


def test_post_only_never_crosses_and_gap_pulls_quotes():
    # Asking for a bid at 101 against a 101 ask is refused rather than taking liquidity.
    tape = Tape().bbo(0.1, 100, 101, 2.0, 2.0).trade(0.2, 101, 3.0, 1)
    res = run(tape.build(), touch_quoter(half_spread=-0.6), record=True)
    assert all(f[1] == -1 or f[2] < 101 for f in res["fills_arr"])
    tape = Tape().bbo(0.1, 100, 101, 2.0, 2.0).gap().trade(0.3, 99, 5.0, -1).bbo(0.4, 99, 100, 1.0, 1.0)
    assert run(tape.build(), touch_quoter(), record=True)["fills"] == 0


def test_perp_led_reference_moves_quotes_by_the_weighted_gap():
    def ask_after_perp_jump(**kw):
        # Basis is fixed by the first spot update (spot 100.5 vs perp 50.5); the perp then rallies
        # 20 ticks while spot stays put, so the perp-led fair value is mid + w * 20.
        tape = Tape().fut(0.0, 50, 51)
        tape.bbo(0.05, 100, 101, 2.0, 2.0).fut(0.1, 70, 71).bbo(0.2, 100, 101, 2.0, 2.0).bbo(0.4, 100, 101, 2.0, 2.0)
        cfg = touch_quoter(half_spread=5.0, requote_threshold=1.0, **kw)
        res = run(tape.build(), cfg, record=True)
        return int(res["quote_log"][-1][0]), [int(x) for x in res["samples_arr"][-2, 5:7]]
    assert ask_after_perp_jump(reference="mid")[1][1] == 106                     # ceil(100.5 + 5)
    assert ask_after_perp_jump(reference="lead", lead_weight=1.0)[1][1] == 126  # ceil(120.5 + 5)
    assert ask_after_perp_jump(reference="lead", lead_weight=0.5)[1][1] == 116  # ceil(110.5 + 5)


def test_cash_and_inventory_reconcile():
    tape = Tape(bids=((100, 0.0),), asks=((101, 0.0),)).bbo(0.1, 100, 101, 0.0, 0.0)
    tape.trade(0.2, 100, 1.0, -1).trade(0.3, 101, 1.0, 1).bbo(0.4, 100, 101, 1, 1)
    res = run(tape.build(), touch_quoter(), record=True)
    assert res["volume_btc"] == 2.0
    assert res["gross_pnl_usd"] == pytest.approx(0.01)       # bought 100, sold 101: one tick of 1 BTC
    assert res["spread_usd"] == pytest.approx(0.01)
    assert res["breakeven_fee_bps"] == pytest.approx(0.01 / (201 * 0.01) * 1e4)


def test_depth_book_sequence_rules():
    b = DepthBook(0.01)
    b.load_snapshot({"lastUpdateId": 10, "bids": [["1.00", "2"]], "asks": [["1.01", "3"]]})
    assert b.apply({"U": 5, "u": 9, "b": [["1.00", "9"]], "a": []}) is None        # older than snapshot
    assert b.apply({"U": 9, "u": 12, "b": [["1.00", "1"]], "a": []}) is not None   # straddles L + 1
    assert b.side[0][100] == 1.0
    assert b.apply({"U": 13, "u": 13, "b": [], "a": [["1.01", "0"]]}) is not None
    assert 101 not in b.side[1]
    assert b.apply({"U": 15, "u": 16, "b": [], "a": []}) is None and b.gaps == 1   # 14 missing
    assert not b.synced


def test_realized_spread_marks_the_maker_side():
    from meridian_research.binance.calibrate import realized_spreads
    # mid 100.5 until t=5, then 110.5; a taker buy at t=2 (maker sold at 101) and a taker sell at t=3
    # (maker bought at 100).
    sbbo = np.array([[0, 100, 101, 1, 1], [5 * NS, 110, 111, 1, 1], [200 * NS, 110, 111, 1, 1]], float)
    fbbo = np.array([[0, 100, 101, 1, 1], [200 * NS, 110, 111, 1, 1]], float)
    strd = np.array([[2 * NS, 101, 1.0, 1, 2000], [3 * NS, 100, 3.0, -1, 3000]], float)
    p = {"t0_ns": 0, "sbbo": sbbo, "fbbo": fbbo, "strd": strd}
    r = realized_spreads(p, 1.0, 100.0, horizons=(1.0, 10.0))
    m = r["groups"]["all"]["markout"]
    assert m["1.0"]["usd_per_btc"] == pytest.approx((-1 * (100.5 - 101) * 1 + 1 * (100.5 - 100) * 3) / 4 * 0.01)
    assert m["10.0"]["usd_per_btc"] == pytest.approx((-1 * (110.5 - 101) * 1 + 1 * (110.5 - 100) * 3) / 4 * 0.01)


def test_paper_session_equals_preprocess_plus_backtest(tmp_path):
    """The live path (raw messages -> Decoder -> MarketMaker) must match the offline path
    (preprocess -> prepare -> run) message for message."""
    from meridian_research.binance.backtest import prepare
    from meridian_research.binance.data import load, preprocess
    from meridian_research.binance.paper import PaperSession
    from meridian_research.binance.recorder import ChunkWriter

    t = 1_700_000_000_000_000_000
    ms = lambda x: t + int(x * 1e9)
    w = ChunkWriter(tmp_path, 1e9)
    w.write(ms(0.00), "spot", "connected", {})
    w.write(ms(0.01), "spot", "btcusdt@depth@100ms", {"U": 101, "u": 105, "b": [["100.00", "2"]], "a": []})
    w.write(ms(0.02), "fut", "btcusdt@bookTicker", {"b": "99.50", "a": "99.60", "B": "1", "A": "1"})
    w.write(ms(0.05), "spot", "snapshot", {"lastUpdateId": 103, "bids": [["100.00", "2"]], "asks": [["100.01", "2"]]})
    w.write(ms(0.06), "spot", "btcusdt@bookTicker", {"b": "100.00", "a": "100.01", "B": "2", "A": "2"})
    w.write(ms(0.10), "spot", "btcusdt@depth@100ms", {"U": 106, "u": 107, "b": [["100.00", "2.5"]], "a": []})
    for k in range(40):
        x = 0.2 + k * 0.05
        px = "100.00" if k % 2 else "100.01"
        w.write(ms(x), "spot", "btcusdt@trade", {"p": px, "q": "0.8", "m": k % 2 == 1, "T": 1})
        w.write(ms(x + 0.01), "fut", "btcusdt@bookTicker", {"b": f"{99.5 + 0.01 * (k % 7):.2f}", "a": "99.70", "B": "1", "A": "1"})
        w.write(ms(x + 0.02), "spot", "btcusdt@bookTicker", {"b": "100.00", "a": "100.01", "B": "1", "A": "1"})
        w.write(ms(x + 0.03), "spot", "btcusdt@depth@100ms", {"U": 108 + k, "u": 108 + k, "b": [["100.00", f"{1 + k % 3}"]], "a": [["100.01", f"{2 + k % 2}"]]})
    w.close()

    cfg = touch_quoter(reference="lead", lead_weight=0.5, requote_interval_s=0.3, end_s=1e9)
    offline = run(prepare(load_after(preprocess, load, tmp_path)), cfg, record=True)
    sess = PaperSession(cfg)
    from meridian_research.binance.data import records
    for rec in records(tmp_path):
        sess.on_message(*rec)
    live = sess.finish()
    assert offline["fills"] > 5
    assert (offline["fills_arr"] == live["fills_arr"]).all()
    assert offline["gross_pnl_usd"] == live["gross_pnl_usd"]
    assert offline["msgs_per_sec"] == live["msgs_per_sec"]


def load_after(preprocess, load, path):
    preprocess(path)
    return load(path)


def test_market_orders_group_trades_and_hawkes_detects_clustering():
    st = np.array([[0, 1, 0.1, 1, 1000], [0, 1, 0.2, 1, 1000], [0, 1, 0.1, -1, 1000], [0, 1, 0.1, 1, 1001]], float)
    mo = _market_orders(st)
    assert mo.tolist() == [[1.0, 1.0, pytest.approx(0.3)], [1.0, -1.0, 0.1], [1.001, 1.0, 0.1]]
    # Simulated self-exciting flow: the fit must find a branching ratio well above zero.
    from meridian_research.hawkes import simulate
    import random
    ev = simulate([0.5, 0.5], [[1.2, 0.3], [0.3, 1.2]], 3.0, 4000.0, random.Random(1))
    rows = [[0, 0, 1.0, 1 if d == 0 else -1, t * 1e3] for t, d in ev]
    p = {"strd": np.array(rows, float), "t0_ns": 0}
    fit = fit_hawkes(p, -1, 1e18)
    assert fit["branching_ratio"] == pytest.approx(0.5, abs=0.12)
    assert fit["loglik"] > fit["poisson_loglik"]
