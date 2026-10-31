import json
import random
import subprocess
from pathlib import Path

import pytest

from meridian_research.lob import BUY, HASH_SEED, SELL, LimitOrderBook, mix_trade
from meridian_research.wire import TapeWriter

ROOT = Path(__file__).resolve().parents[2]
REPLAY = next((p for p in (ROOT / "engine/build/meridian_replay.exe", ROOT / "engine/build/meridian_replay") if p.exists()),
              None)


def test_rests_and_reports_best_prices():
    b = LimitOrderBook()
    b.submit(1, 1, BUY, 99, 10)
    b.submit(1, 2, SELL, 101, 5)
    assert (b.best_bid(), b.best_ask()) == (99, 101)
    assert b.qty_at(BUY, 99) == 10 and b.qty_at(SELL, 101) == 5
    assert b.trade_count == 0


def test_fills_at_maker_price_with_time_priority():
    b = LimitOrderBook()
    b.submit(2, 1, SELL, 100, 2)
    b.submit(3, 2, SELL, 100, 2)
    b.submit(4, 3, SELL, 101, 5)
    ok, fills, leaves = b.submit(1, 9, BUY, 105, 5)
    assert ok and leaves == 0
    assert fills == [(2, 1, 100, 2, 0), (3, 2, 100, 2, 0), (4, 3, 101, 1, 4)]
    assert b.best_ask() == 101 and b.qty_at(SELL, 101) == 4
    assert b.last_trade == 101


def test_partial_fill_rests_remainder_but_ioc_does_not():
    b = LimitOrderBook()
    b.submit(2, 1, SELL, 100, 3)
    _, _, leaves = b.submit(1, 2, BUY, 100, 5)
    assert leaves == 2 and b.best_bid() == 100
    b.submit(2, 3, SELL, 102, 3)
    _, fills, leaves = b.submit(1, 4, BUY, 102, 5, ioc=True)
    assert leaves == 2 and len(fills) == 1
    assert b.leaves(1, 4) == 0


def test_market_order_sweeps_and_needs_ioc():
    b = LimitOrderBook()
    b.submit(2, 1, BUY, 98, 2)
    b.submit(2, 2, BUY, 97, 2)
    assert b.submit(1, 3, SELL, 0, 3, market=True)[0] is False  # GTC market rejected
    ok, fills, leaves = b.submit(1, 4, SELL, 0, 10, market=True, ioc=True)
    assert ok and [f[2] for f in fills] == [98, 97] and leaves == 6
    assert b.best_bid() is None


def test_post_only_rejects_when_crossing():
    b = LimitOrderBook()
    b.submit(2, 1, SELL, 100, 1)
    assert b.submit(1, 2, BUY, 100, 1, post_only=True)[0] is False
    assert b.submit(1, 3, BUY, 99, 1, post_only=True)[0] is True


def test_cancel_and_lazy_queue_cleanup():
    b = LimitOrderBook()
    b.submit(2, 1, SELL, 100, 1)
    b.submit(3, 2, SELL, 100, 1)
    b.submit(4, 3, SELL, 100, 1)
    assert b.cancel(3, 2) == 1
    assert b.cancel(3, 2) == 0  # already gone
    _, fills, _ = b.submit(1, 9, BUY, 100, 2)
    assert [f[1] for f in fills] == [1, 3]  # cancelled order in the middle is skipped
    assert b.best_ask() is None
    b.submit(5, 10, SELL, 100, 1)  # level re-created after being emptied
    assert b.best_ask() == 100 and b.qty_at(SELL, 100) == 1


def test_duplicate_live_id_is_dropped():
    b = LimitOrderBook()
    assert b.submit(1, 1, BUY, 99, 1)[0]
    assert not b.submit(1, 1, BUY, 98, 1)[0]
    assert b.submit(2, 1, BUY, 98, 1)[0]


def test_tape_hash_matches_reference_formula():
    b = LimitOrderBook()
    b.submit(1, 1, SELL, 100, 5)
    b.submit(2, 2, BUY, 100, 3)
    assert b.tape_hash == mix_trade(HASH_SEED, 100, 3, 1, 2)


def random_flow(seed: int, n: int = 20_000) -> LimitOrderBook:
    rng = random.Random(seed)
    book = LimitOrderBook(tape=TapeWriter())
    live: list[tuple[int, int]] = []
    for cl in range(1, n + 1):
        owner = rng.randint(1, 4)
        r = rng.random()
        if r < 0.3 and live:
            o, c = live.pop(rng.randrange(len(live)))
            book.cancel(o, c, ts=cl)
        elif r < 0.4:
            book.submit(owner, cl, rng.randint(0, 1), 0, rng.randint(1, 12), market=True, ioc=True, ts=cl)
        else:
            side = rng.randint(0, 1)
            price = rng.randint(9990, 10010)
            ok, _, leaves = book.submit(owner, cl, side, price, rng.randint(1, 10), ioc=rng.random() < 0.15,
                                        post_only=rng.random() < 0.2, ts=cl)
            if ok and leaves and book.leaves(owner, cl):
                live.append((owner, cl))
    return book


@pytest.mark.skipif(REPLAY is None, reason="C++ engine not built")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cpp_engine_reproduces_python_trade_tape(tmp_path, seed):
    book = random_flow(seed)
    assert book.trade_count > 1000
    tape = book.tape.save(tmp_path / f"flow{seed}.mwp")
    proc = subprocess.run([str(REPLAY), str(tape), "--expect-hash", f"0x{book.tape_hash:016x}"],
                          capture_output=True, text=True, timeout=120)
    out = json.loads(proc.stdout)
    assert proc.returncode == 0, out
    assert out["hash_match"] is True
    assert out["trades"] == book.trade_count
    assert out["volume"] == book.volume
    assert out["resting_orders"] == book.resting_orders()
