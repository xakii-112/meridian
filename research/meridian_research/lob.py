"""Price-time priority limit order book for the research simulator.

Semantics mirror the C++ engine exactly for the order types the simulator uses (GTC/IOC limits,
IOC market orders, post-only, cancels, no self-trade prevention), including the trade-tape hash,
so a simulated session can be replayed through the C++ engine and checked trade-for-trade.

Implementation: per-side dict price -> deque of orders, lazy-deletion heaps for best prices,
and lazy cancellation inside queues (cancelled orders are skipped when they reach the front).
"""

from __future__ import annotations

import heapq
from collections import deque

from .wire import TapeWriter

BUY, SELL = 0, 1
MASK64 = (1 << 64) - 1
HASH_SEED = 14695981039346656037
HASH_PRIME = 1099511628211


def mix_trade(h: int, price: int, qty: int, maker_cl: int, taker_cl: int) -> int:
    """Same 64-bit word hash as meridian::mix_trade."""
    h = ((h ^ (price & MASK64)) * HASH_PRIME) & MASK64
    h = ((h ^ (qty & MASK64)) * HASH_PRIME) & MASK64
    h = ((h ^ (maker_cl & MASK64)) * HASH_PRIME) & MASK64
    return ((h ^ (taker_cl & MASK64)) * HASH_PRIME) & MASK64


class Order:
    __slots__ = ("owner", "cl", "side", "price", "leaves", "alive")

    def __init__(self, owner: int, cl: int, side: int, price: int, leaves: int) -> None:
        self.owner = owner
        self.cl = cl
        self.side = side
        self.price = price
        self.leaves = leaves
        self.alive = True


class LimitOrderBook:
    """Fills are reported as tuples ``(maker_owner, maker_cl, price, qty, maker_leaves)``."""

    def __init__(self, min_price: int = 1, max_price: int = 1 << 20, tape: TapeWriter | None = None) -> None:
        self.min_price = min_price
        self.max_price = max_price
        self.tape = tape
        self._levels: tuple[dict[int, deque], dict[int, deque]] = ({}, {})
        self._qty: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self._heaps: tuple[list[int], list[int]] = ([], [])  # bids stored negated
        self._in_heap: tuple[set[int], set[int]] = (set(), set())
        self._orders: dict[tuple[int, int], Order] = {}
        self.tape_hash = HASH_SEED
        self.trade_count = 0
        self.volume = 0
        self.last_trade: int | None = None

    # ---- queries -----------------------------------------------------------------------------

    def best(self, side: int) -> int | None:
        heap, levels, in_heap = self._heaps[side], self._levels[side], self._in_heap[side]
        while heap:
            p = -heap[0] if side == BUY else heap[0]
            if p in levels:
                return p
            heapq.heappop(heap)
            in_heap.discard(p)
        return None

    def best_bid(self) -> int | None:
        return self.best(BUY)

    def best_ask(self) -> int | None:
        return self.best(SELL)

    def qty_at(self, side: int, price: int) -> int:
        return self._qty[side].get(price, 0)

    def top_excluding(self, side: int, owner: int, cl: int | None) -> tuple[int | None, int]:
        """Best price and size on ``side`` ignoring one order (a market maker's own quote), so the
        reference price a strategy computes is not moved by its own quotes."""
        p = self.best(side)
        if p is None:
            return None, 0
        qty = self._qty[side][p]
        o = self._orders.get((owner, cl)) if cl is not None else None
        if o is None or o.price != p:
            return p, qty
        if qty > o.leaves:
            return p, qty - o.leaves
        others = [x for x in self._qty[side] if x != p]
        if not others:
            return None, 0
        p2 = max(others) if side == BUY else min(others)
        return p2, self._qty[side][p2]

    def leaves(self, owner: int, cl: int) -> int:
        o = self._orders.get((owner, cl))
        return o.leaves if o is not None else 0

    def depth(self, side: int, n: int) -> list[tuple[int, int]]:
        prices = sorted(self._qty[side], reverse=side == BUY)[:n]
        return [(p, self._qty[side][p]) for p in prices]

    def resting_orders(self) -> int:
        return len(self._orders)

    # ---- commands ----------------------------------------------------------------------------

    def submit(self, owner: int, cl: int, side: int, price: int, qty: int, *, ioc: bool = False,
               market: bool = False, post_only: bool = False, ts: int = 0) -> tuple[bool, list, int]:
        """Returns (accepted, fills, leaves_after). Resting happens for GTC remainders."""
        if self.tape is not None:
            self.tape.enter(owner, cl, side, price, qty, ioc, market, post_only, ts)
        if (owner, cl) in self._orders or qty <= 0:
            return False, [], 0
        if market:
            if not ioc or post_only:
                return False, [], 0
            limit = self.max_price if side == BUY else self.min_price
        else:
            if price < self.min_price or price > self.max_price:
                return False, [], 0
            if post_only and ioc:
                return False, [], 0
            limit = price
        if post_only:
            opp = self.best(1 - side)
            if opp is not None and (limit >= opp if side == BUY else limit <= opp):
                return False, [], 0
        fills: list = []
        leaves = self._match(cl, side, limit, qty, fills)
        if leaves > 0 and not ioc:
            self._rest(Order(owner, cl, side, limit, leaves))
        return True, fills, leaves

    def cancel(self, owner: int, cl: int, ts: int = 0) -> int:
        """Returns the cancelled quantity (0 if the order is unknown or already done)."""
        if self.tape is not None:
            self.tape.cancel(owner, cl, ts)
        o = self._orders.pop((owner, cl), None)
        if o is None:
            return 0
        o.alive = False
        qtys = self._qty[o.side]
        remaining = qtys[o.price] - o.leaves
        if remaining == 0:
            del qtys[o.price]
            del self._levels[o.side][o.price]
        else:
            qtys[o.price] = remaining
        return o.leaves

    # ---- internals ---------------------------------------------------------------------------

    def _match(self, taker_cl: int, side: int, limit: int, qty: int, fills: list) -> int:
        opp = 1 - side
        levels, qtys, orders = self._levels[opp], self._qty[opp], self._orders
        h = self.tape_hash
        while qty > 0:
            p = self.best(opp)
            if p is None or (p > limit if side == BUY else p < limit):
                break
            queue = levels[p]
            level_qty = qtys[p]
            while qty > 0:
                m = queue[0]
                if not m.alive:
                    queue.popleft()
                    continue
                f = qty if qty < m.leaves else m.leaves
                m.leaves -= f
                qty -= f
                level_qty -= f
                h = mix_trade(h, p, f, m.cl, taker_cl)
                self.trade_count += 1
                self.volume += f
                fills.append((m.owner, m.cl, p, f, m.leaves))
                if m.leaves == 0:
                    m.alive = False
                    queue.popleft()
                    del orders[(m.owner, m.cl)]
                    if level_qty == 0:
                        break
            self.last_trade = p
            if level_qty == 0:
                del levels[p]
                del qtys[p]
            else:
                qtys[p] = level_qty
        self.tape_hash = h
        return qty

    def _rest(self, o: Order) -> None:
        side, p = o.side, o.price
        levels = self._levels[side]
        queue = levels.get(p)
        if queue is None:
            queue = levels[p] = deque()
            self._qty[side][p] = 0
            if p not in self._in_heap[side]:
                heapq.heappush(self._heaps[side], -p if side == BUY else p)
                self._in_heap[side].add(p)
        queue.append(o)
        self._qty[side][p] += o.leaves
        self._orders[(o.owner, o.cl)] = o
