"""Turns a recording into validated, compact arrays for fast replay.

    python -m meridian_research.binance.data <session_dir>

L2 book reconstruction follows Binance's documented procedure: buffer depth diffs, take a REST
snapshot with ``lastUpdateId = L``, drop diffs with ``u <= L``, require the first applied diff to
straddle ``L + 1`` (``U <= L+1 <= u``), then require ``U == previous u + 1`` for every later diff.
A gap marks the book invalid until the next snapshot re-syncs it. Diffs carry absolute level
quantities, so re-applying an overlapping diff is idempotent.

Audit snapshots (taken every few minutes while recording) are rolled forward with the diffs
received after them and compared level-by-level with the live reconstruction.

Output ``events.npz`` holds one array group per stream plus ``seq_kind``/``seq_idx``, the merged
arrival order, so a backtest is a single loop over integers.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

SBBO, FBBO, STRADE, FTRADE, DEPTH, SYNC, GAP = range(7)
NEAR_TOUCH_TICKS = 20_000  # $200: the part of the book a market maker interacts with
KIND_NAMES = ("spot_bbo", "fut_bbo", "spot_trade", "fut_trade", "depth", "sync", "gap")


def chunks(session: Path) -> list[Path]:
    return sorted(session.glob("chunk_*.jsonl.gz"))  # .part files (still being written) are skipped


def records(session: Path):
    for path in chunks(session):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


class DepthBook:
    """Full-depth L2 book in integer ticks with Binance sequence validation."""

    def __init__(self, tick: float) -> None:
        self.inv_tick = 1.0 / tick
        self.side = ({}, {})  # 0: bids, 1: asks   price_ticks -> qty
        self.last_u: int | None = None
        self.synced = False
        self.gaps = 0

    def px(self, s: str) -> int:
        return int(round(float(s) * self.inv_tick))

    def load_snapshot(self, snap: dict) -> None:
        self.side = ({self.px(p): float(q) for p, q in snap["bids"]}, {self.px(p): float(q) for p, q in snap["asks"]})
        self.last_u = snap["lastUpdateId"]
        self.synced = False  # until a diff straddling L + 1 arrives

    def apply(self, d: dict) -> list[tuple[int, int, float]] | None:
        """Applies a diff; returns the (side, price, qty) changes, or None if it was not applied."""
        U, u = d["U"], d["u"]
        if self.last_u is None:
            return None
        if not self.synced:
            if u <= self.last_u:
                return None  # already contained in the snapshot
            if U > self.last_u + 1:
                self.gaps += 1
                self.last_u = None
                return None
            self.synced = True
        elif U != self.last_u + 1:
            self.gaps += 1
            self.synced = False
            self.last_u = None
            return None
        self.last_u = u
        out = []
        for s, key in ((0, "b"), (1, "a")):
            book = self.side[s]
            for p, q in d[key]:
                price, qty = self.px(p), float(q)
                if qty == 0.0:
                    book.pop(price, None)
                else:
                    book[price] = qty
                out.append((s, price, qty))
        return out

    def best(self) -> tuple[int | None, int | None]:
        b, a = self.side
        return (max(b) if b else None, min(a) if a else None)


def audit(snapshot: dict, recent: deque, live: DepthBook, tick: float) -> dict:
    """Roll an audit snapshot forward with the recent diffs and compare it with the live book."""
    ref = DepthBook(tick)
    ref.load_snapshot(snapshot)
    for d in recent:
        ref.apply(d)
    if not ref.synced or ref.last_u != live.last_u:
        return {"comparable": False}
    # Compare inside the price range the snapshot itself covers (it is truncated at SNAPSHOT_LIMIT
    # levels per side; diffs may add deeper levels the snapshot never listed).
    lo = min(ref.px(p) for p, _ in snapshot["bids"]) if snapshot["bids"] else 0
    hi = max(ref.px(p) for p, _ in snapshot["asks"]) if snapshot["asks"] else 0
    bb, ba = live.best()
    mismatched = compared = near = near_bad = 0
    for s in (0, 1):
        prices = {p for p in ref.side[s] if lo <= p <= hi} | {p for p in live.side[s] if lo <= p <= hi}
        for p in prices:
            compared += 1
            close = abs(p - (bb if s == 0 else ba)) <= NEAR_TOUCH_TICKS
            near += close
            if abs(ref.side[s].get(p, 0.0) - live.side[s].get(p, 0.0)) > 1e-12:
                mismatched += 1
                near_bad += close
    return {"comparable": True, "levels": compared, "mismatched": mismatched, "near_levels": near,
            "near_mismatched": near_bad}


class Decoder:
    """Turns raw messages (recorded or live) into engine events, applying the book-sync rules.

    ``feed`` returns a list of ``(kind, t_ns, payload)``:
      SBBO   (bid, ask, bid_qty, ask_qty)       prices in integer ticks
      FBBO   (bid, ask, bid_qty, ask_qty)       perp prices in spot ticks (float: the perp tick is coarser)
      STRADE (price, qty, taker_side, exch_ms)  taker_side +1 buy / -1 sell
      FTRADE (price, qty, taker_side, exch_ms)
      DEPTH  [(side, price, qty), ...]          level changes; qty 0 removes the level
      SYNC   [(side, price, qty), ...]          the whole book after a (re)sync
      GAP    None                               the book can no longer be trusted
    Recording preprocessing and the live paper trader both use this class, so they see the same
    event stream for the same messages.
    """

    def __init__(self, tick: float = 0.01) -> None:
        self.tick = tick
        self.inv = 1.0 / tick
        self.book = DepthBook(tick)
        self.pending: list[dict] = []           # diffs received before the first snapshot
        self.recent: deque = deque(maxlen=400)  # for audits
        self.open_audits: list[tuple[int, dict]] = []  # a REST snapshot can be newer than the last diff
        self.audits: list[dict] = []
        self.resyncs = 0
        self.crossed = 0

    def _sync(self, t: int) -> tuple:
        return (SYNC, t, [(s, p, q) for s in (0, 1) for p, q in self.book.side[s].items()])

    def feed(self, t: int, source: str, stream: str, d) -> list:
        book, inv = self.book, self.inv
        if source == "spot":
            if stream.endswith("@bookTicker"):
                return [(SBBO, t, (round(float(d["b"]) * inv), round(float(d["a"]) * inv), float(d["B"]), float(d["A"])))]
            if stream.endswith("@trade"):
                return [(STRADE, t, (round(float(d["p"]) * inv), float(d["q"]), -1 if d["m"] else 1, d["T"]))]
            if stream.endswith("@depth@100ms"):
                self.recent.append(d)
                if book.last_u is None and not book.synced:
                    self.pending.append(d)
                    return []
                was_synced = book.synced
                changes = book.apply(d)
                if changes is None:  # a sequence gap: the replay must stop trusting the book
                    return [(GAP, t, None)] if was_synced else []
                out = [] if was_synced else [self._sync(t)]  # first diff after a (re)sync: the whole book
                out.append((DEPTH, t, changes))
                for item in [a for a in self.open_audits if book.last_u > a[1]["lastUpdateId"]]:
                    self.open_audits.remove(item)
                    result = audit(item[1], self.recent, book, self.tick)
                    bb, ba = book.best()
                    self.crossed += int(bb is not None and ba is not None and bb >= ba)
                    self.audits.append({"t": item[0], **result})
                return out
            if stream == "snapshot":
                if book.last_u is None:  # initial sync or re-sync after a gap
                    book.load_snapshot(d)
                    self.resyncs += 1
                    for p in self.pending:
                        book.apply(p)
                    self.pending.clear()
                    return [self._sync(t)] if book.synced else []
                if book.synced:
                    self.open_audits.append((t, d))  # audited once the live book has moved past it
                return []
            if stream in ("connected", "disconnected"):
                out = [(GAP, t, None)] if book.synced else []
                book.last_u, book.synced = None, False  # new connection: must re-sync from a snapshot
                self.pending.clear()
                return out
            return []
        if source == "fut":
            if stream.endswith("@bookTicker"):
                return [(FBBO, t, (float(d["b"]) * inv, float(d["a"]) * inv, float(d["B"]), float(d["A"])))]
            if stream.endswith("@trade"):
                return [(FTRADE, t, (float(d["p"]) * inv, float(d["q"]), -1 if d["m"] else 1, d["T"]))]
        return []


def preprocess(session: Path, out: Path | None = None) -> dict:
    meta_path = session / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tick = meta.get("spot_tick_size", 0.01)
    dec = Decoder(tick)
    buf = {k: [] for k in ("sbbo", "fbbo", "strd", "ftrd", "dt", "doff", "dside", "dpx", "dqty", "sync_t",
                           "sync_off", "sync_side", "sync_px", "sync_qty", "gap_t")}
    seq_kind, seq_idx = [], []
    n_records = 0
    first_t = last_t = None
    rows = {SBBO: "sbbo", FBBO: "fbbo", STRADE: "strd", FTRADE: "ftrd"}

    for t, source, stream, d in records(session):
        n_records += 1
        first_t = t if first_t is None else first_t
        last_t = t
        for kind, te, payload in dec.feed(t, source, stream, d):
            if kind in rows:
                key = rows[kind]
                buf[key].append((te, *payload))
                idx = len(buf[key]) - 1
            elif kind == DEPTH:
                buf["dt"].append(te)
                buf["doff"].append(len(buf["dpx"]))
                for s, p, q in payload:
                    buf["dside"].append(s)
                    buf["dpx"].append(p)
                    buf["dqty"].append(q)
                idx = len(buf["dt"]) - 1
            elif kind == SYNC:
                buf["sync_t"].append(te)
                buf["sync_off"].append(len(buf["sync_px"]))
                for s, p, q in payload:
                    buf["sync_side"].append(s)
                    buf["sync_px"].append(p)
                    buf["sync_qty"].append(q)
                idx = len(buf["sync_t"]) - 1
            else:
                buf["gap_t"].append(te)
                idx = len(buf["gap_t"]) - 1
            seq_kind.append(kind)
            seq_idx.append(idx)

    book, audits, resyncs, crossed = dec.book, dec.audits, dec.resyncs, dec.crossed
    arrays = {
        "sbbo": np.asarray(buf["sbbo"], dtype=np.float64).reshape(-1, 5),
        "fbbo": np.asarray(buf["fbbo"], dtype=np.float64).reshape(-1, 5),
        "strd": np.asarray(buf["strd"], dtype=np.float64).reshape(-1, 5),
        "ftrd": np.asarray(buf["ftrd"], dtype=np.float64).reshape(-1, 5),
        "dt": np.asarray(buf["dt"], dtype=np.int64),
        "doff": np.asarray(buf["doff"] + [len(buf["dpx"])], dtype=np.int64),
        "dside": np.asarray(buf["dside"], dtype=np.int8),
        "dpx": np.asarray(buf["dpx"], dtype=np.int64),
        "dqty": np.asarray(buf["dqty"], dtype=np.float64),
        "sync_t": np.asarray(buf["sync_t"], dtype=np.int64),
        "sync_off": np.asarray(buf["sync_off"] + [len(buf["sync_px"])], dtype=np.int64),
        "sync_side": np.asarray(buf["sync_side"], dtype=np.int8),
        "sync_px": np.asarray(buf["sync_px"], dtype=np.int64),
        "sync_qty": np.asarray(buf["sync_qty"], dtype=np.float64),
        "gap_t": np.asarray(buf["gap_t"], dtype=np.int64),
        "seq_kind": np.asarray(seq_kind, dtype=np.int8),
        "seq_idx": np.asarray(seq_idx, dtype=np.int64),
    }
    comparable = [a for a in audits if a.get("comparable")]
    report = {
        "session": session.name,
        "chunks": len(chunks(session)),
        "records": n_records,
        "minutes": (last_t - first_t) / 6e10 if first_t else 0.0,
        "tick": tick,
        "counts": {name: int(np.sum(arrays["seq_kind"] == k)) for k, name in enumerate(KIND_NAMES)},
        "depth_levels": int(len(arrays["dpx"])),
        "sequence_gaps": book.gaps,
        "syncs": resyncs,
        "audits": len(audits),
        "audits_comparable": len(comparable),
        "audit_levels": int(sum(a["levels"] for a in comparable)),
        "audit_mismatched_levels": int(sum(a["mismatched"] for a in comparable)),
        "audit_near_touch_levels": int(sum(a["near_levels"] for a in comparable)),
        "audit_near_touch_mismatched": int(sum(a["near_mismatched"] for a in comparable)),
        "crossed_at_audit": crossed,
    }
    out = out or session / "events.npz"
    np.savez_compressed(out, **arrays)
    (session / "validation.json").write_text(json.dumps(report, indent=1))
    return report


def load(session: Path) -> dict[str, np.ndarray]:
    with np.load(session / "events.npz") as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    print(json.dumps(preprocess(Path(sys.argv[1])), indent=1))
