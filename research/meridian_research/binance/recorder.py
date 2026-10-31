"""Records Binance public market data to gzip JSON-lines chunks.

    python -m meridian_research.binance.recorder --minutes 90 --out ../data/binance

Streams (read-only, unauthenticated):
  spot     btcusdt@depth@100ms   L2 diffs (U..u update ids)  + REST /api/v3/depth snapshots
           btcusdt@trade         every trade (m = buyer is maker)
           btcusdt@bookTicker    every best bid/ask change
  futures  btcusdt@bookTicker    USD-M perpetual best bid/ask (candidate lead signal)
           btcusdt@trade         perpetual trades

Each line is ``[recv_ns, source, stream, payload]`` with the local receive time in ns. A REST
snapshot is taken after every (re)connect - the book reconstruction needs one that overlaps the
buffered diffs - and every ``--snapshot-every`` seconds to audit the reconstruction. Chunks rotate
every ``--chunk`` seconds so closed chunks can be analysed while recording continues.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import aiohttp
import websockets

SYMBOL = "btcusdt"
SPOT_WS = f"wss://stream.binance.com:9443/stream?streams={SYMBOL}@depth@100ms/{SYMBOL}@trade/{SYMBOL}@bookTicker"
# (aggTrade is no longer served on this route; @trade is, and carries the same information per fill)
FUT_WS = f"wss://fstream.binance.com/stream?streams={SYMBOL}@bookTicker/{SYMBOL}@trade"
SPOT_REST = "https://api.binance.com/api/v3"
SNAPSHOT_LIMIT = 5000


class ChunkWriter:
    def __init__(self, root: Path, chunk_s: float) -> None:
        self.root, self.chunk_s = root, chunk_s
        self.index = 0
        self.lines = 0
        self.counts: dict[str, int] = {}
        self._fh = None
        self._opened = 0.0
        self._rotate()

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
            (self.root / f"chunk_{self.index:04d}.jsonl.gz.part").rename(self.root / f"chunk_{self.index:04d}.jsonl.gz")
            self.index += 1
        self._fh = gzip.open(self.root / f"chunk_{self.index:04d}.jsonl.gz.part", "wt", encoding="utf-8",
                             newline="\n", compresslevel=5)
        self._opened = time.monotonic()

    def write(self, recv_ns: int, source: str, stream: str, payload) -> None:
        if time.monotonic() - self._opened >= self.chunk_s:
            self._rotate()
        self._fh.write(json.dumps([recv_ns, source, stream, payload], separators=(",", ":")))
        self._fh.write("\n")
        self.lines += 1
        key = f"{source}:{stream.split('@', 1)[-1]}"
        self.counts[key] = self.counts.get(key, 0) + 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            (self.root / f"chunk_{self.index:04d}.jsonl.gz.part").rename(self.root / f"chunk_{self.index:04d}.jsonl.gz")
            self._fh = None


Sink = Callable[[int, str, str, object], None]  # (recv_ns, source, stream, payload)


async def snapshot(session: aiohttp.ClientSession, sink: Sink, reason: str) -> None:
    async with session.get(f"{SPOT_REST}/depth", params={"symbol": SYMBOL.upper(), "limit": SNAPSHOT_LIMIT}) as r:
        r.raise_for_status()
        data = await r.json()
    data["reason"] = reason
    sink(time.time_ns(), "spot", "snapshot", data)


async def stream(source: str, url: str, sink: Sink, deadline: float, session: aiohttp.ClientSession,
                 log) -> None:
    backoff = 1.0
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=2**23) as ws:
                sink(time.time_ns(), source, "connected", {"url": url})
                log(f"{source}: connected")
                backoff = 1.0
                if source == "spot":  # snapshot after diffs start buffering, as Binance documents
                    asyncio.get_running_loop().call_later(
                        1.5, lambda: asyncio.ensure_future(snapshot(session, sink, "connect")))
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        break
                    msg = json.loads(raw)
                    sink(time.time_ns(), source, msg["stream"], msg["data"])
        except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
            sink(time.time_ns(), source, "disconnected", {"error": repr(e)})
            log(f"{source}: disconnected ({e!r}), retrying in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def periodic_snapshots(session, sink: Sink, every_s: float, deadline: float) -> None:
    while time.monotonic() + every_s < deadline:
        await asyncio.sleep(every_s)
        try:
            await snapshot(session, sink, "audit")
        except aiohttp.ClientError:
            pass


async def main_async(args) -> Path:
    root = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = time.monotonic() + args.minutes * 60
    writer = ChunkWriter(root, args.chunk)
    log = lambda m: print(f"[{time.time() - started:7.1f}s] {m}", flush=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        async with session.get(f"{SPOT_REST}/exchangeInfo", params={"symbol": SYMBOL.upper()}) as r:
            info = (await r.json())["symbols"][0]
        filters = {f["filterType"]: f for f in info["filters"]}

        async def progress():
            while time.monotonic() < deadline:
                await asyncio.sleep(min(60.0, max(0.0, deadline - time.monotonic())))
                log(f"{writer.lines:,} messages, chunk {writer.index}, " +
                    ", ".join(f"{k}={v:,}" for k, v in sorted(writer.counts.items())))

        await asyncio.gather(stream("spot", SPOT_WS, writer.write, deadline, session, log),
                             stream("fut", FUT_WS, writer.write, deadline, session, log),
                             periodic_snapshots(session, writer.write, args.snapshot_every, deadline), progress())
    writer.close()
    meta = {"symbol": SYMBOL.upper(), "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "ended_utc": datetime.now(timezone.utc).isoformat(), "minutes": args.minutes,
            "spot_tick_size": float(filters["PRICE_FILTER"]["tickSize"]),
            "spot_step_size": float(filters["LOT_SIZE"]["stepSize"]), "messages": writer.lines,
            "counts": writer.counts, "streams": {"spot": SPOT_WS, "fut": FUT_WS}}
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done: {writer.lines:,} messages in {writer.index + 1} chunks -> {root}")
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "binance")
    ap.add_argument("--chunk", type=float, default=600.0, help="seconds per chunk file")
    ap.add_argument("--snapshot-every", type=float, default=300.0)
    args = ap.parse_args(argv)
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
