"""Live paper trading on Binance's public market data.

The backtest's MarketMaker quotes virtual orders against the live spot BTCUSDT book, with the same
queue model, latency and accounting as the replay study. Nothing is ever sent to the exchange: this
project contains no order-entry code and uses no API keys; data comes from the same unauthenticated
streams as the recorder.

    python -m meridian_research.binance.paper [--pick "GLFT · mid"] [--minutes 15] [--record]
    python -m meridian_research.binance.paper --replay <recorded session> [--verify]

``--record`` saves the raw messages exactly as they are processed. Replaying that recording with
``--replay`` drives an identical MarketMaker with the identical event sequence, so the offline result
equals the live one; ``--verify`` also checks it against the array-based backtest (``backtest.run``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .backtest import FEE_BPS, Config, MarketMaker
from .data import DEPTH, FBBO, FTRADE, SBBO, STRADE, SYNC, Decoder, records

ROOT = Path(__file__).resolve().parents[3]
TICK = 0.01


class PaperSession:
    """Raw messages in, MarketMaker events out - the live counterpart of ``prepare`` + ``run``."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.decoder = Decoder(TICK)
        self.mm = MarketMaker(cfg)
        self.t0_ns: int | None = None
        self.last_fmid: float | None = None
        self.done = False
        self.messages = 0
        self.events = 0

    def on_message(self, t_ns: int, source: str, stream: str, payload) -> None:
        if self.done:
            return
        self.messages += 1
        for kind, te, d in self.decoder.feed(t_ns, source, stream, payload):
            if self.t0_ns is None:
                self.t0_ns = te
            # Same reductions as backtest.prepare: perp trades are unused, and a perp quote update
            # that leaves its mid unchanged carries nothing the market maker reads.
            if kind == FTRADE:
                continue
            if kind == FBBO:
                fm = 0.5 * (d[0] + d[1])
                if fm == self.last_fmid:
                    continue
                self.last_fmid = fm
                pl = fm
            elif kind == SBBO:
                pl = (int(d[0]), int(d[1]), d[2], d[3])
            elif kind == STRADE:
                pl = (int(d[0]), d[1], int(d[2]))
            elif kind in (DEPTH, SYNC):
                pl = tuple(list(col) for col in zip(*d)) if d else ([], [], [])
            else:
                pl = None
            t = (float(te) - float(self.t0_ns)) / 1e9  # same float arithmetic as prepare()
            if t > self.cfg.end_s:
                self.done = True
                return
            self.mm.on_event(t, kind, pl)
            self.events += 1

    def finish(self) -> dict:
        return self.mm.finish(record=True)


def load_config(pick: str | None, study_path: Path) -> tuple[str, dict]:
    if study_path.exists():
        study = json.loads(study_path.read_text())
        label = pick or study["champion"]
        for row in study["comparison"]:
            if row["label"] == label:
                return label, dict(row["config"])
        raise SystemExit(f"no strategy {label!r} in {study_path}; choose from "
                         + ", ".join(r["label"] for r in study["comparison"]))
    return "default GLFT", asdict(Config(strategy="glft", reference="mid"))


def fmt_px(ticks) -> str:
    return "-" if ticks is None else f"{ticks * TICK:,.2f}"


def status_line(sess: PaperSession, fee_bps: float) -> str:
    s = sess.mm.status()
    t = s["t"] or 0.0
    clock = f"{int(t // 60):3d}:{int(t % 60):02d}"
    if t < sess.cfg.start_s or not s["book_ok"] or s["mid"] is None:
        return f"[{clock}] warming up: book {'synced' if s['book_ok'] else 'syncing'}, {sess.messages:,} messages"
    fair = (s["lead"] - s["mid"]) * TICK if s["lead"] is not None else 0.0
    quote = lambda o: "-" if o is None else f"{fmt_px(o[0])} ({o[1]:.3f} ahead)"
    net = s["gross_usd"] - s["notional_usd"] * fee_bps / 1e4
    return (f"[{clock}] mid {fmt_px(s['mid'])}  fair {fair:+.2f}  bid {quote(s['bid'])}  ask {quote(s['ask'])}  "
            f"pos {s['q']:+.3f} BTC  fills {s['fills']}  gross ${s['gross_usd']:+.2f}  net@{fee_bps:g}bp ${net:+.2f}")


def report(label: str, result: dict, fee_bps: float) -> str:
    mk = result["markout_usd_per_btc"]
    lines = [
        f"strategy            {label}",
        f"duration            {result['hours'] * 60:.1f} min of quoting",
        f"fills               {result['fills']}  ({result['volume_btc']:.4f} BTC, ${result['notional_usd']:,.0f} notional)",
        f"gross PnL           ${result['gross_pnl_usd']:+.4f}  (${result['gross_pnl_per_hour']:+.2f}/h)",
        f"net PnL @ {fee_bps:g} bp      ${result['gross_pnl_usd'] - result['notional_usd'] * fee_bps / 1e4:+.4f}",
        f"edge per BTC        ${result['gross_edge_per_btc']:+.2f}" if result["volume_btc"] else "edge per BTC        -",
        f"markout 1 s / 10 s  ${mk['1.0']:+.2f} / ${mk['10.0']:+.2f} per BTC" if result["fills"] else "markout             -",
        f"messages/s          {result['msgs_per_sec']:.2f}   quote uptime {result['quote_uptime'] * 100:.0f}%",
    ]
    return "\n".join(lines)


async def live(args, cfg: Config, label: str, log) -> tuple[dict, Path | None]:
    import aiohttp

    from .recorder import FUT_WS, SPOT_WS, ChunkWriter, periodic_snapshots, stream

    sess = PaperSession(cfg)
    rec_dir = None
    writer = None
    if args.record:
        rec_dir = args.out / ("paper_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        rec_dir.mkdir(parents=True, exist_ok=True)
        writer = ChunkWriter(rec_dir, 600.0)

    def sink(t_ns, source, stream_name, payload):
        if writer is not None and not sess.done:
            writer.write(t_ns, source, stream_name, payload)
        sess.on_message(t_ns, source, stream_name, payload)

    deadline = time.monotonic() + cfg.end_s + 3.0
    started = time.time()

    async def display():
        while time.monotonic() < deadline and not sess.done:
            await asyncio.sleep(args.status_every)
            log(status_line(sess, args.fee_bps))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
        tasks = [asyncio.create_task(c) for c in (
            stream("spot", SPOT_WS, sink, deadline, http, log), stream("fut", FUT_WS, sink, deadline, http, log),
            periodic_snapshots(http, sink, 300.0, deadline), display())]
        while not sess.done and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if writer is not None:
        writer.close()
        (rec_dir / "meta.json").write_text(json.dumps({
            "symbol": "BTCUSDT", "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "ended_utc": datetime.now(timezone.utc).isoformat(), "spot_tick_size": TICK, "messages": writer.lines,
            "counts": writer.counts, "paper": {"label": label, "config": asdict(cfg)}}, indent=1))
    return sess.finish(), rec_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick", help="strategy label from the real-data study (default: its champion)")
    ap.add_argument("--study", type=Path, default=ROOT / "results" / "real" / "real_study.json")
    ap.add_argument("--minutes", type=float, default=15.0, help="quoting time after the warm-up")
    ap.add_argument("--warmup", type=float, default=30.0, help="seconds to sync the book and the basis first")
    ap.add_argument("--latency", type=float, help="override the round-trip latency (s)")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="maker fee used for the net figures")
    ap.add_argument("--status-every", type=float, default=10.0)
    ap.add_argument("--record", action="store_true", help="save the raw messages for an exact offline replay")
    ap.add_argument("--replay", type=Path, help="run on a recorded session instead of the live streams")
    ap.add_argument("--verify", action="store_true", help="with --replay: compare against backtest.run")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "paper")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda m: print(f"{datetime.now().strftime('%H:%M:%S')} {m}", flush=True)

    label, base = load_config(args.pick, args.study)
    if args.replay is not None and (args.replay / "meta.json").exists():
        recorded = json.loads((args.replay / "meta.json").read_text()).get("paper")
        if recorded and not args.pick:  # replaying a paper session: use its own configuration
            label, base = recorded["label"], dict(recorded["config"])
            args.warmup, args.minutes = base["start_s"], (base["end_s"] - base["start_s"]) / 60
    base.update(start_s=args.warmup, end_s=args.warmup + args.minutes * 60.0)
    if args.latency is not None:
        base["latency_s"] = args.latency
    cfg = Config(**base)
    log(f"paper trading {label}: {cfg.strategy}/{cfg.reference}, latency {cfg.latency_s * 1e3:.0f} ms, "
        f"quote {cfg.quote_size} BTC, limit {cfg.max_units * cfg.quote_size} BTC - virtual orders only")

    rec_dir = None
    if args.replay is None:
        result, rec_dir = asyncio.run(live(args, cfg, label, log))
        mode = "live"
    else:
        sess = PaperSession(cfg)
        for t_ns, source, stream_name, payload in records(args.replay):
            sess.on_message(t_ns, source, stream_name, payload)
            if sess.done:
                break
        result = sess.finish()
        mode = "replay"

    verify = None
    if args.verify and args.replay is not None:
        from .backtest import prepare, run
        from .data import load, preprocess
        if not (args.replay / "events.npz").exists():
            preprocess(args.replay)
        ref = run(prepare(load(args.replay)), cfg, record=True)
        keys = ("fills", "volume_btc", "gross_pnl_usd", "msgs_per_sec", "markout_usd_per_btc")
        verify = {"identical": all(_same(ref[k], result[k]) for k in keys)
                  and (ref["fills_arr"] == result["fills_arr"]).all(),
                  "backtest_gross_usd": ref["gross_pnl_usd"], "paper_gross_usd": result["gross_pnl_usd"]}
        log(f"replay vs backtest.run: identical = {verify['identical']}")

    print(report(label, result, args.fee_bps), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"mode": mode, "label": label, "fee_bps": args.fee_bps, "wall_s": round(time.time() - started, 1),
           "recording": str(rec_dir) if rec_dir else (str(args.replay) if args.replay else None),
           "verify": verify,
           "net_pnl_usd": {str(b): result["gross_pnl_usd"] - result["notional_usd"] * b / 1e4 for b in FEE_BPS},
           **{k: v for k, v in result.items() if k not in ("samples_arr", "quote_log")},
           "fills_arr": result["fills_arr"].round(6).tolist()}
    path = args.out / f"paper_{mode}_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"saved {path}")
    return 0 if verify is None or verify["identical"] else 2


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


if __name__ == "__main__":
    raise SystemExit(main())
