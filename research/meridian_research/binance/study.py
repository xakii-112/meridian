"""Market-making study on a recorded Binance BTCUSDT session.

    python -m meridian_research.binance.study <session_dir> [--out ../results/real] [--workers N]

The recording is split in time: the first half (after a 30 s warm-up) is in-sample, the second half
out-of-sample. Everything that is fitted or tuned - volatility, Hawkes flow, the fill-intensity
curve, GLFT risk aversion, touch-quoter spread and skew - uses the first half only.

  1. Market statistics, Hawkes fit, perp -> spot lead-lag
  2. Fill intensity lambda(delta) from probe quotes at random depths  -> A, kappa
  3. In-sample tuning: GLFT gamma and touch quoter (half-spread x skew), mid and perp-led references
  4. Out-of-sample comparison, latency sweep, queue-model sensitivity, fee break-even
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..calibration import fill_intensity, fit_exponential
from .backtest import FEE_BPS, Config, prepare, run
from .calibrate import feed_delays, fit_hawkes, lead_lag, market_stats, realized_spreads
from .data import chunks, load, preprocess

ROOT = Path(__file__).resolve().parents[3]
WARMUP_S = 30.0
LATENCY_S = 0.2  # round trip from a home connection to Binance; swept separately
_P = None


def _init(session: str) -> None:
    global _P
    _P = prepare(load(Path(session)))


def _job(cfg: dict) -> dict:
    return run(_P, Config(**cfg))


def _job_record(cfg: dict) -> dict:
    return run(_P, Config(**cfg), record=True)


def _clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("quote_log", "fills_arr", "samples_arr")}


def trace_of(r: dict, tick: float = 0.01) -> dict:
    s, f = r["samples_arr"], r["fills_arr"]
    t, mid, lead, q, cash = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    good = ~np.isnan(mid)
    mid = np.interp(t, t[good], mid[good])
    pnl = (cash + q * mid) * tick
    step = 10  # 1 s
    full = {"t": t[::step].round(2).tolist(), "pnl": pnl[::step].round(4).tolist(), "q": q[::step].round(6).tolist(),
            "mid_usd": (mid[::step] * tick).round(2).tolist()}
    # zoom: 90 s around the largest 1-second move of the mid
    moves = np.abs(mid[step:] - mid[:-step])
    c = int(np.argmax(moves[300:-600])) + 300 if len(moves) > 1200 else len(moves) // 2
    lo, hi = max(0, c - 450), min(len(t), c + 450)
    nan = lambda a: [None if not np.isfinite(x) else round(float(x), 2) for x in a]
    zoom = {"t": t[lo:hi].round(2).tolist(), "mid": nan(mid[lo:hi]), "lead": nan(lead[lo:hi]),
            "bid": nan(s[lo:hi, 5]), "ask": nan(s[lo:hi, 6])}
    sel = (f[:, 0] >= t[lo]) & (f[:, 0] <= t[hi - 1]) if len(f) else np.zeros(0, bool)
    return {"full": full, "zoom": zoom, "zoom_fills": f[sel][:, :4].round(4).tolist() if len(f) else [],
            "zoom_move_usd": float(moves[c] * tick) if len(moves) else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "real")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda m: print(f"[{time.time() - started:6.1f}s] {m}", flush=True)

    npz = args.session / "events.npz"
    newest = max(c.stat().st_mtime for c in chunks(args.session))
    if not npz.exists() or npz.stat().st_mtime < newest:
        log("preprocessing ...")
        validation = preprocess(args.session)
    else:
        validation = json.loads((args.session / "validation.json").read_text())
    log(f"validation: {validation['minutes']:.1f} min, gaps={validation['sequence_gaps']}, near-touch audit "
        f"{validation['audit_near_touch_mismatched']}/{validation['audit_near_touch_levels']} mismatched")

    p = prepare(load(args.session))
    D = p["duration_s"]
    split = WARMUP_S + (D - WARMUP_S) / 2
    IS, OOS = (WARMUP_S, split), (split, D)
    stats = {"full": market_stats(p, WARMUP_S, D), "in_sample": market_stats(p, *IS), "out_of_sample": market_stats(p, *OOS)}
    sigma = stats["in_sample"]["sigma_ticks_per_sqrt_s"]
    hawkes = fit_hawkes(p, *IS)
    ll = lead_lag(p, WARMUP_S, D)
    ll_is = lead_lag(p, *IS)
    lead_w = min(1.0, max(0.0, ll_is["gap_predicts_spot"]["1.0"]["beta"]))  # in-sample E[mid in 1 s] weight
    feeds = feed_delays(p)
    rs_is = realized_spreads(p, *IS)
    realized = {"in_sample": rs_is, "out_of_sample": realized_spreads(p, *OOS, cuts=rs_is["cuts_ticks"])}
    log(f"sigma={sigma:.1f} ticks/sqrt(s)  hawkes BR={hawkes['branching_ratio']:.3f}  lead-lag peak "
        f"{ll['peak_lag_s']:+.2f}s  gap->1s corr {ll['gap_predicts_spot']['1.0']['corr']:.3f}  IS lead weight {lead_w:.3f}")
    del p

    base = {"latency_s": LATENCY_S, "start_s": IS[0], "end_s": IS[1], "quote_size": 0.001, "max_units": 10,
            "lead_weight": lead_w}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(args.session),)) as ex:
        runmap = lambda cfgs: list(ex.map(_job, cfgs, chunksize=1))

        # 2. fill intensity
        probes = runmap([{**base, "strategy": "probe", "probe_lo": 0.0, "probe_hi": 300.0, "requote_interval_s": 2.0,
                          "seed": s} for s in range(8)])
        fi = fill_intensity([r["quote_log"] for r in probes], np.arange(0.0, 300.01, 25.0))
        fit = fit_exponential(fi["depth"], fi["intensity"], fi["fills"], min_fills=5)
        A, kappa = fit["A"], fit["kappa"]
        log(f"fill intensity: A={A:.4f}/s kappa={kappa:.5f}/tick R2={fit['r2']:.3f}")

        # 3. in-sample tuning
        gammas = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        halfs = [0.5, 2.0, 5.0, 15.0, 50.0, 150.0]
        skews = [0.0, 20.0, 100.0]
        model = {"kappa": kappa, "A": A, "sigma": sigma}
        grid = ([("GLFT", ref, {**base, "strategy": "glft", "reference": ref, "gamma": g, **model}, g)
                 for ref in ("mid", "lead") for g in gammas] +
                [("Touch", ref, {**base, "strategy": "symmetric", "reference": ref, "half_spread": h, "skew": k},
                  (h, k)) for ref in ("mid", "lead") for h in halfs for k in skews])
        tuned = runmap([g[2] for g in grid])
        tuning, best = [], {}
        for (fam, ref, cfg, param), r in zip(grid, tuned):
            row = {"family": fam, "reference": ref, "param": param, **{k: r[k] for k in (
                "gross_pnl_per_hour", "minute_pnl_ratio", "fills", "volume_btc", "gross_edge_per_btc",
                "breakeven_fee_bps", "inventory_abs_mean_btc", "quoted_spread_ticks", "msgs_per_sec")}}
            tuning.append(row)
            # Rank by the per-minute PnL ratio among settings that actually trade (>= 20 fills);
            # settings that barely trade rank below all of those, by fill count.
            ratio = r["minute_pnl_ratio"] if math.isfinite(r["minute_pnl_ratio"]) else -1e9
            score = (1, ratio) if r["fills"] >= 20 else (0, r["fills"])
            if (fam, ref) not in best or score > best[(fam, ref)][0]:
                best[(fam, ref)] = (score, cfg, param)
        for key, (score, _, param) in sorted(best.items()):
            log(f"tuned {key[0]}/{key[1]}: {param}  IS score {score}")

        # 4. out of sample
        oos = {"start_s": OOS[0], "end_s": OOS[1]}
        g_mid = best[("GLFT", "mid")][1]
        lineup = [("Touch · mid", {**best[("Touch", "mid")][1], **oos}),
                  ("Touch · perp-led", {**best[("Touch", "lead")][1], **oos}),
                  ("GLFT · mid", {**g_mid, **oos}),
                  ("GLFT · microprice", {**g_mid, "reference": "micro", **oos}),
                  ("GLFT · perp-led", {**best[("GLFT", "lead")][1], **oos})]
        results = runmap([c for _, c in lineup])
        comparison = []
        for (label, cfg), r in zip(lineup, results):
            comparison.append({"label": label, **_clean(r)})
            log(f"OOS {label:18s} gross {r['gross_pnl_per_hour']:+8.2f} $/h  fills {r['fills']:5d}  "
                f"edge {r['gross_edge_per_btc']:+7.2f} $/BTC  break-even fee {r['breakeven_fee_bps']:+.3f} bps")
        champion = max(zip(lineup, results), key=lambda x: x[1]["gross_pnl_per_hour"])
        champ_label, champ_cfg = champion[0]

        latencies = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        lat_runs = runmap([{**champ_cfg, "latency_s": L} for L in latencies])
        latency = [{"latency_s": L, **{k: r[k] for k in ("gross_pnl_per_hour", "fills", "gross_edge_per_btc",
                                                         "breakeven_fee_bps", "markout_usd_per_btc")}}
                   for L, r in zip(latencies, lat_runs)]
        log("latency: " + ", ".join(f"{x['latency_s'] * 1e3:g}ms={x['gross_pnl_per_hour']:+.2f}" for x in latency))
        conservative = _clean(runmap([{**champ_cfg, "queue": "conservative"}])[0])
        trace_run = list(ex.map(_job_record, [champ_cfg]))[0]

    study = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - started, 1),
        "session": args.session.name,
        "recording": json.loads((args.session / "meta.json").read_text()) if (args.session / "meta.json").exists() else None,
        "validation": validation,
        "split_s": {"warmup": WARMUP_S, "in_sample": IS, "out_of_sample": OOS},
        "stats": stats,
        "hawkes": hawkes,
        "lead_lag": ll,
        "lead_weight": lead_w,
        "realized_spreads": realized,
        "feed_delays": feeds,
        "fill_intensity": {"A": A, "kappa": kappa, "r2": fit["r2"], "depth": fi["depth"].tolist(),
                           "intensity": np.nan_to_num(fi["intensity"]).tolist(), "fills": fi["fills"].tolist(),
                           "exposure_s": fi["exposure_s"].tolist()},
        "model": model,
        "tuning": tuning,
        "comparison": comparison,
        "champion": champ_label,
        "latency": latency,
        "queue_sensitivity": {"proportional": next(c for c in comparison if c["label"] == champ_label),
                              "conservative": conservative},
        "fees_bps": list(FEE_BPS),
        "trace": trace_of(trace_run),
        "backtest_defaults": asdict(Config()),
    }
    (args.out / "real_study.json").write_text(json.dumps(study, indent=1, default=_json_default))
    try:
        from .figures import render
        render(study, args.out / "figures")
    except ImportError as e:
        log(f"figures skipped: {e}")
    log(f"done -> {args.out / 'real_study.json'}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    raise SystemExit(main())
