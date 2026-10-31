"""End-to-end market-making study.

    python -m meridian_research.study [--quick] [--workers N] [--out ../results]

Pipeline
  1. Calibration     probe quotes at random depths -> lambda(delta) = A exp(-kappa delta); realised sigma.
  2. Tuning (IS)     each strategy family is tuned on in-sample seeds only:
                     Symmetric half-spread, Avellaneda-Stoikov gamma, GLFT gamma (risk-aversion frontier).
  3. Evaluation (OOS) tuned strategies on disjoint out-of-sample seeds; reference-price ablation
                     (lead vs microprice vs mid) for GLFT.
  4. Latency         GLFT out-of-sample across order-entry latencies (100us .. 1s).
  5. Trace + tape    one session recorded in full; its order flow is written as an MWP tape and
                     replayed through the C++ engine, which must reproduce the identical trade tape.
Outputs ``study.json`` and PNG figures in the output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import strategies as st
from .analytics import S, aggregate, session_metrics
from .calibration import fill_intensity, fit_exponential, realized_sigma
from .market import FILL_COLS, MakerConfig, MarketConfig, run_session

STRATEGIES = {"Symmetric": st.Symmetric, "AvellanedaStoikov": st.AvellanedaStoikov, "GLFT": st.GLFT, "Probe": st.Probe}
ROOT = Path(__file__).resolve().parents[2]


# ---- jobs (top-level so they pickle into worker processes) ----------------------------------

def job(market: MarketConfig, maker: MakerConfig, strategy: str, kwargs: dict, seed: int, extras=()) -> dict:
    return {"market": asdict(market), "maker": asdict(maker), "strategy": strategy, "kwargs": kwargs,
            "seed": seed, "extras": tuple(extras)}


def run_job(j: dict) -> dict:
    strategy = STRATEGIES[j["strategy"]](**j["kwargs"])
    res = run_session(MarketConfig(**j["market"]), MakerConfig(**j["maker"]), strategy, j["seed"],
                      record_tape="tape" in j["extras"])
    out: dict = {"metrics": session_metrics(res)}
    if "quote_log" in j["extras"]:
        out["quote_log"] = res.quote_log
        out["sigma"] = realized_sigma(res.samples)
    if "inventory_hist" in j["extras"]:
        out["inventory_hist"] = np.histogram(res.samples[:, S["inventory"]], bins=np.arange(-51, 52, 2))[0].tolist()
    if "trace" in j["extras"]:
        out["trace"] = build_trace(res)
    if "tape" in j["extras"]:
        out["tape"] = res.tape.to_bytes()
        out["tape_hash"] = res.tape_hash
        out["trades"] = res.trade_count
        out["volume"] = res.market_volume
    return out


def build_trace(res) -> dict:
    s, f = res.samples, res.fills
    t = s[:, S["t"]]
    mid = s[:, S["mid"]].copy()
    good = ~np.isnan(mid)
    mid[~good] = np.interp(t[~good], t[good], mid[good])
    mtm = (s[:, S["cash"]] + s[:, S["inventory"]] * mid + s[:, S["rebates"]]) * res.market["tick_value"]
    step = int(round(1.0 / res.market["sample_dt"]))
    full = {"t": t[::step], "fair": s[::step, S["fair"]], "mid": mid[::step],
            "inventory": s[::step, S["inventory"]], "pnl": mtm[::step]}
    # Zoom on the largest fair-value jump to show requoting around news.
    jumps = np.abs(np.diff(s[:, S["fair"]]))
    j = int(np.argmax(jumps[: len(jumps) - 400])) if len(jumps) > 800 else 0
    lo, hi = max(0, j - 300), min(len(t), j + 300)
    zoom = {k: s[lo:hi, S[k]] for k in ("t", "fair", "best_bid", "best_ask", "mm_bid", "mm_ask", "inventory")}
    sel = (f[:, FILL_COLS.index("t")] >= t[lo]) & (f[:, FILL_COLS.index("t")] < t[hi - 1]) if len(f) else []
    fills = f[sel][:, [FILL_COLS.index(c) for c in ("t", "side", "price", "qty")]] if len(f) else np.empty((0, 4))
    clean = lambda d: {k: [None if (isinstance(x, float) and math.isnan(x)) else round(float(x), 4) for x in v]
                       for k, v in d.items()}
    return {"full": clean(full), "zoom": clean(zoom), "zoom_fills": fills.round(4).tolist(),
            "jump_ticks": float(jumps[j]) if len(jumps) else 0.0}


def run_all(jobs: list[dict], workers: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_job, jobs, chunksize=1))


def summarize(outs: list[dict]) -> dict:
    return aggregate([o["metrics"] for o in outs])


# ---- pipeline ------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="short sessions and few seeds (smoke run)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    market = MarketConfig(duration_s=900.0 if args.quick else 3600.0)
    maker = MakerConfig()
    n = 4 if args.quick else 16
    in_sample = list(range(1001, 1001 + n))
    out_sample = list(range(1, 1 + n))
    W = args.workers
    log = lambda msg: print(f"[{time.time() - started:6.1f}s] {msg}", flush=True)
    sessions = 0

    # 1. Calibration --------------------------------------------------------------------------
    n_cal = 4 if args.quick else 8
    cal_out = run_all([job(market, maker, "Probe", {"lo": 0.0, "hi": 4.0, "seed": s}, 5000 + s, ["quote_log"])
                       for s in range(n_cal)], W)
    sessions += n_cal
    fi = fill_intensity([o["quote_log"] for o in cal_out], np.arange(0.0, 4.51, 0.25))
    fit = fit_exponential(fi["depth"], fi["intensity"], fi["fills"])
    sigma = float(np.mean([o["sigma"] for o in cal_out]))
    A, kappa = fit["A"], fit["kappa"]
    log(f"calibration: A={A:.3f}/s kappa={kappa:.3f}/tick (R2={fit['r2']:.3f}) sigma={sigma:.3f} ticks/sqrt(s)")

    # 2. In-sample tuning ----------------------------------------------------------------------
    gammas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    half_spreads = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    families = {
        "GLFT": [("GLFT", {"gamma": g, "kappa": kappa, "A": A, "sigma": sigma}, g) for g in gammas],
        # Beyond gamma ~0.5 the rolling-horizon AS skew (gamma sigma^2 tau) exceeds tens of ticks per
        # unit of inventory and the quotes degenerate, so its grid stops there.
        "AvellanedaStoikov": [("AvellanedaStoikov", {"gamma": g, "kappa": kappa, "sigma": sigma, "horizon_s": 30.0}, g)
                              for g in gammas if g <= 0.5],
        "Symmetric": [("Symmetric", {"half_spread": h}, h) for h in half_spreads],
    }
    tuning: dict[str, list] = {}
    best: dict[str, tuple] = {}
    for fam, grid in families.items():
        jobs = [job(market, maker, cls, kw, s) for cls, kw, _ in grid for s in in_sample]
        outs = run_all(jobs, W)
        sessions += len(jobs)
        rows = []
        for i, (cls, kw, param) in enumerate(grid):
            agg = summarize(outs[i * n:(i + 1) * n])
            rows.append({"param": param, **{k: agg[k] for k in ("pnl_mean_usd", "pnl_std_usd", "sharpe_annualized",
                                                              "inventory_abs_mean", "inventory_std", "quoted_spread_ticks",
                                                              "lots", "edge_per_lot_ticks", "pnl_per_lot_ticks",
                                                              "max_drawdown_usd", "time_at_limit")}})
        tuning[fam] = rows
        pick = max(range(len(rows)), key=lambda i: rows[i]["sharpe_annualized"] if math.isfinite(rows[i]["sharpe_annualized"]) else -1e18)
        best[fam] = grid[pick]
        log(f"tuned {fam}: best param={grid[pick][2]} IS Sharpe={rows[pick]['sharpe_annualized']:.1f} "
            f"pnl={rows[pick]['pnl_mean_usd']:.0f}")

    g_star = best["GLFT"][2]
    glft_kw = best["GLFT"][1]

    # 3. Out-of-sample evaluation ---------------------------------------------------------------
    lineup = [
        ("Symmetric", "Symmetric", best["Symmetric"][1]),
        ("Avellaneda-Stoikov", "AvellanedaStoikov", best["AvellanedaStoikov"][1]),
        ("GLFT · mid", "GLFT", {**glft_kw, "reference": "mid"}),
        ("GLFT · microprice", "GLFT", {**glft_kw, "reference": "micro"}),
        ("GLFT · lead", "GLFT", glft_kw),
    ]
    jobs = [job(market, maker, cls, kw, s, ["inventory_hist"]) for _, cls, kw in lineup for s in out_sample]
    outs = run_all(jobs, W)
    sessions += len(jobs)
    comparison = []
    for i, (label, cls, kw) in enumerate(lineup):
        chunk = outs[i * n:(i + 1) * n]
        agg = summarize(chunk)
        agg["inventory_hist"] = np.sum([o["inventory_hist"] for o in chunk], axis=0).tolist()
        comparison.append({"label": label, "strategy": cls, "params": kw, **agg})
        log(f"OOS {label:22s} pnl={agg['pnl_mean_usd']:8.0f} ± {agg['pnl_sem_usd']:5.0f}  Sharpe={agg['sharpe_annualized']:7.1f}")

    # 4. Latency sweep ----------------------------------------------------------------------------
    latencies = [0.0001, 0.001, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    jobs = [job(market, MakerConfig(latency_s=lat), "GLFT", glft_kw, s) for lat in latencies for s in out_sample]
    outs = run_all(jobs, W)
    sessions += len(jobs)
    latency = []
    for i, lat in enumerate(latencies):
        agg = summarize(outs[i * n:(i + 1) * n])
        latency.append({"latency_s": lat, **{k: agg[k] for k in ("pnl_mean_usd", "pnl_sem_usd", "sharpe_annualized",
                                                                  "edge_per_lot_ticks", "pnl_per_lot_ticks", "lots",
                                                                  "markout_fair", "inventory_usd", "spread_usd")}})
    log("latency sweep: " + ", ".join(f"{r['latency_s'] * 1e3:g}ms={r['pnl_mean_usd']:.0f}" for r in latency))

    # 5. Trace + tape + differential replay through the C++ engine --------------------------------
    trace_out = run_job(job(market, maker, "GLFT", glft_kw, 7, ["trace", "tape"]))
    sessions += 1
    tape_path = args.out / "sim_tape.mwp"
    tape_path.write_bytes(trace_out["tape"])
    replay = differential_replay(tape_path, trace_out["tape_hash"])
    replay.update({"python_trades": trace_out["trades"], "python_volume": trace_out["volume"],
                   "frames": (len(trace_out["tape"]) - 40) // 48})
    log(f"differential replay: python hash 0x{trace_out['tape_hash']:016x} engine {replay.get('engine_hash')} "
        f"-> match={replay.get('match')}")

    study = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quick": args.quick,
        "runtime_s": round(time.time() - started, 1),
        "sessions": sessions,
        "seeds": {"in_sample": in_sample, "out_of_sample": out_sample},
        "market": asdict(market),
        "maker": asdict(maker),
        "calibration": {"A": A, "kappa": kappa, "r2": fit["r2"], "sigma": sigma,
                        "depth": fi["depth"].tolist(), "intensity": np.nan_to_num(fi["intensity"]).tolist(),
                        "exposure_s": fi["exposure_s"].tolist(), "fills": fi["fills"].tolist()},
        "glft": {"gamma": g_star, "c1": st.GLFT(**glft_kw).c1, "c2": st.GLFT(**glft_kw).c2,
                 "half_spread_ticks": st.GLFT(**glft_kw).half_spread()},
        "tuning": tuning,
        "comparison": comparison,
        "latency": latency,
        "trace": trace_out["trace"],
        "trace_metrics": {k: v for k, v in trace_out["metrics"].items() if not isinstance(v, dict)},
        "replay": replay,
    }
    (args.out / "study.json").write_text(json.dumps(study, indent=1, default=_json_default))
    try:
        from .figures import render_all
        render_all(study, args.out / "figures")
    except ImportError as e:  # matplotlib is optional
        log(f"figures skipped: {e}")
    log(f"done: {sessions} sessions -> {args.out / 'study.json'}")
    return 0 if replay.get("match", True) else 2


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def differential_replay(tape: Path, expected_hash: int) -> dict:
    exe = next((p for p in (ROOT / "engine" / "build" / "meridian_replay.exe", ROOT / "engine" / "build" / "meridian_replay")
                if p.exists()), None)
    if exe is None:
        return {"skipped": "engine not built (cmake --build engine/build)"}
    proc = subprocess.run([str(exe), str(tape), "--expect-hash", f"0x{expected_hash:016x}"],
                          capture_output=True, text=True, timeout=600)
    data = json.loads(proc.stdout)
    return {"engine_hash": data["tape_hash"], "python_hash": f"0x{expected_hash:016x}", "match": data["hash_match"],
            "engine_trades": data["trades"], "engine_volume": data["volume"], "engine_ms": data["elapsed_ms"],
            "engine_frames_per_sec": data["frames_per_sec"]}


if __name__ == "__main__":
    raise SystemExit(main())
