"""Static figures for the study (PNG, for the README); the interactive versions live in the dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
SERIES = ["#2f6fdf", "#d9534f", "#2a9d8f", "#e9a23b", "#7c4dbd"]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def render_all(study: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    cal = study["calibration"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    d, lam, fills = np.array(cal["depth"]), np.array(cal["intensity"]), np.array(cal["fills"])
    ok = (lam > 0) & (fills >= 20)
    ax.scatter(d[ok], lam[ok], s=np.clip(fills[ok] / 20, 8, 120), color=SERIES[0], alpha=0.8, label="empirical")
    xs = np.linspace(0, d.max(), 100)
    ax.plot(xs, cal["A"] * np.exp(-cal["kappa"] * xs), color=SERIES[1],
            label=f"A·e^(-κδ): A={cal['A']:.2f}, κ={cal['kappa']:.2f}, R²={cal['r2']:.3f}")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style(ax, "Fill intensity vs quote depth", "depth from fair value δ (ticks)", "fills per second per side")
    paths.append(_save(fig, out / "calibration.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (fam, rows), c in zip(study["tuning"].items(), SERIES):
        x = [r["pnl_std_usd"] for r in rows]
        y = [r["pnl_mean_usd"] for r in rows]
        ax.plot(x, y, "o-", color=c, label=fam, markersize=4)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, "Risk / return frontier (in-sample tuning)", "PnL std per session (USD)", "mean PnL per session (USD)")
    paths.append(_save(fig, out / "frontier.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for row, c in zip(study["comparison"], SERIES):
        hs = [float(h) for h in row["markout_fair"]]
        ax.plot(hs, list(row["markout_fair"].values()), "o-", color=c, label=row["label"], markersize=3)
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Markouts vs fair value (out-of-sample)", "horizon (s)", "ticks per lot")
    paths.append(_save(fig, out / "markouts.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    lat = study["latency"]
    x = [r["latency_s"] * 1e3 for r in lat]
    ax.errorbar(x, [r["pnl_mean_usd"] for r in lat], yerr=[r["pnl_sem_usd"] for r in lat], fmt="o-", color=SERIES[0],
                capsize=3)
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, "Cost of latency (GLFT · lead, out-of-sample)", "order-entry latency (ms)", "mean PnL per session (USD)")
    paths.append(_save(fig, out / "latency.png"))

    z = study["trace"]["zoom"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    t = np.array(z["t"], dtype=float)
    arr = lambda k: np.array([np.nan if v is None else v for v in z[k]], dtype=float)
    ax.step(t, arr("best_bid"), where="post", color=MUTED, linewidth=0.8, label="best bid/ask")
    ax.step(t, arr("best_ask"), where="post", color=MUTED, linewidth=0.8)
    ax.step(t, arr("mm_bid"), where="post", color=SERIES[0], linewidth=1.4, label="MM bid")
    ax.step(t, arr("mm_ask"), where="post", color=SERIES[1], linewidth=1.4, label="MM ask")
    ax.plot(t, arr("fair"), color=INK, linewidth=1.0, label="fair value")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _style(ax, f"Quoting through a {study['trace']['jump_ticks']:.1f}-tick jump", "", "price (ticks)")
    ax2.step(t, arr("inventory"), where="post", color=SERIES[2])
    _style(ax2, "", "time (s)", "inventory (lots)")
    paths.append(_save(fig, out / "trace.png"))
    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
