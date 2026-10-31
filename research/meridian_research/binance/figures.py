"""Static figures for the real-data study (PNG, for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..figures import SERIES, _save, _style  # noqa: E402


def render(study: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    ll = study["lead_lag"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(ll["lags_s"], ll["corr"], width=ll["dt_s"] * 0.8, color=[SERIES[0] if x > 0 else SERIES[1] if x < 0 else SERIES[2]
                                                                     for x in ll["lags_s"]])
    _style(ax, "Perp → spot lead-lag (50 ms changes)", "lag (s): positive = perp moves first", "correlation")
    paths.append(_save(fig, out / "lead_lag.png"))

    fi = study["fill_intensity"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    d, lam, n = np.array(fi["depth"]), np.array(fi["intensity"]), np.array(fi["fills"])
    ok = (lam > 0) & (n >= 10)
    ax.scatter(d[ok], lam[ok], s=np.clip(n[ok], 8, 120), color=SERIES[0], alpha=0.8)
    xs = np.linspace(0, d.max(), 100)
    ax.plot(xs, fi["A"] * np.exp(-fi["kappa"] * xs), color=SERIES[1],
            label=f"A={fi['A']:.3f}/s, κ={fi['kappa']:.4f}/tick, R²={fi['r2']:.2f}")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style(ax, "BTCUSDT fill intensity vs depth (in-sample)", "depth from mid (ticks of $0.01)", "fills / s")
    paths.append(_save(fig, out / "fill_intensity.png"))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for row, c in zip(study["comparison"], SERIES):
        mk = row["markout_usd_per_btc"]
        ax.plot([float(h) for h in mk], list(mk.values()), "o-", color=c, label=row["label"], markersize=3)
    ax.set_xscale("log")
    ax.axhline(0, color="#6e7781", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Markouts after fills (out-of-sample)", "horizon (s)", "USD per BTC")
    paths.append(_save(fig, out / "real_markouts.png"))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [r["label"] for r in study["comparison"]]
    be = [r["breakeven_fee_bps"] for r in study["comparison"]]
    ax.barh(labels, be, color=[SERIES[0] if v > 0 else SERIES[1] for v in be])
    for fee, name in ((1.0, "1 bp"), (10.0, "10 bp")):
        ax.axvline(fee, color="#6e7781", linestyle="--", linewidth=0.8)
        ax.text(fee, len(labels) - 0.4, name, color="#6e7781", fontsize=8)
    _style(ax, "Break-even maker fee (out-of-sample)", "bps of notional", "")
    paths.append(_save(fig, out / "breakeven.png"))
    return paths
