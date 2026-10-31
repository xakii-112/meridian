"""Performance and risk analytics for market-making sessions.

PnL attribution (ticks x lots, converted to USD with ``tick_value``)
    total      = cash + inventory * mid_T + rebates                 (mark-to-market at the final mid)
    spread     = sum side * (mid_pre - price) * qty                 (edge captured vs the pre-trade mid)
    rebates    = maker_rebate * filled lots
    inventory  = total - spread - rebates                           (mark-to-market drift of the position,
                                                                     where adverse selection shows up)

Markouts: for every fill, side * (mid(t + h) - price) per lot, volume-weighted, for horizons h. A
curve that decays from the spread captured at h=0 measures how much of the edge is given back to
better-informed flow; the same against the fair value isolates true adverse selection.
"""

from __future__ import annotations

import math

import numpy as np

from .market import FILL_COLS, SAMPLE_COLS, SessionResult

HORIZONS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
TRADING_HOURS_PER_YEAR = 252 * 6.5

F = {c: i for i, c in enumerate(FILL_COLS)}
S = {c: i for i, c in enumerate(SAMPLE_COLS)}


def _forward_fill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    valid = ~np.isnan(x)
    if not valid.any():
        return x
    idx = np.where(valid, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: np.argmax(valid)] = x[np.argmax(valid)]
    return out


def markouts(fills: np.ndarray, times: np.ndarray, price: np.ndarray, horizons=HORIZONS) -> dict[float, float]:
    """Volume-weighted markout per lot at each horizon (ticks). Fills too close to the end are dropped."""
    out: dict[float, float] = {}
    if len(fills) == 0:
        return {h: math.nan for h in horizons}
    t, side, px, qty = fills[:, F["t"]], fills[:, F["side"]], fills[:, F["price"]], fills[:, F["qty"]]
    for h in horizons:
        keep = t + h <= times[-1]
        if not keep.any():
            out[h] = math.nan
            continue
        idx = np.searchsorted(times, t[keep] + h, side="right") - 1
        future = price[np.clip(idx, 0, len(price) - 1)]
        out[h] = float(np.sum(side[keep] * (future - px[keep]) * qty[keep]) / np.sum(qty[keep]))
    return out


def max_drawdown(path: np.ndarray) -> float:
    if len(path) == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(path) - path))


def session_metrics(res: SessionResult) -> dict:
    tv = res.market["tick_value"]
    max_inv = res.strategy.get("max_inventory", 20)
    s, f = res.samples, res.fills
    times = s[:, S["t"]]
    mid = _forward_fill(s[:, S["mid"]])
    fair = s[:, S["fair"]]
    inv = s[:, S["inventory"]]
    mtm = s[:, S["cash"]] + inv * mid + s[:, S["rebates"]]

    if len(f):
        qty = f[:, F["qty"]]
        spread = float(np.sum(f[:, F["side"]] * (f[:, F["mid_pre"]] - f[:, F["price"]]) * qty))
        lots = float(qty.sum())
    else:
        spread, lots = 0.0, 0.0
    rebates = lots * res.market["maker_rebate"]
    total = float(mtm[-1]) if len(mtm) else 0.0
    mk_mid = markouts(f, times, mid)
    mk_fair = markouts(f, times, fair)
    both = ~np.isnan(s[:, S["mm_bid"]]) & ~np.isnan(s[:, S["mm_ask"]])
    quoted = s[both, S["mm_ask"]] - s[both, S["mm_bid"]]
    hours = (times[-1] - times[0]) / 3600 if len(times) > 1 else 1.0
    return {
        "seed": res.seed,
        "pnl_usd": total * tv,
        "spread_usd": spread * tv,
        "rebates_usd": rebates * tv,
        "inventory_usd": (total - spread - rebates) * tv,
        "max_drawdown_usd": max_drawdown(mtm) * tv,
        "fills": int(len(f)),
        "lots": lots,
        "volume_share": lots / res.market_volume if res.market_volume else 0.0,
        "pnl_per_lot_ticks": total / lots if lots else 0.0,
        "edge_per_lot_ticks": spread / lots if lots else 0.0,
        "markout_mid": mk_mid,
        "markout_fair": mk_fair,
        "inventory_abs_mean": float(np.mean(np.abs(inv))),
        "inventory_std": float(np.std(inv)),
        "inventory_abs_max": float(np.max(np.abs(inv))) if len(inv) else 0.0,
        "time_at_limit": float(np.mean(np.abs(inv) >= max_inv)),
        "quote_uptime": float(np.mean(both)),
        "quoted_spread_ticks": float(np.mean(quoted)) if len(quoted) else math.nan,
        "decisions_per_sec": res.decisions / max(hours * 3600, 1e-9),
        "hours": hours,
    }


def aggregate(sessions: list[dict]) -> dict:
    """Cross-session statistics; Sharpe annualised assuming i.i.d. sessions over a 6.5h x 252 day year."""
    pnl = np.array([m["pnl_usd"] for m in sessions])
    hours = float(np.mean([m["hours"] for m in sessions]))
    n = len(pnl)
    std = float(np.std(pnl, ddof=1)) if n > 1 else math.nan
    mean = float(np.mean(pnl))
    sharpe = mean / std * math.sqrt(TRADING_HOURS_PER_YEAR / hours) if n > 1 and std > 0 else math.nan

    def avg(key):
        return float(np.nanmean([m[key] for m in sessions]))

    def curve(key):
        hs = sessions[0][key].keys()
        return {str(h): float(np.nanmean([m[key][h] for m in sessions])) for h in hs}

    return {
        "sessions": n,
        "hours_per_session": hours,
        "pnl_mean_usd": mean,
        "pnl_std_usd": std,
        "pnl_sem_usd": std / math.sqrt(n) if n > 1 else math.nan,
        "sharpe_annualized": sharpe,
        "win_rate": float(np.mean(pnl > 0)),
        "spread_usd": avg("spread_usd"),
        "rebates_usd": avg("rebates_usd"),
        "inventory_usd": avg("inventory_usd"),
        "max_drawdown_usd": avg("max_drawdown_usd"),
        "fills": avg("fills"),
        "lots": avg("lots"),
        "volume_share": avg("volume_share"),
        "pnl_per_lot_ticks": avg("pnl_per_lot_ticks"),
        "edge_per_lot_ticks": avg("edge_per_lot_ticks"),
        "inventory_abs_mean": avg("inventory_abs_mean"),
        "inventory_std": avg("inventory_std"),
        "time_at_limit": avg("time_at_limit"),
        "quote_uptime": avg("quote_uptime"),
        "quoted_spread_ticks": avg("quoted_spread_ticks"),
        "markout_mid": curve("markout_mid"),
        "markout_fair": curve("markout_fair"),
        "pnl_by_session": [float(x) for x in pnl],
    }
