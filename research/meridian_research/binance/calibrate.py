"""Statistics and model calibration on recorded Binance data.

* market_stats      spread / touch-size distribution, realised volatility, trade flow
* fit_hawkes        maximum-likelihood bivariate exponential Hawkes fit to market-order arrivals
                    (trades sharing an exchange timestamp and side are one taker order)
* lead_lag          cross-correlation of perpetual and spot mid changes on a 50 ms grid, and how
                    well the perp-implied fair value predicts the spot mid
* realized_spreads  what the resting side of every real trade earned after h seconds, overall and
                    split by whether the perp-implied gap pointed with or against it
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def _window(t: np.ndarray, t0_ns: int, start_s: float, end_s: float) -> np.ndarray:
    rel = (t - t0_ns) / 1e9
    return (rel >= start_s) & (rel < end_s)


def step_at(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    k = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(values) - 1)
    return values[k]


def market_stats(p: dict, start_s: float = 0.0, end_s: float = 1e18) -> dict:
    t0 = p["t0_ns"]
    sb = p["sbbo"]
    m = _window(sb[:, 0], t0, start_s, end_s)
    sb = sb[m]
    ts = (sb[:, 0] - t0) / 1e9
    dwell = np.diff(ts, append=ts[-1])
    spread = sb[:, 2] - sb[:, 1]
    mid = 0.5 * (sb[:, 1] + sb[:, 2])
    grid = np.arange(ts[0], ts[-1], 1.0)
    r1 = np.diff(step_at(ts, mid, grid))
    sigma = float(np.std(r1))
    price = float(np.median(mid)) * 0.01
    st = p["strd"]
    st = st[_window(st[:, 0], t0, start_s, end_s)]
    orders = _market_orders(st)
    hours = (ts[-1] - ts[0]) / 3600
    return {
        "hours": hours,
        "price_usd": price,
        "spread_one_tick_time_frac": float(np.sum(dwell[spread == 1]) / np.sum(dwell)),
        "spread_mean_ticks": float(np.sum(dwell * spread) / np.sum(dwell)),
        "touch_qty_median_btc": float(np.median(np.concatenate([sb[:, 3], sb[:, 4]]))),
        "sigma_ticks_per_sqrt_s": sigma,
        "sigma_annualized_pct": sigma * 0.01 / price * math.sqrt(365 * 86400) * 100,
        "range_usd": float((mid.max() - mid.min()) * 0.01),
        "trades_per_sec": len(st) / (hours * 3600),
        "market_orders_per_sec": len(orders) / (hours * 3600),
        "buy_share": float(np.mean(orders[:, 1] > 0)) if len(orders) else math.nan,
        "order_size_btc_quantiles": {q: float(np.quantile(orders[:, 2], float(q))) for q in ("0.5", "0.9", "0.99")}
        if len(orders) else {},
        "volume_btc_per_hour": float(st[:, 2].sum() / hours),
        "volume_usd_per_hour": float(st[:, 2].sum() / hours * price),
    }


def _market_orders(st: np.ndarray) -> np.ndarray:
    """Group trades by (exchange time, taker side): rows of (time_s, side, qty)."""
    if len(st) == 0:
        return np.empty((0, 3))
    key_t, side, qty = st[:, 4], st[:, 3], st[:, 2]
    new = np.ones(len(st), dtype=bool)
    new[1:] = (key_t[1:] != key_t[:-1]) | (side[1:] != side[:-1])
    groups = np.cumsum(new) - 1
    out = np.zeros((groups[-1] + 1, 3))
    out[:, 0] = key_t[new] / 1e3
    out[:, 1] = side[new]
    np.add.at(out[:, 2], groups, qty)
    return out


def _hawkes_nll(log_params: np.ndarray, t: np.ndarray, d: np.ndarray, horizon: float) -> float:
    mu, a_self, a_cross, beta = np.exp(log_params)
    r0 = r1 = 0.0
    last = 0.0
    ll = 0.0
    exp, log = math.exp, math.log
    for ti, di in zip(t.tolist(), d.tolist()):
        decay = exp(-beta * (ti - last))
        r0 *= decay
        r1 *= decay
        last = ti
        if di == 0:
            ll += log(mu + a_self * r0 + a_cross * r1)
            r0 += 1.0
        else:
            ll += log(mu + a_self * r1 + a_cross * r0)
            r1 += 1.0
    comp = 2 * mu * horizon + (a_self + a_cross) / beta * float(np.sum(1.0 - np.exp(-beta * (horizon - t))))
    return -(ll - comp)


def fit_hawkes(p: dict, start_s: float, end_s: float) -> dict:
    st = p["strd"]
    st = st[_window(st[:, 0], p["t0_ns"], start_s, end_s)]
    orders = _market_orders(st)
    t = orders[:, 0] - orders[0, 0]
    d = (orders[:, 1] < 0).astype(np.int64)  # 0 = buy, 1 = sell
    horizon = float(t[-1]) + 1e-3
    rate = len(t) / horizon / 2
    x0 = np.log([rate * 0.3, 0.5, 0.2, 1.0])
    res = minimize(_hawkes_nll, x0, args=(t, d, horizon), method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-3})
    mu, a_self, a_cross, beta = np.exp(res.x)
    poisson_ll = len(t) * math.log(len(t) / horizon / 2) - len(t)  # two homogeneous Poisson streams
    return {"events": int(len(t)), "mu": float(mu), "alpha_self": float(a_self), "alpha_cross": float(a_cross),
            "beta": float(beta), "branching_ratio": float((a_self + a_cross) / beta),
            "half_life_s": float(math.log(2) / beta), "loglik": float(-res.fun), "poisson_loglik": float(poisson_ll),
            "converged": bool(res.success)}


def lead_lag(p: dict, start_s: float, end_s: float, dt: float = 0.05, max_lag_s: float = 2.0,
             basis_halflife_s: float = 30.0) -> dict:
    t0 = p["t0_ns"]
    sb, fb = p["sbbo"], p["fbbo"]
    st, sm = (sb[:, 0] - t0) / 1e9, 0.5 * (sb[:, 1] + sb[:, 2])
    ft, fm = (fb[:, 0] - t0) / 1e9, 0.5 * (fb[:, 1] + fb[:, 2])
    lo, hi = max(start_s, st[0], ft[0]), min(end_s, st[-1], ft[-1])
    grid = np.arange(lo, hi, dt)
    s, f = step_at(st, sm, grid), step_at(ft, fm, grid)
    rs, rf = np.diff(s), np.diff(f)
    lags = np.arange(-int(max_lag_s / dt), int(max_lag_s / dt) + 1)
    corr = []
    for L in lags:  # corr(perp change at t, spot change at t + L): L > 0 means the perp leads
        a, b = (rf[: len(rf) - L], rs[L:]) if L >= 0 else (rf[-L:], rs[: len(rs) + L])
        corr.append(float(np.corrcoef(a, b)[0, 1]))
    # Does the perp-implied fair value (perp mid + slow EWMA basis) predict the spot mid?
    alpha = 1 - math.exp(-dt * math.log(2) / basis_halflife_s)
    basis = np.empty_like(s)
    acc = s[0] - f[0]
    for i, (x, y) in enumerate(zip(s.tolist(), f.tolist())):
        acc += alpha * ((x - y) - acc)
        basis[i] = acc
    gap = f + basis - s
    pred = {}
    for h in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        k = int(round(h / dt))
        fwd = s[k:] - s[:-k]
        g = gap[:-k]
        c = float(np.corrcoef(g, fwd)[0, 1])
        beta = float(np.dot(g, fwd) / np.dot(g, g))
        pred[str(h)] = {"corr": c, "beta": beta}
    peak = int(np.argmax(corr))
    return {"dt_s": dt, "lags_s": (lags * dt).round(4).tolist(), "corr": corr, "peak_lag_s": float(lags[peak] * dt),
            "peak_corr": corr[peak], "contemporaneous_corr": corr[len(lags) // 2], "gap_predicts_spot": pred,
            "basis_mean_usd": float(np.mean(basis) * 0.01), "gap_std_ticks": float(np.std(gap))}


def feed_delays(p: dict) -> dict:
    """Receive time minus exchange trade time, per market (ms). This is network delay plus any
    local clock offset; the difference between the two markets is what shifts a receive-time
    lead-lag estimate."""
    out = {}
    for name, key in (("spot", "strd"), ("perp", "ftrd")):
        a = p[key]
        d = a[:, 0] / 1e6 - a[:, 4]
        out[name] = {"median_ms": float(np.median(d)), "p90_ms": float(np.quantile(d, 0.9)),
                     "p99_ms": float(np.quantile(d, 0.99))}
    out["perp_minus_spot_median_ms"] = out["perp"]["median_ms"] - out["spot"]["median_ms"]
    return out


def perp_gap(p: dict, grid: np.ndarray, basis_halflife_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Spot mid and (perp mid + EWMA basis - spot mid) on a regular grid, in ticks."""
    t0 = p["t0_ns"]
    sb, fb = p["sbbo"], p["fbbo"]
    s = step_at((sb[:, 0] - t0) / 1e9, 0.5 * (sb[:, 1] + sb[:, 2]), grid)
    f = step_at((fb[:, 0] - t0) / 1e9, 0.5 * (fb[:, 1] + fb[:, 2]), grid)
    dt = float(grid[1] - grid[0])
    alpha = 1 - math.exp(-dt * math.log(2) / basis_halflife_s)
    basis = np.empty_like(s)
    acc = s[0] - f[0]
    for i, (x, y) in enumerate(zip(s.tolist(), f.tolist())):
        acc += alpha * ((x - y) - acc)
        basis[i] = acc
    return s, f + basis - s


def realized_spreads(p: dict, start_s: float, end_s: float, horizons=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
                     dt: float = 0.05, cuts: list[float] | None = None) -> dict:
    """Markout of the passive (maker) side of every recorded trade, USD per BTC, volume-weighted.

    The maker of a trade is short if the taker bought. Trades are split into terciles of
    ``maker_sign * gap`` - the perp-implied gap pointing in the maker's favour (top tercile), roughly
    neutral, or against it (bottom tercile) - to measure what the lead signal is worth to a quoter.
    """
    t0 = p["t0_ns"]
    st = p["strd"]
    t = (st[:, 0] - t0) / 1e9
    keep = (t >= start_s) & (t < end_s - max(horizons))
    t, px, qty, maker = t[keep], st[keep, 1], st[keep, 2], -st[keep, 3]
    grid = np.arange(start_s - 1.0, end_s, dt)
    mid, gap = perp_gap(p, grid)
    before = np.clip(np.searchsorted(grid, t, side="left") - 1, 0, len(grid) - 1)  # state just before the trade
    signal = maker * gap[before]
    if cuts is None:  # terciles of this window; pass in-sample cuts to test out of sample
        cuts = np.quantile(signal, [1 / 3, 2 / 3]).tolist()
    groups = {"all": np.ones(len(t), bool), "gap with maker": signal >= cuts[1],
              "neutral": (signal > cuts[0]) & (signal < cuts[1]), "gap against maker": signal <= cuts[0]}
    out = {"trades": int(len(t)), "btc": float(qty.sum()), "horizons": list(horizons), "cuts_ticks": list(cuts),
           "groups": {}}
    for name, m in groups.items():
        row = {}
        if not m.any():
            out["groups"][name] = {"trades": 0, "btc": 0.0,
                                   "markout": {str(h): {"usd_per_btc": math.nan, "se": math.nan} for h in horizons}}
            continue
        for h in horizons:
            k = np.clip(np.searchsorted(grid, t[m] + h, side="right") - 1, 0, len(grid) - 1)
            pnl = maker[m] * (mid[k] - px[m]) * 0.01
            w = qty[m]
            mean = float(np.sum(pnl * w) / np.sum(w))
            # standard error with trades as independent draws (optimistic: trades cluster in time)
            var = float(np.sum(w * (pnl - mean) ** 2) / np.sum(w))
            n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))
            row[str(h)] = {"usd_per_btc": mean, "se": math.sqrt(var / n_eff)}
        out["groups"][name] = {"trades": int(m.sum()), "btc": float(qty[m].sum()), "markout": row}
    return out
