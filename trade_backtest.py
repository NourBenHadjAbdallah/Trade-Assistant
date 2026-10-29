#!/usr/bin/env python3
"""
trade_backtest.py - test strategies on history and estimate honest probabilities.

Put it in the same folder as trade_ai.py / trade_chat.py. Needs only: requests, pandas, numpy.

What it does
  * Downloads Binance history (paged, cached in ./cache so later runs are fast and offline)
  * 4 strategies: ema_pullback, breakout, mean_reversion, htf_trend
  * Realistic engine: signal on candle close -> entry at NEXT open, fees + slippage,
    stop and target in the same candle = stop first (pessimistic), one trade at a time
  * Results in R multiples (same unit as trade_journal.py)
  * Stats with confidence intervals, Monte Carlo (probability of profit / drawdown / ruin)
  * Anti-overfitting: out-of-sample split, walk-forward, parameter robustness,
    multiple-testing penalty, verdict that says "no edge" when the evidence is weak

Usage
    python trade_backtest.py --symbol ETHUSDT --tf 4h                       # all strategies, default params
    python trade_backtest.py --symbol BTCUSDT --tf 1h --strategy breakout --optimize
    python trade_backtest.py --symbol BTCUSDT --tf 1h --strategy ema_pullback --walkforward
    python trade_backtest.py --demo                                         # synthetic data (sanity check)

From trade_chat.py: see run_for_chat() at the bottom.
"""
import argparse
import os
import time
from itertools import product
from statistics import NormalDist

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
KLINE_URLS = ["https://api.binance.com/api/v3/klines", "https://data-api.binance.vision/api/v3/klines"]
MAX_CANDLES = 250_000          # keeps 1m/5m downloads reasonable
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440, "1w": 10080}
TRADE_COLS = ["entry_idx", "exit_idx", "time", "side", "entry", "stop", "target", "r", "reason"]
COST_PER_SIDE = 0.0015          # 0.10% fee + 0.05% slippage, charged on entry AND exit


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class SymbolError(ValueError):
    pass


def _get_klines(params):
    import requests
    last = None
    for url in KLINE_URLS:
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 400:
                raise SymbolError(f"Binance does not list '{params['symbol']}'.")
            r.raise_for_status()
            return r.json()
        except SymbolError:
            raise
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not reach Binance. Check your internet connection. ({last})")


def fetch_history(symbol, tf, years=2.0, use_cache=True):
    """Page through Binance klines. Cached as CSV; later calls only download the new candles."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.upper()}_{tf}.csv")
    cols = ["time", "open", "high", "low", "close", "volume", "ct", "qv", "trades", "buy_volume", "tbq", "ig"]
    old = None
    if use_cache and os.path.isfile(path):
        old = pd.read_csv(path, parse_dates=["time"])
    ms_bar = TF_MIN[tf] * 60_000
    now_ms = int(time.time() * 1000)
    start = int((old["time"].iloc[-1].timestamp() * 1000) + ms_bar) if old is not None and len(old) \
        else now_ms - int(years * 365 * 86_400_000)
    rows = []
    while start < now_ms:
        batch = _get_klines({"symbol": symbol.upper(), "interval": tf, "startTime": start, "limit": 1000})
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + ms_bar
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    new = pd.DataFrame(rows, columns=cols) if rows else None
    if new is not None:
        for c in ["open", "high", "low", "close", "volume", "buy_volume"]:
            new[c] = new[c].astype(float)
        new["time"] = pd.to_datetime(new["time"], unit="ms")
        new = new[["time", "open", "high", "low", "close", "volume", "buy_volume"]]
    df = pd.concat([x for x in (old, new) if x is not None], ignore_index=True)
    if df.empty:
        raise RuntimeError("No data returned (check the symbol and your connection).")
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    # keep closed candles only (Binance times are UTC). Never trims rows that were already cached.
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = df[df["time"] + pd.Timedelta(minutes=TF_MIN[tf]) <= now_utc].reset_index(drop=True)
    if new is not None:
        df.to_csv(path, index=False)
    cut = now_utc - pd.Timedelta(days=int(years * 365))
    return df[df["time"] >= cut].reset_index(drop=True)


def demo_history(tf="1h", years=2.0, seed=1):
    """Synthetic regime-switching market with NO built-in edge (a good sanity check: should not look profitable)."""
    rng = np.random.default_rng(seed)
    n = int(years * 365 * 1440 / TF_MIN[tf])
    drift = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.integers(100, 600))
        drift[i:i + L] = rng.choice([-1, 0, 0, 1]) * 0.0003
        i += L
    vol = 0.004 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.2)
    close = 100 * np.exp(np.cumsum(rng.normal(drift - 0.5 * vol ** 2, vol)))   # Ito term: no hidden upward drift
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    v = rng.gamma(3, 100, n)
    t = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq=f"{TF_MIN[tf]}min")
    return pd.DataFrame({"time": t, "open": open_, "high": high, "low": low, "close": close,
                         "volume": v, "buy_volume": v * rng.uniform(0.3, 0.7, n)})


# ----------------------------------------------------------------------------
# Indicators (vectorized)
# ----------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def true_range(df):
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).rolling(n, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------------------
# Strategies: each returns an int array (+1 long, -1 short, 0 none) decided on candle CLOSE.
# Exits are handled by the engine through stop_atr / rr / max_hold.
# ----------------------------------------------------------------------------
def _sig(long, short):
    return (np.where(long.fillna(False), 1, 0) - np.where(short.fillna(False), 1, 0)).astype(int)


def s_ema_pullback(df, tf, fast=20, slow=50, **_):
    c, h, l = df["close"], df["high"], df["low"]
    f, s = ema(c, fast), ema(c, slow)
    up, dn = (f > s) & (c > s), (f < s) & (c < s)
    return _sig(up & (l <= f) & (c > f), dn & (h >= f) & (c < f))


def s_breakout(df, tf, lookback=20, vol_mult=1.5, **_):
    c, v = df["close"], df["volume"]
    hh = df["high"].rolling(lookback).max().shift(1)
    ll = df["low"].rolling(lookback).min().shift(1)
    big = v > vol_mult * v.rolling(20).mean()
    return _sig((c > hh) & big, (c < ll) & big)


def s_mean_reversion(df, tf, rsi_lo=30, adx_max=25, **_):
    c = df["close"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    r, ranging = rsi(c), adx(df) < adx_max
    return _sig((r < rsi_lo) & (c < mid - 2 * sd) & ranging, (r > 100 - rsi_lo) & (c > mid + 2 * sd) & ranging)


def htf_state(df, tf, htf_factor=4, htf_ema=50):
    """(up, dn) bool arrays: higher-timeframe trend as known at the START of each HTF bar (no lookahead)."""
    g = df.set_index("time")
    rule = f"{TF_MIN[tf] * htf_factor}min"
    hc = g["close"].resample(rule).last().dropna()
    e = ema(hc, htf_ema)
    up = ((hc > e) & (e > e.shift(3))).shift(1)
    dn = ((hc < e) & (e < e.shift(3))).shift(1)
    up = up.astype(float).reindex(g.index, method="ffill").fillna(0).astype(bool).values
    dn = dn.astype(float).reindex(g.index, method="ffill").fillna(0).astype(bool).values
    return up, dn


def s_htf_trend(df, tf, htf_factor=4, htf_ema=50, **_):
    """Trade RSI-50 re-crosses only in the direction of the higher timeframe trend (no lookahead)."""
    up, dn = htf_state(df, tf, htf_factor, htf_ema)
    r = rsi(df["close"]).values
    rp = np.r_[np.nan, r[:-1]]
    long = pd.Series(up & (rp < 50) & (r >= 50), index=df.index)
    short = pd.Series(dn & (rp > 50) & (r <= 50), index=df.index)
    return _sig(long, short)


def confluence_filters(df, tf, htf_factor=4, adx_min=20, flow_min=0.0, **_):
    """
    Named bool Series per side: every independent check a 'strong' signal must pass.
    Trend (HTF + EMA stack + ADX), pullback to EMA20 that held, decisive candle,
    order flow (net taker delta), participation (volume), momentum not stretched, sane volatility.
    """
    c, o, h, l, v = (df[k] for k in ("close", "open", "high", "low", "volume"))
    e20, e50 = ema(c, 20), ema(c, 50)
    up, dn = (pd.Series(x, index=df.index) for x in htf_state(df, tf, htf_factor, 50))
    rng = (h - l).replace(0, np.nan)
    clv = ((c - l) / rng).fillna(0.5)
    delta = 2 * df["buy_volume"] - v
    flow = delta.rolling(5).sum() / v.rolling(5).sum().replace(0, np.nan)
    r, a = rsi(c), atr(df)
    trending = adx(df) > adx_min
    participation = v > v.rolling(20).mean()
    sane_vol = a.rolling(200, min_periods=50).rank(pct=True) < 0.9
    long = {"htf_trend": up, "ema_stack": (e20 > e50) & (c > e50), "adx": trending,
            "pullback_held": (l.rolling(3).min() <= e20) & (c > e20),
            "strong_candle": (c > o) & (clv > 0.6), "order_flow": flow > flow_min,
            "volume": participation, "rsi_room": r.between(45, 70), "volatility": sane_vol}
    short = {"htf_trend": dn, "ema_stack": (e20 < e50) & (c < e50), "adx": trending,
             "pullback_held": (h.rolling(3).max() >= e20) & (c < e20),
             "strong_candle": (c < o) & (clv < 0.4), "order_flow": flow < -flow_min,
             "volume": participation, "rsi_room": r.between(30, 55), "volatility": sane_vol}
    return long, short


def s_confluence(df, tf, htf_factor=4, adx_min=20, flow_min=0.0, **_):
    """Only fires when ALL confluence filters agree (see confluence_filters)."""
    long, short = confluence_filters(df, tf, htf_factor, adx_min, flow_min)
    all_ = lambda d: np.logical_and.reduce([x.fillna(False).astype(bool).values for x in d.values()])
    return _sig(pd.Series(all_(long), index=df.index), pd.Series(all_(short), index=df.index))


STRATEGIES = {
    "ema_pullback": {
        "fn": s_ema_pullback, "desc": "Buy dips to the fast EMA inside an uptrend (sell rallies in a downtrend).",
        "default": {"fast": 20, "slow": 50, "stop_atr": 1.5, "rr": 2.0, "max_hold": 60},
        "grid": {"fast": [10, 20, 30], "slow": [50, 100], "stop_atr": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 3.0]}},
    "breakout": {
        "fn": s_breakout, "desc": "Close beyond the N-bar range on strong volume.",
        "default": {"lookback": 20, "vol_mult": 1.5, "stop_atr": 1.5, "rr": 2.0, "max_hold": 60},
        "grid": {"lookback": [10, 20, 40, 55], "vol_mult": [1.0, 1.5, 2.0], "stop_atr": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 3.0]}},
    "mean_reversion": {
        "fn": s_mean_reversion, "desc": "Fade RSI + Bollinger extremes, only when ADX says the market is ranging.",
        "default": {"rsi_lo": 30, "adx_max": 25, "stop_atr": 1.5, "rr": 1.5, "max_hold": 40},
        "grid": {"rsi_lo": [20, 25, 30], "adx_max": [20, 25, 30], "stop_atr": [1.5, 2.0, 2.5], "rr": [1.0, 1.5, 2.0]}},
    "htf_trend": {
        "fn": s_htf_trend, "desc": "RSI-50 re-cross in the direction of the higher-timeframe EMA trend.",
        "default": {"htf_factor": 4, "htf_ema": 50, "stop_atr": 1.5, "rr": 2.0, "max_hold": 60},
        "grid": {"htf_factor": [3, 4, 6], "htf_ema": [30, 50, 100], "stop_atr": [1.5, 2.0], "rr": [1.5, 2.0, 3.0]}},
    "confluence": {
        "fn": s_confluence, "desc": "Strong signals only: HTF trend + ADX + held EMA pullback + order flow + volume all agree.",
        "default": {"htf_factor": 4, "adx_min": 20, "flow_min": 0.0, "stop_atr": 2.0, "rr": 2.0, "max_hold": 60},
        "grid": {"htf_factor": [4, 6], "adx_min": [20, 25], "flow_min": [0.0, 0.05], "stop_atr": [1.5, 2.0, 2.5], "rr": [1.5, 2.0, 3.0]}},
}


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
EXIT_PARAMS = ("stop_atr", "rr", "max_hold")


def backtest(df, strategy, tf, params=None, cost=COST_PER_SIDE, start_idx=0, sig=None, atr_arr=None):
    """
    Returns a DataFrame of trades (entry_idx, exit_idx, time, side, entry, r, reason).
    Signals are computed on the whole `df` (so indicators are warmed up) but only trades
    entering at index >= start_idx are kept: this is how out-of-sample slices are tested.
    sig / atr_arr: precomputed arrays for `df` (the optimizer reuses them across exit-only variants).
    """
    spec = STRATEGIES[strategy]
    p = {**spec["default"], **(params or {})}
    stop_atr, rr, max_hold = p["stop_atr"], p["rr"], int(p["max_hold"])
    if sig is None:
        sig = spec["fn"](df, tf, **p)
    a = atr(df).values if atr_arr is None else atr_arr
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    t = df["time"].values
    n, out, i = len(df), [], max(start_idx - 1, 0)
    while i < n - 2:
        if sig[i] == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        side, e = sig[i], i + 1                           # enter at the NEXT candle's open
        entry, risk = o[e], stop_atr * a[i]
        stop, tgt = entry - side * risk, entry + side * rr * risk
        r, reason, j = None, "time", e
        for j in range(e, min(e + max_hold, n - 1) + 1):
            hit_stop = l[j] <= stop if side > 0 else h[j] >= stop
            hit_tgt = h[j] >= tgt if side > 0 else l[j] <= tgt
            if hit_stop:                                  # stop wins ties (pessimistic)
                # a candle that opens beyond the stop fills at its open (gap), not at the stop
                fill = min(o[j], stop) if side > 0 else max(o[j], stop)
                r, reason = side * (fill - entry) / risk, "stop"
                break
            if hit_tgt:
                r, reason = rr, "target"
                break
        if r is None:
            r = side * (c[j] - entry) / risk
        r -= 2 * cost * entry / risk                      # fees + slippage, both sides, in R
        out.append((e, j, t[e], int(side), float(entry), float(stop), float(tgt), float(r), reason))
        i = j                                             # one position at a time
    return pd.DataFrame(out, columns=["entry_idx", "exit_idx", "time", "side", "entry", "stop", "target", "r", "reason"])


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - m)), float(min(1, c + m))


def stats(r, trials=1):
    """trials > 1 widens the intervals (Bonferroni) to account for how many variants were tried."""
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return None
    z = NormalDist().inv_cdf(1 - 0.025 / max(trials, 1))
    wins, losses = r[r > 0], r[r <= 0]
    lo, hi = wilson(int((r > 0).sum()), n, z)
    sd = r.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    cum = np.cumsum(r)
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    gp, gl = wins.sum(), -losses.sum()
    rng = np.random.default_rng(0)
    p_pos = float((rng.choice(r, (2000, n)).mean(axis=1) > 0).mean()) if n >= 10 else float("nan")
    return {"n": n, "win_rate": float((r > 0).mean()), "win_ci": (lo, hi),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "expectancy": float(r.mean()), "exp_ci": (float(r.mean() - z * se), float(r.mean() + z * se)),
            "total_r": float(cum[-1]), "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
            "max_dd_r": float((np.maximum.accumulate(cum) - cum).max()), "max_losing_streak": int(best),
            "prob_edge_positive": p_pos, "trials": int(trials), "z": float(z)}


def monte_carlo(r, n_trades=100, risk_pct=1.0, sims=5000, ruin_dd=30.0, seed=0):
    """Bootstrap the R results: what could the next n_trades look like at risk_pct per trade?"""
    r = np.asarray(r, float)
    if len(r) < 10:
        return None
    rng = np.random.default_rng(seed)
    paths = rng.choice(r, size=(sims, n_trades), replace=True)
    eq = 100 * np.cumprod(np.maximum(1 + paths * risk_pct / 100, 0), axis=1)
    final = eq[:, -1]
    eq0 = np.column_stack([np.full(sims, 100.0), eq])     # include the starting balance as the first peak
    peak = np.maximum.accumulate(eq0, axis=1)
    dd = ((peak - eq0) / peak).max(axis=1) * 100
    return {"n_trades": n_trades, "risk_pct": risk_pct, "prob_profit": float((final > 100).mean()),
            "median_ret": float(np.median(final) - 100), "p5_ret": float(np.percentile(final, 5) - 100),
            "p95_ret": float(np.percentile(final, 95) - 100), "median_dd": float(np.median(dd)),
            "p95_dd": float(np.percentile(dd, 95)), "prob_ruin": float((dd >= ruin_dd).mean()), "ruin_dd": ruin_dd}


def kelly(s, fraction=0.25, cap=2.0):
    """Fractional Kelly risk % per trade from win rate and payoff. 0 unless the edge is statistically proven."""
    if not s or s["avg_loss"] == 0 or s["avg_win"] <= 0:
        return 0.0
    if s["n"] < 30 or s["expectancy"] <= 0 or s["exp_ci"][0] <= 0:
        return 0.0
    b, p = s["avg_win"] / abs(s["avg_loss"]), s["win_rate"]
    k = p - (1 - p) / b
    if s["n"] < 100:                             # proven but thin sample: keep it small
        cap = min(cap, 0.5)
    return float(min(max(k, 0) * fraction * 100, cap))


def verdict(s, mc, min_trades=100):
    if s is None or s["n"] < 30:
        return "NOT ENOUGH DATA: fewer than 30 trades. Nothing can be concluded."
    lo, hi = s["exp_ci"]
    if s["n"] < min_trades:
        return f"WEAK EVIDENCE: only {s['n']} trades (want {min_trades}+). Treat any result as a hint."
    if lo > 0 and mc and mc["prob_profit"] >= 0.8:
        return "POSSIBLE EDGE: expectancy is positive even at the pessimistic end. Still forward-test before real money."
    if s["expectancy"] > 0:
        return "INCONCLUSIVE: positive on average but the confidence range includes zero (could be luck)."
    return "NO EDGE FOUND: expectancy after costs is not positive. Do not trade this as-is."


# ----------------------------------------------------------------------------
# Validation: out-of-sample, optimisation, walk-forward
# ----------------------------------------------------------------------------
def grid_combos(strategy):
    g = STRATEGIES[strategy]["grid"]
    return [dict(zip(g, v)) for v in product(*g.values())]


def optimize(df, strategy, tf, train_end, min_trades=20, cost=COST_PER_SIDE):
    """Pick params on data[:train_end] only. Score = t-stat of expectancy (rewards consistency, not luck)."""
    best, best_score, rows = None, -1e9, []
    combos = grid_combos(strategy)
    train = df.iloc[:train_end]
    spec, a, sigs = STRATEGIES[strategy], atr(train).values, {}
    for prm in combos:
        # signals only depend on entry params: compute once per entry set, reuse for every stop/target variant
        key = tuple(sorted((k, v) for k, v in prm.items() if k not in EXIT_PARAMS))
        if key not in sigs:
            sigs[key] = spec["fn"](train, tf, **{**spec["default"], **prm})
        tr = backtest(train, strategy, tf, prm, cost, sig=sigs[key], atr_arr=a)
        if len(tr) < min_trades:
            continue
        r = tr["r"].values
        sc = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)) + 1e-9)
        rows.append({**prm, "n": len(r), "exp": r.mean(), "score": sc})
        if sc > best_score:
            best, best_score = prm, sc
    return best, pd.DataFrame(rows), len(combos)


def split_test(df, strategy, tf, frac=0.7, do_opt=False, cost=COST_PER_SIDE):
    cut = int(len(df) * frac)
    prm, table, trials = (None, None, 1)
    if do_opt:
        prm, table, trials = optimize(df, strategy, tf, cut, cost=cost)
        if prm is None:
            return {"params": None, "trials": trials, "oos": None, "train": None, "table": table}
    train = backtest(df.iloc[:cut], strategy, tf, prm, cost)
    test = backtest(df, strategy, tf, prm, cost, start_idx=cut)
    return {"params": prm or STRATEGIES[strategy]["default"], "trials": trials, "train": train,
            "oos": test, "table": table, "cut_time": df["time"].iloc[cut]}


def walk_forward(df, strategy, tf, folds=4, cost=COST_PER_SIDE):
    """Expanding window: optimise on everything before the fold, test on the fold. Returns pooled OOS trades."""
    n = len(df)
    first = int(n * 0.4)
    edges = np.linspace(first, n, folds + 1).astype(int)
    pooled, log = [], []
    for k in range(folds):
        a, b = edges[k], edges[k + 1]
        prm, _, _ = optimize(df, strategy, tf, a, cost=cost)
        if prm is None:
            log.append((df["time"].iloc[a], None, 0, float("nan")))
            continue
        tr = backtest(df.iloc[:b], strategy, tf, prm, cost, start_idx=a)
        pooled.append(tr)
        log.append((df["time"].iloc[a], prm, len(tr), float(tr["r"].mean()) if len(tr) else float("nan")))
    out = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame(columns=["r"])
    return out, log, len(grid_combos(strategy)) * folds


def robustness(table, top=5):
    """Is the best setting an isolated spike? Compare it with the median of all settings tried."""
    if table is None or table.empty:
        return "No parameter table."
    t = table.sort_values("score", ascending=False)
    pos = float((t["exp"] > 0).mean())
    return (f"{pos:.0%} of {len(t)} tested settings had positive expectancy on the training data. "
            f"Best {t['exp'].iloc[0]:+.2f}R, median {t['exp'].median():+.2f}R. "
            + ("Edge looks broad." if pos >= 0.6 else "Edge is narrow: the best setting is probably curve-fit."))


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt_stats(s, mc, kel=None, title=""):
    if s is None:
        return f"{title}\n  No trades."
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    L = [title] if title else []
    L.append(f"  Trades {s['n']} | win rate {s['win_rate']:.0%} (95% range {s['win_ci'][0]:.0%}-{s['win_ci'][1]:.0%})"
             + (f" [widened for {s['trials']} variants tried]" if s["trials"] > 1 else ""))
    L.append(f"  Expectancy {s['expectancy']:+.2f}R per trade (range {s['exp_ci'][0]:+.2f} to {s['exp_ci'][1]:+.2f}) "
             f"| profit factor {pf}")
    L.append(f"  Avg win {s['avg_win']:+.2f}R | avg loss {s['avg_loss']:+.2f}R | total {s['total_r']:+.1f}R "
             f"| max drawdown {s['max_dd_r']:.1f}R | longest losing streak {s['max_losing_streak']}")
    if s["prob_edge_positive"] == s["prob_edge_positive"]:
        L.append(f"  Chance the true average is above zero (bootstrap): {s['prob_edge_positive']:.0%}")
    if mc:
        L.append(f"  Next {mc['n_trades']} trades at {mc['risk_pct']:g}% risk: {mc['prob_profit']:.0%} chance of profit | "
                 f"median {mc['median_ret']:+.1f}% | bad-luck (5th pct) {mc['p5_ret']:+.1f}% | good-luck (95th) {mc['p95_ret']:+.1f}%")
        L.append(f"  Drawdown: median {mc['median_dd']:.0f}%, 95th pct {mc['p95_dd']:.0f}% | "
                 f"chance of a {mc['ruin_dd']:.0f}%+ drawdown: {mc['prob_ruin']:.0%}")
    if kel is not None:
        L.append(f"  Suggested risk (quarter-Kelly, capped 2%; 0 unless the edge is proven): {kel:.2f}% per trade" + ("  (no edge: 0)" if kel == 0 else ""))
    L.append("  >> " + verdict(s, mc))
    return "\n".join(L)


def analyse(r, trials=1, risk_pct=1.0):
    s = stats(r, trials)
    return s, monte_carlo(r, 100, risk_pct), kelly(s) if s else None


def report_default(df, strategy, tf, risk_pct=1.0, cost=COST_PER_SIDE):
    spec = STRATEGIES[strategy]
    cut = int(len(df) * 0.7)
    full = backtest(df, strategy, tf, None, cost)
    oos = backtest(df, strategy, tf, None, cost, start_idx=cut)
    L = [f"=== {strategy}: {spec['desc']}", f"  Params (fixed, not fitted): {spec['default']}"]
    L.append(fmt_stats(*analyse(full["r"].values, 1, risk_pct), title="ALL DATA"))
    L.append(fmt_stats(*analyse(oos["r"].values, 1, risk_pct), title=f"LAST 30% ONLY (from {df['time'].iloc[cut]:%Y-%m-%d})"))
    return "\n".join(L), full, oos


def report_optimize(df, strategy, tf, risk_pct=1.0, cost=COST_PER_SIDE):
    res = split_test(df, strategy, tf, 0.7, True, cost)
    L = [f"=== {strategy} OPTIMIZED on first 70%, judged ONLY on the last 30% (from {res.get('cut_time', '?')})"]
    if res["params"] is None:
        return "\n".join(L + ["  No parameter set produced enough trades on the training data."]), None
    L.append(f"  Chosen params: {res['params']}   ({res['trials']} variants tried)")
    L.append("  " + robustness(res["table"]))
    L.append(fmt_stats(*analyse(res["train"]["r"].values, res["trials"], risk_pct), title="IN-SAMPLE (optimistic, ignore)"))
    # the holdout is evaluated once, so no multiple-testing penalty here (it belongs to the in-sample pick)
    L.append(fmt_stats(*analyse(res["oos"]["r"].values, 1, risk_pct), title="OUT-OF-SAMPLE (this is the number that matters)"))
    return "\n".join(L), res


def report_walkforward(df, strategy, tf, risk_pct=1.0, cost=COST_PER_SIDE):
    pooled, log, _ = walk_forward(df, strategy, tf, 4, cost)
    L = [f"=== {strategy} WALK-FORWARD (re-optimised before every test window)"]
    for t, prm, n, e in log:
        L.append(f"  from {t:%Y-%m-%d}: " + (f"{n} trades, {e:+.2f}R avg  params {prm}" if prm else "no usable params"))
    L.append(fmt_stats(*analyse(pooled["r"].values if len(pooled) else [], 1, risk_pct), title="POOLED OUT-OF-SAMPLE"))
    return "\n".join(L), pooled


def verdict_kind(s, mc):
    """(kind, badge) for the result cards: good / maybe / bad / nodata."""
    if s is None or s["n"] < 30:
        return "nodata", "TOO FEW TRADES"
    if s["exp_ci"][1] < 0:
        return "bad", "LOSES MONEY"
    if s["exp_ci"][0] > 0 and s["n"] >= 100 and mc and mc["prob_profit"] >= 0.8:
        return "good", "POSSIBLE EDGE"
    return "maybe", "UNCLEAR"


def make_card(name, r, trials=1, risk_pct=1.0, has_trades=False):
    s, mc, kel = analyse(r, trials, risk_pct)
    kind, badge = verdict_kind(s, mc)
    lines = []
    if s is None:
        lines.append("No trades were generated.")
    else:
        e = s["expectancy"]
        lines.append(f"Took {s['n']} trades: won {s['win_rate']:.0%}, lost {1 - s['win_rate']:.0%}.")
        lines.append(f"On average each trade {'makes' if e > 0 else 'loses'} {abs(e):.2f}R "
                     f"(= {abs(e) * 100:.0f}% of the money you risk per trade).")
        if mc:
            lines.append(f"Next 100 trades risking {risk_pct:g}% each: {mc['prob_profit']:.0%} chance to finish in profit. "
                         f"Typical result {mc['median_ret']:+.0f}%, bad luck {mc['p5_ret']:+.0f}%, good luck {mc['p95_ret']:+.0f}%.")
        else:
            lines.append("Too few trades to estimate future results.")
        lines.append(verdict(s, mc))
    return {"name": name, "kind": kind, "badge": badge, "lines": lines, "has_trades": bool(has_trades),
            "n": s["n"] if s else 0}


def market_phases(df, tf):
    """Split history into thirds so you can see if a strategy only worked in one period."""
    k = len(df) // 3
    return [(df["time"].iloc[i * k], df["time"].iloc[min((i + 1) * k, len(df)) - 1]) for i in range(3)]


def load_data(symbol, tf, years, demo):
    if demo:
        return demo_history(tf, years)
    return fetch_history(symbol, tf, years)


def run_backtest(symbol="BTCUSDT", tf="1h", strategy="all", mode="default", years=2.0,
                 risk_pct=1.0, demo=False, progress=None):
    """
    Shared entry point for the GUI, chat and CLI.
    mode: "default" | "optimize" | "walkforward"
    Returns {"text": report, "curves": {strategy: (r_array, split_index_or_None)}}.
    The curve is what the equity chart should show: all trades (split = where the last 30% starts)
    for default mode, out-of-sample trades only for optimize / walkforward.
    """
    say = progress or (lambda m: None)
    if tf not in TF_MIN:
        raise ValueError(f"Timeframe '{tf}' is not supported for backtests. Use one of: {', '.join(TF_MIN)}.")
    note = ""
    cap = MAX_CANDLES * TF_MIN[tf] / (365 * 1440)
    if years > cap:
        note, years = f" (history limited to {cap:.2f} years to keep the download reasonable)", cap
    say(f"Loading {symbol} {tf} history...")
    df = load_data(symbol, tf, years, demo)
    if len(df) < 500:
        raise ValueError(f"Only {len(df)} candles available: too few to test. Use a higher timeframe or more years.")
    names = list(STRATEGIES) if strategy in (None, "all") else [strategy]
    parts = [f"BACKTEST {symbol} {tf} | {len(df)} candles {df['time'].iloc[0]:%Y-%m-%d} to {df['time'].iloc[-1]:%Y-%m-%d}{note} | "
             f"costs {COST_PER_SIDE * 200:.2f}% round trip | mode {mode} | results in R"]
    curves, trades, cards, cut = {}, {}, [], int(len(df) * 0.7)
    for nme in names:
        say(f"Testing {nme} ({mode})... this can take a minute" if mode != "default" else f"Testing {nme}...")
        if mode == "optimize":
            txt, res = report_optimize(df, nme, tf, risk_pct)
            tdf = res["oos"] if res and res.get("oos") is not None else pd.DataFrame(columns=TRADE_COLS)
            r, split, trials = tdf["r"].values, None, 1
        elif mode == "walkforward":
            txt, pooled = report_walkforward(df, nme, tf, risk_pct)
            tdf = pooled if len(pooled) else pd.DataFrame(columns=TRADE_COLS)
            r, split, trials = tdf["r"].values, None, 1
        else:
            txt, full, _ = report_default(df, nme, tf, risk_pct)
            tdf, r, split, trials = full, full["r"].values, int((full["entry_idx"] < cut).sum()), 1
        curves[nme] = (np.asarray(r, float), split)
        trades[nme] = tdf
        cards.append(make_card(nme, r, trials, risk_pct, len(tdf) > 0))
        parts.append(txt)
    parts.append("Past results estimate, they do not predict. Probabilities assume the future resembles the tested period.")
    return {"text": "\n\n".join(parts), "curves": curves, "trades": trades, "cards": cards, "df": df,
            "symbol": symbol, "tf": tf}


def run_for_chat(arg_string="", demo=False, risk_pct=1.0):
    """
    Plain-text result for trade_chat.py. Example: run_for_chat("ETHUSDT 4h ema_pullback")
    Tokens: SYMBOL, timeframe (1h...), strategy name, 'optimize', 'walkforward'. Order does not matter.
    """
    symbol, tf, strat, mode = "BTCUSDT", "1h", None, "default"
    for tok in arg_string.split():
        low = tok.lower()
        if low in TF_MIN:
            tf = low
        elif low in STRATEGIES:
            strat = low
        elif low in ("optimize", "optimise"):
            mode = "optimize"
        elif low in ("walkforward", "wf"):
            mode = "walkforward"
        else:
            symbol = tok.upper()
    return run_backtest(symbol, tf, strat, mode, 2.0, risk_pct, demo)["text"]


def main():
    p = argparse.ArgumentParser(description="Backtest strategies with honest statistics")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--tf", default="1h", choices=list(TF_MIN))
    p.add_argument("--strategy", default="all", choices=["all"] + list(STRATEGIES))
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--risk", type=float, default=1.0, help="risk per trade %% for the Monte Carlo")
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--demo", action="store_true", help="synthetic data with no edge (sanity check)")
    a = p.parse_args()
    toks = f"{a.symbol} {a.tf}" + ("" if a.strategy == "all" else f" {a.strategy}") \
           + (" optimize" if a.optimize else "") + (" walkforward" if a.walkforward else "")
    print(run_for_chat(toks, a.demo, a.risk))


if __name__ == "__main__":
    main()