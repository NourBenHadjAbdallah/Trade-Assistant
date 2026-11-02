#!/usr/bin/env python3
"""
time_machine.py - replay a trade plan before risking money.

Two independent answers to "how often does this hit the target before the stop?":
  * Simulation: 10,000 futures rebuilt from the coin's own recent candles (block bootstrap,
    so calm and wild stretches stay together), starting at the entry.
  * History: every past bar where the same side's confluence checklist was at least as good
    as now, replayed with the same stop/target distances measured in ATR.
Both follow the app's plan: stop first when stop and target share a candle, half off at T1
with the stop moved to the entry, the rest to T2, closed at market after MAX_HOLD bars,
fees included. Results are in R.

CLI:
    python time_machine.py --symbol BTCUSDT --style Intraday
"""
import argparse

import numpy as np

import trade_backtest as tb

SIMS = 10_000
MAX_HOLD = 60          # candles of the setup timeframe
BLOCK = 10             # bootstrap block length (keeps volatility clustering)
HISTORY_YEARS = 1.0


def _first(mask):
    """Index of the first True per row, or len when none."""
    has = mask.any(axis=1)
    return np.where(has, mask.argmax(axis=1), mask.shape[1])


def outcomes(side, entry, stop, t1, t2, high, low, close, cost=tb.COST_PER_SIDE):
    """
    Vectorized trade outcomes for many price paths (rows) that all start at `entry`.
    high/low/close: arrays (paths, bars). Returns dict of arrays: kind (0 stop, 1 T1 then BE,
    2 T2, 3 time limit), r (managed result in R after fees), t1_first (bool), stop_first (bool).
    """
    d = 1 if side > 0 else -1
    risk = abs(entry - stop)
    up, dn = (high, low) if d > 0 else (-low, -high)            # mirror shorts into longs
    e, s, a1, a2 = (d * x for x in (entry, stop, t1, t2))
    n = high.shape[1]
    i_stop = _first(dn <= s)
    i_t1 = _first(up >= a1)
    stop_first = i_stop <= i_t1                                  # ties: stop first
    hit_stop = stop_first & (i_stop < n)
    reached_t1 = (~stop_first) & (i_t1 < n)
    after = np.arange(n)[None, :] > i_t1[:, None]
    i_be = _first((dn <= e) & after)
    i_t2 = _first((up >= a2) & after)
    fee = 2 * cost * entry / risk
    last = d * close[:, -1]
    r = np.where(hit_stop, -1.0, (last - e) / risk)              # default: still full size at the time limit
    kind = np.full(len(r), 3)
    kind[hit_stop] = 0
    half = 0.5 * (a1 - e) / risk
    be_first = reached_t1 & (i_be <= i_t2) & (i_be < n)
    t2_first = reached_t1 & (i_t2 < i_be)
    r = np.where(be_first, half, r)
    r = np.where(t2_first, half + 0.5 * (a2 - e) / risk, r)
    still = reached_t1 & ~be_first & ~t2_first
    r = np.where(still, half + 0.5 * (last - e) / risk, r)
    kind[be_first], kind[t2_first] = 1, 2
    return {"kind": kind, "r": r - fee, "t1_first": reached_t1, "stop_first": hit_stop}


def simulate(df, side, entry, stop, t1, t2, sims=SIMS, bars=MAX_HOLD, block=BLOCK, seed=0):
    """Rebuild `sims` futures from df's own candle moves (relative to the previous close)."""
    c_prev = df["close"].shift().values[1:]
    rc = np.log(df["close"].values[1:] / c_prev)
    rh = np.log(df["high"].values[1:] / c_prev)
    rl = np.log(df["low"].values[1:] / c_prev)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(rc) - block, size=(sims, -(-bars // block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)[:, :bars]
    log_close = np.cumsum(rc[idx], axis=1)
    prev = np.c_[np.zeros(sims), log_close[:, :-1]]
    close = entry * np.exp(log_close)
    high = entry * np.exp(prev + rh[idx])
    low = entry * np.exp(prev + rl[idx])
    return outcomes(side, entry, stop, t1, t2, high, low, close)


def history(df, tf, side, stop_atr, t1_atr, t2_atr, min_checks, bars=MAX_HOLD):
    """Replay every past bar where this side's confluence checklist passed >= min_checks."""
    long_f, short_f = tb.confluence_filters(df, tf)
    checks = sum(v.fillna(False).astype(int) for v in (long_f if side > 0 else short_f).values())
    a = tb.atr(df).values
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    events = np.flatnonzero((checks.values >= min_checks) & np.isfinite(a))
    events = events[events < len(df) - bars - 1]
    rows, last_exit = [], -1
    for i in events:
        if i <= last_exit:                                       # one trade at a time
            continue
        d = 1 if side > 0 else -1
        e = c[i]
        out = outcomes(side, e, e - d * stop_atr * a[i], e + d * t1_atr * a[i], e + d * t2_atr * a[i],
                       h[None, i + 1:i + 1 + bars], l[None, i + 1:i + 1 + bars], c[None, i + 1:i + 1 + bars])
        rows.append((float(out["r"][0]), int(out["kind"][0])))
        last_exit = i + 3                                        # avoid counting the same move many times
    return np.array([r for r, _ in rows]), np.array([k for _, k in rows])


def run(plan, df=None, sims=SIMS, years=HISTORY_YEARS):
    """Time Machine for a build_plan() result. df: setup-TF history (downloaded when None)."""
    s = plan.get("setup")
    if not s:
        raise ValueError("This plan has no trade to replay (no direction).")
    side = 1 if plan["direction"] == "LONG" else -1
    tf = plan["tfs"][1]
    if df is None:
        df = tb.fetch_history(plan["symbol"], tf, years)
    if len(df) < 300:
        raise ValueError(f"Only {len(df)} {tf} candles of history: too few to replay.")
    entry, stop, t1, t2 = s["entry"], s["stop"], s["t1"], s["t2"]
    sim = simulate(df.tail(2000), side, entry, stop, t1, t2, sims)
    atr_now = float(tb.atr(df).iloc[-1])
    dist = lambda x: abs(x - entry) / atr_now
    met = s.get("confluence_met") or 0
    hr, hk = history(df, tf, side, dist(stop), dist(t1), dist(t2), max(min(met, 9), 5))
    rr = s["rr1"]
    fee_r = 2 * tb.COST_PER_SIDE * entry / abs(entry - stop)
    res = {
        "symbol": plan["symbol"], "tf": tf, "side": plan["direction"], "sims": sims, "bars": MAX_HOLD,
        "p_t1": float(sim["t1_first"].mean()), "p_stop": float(sim["stop_first"].mean()),
        "p_none": float(1 - sim["t1_first"].mean() - sim["stop_first"].mean()),
        "avg_r": float(sim["r"].mean()), "r_p10": float(np.percentile(sim["r"], 10)),
        "r_p90": float(np.percentile(sim["r"], 90)),
        "breakeven": (1 + fee_r) / (1 + rr) if rr > 0 else 1.0, "fee_r": fee_r,
        "hist_n": int(len(hr)), "hist_win": float((hr > 0).mean()) if len(hr) else None,
        "hist_avg_r": float(hr.mean()) if len(hr) else None, "hist_checks": max(min(met, 9), 5),
    }
    res["verdict"] = verdict(res)
    return res


def verdict(r):
    edge = r["avg_r"]
    hist = r["hist_avg_r"]
    hist_ok = r["hist_n"] >= 30
    if edge >= 0.1 and (not hist_ok or hist > 0):
        v = f"Odds lean your way: {edge:+.2f}R per trade on average in the simulation"
    elif edge <= -0.1:
        v = f"Odds are against you: {edge:+.2f}R per trade on average in the simulation"
    else:
        v = f"Close to a coin flip after fees ({edge:+.2f}R per trade)"
    if hist_ok:
        v += f"; history agrees ({hist:+.2f}R over {r['hist_n']} past setups)." if (hist > 0) == (edge > 0) \
            else f"; history disagrees ({hist:+.2f}R over {r['hist_n']} past setups)."
    else:
        v += f"; only {r['hist_n']} similar past setups, so history can't confirm it."
    if r["fee_r"] >= 0.25:
        v += f" Fees alone cost {r['fee_r']:.2f}R here: the stop is tight for this timeframe."
    return v


def report_text(r):
    L = [f"TIME MACHINE  {r['symbol']} {r['side']} on {r['tf']} | {r['sims']:,} simulated futures, {r['bars']} candles each",
         f"  Target 1 first {r['p_t1']:.0%} | stop first {r['p_stop']:.0%} | neither {r['p_none']:.0%}"
         f"   (break-even needs about {r['breakeven']:.0%} at T1)",
         f"  Average {r['avg_r']:+.2f}R after fees; bad luck (10th pct) {r['r_p10']:+.2f}R, good luck (90th) {r['r_p90']:+.2f}R"]
    if r["hist_n"]:
        L.append(f"  History: {r['hist_n']} past setups with {r['hist_checks']}+ checks: "
                 f"won {r['hist_win']:.0%}, average {r['hist_avg_r']:+.2f}R")
    L.append("  >> " + r["verdict"])
    return "\n".join(L)


def main():
    import trade_ai as ai
    p = argparse.ArgumentParser(description="Replay a trade plan 10,000 times before risking money")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--style", choices=list(ai.STYLES), default="Intraday")
    a = p.parse_args()
    plan = ai.build_plan(ai.binance_provider, a.symbol.upper(), a.style)
    print(plan["verdict"])
    if not plan.get("setup"):
        return
    print(report_text(run(plan)))


if __name__ == "__main__":
    main()
