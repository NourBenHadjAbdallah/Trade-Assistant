#!/usr/bin/env python3
"""
trade_ai.py - multi-timeframe strategy engine + AI coach

Sits on top of trade_assistant.py (same folder). Adds:
  * Top-down multi-timeframe analysis (HTF bias -> setup -> trigger)
  * Market regime detection (trend / range / transition) per timeframe
  * Strategy selection WITH reasons (pullback, breakout retest, range rotation, stand aside)
  * Full trade plan: entry zone, stop, 2 targets, R:R, position size from your risk %
  * Confidence score with a visible breakdown (no black box)
  * "What would change my mind" + mistakes to avoid
  * Offline explainer (works without internet/API key)
  * Optional LLM coach (Claude API) that explains the plan and answers questions

CLI:
    python trade_ai.py --symbol BTCUSDT --style Intraday
    python trade_ai.py --demo --style Swing
    python trade_ai.py --symbol ETHUSDT --style Scalp --account 2000 --risk 0.5

LLM coach (optional): set ANTHROPIC_API_KEY (and optionally TRADE_AI_MODEL).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import trade_assistant as ta

DEFAULT_MODEL = os.environ.get("TRADE_AI_MODEL", "claude-sonnet-5-5")

# (higher timeframe, setup timeframe, trigger timeframe)
STYLES = {
    "Scalp":    ["1m", "5m", "15m"],
    "Intraday": ["15m", "1h", "4h"],
    "Swing":    ["1h", "4h", "1d"],
    "Position": ["4h", "1d", "1w"],
}
# STYLES lists are ordered LOW -> HIGH; roles are assigned below
ROLES = ("Trigger", "Setup", "HTF bias")
WEIGHTS = {"Trigger": 0.20, "Setup": 0.35, "HTF bias": 0.45}

GLOSSARY = {
    "poc": "POC (Point of Control): the price with the most traded volume in the window. Price tends to gravitate to it and react at it.",
    "vah": "VAH (Value Area High): upper edge of the zone holding ~70% of volume. Acceptance above it = strength; rejection = fade back into value.",
    "val": "VAL (Value Area Low): lower edge of the ~70% volume zone. Acceptance below = weakness; rejection = bounce back into value.",
    "hvn": "HVN (High Volume Node): a price where lots of volume traded. Acts like a magnet / support-resistance.",
    "lvn": "LVN (Low Volume Node): thin-volume price. Price moves through it fast; breaks there tend to run.",
    "avwap": "Anchored VWAP: the volume-weighted average price since a chosen start (swing high/low, big-volume bar). Shows who is in profit/loss since that event; trend pullbacks often hold it.",
    "vwap": "VWAP: volume-weighted average price. Anchored VWAP starts it from a meaningful event instead of the session open.",
    "delta": "Delta: aggressive buy volume minus aggressive sell volume. Positive = buyers hitting the ask harder than sellers hit the bid.",
    "cvd": "CVD (Cumulative Volume Delta): running total of delta. If price makes a new high but CVD does not, buyers are not really behind the move (divergence).",
    "absorption": "Absorption: heavy one-sided aggressive volume that fails to move price. Passive orders on the other side are soaking it up, often a reversal clue.",
    "atr": "ATR: average true range, the typical candle size. Used here to size stops and measure how far price is from levels.",
    "r multiple": "R multiple: profit or loss measured in units of your initial risk. +2R = you made twice what you risked.",
    "expectancy": "Expectancy: average R per trade (win rate x avg win - loss rate x avg loss). Positive over many trades = an edge.",
    "regime": "Regime: whether the market is trending, ranging, or in transition. Strategies that work in one regime lose in the other.",
    "confluence": "Confluence: several independent levels (POC, AVWAP, HVN...) stacked at the same price, which makes the zone more meaningful.",
}


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    v = (100 - 100 / (1 + rs)).iloc[-1]
    return 50.0 if pd.isna(v) else float(v)


def atr_series(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=3).mean()


def pivots(df, n=3):
    H, L = df["high"].values, df["low"].values
    hi, lo = [], []
    for i in range(n, len(df) - n):
        if H[i] == H[i - n:i + n + 1].max():
            hi.append((i, float(H[i])))
        if L[i] == L[i - n:i + n + 1].min():
            lo.append((i, float(L[i])))
    return hi, lo


def structure(df):
    hi, lo = pivots(df)
    if len(hi) >= 2 and len(lo) >= 2:
        hh, hl = hi[-1][1] > hi[-2][1], lo[-1][1] > lo[-2][1]
        if hh and hl:
            return 1, "higher highs + higher lows"
        if not hh and not hl:
            return -1, "lower highs + lower lows"
        return 0, "mixed swings"
    return 0, "not enough swings"


def _sign(x, thr=0.15):
    return 1 if x > thr else -1 if x < -thr else 0


def _f(x):
    return float(x)


# ----------------------------------------------------------------------------
# Per-timeframe snapshot
# ----------------------------------------------------------------------------
REGIME_TEXT = {
    "trend_up": ("UPTREND", "Favour buying dips; shorts here are countertrend."),
    "trend_down": ("DOWNTREND", "Favour selling rallies; longs here are countertrend."),
    "range": ("RANGE / BALANCE", "Price rotates between value edges: fade the extremes, distrust breakouts without delta."),
    "transition": ("TRANSITION", "No clean trend yet: smaller size or wait for structure."),
}


def snapshot(df, tf):
    vp = ta.volume_profile(df)
    avw = ta.build_avwaps(df)
    of_df, of = ta.order_flow(df)
    res = ta.analyze(df, vp, avw, of, of_df)

    close = df["close"]
    price, a = _f(res["price"]), _f(res["atr"])
    e20, e50 = ema(close, 20), ema(close, 50)
    n = max(5, min(20, len(df) - 1))
    path = close.diff().abs().iloc[-n:].sum()
    er = _f(abs(close.iloc[-1] - close.iloc[-1 - n]) / path) if path > 0 else 0.0   # efficiency ratio

    if price > e20.iloc[-1] > e50.iloc[-1]:
        trend_dir, stance = 1, "price > EMA20 > EMA50"
    elif price < e20.iloc[-1] < e50.iloc[-1]:
        trend_dir, stance = -1, "price < EMA20 < EMA50"
    else:
        trend_dir, stance = 0, "price tangled with the EMAs"
    struct, struct_txt = structure(df)

    if trend_dir != 0 and er >= 0.25:
        regime = "trend_up" if trend_dir > 0 else "trend_down"
    elif er < 0.18:
        regime = "range"
    else:
        regime = "transition"

    vs = atr_series(df).dropna()
    vol_pct = _f((vs.iloc[-100:] <= vs.iloc[-1]).mean()) if len(vs) else 0.5

    direction = 0.4 * trend_dir + 0.4 * float(np.clip(res["score"] / 3.5, -1, 1)) + 0.2 * struct

    snap = {
        "tf": tf, "price": price, "atr": a, "regime": regime, "dir": _f(direction),
        "trend_dir": trend_dir, "stance": stance, "structure": struct_txt, "efficiency": er,
        "rsi": rsi(close), "vol_pct": vol_pct, "delta_bias": _f(of["bias"]),
        "divergence": of["divergence"], "ta_label": ta.label(res["score"]), "ta_score": _f(res["score"]),
        "poc": _f(vp["poc"]), "vah": _f(vp["vah"]), "val": _f(vp["val"]),
        "ema20": _f(e20.iloc[-1]), "price_vs_ema20": 1 if price > e20.iloc[-1] else -1,
        "delta_estimated": bool(df.attrs.get("delta_estimated", False)),
        "meaning": REGIME_TEXT[regime][1],
    }
    snap["_raw"] = {"vp": vp, "res": res}
    return snap


# ----------------------------------------------------------------------------
# Strategy selection + trade plan
# ----------------------------------------------------------------------------
AVOID = {
    "pullback": ["Buying/selling the middle of the move instead of the zone.",
                 "Entering without lower-timeframe delta/AVWAP confirmation.",
                 "Widening the stop when price goes against you."],
    "breakout": ["Chasing the candle that already broke out; wait for the retest of the value edge.",
                 "Trusting a breakout with weak or opposite delta.",
                 "Holding after price accepts back inside value."],
    "range": ["Trading breakouts inside a balance; the edges are for fading.",
              "Fading an edge when higher-timeframe bias is strongly against you.",
              "Targeting beyond the POC/opposite edge; ranges pay modestly."],
    "none": ["Forcing a trade because you want action.",
             "Averaging down in a market with no edge.",
             "Using the lowest timeframe to override the higher-timeframe picture."],
}


def choose_strategy(top, mid, bias):
    a, p, vp = mid["atr"], mid["price"], mid["_raw"]["vp"]
    why = []
    conflict = top["dir"] * mid["dir"] < 0 and abs(top["dir"]) > 0.3 and abs(mid["dir"]) > 0.3
    if conflict:
        why.append(f"HTF ({top['tf']}) leans {'up' if top['dir'] > 0 else 'down'} but setup TF ({mid['tf']}) leans "
                   f"{'up' if mid['dir'] > 0 else 'down'}: timeframes disagree.")
        return 0, "none", "Stand aside (timeframes disagree)", why

    if abs(bias) >= 0.25:
        d = 1 if bias > 0 else -1
        top_ok = top["regime"] == ("trend_up" if d > 0 else "trend_down")
        beyond = (p - vp["vah"]) / a if d > 0 else (vp["val"] - p) / a
        why.append(f"Weighted multi-timeframe bias is {'bullish' if d > 0 else 'bearish'} ({bias:+.2f}).")
        why.append(f"{top['tf']} is {REGIME_TEXT[top['regime']][0].lower()}" +
                   (": clean trend in the trade direction." if top_ok else ": not a clean trend, so conviction is lower."))
        if beyond > 0.2:
            why.append(f"{mid['tf']} price is already outside the value area ({beyond:.1f} ATR): "
                       "entry is the retest of the value edge, not a market chase.")
            return d, "breakout", ("Breakout continuation (buy the retest)" if d > 0
                                   else "Breakdown continuation (sell the retest)"), why
        why.append(f"{mid['tf']} is inside/near value: wait for price to pull back into a confluence zone.")
        return d, "pullback", ("Trend pullback (buy the dip)" if d > 0 else "Trend pullback (sell the rally)"), why

    if mid["regime"] in ("range", "transition"):
        width = max(vp["vah"] - vp["val"], 1e-9)
        pos = (p - vp["val"]) / width
        why.append(f"Bias is neutral ({bias:+.2f}) and {mid['tf']} is {REGIME_TEXT[mid['regime']][0].lower()}: rotation market.")
        why.append(f"Price sits at {pos:.0%} of the value area (0% = VAL, 100% = VAH).")
        if pos <= 0.35:
            return 1, "range", "Range rotation (buy the low edge)", why
        if pos >= 0.65:
            return -1, "range", "Range rotation (sell the high edge)", why
        why.append("Middle of the range: no edge until price reaches an edge.")
        return 0, "none", "No edge (middle of range)", why

    why.append("Mixed signals with no clear regime.")
    return 0, "none", "No edge (mixed signals)", why


def build_levels(d, key, mid):
    a, p = mid["atr"], mid["price"]
    vp, res = mid["_raw"]["vp"], mid["_raw"]["res"]
    clusters = res["clusters"]
    names = []
    if key == "range":
        entry = vp["val"] if d > 0 else vp["vah"]
        stop = entry - d * 0.6 * a
        names = ["VAL" if d > 0 else "VAH"]
    elif key == "breakout":
        entry = vp["vah"] if d > 0 else vp["val"]
        stop = entry - d * 0.8 * a
        names = ["VAH retest" if d > 0 else "VAL retest"]
    else:
        sc = [s for s in res["scenarios"] if s[0].startswith("LONG" if d > 0 else "SHORT")]
        if sc:
            entry, names = float(sc[0][1]["price"]), list(sc[0][1]["names"])
        else:
            entry = p
        stop = entry - d * 0.6 * a
    entry, stop = float(entry), float(stop)
    risk = abs(entry - stop)
    if d > 0:
        cands = sorted(c["price"] for c in clusters if c["price"] > entry + 0.9 * risk)
    else:
        cands = sorted((c["price"] for c in clusters if c["price"] < entry - 0.9 * risk), reverse=True)
    t1 = float(cands[0]) if cands else entry + d * 1.5 * risk
    t2 = float(cands[1]) if len(cands) > 1 else entry + d * 3.0 * risk
    if abs(t2 - t1) < 0.5 * risk:
        t2 = t1 + d * 1.5 * risk

    dist = abs(p - entry) / a
    if dist <= 0.4:
        state = "IN ZONE"
    elif d * (p - entry) > 0:
        state = f"WAIT: price is {dist:.1f} ATR past the entry. Do not chase; let it come back to {entry:.6g}."
    else:
        state = f"WAIT: price is {dist:.1f} ATR on the wrong side of the entry (zone may be failing). Needs a reclaim."
    return {
        "entry": entry, "zone_lo": entry - 0.15 * a, "zone_hi": entry + 0.15 * a, "stop": stop,
        "t1": t1, "t2": t2, "risk": risk, "rr1": abs(t1 - entry) / risk, "rr2": abs(t2 - entry) / risk,
        "state": state, "in_zone": state == "IN ZONE", "zone_names": names,
    }


def check_trigger(d, low):
    side = "above" if d > 0 else "below"
    conds = [
        (f"{low['tf']} delta {'positive' if d > 0 else 'negative'} (last 10 bars): {low['delta_bias']:+.1%}",
         d * low["delta_bias"] > 0.02),
        (f"{low['tf']} price {side} its EMA20", d * low["price_vs_ema20"] > 0),
        (f"No {low['tf']} CVD divergence against the trade",
         low["divergence"] != ("bearish" if d > 0 else "bullish")),
    ]
    return conds


def position_size(account, risk_pct, entry, risk):
    risk_amt = account * risk_pct / 100.0
    units = risk_amt / risk if risk > 0 else 0.0
    notional = units * entry
    return {"account": account, "risk_pct": risk_pct, "risk_amount": risk_amt, "units": units,
            "notional": notional, "leverage_needed": notional / account if account > 0 else 0.0}


def build_plan(provider, symbol, style="Intraday", account=1000.0, risk_pct=1.0):
    """provider(symbol, interval) -> DataFrame (same schema as trade_assistant)."""
    tfs = STYLES[style]
    snaps = []
    for tf, role in zip(tfs, ROLES):
        s = snapshot(provider(symbol, tf), tf)
        s["role"] = role
        snaps.append(s)
    low, mid, top = snaps                      # ordered low -> high
    bias = sum(WEIGHTS[s["role"]] * s["dir"] for s in snaps)
    d, key, strat_name, why = choose_strategy(top, mid, bias)

    plan = {"symbol": symbol, "style": style, "tfs": tfs, "bias": _f(bias),
            "strategy": {"name": strat_name, "key": key, "why": why},
            "direction": "LONG" if d > 0 else "SHORT" if d < 0 else "NONE",
            "setup": None, "confidence": None, "invalidation": [], "avoid": AVOID[key],
            "notes": []}
    if any(s["delta_estimated"] for s in snaps):
        plan["notes"].append("Delta is estimated from candle closes (no aggressor data): order-flow signals are weaker.")

    agree = sum(1 for s in snaps if d != 0 and _sign(s["dir"]) == d)
    plan["alignment"] = f"{agree}/3 timeframes agree" if d != 0 else "n/a"

    if d != 0:
        lv = build_levels(d, key, mid)
        trig = check_trigger(d, low)
        met = sum(1 for _, ok in trig if ok)
        lv["trigger"] = trig
        lv["trigger_met"] = met
        lv["triggered"] = met >= 2
        lv["side"] = plan["direction"]
        lv["size"] = position_size(account, risk_pct, lv["entry"], lv["risk"])
        plan["setup"] = lv

        # confidence with visible breakdown
        factors = [("Base", 10)]
        factors.append((f"Weighted bias strength ({bias:+.2f})", round(abs(bias) * 40)))
        factors.append((f"Alignment ({agree}/3 timeframes)", {3: 20, 2: 10}.get(agree, 0)))
        if lv["rr1"] >= 2:
            factors.append((f"Reward:risk to T1 = {lv['rr1']:.1f}", 15))
        elif lv["rr1"] >= 1.5:
            factors.append((f"Reward:risk to T1 = {lv['rr1']:.1f}", 8))
        else:
            factors.append((f"Weak reward:risk to T1 = {lv['rr1']:.1f}", -10))
        factors.append(("Lower-timeframe trigger " + ("confirmed" if met >= 2 else "not confirmed"),
                        10 if met == 3 else 5 if met == 2 else -5))
        if lv["in_zone"]:
            factors.append(("Price is in the entry zone", 10))
        if top["vol_pct"] > 0.9:
            factors.append((f"{top['tf']} volatility unusually high (percentile {top['vol_pct']:.0%})", -10))
        if (d > 0 and mid["rsi"] > 75) or (d < 0 and mid["rsi"] < 25):
            factors.append((f"{mid['tf']} RSI {mid['rsi']:.0f} is stretched against the entry", -10))
        if key == "range" and top["regime"] in ("trend_up", "trend_down"):
            factors.append((f"Fading a range while {top['tf']} trends", -10))
        if any(s["delta_estimated"] for s in snaps):
            factors.append(("Delta is estimated, not real aggressor data", -5))
        score = int(np.clip(sum(p for _, p in factors), 0, 100))
        lab = "HIGH" if score >= 70 else "MEDIUM" if score >= 45 else "LOW"
        plan["confidence"] = {"score": score, "label": lab, "factors": factors}

        # invalidation
        hi, lo = pivots(provider(symbol, top["tf"]))
        plan["invalidation"].append(f"{mid['tf']} closes beyond the stop {lv['stop']:.6g}: the idea is wrong, exit.")
        if d > 0 and lo and lo[-1][1] < top["price"]:
            plan["invalidation"].append(f"{top['tf']} structure breaks: close below its last swing low {lo[-1][1]:.6g}.")
        if d < 0 and hi and hi[-1][1] > top["price"]:
            plan["invalidation"].append(f"{top['tf']} structure breaks: close above its last swing high {hi[-1][1]:.6g}.")
        plan["invalidation"].append(f"{low['tf']} delta flips against you with CVD divergence at the zone.")

        waiting = []
        if not lv["in_zone"]:
            waiting.append("price to reach the entry zone")
        if not lv["triggered"]:
            waiting.append(f"{low['tf']} confirmation (2 of 3 trigger checks)")
        if score >= 55 and not waiting:
            plan["verdict"], plan["verdict_tag"] = f"ACTIONABLE {plan['direction']}", "long" if d > 0 else "short"
        elif score >= 40:
            plan["verdict"] = f"SETUP FORMING ({plan['direction']}): waiting for " + " and ".join(waiting or ["more confidence"])
            plan["verdict_tag"] = "h"
        else:
            plan["verdict"], plan["verdict_tag"] = f"LOW-QUALITY {plan['direction']} SETUP: skip", "muted"
    else:
        plan["verdict"], plan["verdict_tag"] = "NO TRADE: no edge right now", "muted"
        plan["invalidation"].append("Re-check when price reaches a value-area edge or the timeframes re-align.")

    plan["tf"] = [{k: v for k, v in s.items() if k != "_raw"} for s in snaps[::-1]]   # high -> low
    return plan


# ----------------------------------------------------------------------------
# Rendering (list of (text, tag)); tags: h, long, short, warn, muted, None
# ----------------------------------------------------------------------------
def _vol_word(p):
    return "very high" if p > 0.9 else "high" if p > 0.7 else "normal" if p > 0.3 else "low"


def render_plan_lines(plan):
    L = []
    add = lambda t="", tag=None: L.append((t, tag))
    add(f"{plan['symbol']}  |  {plan['style']} style  |  {' > '.join(plan['tfs'][::-1])}", "h")
    add(plan["verdict"], plan["verdict_tag"])
    add(f"Multi-timeframe bias {plan['bias']:+.2f}   |   {plan['alignment']}")
    for n in plan["notes"]:
        add("! " + n, "warn")
    add()
    add("TOP-DOWN READ", "h")
    for t in plan["tf"]:
        name = REGIME_TEXT[t["regime"]][0]
        add(f"{t['tf']:>4} {t['role']:<9} {name}", "h")
        add(f"      {t['stance']}; {t['structure']}")
        add(f"      delta {t['delta_bias']:+.0%} | RSI {t['rsi']:.0f} | volatility {_vol_word(t['vol_pct'])} | {t['ta_label']}")
        add(f"      => {t['meaning']}", "muted")
    add()
    add("STRATEGY", "h")
    add(plan["strategy"]["name"], "long" if plan["direction"] == "LONG" else "short" if plan["direction"] == "SHORT" else "muted")
    for w in plan["strategy"]["why"]:
        add("  - " + w)
    s = plan["setup"]
    if s:
        add()
        add("TRADE PLAN (reference levels, not a signal)", "h")
        add(f"  Side        {s['side']}")
        add(f"  Entry zone  {s['zone_lo']:.6g} - {s['zone_hi']:.6g}  (mid {s['entry']:.6g})" +
            (f"   [{', '.join(s['zone_names'])}]" if s["zone_names"] else ""))
        add(f"  Stop        {s['stop']:.6g}   (risk {s['risk']:.6g} per unit)")
        add(f"  Target 1    {s['t1']:.6g}   R:R {s['rr1']:.1f}")
        add(f"  Target 2    {s['t2']:.6g}   R:R {s['rr2']:.1f}")
        add(f"  Status      {s['state']}", None if s["in_zone"] else "warn")
        add("  Suggested management: take ~half at T1, move stop to entry, let the rest run to T2.", "muted")
        add()
        add("TRIGGER CHECKS", "h")
        for txt, ok in s["trigger"]:
            add(f"  [{'x' if ok else ' '}] {txt}", None if ok else "muted")
        z = s["size"]
        add()
        add("POSITION SIZE", "h")
        add(f"  Account {z['account']:.2f}, risk {z['risk_pct']:.2f}% = {z['risk_amount']:.2f} at stake")
        add(f"  Size {z['units']:.6g} units  (notional {z['notional']:.2f})")
        if z["leverage_needed"] > 1.0:
            add(f"  Needs ~{z['leverage_needed']:.1f}x leverage for this stop distance. "
                "Tighter risk or a smaller size is safer.", "warn")
        c = plan["confidence"]
        add()
        add(f"CONFIDENCE {c['score']}/100 ({c['label']})", "h")
        for txt, pts in c["factors"]:
            add(f"  {pts:+d}  {txt}")
    add()
    add("WHAT WOULD CHANGE MY MIND", "h")
    for t in plan["invalidation"]:
        add("  - " + t)
    add()
    add("MISTAKES TO AVOID IN THIS SETUP", "h")
    for t in plan["avoid"]:
        add("  - " + t)
    add()
    add("Educational tool, not financial advice. Backtest and journal before risking real money.", "muted")
    return L


def render_plan_text(plan):
    return "\n".join(t for t, _ in render_plan_lines(plan))


# ----------------------------------------------------------------------------
# Offline explainer
# ----------------------------------------------------------------------------
def offline_answer(question, plan, journal_text=None):
    q = question.lower()
    for key in sorted(GLOSSARY, key=len, reverse=True):
        if key in q.replace("?", " ").split() or (len(key) > 4 and key in q):
            return GLOSSARY[key]
    if plan is None:
        return "Run an analysis first, then ask me about it."
    s = plan["setup"]
    if any(w in q for w in ("journal", "my trades", "mistake", "my stats", "habit")):
        return journal_text or "No journal data yet. Add trades in the Journal tab and I will point out your patterns."
    if any(w in q for w in ("size", "how much", "lot", "units", "leverage")) and s:
        z = s["size"]
        return (f"Risking {z['risk_pct']:.2f}% of {z['account']:.2f} = {z['risk_amount']:.2f}. Stop distance is {s['risk']:.6g}, "
                f"so size = {z['units']:.6g} units (notional {z['notional']:.2f}). Size comes from the stop, never the other way round.")
    if any(w in q for w in ("stop", "invalid", "wrong", "exit")):
        return "\n".join(plan["invalidation"])
    if any(w in q for w in ("timeframe", "tf", "higher", "lower")):
        return "\n".join(f"{t['tf']} ({t['role']}): {REGIME_TEXT[t['regime']][0]}, {t['stance']}, {t['structure']}. {t['meaning']}"
                         for t in plan["tf"])
    if any(w in q for w in ("confidence", "score", "sure")) and plan["confidence"]:
        c = plan["confidence"]
        return f"Confidence {c['score']}/100 ({c['label']}):\n" + "\n".join(f"{p:+d} {t}" for t, p in c["factors"])
    if any(w in q for w in ("avoid", "careful", "danger")):
        return "\n".join(plan["avoid"])
    # default: why
    lines = [plan["verdict"], plan["strategy"]["name"]] + plan["strategy"]["why"]
    lines.append("(Set ANTHROPIC_API_KEY or paste a key in the Coach tab for free-form conversation in Arabic, French or English.)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# LLM coach (Claude API)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the AI strategy coach inside a crypto trading desktop app.
The app already computed a multi-timeframe analysis (volume profile, anchored VWAP, order-flow delta/CVD, trend regime, ATR) with deterministic code. You receive it below as JSON, plus the user's trade-journal statistics when available.

Rules:
- Use ONLY numbers present in the data. Never invent prices, indicators, news or fundamentals. If something is not in the data, say so.
- Explain in plain language, step by step: what each timeframe is doing, why this strategy was chosen, where the idea is wrong (invalidation), and how much to risk.
- Be direct and concrete. If the user's idea fights the higher timeframe or has poor risk/reward, say so and explain why.
- Never promise outcomes. Candle-level order flow is an approximation. At most one short reminder that this is not financial advice.
- When journal stats show a pattern (cutting winners, revenge trading, ignoring stops), connect it to the current setup.
- Reply in the same language the user writes in (Arabic, French or English).
- Keep answers focused: short paragraphs or short lists."""

BRIEFING_QUESTION = ("Explain this setup in plain language: the story across the timeframes, why this strategy, "
                     "what must happen before I enter, where I'm wrong, and what mistake I'm most likely to make here.")


def plan_to_json(plan):
    def default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return str(o)
    return json.dumps(plan, default=default, ensure_ascii=False)


def ask_llm(question, plan, history=None, journal_text=None, api_key=None, model=None, timeout=60):
    import requests
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("No API key. Set ANTHROPIC_API_KEY or paste it in the Coach tab.")
    system = SYSTEM_PROMPT
    if plan is not None:
        system += "\n\n<analysis>\n" + plan_to_json(plan) + "\n</analysis>"
    else:
        system += "\n\nNo analysis has been run yet."
    if journal_text:
        system += "\n\n<journal_stats>\n" + journal_text + "\n</journal_stats>"
    msgs = list(history or [])[-12:] + [{"role": "user", "content": question}]
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model or DEFAULT_MODEL, "max_tokens": 1500, "system": system, "messages": msgs},
        timeout=timeout,
    )
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except Exception:
            msg = r.text
        raise RuntimeError(f"API error {r.status_code}: {msg}")
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


# ----------------------------------------------------------------------------
# Providers + CLI
# ----------------------------------------------------------------------------
_PD_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h",
            "4h": "4h", "1d": "1D", "1w": "7D"}


def demo_provider(symbol, tf):
    df = ta.demo_data(n=400, seed=abs(hash(tf)) % 1000)
    df["time"] = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=len(df), freq=_PD_FREQ.get(tf, "15min"))
    return df


def binance_provider(symbol, tf):
    return ta.fetch_binance(symbol, tf, 500)


def main():
    p = argparse.ArgumentParser(description="Multi-timeframe AI strategy plan")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--style", choices=list(STYLES), default="Intraday")
    p.add_argument("--account", type=float, default=1000.0)
    p.add_argument("--risk", type=float, default=1.0, help="risk per trade, percent of account")
    p.add_argument("--demo", action="store_true", help="synthetic data")
    p.add_argument("--explain", action="store_true", help="ask the LLM coach to explain the setup")
    a = p.parse_args()
    plan = build_plan(demo_provider if a.demo else binance_provider, a.symbol.upper(), a.style, a.account, a.risk)
    print(render_plan_text(plan))
    if a.explain:
        print("\n" + "=" * 64 + "\nAI COACH\n")
        try:
            print(ask_llm(BRIEFING_QUESTION, plan))
        except Exception as e:
            print(f"(LLM unavailable: {e})")
            print(offline_answer("why", plan))


if __name__ == "__main__":
    main()