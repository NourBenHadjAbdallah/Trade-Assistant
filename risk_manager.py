#!/usr/bin/env python3
"""
risk_manager.py - check every trade against your own limits before you take it.

Limits live in trader_settings.json (next to this file, not committed). Trades you take are
recorded in signals.db so the daily limits know what you already risked today.

Rules (all must pass):
  * the plan is a real trade idea (not "no trade" / "skip") and the coin is not blocked
  * the stop is on the right side of the entry and not too tight: at least MIN_STOP_ATR x ATR,
    and fees must cost at most MAX_FEE_R of the risk
  * reward:risk to T1 meets your minimum
  * position size at your risk % needs no more leverage than you allow
  * today's total risk (trades already taken + this one) fits your daily loss limit
  * you have not hit your trades-per-day limit
Prints PASS or BLOCK with every reason.

CLI:
    python risk_manager.py --symbol BTCUSDT --style Intraday
"""
import argparse
import json
import os

import pandas as pd

import signal_log as sl
import trade_backtest as tb

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trader_settings.json")
DEFAULTS = {
    "account": 1000.0,             # account size in USDT
    "risk_per_trade_pct": 1.0,     # max loss per trade, % of account (fees included)
    "daily_loss_limit_pct": 3.0,   # max total risk taken per day, % of account
    "max_trades_per_day": 3,
    "min_rr": 1.5,                 # minimum reward:risk to target 1
    "max_leverage": 3.0,           # position value / account
    "min_stop_atr": 0.5,           # stop at least this many ATRs away
    "max_fee_r": 0.25,             # fees may eat at most this share of the risk
    "blocked": [],                 # coins you never trade, e.g. ["DOGEUSDT"]
}
LABELS = {
    "account": "Account size (USDT)", "risk_per_trade_pct": "Max loss per trade (%)",
    "daily_loss_limit_pct": "Daily loss limit (%)", "max_trades_per_day": "Max trades per day",
    "min_rr": "Minimum reward:risk", "max_leverage": "Max leverage (x)",
    "min_stop_atr": "Minimum stop distance (ATR)", "max_fee_r": "Max fees (share of risk)",
    "blocked": "Blocked coins (comma separated)",
}


def load_settings(path=SETTINGS_PATH):
    s = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save_settings(values, path=SETTINGS_PATH):
    s = dict(DEFAULTS)
    for k, v in values.items():
        if k not in DEFAULTS:
            continue
        if k == "blocked":
            v = [x.strip().upper() for x in (v.split(",") if isinstance(v, str) else v) if x.strip()]
        elif k == "max_trades_per_day":
            v = int(float(v))
        else:
            v = float(v)
        if k != "blocked" and v < 0:
            raise ValueError(f"{LABELS[k]} cannot be negative.")
        s[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    return s


# ----------------------------------------------------------------------------
# Trades taken today (worst-case loss each), stored next to the signal log
# ----------------------------------------------------------------------------
def _db(path=None):
    con = sl._db(path)
    con.execute("""CREATE TABLE IF NOT EXISTS taken (
        id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, symbol TEXT, side TEXT, entry REAL, stop REAL,
        units REAL, max_loss REAL)""")
    return con


def taken_today(path=None):
    """Trades recorded today (local calendar day)."""
    with _db(path) as con:
        df = pd.read_sql_query("SELECT * FROM taken ORDER BY id", con)
    df["time"] = pd.to_datetime(df["time"])
    return df[df["time"].dt.date == pd.Timestamp.now().date()].reset_index(drop=True)


def record_taken(plan, check_result, path=None):
    if check_result["verdict"] != "PASS":
        raise ValueError("Only a trade that passed the risk check can be recorded.")
    s, z = plan["setup"], check_result["size"]
    with _db(path) as con:
        con.execute("INSERT INTO taken (time, symbol, side, entry, stop, units, max_loss) VALUES (?,?,?,?,?,?,?)",
                    (str(pd.Timestamp.now().floor("s")), plan["symbol"], plan["direction"], s["entry"], s["stop"],
                     z["units"], z["max_loss"]))


# ----------------------------------------------------------------------------
# The check
# ----------------------------------------------------------------------------
def check(plan, settings=None, today=None, cost=tb.COST_PER_SIDE):
    """Returns {verdict: PASS|BLOCK, rules: [(name, ok, detail)], size: {...} or None}."""
    st = settings or load_settings()
    today = taken_today() if today is None else today
    rules = []
    add = lambda name, ok, detail: rules.append((name, bool(ok), detail))
    s = plan.get("setup")
    acct = float(st["account"])
    if not s:
        add("Trade idea", False, "No trade: the plan has no direction right now.")
        return {"verdict": "BLOCK", "rules": rules, "size": None}

    add("Trade idea", plan.get("grade") in ("strong", "actionable", "forming"),
        f"Plan grade: {plan.get('grade', '?')}" + ("" if plan.get("grade") in ("strong", "actionable", "forming")
                                                   else " (the app itself says skip)"))
    add("Coin allowed", plan["symbol"] not in st["blocked"],
        "On your blocked list." if plan["symbol"] in st["blocked"] else "Not blocked.")

    d = 1 if plan["direction"] == "LONG" else -1
    entry, stop = float(s["entry"]), float(s["stop"])
    risk = abs(entry - stop)
    add("Stop on the right side", risk > 0 and d * (entry - stop) > 0,
        f"Stop {stop:.6g} {'below' if d > 0 else 'above'} entry {entry:.6g}." if risk > 0 and d * (entry - stop) > 0
        else "Stop is missing or on the wrong side of the entry.")
    if risk <= 0 or d * (entry - stop) <= 0:
        return {"verdict": "BLOCK", "rules": rules, "size": None}

    atr = float(plan["tf"][1].get("atr") or 0)
    stop_atr = risk / atr if atr > 0 else float("inf")
    add("Stop not too tight", stop_atr >= st["min_stop_atr"],
        f"Stop is {stop_atr:.2f} ATR away (minimum {st['min_stop_atr']:g})."
        + ("" if stop_atr >= st["min_stop_atr"] else " Normal noise will hit a tighter stop."))
    fee_r = 2 * cost * entry / risk
    add("Fees affordable", fee_r <= st["max_fee_r"] + 1e-9,
        f"Fees cost {fee_r:.2f}R of this trade (maximum {st['max_fee_r']:g}R).")
    add("Reward:risk", s["rr1"] >= st["min_rr"], f"{s['rr1']:.1f}R to T1 (minimum {st['min_rr']:g}).")

    risk_amt = acct * st["risk_per_trade_pct"] / 100
    units = risk_amt / (risk + 2 * cost * entry)          # fees counted inside the max loss
    notional = units * entry
    lev = notional / acct if acct > 0 else float("inf")
    max_loss = units * (risk + 2 * cost * entry)
    add("Leverage", lev <= st["max_leverage"],
        f"Position {notional:,.2f} USDT = {lev:.1f}x your account (maximum {st['max_leverage']:g}x).")

    used = float(today["max_loss"].sum()) if len(today) else 0.0
    limit = acct * st["daily_loss_limit_pct"] / 100
    add("Daily loss limit", used + max_loss <= limit + 1e-9,
        f"Risked today {used:,.2f} + this {max_loss:,.2f} = {used + max_loss:,.2f} USDT (limit {limit:,.2f}).")
    add("Trades per day", len(today) < st["max_trades_per_day"],
        f"{len(today)} taken today (maximum {st['max_trades_per_day']}).")

    size = {"units": units, "notional": notional, "leverage": lev, "max_loss": max_loss, "risk_amount": risk_amt}
    return {"verdict": "PASS" if all(ok for _, ok, _ in rules) else "BLOCK", "rules": rules, "size": size}


def report_text(plan, res):
    L = [f"RISK CHECK  {plan['symbol']} {plan['direction']}: {res['verdict']}"]
    L += [f"  [{'x' if ok else ' '}] {name}: {detail}" for name, ok, detail in res["rules"]]
    z = res["size"]
    if z:
        L.append(f"  Size {z['units']:.6g} units = {z['notional']:,.2f} USDT, max loss {z['max_loss']:,.2f} USDT")
    return "\n".join(L)


def main():
    import trade_ai as ai
    p = argparse.ArgumentParser(description="Check a trade plan against your risk limits")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--style", choices=list(ai.STYLES), default="Intraday")
    a = p.parse_args()
    plan = ai.build_plan(ai.binance_provider, a.symbol.upper(), a.style)
    print(plan["verdict"])
    print(report_text(plan, check(plan)))


if __name__ == "__main__":
    main()
