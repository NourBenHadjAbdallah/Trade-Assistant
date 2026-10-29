#!/usr/bin/env python3
"""
signal_log.py - the app's track record: log every signal, then grade it against what price did next.

Every BUY/SELL, STRONG and WAIT-FOR-ENTRY signal from the scanner is stored in signals.db (SQLite,
next to this file). Each one is then graded on the candles that came AFTER it, following the plan
the app itself recommends:
  * fill   : price reaches the entry (or is already in the entry zone: fills at the next candle's open)
  * stop   : full loss at the stop; a candle that opens beyond the stop fills at its open (gap)
  * T1     : half the position is taken off, the stop moves to the entry (break-even)
  * T2     : the rest is closed at T2
  * never filled within EXPIRE_BARS -> "expired" (no trade);  still open after MAX_HOLD bars -> closed at market
  * stop and target inside the same candle -> stop first (pessimistic); fees as in trade_backtest.py
Results are in R (1R = the risk between entry and stop).

CLI (works without the GUI; a Windows scheduled task runs `scan` every 15 min, log in signal_scan.log):
    python signal_log.py scan --style Intraday --coins 20     # scan, log new signals, grade open ones
    python signal_log.py update                               # grade open signals only
    python signal_log.py report                               # print the track record
"""
import argparse
import os
import sqlite3
import sys
import threading

import config  # noqa: F401  (loads .env before os.environ is read)

import numpy as np
import pandas as pd

import trade_backtest as tb

DB_PATH = os.environ.get("SIGNALS_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.db"))
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_scan.log")
TRACKED = ("strong", "actionable", "forming")      # grades worth logging
EXPIRE_BARS = 24                                    # setup-TF candles a signal may wait for its entry
MAX_HOLD = 60                                       # setup-TF candles a filled trade may stay open
GRADE_LABEL = {"strong": "STRONG", "actionable": "BUY / SELL", "forming": "WAIT FOR ENTRY"}
SCORE_BUCKETS = [(0, 45, "under 45"), (45, 55, "45-54"), (55, 70, "55-69"), (70, 101, "70+")]

COLS = ["id", "created", "symbol", "style", "tf", "side", "grade", "score", "checklist", "price",
        "entry", "zone_lo", "zone_hi", "stop", "t1", "t2", "in_zone",
        "status", "result", "fill_time", "fill", "exit_time", "exit", "r", "updated"]
_LOCK = threading.Lock()


def _db(path=None):
    con = sqlite3.connect(path or DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, symbol TEXT, style TEXT, tf TEXT, side INTEGER,
        grade TEXT, score INTEGER, checklist TEXT, price REAL, entry REAL, zone_lo REAL, zone_hi REAL,
        stop REAL, t1 REAL, t2 REAL, in_zone INTEGER, status TEXT DEFAULT 'pending', result TEXT,
        fill_time TEXT, fill REAL, exit_time TEXT, exit REAL, r REAL, updated TEXT)""")
    return con


def _now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def record(rows, path=None):
    """Log new signals from ai.scan() rows. Skips ideas already being tracked. Returns how many were added."""
    added = 0
    with _LOCK, _db(path) as con:
        for row in rows:
            plan, rec = row.get("plan"), row.get("rec")
            if not plan or not rec or rec["grade"] not in TRACKED or not plan.get("setup"):
                continue
            s, side = plan["setup"], 1 if plan["direction"] == "LONG" else -1
            dup = con.execute("SELECT 1 FROM signals WHERE symbol=? AND style=? AND side=? "
                              "AND status IN ('pending', 'open')", (row["symbol"], plan["style"], side)).fetchone()
            if dup:
                continue
            conf = s.get("confluence") or []
            con.execute(
                "INSERT INTO signals (created, symbol, style, tf, side, grade, score, checklist, price, entry, "
                "zone_lo, zone_hi, stop, t1, t2, in_zone, status, updated) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)",
                (str(_now()), row["symbol"], plan["style"], plan["tfs"][1], side, rec["grade"], int(rec["score"]),
                 f"{s.get('confluence_met', 0)}/{len(conf)}" if conf else "", float(plan["tf"][1]["price"]),
                 float(s["entry"]), float(s["zone_lo"]), float(s["zone_hi"]), float(s["stop"]),
                 float(s["t1"]), float(s["t2"]), int(bool(s["in_zone"])), str(_now())))
            added += 1
    return added


def load(path=None):
    with _db(path) as con:
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id", con)
    for c in ("created", "fill_time", "exit_time", "updated"):
        df[c] = pd.to_datetime(df[c])
    return df


# ----------------------------------------------------------------------------
# Grading (pure: same signal + same candles -> same result)
# ----------------------------------------------------------------------------
def grade(sig, candles, cost=tb.COST_PER_SIDE, expire_bars=EXPIRE_BARS, max_hold=MAX_HOLD):
    """
    sig: mapping with side, entry, stop, t1, t2, in_zone, created.
    candles: CLOSED candles of the setup timeframe (time, open, high, low, close).
    Returns {status, result, fill_time, fill, exit_time, exit, r}. status: pending | open | closed | expired.
    """
    out = {"status": "pending", "result": None, "fill_time": None, "fill": None,
           "exit_time": None, "exit": None, "r": None}
    c = candles[candles["time"] > pd.Timestamp(sig["created"])].reset_index(drop=True)
    d, entry, stop = int(sig["side"]), float(sig["entry"]), float(sig["stop"])
    o, h, l, cl, t = (c[k].values for k in ("open", "high", "low", "close", "time"))

    # 1) fill
    fi = None
    for i in range(min(len(c), expire_bars)):
        if sig["in_zone"] and i == 0:
            fi, fill = 0, float(o[0])
            break
        if (d > 0 and l[i] <= entry) or (d < 0 and h[i] >= entry):
            fi, fill = i, float(min(o[i], entry) if d > 0 else max(o[i], entry))
            break
    if fi is None:
        if len(c) >= expire_bars:
            out.update(status="expired", result="never filled")
        return out
    risk = abs(fill - stop)
    if risk <= 0 or d * (fill - stop) <= 0:            # opened beyond the stop: the idea was already wrong
        out.update(status="expired", result="moved past stop before entry")
        return out
    out.update(status="open", fill_time=t[fi], fill=fill)
    fee = 2 * cost * fill / risk

    def close(i, px, result, banked=0.0, part=1.0):
        out.update(status="closed", result=result, exit_time=t[i], exit=float(px),
                   r=float(banked + part * d * (px - fill) / risk - fee))
        return out

    # 2) manage: full size until T1, then half size with the stop at the entry
    t1, t2, banked, half = float(sig["t1"]), float(sig["t2"]), 0.0, False
    for i in range(fi, min(len(c), fi + max_hold)):
        cur_stop = fill if half else stop
        if (d > 0 and l[i] <= cur_stop) or (d < 0 and h[i] >= cur_stop):
            px = min(o[i], cur_stop) if d > 0 else max(o[i], cur_stop)
            if i == fi:                                  # fill candle: only a real gap/limit price counts
                px = cur_stop
            return close(i, px, "T1 then break-even" if half else "stop", banked, 0.5 if half else 1.0)
        if i == fi:
            continue                                     # targets only on later candles (pessimistic)
        if not half and ((d > 0 and h[i] >= t1) or (d < 0 and l[i] <= t1)):
            banked, half = 0.5 * d * (t1 - fill) / risk, True
            continue
        if half and ((d > 0 and h[i] >= t2) or (d < 0 and l[i] <= t2)):
            return close(i, t2, "T2", banked, 0.5)
    if len(c) - fi >= max_hold:
        i = fi + max_hold - 1
        return close(i, cl[i], "time limit", banked, 0.5 if half else 1.0)
    return out


def fetch_since(symbol, tf, start):
    """Closed candles of `tf` from `start` (UTC) onward, straight from Binance."""
    ms = tb.TF_MIN[tf] * 60_000
    batch = tb._get_klines({"symbol": symbol, "interval": tf, "limit": 1000,
                            "startTime": int(pd.Timestamp(start).floor(f"{tb.TF_MIN[tf]}min").value // 1_000_000)})
    df = pd.DataFrame([b[:5] for b in batch], columns=["time", "open", "high", "low", "close"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df[df["time"] + pd.Timedelta(milliseconds=ms) <= _now()].reset_index(drop=True)


def update(fetch=fetch_since, path=None):
    """Re-grade every pending/open signal. Returns how many changed status."""
    with _db(path) as con:
        rows = con.execute("SELECT id, symbol, tf, side, entry, stop, t1, t2, in_zone, created, status "
                           "FROM signals WHERE status IN ('pending', 'open')").fetchall()
    changed = 0
    for sid, sym, tf, side, entry, stop, t1, t2, in_zone, created, status in rows:
        try:
            g = grade({"side": side, "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                       "in_zone": in_zone, "created": created}, fetch(sym, tf, created))
        except Exception:
            continue                                     # network hiccup: try again next time
        vals = {k: (str(v) if isinstance(v, (pd.Timestamp, np.datetime64)) else v) for k, v in g.items()}
        with _LOCK, _db(path) as con:
            con.execute("UPDATE signals SET status=?, result=?, fill_time=?, fill=?, exit_time=?, exit=?, r=?, "
                        "updated=? WHERE id=?", (vals["status"], vals["result"], vals["fill_time"], vals["fill"],
                                                 vals["exit_time"], vals["exit"], vals["r"], str(_now()), sid))
        changed += g["status"] != status
    return changed


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def _block(df):
    closed = df[df["status"] == "closed"]
    r = closed["r"].astype(float)
    filled = df[df["status"].isin(["open", "closed"])]
    decided = df[df["status"].isin(["open", "closed", "expired"])]
    return {"signals": len(df), "closed": len(closed), "open": int((df["status"] == "open").sum()),
            "pending": int((df["status"] == "pending").sum()), "expired": int((df["status"] == "expired").sum()),
            "fill_rate": len(filled) / len(decided) if len(decided) else None,
            "win_rate": float((r > 0).mean()) if len(r) else None,
            "avg_r": float(r.mean()) if len(r) else None, "total_r": float(r.sum()) if len(r) else 0.0}


def summary(df=None):
    """Overall, per signal type and per confidence bucket: does a higher grade/score really win more?"""
    df = load() if df is None else df
    by_grade = [(GRADE_LABEL[g], _block(df[df["grade"] == g])) for g in TRACKED]
    by_score = [(lab, _block(df[(df["score"] >= lo) & (df["score"] < hi)])) for lo, hi, lab in SCORE_BUCKETS]
    by_style = [(st, _block(df[df["style"] == st])) for st in sorted(df["style"].dropna().unique())]
    closed = df[df["status"] == "closed"].sort_values("exit_time")
    return {"all": _block(df), "by_grade": by_grade, "by_score": by_score, "by_style": by_style,
            "curve": closed["r"].astype(float).cumsum().tolist(), "verdict": verdict(_block(df))}


def verdict(b):
    n = b["closed"]
    if n < 30:
        return f"Too early to judge: {n} closed trade(s). Wait for 30+ (100+ to trust it)."
    if b["avg_r"] > 0:
        return (f"Positive so far: {b['avg_r']:+.2f}R per trade over {n} trades. "
                + ("Still a small sample." if n < 100 else "Encouraging; keep size small and keep tracking."))
    return f"Not working yet: {b['avg_r']:+.2f}R per trade over {n} trades. Do not trade these signals with real money."


def report_text(s=None):
    s = s or summary()
    fmt = lambda b: (f"{b['signals']:>4} signals | {b['closed']:>3} closed | "
                     + (f"win {b['win_rate']:.0%} | avg {b['avg_r']:+.2f}R | total {b['total_r']:+.1f}R"
                        if b["closed"] else "no closed trades yet"))
    L = ["SIGNAL TRACK RECORD (after fees, in R)", "ALL        " + fmt(s["all"]), "", "By signal type"]
    L += [f"  {name:<15}{fmt(b)}" for name, b in s["by_grade"]]
    L += ["", "By confidence score (higher should win more)"]
    L += [f"  {name:<15}{fmt(b)}" for name, b in s["by_score"]]
    L += ["", ">> " + s["verdict"]]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description="Signal track record: log scanner signals and grade them")
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="scan the top coins, log new signals, grade open ones")
    sc.add_argument("--style", default="Intraday")
    sc.add_argument("--coins", type=int, default=20)
    sub.add_parser("update", help="grade open signals")
    sub.add_parser("report", help="print the track record")
    a = p.parse_args()
    if sys.stdout is None:                              # pythonw (scheduled task): no console, keep a log
        sys.stdout = sys.stderr = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        print(f"\n=== {_now()} UTC  {' '.join(sys.argv[1:])}")
    if a.cmd == "scan":
        import trade_ai as ai
        rows = ai.scan(ai.binance_provider, ai.top_symbols(a.coins), a.style)
        print(f"Logged {record(rows)} new signal(s). {update()} signal(s) changed status.")
    elif a.cmd == "update":
        print(f"{update()} signal(s) changed status.")
    print(report_text())


if __name__ == "__main__":
    main()
