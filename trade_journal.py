#!/usr/bin/env python3
"""
trade_journal.py - log your trades and let the code find your patterns.

Stores trades in trade_journal.csv (same folder). Metrics are in R multiples
(profit/loss divided by the risk you set with your stop), so they are comparable
across coins and position sizes.
"""
import os

import numpy as np
import pandas as pd

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.csv")
COLS = ["open_time", "close_time", "symbol", "side", "timeframe", "strategy",
        "entry", "stop", "exit", "size", "fees", "notes"]
NUM = ["entry", "stop", "exit", "size", "fees"]


def load(path=JOURNAL_PATH):
    if not os.path.isfile(path):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    df["side"] = df["side"].astype(str).str.upper()
    return df[COLS]


def add_trade(row, path=JOURNAL_PATH):
    row = dict(row)
    side = str(row.get("side", "")).upper()
    if side not in ("LONG", "SHORT"):
        raise ValueError("Side must be LONG or SHORT.")
    for k in ("symbol", "entry", "stop", "exit", "size"):
        if row.get(k) in (None, ""):
            raise ValueError(f"Missing field: {k}")
    entry, stop = float(row["entry"]), float(row["stop"])
    if entry == stop:
        raise ValueError("Stop must differ from entry (R multiples need a risk distance).")
    if (side == "LONG" and stop > entry) or (side == "SHORT" and stop < entry):
        raise ValueError(f"A {side} stop must be {'below' if side == 'LONG' else 'above'} the entry.")
    now = pd.Timestamp.now().floor("min")
    row["close_time"] = row.get("close_time") or now
    row["open_time"] = row.get("open_time") or now
    row["side"] = side
    out = pd.DataFrame([{c: row.get(c, "") for c in COLS}])
    out.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def enrich(df):
    d = df.copy()
    dirn = np.where(d["side"] == "LONG", 1.0, -1.0)
    risk = (d["entry"] - d["stop"]).abs()
    d["r"] = np.where(risk > 0, dirn * (d["exit"] - d["entry"]) / risk, np.nan)
    d["pnl"] = dirn * (d["exit"] - d["entry"]) * d["size"] - d["fees"].fillna(0)
    d["hour"] = d["open_time"].dt.hour
    return d.dropna(subset=["r"]).sort_values("close_time").reset_index(drop=True)


def _group(d, col, min_n=1):
    rows = []
    for k, g in d.groupby(col):
        if pd.isna(k) or str(k) == "" or len(g) < min_n:
            continue
        rows.append({"key": k, "n": len(g), "win": float((g["r"] > 0).mean()),
                     "avg_r": float(g["r"].mean()), "total_r": float(g["r"].sum())})
    return sorted(rows, key=lambda r: -r["total_r"])


def stats(df):
    d = enrich(df)
    if d.empty:
        return None
    r = d["r"]
    wins, losses = r[r > 0], r[r <= 0]
    gp, gl = d.loc[d["pnl"] > 0, "pnl"].sum(), -d.loc[d["pnl"] <= 0, "pnl"].sum()
    cum = r.cumsum()
    dd = float((cum.cummax() - cum).max())
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)

    # revenge trading: opened within 30 minutes of closing a loser
    revenge = []
    ordered = d.sort_values("open_time").reset_index(drop=True)
    for i in range(1, len(ordered)):
        p, t = ordered.iloc[i - 1], ordered.iloc[i]
        if p["r"] <= 0 and pd.notna(p["close_time"]) and pd.notna(t["open_time"]):
            gap = (t["open_time"] - p["close_time"]).total_seconds() / 60
            if 0 <= gap <= 30:
                revenge.append(float(t["r"]))
    s = {
        "n": len(d), "win_rate": float((r > 0).mean()), "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0, "expectancy_r": float(r.mean()),
        "total_r": float(r.sum()), "total_pnl": float(d["pnl"].sum()),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0,
        "max_dd_r": dd, "max_losing_streak": int(best), "best_r": float(r.max()), "worst_r": float(r.min()),
        "big_losses": float((r < -1.3).sum() / max(len(losses), 1)),
        "revenge_n": len(revenge), "revenge_avg_r": float(np.mean(revenge)) if revenge else None,
        "by_side": _group(d, "side"), "by_timeframe": _group(d, "timeframe"),
        "by_strategy": _group(d, "strategy"), "by_symbol": _group(d, "symbol"),
        "by_hour": _group(d, "hour"),
    }
    return s


def insights(s):
    if s is None:
        return ["No trades logged yet."]
    out = []
    if s["n"] < 30:
        out.append(f"Only {s['n']} trades: patterns below are hints, not proof (aim for 30+).")
    if s["expectancy_r"] > 0:
        out.append(f"Positive expectancy ({s['expectancy_r']:+.2f}R per trade). Protect it: do not change what works.")
    else:
        out.append(f"Negative expectancy ({s['expectancy_r']:+.2f}R per trade): on average each trade costs you. Fix before adding size.")
    if s["big_losses"] >= 0.25 and s["avg_loss_r"] < -1.2:
        out.append(f"{s['big_losses']:.0%} of your losses exceed -1.3R: stops are not being honored (moved or ignored). This is the most expensive habit.")
    if s["win_rate"] < 0.6 and s["avg_win_r"] > 0 and s["avg_win_r"] < 0.8 * abs(s["avg_loss_r"]):
        out.append(f"Avg win {s['avg_win_r']:.2f}R vs avg loss {s['avg_loss_r']:.2f}R with {s['win_rate']:.0%} win rate: you cut winners too early. Use targets at structure and trail the rest.")
    if s["win_rate"] >= 0.6 and s["expectancy_r"] <= 0:
        out.append("High win rate but negative expectancy: small wins, large losses. Classic pattern; cap the loss first.")
    if s["revenge_n"] >= 3 and s["revenge_avg_r"] is not None and s["revenge_avg_r"] < s["expectancy_r"]:
        out.append(f"{s['revenge_n']} trades opened within 30 min of a loss average {s['revenge_avg_r']:+.2f}R (worse than your norm). Add a cool-down after a loser.")
    if s["max_losing_streak"] >= 5:
        out.append(f"Longest losing streak: {s['max_losing_streak']}. Decide in advance the stop-for-the-day rule (e.g. 3 losses).")
    for label, key in (("strategy", "by_strategy"), ("timeframe", "by_timeframe"), ("side", "by_side"),
                       ("coin", "by_symbol"), ("hour of day", "by_hour")):
        g = [x for x in s[key] if x["n"] >= 3]
        if len(g) >= 2:
            b, w = g[0], g[-1]
            if b["avg_r"] > 0 and w["avg_r"] < 0:
                unit = "h" if key == "by_hour" else ""
                out.append(f"Best {label}: {b['key']}{unit} ({b['avg_r']:+.2f}R avg, {b['n']} trades). Worst: {w['key']}{unit} ({w['avg_r']:+.2f}R avg, {w['n']} trades).")
    return out


def format_stats(s):
    if s is None:
        return "No trades logged yet."
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    lines = [
        f"Trades {s['n']} | win rate {s['win_rate']:.0%} | expectancy {s['expectancy_r']:+.2f}R | profit factor {pf}",
        f"Avg win {s['avg_win_r']:+.2f}R | avg loss {s['avg_loss_r']:+.2f}R | best {s['best_r']:+.1f}R | worst {s['worst_r']:+.1f}R",
        f"Total {s['total_r']:+.1f}R ({s['total_pnl']:+.2f} money) | max drawdown {s['max_dd_r']:.1f}R | longest losing streak {s['max_losing_streak']}",
    ]
    for label, key in (("Strategy", "by_strategy"), ("Timeframe", "by_timeframe"), ("Side", "by_side")):
        if s[key]:
            lines.append(f"{label}: " + "; ".join(f"{x['key']} n={x['n']} avg {x['avg_r']:+.2f}R" for x in s[key]))
    lines.append("")
    lines.append("COACHING NOTES")
    lines += ["- " + i for i in insights(s)]
    return "\n".join(lines)