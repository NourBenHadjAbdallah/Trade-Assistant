#!/usr/bin/env python3
"""
price_alerts.py - price alerts that ring once, plus the entry timer.

Alert kinds
  above / below      price crosses a level
  pct_up / pct_down  price moves N% from where it was when the alert was set
  confirm            entry timer for a plan: rings when price has come into the entry zone AND a
                     closed candle of the trigger timeframe confirms (closes back out of the zone in the
                     trade direction, with the candle and its taker delta pointing the same way).
                     Rings "invalid" instead if a candle closes beyond the stop first, "expired" after
                     EXPIRE_BARS candles without a trigger.
add_plan() sets five at once: entry zone reached, entry confirmed, stop, target 1, target 2.

Alerts live in signals.db. They are checked by the app every 30 s while it is open and by the
scheduled scan every 15 min. Each rings once. Optional phone alerts: set TELEGRAM_TOKEN and
TELEGRAM_CHAT_ID in .env (message your bot once, then read the chat id from getUpdates).

CLI:
    python price_alerts.py add BTCUSDT above 82000
    python price_alerts.py add ETHUSDT pct_down 3
    python price_alerts.py list
    python price_alerts.py check
"""
import argparse
import json
import os

import config  # noqa: F401  (loads .env before os.environ is read)

import pandas as pd
import requests

import signal_log as sl
import trade_assistant as ta

PRICE_URLS = ["https://api.binance.com/api/v3/ticker/price", "https://data-api.binance.vision/api/v3/ticker/price"]
KINDS = ("above", "below", "pct_up", "pct_down", "confirm")
EXPIRE_BARS = 24


def _db(path=None):
    con = sl._db(path)
    con.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, symbol TEXT, kind TEXT, level REAL, base REAL,
        side INTEGER, tf TEXT, zone_lo REAL, zone_hi REAL, stop REAL, note TEXT,
        status TEXT DEFAULT 'active', fired_at TEXT, fired_price REAL, message TEXT, parent INTEGER)""")
    if "parent" not in {r[1] for r in con.execute("PRAGMA table_info(alerts)")}:      # older databases
        con.execute("ALTER TABLE alerts ADD COLUMN parent INTEGER")
    return con


def _now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")


# ----------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------
def fetch_prices(symbols):
    """Latest price for several symbols in one request."""
    syms = sorted(set(symbols))
    if not syms:
        return {}
    last = None
    for url in PRICE_URLS:
        try:
            r = requests.get(url, params={"symbols": json.dumps(syms, separators=(",", ":"))}, timeout=10)
            r.raise_for_status()
            return {d["symbol"]: float(d["price"]) for d in r.json()}
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not get prices from Binance ({last}).")


# ----------------------------------------------------------------------------
# Entry timer
# ----------------------------------------------------------------------------
def entry_state(side, zone_lo, zone_hi, stop, df, since=None, expire_bars=EXPIRE_BARS):
    """
    Where a pullback entry stands, from CLOSED candles of the trigger timeframe.
    since: only candles opening after this time count (None = the last 12 candles).
    Returns {state: waiting | in_zone | triggered | invalid | expired, detail, time, price}.
    """
    d = 1 if side > 0 else -1
    bars = df[df["time"] > pd.Timestamp(since)] if since is not None else df.tail(12)
    bars = bars.reset_index(drop=True)
    out = {"state": "waiting", "detail": "", "time": None, "price": None}
    if bars.empty:
        out["detail"] = "Waiting for the first closed candle."
        return out
    touched = None
    for i, b in bars.iterrows():
        if d * (b["close"] - stop) <= 0:
            return {**out, "state": "invalid", "time": b["time"], "price": float(b["close"]),
                    "detail": f"A candle closed beyond the stop ({b['close']:.6g}) before the entry confirmed."}
        if touched is None and ((d > 0 and b["low"] <= zone_hi) or (d < 0 and b["high"] >= zone_lo)):
            touched = i
        if touched is not None:
            delta = 2 * b.get("buy_volume", b["volume"] / 2) - b["volume"]
            out_of_zone = b["close"] > zone_hi if d > 0 else b["close"] < zone_lo
            if out_of_zone and d * (b["close"] - b["open"]) > 0 and d * delta > 0:
                return {**out, "state": "triggered", "time": b["time"], "price": float(b["close"]),
                        "detail": f"Confirmed at {b['close']:.6g}: candle closed {'above' if d > 0 else 'below'} the "
                                  f"zone with {'buyers' if d > 0 else 'sellers'} in control."}
    last = float(bars["close"].iloc[-1])
    if since is not None and len(bars) >= expire_bars:
        return {**out, "state": "expired", "price": last,
                "detail": f"No confirmation within {expire_bars} candles: the setup is stale."}
    if touched is not None:
        return {**out, "state": "in_zone", "price": last,
                "detail": f"Price reached the zone. Waiting for a candle to close {'above' if d > 0 else 'below'} "
                          f"{zone_hi if d > 0 else zone_lo:.6g} with {'buyers' if d > 0 else 'sellers'} in control."}
    edge = zone_hi if d > 0 else zone_lo
    return {**out, "state": "waiting", "price": last,
            "detail": f"Price {last:.6g} is {abs(last - edge) / last:.2%} from the zone ({zone_lo:.6g} - {zone_hi:.6g})."}


def plan_entry_state(plan):
    """Entry timer for a fresh build_plan() result (uses the trigger-TF candles the plan downloaded)."""
    import trade_ai as ai
    s, dfs = plan.get("setup"), plan.get("_dfs") or {}
    tf = plan["tfs"][0]
    if not s or tf not in dfs:
        return None
    side = 1 if plan["direction"] == "LONG" else -1
    return entry_state(side, s["zone_lo"], s["zone_hi"], s["stop"], ai.closed_bars(dfs[tf], tf))


# ----------------------------------------------------------------------------
# Managing alerts
# ----------------------------------------------------------------------------
def add(symbol, kind, level=None, note="", base=None, side=None, tf=None, zone_lo=None, zone_hi=None, stop=None,
        parent=None, path=None):
    """parent: id of another alert; this one stays disarmed ('waiting') until that one rings."""
    if kind not in KINDS:
        raise ValueError(f"Unknown alert kind '{kind}'. Use one of: {', '.join(KINDS)}.")
    if kind != "confirm" and (level is None or float(level) <= 0):
        raise ValueError("Give a positive price (or percent).")
    if kind in ("pct_up", "pct_down") and base is None:
        base = fetch_prices([symbol])[symbol]
    with _db(path) as con:
        cur = con.execute(
            "INSERT INTO alerts (created, symbol, kind, level, base, side, tf, zone_lo, zone_hi, stop, note, status, "
            "parent) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(_now()), symbol.upper(), kind, None if level is None else float(level), base, side, tf,
             zone_lo, zone_hi, stop, note, "waiting" if parent else "active", parent))
        return cur.lastrowid


def add_plan(plan, path=None):
    """
    Five alerts for a plan: zone reached, entry confirmed, stop, T1, T2. Returns how many were added.
    Stop and targets stay disarmed until price reaches the entry zone (no "target hit" before you are in).
    """
    s = plan.get("setup")
    if not s:
        raise ValueError("This plan has no trade to watch.")
    sym, d = plan["symbol"], 1 if plan["direction"] == "LONG" else -1
    near, far = ("below", "above") if d > 0 else ("above", "below")
    tag = f"{plan['direction'].lower()} plan"
    zone = add(sym, near, s["zone_hi"] if d > 0 else s["zone_lo"], f"entry zone reached ({tag})", path=path)
    add(sym, "confirm", note=f"entry confirmed ({tag})", side=d, tf=plan["tfs"][0],
        zone_lo=s["zone_lo"], zone_hi=s["zone_hi"], stop=s["stop"], path=path)
    add(sym, near, s["stop"], f"STOP hit ({tag})", parent=zone, path=path)
    add(sym, far, s["t1"], f"target 1 hit ({tag}): take half, stop to entry", parent=zone, path=path)
    add(sym, far, s["t2"], f"target 2 hit ({tag})", parent=zone, path=path)
    return 5


def list_alerts(status=None, path=None):
    with _db(path) as con:
        q, args = "SELECT * FROM alerts", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        return pd.read_sql_query(q + " ORDER BY id DESC", con, params=args)


def cancel(alert_id, path=None):
    with _db(path) as con:
        con.execute("UPDATE alerts SET status='cancelled' WHERE (id=? OR parent=?) AND status IN ('active', 'waiting')",
                    (int(alert_id), int(alert_id)))


def describe(a):
    """One line for an alert row (dict-like)."""
    sym = a["symbol"].replace("USDT", "")
    k = a["kind"]
    if k == "above":
        what = f"{sym} at or above {a['level']:.6g}"
    elif k == "below":
        what = f"{sym} at or below {a['level']:.6g}"
    elif k == "pct_up":
        what = f"{sym} up {a['level']:g}% (from {a['base']:.6g})"
    elif k == "pct_down":
        what = f"{sym} down {a['level']:g}% (from {a['base']:.6g})"
    else:
        what = f"{sym} entry confirmation on {a['tf']}"
    return what + (f": {a['note']}" if a.get("note") else "")


def _hit(a, price):
    k, lv = a["kind"], a["level"]
    if k == "above":
        return price >= lv
    if k == "below":
        return price <= lv
    if k == "pct_up":
        return price >= a["base"] * (1 + lv / 100)
    if k == "pct_down":
        return price <= a["base"] * (1 - lv / 100)
    return False


def check(prices=None, klines=None, notify=True, path=None):
    """Check every active alert once. Returns the list of {id, symbol, message} that fired now."""
    active = list_alerts("active", path)
    if active.empty:
        return []
    fired = []
    simple = active[active["kind"] != "confirm"]
    if prices is None and not simple.empty:
        prices = fetch_prices(simple["symbol"])
    prices = prices or {}
    for _, a in simple.iterrows():
        px = prices.get(a["symbol"])
        if px is not None and _hit(a, px):
            fired.append((a, px, "🔔 " + describe(a) + f". Now {px:.6g}.", "fired"))
    for _, a in active[active["kind"] == "confirm"].iterrows():
        try:
            df = (klines or (lambda s, tf: ta.fetch_binance(s, tf, 200)))(a["symbol"], a["tf"])
        except Exception:
            continue
        import trade_ai as ai
        st = entry_state(int(a["side"]), a["zone_lo"], a["zone_hi"], a["stop"], ai.closed_bars(df, a["tf"]),
                         since=a["created"])
        if st["state"] in ("triggered", "invalid", "expired"):
            icon = {"triggered": "✅", "invalid": "❌", "expired": "⌛"}[st["state"]]
            status = "fired" if st["state"] == "triggered" else st["state"]
            fired.append((a, st["price"], f"{icon} {a['symbol'].replace('USDT', '')} {a['note']}: {st['detail']}",
                          status))
    out = []
    with _db(path) as con:
        for a, px, msg, status in fired:
            con.execute("UPDATE alerts SET status=?, fired_at=?, fired_price=?, message=? WHERE id=?",
                        (status, str(_now()), px, msg, int(a["id"])))
            con.execute("UPDATE alerts SET status='active', created=? WHERE parent=? AND status='waiting'",
                        (str(_now()), int(a["id"])))                      # arm stop / targets once in the zone
            out.append({"id": int(a["id"]), "symbol": a["symbol"], "message": msg})
    if notify:
        for f in out:
            telegram(f["message"])
    return out


def telegram_ready():
    return bool(os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def telegram(text):
    """Send to your Telegram chat when TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set. Never raises."""
    if not telegram_ready():
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/sendMessage",
                          json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text}, timeout=10)
        return r.ok
    except requests.RequestException:
        return False


def main():
    p = argparse.ArgumentParser(description="Price alerts that ring once")
    sub = p.add_subparsers(dest="cmd", required=True)
    ad = sub.add_parser("add", help="add an alert")
    ad.add_argument("symbol")
    ad.add_argument("kind", choices=[k for k in KINDS if k != "confirm"])
    ad.add_argument("level", type=float, help="price, or percent for pct_up / pct_down")
    sub.add_parser("list", help="show active alerts")
    sub.add_parser("check", help="check alerts now")
    a = p.parse_args()
    if a.cmd == "add":
        sym = a.symbol.upper() if a.symbol.upper().endswith("USDT") else a.symbol.upper() + "USDT"
        add(sym, a.kind, a.level)
        print("Alert added.")
    elif a.cmd == "check":
        for f in check():
            print(f["message"])
    for _, r in list_alerts("active").iterrows():
        print(f"  #{r['id']}  {describe(r)}")


if __name__ == "__main__":
    main()
