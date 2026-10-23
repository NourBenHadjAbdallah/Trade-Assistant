#!/usr/bin/env python3
"""
Strategic assistant: Volume Profile + Anchored VWAP + Order Flow (delta / CVD)

Runs outside TradingView. Install once:
    pip install pandas numpy requests matplotlib

Examples:
    python trade_assistant.py --source binance --symbol BTCUSDT --interval 15m --limit 500
    python trade_assistant.py --source binance --symbol ETHUSDT --interval 5m --anchor "2026-10-22 08:00" --plot
    python trade_assistant.py --source csv --csv mydata.csv
    python trade_assistant.py --source demo            # synthetic data, to test the tool

CSV columns: time, open, high, low, close, volume  (+ optional buy_volume = aggressive buy volume)
Without buy_volume, delta is ESTIMATED from where each candle closed in its range (less reliable).

Order flow note: with Binance data, delta uses real taker-buy volume per candle (aggressor side).
This is candle-level order flow, not a full tick footprint, so it is an approximation.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def fetch_binance(symbol, interval, limit):
    import requests
    r = requests.get(BINANCE_URL, params={"symbol": symbol.upper(), "interval": interval,
                                          "limit": min(limit, 1000)}, timeout=15)
    r.raise_for_status()
    cols = ["time", "open", "high", "low", "close", "volume", "ct", "qv", "trades",
            "buy_volume", "tbq", "ig"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume", "buy_volume"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.attrs["delta_estimated"] = False
    return df[["time", "open", "high", "low", "close", "volume", "buy_volume"]]


def load_csv(path):
    import os
    if not os.path.isfile(path):
        sys.exit(f"CSV file not found: {path}\nCheck the path (use quotes if it has spaces), "
                 f"or use --source binance / --source demo instead.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ["time", "open", "high", "low", "close", "volume"] if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")
    if pd.api.types.is_numeric_dtype(df["time"]):
        unit = "ms" if df["time"].iloc[0] > 1e11 else "s"   # TradingView exports Unix seconds
        df["time"] = pd.to_datetime(df["time"], unit=unit)
    else:
        df["time"] = pd.to_datetime(df["time"])
    est = "buy_volume" not in df.columns
    if est:
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        clv = ((df["close"] - df["low"]) / rng).fillna(0.5)
        df["buy_volume"] = df["volume"] * clv
    df.attrs["delta_estimated"] = est
    return df[["time", "open", "high", "low", "close", "volume", "buy_volume"]]


def demo_data(n=400, seed=7):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-20", periods=n, freq="15min")
    drift = np.sin(np.linspace(0, 6, n)) * 0.0006
    ret = rng.normal(drift, 0.002)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.gamma(3, 100, n)
    buy_share = np.clip(0.5 + (close - open_) / open_ * 120 + rng.normal(0, 0.08, n), 0.05, 0.95)
    df = pd.DataFrame({"time": t, "open": open_, "high": high, "low": low,
                       "close": close, "volume": vol, "buy_volume": vol * buy_share})
    df.attrs["delta_estimated"] = False
    return df


# ----------------------------------------------------------------------------
# Volume profile
# ----------------------------------------------------------------------------
def volume_profile(df, bins=60, va_pct=0.70):
    lo, hi = df["low"].min(), df["high"].max()
    if hi == lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    step = edges[1] - edges[0]
    vol = np.zeros(bins)
    buy = np.zeros(bins)
    for h, l, v, b in zip(df["high"], df["low"], df["volume"], df["buy_volume"]):
        i0 = min(int((l - lo) / step), bins - 1)
        i1 = min(int((h - lo) / step), bins - 1)
        n = i1 - i0 + 1
        vol[i0:i1 + 1] += v / n          # volume spread evenly across the candle's range
        buy[i0:i1 + 1] += b / n
    delta = buy - (vol - buy)

    poc_i = int(np.argmax(vol))
    lo_i = hi_i = poc_i
    acc, target = vol[poc_i], va_pct * vol.sum()
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        dn = vol[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn

    smooth = np.convolve(vol, np.ones(3) / 3, mode="same")
    mean = vol.mean()
    hvn, lvn = [], []
    for i in range(1, bins - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 1.2 * mean and i != poc_i:
            hvn.append(centers[i])
        if smooth[i] < smooth[i - 1] and smooth[i] <= smooth[i + 1] and smooth[i] < 0.6 * mean:
            lvn.append(centers[i])
    return {"edges": edges, "centers": centers, "vol": vol, "delta": delta,
            "poc": centers[poc_i], "vah": edges[hi_i + 1], "val": edges[lo_i],
            "hvn": hvn, "lvn": lvn}


# ----------------------------------------------------------------------------
# Anchored VWAP
# ----------------------------------------------------------------------------
def anchored_vwap(df, anchor_idx):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).iloc[anchor_idx:].cumsum()
    v = df["volume"].iloc[anchor_idx:].cumsum()
    out = pd.Series(np.nan, index=df.index)
    out.iloc[anchor_idx:] = pv / v
    return out


def last_pivots(df, n=5):
    hi_idx = lo_idx = None
    for i in range(len(df) - n - 1, n - 1, -1):
        if hi_idx is None and df["high"].iloc[i] == df["high"].iloc[i - n:i + n + 1].max():
            hi_idx = i
        if lo_idx is None and df["low"].iloc[i] == df["low"].iloc[i - n:i + n + 1].min():
            lo_idx = i
        if hi_idx is not None and lo_idx is not None:
            break
    return hi_idx, lo_idx


def build_avwaps(df, custom_anchor=None):
    anchors = {
        "AVWAP swing high (range)": int(df["high"].values.argmax()),
        "AVWAP swing low (range)": int(df["low"].values.argmin()),
        "AVWAP max-volume bar": int(df["volume"].values.argmax()),
    }
    ph, pl = last_pivots(df)
    if ph is not None:
        anchors["AVWAP last pivot high"] = ph
    if pl is not None:
        anchors["AVWAP last pivot low"] = pl
    if custom_anchor:
        idx = int(df["time"].searchsorted(pd.Timestamp(custom_anchor)))
        idx = min(idx, len(df) - 1)
        anchors[f"AVWAP custom ({custom_anchor})"] = idx
    return {name: (idx, anchored_vwap(df, idx)) for name, idx in anchors.items()}


# ----------------------------------------------------------------------------
# Order flow (candle-level delta / CVD)
# ----------------------------------------------------------------------------
def order_flow(df, lookback=30):
    d = df.copy()
    d["sell_volume"] = d["volume"] - d["buy_volume"]
    d["delta"] = d["buy_volume"] - d["sell_volume"]
    d["cvd"] = d["delta"].cumsum()
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["clv"] = ((d["close"] - d["low"]) / rng).fillna(0.5)

    last = d.tail(10)
    bias = last["delta"].sum() / max(last["volume"].sum(), 1e-9)   # -1 .. +1

    # CVD divergence vs price over lookback
    w = d.tail(lookback)
    divergence = None
    if w["high"].iloc[-1] >= w["high"].max() * 0.9995 and w["cvd"].iloc[-1] < w["cvd"].max() * 0.98 + 0.02 * w["cvd"].min():
        divergence = "bearish"
    elif w["low"].iloc[-1] <= w["low"].min() * 1.0005 and w["cvd"].iloc[-1] > w["cvd"].min() * 0.98 + 0.02 * w["cvd"].max():
        divergence = "bullish"

    # Absorption: big volume, strong one-sided delta, but price moves the other way / holds
    avg_vol = d["volume"].rolling(20, min_periods=5).mean()
    events = []
    for i in range(max(len(d) - 12, 0), len(d)):
        r = d.iloc[i]
        if avg_vol.iloc[i] and r["volume"] > 1.5 * avg_vol.iloc[i] and abs(r["delta"]) / r["volume"] > 0.20:
            if r["delta"] < 0 and r["clv"] > 0.5:
                events.append((r["time"], r["close"], "buy absorption (sellers hit, price held)"))
            elif r["delta"] > 0 and r["clv"] < 0.5:
                events.append((r["time"], r["close"], "sell absorption (buyers lifted, price rejected)"))
    return d, {"bias": bias, "divergence": divergence, "absorption": events}


# ----------------------------------------------------------------------------
# Strategic assistant
# ----------------------------------------------------------------------------
def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n, min_periods=3).mean().iloc[-1])


def cluster_levels(levels, tol):
    levels = sorted(levels, key=lambda x: x[1])
    clusters, cur = [], [levels[0]]
    for lv in levels[1:]:
        if lv[1] - cur[-1][1] <= tol:
            cur.append(lv)
        else:
            clusters.append(cur)
            cur = [lv]
    clusters.append(cur)
    return [{"price": float(np.mean([p for _, p in c])), "names": [n for n, _ in c], "count": len(c)}
            for c in clusters]


def analyze(df, vp, avwaps, of, of_df):
    price = float(df["close"].iloc[-1])
    a = atr(df)
    notes, score = [], 0.0

    # 1) Location vs value
    if price > vp["vah"]:
        loc, s = "ABOVE value area (acceptance higher or extended)", 1.0
    elif price < vp["val"]:
        loc, s = "BELOW value area (acceptance lower or extended)", -1.0
    else:
        loc, s = "INSIDE value area (balance / rotational)", (0.5 if price > vp["poc"] else -0.5)
    score += s
    notes.append(f"Location: {loc}; POC {vp['poc']:.4f} is {(price - vp['poc']) / a:+.1f} ATR away.")

    # 2) AVWAP stance
    above = [n for n, (_, s_) in avwaps.items() if price > s_.iloc[-1]]
    below = [n for n in avwaps if n not in above]
    score += 1.5 * (len(above) - len(below)) / len(avwaps)
    notes.append(f"AVWAP: price above {len(above)} / below {len(below)} of {len(avwaps)} anchors.")

    # 3) Order flow
    b = of["bias"]
    score += float(np.clip(b * 4, -1, 1))
    notes.append(f"Order flow: net delta over last 10 bars = {b:+.1%} of volume "
                 f"({'buyers' if b > 0 else 'sellers'} more aggressive).")
    if of["divergence"]:
        score += -1 if of["divergence"] == "bearish" else 1
        notes.append(f"CVD divergence: {of['divergence'].upper()} (price extreme not confirmed by delta).")
    for t, p, what in of["absorption"][-3:]:
        score += 0.5 if what.startswith("buy") else -0.5
        notes.append(f"Absorption {t:%m-%d %H:%M} @ {p:.4f}: {what}.")

    # 4) Levels and confluence
    levels = [("POC", vp["poc"]), ("VAH", vp["vah"]), ("VAL", vp["val"])]
    levels += [("HVN", p) for p in vp["hvn"]] + [("LVN", p) for p in vp["lvn"]]
    levels += [(n, s_.iloc[-1]) for n, (_, s_) in avwaps.items()]
    clusters = cluster_levels(levels, tol=0.3 * a)
    sup = [c for c in clusters if c["price"] < price]
    res = [c for c in clusters if c["price"] > price]

    def pick(cands):
        cands = [c for c in cands if abs(c["price"] - price) <= 3 * a]
        return max(cands, key=lambda c: c["count"] - 0.3 * abs(c["price"] - price) / a, default=None)

    scen = []
    s_zone, r_zone = pick(sup), pick(res)
    next_up = min(res, key=lambda c: c["price"] - price, default=None)
    next_dn = max(sup, key=lambda c: c["price"], default=None)
    if s_zone:
        tgt = next_up["price"] if next_up else None
        stop = s_zone["price"] - 0.5 * a
        scen.append(("LONG idea (buy the pullback)", s_zone, stop, tgt,
                     "Needs: absorption / positive delta / CVD holding at the zone. No flow confirmation = no trade."))
    if r_zone:
        tgt = next_dn["price"] if next_dn else None
        stop = r_zone["price"] + 0.5 * a
        scen.append(("SHORT idea (sell the rally)", r_zone, stop, tgt,
                     "Needs: sell absorption / negative delta / bearish CVD divergence at the zone."))
    return {"price": price, "atr": a, "score": score, "notes": notes,
            "clusters": clusters, "scenarios": scen}


def label(score):
    if score >= 2.5: return "STRONG BULLISH lean"
    if score >= 1.0: return "bullish lean"
    if score > -1.0: return "NEUTRAL / mixed"
    if score > -2.5: return "bearish lean"
    return "STRONG BEARISH lean"


def report(df, vp, avwaps, of, res, symbol):
    line = "=" * 64
    print(line)
    print(f" STRATEGIC ASSISTANT  |  {symbol}  |  {len(df)} candles")
    print(f" {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    if df.attrs.get("delta_estimated"):
        print(" ! Delta is ESTIMATED from candle close position (no aggressor data).")
    print(line)
    print(f" Price {res['price']:.4f}   ATR(14) {res['atr']:.4f}")
    print(f" Bias score {res['score']:+.1f}  ->  {label(res['score'])}\n")

    print("VOLUME PROFILE")
    print(f"  VAH {vp['vah']:.4f} | POC {vp['poc']:.4f} | VAL {vp['val']:.4f}")
    print(f"  HVN: {', '.join(f'{p:.4f}' for p in vp['hvn']) or '-'}")
    print(f"  LVN: {', '.join(f'{p:.4f}' for p in vp['lvn']) or '-'}\n")

    print("ANCHORED VWAP")
    for n, (idx, s) in avwaps.items():
        v = s.iloc[-1]
        print(f"  {n:<34} {v:.4f}  ({'above' if res['price'] > v else 'below'}, anchor {df['time'].iloc[idx]:%m-%d %H:%M})")

    print("\nREAD")
    for n in res["notes"]:
        print("  - " + n)

    print("\nCONFLUENCE ZONES (2+ levels stacked)")
    zones = [c for c in res["clusters"] if c["count"] >= 2]
    for c in sorted(zones, key=lambda c: abs(c["price"] - res["price"]))[:6]:
        side = "above" if c["price"] > res["price"] else "below"
        print(f"  {c['price']:.4f} ({side}, {abs(c['price'] - res['price']) / res['atr']:.1f} ATR) <- {', '.join(c['names'])}")
    if not zones:
        print("  none close enough")

    print("\nSCENARIOS (reference levels, not signals)")
    for name, zone, stop, tgt, cond in res["scenarios"]:
        print(f"  {name}")
        print(f"    zone {zone['price']:.4f}  [{', '.join(zone['names'])}]")
        print(f"    invalidation {stop:.4f}" + (f"   first target {tgt:.4f}" if tgt else ""))
        if tgt:
            risk = abs(zone["price"] - stop)
            if risk > 0:
                print(f"    approx R:R {abs(tgt - zone['price']) / risk:.1f}")
        print(f"    {cond}")
    if not res["scenarios"]:
        print("  no clean zone within 3 ATR; wait for price to reach a level.")
    print("\nEducational tool, not financial advice. Backtest before risking money.")


def plot(df, vp, avwaps, of_df, symbol):
    import matplotlib.pyplot as plt
    fig, (ax, ax2, axv) = plt.subplots(3, 1, figsize=(13, 9), sharex=False,
                                       gridspec_kw={"height_ratios": [3, 1, 0.01]})
    axv.remove()
    ax.plot(df["time"], df["close"], color="black", lw=1, label="close")
    for n, (_, s) in avwaps.items():
        ax.plot(df["time"], s, lw=1.1, label=n)
    for name, y, c in [("POC", vp["poc"], "red"), ("VAH", vp["vah"], "gray"), ("VAL", vp["val"], "gray")]:
        ax.axhline(y, color=c, ls="--", lw=0.8)
        ax.text(df["time"].iloc[0], y, name, color=c, va="bottom", fontsize=8)
    for y in vp["hvn"]:
        ax.axhline(y, color="green", ls=":", lw=0.6)
    for y in vp["lvn"]:
        ax.axhline(y, color="orange", ls=":", lw=0.6)
    inset = ax.inset_axes([0.0, 0.0, 0.12, 1.0], sharey=ax)
    inset.barh(vp["centers"], vp["vol"], height=vp["centers"][1] - vp["centers"][0], color="steelblue", alpha=0.5)
    inset.axis("off")
    ax.set_title(f"{symbol}: volume profile + AVWAP")
    ax.legend(fontsize=7, loc="upper right")
    ax2.bar(of_df["time"], of_df["delta"], color=np.where(of_df["delta"] >= 0, "g", "r"), width=0.0007 * 30)
    ax2b = ax2.twinx()
    ax2b.plot(of_df["time"], of_df["cvd"], color="blue", lw=1)
    ax2.set_title("delta (bars) and CVD (line)")
    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Volume profile + AVWAP + order flow assistant")
    p.add_argument("--source", choices=["binance", "csv", "demo"], default="demo")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--csv")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--anchor", help='custom AVWAP anchor, e.g. "2026-10-22 08:00"')
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    if a.source == "binance":
        df = fetch_binance(a.symbol, a.interval, a.limit)
    elif a.source == "csv":
        if not a.csv:
            sys.exit("--csv path required")
        df = load_csv(a.csv)
    else:
        df = demo_data()
        a.symbol = "DEMO"

    vp = volume_profile(df, bins=a.bins)
    avwaps = build_avwaps(df, a.anchor)
    of_df, of = order_flow(df)
    res = analyze(df, vp, avwaps, of, of_df)
    report(df, vp, avwaps, of, res, a.symbol)
    if a.plot:
        plot(df, vp, avwaps, of_df, a.symbol)


if __name__ == "__main__":
    main()