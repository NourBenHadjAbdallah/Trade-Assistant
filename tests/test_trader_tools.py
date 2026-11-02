"""
tests/test_trader_tools.py - time machine, risk manager, price alerts and entry timer on hand-made data.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

import price_alerts as pa
import risk_manager as rm
import time_machine as tm


# ----------------------------------------------------------------------------
# Time machine: long entry 100, stop 98 (risk 2), T1 104, T2 108; costs off
# ----------------------------------------------------------------------------
def paths(*rows):
    """rows of (high, low, close) per bar for one path each."""
    arr = np.array(rows, dtype=float)               # (paths, bars, 3)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def run_paths(*rows, side=1, levels=(100, 98, 104, 108)):
    h, l, c = paths(*rows)
    return tm.outcomes(side, *levels, h, l, c, cost=0.0)


def test_outcome_kinds_and_r():
    out = run_paths(
        [(101, 97, 97.5), (100, 99, 99.5)],                  # stop first
        [(104.5, 100.5, 104), (103, 99.5, 100)],             # T1 then break-even
        [(104.5, 100.5, 104), (108.5, 103, 108)],            # T1 then T2
        [(105, 97, 101), (110, 100, 109)],                   # stop and T1 in one bar: stop wins
        [(101, 99, 100.5), (102, 99.5, 101)],                # nothing: closed at the time limit
    )
    assert out["kind"].tolist() == [0, 1, 2, 0, 3]
    assert out["r"] == pytest.approx([-1.0, 1.0, 3.0, -1.0, 0.5])
    assert out["t1_first"].tolist() == [False, True, True, False, False]


def test_short_mirrors_long():
    out = run_paths([(99.5, 95.5, 96), (97, 91.5, 92)], side=-1, levels=(100, 102, 96, 92))
    assert out["kind"][0] == 2 and out["r"][0] == pytest.approx(3.0)


def test_fees_lower_every_result():
    h, l, c = paths([(101, 97, 97.5)])
    out = tm.outcomes(1, 100, 98, 104, 108, h, l, c, cost=0.0015)
    assert out["r"][0] == pytest.approx(-1.0 - 2 * 0.0015 * 100 / 2)


def test_simulation_probabilities_add_up():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3000)))
    df = pd.DataFrame({"open": np.r_[100, c[:-1]], "close": c})
    df["high"], df["low"] = df[["open", "close"]].max(axis=1) * 1.001, df[["open", "close"]].min(axis=1) * 0.999
    out = tm.simulate(df, 1, 100, 98, 104, 108, sims=2000)
    p_t1, p_stop = out["t1_first"].mean(), out["stop_first"].mean()
    assert len(out["r"]) == 2000 and 0.5 < p_t1 + p_stop <= 1.0
    assert p_stop > p_t1                                       # the stop is twice as close as T1


# ----------------------------------------------------------------------------
# Risk manager
# ----------------------------------------------------------------------------
def plan(entry=100.0, stop=96.0, t1=108.0, atr=4.0, grade="actionable", direction="LONG", symbol="BTCUSDT"):
    risk = abs(entry - stop)
    return {"symbol": symbol, "direction": direction, "grade": grade, "tf": [{}, {"atr": atr}, {}],
            "setup": {"entry": entry, "stop": stop, "t1": t1, "t2": t1 + risk, "rr1": abs(t1 - entry) / risk}}


NO_TRADES = pd.DataFrame(columns=["time", "symbol", "max_loss"])
RULES = {**rm.DEFAULTS}


def failed(res):
    return {name for name, ok, _ in res["rules"] if not ok}


def test_good_trade_passes_and_is_sized():
    res = rm.check(plan(), RULES, NO_TRADES)
    assert res["verdict"] == "PASS", failed(res)
    z = res["size"]
    assert z["max_loss"] == pytest.approx(10.0)              # 1% of 1000, fees included
    assert z["units"] * 100 == pytest.approx(z["notional"])


@pytest.mark.parametrize("kw, rule", [
    ({"stop": 99.5, "t1": 101}, "Fees affordable"),               # tight stop: fees 0.6R
    ({"atr": 20.0}, "Stop not too tight"),                        # stop only 0.2 ATR
    ({"t1": 104.0}, "Reward:risk"),                               # 1R < 1.5
    ({"grade": "skip"}, "Trade idea"),
    ({"symbol": "DOGEUSDT"}, "Coin allowed"),
])
def test_each_rule_can_block(kw, rule):
    rules = {**RULES, "blocked": ["DOGEUSDT"]}
    res = rm.check(plan(**kw), rules, NO_TRADES)
    assert res["verdict"] == "BLOCK" and rule in failed(res)


def test_daily_limits_block():
    today = pd.DataFrame({"time": [pd.Timestamp.now()] * 3, "symbol": ["A"] * 3, "max_loss": [10.0] * 3})
    res = rm.check(plan(), RULES, today)
    assert {"Daily loss limit", "Trades per day"} <= failed(res)


def test_wrong_side_stop_blocks_without_size():
    res = rm.check(plan(stop=101.0), RULES, NO_TRADES)
    assert res["verdict"] == "BLOCK" and res["size"] is None


def test_settings_round_trip(tmp_path):
    path = tmp_path / "s.json"
    rm.save_settings({"account": "2500", "max_trades_per_day": "5", "blocked": "doge, pepe"}, path)
    s = rm.load_settings(path)
    assert s["account"] == 2500 and s["max_trades_per_day"] == 5 and s["blocked"] == ["DOGE", "PEPE"]
    with pytest.raises(ValueError):
        rm.save_settings({"account": "-1"}, path)


def test_record_taken_counts_today(tmp_path):
    db = str(tmp_path / "s.db")
    p = plan()
    rm.record_taken(p, rm.check(p, RULES, NO_TRADES), db)
    today = rm.taken_today(db)
    assert len(today) == 1 and today["max_loss"].iloc[0] == pytest.approx(10.0)
    blocked = rm.check(plan(grade="skip"), RULES, NO_TRADES)
    with pytest.raises(ValueError):
        rm.record_taken(p, blocked, db)


# ----------------------------------------------------------------------------
# Entry timer: long, zone 99.8-100.2, stop 98
# ----------------------------------------------------------------------------
def bars(*rows, start="2026-01-01 01:00"):
    """rows of (open, high, low, close, buy_share)"""
    t = pd.date_range(start, periods=len(rows), freq="15min")
    return pd.DataFrame([(ti, o, h, l, c, 100.0, 100.0 * b) for ti, (o, h, l, c, b) in zip(t, rows)],
                        columns=["time", "open", "high", "low", "close", "volume", "buy_volume"])


def state(df, since=None, **kw):
    return pa.entry_state(1, 99.8, 100.2, 98.0, df, since=since, **kw)["state"]


def test_entry_timer_states():
    assert state(bars((102, 103, 101, 102, .5))) == "waiting"
    assert state(bars((102, 102, 100, 100.1, .5))) == "in_zone"
    assert state(bars((102, 102, 100, 100.1, .5), (100.1, 101, 100, 100.8, .7))) == "triggered"
    # confirmation candle with sellers in control does not count
    assert state(bars((102, 102, 100, 100.1, .5), (100.1, 101, 100, 100.8, .3))) == "in_zone"
    assert state(bars((102, 102, 99, 97.5, .5))) == "invalid"
    assert state(bars(*[(102, 103, 101, 102, .5)] * 4), since="2026-01-01 00:00", expire_bars=4) == "expired"


# ----------------------------------------------------------------------------
# Price alerts
# ----------------------------------------------------------------------------
def test_alerts_ring_once(tmp_path):
    db = str(tmp_path / "a.db")
    pa.add("BTCUSDT", "above", 100, path=db)
    pa.add("BTCUSDT", "below", 90, path=db)
    pa.add("ETHUSDT", "pct_down", 5, base=200.0, path=db)
    fired = pa.check(prices={"BTCUSDT": 101, "ETHUSDT": 189}, notify=False, path=db)
    assert len(fired) == 2
    assert pa.check(prices={"BTCUSDT": 101, "ETHUSDT": 189}, notify=False, path=db) == []
    assert len(pa.list_alerts("active", db)) == 1


def test_plan_alerts_and_confirmation(tmp_path):
    db = str(tmp_path / "a.db")
    p = {"symbol": "BTCUSDT", "direction": "LONG", "tfs": ["15m", "1h", "4h"],
         "setup": {"zone_lo": 99.8, "zone_hi": 100.2, "stop": 98.0, "t1": 104.0, "t2": 108.0}}
    assert pa.add_plan(p, db) == 5
    with pa._db(db) as con:                                    # pretend the alerts were set before these candles
        con.execute("UPDATE alerts SET created='2026-01-01 00:00:00'")
    candles = bars((102, 102, 100, 100.1, .5), (100.1, 101, 100, 100.8, .7))
    candles = pd.concat([candles, bars((100.8, 101, 100.5, 100.9, .5), start="2099-01-01")])  # still-forming bar
    fired = pa.check(prices={"BTCUSDT": 100.9}, klines=lambda s, tf: candles, notify=False, path=db)
    msgs = " | ".join(f["message"] for f in fired)
    assert "entry zone reached" not in msgs                    # 100.9 is above the zone now
    assert "entry confirmed" in msgs and len(fired) == 1
    assert len(pa.list_alerts("waiting", db)) == 3              # stop, T1, T2 wait for the zone
    with pytest.raises(ValueError):
        pa.add("BTCUSDT", "sideways", 1, path=db)


def test_plan_targets_stay_disarmed_until_entry(tmp_path):
    db = str(tmp_path / "a.db")
    p = {"symbol": "BTCUSDT", "direction": "SHORT", "tfs": ["15m", "1h", "4h"],
         "setup": {"zone_lo": 99.8, "zone_hi": 100.2, "stop": 102.0, "t1": 96.0, "t2": 92.0}}
    pa.add_plan(p, db)
    no_klines = lambda s, tf: bars((90, 91, 89, 90, .5))
    # price already below both targets, but the short was never entered: nothing about targets may ring
    fired = pa.check(prices={"BTCUSDT": 90.0}, klines=no_klines, notify=False, path=db)
    assert not any("target" in f["message"] for f in fired)
    # price rallies into the zone: the zone alert rings and arms stop / T1 / T2
    fired = pa.check(prices={"BTCUSDT": 100.0}, klines=no_klines, notify=False, path=db)
    assert any("entry zone reached" in f["message"] for f in fired)
    assert len(pa.list_alerts("waiting", db)) == 0
    fired = pa.check(prices={"BTCUSDT": 95.5}, klines=no_klines, notify=False, path=db)
    assert [f["message"] for f in fired if "target 1" in f["message"]]
    zone_id = int(pa.list_alerts(path=db).query("note.str.startswith('entry zone')")["id"].iloc[0])
    pa.cancel(zone_id, db)                                      # cancelling a parent also cancels its children


# ----------------------------------------------------------------------------
# Plan stops: ATR based, widened so fees never eat more than MAX_FEE_R
# ----------------------------------------------------------------------------
def test_stop_distance_uses_atr_or_fee_floor():
    import trade_ai as ai
    import trade_backtest as tb
    assert ai.stop_distance(100.0, 2.0) == pytest.approx(ai.STOP_ATR * 2.0)        # wide ATR: ATR rule
    tight = ai.stop_distance(100.0, 0.1)                                            # tiny ATR: fee floor
    assert 2 * tb.COST_PER_SIDE * 100.0 / tight == pytest.approx(ai.MAX_FEE_R)


def test_demo_plans_respect_fee_limit():
    import trade_ai as ai
    import trade_backtest as tb
    for style in ("Intraday", "Swing"):
        p = ai.build_plan(ai.demo_provider, "DEMO", style)
        s = p["setup"]
        if s:
            fee_r = 2 * tb.COST_PER_SIDE * s["entry"] / abs(s["entry"] - s["stop"])
            assert fee_r <= ai.MAX_FEE_R + 1e-9
            assert "Fees affordable" not in failed(rm.check(p, RULES, NO_TRADES))
