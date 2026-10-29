"""
tests/test_signal_log.py - the track record's grading rules, on hand-made candles.
Long signal: entry 100, stop 98 (risk 2), T1 104 (2R), T2 108 (4R). Costs off unless stated.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

import signal_log as sl

CREATED = pd.Timestamp("2026-01-01 00:30")
SIG = {"side": 1, "entry": 100.0, "stop": 98.0, "t1": 104.0, "t2": 108.0, "in_zone": 0, "created": CREATED}


def candles(*bars, start="2026-01-01 01:00"):
    """bars: (open, high, low, close)"""
    t = pd.date_range(start, periods=len(bars), freq="1h")
    return pd.DataFrame([(ti, *b) for ti, b in zip(t, bars)], columns=["time", "open", "high", "low", "close"])


def g(c, sig=SIG, **kw):
    return sl.grade(sig, c, cost=0.0, **kw)


def test_waits_then_expires_without_fill():
    c = candles(*[(102, 103, 101, 102)] * 3)
    assert g(c)["status"] == "pending"
    assert g(c, expire_bars=3)["status"] == "expired"


def test_stop_is_full_loss():
    out = g(candles((102, 102, 99.5, 100), (100, 100.5, 97, 97.5)))
    assert out["status"] == "closed" and out["result"] == "stop"
    assert out["r"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    out = g(candles((101, 101, 99.5, 100), (95, 96, 94, 95)))
    assert out["r"] == pytest.approx((95 - 100) / 2)          # -2.5R, not -1R


def test_t1_then_break_even():
    out = g(candles((101, 101, 99.5, 100.5), (101, 104.5, 100.5, 104), (103, 103, 99.5, 100)))
    assert out["result"] == "T1 then break-even"
    assert out["r"] == pytest.approx(0.5 * 2.0)                # half at +2R, half at 0


def test_t1_then_t2():
    out = g(candles((101, 101, 99.5, 100.5), (101, 104.5, 100.5, 104), (104, 108.5, 103, 108)))
    assert out["result"] == "T2"
    assert out["r"] == pytest.approx(0.5 * 2.0 + 0.5 * 4.0)    # +3R


def test_stop_wins_when_stop_and_target_share_a_candle():
    out = g(candles((101, 101, 99.5, 100.5), (100, 105, 97, 101)))
    assert out["result"] == "stop" and out["r"] == pytest.approx(-1.0)


def test_in_zone_fills_at_next_open_and_ignores_older_candles():
    old = candles((50, 50, 1, 50), start="2025-12-31 22:00")    # before the signal: must be ignored
    c = pd.concat([old, candles((100.5, 101, 100, 100.5), (101, 104.5, 100.5, 104), (104, 108.5, 103, 108))])
    out = g(c, {**SIG, "in_zone": 1})
    assert out["fill"] == 100.5 and out["result"] == "T2"


def test_time_limit_closes_at_market():
    c = candles((101, 101, 99.5, 100.5), *[(101, 102, 100.5, 101)] * 5)
    out = g(c, max_hold=4)
    assert out["result"] == "time limit"
    assert out["r"] == pytest.approx((101 - 100) / 2)


def test_short_signal_mirrors_long():
    short = {"side": -1, "entry": 100.0, "stop": 102.0, "t1": 96.0, "t2": 92.0, "in_zone": 0, "created": CREATED}
    out = g(candles((99, 100.5, 98.5, 99), (99, 99.5, 95.5, 96), (96, 97, 91.5, 92)), short)
    assert out["result"] == "T2" and out["r"] == pytest.approx(3.0)


def test_fees_reduce_r():
    c = candles((101, 101, 99.5, 100), (100, 100.5, 97, 97.5))
    out = sl.grade(SIG, c, cost=0.0015)
    assert out["r"] == pytest.approx(-1.0 - 2 * 0.0015 * 100 / 2)


def test_record_dedupes_and_summary(tmp_path):
    db = str(tmp_path / "s.db")
    plan = {"style": "Intraday", "tfs": ["15m", "1h", "4h"], "direction": "LONG",
            "tf": [{"price": 105.0}, {"price": 101.0}, {"price": 99.0}],
            "setup": {"entry": 100.0, "zone_lo": 99.8, "zone_hi": 100.2, "stop": 98.0, "t1": 104.0, "t2": 108.0,
                      "in_zone": False, "confluence": [("x", True)] * 9, "confluence_met": 7}}
    row = {"symbol": "BTCUSDT", "plan": plan, "rec": {"grade": "actionable", "score": 62}}
    skip = {"symbol": "ETHUSDT", "plan": plan, "rec": {"grade": "skip", "score": 30}}
    assert sl.record([row, skip], db) == 1
    assert sl.record([row], db) == 0                             # same idea still open: not logged twice
    df = sl.load(db)
    assert len(df) == 1 and df.loc[0, "price"] == 101.0 and df.loc[0, "checklist"] == "7/9"
    s = sl.summary(df)
    assert s["all"]["signals"] == 1 and s["all"]["pending"] == 1 and "Too early" in s["verdict"]
    assert "SIGNAL TRACK RECORD" in sl.report_text(s)
