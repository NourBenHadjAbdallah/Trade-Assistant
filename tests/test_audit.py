"""
tests/test_audit.py - regression tests for the audit findings (AUDIT.md).
Verifies:
1. No-edge test: Random walk data must yield <= 0 expectancy after costs.
2. Cost drag quantification on narrow stops.
3. The audit bugs stay fixed (gap fills, drawdown math, Kelly sizing, cache truncation, revenge trades).
4. Recommendation engine and Claude coach helpers.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution context
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

import trade_backtest as tb
import trade_journal as tj


def generate_random_walk(n=25000, seed=42, sigma=0.005, start_price=100.0, freq="15min"):
    """Generate a clean zero-drift geometric random walk with realistic intra-candle wicks."""
    rng = np.random.default_rng(seed)
    # Ito correction to ensure martingale property (E[S_t] = S_0)
    ret = rng.normal(-0.5 * sigma**2, sigma, n)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.r_[start_price, close[:-1]]
    
    # Intra-candle wicks
    wick_up = np.abs(rng.normal(0, sigma * 0.4, n))
    wick_dn = np.abs(rng.normal(0, sigma * 0.4, n))
    high = np.maximum(open_, close) * (1.0 + wick_up)
    low = np.minimum(open_, close) * (1.0 - wick_dn)
    
    volume = rng.gamma(shape=3.0, scale=100.0, size=n)
    buy_volume = volume * rng.uniform(0.35, 0.65, size=n)
    times = pd.date_range("2024-01-01", periods=n, freq=freq)
    
    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "buy_volume": buy_volume,
    })


# ----------------------------------------------------------------------------
# 1. NO-EDGE TESTS
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("strat_name", list(tb.STRATEGIES.keys()))
def test_random_walk_no_edge_after_costs(strat_name):
    """
    Core integrity test: on random-walk data with zero drift,
    every strategy MUST show roughly zero or negative expectancy after costs.
    If it shows a positive edge after 0.30% round-trip costs, the engine is leaking.
    """
    df = generate_random_walk(n=30000, seed=123, sigma=0.004)
    trades = tb.backtest(df, strat_name, "15m", cost=tb.COST_PER_SIDE)
    assert len(trades) > 10, f"Strategy {strat_name} produced too few trades ({len(trades)}) to evaluate."
    
    mean_r = float(trades["r"].mean())
    # Expectancy after costs must be strictly negative on random noise
    assert mean_r <= 0.0, f"Leakage detected! {strat_name} produced positive expectancy {mean_r:+.4f}R on random walk after costs."


def test_zero_cost_martingale_bound():
    """
    On pure random-walk data without costs (frictionless),
    expectancy should center near 0.0R within reasonable sampling error.
    """
    df = generate_random_walk(n=25000, seed=999, sigma=0.005)
    for strat in ["ema_pullback", "breakout"]:
        trades = tb.backtest(df, strat, "15m", cost=0.0)
        if len(trades) >= 50:
            mean_r = float(trades["r"].mean())
            # Without costs, expectancy on noise must be within +/- 0.15 R (standard sampling error)
            assert abs(mean_r) < 0.15, f"{strat} showed unexpected drift {mean_r:+.4f}R on zero-cost noise."


# ----------------------------------------------------------------------------
# 2. COST DRAG & STOP-WIDTH ANALYSIS
# ----------------------------------------------------------------------------
def test_cost_drag_proportional_to_stop_width():
    """
    Verify the cost formula: cost_R = 2 * cost_per_side * entry / risk.
    When stop is narrow (e.g. 1.5 ATR on 15m ~ 0.6% risk), round-trip 0.30% costs
    consume ~0.45R to ~0.50R per trade.
    """
    entry = 100.0
    cost_per_side = 0.0015  # 0.30% round trip
    
    # 15m narrow stop: risk = 0.60 (0.60% stop distance)
    risk_narrow = 0.60
    cost_r_narrow = 2 * cost_per_side * entry / risk_narrow
    assert pytest.approx(cost_r_narrow, rel=1e-3) == 0.50
    
    # Daily wide stop: risk = 4.0 (4.0% stop distance)
    risk_wide = 4.0
    cost_r_wide = 2 * cost_per_side * entry / risk_wide
    assert pytest.approx(cost_r_wide, rel=1e-3) == 0.075


# ----------------------------------------------------------------------------
# 3. GAP EXECUTION REALISM
# ----------------------------------------------------------------------------
def test_gap_stop_fills_at_open():
    """
    Audit FLAW-02 (fixed): a candle that opens beyond the stop fills at its open, not at the stop.
    """
    times = pd.date_range("2024-01-01", periods=25, freq="1h")
    # 25 bars so ATR(14) is warm.
    # Bar 14: baseline, bar 15: entry at 100, bar 16: flash crash gap down to 80
    opens = [100.0] * 25
    highs = [101.0] * 25
    lows = [99.0] * 25
    closes = [100.0] * 25
    
    opens[16] = 80.0
    highs[16] = 82.0
    lows[16] = 75.0
    closes[16] = 78.0
    
    df = pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000.0] * 25,
        "buy_volume": [500.0] * 25
    })
    
    # Long signal on bar 14 -> enter bar 15 at 100.0, risk ~1.5 -> stop at 98.5
    # Bar 16 opens at 80.0 (gapping down past stop)
    def dummy_strat(df, tf, **_):
        s = np.zeros(len(df), dtype=int)
        s[14] = 1
        return s
    
    try:
        tb.STRATEGIES["gap_test"] = {
            "fn": dummy_strat,
            "default": {"stop_atr": 1.0, "rr": 2.0, "max_hold": 5}
        }
        
        trades = tb.backtest(df, "gap_test", "1h", cost=0.0)
        assert len(trades) == 1
        trade = trades.iloc[0]
        
        # entry 100, ATR 2 -> stop 98; bar 16 opens at 80 -> fill 80 -> (80 - 100) / 2 = -10R
        assert trade["reason"] == "stop"
        assert trade["r"] == pytest.approx(-10.0, rel=1e-3)
    finally:
        tb.STRATEGIES.pop("gap_test", None)


# ----------------------------------------------------------------------------
# 4. MONTE CARLO DRAWDOWN CALCULATION FLAW
# ----------------------------------------------------------------------------
def test_monte_carlo_drawdown_counts_from_starting_equity():
    """
    Audit STAT-04 (fixed): drawdown is measured from the starting balance (100), not the first trade.
    """
    # Force consecutive losses of -1.0R
    r = np.array([-1.0] * 20)
    # Risk 1% per trade -> 1st trade = 99, 2nd trade = 98.01, 3rd = 97.0299
    mc = tb.monte_carlo(r, n_trades=5, risk_pct=1.0, sims=10, seed=42)
    assert mc is not None
    
    # 100 -> 100 * 0.99^5 = 95.099 -> drawdown 4.90%
    assert mc["median_dd"] == pytest.approx(100 * (1 - 0.99 ** 5), rel=1e-6)


# ----------------------------------------------------------------------------
# 5. KELLY CRITERION SIZING WITH NO EDGE
# ----------------------------------------------------------------------------
def test_kelly_zero_without_statistical_edge():
    """
    Audit STAT-03 (fixed): no risk is suggested while the 95% range still includes zero.
    """
    # Sample with positive sample mean (+0.05R) but wide confidence interval [-0.15, +0.25]
    dummy_stats = {
        "n": 50,
        "win_rate": 0.52,
        "avg_win": 1.5,
        "avg_loss": -1.0,
        "expectancy": 0.05,
        "exp_ci": (-0.15, 0.25),  # Lower bound <= 0
    }
    assert tb.kelly(dummy_stats) == 0.0
    proven = {**dummy_stats, "n": 300, "expectancy": 0.25, "exp_ci": (0.05, 0.45)}
    assert 0.0 < tb.kelly(proven) <= 2.0


# ----------------------------------------------------------------------------
# 6. CACHE FILE TRUNCATION ON REPEATED OFFLINE CALLS
# ----------------------------------------------------------------------------
def test_fetch_history_keeps_cache_when_nothing_new(monkeypatch, tmp_path):
    """
    Audit BUG-01 (fixed): a run that downloads nothing must not shave a candle off the cache.
    """
    monkeypatch.setattr(tb, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tb, "_get_klines", lambda params: [])
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h") - pd.Timedelta(hours=2)
    cached = pd.DataFrame({"time": pd.date_range(end=end, periods=10, freq="1h"),
                           "open": range(10), "high": range(10), "low": range(10),
                           "close": range(10), "volume": range(10), "buy_volume": range(10)})
    path = tmp_path / "TEST_1h.csv"
    cached.to_csv(path, index=False)
    for _ in range(3):
        out = tb.fetch_history("TEST", "1h", years=1.0)
    assert len(out) == 10
    assert len(pd.read_csv(path)) == 10


# ----------------------------------------------------------------------------
# 7. REVENGE TRADING CALCULATION IN TRADE JOURNAL
# ----------------------------------------------------------------------------
def test_trade_journal_revenge_overlapping_trades():
    """
    Audit BUG-09 (fixed): a trade opened right after a losing close counts as revenge,
    even when another position overlapped in between.
    """
    trades_data = pd.DataFrame([
        # Trade 0: opens 10:00, closes 12:00 (loss)
        {"open_time": "2024-01-01 10:00", "close_time": "2024-01-01 12:00", "symbol": "BTCUSDT",
         "side": "LONG", "entry": 100, "stop": 90, "exit": 85, "size": 1, "fees": 0, "timeframe": "1h", "strategy": "s"},
        # Trade 1: opens 10:30, closes 11:00 (overlapping trade)
        {"open_time": "2024-01-01 10:30", "close_time": "2024-01-01 11:00", "symbol": "ETHUSDT",
         "side": "LONG", "entry": 100, "stop": 90, "exit": 105, "size": 1, "fees": 0, "timeframe": "1h", "strategy": "s"},
        # Trade 2: opens 12:10 (10 mins after Trade 0 loss -> classic revenge trade!)
        {"open_time": "2024-01-01 12:10", "close_time": "2024-01-01 13:00", "symbol": "BTCUSDT",
         "side": "LONG", "entry": 100, "stop": 90, "exit": 80, "size": 1, "fees": 0, "timeframe": "1h", "strategy": "s"},
    ])
    trades_data["open_time"] = pd.to_datetime(trades_data["open_time"])
    trades_data["close_time"] = pd.to_datetime(trades_data["close_time"])
    st = tj.stats(trades_data)
    assert st is not None
    # Trade 2 opened 10 min after Trade 0 closed at a loss
    assert st["revenge_n"] == 1


# ----------------------------------------------------------------------------
# 8. RECOMMENDATIONS AND CLAUDE COACH
# ----------------------------------------------------------------------------
def test_recommend_and_scan_on_demo_data():
    import trade_ai as ai
    rows = ai.scan(ai.demo_provider, ["AAA", "BBB"], "Swing", workers=2)
    assert len(rows) == 2 and all(r["error"] is None for r in rows)
    assert [ai.scan_key(r) for r in rows] == sorted(ai.scan_key(r) for r in rows)
    rec = rows[0]["rec"]
    assert rec["action"] in ("BUY", "SELL", "WAIT FOR ENTRY", "AVOID", "STAND ASIDE")
    assert rec["max_risk_pct"] <= 1.0 and isinstance(rec["pro"], list) and isinstance(rec["con"], list)


def test_claude_transcript_and_mode(monkeypatch):
    import claude_coach as cc
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "entry?"}]
    t = cc._transcript(msgs)
    assert "User: hi" in t and "Coach: hello" in t and t.endswith("entry?")
    assert cc._transcript(msgs[:1]) == "hi"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert cc.mode() == "api"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(cc, "cli_path", lambda: None)
    assert cc.mode() is None
    with pytest.raises(RuntimeError):
        cc.ask("sys", msgs)
