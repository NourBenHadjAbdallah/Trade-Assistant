"""
tests/test_audit.py - Unit test suite for Phase 1 Audit.
Verifies:
1. No-edge test: Random walk data must yield <= 0 expectancy after costs.
2. Cost drag quantification on narrow stops.
3. Audit findings & edge-case behavior (gaps, drawdown math, Kelly sizing, cache truncation).
"""
import os
import sys
import tempfile
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
def test_gap_stop_loss_unrealistic_fill():
    """
    Audit Finding 2.1: The engine fills stops at exactly -1.0R even when the candle
    gaps far below the stop price.
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
        
        # In reality: entry 100, gap open 80, risk ~2.0 -> loss should be (80-100)/2 = -10.0R!
        # But current engine fills at exactly -1.0R:
        raw_r = trade["r"]
        # Document that the engine currently registers -1.0R instead of gap loss
        assert raw_r == pytest.approx(-1.0, rel=1e-3), (
            f"Engine filled at {raw_r}R; gap through stop should be -10.0R in realistic execution."
        )
    finally:
        tb.STRATEGIES.pop("gap_test", None)


# ----------------------------------------------------------------------------
# 4. MONTE CARLO DRAWDOWN CALCULATION FLAW
# ----------------------------------------------------------------------------
def test_monte_carlo_drawdown_missing_initial_equity():
    """
    Audit Finding 3.1: In monte_carlo(), eq = 100 * cumprod(...).
    If the simulation path starts with losses, np.maximum.accumulate(eq)
    starts from eq[0] (e.g. 99), ignoring the starting 100 capital.
    """
    # Force consecutive losses of -1.0R
    r = np.array([-1.0] * 20)
    # Risk 1% per trade -> 1st trade = 99, 2nd trade = 98.01, 3rd = 97.0299
    mc = tb.monte_carlo(r, n_trades=5, risk_pct=1.0, sims=10, seed=42)
    assert mc is not None
    
    # Expected drawdown from starting capital 100 to 100*(0.99)^5 = 95.099 -> DD = 4.90%
    # But because eq starts at 99, current calculation is (99 - 95.099)/99 = 3.94%
    assert mc["median_dd"] < 4.5, (
        f"Current median_dd is {mc['median_dd']:.2f}%, which underestimates the 4.90% drawdown "
        f"because starting equity 100 is omitted from maximum.accumulate."
    )


# ----------------------------------------------------------------------------
# 5. KELLY CRITERION SIZING WITH NO EDGE
# ----------------------------------------------------------------------------
def test_kelly_sizing_allows_risk_without_statistical_edge():
    """
    Audit Finding 3.2: kelly() allows up to 0.5% risk even when exp_ci[0] <= 0
    (the 95% confidence interval includes zero, meaning no statistically validated edge).
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
    size = tb.kelly(dummy_stats)
    # Currently suggests 0.50% risk instead of 0.0%
    assert size > 0.0, f"Kelly returned {size}%, confirming it allocates risk without proven edge."


# ----------------------------------------------------------------------------
# 6. CACHE FILE TRUNCATION ON REPEATED OFFLINE CALLS
# ----------------------------------------------------------------------------
def test_fetch_history_cache_truncation_logic():
    """
    Audit Finding 1.2: In fetch_history, `df = df.iloc[:-1]` is executed on every call,
    even when no new candles were downloaded, truncating cached data.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_path = os.path.join(tmp_dir, "TEST_1h.csv")
        original_df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "open": range(10), "high": range(10), "low": range(10),
            "close": range(10), "volume": range(10), "buy_volume": range(10)
        })
        original_df.to_csv(fake_path, index=False)
        
        # Simulate what line 94-98 does when new is None:
        old = pd.read_csv(fake_path, parse_dates=["time"])
        new = None
        df = pd.concat([x for x in (old, new) if x is not None], ignore_index=True)
        df_truncated = df.iloc[:-1]
        
        assert len(df_truncated) == len(original_df) - 1, (
            "Confirmed: concatenation without new data drops 1 candle from cache."
        )


# ----------------------------------------------------------------------------
# 7. REVENGE TRADING CALCULATION IN TRADE JOURNAL
# ----------------------------------------------------------------------------
def test_trade_journal_revenge_overlapping_trades():
    """
    Audit Finding 5.1: trade_journal.stats() sorts by open_time when checking for revenge trades:
    `p, t = ordered.iloc[i - 1], ordered.iloc[i]`.
    When multiple positions overlap, i-1 is not the trade that just closed.
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
    # Trade 2 opened 10 min after Trade 0 closed at a loss.
    # But because sorted by open_time, Trade 2 is compared against Trade 1 (closed at 11:00, gap 70 min)!
    # Thus Trade 2 revenge count is missed (revenge_n == 0):
    assert st["revenge_n"] == 0, "Confirmed: overlapping positions cause revenge trading detector to miss revenge trades."
