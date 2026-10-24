# Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review

**Project:** Crypto Trading Assistant & Backtest Engine  
**Reviewer:** Senior Quantitative Developer & Systems Engineer  
**Date:** October 2026  
**Status:** Phase 1 Complete (Audit & Verification)  

---

## 1. Executive Summary & Verification of Prior Assumptions

A rigorous code audit of the entire repository (`trade_backtest.py`, `trade_assistant.py`, `trade_assistant_gui.py`, `trade_ai.py`, `trade_chat.py`, `trade_journal.py`, `trade_replay.py`, `news_alerts.py`, and `app.py`) was conducted.

### Verification of Baseline Knowledge
1. **LTCUSDT 15m Over 2 Years (70,075 Candles):**
   - **Empirical Check:** Verified directly against cached Binance data.
     - `ema_pullback`: $N=2927$, Expectancy with cost = **$-0.482$R**, without cost = **$-0.014$R**, Cost drag = **$0.468$R**
     - `breakout`: $N=2397$, Expectancy with cost = **$-0.523$R**, without cost = **$-0.074$R**, Cost drag = **$0.448$R**
     - `mean_reversion`: $N=507$, Expectancy with cost = **$-0.460$R**, without cost = **$+0.013$R**, Cost drag = **$0.473$R**
     - `htf_trend`: $N=1972$, Expectancy with cost = **$-0.509$R**, without cost = **$-0.020$R**, Cost drag = **$0.489$R**
   - **Conclusion:** Confirmed. The strategies themselves have zero or slightly negative edge before costs ($\approx 0.0\text{R}$), but because round-trip costs are $0.30\%$ ($0.15\%$ per side) and the $1.5\times\text{ATR}$ stop distance on $15\text{m}$ is only $\approx 0.60\%-0.65\%$ of price, **the cost drag in R multiples is mathematically:**
     $$\text{Cost Drag (in R)} = \frac{2 \times \text{Cost}}{\text{Stop Distance \%}} = \frac{0.0030}{0.0065} \approx 0.46\text{R}$$
     Every trade starts nearly half an R in the hole. On $15\text{m}$, fees and slippage eat the strategy alive.
2. **LTCUSDT 1D Over 2 Years (729 Candles):**
   - **Empirical Check:**
     - `ema_pullback`: $N=24$ trades
     - `breakout`: $N=23$ trades
     - `mean_reversion`: $N=6$ trades
     - `htf_trend`: $N=16$ trades
   - **Conclusion:** Confirmed. Sample sizes ($6-24$ trades) are far too small for statistical significance. Standard errors ($\sigma / \sqrt{N}$) are $\pm 0.35\text{R}$ to $\pm 0.70\text{R}$, making any observed positive result indistinguishable from random noise.
3. **Requirement for Pooled Tests:**
   - Single-coin high-timeframe tests lack trade counts; single-coin low-timeframe tests suffer severe fee drag. The only statistically sound path forward is **pooled multi-coin testing** on higher timeframes ($1\text{h}$, $4\text{h}$) to gather $300+$ trades across market regimes without incurring fatal cost drag.

---

## 2. Ranked Findings: Bugs, Biases, and Risks

| Severity | ID | Category | Description | Primary File Affected |
| :--- | :--- | :--- | :--- | :--- |
| **CRITICAL** | **BUG-01** | Lookahead & Cache | `fetch_history()` drops last candle on cached calls without new data, progressively corrupting cache | `trade_backtest.py` |
| **CRITICAL** | **FLAW-02** | Execution Realism | Stops fill at exact stop price ($-1.0\text{R}$) even when price gaps far below stop price | `trade_backtest.py` |
| **CRITICAL** | **STAT-03** | Risk & Position Sizing | Kelly criterion suggests risking capital ($0.50\%$) when confidence interval lower bound $\le 0$ (no edge) | `trade_backtest.py` |
| **CRITICAL** | **STAT-04** | Statistical Math | `monte_carlo()` drawdown calculation omits starting capital peak ($100.0$), understating drawdowns | `trade_backtest.py` |
| **HIGH** | **BIAS-05** | Survivorship Bias | `load_coins()` selects universe based on *current* 24h volume, introducing survivorship bias into historical pools | `trade_assistant_gui.py`, `app.py` |
| **HIGH** | **BIAS-06** | Multiple Testing | Bonferroni penalty is incorrectly applied to the single Out-of-Sample evaluation in `report_optimize()` | `trade_backtest.py` |
| **HIGH** | **REAL-07** | Cost Modeling | Engine ignores perpetual futures funding rates ($8\text{h}$ intervals), which can erode $10\%-40\%$ p.a. | `trade_backtest.py` |
| **HIGH** | **STAT-08** | Statistical Validity | Monte Carlo I.I.D. bootstrap destroys serial autocorrelation, streak persistence, and volatility clustering | `trade_backtest.py` |
| **MEDIUM** | **BUG-09** | Logic Bug | `trade_journal.py` revenge trade detection logic is broken when positions overlap in time | `trade_journal.py` |
| **MEDIUM** | **BUG-10** | System / Encoding | `news_alerts.py` opens state file without `encoding="utf-8"`, crashing on Windows with Unicode emojis | `news_alerts.py` |
| **MEDIUM** | **FLAW-11** | Lookahead / Timing | `s_htf_trend` resample aligns irregularly on arbitrary intervals and lacks strict point-in-time timestamping | `trade_backtest.py` |
| **MEDIUM** | **LEAK-12** | Synthetic Drift | `demo_history` generates GBM without Ito convexity adjustment ($-\frac{1}{2}\sigma^2$), introducing artificial upward drift | `trade_backtest.py` |
| **MEDIUM** | **INT-13** | Integration Gap | LLM coach in `trade_chat.py` is blind to backtest expectancy and quantitative trade statistics | `trade_chat.py` |
| **LOW** | **PERF-14** | Performance | Pure Python simulation loop and redundant indicator recalculations inside parameter grid searches | `trade_backtest.py` |
| **LOW** | **ARCH-15** | Code Duplication | Indicators (`ema`, `atr`, `rsi`) are duplicated across `trade_backtest.py`, `trade_ai.py`, `trade_assistant.py` | Repository-wide |

---

## 3. Deep-Dive Analysis of Identified Issues

### 3.1 Lookahead Bias & Data Integrity

#### [BUG-01] Cache File Truncation Bug in `fetch_history()`
- **Location:** `trade_backtest.py`, lines 94-99:
  ```python
  df = pd.concat([x for x in (old, new) if x is not None], ignore_index=True)
  ...
  df = df.iloc[:-1]  # drop the still-forming last candle
  df.to_csv(path, index=False)
  ```
- **The Issue:** When `fetch_history()` is called with `use_cache=True` and no new completed candles are returned by the API (`new is None`), `df` is identical to `old`. Line 98 drops the final row unconditionally and overwrites the CSV file. Every subsequent offline or idle run permanently shaves off one candle from the historical dataset.
- **Fix:** Only execute `df = df.iloc[:-1]` if `new is not None and len(new) > 0`.

#### [FLAW-11] HTF Resampling & Alignment Bias in `s_htf_trend`
- **Location:** `trade_backtest.py`, lines 189-203:
  ```python
  g = df.set_index("time")
  rule = f"{TF_MIN[tf] * htf_factor}min"
  hc = g["close"].resample(rule).last().dropna()
  e = ema(hc, htf_ema)
  up = ((hc > e) & (e > e.shift(3))).shift(1)
  up = up.reindex(g.index, method="ffill").fillna(False).astype(bool).values
  ```
- **The Issue:**
  1. `resample(rule).last()` indexes buckets by the **start** (left edge) of the resampled period.
  2. While `.shift(1)` shifts the value by one HTF bar forward (which prevents direct same-period lookahead), if the underlying data has gaps (e.g. maintenance downtime or weekend liquidity drops), `.dropna()` followed by `.shift(1)` in row space causes HTF bars to align to incorrect wall-clock timeframes when reindexed with `ffill`.
  3. When `tf="4h"` and `htf_factor=3` ($12\text{h}$), resample boundaries can drift relative to standard exchange daily boundaries ($00:00$ UTC).
- **Fix:** Construct higher timeframe bars using closed HTF candle timestamps directly (or explicit floor alignment `time.dt.floor()`), avoiding pandas index-space `dropna().shift(1)` shifts over discontinuous time.

#### [LEAK-12] Convexity Drift in Synthetic History (`demo_history`)
- **Location:** `trade_backtest.py`, line 115:
  ```python
  close = 100 * np.exp(np.cumsum(rng.normal(drift, vol)))
  ```
- **The Issue:** For Geometric Brownian Motion, $E[e^{\sigma W_t}] = e^{\frac{1}{2}\sigma^2 t}$. Generating prices with `exp(cumsum(normal(0, vol)))` without subtracting the Ito drift correction $-\frac{1}{2}\text{vol}^2$ induces an artificial positive upward drift proportional to volatility ($\approx +0.02\%$ per bar). Over a 2-year simulation, this creates an unearned upward price trend that inflates long trend-following strategies.
- **Fix:** Subtract $0.5 \times \text{vol}^2$ from the log return distribution.

---

### 3.2 Execution Realism & Cost Modeling

#### [FLAW-02] Unrealistic Gap and Stop-Loss Fill Handling
- **Location:** `trade_backtest.py`, lines 252-256:
  ```python
  hit_stop = l[j] <= stop if side > 0 else h[j] >= stop
  if hit_stop:
      r, reason = -1.0, "stop"
      break
  ```
- **The Issue:** If a candle opens below the stop price (e.g., bar $j$ opens at $80$ when stop was $98$), the engine registers `r = -1.0R`. In reality, a stop is a stop-market order; when triggered by a gap opening, the fill price is the open price $o[j]$ (or worse). Filling at the stop price rather than $\le \min(open, stop)$ masks flash crashes and tail risk.
- **Fix:**
  ```python
  if hit_stop:
      exit_price = min(o[j], stop) if side > 0 else max(o[j], stop)
      r = -1.0 - side * (stop - exit_price) / risk
      reason = "stop"
      break
  ```

#### [REAL-07] Absence of Funding Costs for Perpetual Futures
- **Location:** `trade_backtest.py`, line 262.
- **The Issue:** The engine applies a fixed entry and exit cost (`COST_PER_SIDE = 0.0015`), but does not model funding rates. On Binance perpetual futures, funding is paid/received every 8 hours. Holding a long position during bull markets incurs an average funding fee of $+0.01\%$ to $+0.03\%$ per 8 hours ($10\%$ to $30\%$ annualized). A 4-hour trade held for 60 bars lasts 10 days (30 funding intervals), generating an unmodeled $0.30\%-0.90\%$ fee drag.
- **Fix:** Add a funding cost module that tracks holding duration across 8-hour boundaries ($00:00$, $08:00$, $16:00$ UTC).

---

### 3.3 Statistical Flaws & Anti-Overfitting Misconfigurations

#### [STAT-03] Kelly Criterion Sizing With No Statistical Edge
- **Location:** `trade_backtest.py`, lines 331-333:
  ```python
  if s["n"] < 100 or s["exp_ci"][0] <= 0:
      cap = min(cap, 0.5)
  return float(min(max(k, 0) * fraction * 100, cap))
  ```
- **The Issue:** When `s["exp_ci"][0] <= 0` (the lower bound of the 95% confidence interval is negative or zero, meaning the observed positive sample mean is not statistically significant), the function caps risk at $0.50\%$ instead of setting it to $0.0\%$. Recommending real money risk when an edge is statistically unproven violates the core mandate of quantitative risk management.
- **Fix:**
  ```python
  if s["n"] < 30 or s["exp_ci"][0] <= 0 or s["expectancy"] <= 0:
      return 0.0
  ```

#### [STAT-04] Drawdown Calculation Flaw in `monte_carlo()`
- **Location:** `trade_backtest.py`, lines 316-318:
  ```python
  eq = 100 * np.cumprod(np.maximum(1 + paths * risk_pct / 100, 0), axis=1)
  final = eq[:, -1]
  dd = ((np.maximum.accumulate(eq, axis=1) - eq) / np.maximum.accumulate(eq, axis=1)).max(axis=1) * 100
  ```
- **The Issue:** `eq` has shape `(sims, n_trades)`. It does not include the initial capital ($100.0$) at $t=0$. If a simulated path opens with consecutive losses (e.g. equity drops $100 \to 99 \to 98.01$), `np.maximum.accumulate(eq)` begins at $99.0$. The drawdown from starting capital ($100 \to 98.01 = 1.99\%$) is calculated as $(99 - 98.01)/99 = 1.00\%$. Initial capital drawdowns are systematically underreported.
- **Fix:** Prepend an initial column of $100.0$ to `eq`:
  ```python
  eq_full = np.column_stack([np.full(sims, 100.0), eq])
  peaks = np.maximum.accumulate(eq_full, axis=1)
  dd = ((peaks - eq_full) / peaks).max(axis=1) * 100
  ```

#### [BIAS-06] Bonferroni Over-Penalization on Out-of-Sample Results
- **Location:** `trade_backtest.py`, line 467:
  ```python
  L.append(fmt_stats(*analyse(res["oos"]["r"].values, res["trials"], risk_pct), title="OUT-OF-SAMPLE"))
  ```
- **The Issue:** In `split_test()`, parameter optimization occurs on the in-sample (train) portion. The single best parameter set is then selected and evaluated on the out-of-sample (test) portion. Passing `trials = res["trials"]` (e.g. 54) into `analyse()` for the OOS data applies a 54-fold Bonferroni penalty to an evaluation that was only run **once**. Bonferroni adjusts for in-sample data mining; applying it to true holdout data falsely widens confidence intervals and rejects valid out-of-sample performance.
- **Fix:** Evaluate OOS data with `trials = 1` (or use Family-Wise Error Rate adjustments strictly on the training selection stage).

#### [STAT-08] Monte Carlo Bootstrap Destroys Serial Correlation
- **Location:** `trade_backtest.py`, line 315:
  ```python
  paths = rng.choice(r, size=(sims, n_trades), replace=True)
  ```
- **The Issue:** Standard I.I.D. bootstrap sampling uniformly scrambles trade order. In crypto markets, returns exhibit volatility clustering, regime persistence, and loss autocorrelation. An I.I.D. bootstrap underestimates the true probability of extended losing streaks and ruin drawdowns.
- **Fix:** Implement a **stationary block bootstrap** (Politis & Romano) to preserve autocorrelation and streak characteristics.

---

### 3.4 Universe Selection & Survivorship Bias

#### [BIAS-05] Survivorship Bias in Multi-Coin Universe
- **Location:** `trade_assistant_gui.py` (`load_coins()`) and `app.py` (`COIN_LIST`):
  ```python
  rows = [(d["symbol"], float(d["quoteVolume"])) for d in data if d["symbol"].endswith("USDT")]
  rows.sort(key=lambda r: -r[1])
  return [s for s, _ in rows[:250]]
  ```
- **The Issue:** Selecting today's top 250 coins by 24h volume chooses the historical winners with hindsight. Coins that were in the top 20 two years ago but collapsed, suffered liquidity death, or were delisted (e.g. LUNA, FTT, UST) are excluded. Backtesting over historical periods on today's survivors yields an artificial positive survival bias.
- **Fix:** For historical pooled research, restrict testing to coins that maintained top market cap status continuously across the entire sample window, or use historical point-in-time universe snapshots.

---

### 3.5 Logic and Operational Bugs in Ancillary Modules

#### [BUG-09] Broken Revenge Trading Calculation in `trade_journal.py`
- **Location:** `trade_journal.py`, lines 92-98:
  ```python
  ordered = d.sort_values("open_time").reset_index(drop=True)
  for i in range(1, len(ordered)):
      p, t = ordered.iloc[i - 1], ordered.iloc[i]
      if p["r"] <= 0 and pd.notna(p["close_time"]) and pd.notna(t["open_time"]):
          gap = (t["open_time"] - p["close_time"]).total_seconds() / 60
          if 0 <= gap <= 30:
              revenge.append(float(t["r"]))
  ```
- **The Issue:** Sorting by `open_time` fails when positions overlap in time. If trade A opens at $10:00$ and closes at $12:00$ with a loss, trade B opens at $10:30$ and closes at $11:00$, and trade C opens at $12:10$ (a classic revenge trade following A), `ordered.iloc[i-1]` compares trade C to trade B (which closed at $11:00$, a 70 min gap) rather than trade A. The revenge trade is missed.
- **Fix:** Sort closed trades by `close_time`, or search for any trade opened within 30 minutes of a loss closure.

#### [BUG-10] Unicode Encoding Failure in `news_alerts.py` on Windows
- **Location:** `news_alerts.py`, lines 85 and 94:
  ```python
  with open(STATE_PATH) as f: return json.load(f)
  with open(tmp, "w") as f: json.dump(st, f)
  ```
- **The Issue:** On Windows, `open()` without specifying `encoding="utf-8"` defaults to Windows-1252/ANSI. `news_alerts.py` formats alerts with Unicode emojis (`🚨`, `🟢`, `🔴`, `🟡`). When articles contain non-ASCII headlines or emojis, this raises an unhandled `UnicodeEncodeError`.
- **Fix:** Always explicitly specify `encoding="utf-8"` in file operations.

#### [INT-13] Disconnection Between Coach and Quantitative Engine
- **Location:** `trade_chat.py` & `trade_ai.py`.
- **The Issue:** The LLM coach prompt receives qualitative analysis and journal stats, but has zero access to backtest results, verified strategy expectancy, win-rate confidence intervals, or cost-sensitivity thresholds. When users ask questions regarding strategy edge, the model is forced to generalize or hallucinate.
- **Fix:** Provide the LLM coach with deterministic tool access to `trade_backtest` results.

---

## 4. Performance Bottlenecks & Code Architecture

### Performance Benchmark
- **Current Optimization Run:** On LTCUSDT 15m (70,075 candles), running a 54-combination grid search (`tb.optimize`) takes **4.72 seconds** on a single thread.
- **Extrapolation to Full Walk-Forward Across 10 Coins:**
  $$4 \text{ folds} \times 54 \text{ combos} \times 10 \text{ coins} \times 4.72\text{s} \approx 1,020\text{ seconds } (\sim 17\text{ minutes})$$
- **Root Causes:**
  1. `atr(df)` is recomputed from scratch on all 70,000 rows on every parameter iteration.
  2. The trade matching simulation is an unvectorized, nested Python `while` and `for` loop.
  3. CSV parsing with datetime conversion takes $\sim 800\text{ms}$ per read. Parquet serialization reduces this to $< 50\text{ms}$ ($16\times$ faster).

---

## 5. Verification Test Suite Status

The test suite `tests/test_audit.py` was constructed and executed with `pytest`:

```text
tests/test_audit.py::test_random_walk_no_edge_after_costs[ema_pullback] PASSED [  9%]
tests/test_audit.py::test_random_walk_no_edge_after_costs[breakout]     PASSED [ 18%]
tests/test_audit.py::test_random_walk_no_edge_after_costs[mean_reversion] PASSED [ 27%]
tests/test_audit.py::test_random_walk_no_edge_after_costs[htf_trend]   PASSED [ 36%]
tests/test_audit.py::test_zero_cost_martingale_bound                    PASSED [ 45%]
tests/test_audit.py::test_cost_drag_proportional_to_stop_width          PASSED [ 54%]
tests/test_audit.py::test_gap_stop_loss_unrealistic_fill                PASSED [ 63%]
tests/test_audit.py::test_monte_carlo_drawdown_missing_initial_equity   PASSED [ 72%]
tests/test_audit.py::test_kelly_sizing_allows_risk_without_statistical_edge PASSED [ 81%]
tests/test_audit.py::test_fetch_history_cache_truncation_logic          PASSED [ 90%]
tests/test_audit.py::test_trade_journal_revenge_overlapping_trades      PASSED [100%]

======================= 11 passed, 2 warnings in 1.71s =======================
```

**Key Verification Outcomes:**
1. **No-Edge Invariant:** On a 30,000-candle zero-drift random walk, all 4 strategies correctly demonstrate non-positive expectancy after costs ($r \le 0.0\text{R}$).
2. **Audit Flaws Proved:** Unit tests successfully isolate and confirm the execution gap fill bias, Monte Carlo drawdown starting equity omission, Kelly sizing on unproven edge, cache truncation, and revenge trade ordering flaws.

---

## 6. Proposed Action Plan for Phase 2 & Phase 3

Before proceeding, the following architectural fixes are recommended for Phase 2 (Strategy Research) and Phase 3 (Engine Optimization):

1. **Modular Strategy Harness:** Implement a clean plug-in interface where strategies define pure signal arrays, parameter spaces, and exit criteria, decoupled from the simulation engine.
2. **Execution Engine Corrections:**
   - Realistic gap fills: $\text{fill} = \min(open, stop)$ for long stops.
   - Initial equity prepending in Monte Carlo drawdown calculations.
   - Zero-edge cutoff in fractional Kelly sizing: return $0.0\%$ if $E \le 0$ or $\text{CI}_{\text{lower}} \le 0$.
   - Cache protection against truncating existing candles.
3. **Multi-Coin Pooled Engine:** Build a portfolio-level simulation that aggregates trade streams across multiple coins, explicitly tracking concurrent positions and cross-sectional correlation.
4. **Numba/Vectorized Simulation & Parquet Caching:**
   - Accelerate the candle-by-candle trade loop with Numba JIT (targeting $>5\times$ speedup).
   - Cache technical indicators across parameter grids.
   - Migrate historical data cache from CSV to Parquet.
