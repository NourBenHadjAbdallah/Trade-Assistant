# Graph Report - Trade  (2026-10-29)

## Corpus Check
- 20 files · ~28,230 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 3, .example 1, (none) 1)

## Summary
- 428 nodes · 950 edges · 18 communities (15 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5f727ec3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- trade_backtest.py
- App
- news_alerts.py
- trade_ai.py
- trade_chat.py
- test_audit.py
- trade_assistant_gui.py
- pandas
- Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review
- signal_log.py
- CLAUDE.md
- trade-assistant
- test_signal_log.py
- fetch_history
- analyse
- confluence_filters
- backtest
- _get_klines

## God Nodes (most connected - your core abstractions)
1. `App` - 63 edges
2. `build_plan()` - 16 edges
3. `get_user()` - 13 edges
4. `candles()` - 13 edges
5. `snapshot()` - 13 edges
6. `main()` - 12 edges
7. `backtest()` - 12 edges
8. `g()` - 11 edges
9. `text()` - 11 edges
10. `run_backtest()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `[BIAS-05] Survivorship Bias in Multi-Coin Universe` --references--> `load_coins()`  [INFERRED]
  AUDIT.md → trade_assistant_gui.py
- `[BUG-01] Cache File Truncation Bug in `fetch_history()`` --references--> `fetch_history()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `[LEAK-12] Convexity Drift in Synthetic History (`demo_history`)` --references--> `demo_history()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `[FLAW-11] HTF Resampling & Alignment Bias in `s_htf_trend`` --references--> `s_htf_trend()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `2. Ranked Findings: Bugs, Biases, and Risks` --references--> `monte_carlo()`  [INFERRED]
  AUDIT.md → trade_backtest.py

## Import Cycles
- None detected.

## Communities (18 total, 3 thin omitted)

### Community 0 - "trade_backtest.py"
Cohesion: 0.16
Nodes (12): fmt_stats(), main(), make_card(), market_phases(), report_default(), report_optimize(), report_walkforward(), robustness() (+4 more)

### Community 1 - "App"
Cohesion: 0.06
Nodes (11): get_user(), App, work(), work(), work(), card(), draw_chart(), normalize_symbol() (+3 more)

### Community 2 - "news_alerts.py"
Cohesion: 0.07
Nodes (30): coin_keyboard(), cron(), get_ai_response(), handle_callback(), handle_command(), health_check(), send(), send_coin_picker() (+22 more)

### Community 3 - "trade_ai.py"
Cohesion: 0.08
Nodes (31): atr_series(), binance_provider(), build_levels(), build_plan(), check_trigger(), choose_strategy(), closed_bars(), confluence_checks() (+23 more)

### Community 4 - "trade_chat.py"
Cohesion: 0.10
Nodes (15): ask(), ask_api(), ask_cli(), cli_path(), mode(), status_text(), _transcript(), ask() (+7 more)

### Community 5 - "test_audit.py"
Cohesion: 0.09
Nodes (8): generate_random_walk(), test_cost_drag_proportional_to_stop_width(), test_fetch_history_keeps_cache_when_nothing_new(), test_gap_stop_fills_at_open(), test_kelly_zero_without_statistical_edge(), test_random_walk_no_edge_after_costs(), test_trade_journal_revenge_overlapping_trades(), test_zero_cost_martingale_bound()

### Community 6 - "trade_assistant_gui.py"
Cohesion: 0.11
Nodes (6): _fetch(), bullet_list(), ensure_deps(), get_coin_icon(), ReplayWindow, _style()

### Community 7 - "pandas"
Cohesion: 0.14
Nodes (24): demo_provider(), analyze(), anchored_vwap(), atr(), build_avwaps(), cluster_levels(), demo_data(), fetch_binance() (+16 more)

### Community 8 - "Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review"
Cohesion: 0.11
Nodes (17): 1. Executive Summary & Verification of Prior Assumptions, 3.2 Execution Realism & Cost Modeling, 3.4 Universe Selection & Survivorship Bias, 3.5 Logic and Operational Bugs in Ancillary Modules, 3. Deep-Dive Analysis of Identified Issues, 4. Performance Bottlenecks & Code Architecture, 5. Verification Test Suite Status, 6. Proposed Action Plan for Phase 2 & Phase 3 (+9 more)

### Community 9 - "signal_log.py"
Cohesion: 0.16
Nodes (14): _block(), _db(), fetch_since(), load(), main(), _now(), record(), report_text() (+6 more)

### Community 12 - "test_signal_log.py"
Cohesion: 0.27
Nodes (13): grade(), candles(), g(), test_fees_reduce_r(), test_gap_through_stop_fills_at_open(), test_in_zone_fills_at_next_open_and_ignores_older_candles(), test_short_signal_mirrors_long(), test_stop_is_full_loss() (+5 more)

### Community 13 - "fetch_history"
Cohesion: 0.16
Nodes (10): 2. Ranked Findings: Bugs, Biases, and Risks, 3.1 Lookahead Bias & Data Integrity, [BUG-01] Cache File Truncation Bug in `fetch_history()`, [FLAW-11] HTF Resampling & Alignment Bias in `s_htf_trend`, [LEAK-12] Convexity Drift in Synthetic History (`demo_history`), load_coins(), demo_history(), fetch_history() (+2 more)

### Community 14 - "analyse"
Cohesion: 0.14
Nodes (11): 3.3 Statistical Flaws & Anti-Overfitting Misconfigurations, [BIAS-06] Bonferroni Over-Penalization on Out-of-Sample Results, [STAT-03] Kelly Criterion Sizing With No Statistical Edge, [STAT-04] Drawdown Calculation Flaw in `monte_carlo()`, [STAT-08] Monte Carlo Bootstrap Destroys Serial Correlation, test_monte_carlo_drawdown_counts_from_starting_equity(), analyse(), kelly() (+3 more)

### Community 15 - "confluence_filters"
Cohesion: 0.18
Nodes (12): adx(), atr(), confluence_filters(), ema(), htf_state(), rsi(), s_breakout(), s_confluence() (+4 more)

### Community 16 - "backtest"
Cohesion: 0.36
Nodes (5): backtest(), grid_combos(), optimize(), split_test(), walk_forward()

## Knowledge Gaps
- **1 isolated node(s):** `trade-assistant`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 119 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `App` connect `App` to `news_alerts.py`, `trade_chat.py`, `trade_assistant_gui.py`, `pandas`, `signal_log.py`, `fetch_history`?**
  _High betweenness centrality (0.228) - this node is a cross-community bridge._
- **What connects `trade-assistant` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `App` be split into smaller, more focused modules?**
  _Cohesion score 0.06057945566286216 - nodes in this community are weakly interconnected._
- **Why does `2. Ranked Findings: Bugs, Biases, and Risks` connect `fetch_history` to `trade_backtest.py`, `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review`, `analyse`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Should `news_alerts.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0707070707070707 - nodes in this community are weakly interconnected._
- **Why does `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review` connect `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review` to `fetch_history`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Should `trade_ai.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08130081300813008 - nodes in this community are weakly interconnected._