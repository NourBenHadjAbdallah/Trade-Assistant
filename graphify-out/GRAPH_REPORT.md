# Graph Report - Trade  (2026-11-02)

## Corpus Check
- 24 files · ~34,502 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 3, .example 1, (none) 1)

## Summary
- 524 nodes · 1225 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `92c21dc5`
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
- price_alerts.py
- Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review
- signal_log.py
- CLAUDE.md
- trade-assistant
- test_trader_tools.py

## God Nodes (most connected - your core abstractions)
1. `App` - 75 edges
2. `check()` - 16 edges
3. `build_plan()` - 16 edges
4. `text()` - 15 edges
5. `get_user()` - 13 edges
6. `candles()` - 13 edges
7. `snapshot()` - 13 edges
8. `check()` - 12 edges
9. `main()` - 12 edges
10. `backtest()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `[STAT-04] Drawdown Calculation Flaw in `monte_carlo()`` --references--> `monte_carlo()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `work()` --indirect_call--> `plan()`  [INFERRED]
  trade_assistant_gui.py → tests/test_trader_tools.py
- `2. Ranked Findings: Bugs, Biases, and Risks` --references--> `load_coins()`  [INFERRED]
  AUDIT.md → trade_assistant_gui.py
- `[BIAS-05] Survivorship Bias in Multi-Coin Universe` --references--> `load_coins()`  [INFERRED]
  AUDIT.md → trade_assistant_gui.py
- `[BUG-01] Cache File Truncation Bug in `fetch_history()`` --references--> `fetch_history()`  [INFERRED]
  AUDIT.md → trade_backtest.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "trade_backtest.py"
Cohesion: 0.05
Nodes (55): 2. Ranked Findings: Bugs, Biases, and Risks, 3.1 Lookahead Bias & Data Integrity, [BIAS-06] Bonferroni Over-Penalization on Out-of-Sample Results, [BUG-01] Cache File Truncation Bug in `fetch_history()`, [FLAW-11] HTF Resampling & Alignment Bias in `s_htf_trend`, [LEAK-12] Convexity Drift in Synthetic History (`demo_history`), test_simulation_probabilities_add_up(), _first() (+47 more)

### Community 1 - "App"
Cohesion: 0.05
Nodes (14): get_user(), App, work(), work(), work(), _fetch(), bullet_list(), card() (+6 more)

### Community 2 - "news_alerts.py"
Cohesion: 0.07
Nodes (31): coin_keyboard(), cron(), get_ai_response(), handle_callback(), handle_command(), health_check(), send(), send_coin_picker() (+23 more)

### Community 3 - "trade_ai.py"
Cohesion: 0.06
Nodes (47): atr_series(), binance_provider(), build_levels(), build_plan(), check_trigger(), choose_strategy(), closed_bars(), confluence_checks() (+39 more)

### Community 4 - "trade_chat.py"
Cohesion: 0.07
Nodes (25): ask(), ask_api(), ask_cli(), cli_path(), mode(), status_text(), _transcript(), test_trade_journal_revenge_overlapping_trades() (+17 more)

### Community 5 - "test_audit.py"
Cohesion: 0.10
Nodes (8): generate_random_walk(), test_cost_drag_proportional_to_stop_width(), test_fetch_history_keeps_cache_when_nothing_new(), test_gap_stop_fills_at_open(), test_kelly_zero_without_statistical_edge(), test_monte_carlo_drawdown_counts_from_starting_equity(), test_random_walk_no_edge_after_costs(), test_zero_cost_martingale_bound()

### Community 6 - "trade_assistant_gui.py"
Cohesion: 0.12
Nodes (3): ensure_deps(), ReplayWindow, _style()

### Community 7 - "price_alerts.py"
Cohesion: 0.12
Nodes (22): add(), add_plan(), cancel(), check(), _db(), describe(), entry_state(), fetch_prices() (+14 more)

### Community 8 - "Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review"
Cohesion: 0.08
Nodes (22): 1. Executive Summary & Verification of Prior Assumptions, 3.2 Execution Realism & Cost Modeling, 3.3 Statistical Flaws & Anti-Overfitting Misconfigurations, 3.4 Universe Selection & Survivorship Bias, 3.5 Logic and Operational Bugs in Ancillary Modules, 3. Deep-Dive Analysis of Identified Issues, 4. Performance Bottlenecks & Code Architecture, 5. Verification Test Suite Status (+14 more)

### Community 9 - "signal_log.py"
Cohesion: 0.15
Nodes (15): _block(), _db(), fetch_since(), grade(), load(), main(), _now(), record() (+7 more)

### Community 12 - "test_trader_tools.py"
Cohesion: 0.09
Nodes (36): check(), _db(), load_settings(), main(), record_taken(), report_text(), save_settings(), taken_today() (+28 more)

## Knowledge Gaps
- **1 isolated node(s):** `trade-assistant`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 142 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `App` connect `App` to `trade_ai.py`, `trade_chat.py`, `trade_assistant_gui.py`, `price_alerts.py`, `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review`, `signal_log.py`, `test_trader_tools.py`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **What connects `trade-assistant` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `trade_backtest.py` be split into smaller, more focused modules?**
  _Cohesion score 0.051201671891327065 - nodes in this community are weakly interconnected._
- **Why does `2. Ranked Findings: Bugs, Biases, and Risks` connect `trade_backtest.py` to `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Should `App` be split into smaller, more focused modules?**
  _Cohesion score 0.05269497139415839 - nodes in this community are weakly interconnected._
- **Why does `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review` connect `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review` to `trade_backtest.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Should `news_alerts.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0707070707070707 - nodes in this community are weakly interconnected._