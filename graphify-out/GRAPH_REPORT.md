# Graph Report - Trade  (2026-10-29)

## Corpus Check
- 18 files · ~25,268 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 3, .example 1, (none) 1)

## Summary
- 381 nodes · 818 edges · 11 communities (8 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7fff69af`
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
- Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review
- ReplayWindow
- CLAUDE.md
- trade-assistant

## God Nodes (most connected - your core abstractions)
1. `App` - 56 edges
2. `build_plan()` - 16 edges
3. `get_user()` - 13 edges
4. `snapshot()` - 13 edges
5. `main()` - 12 edges
6. `backtest()` - 12 edges
7. `run_backtest()` - 11 edges
8. `ReplayWindow` - 11 edges
9. `run_check()` - 10 edges
10. `fetch_history()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `[BUG-01] Cache File Truncation Bug in `fetch_history()`` --references--> `fetch_history()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `[LEAK-12] Convexity Drift in Synthetic History (`demo_history`)` --references--> `demo_history()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `[FLAW-11] HTF Resampling & Alignment Bias in `s_htf_trend`` --references--> `s_htf_trend()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `[STAT-04] Drawdown Calculation Flaw in `monte_carlo()`` --references--> `monte_carlo()`  [INFERRED]
  AUDIT.md → trade_backtest.py
- `2. Ranked Findings: Bugs, Biases, and Risks` --references--> `load_coins()`  [INFERRED]
  AUDIT.md → trade_assistant_gui.py

## Import Cycles
- None detected.

## Communities (11 total, 3 thin omitted)

### Community 0 - "trade_backtest.py"
Cohesion: 0.07
Nodes (44): 2. Ranked Findings: Bugs, Biases, and Risks, [BIAS-06] Bonferroni Over-Penalization on Out-of-Sample Results, demo_provider(), demo_data(), adx(), analyse(), atr(), backtest() (+36 more)

### Community 1 - "App"
Cohesion: 0.07
Nodes (10): get_user(), App, work(), work(), work(), _fetch(), get_coin_icon(), normalize_symbol() (+2 more)

### Community 2 - "news_alerts.py"
Cohesion: 0.07
Nodes (31): coin_keyboard(), cron(), get_ai_response(), handle_callback(), handle_command(), health_check(), send(), send_coin_picker() (+23 more)

### Community 3 - "trade_ai.py"
Cohesion: 0.07
Nodes (44): atr_series(), binance_provider(), build_levels(), build_plan(), check_trigger(), choose_strategy(), closed_bars(), confluence_checks() (+36 more)

### Community 4 - "trade_chat.py"
Cohesion: 0.08
Nodes (23): ask(), ask_api(), ask_cli(), cli_path(), mode(), status_text(), _transcript(), test_trade_journal_revenge_overlapping_trades() (+15 more)

### Community 5 - "test_audit.py"
Cohesion: 0.09
Nodes (8): generate_random_walk(), test_cost_drag_proportional_to_stop_width(), test_fetch_history_keeps_cache_when_nothing_new(), test_gap_stop_fills_at_open(), test_kelly_zero_without_statistical_edge(), test_monte_carlo_drawdown_counts_from_starting_equity(), test_random_walk_no_edge_after_costs(), test_zero_cost_martingale_bound()

### Community 6 - "trade_assistant_gui.py"
Cohesion: 0.11
Nodes (7): bullet_list(), card(), draw_chart(), ensure_deps(), pill(), scan_symbols(), _style()

### Community 8 - "Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review"
Cohesion: 0.07
Nodes (26): 1. Executive Summary & Verification of Prior Assumptions, 3.1 Lookahead Bias & Data Integrity, 3.2 Execution Realism & Cost Modeling, 3.3 Statistical Flaws & Anti-Overfitting Misconfigurations, 3.4 Universe Selection & Survivorship Bias, 3.5 Logic and Operational Bugs in Ancillary Modules, 3. Deep-Dive Analysis of Identified Issues, 4. Performance Bottlenecks & Code Architecture (+18 more)

## Knowledge Gaps
- **1 isolated node(s):** `trade-assistant`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 109 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `App` connect `App` to `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review`, `trade_ai.py`, `trade_chat.py`, `trade_assistant_gui.py`?**
  _High betweenness centrality (0.231) - this node is a cross-community bridge._
- **What connects `trade-assistant` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `trade_backtest.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06564364876385337 - nodes in this community are weakly interconnected._
- **Why does `2. Ranked Findings: Bugs, Biases, and Risks` connect `trade_backtest.py` to `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Should `App` be split into smaller, more focused modules?**
  _Cohesion score 0.0677555958862674 - nodes in this community are weakly interconnected._
- **Why does `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review` connect `Comprehensive Audit Report: Quantitative Integrity, Statistics & Code Review` to `trade_backtest.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Should `news_alerts.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0707070707070707 - nodes in this community are weakly interconnected._