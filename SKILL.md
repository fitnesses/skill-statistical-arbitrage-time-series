---
name: statistical-arbitrage-time-series
name_zh: 统计套利与时间序列建模
description: Generate a sourced, reproducible statistical-arbitrage research dossier
  for Chinese commodity futures (calendar spreads and cross-commodity spreads, daily
  data via the zeus MCP), covering point-in-time contract mapping and rolls, pair
  selection, cointegration and stationarity testing (train-window ADF + KPSS), spread
  modeling with hedge-ratio stability, mean-reversion estimation, z-score signal
  construction, integer-lot actual-contract backtesting with futures costs (per-leg
  fees, slippage, margin and liquidity constraints) and Sharpe significance
  testing, and robustness risk flags. Use when the
  user asks for 统计套利研究、配对交易回测、协整检验、价差均值回归建模、信号有效性验证、回测偏差排查,
  or a one-stop stat-arb strategy due-diligence report.
description_zh: 面向国内商品期货（跨期、跨品种价差，日线，经 zeus MCP 取数），输入候选价差，输出一份可复现、可溯源的统计套利研究报告：按时点安全的合约映射与换月、配对筛选、协整与平稳性检验（训练窗
  ADF+KPSS+EG）、价差建模与对冲比率稳定性、均值回归估计、z-score 信号构建、真实合约整手回测（逐腿手续费、滑点、保证金、流动性约束）及
  Sharpe 显著性检验、稳健性风险清单。适用于统计套利研究、配对交易回测、协整检验、价差均值回归建模、信号有效性验证、回测偏差排查等场景。
metadata:
  organization: QuantSkills
  organization_url: https://github.com/quantskills
  repository: skill-statistical-arbitrage-time-series
  repository_url: https://github.com/fitnesses/skill-statistical-arbitrage-time-series
  project_type: skill
  collection: statistical-arbitrage-time-series
quantSkills:
  project_type: skill
  category: research
  tags:
  - statistical-arbitrage
  - pairs-trading
  - cointegration
  - time-series
  - mean-reversion
  - backtesting
  - commodity-futures
  - calendar-spread
  platforms:
  - claude-code
  - codex
  - openclaw
  - cursor
  status: stable
  validation_level: runnable
  maintainer_type: community
  summary_zh: 输入一组候选标的，输出一份可复现的统计套利研究报告：配对筛选、训练窗协整与平稳性检验、价差均值回归与对冲比率稳定性、信号构建、含期货成本与
    Sharpe 显著性的带偏差控制回测，一次验真伪。
  summary_en: A statistical-arbitrage skill that screens pairs, tests cointegration
    on the training window (ADF + KPSS), models spread mean-reversion and hedge-ratio
    stability, builds z-score signals, and runs a bias-controlled backtest with
    futures costs and Sharpe significance on actual contracts in integer lots to judge
    whether an apparent edge is real.
  license: GPL-3.0-only
  requires:
  - python>=3.10
  - statsmodels>=0.14
  - zeus MCP (fut_daily)
---

# Statistical Arbitrage & Time Series Modeling

Use this skill to turn a Chinese commodity-futures spread — a **calendar spread** (same product, e.g. RB dominant vs RB next) or a **cross-commodity spread** with an industrial-chain rationale (e.g. HC vs RB) — into a sourced, reproducible research dossier: zeus MCP data snapshot, point-in-time contract mapping and roll events, cointegration and stationarity tests, spread modeling, mean-reversion estimation, signal construction, an integer-lot backtest on actual contracts with futures costs and margin, and robustness risk flags.

The single most important job of this skill is to decide **whether an apparent edge is real or spurious**. A pretty out-of-sample equity curve means nothing until it is statistically significant (the Sharpe has a meaningful t-statistic), free of look-ahead leakage, survives realistic **futures costs (per-leg fees, slippage) and margin**, and is not just directional beta in disguise. Default to skepticism: design the analysis to disprove the edge, and when the evidence is only indicative, say so and recommend continued testing rather than a tradable conclusion.

## Core Workflow

Paths such as `scripts/run_statarb.py` are relative to **this skill's directory**; call the script by its absolute path and keep configs, snapshots and run outputs in the user's workspace (e.g. `./statarb_runs/<name>/`), not inside the skill. zeus connection details are found automatically, in this order: process env `ZEUS_MCP_URL`/`ZEUS_MCP_TOKEN` → `.env` in the workspace → `.env` in the skill directory → the `zeus` MCP server already configured in Claude Code (`.mcp.json` / `~/.claude.json`). If none is found the script says so; then ask the user to copy `.env.example` to `.env` in the workspace and fill it in. Never print the token or write it into any file you create.

0. Run the script with **`uv run <skill-dir>/scripts/run_statarb.py …`**. The script declares its dependencies inline (PEP 723), so the first run creates an isolated environment with numpy/pandas/scipy/**statsmodels** automatically (~30 s once, cached afterwards); the user never installs Python packages. If `uv` is missing, ask the user's permission and install it (Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`; macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`). Only if that is impossible, fall back to `python -m pip install -r <skill-dir>/requirements.txt` and plain `python`. `statsmodels` is mandatory — the script refuses to run without it, so every p-value is a real ADF/KPSS/Engle-Granger result.
1. Normalize the input into two legs. A leg is either a product rule — `{"product": "RB", "exchange": "SHF", "select": "dominant"}` (前一交易日持仓量最大的主力) or `"select": "second"` (second leg only: the most-held contract delivering after the first leg, i.e. a calendar far leg) — or a fixed contract `"RB2501.SHF"`. Name the exact products/contracts used; ask only when the input is ambiguous.
2. Confirm the study scope. Default to daily bars over roughly three to five years, a chronological train/test split with the most recent ~30% held out of sample, next-day-open execution with settle marking, and per-leg futures costs. **Before running, ask the user for each product's fee and margin** — exchange fee as a rate on turnover (`fee_rate`, e.g. 0.0001 = 万分之一) and/or per lot (`fee_per_lot`, 元/手), and the margin rate (`margin_rate`, e.g. 0.10) — preferably their own broker's numbers. These have no data source; never invent them silently. If the user has no numbers, propose the exchange's published values as an explicit assumption and get confirmation. A **cross-commodity** spread requires a written economic/industrial-chain `rationale` (correlation is not a reason); record how many candidates were screened in `screening`.
3. Read `references/statarb-guide.md` before the first dossier in a session. Use it for the stage-by-stage method map, the formulas for derived metrics, the default robustness thresholds, the report blueprint, the explicit "implemented vs. agent-supplied" split, and the appendix requirements.
4. Fetch data **only through the zeus MCP**; never use third-party sources or connect to DolphinDB. The script calls zeus itself (`tools/list`, then `fut_basic` → `fut_daily` per contract). If zeus lacks `fut_basic` it is recorded as a capability gap: contracts are probed by product+YYMM and the multiplier/tick come from `assumptions` (see `references/zeus-mcp-interface.md`). Fees and margin always come from `assumptions`; a missing required value stops the run and names the config key — nothing is silently set to zero. Write a research config next to where the snapshot should live:
   ```json
   {"snapshot": "snapshot.json", "start_date": "20210104", "end_date": "20241231",
    "legs": [{"name": "HC", "product": "HC", "exchange": "SHF", "select": "dominant"},
             {"name": "RB", "product": "RB", "exchange": "SHF", "select": "dominant"}],
    "rationale": "热卷与螺纹同为钢坯下游成材，共享铁矿/焦炭成本端，价差反映板材与建材需求相对强弱",
    "screening": {"n_candidates": 1, "procedure": "预设单一配对"},
    "roll": {"exclude_months_before_delivery": 1},
    "execution": {"base_lots": 10, "exec_price": "open", "mark_price": "settle", "slippage_ticks": 1,
                  "max_participation": 0.05, "capital": 1000000, "funding_rate_annual": 0.0,
                  "broker_fee_multiplier": 1.0, "broker_margin_add": 0.0, "roll_slippage_ticks": 0},
    "assumptions": {"HC": {"multiplier": 10, "price_tick": 1, "fee_rate": 0.0001, "fee_per_lot": 0, "margin_rate": 0.10},
                    "RB": {"multiplier": 10, "price_tick": 1, "fee_rate": 0.0001, "fee_per_lot": 0, "margin_rate": 0.10}}}
   ```
   `assumptions.<品种>` must give `margin_rate` and `fee_rate` and/or `fee_per_lot` for every product (state where each number came from); `multiplier`/`price_tick` are only needed while zeus lacks `fut_basic`. `execution` and `roll` have the defaults shown. Limit-up/limit-down locks are not modeled yet — say so in the report. Then:
   ```bash
   uv run <skill-dir>/scripts/run_statarb.py --check-zeus RB2501.SHF    # 可选：zeus 连接与字段检查
   uv run <skill-dir>/scripts/run_statarb.py --config statarb_runs/hc_rb/config.json --fetch --out-dir statarb_runs/hc_rb/out
   uv run <skill-dir>/scripts/run_statarb.py --config statarb_runs/hc_rb/config.json --out-dir statarb_runs/hc_rb/out2   # 离线复跑
   ```
   `--fetch` writes the verbatim responses to an immutable snapshot (refuses to overwrite). Each replay writes `report.md`, `manifest.json` (run_id, config/snapshot/code hashes, MCP calls and params, server version, capability gaps, contract-spec sources, fitted parameters, statistical and executable metrics, output hashes), `mapping.csv` (per-day contracts, research log price, executable price), `rolls.csv`, `trades.csv`, `daily.csv` and `events.csv` (deferred fills, liquidity flags). Malformed inputs fail with the exact location (e.g. `calls[1].rows[3]: missing 'close'`). No-trade days (`vol=0`, null open/high/low) are valid data: they are marked at settle but never filled — both legs wait.
5. Collect evidence first, then analyze. Keep raw price tables, alignment diagnostics, test statistics, p-values, estimated coefficients, and trade logs long enough to cite the data window, sample size, and missing-data status in the final report.
6. Produce Markdown by default. If the user asks for Word, PDF, or a polished deliverable, generate the analytical content here first, then use the relevant document skill for final layout.

## Analysis Rules

- Separate facts, derived metrics, and judgment. Label every derived quantity (hedge ratio, ADF/KPSS statistics and p-values, Johansen trace, half-life, z-score, Sharpe and **its t-statistic**, max drawdown, cost-adjusted returns) with its formula and the field/series names it was computed from.
- Align series before any test. State how prices were aligned (common trading days, forward-fill policy, log vs. level), and report the usable sample size after alignment.
- Test stationarity and cointegration explicitly on the **training window**, never assume it and never test it on the full sample while estimating β on the training window only (that is look-ahead leakage). Report ADF and KPSS (opposite nulls; they should agree), the statistic, the p-value, and the critical values; add Engle-Granger/Johansen when the question warrants.
- Always separate in-sample and out-of-sample. Estimate the hedge ratio, thresholds, and any tuned parameter on the training window only, then evaluate on the untouched test window. Check **both** failure modes: in-sample good / out-of-sample bad (overfitting) **and** in-sample bad / out-of-sample good (regime dependence or luck — an edge with no in-sample support is not a stable edge).
- Report hedge-ratio stability, not just a point estimate. Show chunk-wise or rolling β and its drift; a spread whose own definition drifts is not tradable.
- Keep statistical evidence and trading feasibility apart. Statistics run on the **research continuous series** (same-contract daily returns chained across rolls, no splice gaps); PnL comes only from the **executable backtest on actual contracts in integer lots** (signal at close t, fill at t+1, roll = close old + open new with the same lots). A stationary spread is not a tradable strategy until the executable backtest says so.
- Contract mapping must be point-in-time safe: dominant/second selection uses only the previous day's open interest, rolls only move to later deliveries, and contracts inside the pre-delivery window are never held. Cite `rolls.csv` for every roll.
- Model costs as evidence, not an afterthought. Per-leg fees (rate and/or per lot, open and close each charged, with the broker multiplier), slippage in ticks, margin at the user-specified rate, optional funding on margin; a missing bar on either leg defers **both** legs (no legging), thin volume is flagged; limit locks are not modeled (disclose as a limitation). Report gross and net, costs by type, drawdown, turnover, round trips, margin used, return on margin, and the stress scenario (fees ×2, slippage +1 tick).
- Report Sharpe with its significance. Give the approximate t-statistic and an overlap-deflated effective t (positions held ~half-life days are autocorrelated). A Sharpe with |t| < ~2 is not distinguishable from zero — never present it as an edge.
- Watch for leaked beta. When the hedge ratio is low and both legs co-move strongly, spread PnL may contain directional market/sector exposure; recommend or run a factor regression of spread returns before concluding the edge is alpha.
- Treat empty or insufficient results as evidence. State "无数据" or "样本不足" with the method name, window, and usable sample size instead of silently omitting the section.
- Use high/medium/low robustness levels only when a rule in `references/statarb-guide.md` or a user-provided rule is triggered. Include the triggering rule text beside each flag.
- Be honest about provenance. The bundled script does point-in-time contract mapping and rolls, OLS β, ADF+KPSS (on the training-window spread, with full/OOS cross-checks), Engle-Granger cointegration on the training window, **returns-space hedge-ratio stability (a Chow-style first-half-vs-second-half β break test plus a noise-corrected chunk-dispersion diagnostic)**, half-life, an adaptive z-score backtest with a multi-component cost model, Sharpe t-stats with overlap deflation, and **two-level factor attribution** (strategy-PnL alpha/beta and spread-return market-neutrality vs a market proxy). an integer-lot executable backtest on actual contracts, and futures risk rules. Johansen, Kalman dynamic hedging, full CUSUM, true walk-forward, and seasonality/regime analysis are agent extensions; if not run, report them as "未做" rather than implying they were done.
- End every report with this disclaimer: `本报告基于公开数据与规则化分析生成，仅供研究参考，不构成任何投资建议。`

## Resource Guide

- `references/statarb-guide.md`: stage-by-stage method map, derived-metric formulas, robustness/risk rules, the implemented-vs-agent-supplied split, report blueprint, and final QA checklist.
- `references/sample_report_futures_calendar.md`, `references/sample_report_futures_cross.md`: complete futures dossiers generated from the synthetic contract fixtures (not market data) — use them as the target shape of a report; `sample_report_{greenlight,reject,betadrift}.md` are statistical self-test outputs.
- `references/zeus-mcp-interface.md`: the zeus MCP tool contract (`fut_daily` and `fut_basic` both live), required fields, and what the script does while each tool is missing.
- `scripts/futures.py`: futures rules — contract-code parsing, point-in-time dominant/second mapping and roll events, research continuous series, contract specs (multiplier/tick from zeus `fut_basic` or config; fees and margin from config; else error), and the integer-lot executable backtest with fees, slippage, margin, missing-bar deferral and liquidity flags.
- `scripts/run_statarb.py`: runnable backbone — research config validation, zeus fetch into an immutable snapshot (`--fetch`), replay (`--config`), `--check-zeus`, or synthetic data for self-test; tests cointegration on the training window (ADF + KPSS, with full/OOS cross-checks), estimates the hedge ratio plus its returns-space stability (Chow-style half-sample β break test + noise-corrected chunk dispersion) and half-life, builds an adaptive z-score signal, runs a bias-controlled gross/net backtest with per-leg futures fees + slippage and margin usage, computes Sharpe with a t-statistic and overlap-deflated effective t, runs two-level factor attribution (strategy-PnL alpha/beta and spread-return market-neutrality), applies the robustness rules, and writes the Markdown report. Built-in offline self-test: `python scripts/run_statarb.py --source synthetic --mode {coint,nocoint,strong,inversion,leaked,drift}` exercises each verdict branch (e.g. `strong`→green light, `inversion`→IS/OOS-mismatch flag, `leaked`→spread-not-market-neutral flag, `drift`→hedge-ratio-break flag) without network access.

## Quality Bar

- Every material claim traces to a data source, a window/sample size, and a named test or formula.
- The hedge ratio and all thresholds are estimated on the training window only; cointegration is tested on the training-window spread; report any residual look-ahead or survivorship risk explicitly.
- The headline Sharpe is reported with its t-statistic; a sub-2 t is called out as statistically insignificant rather than presented as an edge.
- Executable results use actual contract prices, integer lots, per-leg futures fees and slippage, and report margin usage; every contract-spec value is traceable to zeus or to a disclosed config assumption.
- Contract mappings and roll events are reproducible from the snapshot and use only information available at each decision time.
- Cross-commodity spreads carry a written economic rationale; statistical evidence and trading feasibility get separate verdicts.
- Multiple-testing is disclosed: if many pairs were screened, say how many and whether a correction (e.g. Bonferroni, FDR) or a held-out confirmation was applied.
- Advanced steps not performed (Johansen, Kalman, Chow/CUSUM, walk-forward, seasonality/regime) and every zeus capability gap are listed as not done / assumed, never implied as automatic.
- Do not overstate an edge. Prefer "可能存在均值回归", "需要样本外确认", "Sharpe 与 0 不可区分", "扣费后优势消失", and never use buy/sell language.
