# Statistical Arbitrage Guide

Read this guide when generating or revising a statistical-arbitrage dossier. Use it as a compact operating checklist, not as a replacement for the exact data-source documentation or the statistical-library references.

The single job of this skill is to decide **whether an apparent edge is real or spurious**. Default to disproof, not confirmation: a pretty out-of-sample equity curve is not evidence until it is statistically significant, leakage-free, survives realistic futures costs (per-leg fees, slippage, margin), and is not just directional beta in disguise. When the data is only indicative, say so and stop.

## Default Scope

- Target: one candidate pair `[A, B]`, a small basket, or a universe plus a screening rule.
- Price window: daily close (and volume when needed) over roughly three to five years unless the user requests another window.
- Split: chronological train/test split with the most recent ~30% held strictly out of sample; never tune on the test window.
- Costs: futures costs per leg — exchange/broker fee (rate and/or per lot, charged on open and on close), slippage in ticks × tick size, and margin usage (add market impact for large notionals). Contract specs come from the config and are assumptions to be checked against current exchange/broker rules.
- Output: Markdown report unless the user requests HTML, Word, PDF, or another deliverable.

## What The Bundled Script Computes vs. What The Agent Must Add

Be honest in the report about provenance. `scripts/run_statarb.py` implements:

- zeus MCP fetch into an immutable snapshot (or synthetic data for self-test), with capability gaps recorded (`references/zeus-mcp-interface.md`);
- **point-in-time contract mapping**: dominant = most prior-day open interest (sticky, forward-only, forced out of the pre-delivery window); second = most-held contract delivering after the dominant leg; fixed contracts never roll; every roll logged with its decision day and evidence;
- a **research continuous series** (same-contract daily log returns chained across rolls — no splice gaps) used only for statistics, kept distinct from executable contract prices;
- OLS hedge ratio on the **training window only**, AR(1) half-life;
- **chunk-wise β** as a cheap structural-stability proxy;
- ADF + KPSS, run on the **training-window spread** for the headline decision (full-sample and OOS shown only as a cross-check), plus **Engle-Granger cointegration** (MacKinnon p-value) on training-window log prices; all tests come from `statsmodels` (mandatory, no approximation);
- a vectorized z-score backtest with an adaptive lookback, in/out-of-sample split, a multi-component cost model, and completed-round-trip counting;
- Sharpe with an approximate **t-statistic** and an overlap-deflated effective t;
- an **integer-lot executable backtest on actual contracts**: signal at close t → fill at t+1 (open by default) → mark at settle; leg B lots = round(|β|·lots_A·P_A·M_A / (P_B·M_B)); rolls close the old and open the new contract with the same lots; per-leg fees (rate + per lot, broker multiplier), slippage in ticks, margin at the user-specified rate, optional funding; a missing bar on either leg defers **both** legs; limit locks are not modeled; thin volume is flagged; IS/OOS/stress (fees ×2, slippage +1 tick) money metrics;
- contract specs: multiplier/tick from zeus `fut_basic` (config fallback, disclosed); fee rate/per-lot fee and margin rate specified by the user per product in the config; an error when a required value is missing;
- the robustness rule engine below (statistical rules + futures feasibility rules).

The following are **NOT** in the script and must be added by the agent when the question warrants, and must never be reported as "automatically done": Johansen cointegration, Kalman/dynamic hedge ratio, a full Chow/CUSUM break-test suite, true walk-forward re-estimation, seasonality/regime analysis, and intraday execution. If the agent did not run them, say "未做" with the reason.

**Now implemented in the script** (previously agent-only): factor attribution at two levels — (a) the strategy's OOS net PnL regressed on a market proxy (reports market β, residual α, and their t-stats; insignificant residual α flags an edge that may be leaked beta), and (b) the spread's daily returns regressed on the market proxy (market-neutrality check that directly addresses the low-hedge-ratio / co-moving-legs concern). The market proxy is an independent synthetic factor in self-test mode; zeus MCP runs currently have no market proxy, so attribution is reported as not done; if no proxy is available the report states attribution was not done.

## Stage Map

| Stage | Methods / Tests | Use |
|---|---|---|
| Data preparation | price load via `scripts/run_statarb.py`, calendar alignment, log-price transform, return series, gap/outlier audit | Build clean, aligned, same-currency series; report usable sample size after alignment. |
| Pair / universe selection | correlation screen, sector/cluster filter, distance method (SSD), preliminary cointegration scan | Narrow to economically related pairs before formal testing; record how many pairs were screened (multiple-testing exposure). |
| Cointegration & stationarity | ADF (`adfuller`) + KPSS (`kpss`) on the **training-window** spread; Engle-Granger (`coint`) / Johansen (`coint_johansen`) when the agent adds them | Decide whether a stationary tradable spread exists; report statistic, p-value, and critical values. Correlation is not cointegration. |
| Spread modeling & mean reversion | OLS/TLS hedge ratio (train only), chunk-wise/rolling/Kalman β, AR(1)/OU fit, half-life | Build the spread, estimate hedge ratio + reversion speed; report β stability. |
| Signal construction | rolling mean/std, z-score with lookback **≥ half-life**, entry/exit/stop bands, holding cap | Convert the spread into signals with explicit, past-only thresholds. |
| Backtesting & bias control | vectorized backtest, train/test isolation, walk-forward (agent), futures cost model (per-leg fees, slippage, margin), structural-break tests (agent) | Evaluate out of sample with realistic frictions; check parameter drift. |
| Performance & risk | annualized return, Sharpe **+ its t-stat**, Sortino, max drawdown, Calmar, hit rate, holding period, turnover | Summarize gross and net performance, downside profile, and statistical significance. |

## Metrics To Derive

State the formula and the series/field names used whenever a metric is derived.

- **Hedge ratio (β)**: OLS `log(A) = α + β·log(B) + ε`, estimated on the training window only; report chunk-wise/rolling β and its drift `(max−min)/|mean|`.
- **Spread**: `spread_t = log(A_t) − β·log(B_t)`; report the construction used.
- **Cointegration**: ADF statistic + p-value and KPSS statistic + p-value on the **training-window** spread; ADF and KPSS test opposite nulls and ideally agree (ADF rejects unit root *and* KPSS fails to reject stationarity). Add Engle-Granger/Johansen when needed.
- **Mean-reversion half-life**: AR(1) `Δspread_t = a + b·spread_{t−1} + e_t`, `half-life = −ln(2) / ln(1+b)`. Flag half-lives too long to be tradable after carry.
- **Z-score**: `z_t = (spread_t − rolling_mean_t) / rolling_std_t`; lookback uses only past data and should be **at least the half-life** (a window shorter than the half-life manufactures false crossings).
- **Performance**: `Sharpe = mean(daily PnL)/std × √252`; Sortino; max drawdown; `Calmar = annual return / |max DD|`; hit rate; holding period; turnover.
- **Sharpe significance**: approximate `t ≈ Sharpe_annual × √years`. Because positions are held for ~half-life days, daily PnL is autocorrelated and this t is optimistic; also report an overlap-deflated `t_eff ≈ Sharpe_annual × √(years / half-life)` and the implied number of independent bets ≈ `days / half-life`. **A Sharpe with |t| < ~2 is not distinguishable from zero — do not call it an edge.**
- **Cost-adjusted edge**: per-round-trip edge minus modeled round-trip cost = per-leg fees + slippage, legs weighted 1 : |β| by notional; report gross and net, the breakeven cost, and net return on margin.

## Robustness / Risk Rules

Use these defaults unless the user supplies thresholds. If an input is missing (no OOS window, no cost assumption), downgrade the rule to a qualitative watch item and say what is missing.

| Level | Trigger |
|---|---|
| High | No cointegration: spread ADF p-value > 0.10 (and/or KPSS rejects stationarity on the training-window spread). |
| High | Does not revert: AR(1) coefficient ≥ 0 (half-life = ∞). |
| High | Half-life > 60 trading days — reversion too slow to survive carry. |
| High | Hedge ratio breaks: a Chow-style first-half-vs-second-half β test (in log-return space) is significant (|z| ≥ 4) with a material relative gap (> 0.2). Medium at |z| ≥ 2.5. A secondary noise-corrected chunk-dispersion `excess_drift` is reported as color. Returns space is used because chunk OLS on I(1) price levels gives spurious instability when the regressor barely moves within a window. Prefer a full Chow/CUSUM break test as the authoritative check. |
| High | Edge vanishes after costs: net OOS Sharpe ≤ 0, or per-trade edge below modeled round-trip cost. |
| High | **Edge has no in-sample support: in-sample net Sharpe ≈ 0 or negative (≤ 0.05, or its t < 1) while out-of-sample net Sharpe is clearly positive (> 0.5)** — the apparent edge exists only out of sample, which points to regime dependence or luck, not a stable relationship. (This is the failure mode the engine previously missed.) |
| High | Overfitting: OOS net Sharpe < ~half of in-sample net Sharpe. |
| High | **Leaked beta: spread daily returns load significantly on the market (|t|≥1.96) with material R² (≥0.10)** — the spread is not market-neutral, so "edge" may be directional beta. The script also flags HIGH when the *strategy's* net PnL has significant market β but insignificant residual α. |
| Medium | Weak cointegration: spread ADF p between 0.05 and 0.10, or ADF and KPSS disagree. |
| Medium | **Sharpe not significant: out-of-sample Sharpe |t| < ~1.96** (and worse once overlap-deflated). |
| Medium | **Residual alpha not significant: after regressing strategy PnL on the market, |t(α)| < 1.96** — what looks like edge survives only as noise once beta is removed. |
| Medium | **Spread mildly non-neutral: market β significant but R² < 0.10** — modest directional leakage. |
| Medium | Low power: fewer than ~30 completed round-trip trades in the test window. |
| Medium | Data snooping: many pairs screened without a multiple-testing correction or a held-out confirmation set. |
| Medium | Half-life between 20 and 60 days, or a Chow-style β break with 2.5 ≤ |z| < 4. |
| Medium | z-score lookback shorter than the half-life. |
| High | **Executable backtest loses after costs**: out-of-sample net PnL ≤ 0 on actual contracts in integer lots. |
| Medium | Executable OOS net positive but its Sharpe |t| < 1.96. |
| Medium | Stress scenario (fees ×2, slippage +1 tick) turns OOS net PnL non-positive. |
| Medium | Fills deferred because a leg had no bar (both legs wait). |
| Medium | Liquidity: an order exceeds `max_participation` × that day's volume. |
| Medium | Multiplier/tick taken from config because zeus lacks `fut_basic` (fees and margin are always user-specified and disclosed). |
| Medium | Multiple testing: `screening.n_candidates` > 1 and the training-window EG p-value exceeds the Bonferroni threshold 0.05/n. |
| Medium | Lot rounding moves the realized notional hedge more than 10 % away from |β|. |
| Low | Minor data gaps, a single outlier, or a borderline single metric; record in the appendix rather than the headline flag list. |

Green light (verdict = "证据指向可进一步研究") requires **all** of: ADF p ≤ 0.05 on the training-window spread, finite half-life ≤ 60 days, OOS net Sharpe |t| ≥ 1.96, ≥ 30 completed round-trips, and (no attribution OR significant residual α). Slow-reverting or thin-sample pairs will therefore usually be rejected — that is the tool working as a disproof engine, not a bug.

Name combined signals explicitly, e.g. `协整边际 + Sharpe不显著`, `半衰期过长 + 扣费后归零`, `样本外有效但样本内无效 + 样本量不足`.

## Report Blueprint

Use this chapter order unless the user asks for a custom structure:

1. `摘要与结论`: a one-line overall verdict (tradable-evidence vs. reject/keep-disproving), the candidate(s), in-sample vs. out-of-sample net result **with the Sharpe t-stat**, net-of-cost edge, and the top robustness flags.
2. `数据、合约映射与换月`: zeus snapshot provenance (run_id, sha256, calls, missing capabilities), research window and train/test split date, each leg's selection rule, the roll table, and the research-series vs executable-price distinction.
3. `候选与经济逻辑`: spread family, the economic/industrial-chain rationale (required for cross-commodity), screening count and procedure, multiple-testing threshold.
4. `协整与平稳性检验`: ADF + KPSS on the **training-window** spread (headline) with full-sample/OOS as cross-checks; Engle-Granger/Johansen if run; the cointegration conclusion.
5. `价差建模与均值回归`: hedge-ratio estimation **and returns-space stability (Chow-style half-sample β break + `excess_drift`)**, spread construction, AR(1)/OU fit, half-life with its formula.
6. `交易可行性`: signal/execution timing, lot rule and realized hedge, cost and margin model, IS/OOS/full/stress money table (gross, fees, slippage, funding, net, Sharpe, t, drawdown, round trips, lots, turnover, margin, return on margin), deferred fills, liquidity flags, spec sources, and daily-bar execution limitations.
7. `回测与偏差控制`: in-sample vs. out-of-sample, the futures cost model (per-leg fees + slippage, margin usage), gross vs. net, **Sharpe t-stat / effective independent bets**, **factor attribution (strategy α/β + spread market-neutrality)**, and any structural-break / walk-forward work (or its absence).
8. `稳健性与风险信号清单`: table with level, signal, triggering rule, evidence, window/sample, and the test or formula used.
9. `方法附录`: stage-by-stage source table with data window, usable rows, test names, key statistics, and caveats — including an explicit list of advanced steps **not** performed.
10. `未做的分析、数据限制与假设`: analyses not run, every zeus capability gap with its fallback, config assumptions, and execution parameters.
11. `敏感性分析`: the same snapshot re-run with one parameter changed per row (fees ×2, slippage +1 tick, train 60%/80%, 20 lots, one more pre-delivery month); a conclusion that flips is not robust.
12. `自动对账`: independent recomputation from the output CSVs (net = gross − costs, per-trade fees and slippage, gross PnL from fills + marks, roll decision days, training β and EG p) — every row must pass.

Above chapter 1 the report shows a key-number table, the reconciliation/sensitivity status, a **checklist (each test's result next to its pass standard and a ✓/△/✗ verdict, so readers never need to remember thresholds)** and an overview figure (four charts stacked vertically); each figure carries a "看图要点" line saying what normal looks like. `report.html` is the same report, self-contained.

The header states two separate verdicts — **statistical evidence** and **trading feasibility** — because a stationary spread is not automatically a tradable strategy.

## Evidence And Output Requirements

- Include at least one source/method table in the appendix: `分析阶段`, `数据来源/方法`, `查询或样本窗口`, `可用样本量`, `关键统计量/参数`, `备注`.
- For each robustness flag, include the test name, statistic/p-value, and window in the same row or the next sentence.
- Report cointegration on the training window, never silently on the full sample alongside a train-only β.
- Always report a Sharpe **with its t-statistic**; never present a Sharpe number as an edge without its significance.
- Report both gross and net; the net cost model uses per-leg futures fees and slippage, and margin usage is reported.
- If a section has no usable data or insufficient sample, keep the heading and state method, window, and what is missing.
- Disclose multiple-testing exposure when a universe was screened.
- Keep the tone analytical and non-promotional; avoid buy/sell language. Prefer "可能存在均值回归", "需要样本外确认", "Sharpe 与 0 不可区分", "扣费后优势消失".

## Final QA Checklist

- Symbols and the studied pair are normalized and displayed consistently.
- Data window and usable sample size appear in each major section.
- Derived metrics have formulas or field/series names.
- Cointegration is tested **on the training window** (ADF + KPSS), reported with critical values, not assumed.
- Hedge ratio and thresholds were estimated on the training window only; an untouched OOS result is reported.
- The Sharpe is reported with its t-statistic, and a sub-2 t is called out as non-significant.
- Costs are modeled per leg (fees + slippage) with margin usage; both gross and net are shown.
- Every roll in `rolls.csv` was decided from prior-day information; the report distinguishes the research continuous series from executable contract prices.
- Executable results use actual contracts and integer lots; the lot rule, realized hedge, deferred fills, and liquidity flags are reported.
- Each contract-spec value is traced to zeus or to a disclosed config assumption; every zeus capability gap is listed with its fallback.
- Cross-commodity spreads state their economic rationale; statistical evidence and trading feasibility have separate verdicts.
- The report carries the run_id and snapshot hash so it can be replayed offline.
- In-sample/out-of-sample inversion (good OOS, bad IS) is checked, not just IS-good/OOS-bad overfitting.
- Advanced steps not performed (Johansen, Kalman, Chow/CUSUM, walk-forward) are listed as not done, not implied.
- Multiple-testing exposure is disclosed when a universe was screened.
- Empty or insufficient data is disclosed rather than hidden.
- Final disclaimer is present exactly as required by `SKILL.md`.
