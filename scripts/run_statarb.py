#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "statsmodels>=0.14"]
# ///
"""
run_statarb.py — Statistical Arbitrage & Time Series Skill 的可执行骨架（优化版）。

设计目标：把 SKILL.md / statarb-guide.md 里"测什么、怎么判"翻译成真能跑出
ADF/KPSS 统计量、半衰期、对冲比率稳定性、样本外 Sharpe 及其显著性、扣费净值
的代码，并按报告蓝图输出 Markdown。本版本的核心是**主动证伪**：宁可漏掉一个
机会，也不要把一个不显著 / 有泄漏 / 扣费即亏的价差当成可交易结论放出去。

相对上一版的关键修正（每条都对应一个真实踩过的坑）：
  1. 协整检验改在"训练窗价差"上做（标题结论），并同时给全样本 / 样本外对照，
     消除"β 用训练窗、ADF 却用全样本"的前视泄漏。
  2. 新增 KPSS（与 ADF 互为对偶零假设），并实现 ADF×KPSS 矛盾的中风险规则。
  3. Sharpe 给出 t 统计量与"有效独立下注次数"（按持仓重叠/半衰期折减），
     新增"Sharpe 与 0 不可区分"的风险规则——0.95 的样本外 Sharpe 若 t≈1.3 不算证据。
  4. 修补规则盲区：样本内不赚钱但样本外赚钱（IS≤0 且 OOS>0）现在会被判为高风险，
     而不是被原"过拟合"规则悄悄放过。
  5. 对冲比率稳定性：分段估计 β，报告漂移幅度，作为结构断点的廉价代理（高/中规则）。
  6. z-score 回看窗默认由半衰期自适应；若窗 < 半衰期会显式告警（短窗制造伪信号）。
  7. 成本为期货口径：逐腿 手续费(费率/每手) + 滑点(跳数×最小变动价位)，开/平各计单边，
     两腿按 1:|β| 名义加权；报告保证金占用与保证金年化收益。合约参数必须在配置里逐腿给出。
  8. 交易计数改为"完成往返"(开→平 成对)，并报告期末未平仓，避免把开仓次数当往返。
  9. 半衰期公式与 guide 对齐：half-life = -ln2 / ln(1+b)。

两条路径：
  - 研究路径：数据只来自 zeus MCP（期货行情，DolphinDB 在 MCP 之后，脚本不直连）。
    --fetch 按配置调 fut_daily，把响应原样写成快照 JSON；随后（或任何时候）用
    --config 离线回放，输出 report.md + manifest.json；统计优先用 statsmodels。
  - 自测路径（无数据/无 statsmodels）：--source synthetic，ADF/KPSS 用本文件内置的
    numpy 实现（近似，仅供机器自测）；真实研究请装 statsmodels 用其精确 p 值。

注意：本骨架实现的是 OLS 对冲比率 + ADF/KPSS + 分段 β 稳定性 + 单次样本外回测。
Johansen、Kalman 动态对冲、Chow/CUSUM、滚动前推(walk-forward)、价差收益的因子归因
属于 Agent 在 references/statarb-guide.md 指引下补充的进阶分析，不在本脚本内，
请勿在报告里把它们写成"已自动完成"。

用法示例：
  export ZEUS_MCP_URL=http://<host>:8000/mcp ZEUS_MCP_TOKEN=<token>
  python run_statarb.py --config run1/config.json --fetch --out-dir run1   # 取数 + 回放
  python run_statarb.py --config run1/config.json --out-dir run1b          # 仅回放已有快照
  python run_statarb.py --source synthetic --mode strong                   # 离线自测
"""
import argparse, datetime as dt, hashlib, json, os, subprocess, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- statsmodels 是硬依赖：p 值必须来自真实检验，不做近似回退 ----------
try:
    import statsmodels
    from statsmodels.tsa.stattools import adfuller as _sm_adf, kpss as _sm_kpss, coint as _sm_coint
except ImportError as _e:
    raise SystemExit("需要 statsmodels 才能得到真实的 ADF/KPSS/协整 p 值：pip install -r requirements.txt") from _e

import futures as fx   # 期货规则层：合约映射/换月、研究序列、整手可执行回测


# ============ 1. 数据层 ============
# 真实行情只经 zeus MCP 获取：fetch_snapshot() 调 fut_daily 落成快照 JSON，
# replay() 只读快照；不直连数据库。合成数据仅供离线自测。
LEG_SPEC_KEYS = ("multiplier", "tick_size", "fee_rate_bps", "fee_per_lot",
                 "slippage_ticks", "margin_rate")
# 合成自测用的示意合约参数（非任何真实品种）；真实研究必须在配置里逐腿给出
SYNTH_LEG = {"multiplier": 10, "tick_size": 0.01, "fee_rate_bps": 1.0, "fee_per_lot": 0,
             "slippage_ticks": 1, "margin_rate": 0.10}


_SYNTH_MKT = None   # 合成市场因子日收益，供 load_market 取用


def _synthetic_pair(n=1200, cointegrated=True, seed=7, mode=None):
    """生成可控的合成价格对（对数空间，β_true≈1，价差=平稳 OU）。
    mode: None→沿用 cointegrated；'coint'/'nocoint'/'strong'/'inversion'/'leaked'/'drift'。
    A 直接由 B 定义（log_a = log_b + 价差），故价差恰为平稳 OU、对冲比率稳定。
    leaked 模式把市场因子注入 OU 的**创新项**：价差仍均值回归，但其日变化与市场相关
    → 用于演示"价差非市场中性 / 漏 beta"。市场因子 M 与 B 独立，作为回归代理。
    drift 模式让 A 对 B 的同期载荷随时间从 0.6 线性升到 1.6 → 真实对冲比率漂移。"""
    global _SYNTH_MKT
    if mode is None:
        mode = "coint" if cointegrated else "nocoint"
    if mode == "strong":
        n = 2600
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)

    m_ret = rng.normal(0, 0.008, n)               # 独立市场因子日收益（回归代理）
    _SYNTH_MKT = pd.Series(m_ret, index=idx, name="mkt")

    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.012, n))

    if mode == "nocoint":                         # A 与 B 各自独立游走，不协整
        log_a = np.log(90) + np.cumsum(rng.normal(0, 0.013, n))
        return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)

    if mode == "drift":                           # 同期载荷随时间漂移 → 真实对冲比率不稳
        rb = np.diff(log_b, prepend=log_b[0])
        k = np.linspace(0.6, 1.6, n)              # 载荷 0.6 → 1.6
        log_a = np.log(90) + np.cumsum(k * rb + rng.normal(0, 0.004, n))
        return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)

    # 价差为平稳 OU；inversion 前70%不回归、后30%回归；leaked 在创新项注入市场
    phi_default = 0.87 if mode == "strong" else 0.94
    cut = int(n * 0.70)
    gamma = 1.5 if mode == "leaked" else 0.0
    spread = np.zeros(n)
    for t in range(1, n):
        phi = (1.0 if t < cut else 0.80) if mode == "inversion" else phi_default
        spread[t] = phi * spread[t-1] + rng.normal(0, 0.04) + gamma * m_ret[t]

    log_a = np.log(90) + (log_b - np.log(50)) + spread   # log_a − log_b = const + 平稳价差
    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)


# ============ 2. 检验工具 ============
def adf_test(x):
    """ADF（H0: 有单位根）。返回 (统计量, p, 结论文本)。"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stat, p, *_ = _sm_adf(x, autolag="AIC")
    return float(stat), float(p), _adf_verdict(p)


def kpss_test(x):
    """KPSS（H0: 平稳）。返回 (统计量, p, 结论文本)。与 ADF 互为对偶：两者都指向平稳才是干净证据。"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = _sm_kpss(x, regression="c", nlags="auto")
    return float(stat), float(p), _kpss_verdict(p)


def seasonality_test(spread):
    """价差日变化按自然月分组的 Kruskal-Wallis 检验（H0: 各月分布相同），只应在训练窗上调用。
    返回 (H, p, 月份数)；可用月份 <3（每月 ≥10 个观测）时返回 None。"""
    from scipy.stats import kruskal
    d = pd.Series(np.asarray(spread, float), index=spread.index).diff().dropna()
    groups = [g.to_numpy() for _, g in d.groupby(d.index.month) if len(g) >= 10]
    if len(groups) < 3:
        return None
    h, p = kruskal(*groups)
    return float(h), float(p), len(groups)


def eg_test(log_a, log_b):
    """Engle-Granger 协整（H0: 不协整），MacKinnon p 值；只应在训练窗上调用。返回 (统计量, p)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _ = _sm_coint(np.asarray(log_a, float), np.asarray(log_b, float))
    return float(stat), float(p)


def _adf_verdict(p):
    if p <= 0.05:
        return "拒绝单位根 → 价差平稳，可能存在可交易协整关系"
    if p <= 0.10:
        return "协整边际，需样本外确认"
    return "无法拒绝单位根 → 不协整，无可交易价差"


def _kpss_verdict(p):
    # KPSS 的 p 被截断在 [0.01, 0.10]；p 小=拒绝平稳
    if p <= 0.05:
        return "拒绝平稳（与可交易价差相悖）"
    return "无法拒绝平稳（与 ADF 若一致则为干净证据）"


def half_life(spread):
    """AR(1)：Δs_t = a + b s_{t-1} + e；half-life = -ln2 / ln(1+b)（与 guide 对齐）。"""
    s = np.asarray(spread, float)
    s = s[~np.isnan(s)]
    ds, lag = np.diff(s), s[:-1]
    b = np.polyfit(lag, ds, 1)[0]
    if b >= 0 or b <= -1:        # 不回归 或 过冲不稳定
        return np.inf
    return -np.log(2) / np.log(1 + b)


def _ols_beta_se(x, y):
    """单变量 OLS 斜率及其标准误。"""
    bk, ak = np.polyfit(x, y, 1)
    resid = y - (ak + bk * x)
    dof = max(len(x) - 2, 1)
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt((resid @ resid / dof) / sxx)) if sxx > 0 else np.nan
    return float(bk), se


def beta_stability(log_a, log_b, n_chunks=6):
    """对冲比率稳定性诊断（在**对数收益**空间，避免对 I(1) 价格分段回归的伪不稳定）。

    两个互补量：
      1) Chow 式半样本断点检验（主信号、有功效）：前半 vs 后半各估 β，
         z=(β1−β2)/sqrt(SE1²+SE2²)；显著且经济幅度可观即判结构漂移。
      2) 6 段噪声校正离散度 excess_drift（辅助色彩、功效弱）：
         sqrt(max(0, var(β块)−mean(SE块²)))/max(|mean|,0.1)。
    价差所用对冲比率仍是水平协整回归的 β，这里仅作稳定性诊断；权威检验用 Chow/CUSUM。
    """
    ra = np.diff(np.asarray(log_a, float))
    rb = np.diff(np.asarray(log_b, float))
    n = len(ra)

    # --- 6 段离散度（辅助）---
    betas, ses = [], []
    for k in range(n_chunks):
        s, e = k * n // n_chunks, (k + 1) * n // n_chunks
        if e - s < 20:
            continue
        bk, se = _ols_beta_se(rb[s:e], ra[s:e])
        betas.append(bk); ses.append(se)
    betas = np.array(betas, float); ses = np.array(ses, float)
    mean = float(betas.mean()) if len(betas) else float("nan")
    denom = max(abs(mean), 0.1)
    raw_drift = float((betas.max() - betas.min()) / denom) if len(betas) else float("inf")
    between_var = float(np.var(betas, ddof=1)) if len(betas) > 1 else 0.0
    samp_var = float(np.nanmean(ses ** 2)) if len(ses) else 0.0
    excess_drift = float(np.sqrt(max(0.0, between_var - samp_var)) / denom)

    # --- Chow 式半样本断点（主信号）---
    half = n // 2
    b1, se1 = _ols_beta_se(rb[:half], ra[:half])
    b2, se2 = _ols_beta_se(rb[half:], ra[half:])
    se_diff = float(np.sqrt(np.nansum([se1 ** 2, se2 ** 2])))
    chow_z = float((b1 - b2) / se_diff) if se_diff > 0 else 0.0
    chow_rel = float(abs(b1 - b2) / denom)

    return dict(mean=mean, raw_drift=raw_drift, excess_drift=excess_drift,
                between_std=float(np.sqrt(between_var)),
                samp_std=float(np.sqrt(samp_var)),
                betas=betas.round(4).tolist(),
                b_first=b1, b_second=b2, chow_z=chow_z, chow_rel=chow_rel)


def load_market(px):
    """合成自测用的市场因子日收益（因子归因演示）。回放路径无市场代理 → 不做归因。"""
    if _SYNTH_MKT is not None:
        return _SYNTH_MKT.reindex(px.index)
    return np.log(px[px.columns[1]]).diff()


def factor_attribution(pnl, mkt_ret):
    """把策略日收益对市场代理收益做 OLS，分离方向性 beta 与残差 alpha。
    返回 beta, t_beta, alpha_annual, t_alpha, r2, n；样本不足返回 None。
    判读：若 t_beta 显著而 t_alpha 不显著，则"优势"很可能只是漏进来的板块 beta。"""
    df = pd.concat([pd.Series(pnl).rename("y"),
                    pd.Series(mkt_ret).rename("x")], axis=1, sort=False).dropna()
    if len(df) < 30 or df["x"].std() == 0:
        return None
    x = df["x"].to_numpy(); y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - 2, 1)
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    alpha, beta = float(coef[0]), float(coef[1])
    t_alpha = alpha / se[0] if se[0] > 0 else 0.0
    t_beta = beta / se[1] if se[1] > 0 else 0.0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=beta, t_beta=float(t_beta), alpha_annual=float(alpha * 252),
                t_alpha=float(t_alpha), r2=float(r2), n=int(len(df)))


def _fmt_p(p, lo=0.01, hi=0.10):
    """被截断在 [lo,hi] 的 p 值显示为 >hi / <lo（各按其自然精度），
    避免把 0.100 误读成精确等于、或把下界 0.001 截断成 0.00。"""
    def _num(x):
        for d in (2, 3, 4):
            s = f"{x:.{d}f}"
            if float(s) == x:
                return s
        return f"{x:.4f}"
    if p is None:
        return "—"
    if p >= hi:
        return f">{_num(hi)}"
    if p <= lo:
        return f"<{_num(lo)}"
    return f"{p:.3f}"


# ============ 3. 回测（样本外隔离 + 现实成本 + 显著性） ============
def _unit_cost(price, spec):
    """每单位名义本金的单边成本（分数）= 手续费率 + (每手固定费 + 滑点跳数·跳价·乘数)/(价格·乘数)。"""
    return (spec["fee_rate_bps"] / 1e4
            + (spec["fee_per_lot"] + spec["slippage_ticks"] * spec["tick_size"] * spec["multiplier"])
            / (price * spec["multiplier"]))


def backtest(px, legs, train_frac=0.70, window=None, entry=2.0, exit=0.5, stop=3.5):
    """
    期货口径成本（以"价差对数收益"近似的分数计；1 单位价差 = A 腿 1 份名义 + B 腿 |β| 份名义）：
      - legs[腿] = {multiplier 合约乘数, tick_size 最小变动价位, fee_rate_bps 按成交额手续费,
                    fee_per_lot 每手固定手续费(元), slippage_ticks 每次成交滑点跳数, margin_rate 保证金率}
      - 开仓、平仓各计一次单边成本（日线无平今，不区分）；期货无印花税、无融券 carry。
      - 保证金占用/单位价差名义 = margin_A + |β|·margin_B；报告样本外年化收益 / 保证金。
    window=None 时，z-score 回看窗按半衰期自适应（clamp 到 [20,120]）。
    """
    a, b = px.columns
    log = np.log(px)
    split = int(len(px) * train_frac)

    # 对冲比率仅用训练窗估计（防 β 前视）
    beta = float(np.polyfit(log[b].iloc[:split], log[a].iloc[:split], 1)[0])
    spread = log[a] - beta * log[b]

    # 半衰期（训练窗）→ 自适应 z 窗
    hl_train = half_life(spread.iloc[:split])
    if window is None:
        hl_for_win = hl_train if np.isfinite(hl_train) else 30
        window = int(min(120, max(20, round(hl_for_win))))

    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    z = (spread - mu) / sd

    pos = np.zeros(len(px)); state = 0
    for t in range(len(px)):
        zt = z.iloc[t]
        if np.isnan(zt):
            pos[t] = 0; continue
        if state == 0:
            if zt > entry: state = -1
            elif zt < -entry: state = 1
        elif state == 1 and (zt >= -exit or zt < -stop):
            state = 0
        elif state == -1 and (zt <= exit or zt > stop):
            state = 0
        pos[t] = state
    pos = pd.Series(pos, index=px.index)

    dspread = spread.diff().fillna(0)
    gross = pos.shift(1).fillna(0) * dspread

    turn = pos.diff().abs().fillna(0)             # 每次开/平 = 1 次单边
    cost_a = turn * _unit_cost(px[a], legs[a])
    cost_b = turn * abs(beta) * _unit_cost(px[b], legs[b])
    net = gross - cost_a - cost_b
    margin = legs[a]["margin_rate"] + abs(beta) * legs[b]["margin_rate"]

    def metrics(r, n_eff_div=1.0):
        r = r.dropna()
        if len(r) == 0 or r.std() == 0:
            return dict(sharpe=0.0, mdd=0.0, ret=0.0, t=0.0, t_eff=0.0, n=len(r))
        sharpe = r.mean() / r.std() * np.sqrt(252)
        eq = r.cumsum()
        mdd = (eq - eq.cummax()).min()
        years = len(r) / 252.0
        t_naive = sharpe * np.sqrt(years)                    # iid 假设下的 Sharpe t
        t_eff = sharpe * np.sqrt(years / max(1.0, n_eff_div))  # 持仓重叠折减后的 t
        return dict(sharpe=float(sharpe), mdd=float(mdd), ret=float(eq.iloc[-1]),
                    t=float(t_naive), t_eff=float(t_eff), n=len(r))

    is_slice, oos_slice = slice(0, split), slice(split, None)
    # 持仓重叠：相邻日强相关，有效独立下注数 ≈ 天数 / 半衰期
    n_eff_div = hl_train if np.isfinite(hl_train) and hl_train > 1 else 1.0

    # 完成往返 = 开仓(0→±1) 后又平仓(±1→0) 的成对数；并统计期末未平仓
    p_oos = pos.iloc[oos_slice].to_numpy()
    completed, open_at_end = _count_round_trips(p_oos)

    oos_n = len(px) - split
    oos_ret_ann = float(net.iloc[oos_slice].sum()) / max(oos_n / 252.0, 1e-9)
    return dict(
        oos_return_on_margin_ann=oos_ret_ann / margin if margin > 0 else float("nan"),
        beta=beta, spread=spread, z=z, split=split, window=window,
        hl_train=hl_train,
        is_gross=metrics(gross.iloc[is_slice]),
        is_net=metrics(net.iloc[is_slice]),
        oos_gross=metrics(gross.iloc[oos_slice], n_eff_div),
        oos_net=metrics(net.iloc[oos_slice], n_eff_div),
        n_round_trips_oos=completed, open_at_end_oos=open_at_end,
        oos_net_pnl=net.iloc[oos_slice],
        pos=pos, margin_per_unit=float(margin), legs=legs,
        cost=dict(oos_total=float((cost_a + cost_b).iloc[oos_slice].sum()),
                  oos_by_leg={a: float(cost_a.iloc[oos_slice].sum()),
                              b: float(cost_b.iloc[oos_slice].sum())}),
    )


def _count_round_trips(pos_arr):
    """统计完成往返（进场→出场成对）次数与期末未平仓数。"""
    completed, in_pos = 0, False
    for p in pos_arr:
        if not in_pos and p != 0:
            in_pos = True
        elif in_pos and p == 0:
            completed += 1
            in_pos = False
    return completed, (1 if in_pos else 0)


# ============ 4. 稳健性规则引擎（对应 statarb-guide.md） ============
def robustness_flags(adf_p, kpss_p, hl, bt, betastab, attr=None, attr_sp=None):
    flags = []

    # --- 协整 / 平稳 ---
    if adf_p > 0.10:
        flags.append(("🔴 高", "不协整", "价差 ADF p>0.10", f"ADF p={_fmt_p(adf_p, 0.001, 0.99)}"))
    elif adf_p > 0.05:
        flags.append(("🟡 中", "协整边际", "ADF p∈(0.05,0.10]", f"ADF p={_fmt_p(adf_p, 0.001, 0.99)}"))
    if (kpss_p is not None) and (adf_p <= 0.05) and (kpss_p <= 0.05):
        flags.append(("🟡 中", "ADF×KPSS 矛盾", "ADF判平稳但KPSS拒绝平稳",
                      f"ADF p={_fmt_p(adf_p, 0.001, 0.99)} / KPSS p={_fmt_p(kpss_p)}"))

    # --- 半衰期 ---
    if not np.isfinite(hl):
        flags.append(("🔴 高", "不回归", "AR(1) 系数≥0，价差不均值回归", "half-life=∞"))
    elif hl > 60:
        flags.append(("🔴 高", "半衰期过长", "half-life>60 交易日", f"{hl:.1f} 天"))
    elif hl > 20:
        flags.append(("🟡 中", "半衰期偏长", "half-life∈(20,60]", f"{hl:.1f} 天"))

    # --- 对冲比率稳定性：Chow 式半样本断点为主信号，excess_drift 为辅助 ---
    cz, crel = abs(betastab["chow_z"]), betastab["chow_rel"]
    ed = betastab["excess_drift"]
    if cz >= 2.5 and crel > 0.2:        # 统计显著 + 经济幅度可观
        lvl = "🔴 高" if cz >= 4 else "🟡 中"
        flags.append((lvl, "对冲比率结构漂移",
                      "前/后半样本 β 显著不同（Chow 式，收益空间）",
                      f"β_前={betastab['b_first']:.2f} vs β_后={betastab['b_second']:.2f}, "
                      f"z={betastab['chow_z']:.2f}, 相对差={crel:.2f}, excess_drift={ed:.2f}；"
                      "建议 Chow/CUSUM 正式复核"))

    # --- 回测：扣费 / 过拟合 / 样本内外倒挂 / 显著性 ---
    is_s = bt["is_net"]["sharpe"]
    oos_s = bt["oos_net"]["sharpe"]
    oos_t = bt["oos_net"]["t"]
    oos_t_eff = bt["oos_net"]["t_eff"]

    if oos_s <= 0:
        flags.append(("🔴 高", "扣费后归零", "样本外净Sharpe≤0", f"OOS净Sharpe={oos_s:.2f}"))
    is_t = bt["is_net"]["t"]
    if (is_s <= 0.05 or is_t < 1.0) and oos_s > 0.5:     # 样本内与 0 不可区分（Sharpe≈0 或 t<1）
        flags.append(("🔴 高", "样本外优势缺乏样本内支撑",
                      "样本内净Sharpe≈0/负或t<1，但样本外>0.5（疑似区制依赖/机缘，非稳定edge）",
                      f"IS净={is_s:.2f}(t={is_t:.2f}) / OOS净={oos_s:.2f}"))
    if is_s > 0 and oos_s < 0.5 * is_s:
        flags.append(("🔴 高", "疑似过拟合", "样本外Sharpe<样本内一半",
                      f"IS净={is_s:.2f} / OOS净={oos_s:.2f}"))
    if oos_s > 0 and abs(oos_t) < 1.96:
        flags.append(("🟡 中", "Sharpe不显著", "样本外Sharpe的|t|<1.96，与0不可区分",
                      f"Sharpe={oos_s:.2f}, t≈{oos_t:.2f}（重叠折减后 t≈{oos_t_eff:.2f}）"))

    # --- 样本量 ---
    if bt["n_round_trips_oos"] < 30:
        flags.append(("🟡 中", "样本不足", "样本外完成往返<30笔",
                      f"{bt['n_round_trips_oos']} 笔（期末未平仓 {bt['open_at_end_oos']}）"))

    # --- z 窗 vs 半衰期 ---
    if np.isfinite(hl) and bt["window"] < hl:
        flags.append(("🟡 中", "z窗短于半衰期", "回看窗<半衰期，易制造伪信号",
                      f"window={bt['window']} < half-life={hl:.1f}"))

    # --- 因子归因：优势是否只是漏进来的方向性 beta ---
    # (1) 价差本身是否市场中性（对应低对冲比率/两腿同向的担忧）
    if attr_sp is not None and abs(attr_sp["t_beta"]) >= 1.96:
        lvl = "🔴 高" if attr_sp["r2"] >= 0.10 else "🟡 中"
        flags.append((lvl, "价差非市场中性",
                      "价差日收益对市场回归 beta 显著",
                      f"β_spread={attr_sp['beta']:.2f}(t={attr_sp['t_beta']:.2f}), R²={attr_sp['r2']:.2f}"
                      "（价差含方向性敞口，收益可能非纯 alpha）"))
    # (2) 已实现策略收益扣除市场后是否仍有 alpha
    if attr is not None and oos_s > 0:
        if abs(attr["t_beta"]) >= 1.96 and abs(attr["t_alpha"]) < 1.96:
            flags.append(("🔴 高", "疑似漏入方向性beta",
                          "策略收益市场beta显著但残差alpha不显著",
                          f"β_mkt={attr['beta']:.2f}(t={attr['t_beta']:.2f}), "
                          f"α年化={attr['alpha_annual']:.3f}(t={attr['t_alpha']:.2f}), R²={attr['r2']:.2f}"))
        elif abs(attr["t_alpha"]) < 1.96:
            flags.append(("🟡 中", "残差alpha不显著",
                          "扣除市场后残差alpha的|t|<1.96",
                          f"α年化={attr['alpha_annual']:.3f}(t={attr['t_alpha']:.2f})"))

    if not flags:
        flags.append(("🟢 低", "未触发高/中规则", "—", "通过基础稳健性检查"))

    # 高风险优先排前
    order = {"🔴 高": 0, "🟡 中": 1, "🟢 低": 2}
    flags.sort(key=lambda x: order.get(x[0], 9))
    return flags


# ============ 5. 报告输出（对应 9 章蓝图，精简版） ============
def _stat_tradable(s):
    f, attr = s["bt"], s["attr"]
    alpha_ok = (attr is None) or (abs(attr["t_alpha"]) >= 1.96)
    return (s["adf_tr"][1] <= 0.05 and np.isfinite(s["hl_tr"]) and s["hl_tr"] <= 60
            and f["oos_net"]["sharpe"] > 0 and abs(f["oos_net"]["t"]) >= 1.96
            and f["n_round_trips_oos"] >= 30 and alpha_ok)


def write_report(a, b, source, px, s, flags, path, header=(), insert=None):
    """s = run_stats() 的结果；header 插在标题后；insert = {章节标题前缀: 行列表} 插在该章节之前。"""
    bt = f = s["bt"]
    adf_tr, kpss_tr, adf_full, adf_oos = s["adf_tr"], s["kpss_tr"], s["adf_full"], s["adf_oos"]
    hl_tr, hl_full, betastab, attr, attr_sp, eg = (s["hl_tr"], s["hl_full"], s["betastab"], s["attr"],
                                                  s["attr_sp"], s["eg"])
    kernel = f"statsmodels {statsmodels.__version__}"
    top = flags[0][0]
    verdict = ("证据指向可进一步研究" if _stat_tradable(s)
               else "证据不足以支持可交易结论（建议否决/继续证伪）")
    tr_pct = round(100 * bt["split"] / len(px))

    if attr is None:
        attr_line = ("- 因子归因：未做（缺市场代理序列；zeus 暂无指数数据）。β 低且两腿同向时，"
                     "样本外收益可能混入方向性 beta，有指数序列后应对其回归再下结论。")
    else:
        sp = ""
        if attr_sp is not None:
            neutral = "市场中性成立" if abs(attr_sp["t_beta"]) < 1.96 else "**价差非市场中性**"
            sp = (f" 价差中性检验：β_spread={attr_sp['beta']:.2f}(t={attr_sp['t_beta']:.2f}), "
                  f"R²={attr_sp['r2']:.2f} → {neutral}。")
        a_sig = abs(attr["t_alpha"]) >= 1.96
        b_sig = abs(attr["t_beta"]) >= 1.96
        if a_sig:
            judge = "残差 α 显著，优势不主要来自方向性 beta。"
        elif b_sig:
            judge = "**残差 α 不显著且市场 β 显著：优势很可能只是漏进来的板块 beta。**"
        else:
            judge = "**残差 α 不显著（市场 β 也不显著）：优势在统计上与 0 不可区分，更像噪声而非真 alpha。**"
        attr_line = (f"- 因子归因（样本外）：策略净收益 β_mkt={attr['beta']:.2f}"
                     f"(t={attr['t_beta']:.2f})，残差 α 年化={attr['alpha_annual']:.3f}"
                     f"(t={attr['t_alpha']:.2f})，R²={attr['r2']:.2f}。"
                     + judge + sp)

    lines = [
        f"# 统计套利研究报告：{a} × {b}",
        *header,
        f"\n> 数据源：`{source}`｜统计内核：{kernel}"
        f"｜样本：{px.index.min().date()} ~ {px.index.max().date()}，共 {len(px)} 个交易日"
        f"｜训练/测试={tr_pct}/{100 - tr_pct}（最近 {100 - tr_pct}% 留出）\n",

        "## 1. 摘要与结论",
        f"- **总体判断：{verdict}**（最高风险等级 {top}，详见第 8 章）",
        f"- 对冲比率 β（仅训练窗估计）= **{f['beta']:.4f}**；价差 = log({a}) − β·log({b})",
        f"- 协整（训练窗价差）：ADF stat={adf_tr[0]:.3f}, p={_fmt_p(adf_tr[1], 0.001, 0.99)}；"
        f"KPSS stat={kpss_tr[0]:.3f}, p={_fmt_p(kpss_tr[1])} → {adf_tr[2]}",
        f"- 均值回归半衰期（训练窗）= **{hl_tr:.1f} 交易日**（z 回看窗自适应={f['window']}）",
        f"- 样本外（研究口径：价差对数收益、近似成本）：Sharpe **{f['oos_net']['sharpe']:.2f}**"
        f"（t≈{f['oos_net']['t']:.2f}，持仓重叠折减后 t≈{f['oos_net']['t_eff']:.2f}）；"
        f"最大回撤 {f['oos_net']['mdd']:.4f}；完成往返 {f['n_round_trips_oos']} 笔",
        attr_line,
        "- 提示：样本外 Sharpe 的 |t|<2 即无法在统计上与 0 区分；绿灯门槛同时要求"
        "完成往返≥30 笔且残差 α 显著，故慢回归/样本少的对子通常会被判为否决。",

        "\n## 4. 协整与平稳性检验",
        "| 检验 | 窗口 | 统计量 | p | 结论 |",
        "|---|---|---|---|---|",
        f"| ADF | 训练窗（标题判定） | {adf_tr[0]:.3f} | {_fmt_p(adf_tr[1], 0.001, 0.99)} | {adf_tr[2]} |",
        f"| KPSS | 训练窗 | {kpss_tr[0]:.3f} | {_fmt_p(kpss_tr[1])} | {kpss_tr[2]} |",
        f"| Engle-Granger 协整 | 训练窗（对数价） | {eg[0]:.3f} | {_fmt_p(eg[1], 0.001, 0.99)} | "
        f"{'拒绝“不协整”' if eg[1] <= 0.05 else '无法拒绝“不协整”'}（MacKinnon p 值） |",
        f"| ADF | 全样本（对照） | {adf_full[0]:.3f} | {_fmt_p(adf_full[1], 0.001, 0.99)} | — |",
        f"| ADF | 样本外（对照） | {adf_oos[0]:.3f} | {_fmt_p(adf_oos[1], 0.001, 0.99)} | — |",
        "\n说明：标题协整结论只用训练窗价差，避免 β 用训练窗、ADF 用全样本造成的前视泄漏；"
        "全样本/样本外仅作对照。ADF 与 KPSS 零假设相反，两者都指向平稳才是干净证据。"
        "KPSS 的 p 被截断在 [0.01,0.10]，故显示为 >0.10 / <0.01。",

        "\n## 5. 价差建模与均值回归",
        f"对冲比率 β={f['beta']:.4f}（OLS，仅训练窗，水平协整回归，价差据此构造）。"
        f"分段 β 稳定性诊断（在**对数收益**空间估计，避免对 I(1) 价格分段回归的伪不稳定）："
        f"mean={betastab['mean']:.4f}，块间std={betastab['between_std']:.4f}，"
        f"抽样std={betastab['samp_std']:.4f}，**excess_drift={betastab['excess_drift']:.2f}**"
        f"（raw_drift={betastab['raw_drift']:.2f}），分段值={betastab['betas']}。"
        f"excess_drift 已扣除小样本抽样噪声，仅超出噪声的系统性漂移才计为风险；"
        f"半样本断点（Chow 式）：β_前={betastab['b_first']:.3f} vs β_后={betastab['b_second']:.3f}, "
        f"z={betastab['chow_z']:.2f}（|z|≥2.5 且相对差>0.2 判结构漂移）。"
        f"权威断点检验请用 Chow/CUSUM。",
        f"半衰期 = −ln2/ln(1+b)：训练窗 **{hl_tr:.1f} 天**，全样本 {hl_full:.1f} 天。",
        (f"季节性（训练窗价差日变化按月 Kruskal-Wallis，{s['season'][2]} 个月）：H={s['season'][0]:.2f}，"
         f"p={_fmt_p(s['season'][1], 0.001, 0.99)} → {'各月分布存在显著差异，需按季节拆分验证' if s['season'][1] < 0.05 else '未见显著季节性'}。"
         if s.get("season") else "季节性：训练窗可用月份不足 3 个（每月需 ≥10 个观测），未检验。"),

        "\n## 7. 回测与偏差控制（研究口径：价差对数收益；毛 vs 净；样本内 vs 样本外）",
        "| 口径 | Sharpe | t | 最大回撤 | 累计价差收益 |",
        "|---|---|---|---|---|",
        f"| 样本内·毛 | {f['is_gross']['sharpe']:.2f} | {f['is_gross']['t']:.2f} | {f['is_gross']['mdd']:.4f} | {f['is_gross']['ret']:.4f} |",
        f"| 样本内·净 | {f['is_net']['sharpe']:.2f} | {f['is_net']['t']:.2f} | {f['is_net']['mdd']:.4f} | {f['is_net']['ret']:.4f} |",
        f"| 样本外·毛 | {f['oos_gross']['sharpe']:.2f} | {f['oos_gross']['t']:.2f} | {f['oos_gross']['mdd']:.4f} | {f['oos_gross']['ret']:.4f} |",
        f"| **样本外·净** | **{f['oos_net']['sharpe']:.2f}** | **{f['oos_net']['t']:.2f}** | {f['oos_net']['mdd']:.4f} | {f['oos_net']['ret']:.4f} |",
        "\n成本（期货口径，按腿；开/平各计一次单边；1 单位价差 = A 腿 1 份名义 + B 腿 |β| 份名义）：",
        "| 腿 | 乘数 | 最小变动价位 | 手续费率(bp) | 每手费(元) | 滑点(跳) | 保证金率 | 样本外成本(价差单位) |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {leg} | {sp['multiplier']:g} | {sp['tick_size']:g} | {sp['fee_rate_bps']:g} | "
          f"{sp['fee_per_lot']:g} | {sp['slippage_ticks']:g} | {sp['margin_rate']:.0%} | "
          f"{f['cost']['oos_by_leg'][leg]:.4f} |" for leg, sp in ((a, f['legs'][a]), (b, f['legs'][b]))],
        f"\n样本外累计成本 ≈ {f['cost']['oos_total']:.4f}；保证金占用 ≈ {f['margin_per_unit']:.3f}"
        f"（每单位价差名义）；研究口径样本外年化净收益 / 保证金 ≈ {f['oos_return_on_margin_ann']:.1%}（金额口径见交易可行性章节）。"
        "研究口径近似成本（合约参数取切分日的值）；换月、涨跌停、整手与金额口径见交易可行性章节（如有）。",
        "t 为 Sharpe 的近似 t 统计量（iid 假设）；持仓重叠会高估 t，"
        f"按半衰期折减后的有效 t≈{f['oos_net']['t_eff']:.2f}，"
        f"有效独立下注 ≈ 天数/半衰期。",

        "\n## 8. 稳健性与风险信号清单",
        "| 风险等级 | 信号 | 触发规则 | 证据 |",
        "|---|---|---|---|",
    ]
    for lv, sig, rule, ev in flags:
        lines.append(f"| {lv} | {sig} | {rule} | {ev} |")

    lines += [
        "\n## 9. 方法附录",
        "| 分析阶段 | 方法 | 样本窗口 | 关键参数/统计量 | 备注 |",
        "|---|---|---|---|---|",
        f"| 协整检验 | ADF+KPSS({kernel}) | 训练窗（标题）+全样本/样本外对照 | "
        f"ADF_tr stat={adf_tr[0]:.3f} p={_fmt_p(adf_tr[1], 0.001, 0.99)}；KPSS_tr p={_fmt_p(kpss_tr[1])} | 防前视 |",
        f"| 价差建模 | OLS 对冲比率 + 收益空间β稳定性(Chow式半样本断点+excess_drift) + AR(1) 半衰期 | 训练窗 | "
        f"β={f['beta']:.4f}, excess_drift={betastab['excess_drift']:.2f}, "
        f"Chow z={betastab['chow_z']:.2f}, hl={hl_tr:.1f} | — |",
        f"| 信号/回测 | 向量化 z-score + 近似成本（研究口径） | 样本内/外 | "
        f"窗={f['window']}(自适应), 开={2.0}, 平={0.5}, 止={3.5} | 完成往返 {f['n_round_trips_oos']} |",
        f"| 显著性 | Sharpe t 统计量 + 重叠折减 | 样本外 | "
        f"t={f['oos_net']['t']:.2f}, t_eff={f['oos_net']['t_eff']:.2f} | |t|<2 视为不显著 |",
        (f"| 因子归因 | 样本外净收益 OLS 回归市场代理 | 样本外 | "
         f"β_mkt={attr['beta']:.2f}(t={attr['t_beta']:.2f}), α年化={attr['alpha_annual']:.3f}"
         f"(t={attr['t_alpha']:.2f}), R²={attr['r2']:.2f} | 残差α不显著则优势存疑(β显著→漏beta；β也不显著→噪声) |"
         if attr is not None else
         "| 因子归因 | 样本外净收益回归市场代理 | 样本外 | 未做（缺市场代理序列） | 待 zeus 提供指数序列 |"),
        "\n**本脚本已实现的稳健性内核**：ADF+KPSS（训练窗，全样本/OOS 对照）、收益空间对冲比率"
        "稳定性（Chow 式半样本 β 断点 + 噪声校正离散度）、AR(1) 半衰期、含成本的样本外回测、"
        "Sharpe t 与重叠折减、两级因子归因（策略 α/β + 价差市场中性）。",
        "**未实现、需 Agent 另行补充的进阶项**：Johansen 协整、Kalman 动态对冲比率、"
        "完整 Chow/CUSUM 多断点/路径检验、滚动前推(walk-forward)、季节性与区制分析。报告中不得把这些写成已自动完成。",
        "\n---",
        "本报告基于公开数据与规则化分析生成，仅供研究参考，不构成任何投资建议。",
    ]
    for key, extra in (insert or {}).items():
        i = next(i for i, l in enumerate(lines) if l.lstrip("\n").startswith(key))
        lines[i:i] = extra
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ============ 分析流水线（CLI 与回放共用） ============
def run_stats(px, legs, mkt=None, window=None, train_frac=0.70):
    """统计证据：训练窗 β/窗口/半衰期 → ADF+KPSS+EG → β 稳定性 → 研究口径回测 → 因子归因（有市场代理时）。"""
    bt = backtest(px, legs, train_frac=train_frac, window=window, entry=ENTRY, exit=EXIT, stop=STOP)
    split, spread = bt["split"], bt["spread"]
    log = np.log(px)
    a, b = px.columns
    s = {"bt": bt, "adf_tr": adf_test(spread.iloc[:split]), "kpss_tr": kpss_test(spread.iloc[:split]),
         "adf_full": adf_test(spread), "adf_oos": adf_test(spread.iloc[split:]),
         "eg": eg_test(log[a].iloc[:split], log[b].iloc[:split]),
         "hl_tr": bt["hl_train"], "hl_full": half_life(spread),
         "betastab": beta_stability(log[a].iloc[:split], log[b].iloc[:split]),
         "season": seasonality_test(spread.iloc[:split]), "attr": None, "attr_sp": None}
    if mkt is not None:
        s["attr"] = factor_attribution(bt["oos_net_pnl"], mkt)                  # 策略收益层
        s["attr_sp"] = factor_attribution(spread.diff().iloc[split:], mkt)      # 价差中性层
    return s


def analyze(px, a, b, source, mkt, out, legs, window=None):
    """合成数据自测路径：统计证据 + 规则 + 报告。"""
    s = run_stats(px, legs, mkt, window)
    bt, attr, attr_sp = s["bt"], s["attr"], s["attr_sp"]
    flags = robustness_flags(s["adf_tr"][1], s["kpss_tr"][1], s["hl_tr"], bt, s["betastab"], attr, attr_sp)
    write_report(a, b, source, px, s, flags, out)
    print(f"[done] {a}×{b} via {source}")
    print(f"  ADF_tr p={_fmt_p(s['adf_tr'][1], 0.001, 0.99)} | EG p={_fmt_p(s['eg'][1], 0.001, 0.99)} "
          f"| KPSS_tr p={_fmt_p(s['kpss_tr'][1])} | half-life={s['hl_tr']:.1f}d | z-win={bt['window']}")
    attr_s = (f"strat β={attr['beta']:.2f}(t={attr['t_beta']:.2f}) α_t={attr['t_alpha']:.2f}"
              + (f" | spread β={attr_sp['beta']:.2f}(t={attr_sp['t_beta']:.2f})" if attr_sp else "")
              if attr is not None else "未做")
    print(f"  OOS net Sharpe={bt['oos_net']['sharpe']:.2f} (t={bt['oos_net']['t']:.2f}) "
          f"| round-trips={bt['n_round_trips_oos']} | factor: {attr_s} | top flag={flags[0][0]} {flags[0][1]}")
    print(f"  report -> {out}")
    return s


# ============ 期货规则与报告章节（回放专用） ============
def combine_flags(stat_flags, fut_flags):
    flags = [f for f in stat_flags if f[1] != "未触发高/中规则"] + list(fut_flags)
    if not flags:
        flags = [("🟢 低", "未触发高/中规则", "—", "通过基础稳健性检查")]
    order = {"🔴 高": 0, "🟡 中": 1, "🟢 低": 2}
    return sorted(flags, key=lambda x: order.get(x[0], 9))


def futures_flags(s, ctx):
    """交易可行性与数据来源相关的期货规则（统计规则见 robustness_flags）。"""
    cfg, exe = ctx["cfg"], ctx["exe"]
    m = exe["metrics"].get("oos", {})
    out = []
    if m.get("days"):
        if m["net"] <= 0:
            out.append(("🔴 高", "整手真实合约回测扣费后不赚钱", "可执行回测样本外净 PnL≤0",
                        f"样本外净 {m['net']:,.0f} 元（毛 {m['gross']:,.0f}，手续费 {m['fees']:,.0f}，"
                        f"滑点 {m['slippage']:,.0f}）"))
        elif abs(m["t"]) < 1.96:
            out.append(("🟡 中", "可执行回测收益不显著", "样本外净收益 Sharpe |t|<1.96",
                        f"Sharpe={m['sharpe']:.2f}, t≈{m['t']:.2f}"))
        st = ctx["stress"].get("oos") or {}
        if m["net"] > 0 and st.get("days") and st["net"] <= 0:
            out.append(("🟡 中", "压力成本下转亏", "手续费×2、滑点+1 跳后样本外净 PnL≤0",
                        f"压力净 {st['net']:,.0f} 元"))
    ev = exe["events"]
    n_def, n_liq = int((ev.kind == "deferred").sum()), int((ev.kind == "liquidity").sum())
    if n_def:
        out.append(("🟡 中", "成交被顺延", "一腿当日无行情 → 两腿都不成交，次日重试",
                    f"{n_def} 个交易日（见 events.csv）"))
    if n_liq:
        out.append(("🟡 中", "流动性不足", f"单笔手数 > {cfg['execution']['max_participation']:.0%}×当日成交量",
                    f"{n_liq} 笔（见 events.csv）"))
    assumed = sorted(k for k in ("multiplier", "price_tick") if "config" in ctx["sources"].get(k, {}))
    if assumed:
        out.append(("🟡 中", "合约乘数/跳价来自配置", "zeus fut_basic 缺失或无对应合约",
                    f"字段 {assumed} 用 config assumptions，请核对交易所合约规格"))
    if s.get("season") and s["season"][1] < 0.05:
        out.append(("🟡 中", "价差存在季节性", "训练窗按月 Kruskal-Wallis p<0.05",
                    f"H={s['season'][0]:.2f}, p={s['season'][1]:.4f}；全样本统计可能掩盖分月差异"))
    if "funding_rate_annual" not in cfg["raw"].get("execution", {}):
        out.append(("🟢 低", "保证金资金成本按 0 计", "config.execution.funding_rate_annual 未设置",
                    "未计保证金占用的机会成本；如需计入请显式配置年化利率"))
    n = cfg["screening"]["n_candidates"]
    if n > 1 and s["eg"][1] > 0.05 / n:
        out.append(("🟡 中", "多重检验未通过", f"筛选 {n} 个候选，Bonferroni 阈值 p≤{0.05 / n:.4g}",
                    f"EG 协整 p={s['eg'][1]:.4f}"))
    bad = [e for e in exe["entries"] if abs(e["realized_hedge"] / max(abs(e["beta"]), 1e-12) - 1) > 0.10]
    if bad:
        out.append(("🟡 中", "整手取整偏离模型对冲比", "实际名义对冲比与 |β| 相差 >10%",
                    f"{len(bad)}/{len(exe['entries'])} 次开仓；可增大 base_lots"))
    return out


def _leg_desc(leg, excl):
    if leg["select"] == "fixed":
        return f"`{leg['ts_code']}`（固定合约，不换月）"
    if leg["select"] == "dominant":
        return (f"{leg['product']}.{leg['exchange']} 主力（前一交易日持仓量最大；粘性、只向远月换；"
                f"交割月前 {excl} 个月起禁持并强制换出）")
    return f"{leg['product']}.{leg['exchange']} 次主力（交割月晚于近月腿的合约中，前一交易日持仓量最大）"


def _money_row(label, m):
    if not m or not m.get("days"):
        return f"| {label} |" + " — |" * 14
    return (f"| {label} | {m['gross']:,.0f} | {m['fees']:,.0f} | {m['slippage']:,.0f} | {m['funding']:,.0f} | "
            f"**{m['net']:,.0f}** | {m['ann_return']:.1%} | {m['sharpe']:.2f} | {m['t']:.2f} | "
            f"{m['max_drawdown']:,.0f} ({m['max_drawdown_pct']:.1%}) | {m['round_trips']} | {m['lots_traded']} | "
            f"{m['turnover_ann']:.1f}x | {m['margin_max']:,.0f} | {m['return_on_margin_ann']:.1%} |")


def _leg_cost_rows(trades, mapping):
    rows = []
    for leg in mapping.columns:
        t = trades[trades.ts_code.isin(set(mapping[leg].dropna()))]
        if t.empty:
            rows.append(f"| {leg} | 0 | 0 | 0 | 0 | 0 |")
            continue
        by = lambda r: t[t.reason == r]
        closes = t[t.reason.isin(["close", "adjust"])]
        rows.append(f"| {leg} | {by('open').fee.sum():,.0f} | {closes.fee.sum():,.0f} | {by('roll').fee.sum():,.0f} | "
                    f"{t.slippage.sum():,.0f} | {by('roll').slippage.sum():,.0f} |")
    return rows


def futures_report_parts(a, b, s, ctx, flags):
    cfg, exe, snap = ctx["cfg"], ctx["exe"], ctx["snap"]
    ex, M, bt = cfg["execution"], exe["metrics"], s["bt"]
    oos = M.get("oos", {})
    stat_ok = _stat_tradable(s)
    exe_ok = bool(oos.get("days")) and oos["net"] > 0 and abs(oos["t"]) >= 1.96 and \
        not any(fl[0] == "🔴 高" for fl in flags)
    rolls, research = ctx["rolls"], ctx["research"]
    n = cfg["screening"]["n_candidates"]
    ev = exe["events"]
    hedges = [e["realized_hedge"] for e in exe["entries"]]
    header = [
        f"\n> run_id `{ctx['run_id']}`｜快照 sha256 `{ctx['snap_sha'][:16]}…`｜数据：{snap['provenance']}"
        f"{' v' + str(snap['server_version']) if snap.get('server_version') else ''}",
        f"> **统计证据：{'支持继续研究' if stat_ok else '不足'}｜交易可行性："
        f"{'整手真实合约扣费后样本外为正且显著（实盘前仍需复核）' if exe_ok else '不足/未证实'}**。"
        "统计上平稳的价差不等于可交易策略，两者分开判断（第 1、4、5 章为统计证据，第 6 章为交易可行性）。",
    ]
    sec2 = [
        "\n## 2. 数据、合约映射与换月",
        f"- 数据来源：{snap['server']} {snap.get('server_version') or ''} MCP；快照 `{cfg['snapshot']}`"
        f"（sha256 `{ctx['snap_sha']}`），{len(snap['calls'])} 次调用；回放只读快照，不访问实时数据。",
        f"- 可用工具：{snap.get('tools_available') or '（v1 快照未记录）'}；缺失能力："
        + ("无" if not ctx["missing"] else "；".join(f"`{m['tool']}`（{m['purpose']}）→ {m['fallback']}"
                                                   for m in ctx["missing"])),
        f"- 研究区间：{research.index[0]:%Y-%m-%d} ~ {research.index[-1]:%Y-%m-%d}，{len(research)} 个可用交易日；"
        f"训练/测试切分日 {ctx['split_date']:%Y-%m-%d}（训练 {cfg['train_frac']:.0%}，其后为未触碰的样本外）。",
        f"- 腿 {a}：{_leg_desc(cfg['legs'][0], cfg['exclude_months'])}；腿 {b}："
        f"{_leg_desc(cfg['legs'][1], cfg['exclude_months'])}。映射只用决策日（前一交易日）收盘后已知信息，逐日记录于 mapping.csv。",
        "- **研究用连续序列**：对数价 = 累加“当日所持合约自身的日收益”，换月日不产生拼接跳空，只用于统计诊断；"
        "**回测盈亏全部按真实合约价格计算**（mapping.csv 的 *_exec_close 列为真实合约价格）。",
        f"- 换月事件 {len(rolls)} 次（rolls.csv）：",
        "| 生效日 | 决策日 | 腿 | 从 | 到 | 原因 | 决策日持仓量 从→到 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r.date:%Y-%m-%d} | {r.decided_on:%Y-%m-%d} | {r.leg} | {r.from_code} | {r.to_code} | {r.reason} | "
          f"{r.oi_from:,.0f} → {r.oi_to:,.0f} |" for r in rolls.head(20).itertuples()],
        "（仅列前 20 次，完整见 rolls.csv）" if len(rolls) > 20 else "",
    ]
    sec3 = [
        "\n## 3. 候选与经济逻辑",
        f"- 价差类型：{'跨期（同品种不同交割月）' if cfg['family'] == 'calendar' else '跨品种'}。",
        f"- 经济/产业链逻辑：{cfg['rationale'] or '跨期价差：同一品种不同交割月之间的期限结构与持有成本关系'}",
        f"- 筛选：候选数 {n}，方法：{cfg['screening']['procedure']}。"
        + (f" 多重检验：Bonferroni 阈值 p≤{0.05 / n:.4g}，训练窗 EG 协整 p={s['eg'][1]:.4f}。" if n > 1 else ""),
        "- 相关性不作为入选理由；协整与均值回归证据见第 4、5 章。",
    ]
    sec6 = [
        "\n## 6. 交易可行性（可执行回测：真实合约·整手·期货成本）",
        f"- 信号：第 t 日收盘按研究价差 z-score 决定目标仓位（开 |z|>{ENTRY}、平 |z|<{EXIT}、止损 |z|>{STOP}，"
        f"窗口与 β 仅用训练窗）→ 第 t+1 日按 **{ex['exec_price']}** 价成交，按 **{ex['mark_price']}** 价逐日盯市。",
        f"- 手数规则：{a} 腿 {ex['base_lots']} 手；{b} 腿 = round(|β|×{ex['base_lots']}×P_{a}×乘数_{a} ÷ "
        f"(P_{b}×乘数_{b}))，至少 1 手（P 为决策日收盘价）；β={bt['beta']:.4f}"
        f"{'>0 时两腿反向' if bt['beta'] > 0 else '≤0 时两腿同向'}。持仓期间手数不变；换月按原手数平旧开新。",
        (f"- 开仓 {len(hedges)} 次；取整后实际名义对冲比 均值 {np.mean(hedges):.3f}（|β|={abs(bt['beta']):.3f}），"
         f"范围 {min(hedges):.3f}~{max(hedges):.3f}。") if hedges else "- 回测期内未开仓。",
        "- 手续费率与保证金率由用户在配置中按品种指定（整段常数，非逐日历史）："
        + "；".join(f"{p} 费率 {v.get('fee_rate', 0):g}、每手 {v.get('fee_per_lot', 0):g} 元、保证金 {v['margin_rate']:.0%}"
                   for p, v in cfg["assumptions"].items() if "margin_rate" in v) + "。",
        f"- 成本：手续费 = 手数×(成交价×乘数×费率 + 每手费)×{ex['broker_fee_multiplier']:g}（经纪商倍数），开、平各计；"
        f"滑点 {ex['slippage_ticks']:g} 跳/手/次；保证金 = Σ|手数|×盯市价×乘数×(保证金率 + "
        f"{ex['broker_margin_add']:g})；保证金资金成本 {ex['funding_rate_annual']:.2%}/年；本金 {ex['capital']:,.0f} 元。",
        "| 区间 | 毛PnL | 手续费 | 滑点 | 资金成本 | 净PnL | 年化(本金) | Sharpe | t | 最大回撤(占本金) | "
        "完成往返 | 成交手数 | 年化换手 | 最大保证金 | 年化收益/平均保证金 |",
        "|" + "---|" * 15,
        _money_row("样本内", M.get("is")), _money_row("**样本外**", oos), _money_row("全样本", M["all"]),
        _money_row("样本外·压力(费×2, 滑点+1跳)", ctx["stress"].get("oos")),
        f"- 换月执行 {M['all'].get('rolls', 0)} 次；成交顺延 {int((ev.kind == 'deferred').sum())} 日；流动性标记 "
        f"{int((ev.kind == 'liquidity').sum())} 笔（events.csv）。一腿当日无行情则两腿都不成交，次日重试"
        "（不留单腿敞口）。**未考虑涨跌停**：一字板日按假设价成交，可能高估可成交性。",
        "- 成本按腿与开/平拆分（全样本，元）：",
        "| 腿 | 开仓手续费 | 平仓手续费 | 换月手续费 | 滑点 | 其中换月滑点 |",
        "|---|---|---|---|---|---|",
        *_leg_cost_rows(exe["trades"], ctx["mapping"]),
        f"  开、平按同一费率计（日线不区分平今，`offset_today_fee` 未使用）；换月额外滑点 "
        f"{ex['roll_slippage_ticks']:g} 跳/手（config.execution.roll_slippage_ticks）。",
        "- 成交记录（前 20 笔，完整见 trades.csv）：",
        "| 信号日 | 成交日 | 合约 | 手数 | 成交价 | 手续费 | 滑点 | 原因 |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {t.signal_date:%Y-%m-%d} | {t.date:%Y-%m-%d} | {t.ts_code} | {t.lots:+d} | {t.price:g} | "
          f"{t.fee:,.1f} | {t.slippage:,.1f} | {t.reason} |" for t in exe["trades"].head(20).itertuples()],
        "- 合约参数来源（按合约日计）：" + "；".join(
            f"{k}: " + ", ".join(f"{src}×{c}" for src, c in sorted(v.items())) for k, v in sorted(ctx["sources"].items())),
        "- **执行局限**：日线只说明当日价格区间，不能证明按假设价成交；未建模盘口深度、平今手续费、"
        "交割与限仓规则、夜盘跳空对开盘成交价的影响。逐笔成交见 trades.csv，逐日盈亏与保证金见 daily.csv。",
    ]
    sec10 = [
        "\n## 10. 未做的分析、数据限制与假设",
        "- 未做（需 Agent 另行补充，不得写成已完成）：Johansen 协整、Kalman 动态对冲、完整 Chow/CUSUM、"
        "滚动前推（walk-forward）、区制分析、因子归因（zeus 暂无市场代理序列）、盘中/分钟级执行、涨跌停封板约束。",
        *[f"- 数据缺口：zeus 缺 `{m['tool']}`（{m['purpose']}）→ {m['fallback']}。" for m in ctx["missing"]],
        f"- 用户指定的合约参数：{json.dumps(cfg['assumptions'], ensure_ascii=False)}。",
        f"- 执行参数：{json.dumps(ex, ensure_ascii=False)}。",
        "- 第 7 章为研究口径（价差对数收益 + 近似成本），只作统计证据；金额口径以第 6 章为准。",
    ]
    return {"header": header, "insert": {"## 4.": sec2 + sec3, "## 7.": sec6, "---": sec10}}


# ============ 研究配置 ============
class ReplayInputError(ValueError):
    """配置或快照缺字段/格式错误；消息指明位置与缺什么。"""


EXEC_DEFAULTS = {"base_lots": 10, "exec_price": "open", "mark_price": "settle", "slippage_ticks": 1,
                 "max_participation": 0.05, "capital": 1_000_000, "funding_rate_annual": 0.0,
                 "broker_fee_multiplier": 1.0, "broker_margin_add": 0.0, "roll_slippage_ticks": 0}
ASSUME_KEYS = ("multiplier", "price_tick", "fee_rate", "fee_per_lot", "margin_rate")
ENTRY, EXIT, STOP = 2.0, 0.5, 3.5
# zeus 工具 → (用途, 缺失时的替代)；接口定义见 references/zeus-mcp-interface.md
OPTIONAL_TOOLS = {
    "fut_basic": ("合约列表、合约乘数、最小变动价位", "按 品种+YYMM 枚举合约代码探测；乘数/跳价用 config assumptions"),
}
DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "settle", "vol", "oi")
# 各工具必需字段（与 references/zeus-mcp-interface.md 一致）；快照校验与 --check-zeus 共用
TOOL_FIELDS = {
    "fut_daily": DAILY_FIELDS,
    "fut_basic": ("ts_code", "fut_code", "exchange", "multiplier", "price_tick", "list_date", "delist_date"),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(obj, keys, where):
    if not isinstance(obj, dict):
        raise ReplayInputError(f"{where}: expected an object, got {type(obj).__name__}")
    for k in keys:
        if k not in obj:
            raise ReplayInputError(f"{where}: missing '{k}'")


def _read_json(path, where):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ReplayInputError(f"{where}: cannot read {path}: {e}") from e


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def _check_yyyymmdd(v, where):
    if not (isinstance(v, str) and len(v) == 8 and v.isdigit()):
        raise ReplayInputError(f"{where} must be YYYYMMDD, got {v!r}")


def _num(v, where, lo=0.0, hi=None, integer=False, strict=False):
    ok = _is_num(v) and (v > lo if strict else v >= lo) and (hi is None or v <= hi)
    if integer:
        ok = ok and float(v).is_integer()
    if not ok:
        raise ReplayInputError(f"{where} must be a {'integer' if integer else 'number'} "
                               f"{'>' if strict else '≥'}{lo:g}{'' if hi is None else f' and ≤{hi:g}'}, got {v!r}")
    return int(v) if integer else float(v)


def load_config(config_path):
    """读取并规范化研究配置；所有必填项/取值在这里校验，错误指明位置。"""
    config_path = Path(config_path)
    cfg = _read_json(config_path, "config")
    _require(cfg, ("snapshot", "legs"), "config")
    raw = cfg["legs"]
    if not (isinstance(raw, list) and len(raw) == 2):
        raise ReplayInputError(f"config: 'legs' must be a list of two legs, got {raw!r}")
    legs = []
    for i, leg in enumerate(raw):
        where = f"config.legs[{i}]"
        if isinstance(leg, str):                 # 简写：固定合约代码
            leg = {"ts_code": leg}
        if not isinstance(leg, dict):
            raise ReplayInputError(f"{where}: expected an object or ts_code string")
        if "ts_code" in leg:
            try:
                prod, _, exch = fx.parse_code(leg["ts_code"])
            except (ValueError, TypeError) as e:
                raise ReplayInputError(f"{where}: {e}") from e
            legs.append({"name": leg.get("name", leg["ts_code"]), "ts_code": leg["ts_code"],
                         "product": prod, "exchange": exch, "select": "fixed"})
            continue
        _require(leg, ("product", "exchange", "select"), where)
        if leg["select"] not in ("dominant", "second"):
            raise ReplayInputError(f"{where}.select must be 'dominant' or 'second', got {leg['select']!r}")
        legs.append({"name": leg.get("name", f"{leg['product']}_{leg['select']}"),
                     "product": str(leg["product"]).upper(), "exchange": str(leg["exchange"]).upper(),
                     "select": leg["select"]})
    if legs[0]["select"] == "second":
        raise ReplayInputError("config.legs[0]: 'second' 只能用于第二条腿（相对第一条腿的远月）")
    if legs[1]["select"] == "second" and (legs[0]["select"] != "dominant" or legs[0]["product"] != legs[1]["product"]):
        raise ReplayInputError("config.legs[1]: 'second' 需要第一条腿为同品种 'dominant'")
    same = ({k: legs[0].get(k) for k in ("product", "exchange", "select", "ts_code")}
            == {k: legs[1].get(k) for k in ("product", "exchange", "select", "ts_code")})
    if same:
        raise ReplayInputError("config.legs: 两条腿会映射到同一合约（价差恒为 0）；跨期请用 dominant + second")
    if legs[0]["name"] == legs[1]["name"]:
        raise ReplayInputError(f"config.legs: 两条腿名称重复 {legs[0]['name']!r}")
    family = "calendar" if legs[0]["product"] == legs[1]["product"] else "cross"
    if cfg.get("family", family) != family:
        raise ReplayInputError(f"config: family={cfg['family']!r} 与两腿品种不符（应为 {family!r}）")
    rationale = cfg.get("rationale", "")
    if family == "cross" and not (isinstance(rationale, str) and rationale.strip()):
        raise ReplayInputError("config: cross-commodity spread requires 'rationale'（产业链/经济逻辑）；"
                               "相关性本身不能作为理由")
    scr = cfg.get("screening", {"n_candidates": 1, "procedure": "预设单一配对（未做批量筛选）"})
    _require(scr, ("n_candidates", "procedure"), "config.screening")
    screening = {"n_candidates": _num(scr["n_candidates"], "config.screening.n_candidates", 1, integer=True),
                 "procedure": str(scr["procedure"])}
    roll = cfg.get("roll", {})
    excl = _num(roll.get("exclude_months_before_delivery", 1),
                "config.roll.exclude_months_before_delivery", 0, integer=True)
    ex_raw = cfg.get("execution", {})
    unknown = set(ex_raw) - set(EXEC_DEFAULTS)
    if unknown:
        raise ReplayInputError(f"config.execution: unknown keys {sorted(unknown)}")
    ex = {**EXEC_DEFAULTS, **ex_raw}
    if ex["exec_price"] not in ("open", "close", "settle"):
        raise ReplayInputError("config.execution.exec_price must be open/close/settle")
    if ex["mark_price"] not in ("close", "settle"):
        raise ReplayInputError("config.execution.mark_price must be close/settle")
    ex["base_lots"] = _num(ex["base_lots"], "config.execution.base_lots", 1, integer=True)
    ex["capital"] = _num(ex["capital"], "config.execution.capital", 0, strict=True)
    ex["max_participation"] = _num(ex["max_participation"], "config.execution.max_participation", 0, 1, strict=True)
    for k in ("slippage_ticks", "funding_rate_annual", "broker_fee_multiplier", "broker_margin_add",
              "roll_slippage_ticks"):
        ex[k] = _num(ex[k], f"config.execution.{k}")
    assumptions = {}
    for prod, a in (cfg.get("assumptions") or {}).items():
        if not isinstance(a, dict):
            raise ReplayInputError(f"config.assumptions.{prod}: expected an object")
        unknown = set(a) - set(ASSUME_KEYS)
        if unknown:
            raise ReplayInputError(f"config.assumptions.{prod}: unknown keys {sorted(unknown)}")
        assumptions[prod.upper()] = {k: _num(v, f"config.assumptions.{prod}.{k}",
                                             strict=k in ("multiplier", "price_tick")) for k, v in a.items()}
    for prod in dict.fromkeys(l["product"] for l in legs):     # 手续费与保证金没有数据源，必须由用户给出
        a = assumptions.get(prod, {})
        if "margin_rate" not in a:
            raise ReplayInputError(f"config.assumptions.{prod}: missing 'margin_rate'（保证金率需由用户指定，如 0.10）")
        if "fee_rate" not in a and "fee_per_lot" not in a:
            raise ReplayInputError(f"config.assumptions.{prod}: 需指定 'fee_rate'（按成交额，如 0.0001）"
                                   f"或 'fee_per_lot'（元/手）")
    window = _num(cfg.get("window", 0), "config.window", 0, integer=True)
    train_frac = _num(cfg.get("train_frac", 0.7), "config.train_frac", 0.3, 0.9)
    for k in ("start_date", "end_date"):
        if k in cfg:
            _check_yyyymmdd(cfg[k], f"config: '{k}'")
    if cfg.get("start_date") and cfg.get("end_date") and cfg["start_date"] > cfg["end_date"]:
        raise ReplayInputError(f"config: 'start_date' {cfg['start_date']} 晚于 'end_date' {cfg['end_date']}")
    return {"path": config_path, "raw": cfg, "snapshot": cfg["snapshot"], "legs": legs, "family": family,
            "rationale": rationale, "screening": screening, "exclude_months": excl, "execution": ex,
            "assumptions": assumptions, "window": window, "train_frac": train_frac,
            "start_date": cfg.get("start_date"), "end_date": cfg.get("end_date")}


# ============ 取数：zeus MCP（Streamable HTTP, JSON-RPC）→ 快照 ============
class ZeusError(RuntimeError):
    """zeus MCP 连接/鉴权/工具调用失败，或返回空数据。"""


class ZeusClient:
    """最小 MCP 客户端（仅标准库）：initialize → notifications/initialized → tools/list / tools/call。
    ponytail: 只实现 zeus 需要的请求-应答；无流式通知/重连，需要时换官方 mcp SDK。"""

    def __init__(self, url, token=None, timeout=120, headers=None):
        self.url, self.token, self.timeout = url, token, timeout
        self.extra_headers = dict(headers or {})
        self.session = None
        self._id = 0
        init = self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                        "clientInfo": {"name": "run_statarb", "version": "2"}})
        self.server_info = init.get("serverInfo", {})
        self._rpc("notifications/initialized", notify=True)

    def _rpc(self, method, params=None, notify=False):
        import urllib.request, urllib.error
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self._id += 1
            msg["id"] = self._id
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        headers.update(self.extra_headers)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.session = r.headers.get("Mcp-Session-Id") or self.session
                body = r.read().decode("utf-8")
                ctype = r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            raise ZeusError(f"zeus {method}: HTTP {e.code} {e.reason}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ZeusError(f"zeus {method}: cannot reach {self.url}: {e}") from e
        if notify:
            return None
        if "text/event-stream" in ctype:   # SSE：取与本请求 id 对应的 data 行
            msgs = [json.loads(l[5:]) for l in body.splitlines() if l.startswith("data:")]
            resp = next((m for m in msgs if m.get("id") == msg["id"]), None)
        else:
            resp = json.loads(body) if body else None
        if resp is None:
            raise ZeusError(f"zeus {method}: no response for request id {msg['id']}")
        if "error" in resp:
            raise ZeusError(f"zeus {method}: {resp['error']}")
        return resp["result"]

    def list_tools(self):
        names, cursor = [], None
        while True:
            res = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            names += [t["name"] for t in res.get("tools", [])]
            cursor = res.get("nextCursor")
            if not cursor:
                return names

    def call(self, tool, args):
        """调用工具，返回行列表（FastMCP 把 list 返回拆成每行一个 text 项）。"""
        res = self._rpc("tools/call", {"name": tool, "arguments": args})
        texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
        if res.get("isError"):
            raise ZeusError(f"zeus {tool}{args}: {' '.join(texts)}")
        rows = []
        for t in texts:
            try:
                v = json.loads(t)
            except json.JSONDecodeError:
                raise ZeusError(f"zeus {tool}{args}: 返回的不是 JSON 行数据：{t[:200]!r}") from None
            rows.extend(v if isinstance(v, list) else [v])
        return rows


SKILL_DIR = Path(__file__).resolve().parents[1]


def _read_dotenv(path):
    """最小 .env 解析：KEY=VALUE，支持 # 注释、export 前缀、单/双引号。"""
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        v = v.strip()
        if v[:1] in "\"'" and v[:1] and v.count(v[0]) >= 2:
            v = v[1:v.index(v[0], 1)]
        else:
            v = v.split(" #")[0].strip()
        out[k.strip()] = v
    return out


def _mcp_zeus(conf, cwd):
    """从 Claude Code 风格的配置中取名为 zeus 的 MCP server（顶层或当前项目下）。"""
    servers = [conf.get("mcpServers", {}).get("zeus")]
    for proj, pc in (conf.get("projects") or {}).items():
        if Path(proj).resolve() == Path(cwd).resolve():
            servers.insert(0, (pc.get("mcpServers") or {}).get("zeus"))
    return next((z for z in servers if z and z.get("url")), None)


def resolve_zeus(cwd=None, skill_dir=None, home=None, environ=None):
    """zeus 连接信息，按顺序取第一个可用来源 → (url, headers, 来源说明)；来源说明不含 token。
      1. 进程环境变量 ZEUS_MCP_URL / ZEUS_MCP_TOKEN
      2. 工作目录 .env   3. skill 目录 .env
      4. Claude Code MCP 配置里名为 zeus 的 server：工作目录 .mcp.json → ~/.claude.json（项目级优先）"""
    cwd, skill_dir = Path(cwd or Path.cwd()), Path(skill_dir or SKILL_DIR)
    home, environ = Path(home or Path.home()), os.environ if environ is None else environ

    def pack(url, token):
        return url, ({"Authorization": f"Bearer {token}"} if token else {})

    if environ.get("ZEUS_MCP_URL"):
        return (*pack(environ["ZEUS_MCP_URL"], environ.get("ZEUS_MCP_TOKEN")), "环境变量")
    for f in (cwd / ".env", skill_dir / ".env"):
        if f.is_file():
            env = _read_dotenv(f)
            if env.get("ZEUS_MCP_URL"):
                return (*pack(env["ZEUS_MCP_URL"], env.get("ZEUS_MCP_TOKEN")), str(f))
    for f in (cwd / ".mcp.json", home / ".claude.json"):
        if f.is_file():
            try:
                z = _mcp_zeus(json.loads(f.read_text(encoding="utf-8")), cwd)
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
            if z:
                headers = {k: os.path.expandvars(str(v)) for k, v in (z.get("headers") or {}).items()}
                return z["url"], headers, f"{f}（mcpServers.zeus）"
    raise ZeusError(f"找不到 zeus 连接信息：请把 {SKILL_DIR / '.env.example'} 复制为工作目录下的 .env "
                    "并填写 ZEUS_MCP_URL / ZEUS_MCP_TOKEN，或在 Claude Code 中添加名为 zeus 的 MCP server")


def _probe_codes(product, exch, start, end):
    """无 fut_basic 时：枚举 品种+YYMM（区间起始月 → 结束月后 13 个月）作为候选合约代码。"""
    y, m = int(start[:4]), int(start[4:6])
    last = int(end[:4]) * 12 + int(end[4:6]) + 13
    out = []
    while y * 12 + m <= last:
        out.append(f"{product}{y % 100:02d}{m:02d}.{exch}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fetch_snapshot(config_path, url, token=None, headers=None):
    """按研究配置调 zeus：fut_basic（或枚举）→ 每张合约 fut_daily，
    响应原样写入快照（不可变：已存在则拒绝；任何失败都不写文件）。缺失的可选工具记为能力缺口。"""
    cfg = load_config(config_path)
    start, end = cfg["start_date"], cfg["end_date"]
    if not (start and end):
        raise ReplayInputError("config: --fetch 需要 'start_date' 与 'end_date'（YYYYMMDD）")
    snap_path = cfg["path"].parent / cfg["snapshot"]
    if snap_path.exists():
        raise ReplayInputError(f"snapshot {snap_path} already exists; 快照不可变，请换文件名")

    client = ZeusClient(url, token, headers=headers)
    tools = client.list_tools()
    if "fut_daily" not in tools:
        raise ZeusError(f"zeus 缺少必需工具 fut_daily（现有：{tools}）")
    calls = []

    def call(tool, args):
        retrieved_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            rows = client.call(tool, args)
        except ZeusError as e:
            raise ZeusError(f"{args.get('ts_code') or args.get('fut_code')}: {e}") from e
        calls.append({"tool": tool, "params": args, "retrieved_at": retrieved_at, "rows": rows})
        return rows

    codes, product_codes = [], {}
    for leg in cfg["legs"]:
        if leg["select"] == "fixed":
            codes.append(leg["ts_code"])
    for prod, exch in dict.fromkeys((l["product"], l["exchange"]) for l in cfg["legs"]):
        roll_leg = any(l["select"] != "fixed" and l["product"] == prod for l in cfg["legs"])
        if "fut_basic" in tools:                       # 固定合约腿也要取合约乘数/最小变动价位
            rows = call("fut_basic", {"exchange": exch, "fut_code": prod})
            cs = [r["ts_code"] for r in rows
                  if str(r.get("list_date") or "0") <= end and str(r.get("delist_date") or "99999999") >= start]
        else:
            cs = _probe_codes(prod, exch, start, end)
        if roll_leg:
            product_codes[prod] = cs
            codes += cs
    n_rows = {}
    for c in dict.fromkeys(codes):
        n_rows[c] = len(call("fut_daily", {"ts_code": c, "start_date": start, "end_date": end}))
    for leg in cfg["legs"]:
        if leg["select"] == "fixed" and not n_rows[leg["ts_code"]]:
            raise ZeusError(f"{leg['ts_code']}: fut_daily returned no rows for {start}~{end}"
                            "（合约代码/区间？MCP 缺数据？）")
    for prod, cs in product_codes.items():
        if not any(n_rows[c] for c in cs):
            raise ZeusError(f"{prod}: fut_daily returned no rows for any contract in {start}~{end}")
    missing = []
    for tool, (purpose, fallback) in OPTIONAL_TOOLS.items():
        if tool not in tools:
            missing.append({"tool": tool, "purpose": purpose, "fallback": fallback})
            continue

    info = client.server_info
    snap = {"snapshot_version": 2,
            "server": info.get("name", "zeus"), "server_version": info.get("version"),
            "provenance": f"{info.get('name', 'zeus')} MCP（原样响应）",
            "tools_available": sorted(tools), "missing_capabilities": missing, "calls": calls}
    snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    return snap


def check_zeus(url, token, sample_code, start, end, headers=None):
    """真实 MCP 兼容性检查（不进普通测试）：必需 fut_daily 字段齐全；可选工具存在且字段符合接口定义。"""
    client = ZeusClient(url, token, headers=headers)
    tools = client.list_tools()
    prod, _, exch = fx.parse_code(sample_code)

    def probe(tool):
        if tool not in tools:
            return "absent" + ("" if tool == "fut_daily" else f"（替代：{OPTIONAL_TOOLS[tool][1]}）")
        args = ({"exchange": exch, "fut_code": prod} if tool == "fut_basic"
                else {"ts_code": sample_code, "start_date": start, "end_date": end})
        try:
            rows = client.call(tool, args)
        except ZeusError as e:
            return f"error: {e}"
        if not rows:
            return f"no rows for {args}"
        lack = [f for f in TOOL_FIELDS[tool] if f not in rows[0]]
        return "ok" if not lack else f"missing fields {lack}"

    daily = probe("fut_daily")
    return {"server": client.server_info, "tools": tools, "ok": daily == "ok",
            "required": {"fut_daily": daily}, "optional": {t: probe(t) for t in OPTIONAL_TOOLS}}


# ============ 回放：研究配置 + 固定 MCP 快照 → 报告 + 可审计产物（不取实时数据） ============
def load_snapshot(snap):
    """校验快照并按工具归集行；逐行校验必需字段，错误指明 calls[i].rows[j]。"""
    _require(snap, ("server", "provenance", "calls"), "snapshot")
    out = {"fut_daily": [], "fut_basic": []}
    # 快照只强制主键与会被使用的数值字段；fut_basic 其余字段缺失时按行回落到配置
    required = {"fut_daily": DAILY_FIELDS, "fut_basic": ("ts_code",)}
    # 无成交日（vol=0）交易所只给 close/settle，open/high/low 为 null：允许为空，但不能是非数值
    numeric = {"fut_daily": ("close", "settle", "vol", "oi")}
    nullable = {"fut_daily": ("open", "high", "low")}
    seen = set()
    for i, call in enumerate(snap["calls"]):
        _require(call, ("tool", "params", "retrieved_at", "rows"), f"calls[{i}]")
        tool = call["tool"]
        if tool not in out:
            continue
        for j, row in enumerate(call["rows"]):
            where = f"calls[{i}].rows[{j}]"
            _require(row, required[tool], where)
            if "trade_date" in row:
                d = str(row["trade_date"])
                if len(d) != 8 or not d.isdigit():
                    raise ReplayInputError(f"{where}: trade_date '{d}' is not YYYYMMDD")
            for f in numeric.get(tool, ()):
                if not _is_num(row[f]):
                    raise ReplayInputError(f"{where}: '{f}' is not numeric: {row[f]!r}")
            for f in nullable.get(tool, ()):
                if row[f] is not None and not _is_num(row[f]):
                    raise ReplayInputError(f"{where}: '{f}' is not numeric or null: {row[f]!r}")
            if tool == "fut_daily":
                key = (row["ts_code"], row["trade_date"])
                if key in seen:     # 分页重叠等导致的重复行：不静默覆盖
                    raise ReplayInputError(f"{where}: duplicate {row['ts_code']} {row['trade_date']}")
                seen.add(key)
            out[tool].append(row)
    if not out["fut_daily"]:
        raise ReplayInputError("snapshot: no fut_daily rows")
    return out


def _git_version():
    try:
        cwd = Path(__file__).parent
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                               text=True, check=True).stdout.strip() != ""
        return {"git_commit": head, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def _lib_versions():
    import platform, statsmodels
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__}


def _to_csv(df, path, date_cols=()):
    df = df.copy()
    for k in date_cols:
        df[k] = pd.to_datetime(df[k]).dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def replay(config_path, out_dir):
    """研究配置 + 快照 → 合约映射/换月 → 研究序列统计证据 → 整手真实合约回测 → 报告与产物。
    run_id 只由 配置/快照/脚本 的内容哈希决定：输入不变 → run_id 与全部产物不变。"""
    cfg = load_config(config_path)
    out_dir = Path(out_dir)
    snap_path = cfg["path"].parent / cfg["snapshot"]
    snap = _read_json(snap_path, "snapshot")
    data = load_snapshot(snap)
    legs = cfg["legs"]
    a, b = legs[0]["name"], legs[1]["name"]

    tabs = fx.bar_tables(data["fut_daily"])
    for leg in legs:
        if leg["select"] == "fixed" and leg["ts_code"] not in tabs["close"].columns:
            raise ReplayInputError(f"snapshot has no rows for {leg['ts_code']}; available: "
                                   f"{sorted(tabs['close'].columns)[:20]}")
    mapping, rolls = fx.build_mapping(tabs, legs, cfg["exclude_months"])
    research = fx.research_prices(tabs, mapping).dropna()
    if len(research) < 60:
        raise ReplayInputError(f"legs {[a, b]} share only {len(research)} usable dates in snapshot; need ≥60")
    mapping = mapping.loc[research.index[0]:research.index[-1]]
    ex = cfg["execution"]
    make_specs = lambda fee_mult: fx.ContractSpecs(data["fut_basic"], cfg["assumptions"], fee_mult,
                                                   ex["broker_margin_add"])
    specs = make_specs(ex["broker_fee_multiplier"])
    split_i = int(len(research) * cfg["train_frac"])
    split_date, train_end = research.index[split_i], research.index[split_i - 1]
    research_legs = {}
    for n in (a, b):            # 研究口径近似成本：取训练窗最后一日的合约参数（不看样本外）
        sp = specs.spec(mapping.at[train_end, n], train_end)
        research_legs[n] = {"multiplier": sp["multiplier"], "tick_size": sp["price_tick"],
                            "fee_rate_bps": sp["fee_rate"] * 1e4, "fee_per_lot": sp["fee_per_lot"],
                            "slippage_ticks": ex["slippage_ticks"],
                            "margin_rate": max(sp["margin_long"], sp["margin_short"])}
    px = np.exp(research)
    s = run_stats(px, research_legs, None, cfg["window"] or None, cfg["train_frac"])
    bt = s["bt"]
    target = bt["pos"].reindex(mapping.index).ffill().fillna(0)
    exe = fx.executable_backtest(tabs, mapping, target, bt["beta"], specs, ex, split_date)
    stress_ex = {**ex, "slippage_ticks": ex["slippage_ticks"] + 1}     # 压力：手续费×2、滑点+1 跳
    stress = fx.executable_backtest(tabs, mapping, target, bt["beta"], make_specs(2 * ex["broker_fee_multiplier"]),
                                    stress_ex, split_date)

    missing = snap.get("missing_capabilities")
    if missing is None:        # v1 快照：未记录工具清单 → 从调用推断
        called = {c["tool"] for c in snap["calls"]}
        missing = [{"tool": t, "purpose": p, "fallback": f} for t, (p, f) in OPTIONAL_TOOLS.items() if t not in called]
    sources = {}
    for (field, src), n in specs.used.items():
        sources.setdefault(field, {})[src] = n
    ctx = {"cfg": cfg, "snap": snap, "missing": missing, "sources": sources, "rolls": rolls, "mapping": mapping,
           "exe": exe, "stress": stress["metrics"], "split_date": split_date, "research": research}

    snap_sha, cfg_sha = _sha256(snap_path), _sha256(cfg["path"])
    code_sha = hashlib.sha256((_sha256(__file__) + _sha256(fx.__file__)).encode()).hexdigest()
    run_id = hashlib.sha256(f"{cfg_sha}{snap_sha}{code_sha}".encode()).hexdigest()[:16]
    ctx["run_id"], ctx["snap_sha"] = run_id, snap_sha

    out_dir.mkdir(parents=True, exist_ok=True)
    exec_close = fx.executable_prices(tabs, mapping, "close")
    mtab = mapping.copy()
    for n in (a, b):
        mtab[f"{n}_research_logp"] = research[n].reindex(mapping.index)
        mtab[f"{n}_exec_close"] = exec_close[n]
    mtab.insert(0, "date", mtab.index.strftime("%Y-%m-%d"))
    _to_csv(mtab, out_dir / "mapping.csv")
    _to_csv(rolls, out_dir / "rolls.csv", ("date", "decided_on"))
    _to_csv(exe["trades"], out_dir / "trades.csv", ("date", "signal_date"))
    _to_csv(exe["daily"].reset_index(), out_dir / "daily.csv", ("date",))
    _to_csv(exe["events"], out_dir / "events.csv", ("date",))

    flags = combine_flags(robustness_flags(s["adf_tr"][1], s["kpss_tr"][1], s["hl_tr"], bt, s["betastab"]),
                          futures_flags(s, ctx))
    report = out_dir / "report.md"
    write_report(a, b, "zeus MCP", px, s, flags, report, **futures_report_parts(a, b, s, ctx, flags))

    outputs = {f: _sha256(out_dir / f) for f in
               ("report.md", "mapping.csv", "rolls.csv", "trades.csv", "daily.csv", "events.csv")}
    fitted = {"beta": round(bt["beta"], 12), "window": bt["window"], "hl_train": round(float(s["hl_tr"]), 9),
              "entry": ENTRY, "exit": EXIT, "stop": STOP, "train_frac": cfg["train_frac"],
              "split_date": split_date.strftime("%Y-%m-%d")}
    manifest = {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "family": cfg["family"], "legs": legs, "rationale": cfg["rationale"], "screening": cfg["screening"],
        "config": cfg["raw"], "config_sha256": cfg_sha,
        "execution": ex, "assumptions": cfg["assumptions"], "specs_sources": sources,
        "snapshot": {
            "path": cfg["snapshot"], "sha256": snap_sha,
            "snapshot_version": snap.get("snapshot_version"),
            "server": snap["server"], "server_version": snap.get("server_version"),
            "provenance": snap["provenance"], "tools_available": snap.get("tools_available"),
            "missing_capabilities": missing,
            "calls": [{"tool": c["tool"], "params": c["params"],
                       "retrieved_at": c["retrieved_at"], "n_rows": len(c["rows"])} for c in snap["calls"]],
        },
        "code": {"sha256": code_sha, **_git_version(), "libs": _lib_versions()},
        "data_window": {"start": research.index[0].strftime("%Y-%m-%d"),
                        "end": research.index[-1].strftime("%Y-%m-%d"), "n_research_days": len(research),
                        "n_rolls": len(rolls)},
        "fitted": fitted,
        "metrics": {"statistical": {"eg_p": s["eg"][1], "adf_train_p": s["adf_tr"][1],
                                    "seasonality_p": s["season"][1] if s["season"] else None,
                                    "oos_net_sharpe": bt["oos_net"]["sharpe"], "oos_net_t": bt["oos_net"]["t"]},
                    "executable": exe["metrics"], "stress_oos": stress["metrics"].get("oos")},
        "flags": [list(f) for f in flags],
        "outputs": outputs,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return json.loads(json.dumps(manifest, default=str))


# ============ main ============
def main():
    sys.stdout.reconfigure(errors="replace")   # gbk 控制台打印 emoji 不崩
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="研究配置 JSON（两条腿、快照路径、区间、执行/成本假设）")
    ap.add_argument("--fetch", action="store_true",
                    help="先从 zeus MCP 取数写快照（需环境变量 ZEUS_MCP_URL，可选 ZEUS_MCP_TOKEN），再回放")
    ap.add_argument("--out-dir", default="statarb_run", help="回放输出目录")
    ap.add_argument("--check-zeus", metavar="TS_CODE",
                    help="检查真实 zeus MCP 的工具与字段（如 RB2501.SHF），配合 --start/--end")
    ap.add_argument("--start", default="20240102")
    ap.add_argument("--end", default="20240131")
    ap.add_argument("--source", default="synthetic", choices=["synthetic"],
                    help="无 --config 时仅支持合成数据自测；真实数据请用 --config")
    ap.add_argument("--cointegrated", type=int, default=1)
    ap.add_argument("--mode", default="",
                    choices=["", "coint", "nocoint", "strong", "inversion", "leaked", "drift"],
                    help="合成数据分支自测；留空则用 --cointegrated")
    ap.add_argument("--window", type=int, default=0, help="z 回看窗；0=按半衰期自适应")
    ap.add_argument("--fee-bps", type=float, default=SYNTH_LEG["fee_rate_bps"], help="自测：每腿手续费率(bp)")
    ap.add_argument("--slippage-ticks", type=float, default=SYNTH_LEG["slippage_ticks"], help="自测：每腿滑点跳数")
    ap.add_argument("--out", default="statarb_report.md")
    args = ap.parse_args()

    def zeus():
        url, headers, source = resolve_zeus()
        print(f"[zeus] {url}（连接信息来自 {source}）")
        return url, headers

    try:
        if args.check_zeus:
            url, headers = zeus()
            res = check_zeus(url, None, args.check_zeus, args.start, args.end, headers=headers)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            raise SystemExit(0 if res["ok"] else 1)
        if args.config:
            cfg = load_config(args.config) if args.fetch else None
            existing = cfg and cfg["path"].parent / cfg["snapshot"]
            if args.fetch and existing.exists():       # 快照不可变：不覆盖，直接用它回放
                print(f"[fetch] 快照已存在，直接用它回放：{existing}（如需重新取数，请在配置里换一个 snapshot 文件名）")
            elif args.fetch:
                url, headers = zeus()
                snap = fetch_snapshot(args.config, url, headers=headers)
                n = sum(len(c["rows"]) for c in snap["calls"] if c["tool"] == "fut_daily")
                print(f"[fetch] {snap['server']} {snap['server_version']}: {len(snap['calls'])} calls, "
                      f"{n} fut_daily rows; missing: {[m['tool'] for m in snap['missing_capabilities']] or 'none'}")
            m = replay(args.config, args.out_dir)
            print(f"[done] run_id={m['run_id']} -> {args.out_dir}")
            for lv, sig, rule, ev in m["flags"][:5]:
                print(f"  {lv} {sig}: {ev}")
            return
    except (ReplayInputError, fx.MissingData) as e:
        raise SystemExit(f"[input error] {e}")
    except ZeusError as e:
        raise SystemExit(f"[zeus error] {e}")

    px = _synthetic_pair(cointegrated=bool(args.cointegrated), mode=args.mode or None)
    leg = {**SYNTH_LEG, "fee_rate_bps": args.fee_bps, "slippage_ticks": args.slippage_ticks}
    analyze(px, "A", "B", "synthetic", load_market(px), args.out, {"A": leg, "B": leg},
            window=args.window or None)


if __name__ == "__main__":
    main()
