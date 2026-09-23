#!/usr/bin/env python3
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
  7. 成本模型拆成 佣金(双腿) + 卖出印花税(单边) + 融券/融资日度carry，扣费贴近现实；
     期货回放应在配置里显式给出成本（期货无印花税/融券，置 0）。
  8. 交易计数改为"完成往返"(开→平 成对)，并报告期末未平仓，避免把开仓次数当往返。
  9. 半衰期公式与 guide 对齐：half-life = -ln2 / ln(1+b)。

两条路径：
  - 研究路径：数据只来自 zeus MCP。Agent 调 fut_daily 把响应落成快照 JSON，
    脚本用 --config 离线回放，输出 report.md + manifest.json；统计优先用 statsmodels。
  - 自测路径（无数据/无 statsmodels）：--source synthetic，ADF/KPSS 用本文件内置的
    numpy 实现（近似，仅供机器自测）；真实研究请装 statsmodels 用其精确 p 值。

注意：本骨架实现的是 OLS 对冲比率 + ADF/KPSS + 分段 β 稳定性 + 单次样本外回测。
Johansen、Kalman 动态对冲、Chow/CUSUM、滚动前推(walk-forward)、价差收益的因子归因
属于 Agent 在 references/statarb-guide.md 指引下补充的进阶分析，不在本脚本内，
请勿在报告里把它们写成"已自动完成"。

用法示例：
  python run_statarb.py --source synthetic --cointegrated 1      # 离线自测：协整对
  python run_statarb.py --source synthetic --cointegrated 0      # 离线自测：非协整对
  python run_statarb.py --config tests/fixtures/replay_config.json --out-dir run1
      # 回放：研究配置 + 固定 MCP 快照 → report.md + manifest.json（不取实时数据）
"""
import argparse, datetime as dt, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- 是否有 statsmodels（生产精确路径） ----------
try:
    from statsmodels.tsa.stattools import adfuller as _sm_adf, kpss as _sm_kpss
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# ============ 1. 数据层 ============
# 真实行情只经 zeus MCP 获取：Agent 调 fut_daily 落成快照 JSON，再用 --config 回放
# （见 replay()）。脚本本身不联网、不直连数据库；合成数据仅供离线自测。
COST_DEFAULTS = {"commission_bps": 5.0, "stamp_duty_bps": 5.0, "borrow_annual_bps": 800.0}


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
    if HAVE_SM:
        stat, p, *_ = _sm_adf(x, autolag="AIC")
        return float(stat), float(p), _adf_verdict(p)
    # --- numpy 回退：Δx_t = a + b x_{t-1} + Σ γ Δx_{t-i} + ε，取 b 的 t 值 ---
    lags = 1
    dx = np.diff(x)
    y = dx[lags:]
    cols = [np.ones(len(y)), x[lags:-1]]
    for i in range(1, lags + 1):
        cols.append(dx[lags - i: -i])
    M = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ beta
    dof = max(len(y) - M.shape[1], 1)
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(M.T @ M)))
    tstat = beta[1] / se[1]
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}  # ADF(常数项, 大样本) 近似临界值
    p = 0.01 if tstat < crit["1%"] else 0.05 if tstat < crit["5%"] else \
        0.10 if tstat < crit["10%"] else 0.20
    return float(tstat), float(p), _adf_verdict(p) + "（numpy 近似，建议装 statsmodels 复核）"


def kpss_test(x):
    """KPSS（H0: 平稳）。返回 (统计量, p, 结论文本)。
    与 ADF 互为对偶：两者都指向平稳才是干净证据。"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if HAVE_SM:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, *_ = _sm_kpss(x, regression="c", nlags="auto")
        return float(stat), float(p), _kpss_verdict(p)
    # --- numpy 回退：水平平稳 KPSS，Bartlett 长期方差 ---
    T = len(x)
    e = x - x.mean()
    S = np.cumsum(e)
    l = int(np.floor(4 * (T / 100.0) ** 0.25))
    g0 = (e @ e) / T
    lrv = g0
    for j in range(1, l + 1):
        gj = (e[j:] @ e[:-j]) / T
        lrv += 2.0 * (1 - j / (l + 1)) * gj
    lrv = max(lrv, 1e-12)
    stat = (S @ S) / (T ** 2 * lrv)
    cv = [(0.347, 0.10), (0.463, 0.05), (0.574, 0.025), (0.739, 0.01)]
    p = 0.10
    for c, pv in cv:
        if stat > c:
            p = pv
    return float(stat), float(p), _kpss_verdict(p) + "（numpy 近似，建议装 statsmodels 复核）"


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
def backtest(px, train_frac=0.70, window=None,
             entry=2.0, exit=0.5, stop=3.5,
             commission_bps=COST_DEFAULTS["commission_bps"],
             stamp_duty_bps=COST_DEFAULTS["stamp_duty_bps"],
             borrow_annual_bps=COST_DEFAULTS["borrow_annual_bps"]):
    """
    成本拆解（均以"价差对数收益"近似的分数计）：
      - commission_bps：单腿单边佣金 (bp)；一次换手交易两条腿 → 每单位换手 2×commission。
      - stamp_duty_bps：卖出印花税 (bp)，A 股单边、仅卖出腿，约半数换手承担。
        （注：A 股印花税率会调整，请按当期规则核对；美股置 0。）
      - borrow_annual_bps：做空腿的年化融券/融资成本 (bp)，按持仓天数计 carry。
        A 股个股融券常常无券可融或成本高且不稳定，这是比统计问题更硬的现实约束。
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

    turn = pos.diff().abs().fillna(0)
    c_frac = commission_bps / 1e4
    s_frac = stamp_duty_bps / 1e4
    borrow_daily = (borrow_annual_bps / 1e4) / 252.0
    commission_cost = turn * 2 * c_frac          # 双腿
    stamp_cost = turn * 0.5 * s_frac             # 卖出腿、约半数换手
    borrow_cost = pos.abs() * borrow_daily       # 持仓期做空腿 carry
    net = gross - commission_cost - stamp_cost - borrow_cost

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

    return dict(
        beta=beta, spread=spread, z=z, split=split, window=window,
        hl_train=hl_train,
        is_gross=metrics(gross.iloc[is_slice]),
        is_net=metrics(net.iloc[is_slice]),
        oos_gross=metrics(gross.iloc[oos_slice], n_eff_div),
        oos_net=metrics(net.iloc[oos_slice], n_eff_div),
        n_round_trips_oos=completed, open_at_end_oos=open_at_end,
        oos_net_pnl=net.iloc[oos_slice],
        cost=dict(commission_bps=commission_bps, stamp_duty_bps=stamp_duty_bps,
                  borrow_annual_bps=borrow_annual_bps,
                  total_cost=float((commission_cost + stamp_cost + borrow_cost)
                                   .iloc[oos_slice].sum())),
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
    if is_s <= 0.05 and oos_s > 0.5:
        flags.append(("🔴 高", "样本外优势缺乏样本内支撑",
                      "样本内净Sharpe≈0/负但样本外显著>0（疑似区制依赖/机缘，非稳定edge）",
                      f"IS净={is_s:.2f} / OOS净={oos_s:.2f}"))
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
def write_report(a, b, source, px, adf_tr, kpss_tr, adf_full, adf_oos,
                 hl_tr, hl_full, bt, betastab, flags, path, attr=None, attr_sp=None):
    f = bt
    kernel = "statsmodels" if HAVE_SM else "numpy 近似（自测，非精确 p 值）"
    top = flags[0][0]
    alpha_ok = (attr is None) or (abs(attr["t_alpha"]) >= 1.96)
    tradable = (adf_tr[1] <= 0.05 and np.isfinite(hl_tr) and hl_tr <= 60
                and f["oos_net"]["sharpe"] > 0 and abs(f["oos_net"]["t"]) >= 1.96
                and bt["n_round_trips_oos"] >= 30 and alpha_ok)
    verdict = "证据指向可进一步研究" if tradable else "证据不足以支持可交易结论（建议否决/继续证伪）"

    if attr is None:
        attr_line = ("- 因子归因：未做（缺市场代理序列）。β 低且两腿同向时，"
                     "样本外收益可能混入方向性 beta，建议联网后对市场指数回归再下结论。")
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
        f"\n> 数据源：`{source}`｜统计内核：{kernel}"
        f"｜样本：{px.index.min().date()} ~ {px.index.max().date()}，共 {len(px)} 个交易日"
        f"｜训练/测试={int(0.7*100)}/{int(0.3*100)}（最近 30% 留出）\n",

        "## 1. 摘要与结论",
        f"- **总体判断：{verdict}**（最高风险等级 {top}，详见第 8 章）",
        f"- 对冲比率 β（仅训练窗估计）= **{f['beta']:.4f}**；价差 = log({a}) − β·log({b})",
        f"- 协整（训练窗价差）：ADF stat={adf_tr[0]:.3f}, p={_fmt_p(adf_tr[1], 0.001, 0.99)}；"
        f"KPSS stat={kpss_tr[0]:.3f}, p={_fmt_p(kpss_tr[1])} → {adf_tr[2]}",
        f"- 均值回归半衰期（训练窗）= **{hl_tr:.1f} 交易日**（z 回看窗自适应={f['window']}）",
        f"- 样本外（净，扣费）：Sharpe **{f['oos_net']['sharpe']:.2f}**"
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

        "\n## 7. 回测与偏差控制（毛 vs 净；样本内 vs 样本外）",
        "| 口径 | Sharpe | t | 最大回撤 | 累计价差收益 |",
        "|---|---|---|---|---|",
        f"| 样本内·毛 | {f['is_gross']['sharpe']:.2f} | {f['is_gross']['t']:.2f} | {f['is_gross']['mdd']:.4f} | {f['is_gross']['ret']:.4f} |",
        f"| 样本内·净 | {f['is_net']['sharpe']:.2f} | {f['is_net']['t']:.2f} | {f['is_net']['mdd']:.4f} | {f['is_net']['ret']:.4f} |",
        f"| 样本外·毛 | {f['oos_gross']['sharpe']:.2f} | {f['oos_gross']['t']:.2f} | {f['oos_gross']['mdd']:.4f} | {f['oos_gross']['ret']:.4f} |",
        f"| **样本外·净** | **{f['oos_net']['sharpe']:.2f}** | **{f['oos_net']['t']:.2f}** | {f['oos_net']['mdd']:.4f} | {f['oos_net']['ret']:.4f} |",
        f"\n成本：佣金 {f['cost']['commission_bps']:.0f}bp/腿/边（双腿）"
        f" + 卖出印花税 {f['cost']['stamp_duty_bps']:.0f}bp（单边，约半数换手）"
        f" + 做空腿融券 carry {f['cost']['borrow_annual_bps']:.0f}bp/年（按持仓天数）。"
        f"样本外累计成本（价差单位）≈ {f['cost']['total_cost']:.4f}。",
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
        f"| 信号/回测 | 向量化 z-score + 现实成本 | 样本内/外 | "
        f"窗={f['window']}(自适应), 开={2.0}, 平={0.5}, 止={3.5} | 完成往返 {f['n_round_trips_oos']} |",
        f"| 显著性 | Sharpe t 统计量 + 重叠折减 | 样本外 | "
        f"t={f['oos_net']['t']:.2f}, t_eff={f['oos_net']['t_eff']:.2f} | |t|<2 视为不显著 |",
        (f"| 因子归因 | 样本外净收益 OLS 回归市场代理 | 样本外 | "
         f"β_mkt={attr['beta']:.2f}(t={attr['t_beta']:.2f}), α年化={attr['alpha_annual']:.3f}"
         f"(t={attr['t_alpha']:.2f}), R²={attr['r2']:.2f} | 残差α不显著则优势存疑(β显著→漏beta；β也不显著→噪声) |"
         if attr is not None else
         "| 因子归因 | 样本外净收益回归市场代理 | 样本外 | 未做（缺市场代理序列） | 联网后补 |"),
        "\n**本脚本已实现的稳健性内核**：ADF+KPSS（训练窗，全样本/OOS 对照）、收益空间对冲比率"
        "稳定性（Chow 式半样本 β 断点 + 噪声校正离散度）、AR(1) 半衰期、含成本的样本外回测、"
        "Sharpe t 与重叠折减、两级因子归因（策略 α/β + 价差市场中性）。",
        "**未实现、需 Agent 另行补充的进阶项**：Johansen 协整、Kalman 动态对冲比率、"
        "完整 Chow/CUSUM 多断点/路径检验、滚动前推(walk-forward)。报告中不得把这些写成已自动完成。",
        "\n---",
        "本报告基于公开数据与规则化分析生成，仅供研究参考，不构成任何投资建议。",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ============ 分析流水线（CLI 与回放共用） ============
def analyze(px, a, b, source, mkt, out, window=None, **costs):
    bt = backtest(px, window=window, **costs)

    split = bt["split"]
    spread = bt["spread"]
    log = np.log(px)

    # 协整：训练窗（标题）+ 全样本/样本外对照
    adf_tr = adf_test(spread.iloc[:split])
    kpss_tr = kpss_test(spread.iloc[:split])
    adf_full = adf_test(spread)
    adf_oos = adf_test(spread.iloc[split:])

    hl_tr = bt["hl_train"]
    hl_full = half_life(spread)
    betastab = beta_stability(log[px.columns[0]], log[px.columns[1]])

    # 因子归因：样本外净收益 vs 市场代理（查优势是否漏入方向性 beta）
    attr = attr_sp = None
    if mkt is not None:
        attr = factor_attribution(bt["oos_net_pnl"], mkt)                      # 策略收益层
        spread_ret_oos = spread.diff().iloc[split:]                            # 价差中性层
        attr_sp = factor_attribution(spread_ret_oos, mkt)

    flags = robustness_flags(adf_tr[1], kpss_tr[1], hl_tr, bt, betastab, attr, attr_sp)
    write_report(a, b, source, px, adf_tr, kpss_tr, adf_full, adf_oos,
                 hl_tr, hl_full, bt, betastab, flags, out, attr, attr_sp)

    print(f"[done] {a}×{b} via {source}")
    print(f"  ADF_tr stat={adf_tr[0]:.3f} p={_fmt_p(adf_tr[1], 0.001, 0.99)} "
          f"| KPSS_tr p={_fmt_p(kpss_tr[1])} "
          f"| half-life={hl_tr:.1f}d | z-win={bt['window']} | excess_drift={betastab['excess_drift']:.2f}")
    attr_s = (f"strat β={attr['beta']:.2f}(t={attr['t_beta']:.2f}) α_t={attr['t_alpha']:.2f}"
              + (f" | spread β={attr_sp['beta']:.2f}(t={attr_sp['t_beta']:.2f})" if attr_sp else "")
              if attr is not None else "未做")
    print(f"  OOS net Sharpe={bt['oos_net']['sharpe']:.2f} (t={bt['oos_net']['t']:.2f}, "
          f"t_eff={bt['oos_net']['t_eff']:.2f}) | round-trips={bt['n_round_trips_oos']} "
          f"| factor: {attr_s} | top flag={flags[0][0]} {flags[0][1]}")
    print(f"  report -> {out}")


# ============ 回放：研究配置 + 固定 MCP 快照 → 报告 + 清单（不取实时数据） ============
class ReplayInputError(ValueError):
    """配置或快照缺字段/格式错误；消息指明位置与缺什么。"""


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


def _snapshot_prices(snap, legs, field):
    """快照 → 对齐收盘价 DataFrame（列 = legs）。逐行校验必需字段。"""
    _require(snap, ("server", "provenance", "calls"), "snapshot")
    series = {}
    for i, call in enumerate(snap["calls"]):
        _require(call, ("tool", "params", "retrieved_at", "rows"), f"calls[{i}]")
        for j, row in enumerate(call["rows"]):
            where = f"calls[{i}].rows[{j}]"
            _require(row, ("ts_code", "trade_date", field), where)
            d = str(row["trade_date"])
            if len(d) != 8 or not d.isdigit():
                raise ReplayInputError(f"{where}: trade_date '{d}' is not YYYYMMDD")
            if not _is_num(row[field]):
                raise ReplayInputError(f"{where}: '{field}' is not numeric: {row[field]!r}")
            leg = series.setdefault(row["ts_code"], {})
            if pd.Timestamp(d) in leg:     # 分页重叠等导致的重复行：不静默覆盖
                raise ReplayInputError(f"{where}: duplicate {row['ts_code']} {d}")
            leg[pd.Timestamp(d)] = float(row[field])
    missing = [leg for leg in legs if leg not in series]
    if missing:
        raise ReplayInputError(
            f"snapshot has no rows for {missing}; available: {sorted(series)}")
    px = pd.DataFrame({leg: pd.Series(series[leg]) for leg in legs}).sort_index().dropna()
    if len(px) < 60:
        raise ReplayInputError(f"legs {legs} share only {len(px)} dates in snapshot; need ≥60")
    return px, {leg: len(series[leg]) for leg in legs}


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
    import platform
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    if HAVE_SM:
        import statsmodels
        v["statsmodels"] = statsmodels.__version__
    return v


def replay(config_path, out_dir):
    """用研究配置 + 固定 MCP 响应快照离线复跑，写 report.md 与 manifest.json。
    config: {"snapshot": 相对配置文件的路径, "legs": [ts_code_a, ts_code_b],
             "price_field": "close", "window": 0, "costs": {commission_bps,...}}
    run_id 只由 配置/快照/脚本 的内容哈希决定：输入不变 → run_id 与报告不变。"""
    config_path, out_dir = Path(config_path), Path(out_dir)
    cfg = _read_json(config_path, "config")
    _require(cfg, ("snapshot", "legs"), "config")
    legs = cfg["legs"]
    if not (isinstance(legs, list) and len(legs) == 2 and all(isinstance(x, str) for x in legs)):
        raise ReplayInputError(f"config: 'legs' must be two ts_code strings, got {legs!r}")
    field = cfg.get("price_field", "close")
    if not isinstance(field, str):
        raise ReplayInputError(f"config: 'price_field' must be a string, got {field!r}")
    window = cfg.get("window", 0)
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ReplayInputError(f"config: 'window' must be an integer ≥0 (0=自适应), got {window!r}")
    costs = cfg.get("costs", {})
    if not isinstance(costs, dict):
        raise ReplayInputError(f"config: 'costs' must be an object, got {costs!r}")
    unknown = set(costs) - set(COST_DEFAULTS)
    if unknown:
        raise ReplayInputError(f"config.costs: unknown keys {sorted(unknown)}")
    for k, v in costs.items():
        if not _is_num(v):
            raise ReplayInputError(f"config.costs: '{k}' must be a number, got {v!r}")
    costs_used = {k: float(costs.get(k, d)) for k, d in COST_DEFAULTS.items()}

    snap_path = config_path.parent / cfg["snapshot"]
    snap = _read_json(snap_path, "snapshot")
    px, rows_per_leg = _snapshot_prices(snap, legs, field)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    analyze(px, legs[0], legs[1], "mcp", None, str(report),
            window=window or None, **costs_used)

    snap_sha, cfg_sha = _sha256(snap_path), _sha256(config_path)
    script_sha = _sha256(__file__)
    manifest = {
        "run_id": hashlib.sha256(f"{cfg_sha}{snap_sha}{script_sha}".encode()).hexdigest()[:16],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "config_sha256": cfg_sha,
        "costs_used": costs_used,
        "snapshot": {
            "path": cfg["snapshot"], "sha256": snap_sha,
            "snapshot_version": snap.get("snapshot_version"),
            "server": snap["server"], "server_version": snap.get("server_version"),
            "provenance": snap["provenance"],
            "calls": [{"tool": c["tool"], "params": c["params"],
                       "retrieved_at": c["retrieved_at"], "n_rows": len(c["rows"])}
                      for c in snap["calls"]],
        },
        "code": {"script_sha256": script_sha, **_git_version(), "libs": _lib_versions()},
        "data_window": {"start": px.index[0].strftime("%Y-%m-%d"),
                        "end": px.index[-1].strftime("%Y-%m-%d"), "n_aligned": len(px),
                        "rows_per_leg": rows_per_leg},
        "outputs": {"report.md": _sha256(report)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ============ main ============
def main():
    sys.stdout.reconfigure(errors="replace")   # gbk 控制台打印 emoji 不崩
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="回放：研究配置 JSON（含 zeus MCP 快照路径），不取实时数据")
    ap.add_argument("--out-dir", default="statarb_run", help="回放输出目录")
    ap.add_argument("--source", default="synthetic", choices=["synthetic"],
                    help="无 --config 时仅支持合成数据自测；真实数据请用 --config 回放 MCP 快照")
    ap.add_argument("--cointegrated", type=int, default=1)
    ap.add_argument("--mode", default="",
                    choices=["", "coint", "nocoint", "strong", "inversion", "leaked", "drift"],
                    help="合成数据分支自测；留空则用 --cointegrated")
    ap.add_argument("--window", type=int, default=0, help="z 回看窗；0=按半衰期自适应")
    for k, d in COST_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=d)
    ap.add_argument("--out", default="statarb_report.md")
    args = ap.parse_args()

    if args.config:
        try:
            m = replay(args.config, args.out_dir)
        except ReplayInputError as e:
            raise SystemExit(f"[replay input error] {e}")
        print(f"[done] replay run_id={m['run_id']} -> {args.out_dir}")
        return

    px = _synthetic_pair(cointegrated=bool(args.cointegrated), mode=args.mode or None)
    analyze(px, "A", "B", "synthetic", load_market(px), args.out,
            window=args.window or None, **{k: getattr(args, k) for k in COST_DEFAULTS})


if __name__ == "__main__":
    main()
