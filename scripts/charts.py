"""charts.py — 报告图表（PNG）。只读回放已产出的数据，不参与任何计算（HTML 转换见 run_statarb.to_html）。

配色取 dataviz 参考色板（浅色）：系列 1 蓝 #2a78d6、系列 2 橙 #eb6834、系列 3 青 #1baf7a
（前三色已用 validate_palette.js 校验；青色对比度 2.74:1 < 3:1，按规则配直接标注/表格）。
坐标轴与图例用英文，避免服务器缺中文字体时出现方框；中文说明写在报告正文里。
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
S1_LIGHT, S2_LIGHT = "#86b6ef", "#f4a886"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OOS_WASH = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK2, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "axes.axisbelow": True,
})


# ---------------- 基础元素 ----------------
def _oos(ax, split, end):
    """样本外区域（split → end）铺浅灰底；用明确的结束日期，不依赖当时的坐标范围。顺带设置紧凑日期刻度。"""
    ax.axvspan(split, end, color=OOS_WASH, zorder=0, lw=0)
    ax.axvline(split, color=MUTED, lw=0.8)
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _oos_label(ax, split):
    ax.annotate("out-of-sample", xy=(split, 1), xycoords=("data", "axes fraction"), xytext=(4, -10),
                textcoords="offset points", color=MUTED, fontsize=8)


def _end_label(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK2, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=color, lw=0.8))


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _fills(daily):
    """由逐日持有的价差仓位推出开/平仓，返回它们的**信号日**（成交日的前一交易日：信号在收盘产生、次日成交）。"""
    pos = daily.spread_pos
    prev = pos.shift(1).fillna(0)
    signal_day = pd.Series(daily.index, index=daily.index).shift(1)
    pick = lambda mask: pd.DatetimeIndex(signal_day[mask].dropna())
    return pick((pos == 1) & (prev != 1)), pick((pos == -1) & (prev != -1)), pick((prev != 0) & (pos != prev))


def _leg_labels(p):
    """腿名是英文就用腿名，否则用 “leg N (品种)”（坐标轴只用英文，避免缺中文字体）。"""
    out = []
    for i, leg in enumerate(p["mapping"].columns):
        code = next((c for c in p["mapping"][leg] if isinstance(c, str)), "")
        prod = re.sub(r"^([A-Za-z]+).*$", r"\1", code)
        out.append(leg if str(leg).isascii() else f"leg {i + 1} ({prod})")
    return out


# ---------------- 单图绘制（可放进总览网格） ----------------
def draw_zscore(ax, p, labels=True):
    z = p["z"].clip(-6, 6)
    ax.plot(z.index, z.values, color=S1, lw=1.1, label="z-score")
    for lvl, c, name in ((p["entry"], MUTED, "entry ±%g" % p["entry"]), (p["exit"], BASE, "exit ±%g" % p["exit"]),
                         (p["stop"], INK2, "stop ±%g" % p["stop"])):
        for s in (1, -1):
            ax.axhline(s * lvl, color=c, lw=0.8, ls=(0, (4, 3)), label=name if s == 1 else None)
    li, si, ex = _fills(p["daily"])
    zz = z.reindex(p["daily"].index).ffill()
    ax.scatter(li, zz.reindex(li), marker="^", s=36, color=S3, edgecolor=SURFACE, lw=1.2, zorder=5, label="long spread (signal day)")
    ax.scatter(si, zz.reindex(si), marker="v", s=36, color=S2, edgecolor=SURFACE, lw=1.2, zorder=5, label="short spread (signal day)")
    ax.scatter(ex, zz.reindex(ex), marker="o", s=18, color=MUTED, edgecolor=SURFACE, lw=1.0, zorder=5, label="exit (signal day)")
    ax.set_xlim(z.index[0], z.index[-1])
    _oos(ax, p["split"], p["end"])
    ax.set_ylabel("z-score (research spread)")
    ax.set_title("Spread z-score, thresholds and trade signals (filled next trading day)")
    if labels:
        _oos_label(ax, p["split"])
        ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.32))
    else:
        ax.legend(handles=[h for h, l in zip(*ax.get_legend_handles_labels()) if "signal" in l or l == "z-score"],
                  ncol=4, loc="lower left", fontsize=7, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.9)


def draw_equity(ax, p, labels=True, stress=True):
    d = p["daily"]
    series = [("gross", d.gross.cumsum(), S1), ("net", d.net.cumsum(), S2)]
    if stress and p.get("stress_daily") is not None:
        series.append(("net, stress", p["stress_daily"].net.cumsum(), S3))
    for name, s, c in series:
        ax.plot(s.index, s.values, color=c, label=name)
        if labels:
            _end_label(ax, s.index[-1], s.values[-1], f"{name} {s.values[-1]:,.0f}", c)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.set_xlim(d.index[0], d.index[-1])
    _oos(ax, p["split"], p["end"])
    ax.set_ylabel("cumulative PnL (CNY)")
    ax.set_title("Executable backtest: cumulative PnL (actual contracts, integer lots)")
    ax.legend(loc="upper left")
    if labels:
        _oos_label(ax, p["split"])


def draw_drawdown(ax, p):
    """样本内、样本外分段计算回撤（各自从本区间的净值高点起算），与报告表格口径一致。"""
    net = p["daily"].net
    parts = []
    for seg, name in ((net[net.index < p["split"]], "IS"), (net[net.index >= p["split"]], "OOS")):
        if seg.empty:
            continue
        eq = seg.cumsum()
        dd = eq - eq.cummax()
        parts.append(dd)
        i = dd.values.argmin()
        _end_label(ax, dd.index[i], dd.values[i], f"{name} max {dd.values[i]:,.0f}", S2)
    dd = pd.concat(parts)
    ax.fill_between(dd.index, dd.values, 0, color=S2, alpha=0.18, lw=0)
    ax.plot(dd.index, dd.values, color=S2, lw=1.2, label="net drawdown")
    ax.set_xlim(dd.index[0], dd.index[-1])
    _oos(ax, p["split"], p["end"])
    ax.set_ylabel("drawdown (CNY)")
    ax.set_title("Net drawdown (in-sample and out-of-sample measured separately)")


def draw_timeline(ax, p):
    m, legs = p["mapping"], list(p["mapping"].columns)
    tones = [(S1, S1_LIGHT), (S2, S2_LIGHT)]
    for row, leg in enumerate(legs):
        codes = m[leg]
        seg_id = (codes != codes.shift()).cumsum()
        k = -1
        for _, seg in codes.groupby(seg_id):
            code = seg.iloc[0]
            if not isinstance(code, str):
                continue
            k += 1                                   # 只数有合约的段，保证相邻合约深浅交替
            x0, x1 = seg.index[0], seg.index[-1]
            ax.barh(row, (x1 - x0).days + 1, left=x0, height=0.55, color=tones[row % 2][k % 2],
                    edgecolor=SURFACE, lw=1.5)
            if (x1 - x0).days > 45:
                ax.text(x0 + (x1 - x0) / 2, row, re.sub(r"^[A-Za-z]+(\d+)\..*$", r"\1", code),
                        ha="center", va="center", fontsize=7, color=SURFACE if k % 2 == 0 else INK)
    r = p["rolls"]
    for row, leg in enumerate(legs):
        cross = r[(r.leg == leg) & (r.reason == "oi_crossover")]
        forced = r[(r.leg == leg) & (r.reason != "oi_crossover")]
        ax.scatter(cross.date, [row + 0.42] * len(cross), marker="|", s=60, color=MUTED, zorder=5,
                   label="roll: open-interest crossover" if row == 0 else None)
        ax.scatter(forced.date, [row + 0.42] * len(forced), marker="v", s=22, color=INK2, zorder=5,
                   label="roll: forced (delivery window / near leg)" if row == 0 else None)
    ax.set_yticks(range(len(legs)), _leg_labels(p))
    ax.set_ylim(-0.6, len(legs) - 0.2)
    ax.set_xlim(m.index[0], m.index[-1])
    ax.grid(axis="y", visible=False)
    _oos(ax, p["split"], p["end"])
    ax.set_title("Contract held by each leg (label = YYMM); rolls at segment boundaries")
    if len(r):
        ax.legend(loc="lower left", bbox_to_anchor=(0, -0.35), ncol=2)


def draw_research_vs_exec(axes, p):
    names = _leg_labels(p)
    for i, (ax, leg) in enumerate(zip(axes, p["mapping"].columns)):
        ex = p["exec_close"][leg]
        rs_ = np.exp(p["research"][leg].reindex(ex.index))
        ax.plot(ex.index, ex.values, color=MUTED, lw=1.0, label="actual contract close (executable)")
        ax.plot(rs_.index, rs_.values, color=S1 if i == 0 else S2, lw=1.4, label="research continuous series")
        for d in p["rolls"][p["rolls"].leg == leg].date:
            ax.axvline(d, color=GRID, lw=0.9, zorder=1)
        ax.set_xlim(ex.index[0], ex.index[-1])
        _oos(ax, p["split"], p["end"])
        ax.set_ylabel(f"{names[i]} price")
        ax.set_title(f"{names[i]}: research series vs actual contract price (vertical lines = rolls)")
        ax.legend(loc="upper left")


def draw_waterfall(ax, m, title):
    if not m or not m.get("days"):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.set_title(title)
        return
    steps = [("gross", m["gross"], "total"), ("fees", -m["fees"], "step"), ("slippage", -m["slippage"], "step"),
             ("funding", -m["funding"], "step"), ("net", m["net"], "total")]
    run = 0.0
    for i, (name, v, kind) in enumerate(steps):
        bottom = 0.0 if kind == "total" else run
        ax.bar(i, v, bottom=bottom, width=0.62, color=S1 if kind == "total" else S2,
               edgecolor=SURFACE, lw=2)
        run = v if kind == "total" else run + v
        top = bottom + v
        ax.annotate(f"{v:,.0f}", xy=(i, max(top, bottom)), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=BASE, lw=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.14 * (hi - lo))      # 给数值标签留空间
    ax.set_xticks(range(len(steps)), [s[0] for s in steps])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("CNY")
    ax.set_title(title)


def draw_rolling(axes, p):
    r = p["rolling"]
    if r is None or r.empty:
        axes[0].text(0.5, 0.5, "not enough data for rolling windows", ha="center", va="center",
                     transform=axes[0].transAxes, color=MUTED)
        return
    finite = r.half_life[np.isfinite(r.half_life)]
    cap = max(60.0, 1.15 * float(finite.max())) if len(finite) else 60.0
    hl = r.half_life.clip(upper=cap)                    # 无穷半衰期（不回归）画在顶部，不留空白
    inf = r.half_life.index[~np.isfinite(r.half_life)]
    axes[1].scatter(inf, [cap] * len(inf), marker="^", s=24, color=S2, zorder=5,
                    label="∞: no mean reversion in window")
    panels = [(axes[0], r.beta, "rolling hedge ratio β", p["beta"], "fitted β (train)"),
              (axes[1], hl, "rolling half-life (days)", None, None),
              (axes[2], r.adf_p, "rolling ADF p-value", 0.05, "p = 0.05")]
    for ax, s, name, ref, ref_name in panels:
        ax.plot(s.index, s.values, color=S1, lw=1.3, label=name)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=0.8, ls=(0, (4, 3)), label=ref_name)
        ax.set_xlim(p["mapping"].index[0], p["mapping"].index[-1])
        _oos(ax, p["split"], p["end"])
        ax.set_title(name + f"  ({p['rolling_window']}-day trailing window, diagnostic only)")
        ax.legend(loc="upper left")


def draw_seasonality(axes, p):
    """训练窗：上=价差日变化按月箱线图（季节性检验对象），下=价差水平相对训练均值的偏离按月。"""
    s = p["spread_train"].dropna()
    flagged = set((p.get("season_months") or {}).get("months", []))
    series = [(axes[0], s.diff().dropna(), "Training window: daily spread change by calendar month"),
              (axes[1], s - s.mean(), "Training window: spread level minus training mean, by calendar month")]
    for ax, x, title in series:
        months = [m for m in range(1, 13) if (x.index.month == m).sum() > 0]
        data = [x[x.index.month == m].values for m in months]
        bp = ax.boxplot(data, positions=months, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED, lw=0.8),
                        capprops=dict(color=MUTED, lw=0.8))
        for m, box in zip(months, bp["boxes"]):
            box.set(facecolor=S2 if m in flagged else S1_LIGHT, edgecolor=SURFACE, lw=1.5)
        ax.axhline(0, color=BASE, lw=0.8)
        ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=S1_LIGHT, label="other months"),
                            Patch(facecolor=S2, label="month flagged by per-month test")], loc="upper left")


# ---------------- 输出 ----------------
def render(out_dir, p):
    """p：回放产出的数据。返回 {图名: 相对路径}。"""
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    def save(name, fig):
        _save(fig, fig_dir / f"{name}.png")
        out[name] = f"figures/{name}.png"

    fig, ax = plt.subplots(figsize=(11, 4.2))
    draw_zscore(ax, p)
    save("zscore_signals", fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    draw_equity(a1, p)
    draw_drawdown(a2, p)
    save("equity_drawdown", fig)

    fig, ax = plt.subplots(figsize=(11, 2.6))
    draw_timeline(ax, p)
    save("roll_timeline", fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    draw_research_vs_exec(axes, p)
    save("research_vs_exec", fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))         # 两个区间各自刻度：量级常差一个数量级
    draw_waterfall(a1, p["metrics"].get("is"), "In-sample: gross → costs → net")
    draw_waterfall(a2, p["metrics"].get("oos"), "Out-of-sample: gross → costs → net")
    save("cost_waterfall", fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    draw_rolling(axes, p)
    save("rolling_stability", fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"hspace": 0.45})
    draw_seasonality(axes, p)
    save("seasonality", fig)

    fig = plt.figure(figsize=(11, 15))                  # 4 行 1 列：一行一张，横向空间留给时间轴
    g = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 0.75, 0.95], hspace=0.6)
    ax = fig.add_subplot(g[0]); draw_zscore(ax, p, labels=False); ax.set_title("1. Spread z-score & trade signals")
    ax = fig.add_subplot(g[1]); draw_equity(ax, p, labels=False, stress=False)
    ax.set_title("2. Executable backtest: cumulative PnL (CNY)")
    ax = fig.add_subplot(g[2]); draw_timeline(ax, p); ax.set_title("3. Contracts held by each leg (rolls)")
    draw_waterfall(fig.add_subplot(g[3]), p["metrics"].get("oos"), "4. Out-of-sample: gross → costs → net")
    save("overview", fig)
    return out
