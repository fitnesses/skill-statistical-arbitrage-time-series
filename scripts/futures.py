"""futures.py — 国内商品期货规则层。

  1. 合约代码：<品种><YYMM>.<交易所>（如 RB2501.SHF）；连续代码（RB.SHF）无交割月。
  2. 合约映射与换月（按时点安全）：交易日 t 用哪张合约，只看 t-1 收盘后已知的信息
     （前一日持仓量、上市合约、按合约代码可知的交割月）。
       - dominant：前一日持仓量最大者；粘性且只向更远月份换（新合约前一日持仓量
         超过当前合约才换），进入“交割月前 N 个月”禁持窗口时强制换出。
       - second：交割月晚于同日 dominant 腿的合约中，前一日持仓量最大者（跨期远月腿）。
       - ts_code：固定合约，不换月。
  3. 研究序列 vs 可执行价格：研究用连续对数价 = 累加“当日所持合约自身的日收益”，
     换月日不产生拼接跳空，只用于统计诊断；回测盈亏一律用真实合约价格。
  4. 可执行回测：整手、真实合约、成交价/盯市价可配，手续费（费率+每手）与滑点（跳）
     逐笔计，保证金按多空费率逐日计；一腿涨跌停封板或无行情 → 两腿都不成交（顺延），
     成交量不足按参与率上限标记。
"""
import bisect, re

import numpy as np
import pandas as pd

BAR_FIELDS = ("open", "high", "low", "close", "settle", "vol", "oi")
_CODE = re.compile(r"^([A-Za-z]+?)(\d+)?\.([A-Za-z]+)$")


class MissingData(ValueError):
    """缺少必需的合约参数/行情；消息指明缺什么、应由哪个 zeus 工具提供、配置里的替代键。"""


# ---------------- 合约代码 ----------------
def parse_code(ts_code):
    """RB2501.SHF → ('RB', (2025, 1), 'SHF')；RB.SHF → ('RB', None, 'SHF')。"""
    m = _CODE.match(ts_code or "")
    if not m:
        raise ValueError(f"无法解析合约代码 {ts_code!r}")
    prod, digits, exch = m.groups()
    if digits is None:
        return prod.upper(), None, exch.upper()
    if len(digits) != 4:
        raise ValueError(f"合约代码 {ts_code!r} 的年月不是 4 位 YYMM，无法唯一确定交割月")
    return prod.upper(), (2000 + int(digits[:2]), int(digits[2:])), exch.upper()


def months_to_delivery(ts_code, day):
    ym = parse_code(ts_code)[1]
    return (ym[0] * 12 + ym[1]) - (day.year * 12 + day.month) if ym else np.inf


def bar_tables(rows):
    """fut_daily 行 → {字段: DataFrame(index=交易日, columns=合约)}。"""
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    fields = [f for f in BAR_FIELDS + ("pre_settle",) if f in df.columns]
    return {f: df.pivot(index="trade_date", columns="ts_code", values=f).sort_index()
            for f in fields}


# ---------------- 映射与换月 ----------------
def _product_codes(columns, product, exchange):
    out = []
    for c in columns:
        try:
            p, ym, e = parse_code(c)
        except ValueError:
            continue
        if p == product and e == exchange and ym is not None:
            out.append(c)
    return out


def build_mapping(tabs, legs, exclude_months=1):
    """返回 (mapping, rolls)。mapping.loc[t, 腿] = 交易日 t 该腿持有的合约（首日无前一日 → 不映射）。"""
    oi = tabs["oi"]
    dates = oi.index
    names = [leg["name"] for leg in legs]
    codes = {leg["name"]: _product_codes(oi.columns, leg["product"], leg["exchange"])
             for leg in legs if "ts_code" not in leg}
    cur = {n: None for n in names}
    out, rolls = [], []
    for i in range(1, len(dates)):
        t, prev = dates[i], dates[i - 1]
        row = {}
        for leg in legs:
            n = leg["name"]
            if "ts_code" in leg:
                row[n] = leg["ts_code"]
                continue
            oi_prev = oi.loc[prev, codes[n]].dropna()
            near = row.get(names[0]) if leg["select"] == "second" else None
            ok = lambda c: (months_to_delivery(c, t) > exclude_months
                            and (near is None or parse_code(c)[1] > parse_code(near)[1]))
            cands = [c for c in oi_prev.index if ok(c)]
            if leg["select"] == "second" and not near:
                cands = []
            key = lambda c: (oi_prev[c], parse_code(c)[1])
            old, new, reason = cur[n], cur[n], None
            if old is not None and ok(old) and old not in oi_prev.index:
                pass                                  # 持有合约前一日缺行情：继续持有，不因缺数据换月
            elif not cands:
                new = None
            elif old is None or old not in cands:
                fwd = [c for c in cands if old is None or parse_code(c)[1] > parse_code(old)[1]]
                new = max(fwd, key=key) if fwd else None
                if old is not None:
                    near = row.get(names[0])
                    reason = ("delivery_exclusion" if months_to_delivery(old, t) <= exclude_months
                              else "near_leg_rolled_into_it" if leg["select"] == "second" and near
                              and parse_code(old)[1] <= parse_code(near)[1] else "not_eligible")
            else:
                later = [c for c in cands if parse_code(c)[1] > parse_code(old)[1]]
                best = max(later, key=key) if later else None
                if best and oi_prev[best] > oi_prev[old]:
                    new, reason = best, "oi_crossover"
            if old is not None and new is not None and new != old:
                rolls.append({"date": t, "decided_on": prev, "leg": n, "from_code": old, "to_code": new,
                              "reason": reason, "oi_from": float(oi_prev.get(old, np.nan)),
                              "oi_to": float(oi_prev[new]),
                              "from_delivery": parse_code(old)[1], "to_delivery": parse_code(new)[1]})
            cur[n] = new
            row[n] = new
        out.append(row)
    mapping = pd.DataFrame(out, index=dates[1:], columns=names)
    cols = ["date", "decided_on", "leg", "from_code", "to_code", "reason", "oi_from", "oi_to",
            "from_delivery", "to_delivery"]
    return mapping, pd.DataFrame(rolls, columns=cols)


def executable_prices(tabs, mapping, field):
    """每日各腿所映射合约的真实价格（可执行价格，含换月跳空）。"""
    tab = tabs[field]
    return pd.DataFrame({n: [tab.at[d, c] if isinstance(c, str) and c in tab.columns else np.nan
                             for d, c in mapping[n].items()]
                         for n in mapping.columns}, index=mapping.index)


def research_prices(tabs, mapping, field="close"):
    """研究用连续对数价：r_t = ln P(c_t, t) − ln P(c_t, t−1)，c_t 为 t 日映射合约；累加得序列。"""
    tab = np.log(tabs[field])
    last = tab.ffill()                     # 同一合约最近一个已知收盘（只向前填，不看未来）
    dates = list(tab.index)
    pos = {d: i for i, d in enumerate(dates)}
    out = {}
    for n in mapping.columns:
        vals, level = [], None
        for d, c in mapping[n].items():
            prev = dates[pos[d] - 1]
            if not isinstance(c, str) or c not in tab.columns or np.isnan(tab.at[d, c]):
                vals.append(np.nan)
                continue
            if level is None:
                level = tab.at[d, c]
            elif not np.isnan(last.at[prev, c]):
                level += tab.at[d, c] - last.at[prev, c]
            else:
                vals.append(np.nan)
                continue
            vals.append(level)
        out[n] = vals
    return pd.DataFrame(out, index=mapping.index)


# ---------------- 合约参数（zeus 优先，配置假设兜底，缺失即报错） ----------------
class ContractSpecs:
    """合约参数来源优先级：zeus fut_basic（乘数/最小变动价位）、fut_settle（按日生效的费率/保证金）、
    ft_limit（涨跌停价）→ 配置 assumptions[品种]（整段常数，报告披露为假设）→ 缺失报错。"""

    def __init__(self, basic, settle, limits, assumptions, broker_fee_multiplier=1.0,
                 broker_margin_add=0.0):
        self.basic = {r["ts_code"]: r for r in basic}
        self.assumptions = assumptions or {}
        self.fee_mult, self.margin_add = broker_fee_multiplier, broker_margin_add
        self.settle = {}
        for r in sorted(settle, key=lambda r: r["trade_date"]):
            self.settle.setdefault(r["ts_code"], []).append((pd.Timestamp(r["trade_date"]), r))
        self.limits = {(r["ts_code"], pd.Timestamp(r["trade_date"])): (r["up_limit"], r["down_limit"])
                       for r in limits}
        self.used = {}       # (字段, 来源) → 合约日数，用于披露
        self._cache = {}

    def _assume(self, code, field, tool):
        prod = parse_code(code)[0]
        v = self.assumptions.get(prod, {}).get(field)
        if v is None:
            raise MissingData(f"{code}: missing {field} — zeus {tool} has no value for it and "
                              f"config assumptions.{prod}.{field} is not set")
        return v

    def _count(self, field, src):
        self.used[(field, src)] = self.used.get((field, src), 0) + 1

    def spec(self, code, day):
        key = (code, day)
        if key not in self._cache:
            self._cache[key] = self._spec(code, day)
        return self._cache[key]

    def _spec(self, code, day):
        src = {}
        b = self.basic.get(code, {})
        out = {}
        for field, key in (("multiplier", "multiplier"), ("price_tick", "price_tick")):
            if b.get(key) is not None:
                out[field], src[field] = float(b[key]), "fut_basic"
            else:
                out[field], src[field] = float(self._assume(code, field, "fut_basic")), "assumption"
        rows = self.settle.get(code, [])
        i = bisect.bisect_right(rows, day, key=lambda x: x[0])   # 最近一条 trade_date ≤ day（as-of）
        row = rows[i - 1][1] if i else None
        pairs = (("fee_rate", "trading_fee_rate", "fee_rate"), ("fee_per_lot", "trading_fee", "fee_per_lot"),
                 ("margin_long", "long_margin_rate", "margin_rate"),
                 ("margin_short", "short_margin_rate", "margin_rate"))
        for field, zkey, akey in pairs:
            if row is not None and row.get(zkey) is not None:
                out[field], src[field] = float(row[zkey]), "fut_settle"
            else:
                out[field], src[field] = float(self._assume(code, akey, "fut_settle")), "assumption"
        out["fee_rate"] *= self.fee_mult
        out["fee_per_lot"] *= self.fee_mult
        out["margin_long"] += self.margin_add
        out["margin_short"] += self.margin_add
        for f, s in src.items():
            self._count(f, s)
        out["source"] = src
        return out

    def limit(self, code, day):
        return self.limits.get((code, day))


# ---------------- 可执行回测 ----------------
def _blocked(tabs, code, day, qty, specs):
    """返回阻塞原因或 None。优先用 zeus 涨跌停价；缺失时按 OHLC 一字板推断。"""
    try:
        hi, lo = tabs["high"].at[day, code], tabs["low"].at[day, code]
    except KeyError:
        return "no_bar"
    if np.isnan(hi) or np.isnan(lo):
        return "no_bar"
    lim = specs.limit(code, day)
    if lim is not None:
        up, down = lim
        if qty > 0 and lo >= up:
            return "limit_up_locked"
        if qty < 0 and hi <= down:
            return "limit_down_locked"
        return None
    if hi != lo:
        return None
    pre = tabs.get("pre_settle")
    ps = pre.at[day, code] if pre is not None and code in pre.columns else np.nan
    c = tabs["close"].at[day, code]
    if np.isnan(ps) or c == ps:
        return "one_price_bar(inferred)"
    if qty > 0 and c > ps:
        return "limit_up_locked(inferred)"
    if qty < 0 and c < ps:
        return "limit_down_locked(inferred)"
    return None


def executable_backtest(tabs, mapping, target, beta, specs, ex, split_date=None):
    """target.loc[d] = d 日收盘决定的目标价差仓位（-1/0/1），在下一交易日按 ex['exec_price'] 成交。

    手数规则（文档化）：A 腿 = base_lots；B 腿 = round(|β|·A手·P_A·M_A / (P_B·M_B))，至少 1 手，
    P 取决策日收盘价（映射合约）；β>0 时 B 腿与 A 腿反向。持仓期间手数不再调整，换月按原手数平旧开新。
    """
    a, b = mapping.columns
    dates = list(mapping.index)
    exec_tab, mark_tab = tabs[ex["exec_price"]], tabs[ex["mark_price"]].ffill()
    close = tabs["close"]
    sign_b = -1 if beta > 0 else 1
    hold, cur_pos, lots = {}, 0, None
    trades, daily, events, entries = [], [], [], []
    roll_ticks = ex.get("roll_slippage_ticks", 0)
    rolls_done = 0
    for i, d in enumerate(dates):
        prev = dates[i - 1] if i else None
        want = int(target.get(prev, 0)) if prev is not None else 0
        ca, cb = mapping.at[d, a], mapping.at[d, b]
        desired, new_lots = {}, lots
        if want != 0 and isinstance(ca, str) and isinstance(cb, str):
            if cur_pos != want or lots is None:
                sa, sb = specs.spec(ca, d), specs.spec(cb, d)
                pa, pb = close.at[prev, ca], close.at[prev, cb]
                la = int(ex["base_lots"])
                lb = max(1, int(round(abs(beta) * la * pa * sa["multiplier"] / (pb * sb["multiplier"]))))
                new_lots = (la, lb, pa, pb, sa["multiplier"], sb["multiplier"])
            desired = {ca: want * new_lots[0], cb: sign_b * want * new_lots[1]}
        elif want != 0:
            events.append({"date": d, "kind": "no_mapping", "detail": f"{a}={ca}, {b}={cb}"})
            desired = dict(hold)
        orders = {c: desired.get(c, 0) - hold.get(c, 0) for c in set(desired) | set(hold)}
        orders = {c: q for c, q in orders.items() if q != 0}
        blocks = {c: r for c, q in orders.items() if (r := _blocked(tabs, c, d, q, specs))}
        fills, fees, slip = [], 0.0, 0.0
        if blocks:
            events.append({"date": d, "kind": "deferred",
                           "detail": "; ".join(f"{c}:{r}" for c, r in blocks.items())})
        elif orders:
            is_roll = cur_pos == want and want != 0
            for c, q in sorted(orders.items()):
                sp = specs.spec(c, d)
                px = exec_tab.at[d, c]
                fee = abs(q) * (px * sp["multiplier"] * sp["fee_rate"] + sp["fee_per_lot"])
                ticks = ex["slippage_ticks"] + (roll_ticks if is_roll else 0)
                slip_q = abs(q) * ticks * sp["price_tick"] * sp["multiplier"]
                fees, slip = fees + fee, slip + slip_q
                reason = "roll" if is_roll else ("open" if q * (hold.get(c, 0) + q) > 0 and hold.get(c, 0) == 0
                                                 else "close" if hold.get(c, 0) + q == 0 else "adjust")
                vol = tabs["vol"].at[d, c]
                if abs(q) > ex["max_participation"] * vol:
                    events.append({"date": d, "kind": "liquidity",
                                   "detail": f"{c}: {abs(q)} 手 > {ex['max_participation']:.0%}×成交量 {vol:g}"})
                fills.append((c, q, px, sp))
                trades.append({"date": d, "signal_date": prev, "ts_code": c, "lots": q, "price": px,
                               "multiplier": sp["multiplier"],
                               "exec_price_field": ex["exec_price"], "fee": fee, "slippage": slip_q,
                               "reason": reason, "target_spread_pos": want})
            if is_roll:
                rolls_done += 1
            if want != 0 and cur_pos != want:         # 开仓或反手 → 新一笔交易
                la, lb, pa, pb, ma, mb = new_lots
                entries.append({"date": d, "lots_a": la, "lots_b": lb, "beta": beta,
                                "realized_hedge": lb * pb * mb / (la * pa * ma)})
            cur_pos, lots = want, (new_lots if want else None)
        # 盯市：持仓 × (今日盯市价 − 昨日盯市价) + 成交 × (今日盯市价 − 成交价)
        gross = 0.0
        for c, h in hold.items():
            m = specs.spec(c, d)["multiplier"]
            gross += h * (mark_tab.at[d, c] - mark_tab.at[prev, c]) * m
        for c, q, px, sp in fills:
            gross += q * (mark_tab.at[d, c] - px) * sp["multiplier"]
            hold[c] = hold.get(c, 0) + q
            if hold[c] == 0:
                del hold[c]
        margin = 0.0
        for c, h in hold.items():
            sp = specs.spec(c, d)
            margin += abs(h) * mark_tab.at[d, c] * sp["multiplier"] * (sp["margin_long"] if h > 0 else sp["margin_short"])
        funding = margin * ex["funding_rate_annual"] / 252.0
        daily.append({"date": d, "gross": gross, "fees": fees, "slippage": slip, "funding": funding,
                      "net": gross - fees - slip - funding, "margin": margin, "spread_pos": cur_pos,
                      "holdings": ";".join(f"{c}:{h:+d}" for c, h in sorted(hold.items()))})
    daily = pd.DataFrame(daily).set_index("date")
    trades = pd.DataFrame(trades, columns=["date", "signal_date", "ts_code", "lots", "price", "multiplier",
                                           "exec_price_field",
                                           "fee", "slippage", "reason", "target_spread_pos"])
    events = pd.DataFrame(events, columns=["date", "kind", "detail"])
    metrics = {"all": money_metrics(daily, trades, ex["capital"])}
    if split_date is not None:
        metrics["is"] = money_metrics(daily.loc[:split_date].iloc[:-1], trades[trades.date < split_date], ex["capital"])
        metrics["oos"] = money_metrics(daily.loc[split_date:], trades[trades.date >= split_date], ex["capital"])
    metrics["all"]["rolls"] = rolls_done
    return {"trades": trades, "daily": daily, "events": events, "entries": entries, "metrics": metrics}


def money_metrics(daily, trades, capital):
    """金额口径绩效：毛/净 PnL、成本拆分、Sharpe(净收益/资本)、回撤、换手、完成往返、保证金。"""
    n = len(daily)
    if n == 0:
        return {"days": 0}
    years = n / 252.0
    ret = daily.net / capital
    sd = ret.std()
    sharpe = float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    eq = daily.net.cumsum()
    mdd = float((eq - eq.cummax()).min())
    pos = daily.spread_pos.to_numpy()
    rt = int(sum(1 for i in range(1, n) if pos[i - 1] != 0 and pos[i] != pos[i - 1]))   # 平仓或反手
    held = daily.margin[daily.margin > 0]
    notional = float((trades.lots.abs() * trades.price * trades.multiplier).sum()) if len(trades) else 0.0
    return {"days": n, "gross": float(daily.gross.sum()), "fees": float(daily.fees.sum()),
            "slippage": float(daily.slippage.sum()), "funding": float(daily.funding.sum()),
            "net": float(daily.net.sum()), "ann_return": float(daily.net.sum() / capital / years),
            "sharpe": sharpe, "t": sharpe * np.sqrt(years), "max_drawdown": mdd,
            "max_drawdown_pct": mdd / capital, "round_trips": rt,
            "lots_traded": int(trades.lots.abs().sum()) if len(trades) else 0,
            "notional_traded": notional, "turnover_ann": notional / capital / years,
            "margin_max": float(daily.margin.max()), "margin_avg_held": float(held.mean()) if len(held) else 0.0,
            "return_on_margin_ann": (float(daily.net.sum() / years / held.mean()) if len(held) else float("nan"))}
