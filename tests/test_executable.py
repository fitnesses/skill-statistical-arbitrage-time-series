"""可执行回测：整手、真实合约价格、成本/保证金、一腿无行情则两腿都不成交、流动性标记。"""

import numpy as np
import pandas as pd
import pytest

import futures as fx

D = [f"2024010{i}" for i in range(2, 9)]          # 7 个交易日
PX = {  # code: [(open, settle)] 每日
    "AA2405.SHF": [(100, 100), (101, 102), (103, 104), (104, 105), (106, 106), (107, 108), (108, 108)],
    "AA2409.SHF": [(110, 110), (111, 112), (113, 114), (115, 116), (116, 117), (118, 118), (119, 120)],
    "BB2405.SHF": [(60, 60), (60, 61), (61, 61), (62, 63), (63, 63), (64, 64), (64, 65)],
}
SPEC = {"multiplier": 10, "price_tick": 1, "fee_rate": 0.0001, "fee_per_lot": 1.0, "margin_rate": 0.1}
EX = {"base_lots": 10, "exec_price": "open", "mark_price": "settle", "slippage_ticks": 1,
      "max_participation": 0.05, "capital": 1_000_000, "funding_rate_annual": 0.0,
      "broker_fee_multiplier": 1.0, "broker_margin_add": 0.0}


def rows(missing=None, vol=1e6):
    out = []
    for code, series in PX.items():
        for d, (o, s) in zip(D, series):
            hi, lo = max(o, s) + 1, min(o, s) - 1
            if missing and (code, d) == missing:
                continue                                     # 当日无行情
            out.append({"ts_code": code, "trade_date": d, "open": float(o), "high": float(hi),
                        "low": float(lo), "close": float(s), "settle": float(s),
                        "pre_settle": float(s), "vol": vol, "oi": 1000.0})
    return out


def specs():
    return fx.ContractSpecs(basic=[], assumptions={"AA": SPEC, "BB": SPEC})


def run(rs, target, mapping=None, sp=None, ex=EX, beta=0.5):
    tabs = fx.bar_tables(rs)
    idx = tabs["close"].index
    if mapping is None:
        mapping = pd.DataFrame({"a": "AA2405.SHF", "b": "BB2405.SHF"}, index=idx)
    tgt = pd.Series(target, index=idx[:len(target)]).reindex(idx).fillna(0)
    return fx.executable_backtest(tabs, mapping, tgt, beta, sp or specs(), ex)


def test_integer_lots_pnl_costs_and_margin_from_actual_contract_prices():
    res = run(rows(), [1, 1, 0])       # 第 0 日收盘开多价差 → 第 1 日开盘成交；第 2 日收盘平 → 第 3 日开盘平
    tr = res["trades"]
    opens = tr[tr.date == pd.Timestamp(D[1])]
    # 手数：A=10，B=round(0.5·10·100·10/(60·10))=8（用决策日收盘价）；β>0 → 多 A 空 B
    assert dict(zip(opens.ts_code, opens.lots)) == {"AA2405.SHF": 10, "BB2405.SHF": -8}
    assert np.isclose(res["entries"][0]["realized_hedge"], 8 * 60 / (10 * 100))
    closes = tr[tr.date == pd.Timestamp(D[3])]
    assert dict(zip(closes.ts_code, closes.lots)) == {"AA2405.SHF": -10, "BB2405.SHF": 8}

    # 毛 PnL（元）：成交价→结算价，再逐日结算价盯市
    m = 10
    exp_gross = (10 * (102 - 101) - 8 * (61 - 60)) * m            # D1
    exp_gross += (10 * (104 - 102) - 8 * (61 - 61)) * m           # D2
    exp_gross += (10 * (104 - 104) - 8 * (62 - 61)) * m           # D3 平仓价(开盘)对前结算
    daily = res["daily"]
    assert np.isclose(daily.gross.sum(), exp_gross)
    fee = lambda q, p: abs(q) * (p * m * 0.0001 + 1.0)
    exp_fee = fee(10, 101) + fee(8, 60) + fee(10, 104) + fee(8, 62)
    assert np.isclose(daily.fees.sum(), exp_fee)
    assert np.isclose(daily.slippage.sum(), (10 + 8 + 10 + 8) * 1 * 1 * m)
    assert np.isclose(daily.net.sum(), exp_gross - exp_fee - daily.slippage.sum())
    d1 = daily.loc[pd.Timestamp(D[1])]
    assert np.isclose(d1.margin, (10 * 102 + 8 * 61) * m * 0.1)
    assert res["metrics"]["all"]["round_trips"] == 1


def test_roll_while_holding_closes_old_and_opens_new_contract_with_same_lots():
    tabs = fx.bar_tables(rows())
    idx = tabs["close"].index
    mapping = pd.DataFrame({"a": ["AA2405.SHF"] * 3 + ["AA2409.SHF"] * 4, "b": "BB2405.SHF"}, index=idx)
    res = run(rows(), [1, 1, 1, 1, 1], mapping=mapping)
    tr = res["trades"]
    roll = tr[tr.date == idx[3]]
    assert dict(zip(roll.ts_code, roll.lots)) == {"AA2405.SHF": -10, "AA2409.SHF": 10}
    assert set(roll.reason) == {"roll"}
    assert res["metrics"]["all"]["rolls"] == 1


def test_one_leg_without_a_bar_defers_both_legs():
    res = run(rows(missing=("BB2405.SHF", D[1])), [1, 1, 1, 0])
    tr, ev = res["trades"], res["events"]
    assert tr[tr.date == pd.Timestamp(D[1])].empty                 # 两腿都未成交
    assert ((ev.date == pd.Timestamp(D[1])) & (ev.kind == "deferred")).any()
    assert "BB2405.SHF" in ev[ev.kind == "deferred"].detail.iloc[0]
    assert set(tr[tr.date == pd.Timestamp(D[2])].ts_code) == {"AA2405.SHF", "BB2405.SHF"}


def test_one_price_bar_is_not_treated_as_limit_lock():
    """涨跌停暂不建模：一字板照常按假设价成交（报告中披露为局限）。"""
    rs = rows()
    for r in rs:
        if (r["ts_code"], r["trade_date"]) == ("BB2405.SHF", D[1]):
            r.update(open=55.0, high=55.0, low=55.0, close=55.0, settle=55.0)
    res = run(rs, [1, 1, 1, 0])
    assert not res["trades"][res["trades"].date == pd.Timestamp(D[1])].empty


def test_thin_volume_is_flagged():
    res = run(rows(vol=100.0), [1, 1, 0])
    assert (res["events"].kind == "liquidity").any()


def test_costs_change_net_not_gross():
    cheap = run(rows(), [1, 1, 0])
    dear = run(rows(), [1, 1, 0], ex={**EX, "slippage_ticks": 3, "broker_fee_multiplier": 2.0})
    assert np.isclose(cheap["daily"].gross.sum(), dear["daily"].gross.sum())
    assert dear["daily"].net.sum() < cheap["daily"].net.sum()


def test_missing_spec_names_zeus_tool_and_config_key():
    sp = fx.ContractSpecs(basic=[], assumptions={"AA": SPEC})
    with pytest.raises(fx.MissingData, match=r"BB2405.SHF.*fut_basic.*assumptions.BB.multiplier"):
        run(rows(), [1, 1, 0], sp=sp)


def test_fut_basic_sets_multiplier_and_tick_while_fees_and_margin_come_from_config():
    basic = [{"ts_code": "AA2405.SHF", "multiplier": 5, "price_tick": 2}]
    sp = fx.ContractSpecs(basic=basic, assumptions={"AA": SPEC, "BB": SPEC})
    s = sp.spec("AA2405.SHF", pd.Timestamp(D[4]))
    assert (s["multiplier"], s["price_tick"]) == (5.0, 2.0) and s["source"]["multiplier"] == "fut_basic"
    assert s["fee_rate"] == 0.0001 and s["margin_long"] == s["margin_short"] == 0.1
    assert s["source"]["fee_rate"] == "config"
    assert sp.spec("BB2405.SHF", pd.Timestamp(D[4]))["source"]["multiplier"] == "config"


def test_direct_reversal_records_entry_and_round_trip():
    res = run(rows(), [1, -1, -1, 0])
    assert len(res["entries"]) == 2
    assert res["metrics"]["all"]["round_trips"] == 2


def test_zero_volume_day_defers_both_legs_even_with_a_quoted_settle():
    rs = rows()
    for r in rs:
        if (r["ts_code"], r["trade_date"]) == ("BB2405.SHF", D[1]):
            r.update(open=None, high=None, low=None, vol=0.0)        # 无成交日：交易所只给收盘/结算
    res = run(rs, [1, 1, 1, 0], ex={**EX, "exec_price": "settle"})
    tr, ev = res["trades"], res["events"]
    assert tr[tr.date == pd.Timestamp(D[1])].empty
    assert "no_trade" in ev[(ev.kind == "deferred") & (ev.date == pd.Timestamp(D[1]))].detail.iloc[0]
