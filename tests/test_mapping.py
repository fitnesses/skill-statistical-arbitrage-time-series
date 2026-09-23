"""合约映射与换月：按时点安全（只用前一交易日信息）、可复现、研究序列与可执行价格分离。"""

import numpy as np
import pandas as pd
import pytest

import futures as fx
import zeus_fixtures as zf

CAL = [{"name": "near", "product": "RB", "exchange": "SHF", "select": "dominant"},
       {"name": "far", "product": "RB", "exchange": "SHF", "select": "second"}]
CROSS = [{"name": "HC", "product": "HC", "exchange": "SHF", "select": "dominant"},
         {"name": "RB", "product": "RB", "exchange": "SHF", "select": "dominant"}]


@pytest.fixture(scope="module")
def tabs():
    return fx.bar_tables(zf.market()["fut_daily"])


def test_parse_code():
    assert fx.parse_code("RB2501.SHF") == ("RB", (2025, 1), "SHF")
    assert fx.parse_code("RB.SHF") == ("RB", None, "SHF")
    with pytest.raises(ValueError, match="RB501.SHF"):
        fx.parse_code("RB501.SHF")   # 3 位年月有歧义，不猜


def test_rolls_are_decided_on_previous_day_open_interest(tabs):
    mapping, rolls = fx.build_mapping(tabs, CAL)
    oi = tabs["oi"]
    dates = list(oi.index)
    crossovers = rolls[rolls.reason == "oi_crossover"]
    assert len(crossovers) >= 3
    for r in crossovers.itertuples():
        i = dates.index(r.date)
        assert r.decided_on == dates[i - 1]
        assert oi.at[r.decided_on, r.to_code] > oi.at[r.decided_on, r.from_code]
        # 同一天(当日)的持仓量不参与决策：前一天之前尚未交叉
        assert not (oi.at[dates[i - 2], r.to_code] > oi.at[dates[i - 2], r.from_code]) or r.leg == "far"


def test_mapping_prefix_is_invariant_to_future_data(tabs):
    full, full_rolls = fx.build_mapping(tabs, CAL)
    for cut in (120, 260, 400):
        t = tabs["oi"].index[cut]
        part, _ = fx.build_mapping({k: v.loc[:t] for k, v in tabs.items()}, CAL)
        pd.testing.assert_frame_equal(part, full.loc[:t])


def test_calendar_far_leg_delivers_after_near_and_never_in_delivery_window(tabs):
    mapping, rolls = fx.build_mapping(tabs, CAL, exclude_months=1)
    for d, row in mapping.dropna().iterrows():
        near, far = fx.parse_code(row["near"])[1], fx.parse_code(row["far"])[1]
        assert far > near
        for c in row:
            assert fx.months_to_delivery(c, d) > 1
    assert (rolls.to_delivery > rolls.from_delivery).all()        # 只向更远月份换
    assert set(rolls.columns) >= {"date", "decided_on", "leg", "from_code", "to_code", "reason",
                                  "oi_from", "oi_to"}


def test_research_series_has_no_roll_jump_and_is_distinct_from_executable(tabs):
    mapping, rolls = fx.build_mapping(tabs, CROSS)
    research = fx.research_prices(tabs, mapping)
    close = tabs["close"]
    dates = list(close.index)
    for r in rolls.itertuples():
        i = dates.index(r.date)
        prev = dates[i - 1]
        got = research.at[r.date, r.leg] - research.at[prev, r.leg]
        own = np.log(close.at[r.date, r.to_code]) - np.log(close.at[prev, r.to_code])
        assert np.isclose(got, own)        # 换月日收益 = 新合约自身收益，无拼接跳空
    raw = fx.executable_prices(tabs, mapping, "close")
    assert not np.allclose(np.exp(research["RB"].dropna()), raw["RB"].loc[research["RB"].dropna().index])


def test_fixed_leg_never_rolls(tabs):
    legs = [{"name": "a", "ts_code": "HC2205.SHF"}, {"name": "b", "ts_code": "RB2205.SHF"}]
    mapping, rolls = fx.build_mapping(tabs, legs)
    assert rolls.empty
    assert set(mapping["a"].dropna()) == {"HC2205.SHF"}


def test_missing_bar_on_held_contract_does_not_roll_backward(tabs):
    mapping, rolls = fx.build_mapping(tabs, CROSS)
    r = rolls[rolls.leg == "RB"].iloc[1]                  # 第二次换月之后持有 r.to_code
    held, dates = r.to_code, list(tabs["oi"].index)
    i = dates.index(r.date) + 10
    gap = {k: v.copy() for k, v in tabs.items()}
    for k in gap:
        gap[k].loc[dates[i], held] = np.nan                   # 持有合约缺一天行情
    m2, r2 = fx.build_mapping(gap, CROSS)
    assert len(r2) == len(rolls)                              # 不因缺行情多出往返换月
    assert (r2.to_delivery > r2.from_delivery).all()


def test_research_series_bridges_a_missing_bar_within_the_same_contract(tabs):
    mapping, rolls = fx.build_mapping(tabs, CROSS)
    dates = list(tabs["close"].index)
    d = dates[200]
    held = mapping.at[d, "RB"]
    gap = {k: v.copy() for k, v in tabs.items()}
    gap["close"].loc[dates[199], held] = np.nan
    full = fx.research_prices(tabs, mapping)
    bridged = fx.research_prices(gap, mapping)
    assert np.isnan(bridged.at[dates[199], "RB"])
    assert np.isclose(bridged.at[d, "RB"], full.at[d, "RB"])  # t−2→t 收益被接上，没有丢
