"""完整研究运行：研究配置 → zeus 取数(假服务) → 快照 → 回放 → 报告与可审计产物。覆盖跨期与跨品种。"""
import hashlib, json
from pathlib import Path

import pandas as pd
import pytest

import run_statarb as rs
import zeus_fixtures as zf

START, END = zf.window_of()
COSTS = {p: {"fee_rate": 0.0001, "fee_per_lot": 0, "margin_rate": 0.1} for p in ("RB", "HC")}
ASSUME = {p: {"multiplier": 10, "price_tick": 1, **COSTS[p]} for p in ("RB", "HC")}
CAL = {"legs": [{"name": "near", "product": "RB", "exchange": "SHF", "select": "dominant"},
                {"name": "far", "product": "RB", "exchange": "SHF", "select": "second"}]}
CROSS = {"legs": [{"name": "HC", "product": "HC", "exchange": "SHF", "select": "dominant"},
                  {"name": "RB", "product": "RB", "exchange": "SHF", "select": "dominant"}],
         "rationale": "热卷与螺纹同为钢坯下游成材，成本端共享铁矿与焦炭，价差反映板材/建材需求相对强弱",
         "screening": {"n_candidates": 1, "procedure": "预设单一配对"}}
ARTIFACTS = ("report.md", "manifest.json", "mapping.csv", "rolls.csv", "trades.csv", "daily.csv", "events.csv")


@pytest.fixture(scope="module")
def market():
    return zf.market()



@pytest.fixture(scope="module")
def full_zeus(market):
    url, srv = zf.serve(market)
    yield url
    srv.shutdown()


@pytest.fixture(scope="module")
def daily_only_zeus(market):
    url, srv = zf.serve(market, tools=("fut_daily",))
    yield url
    srv.shutdown()


def write_cfg(tmp, extra, **kw):
    cfg = {"snapshot": "snapshot.json", "start_date": START, "end_date": END, "assumptions": COSTS,
           "report": {"charts": False, "sensitivity": False}, **extra, **kw}
    p = tmp / "config.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


def fetch_and_run(tmp, url, extra, **kw):
    cfg = write_cfg(tmp, extra, **kw)
    rs.fetch_snapshot(cfg, url, zf.TOKEN)
    return cfg, rs.replay(cfg, tmp / "out")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_calendar_spread_end_to_end(tmp_path, full_zeus):
    cfg, man = fetch_and_run(tmp_path, full_zeus, CAL)
    out = tmp_path / "out"
    for f in ARTIFACTS:
        assert (out / f).exists(), f
        if f != "manifest.json":
            assert man["outputs"][f] == sha(out / f)
    assert man["family"] == "calendar"
    mapping = pd.read_csv(out / "mapping.csv")
    assert {"date", "near", "far", "near_research_logp", "near_exec_close"} <= set(mapping.columns)
    rolls = pd.read_csv(out / "rolls.csv")
    assert len(rolls) >= 3 and set(rolls.leg) == {"near", "far"}
    trades = pd.read_csv(out / "trades.csv")
    assert len(trades) and (trades.lots == trades.lots.round()).all()
    assert set(trades.ts_code) <= set(mapping.near) | set(mapping.far)
    assert {"signal_date", "date", "price", "fee", "slippage", "reason"} <= set(trades.columns)
    text = (out / "report.md").read_text(encoding="utf-8")
    for s in ("合约映射与换月", "交易可行性", "统计证据", "未做的分析", man["run_id"], "不构成任何投资建议",
              "研究用连续序列"):
        assert s in text, s
    assert man["snapshot"]["missing_capabilities"] == []
    assert set(man["specs_sources"]["multiplier"]) == {"fut_basic"}
    assert set(man["specs_sources"]["fee_rate"]) == {"config"}
    ex = man["metrics"]["executable"]
    for k in ("gross", "net", "fees", "slippage", "max_drawdown", "round_trips", "margin_max",
              "return_on_margin_ann", "turnover_ann"):
        assert k in ex["oos"], k
    assert "stress_oos" in man["metrics"]


def test_cross_commodity_end_to_end_records_rationale_and_screening(tmp_path, full_zeus):
    cfg, man = fetch_and_run(tmp_path, full_zeus, CROSS)
    assert man["family"] == "cross"
    assert man["screening"] == {"n_candidates": 1, "procedure": "预设单一配对"}
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert CROSS["rationale"] in text and "筛选" in text
    trades = pd.read_csv(tmp_path / "out" / "trades.csv")
    assert {c[:2] for c in trades.ts_code} == {"HC", "RB"}


def test_cross_commodity_requires_rationale(tmp_path):
    cfg = write_cfg(tmp_path, {**CROSS, "rationale": ""})
    with pytest.raises(rs.ReplayInputError, match="rationale"):
        rs.load_config(cfg)


def test_multiple_testing_threshold_is_reported(tmp_path, full_zeus):
    _, man = fetch_and_run(tmp_path, full_zeus, {**CROSS, "screening": {"n_candidates": 20, "procedure": "20 对钢材链"}})
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "0.0025" in text                       # Bonferroni 0.05/20


def test_missing_zeus_capabilities_fall_back_to_disclosed_assumptions(tmp_path, daily_only_zeus):
    cfg, man = fetch_and_run(tmp_path, daily_only_zeus, CROSS, assumptions=ASSUME)
    missing = {m["tool"] for m in man["snapshot"]["missing_capabilities"]}
    assert missing == {"fut_basic"}
    assert set(man["specs_sources"]["multiplier"]) == {"config"}
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "fut_basic" in text and "未考虑涨跌停" in text


def test_missing_capability_without_assumption_names_tool_and_config_key(tmp_path, daily_only_zeus):
    cfg = write_cfg(tmp_path, CROSS)
    rs.fetch_snapshot(cfg, daily_only_zeus, zf.TOKEN)
    with pytest.raises(rs.fx.MissingData, match=r"fut_basic.*assumptions\.HC\.multiplier"):
        rs.replay(cfg, tmp_path / "out")


def test_replay_is_deterministic_and_offline(tmp_path, full_zeus, monkeypatch):
    cfg, first = fetch_and_run(tmp_path, full_zeus, CAL)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net")))
    second = rs.replay(cfg, tmp_path / "out2")
    assert first["run_id"] == second["run_id"] and first["outputs"] == second["outputs"]


def test_test_period_data_cannot_change_fitted_parameters(tmp_path, full_zeus):
    cfg, base = fetch_and_run(tmp_path, full_zeus, CROSS)
    split = base["fitted"]["split_date"].replace("-", "")
    snap_p = tmp_path / "snapshot.json"
    snap = json.loads(snap_p.read_text(encoding="utf-8"))
    for c in snap["calls"]:
        if c["tool"] == "fut_daily":
            for r in c["rows"]:
                if r["trade_date"] >= split and r["ts_code"].startswith("HC"):
                    for k in ("open", "high", "low", "close", "settle"):
                        r[k] = round(r[k] * 1.1)
    snap_p.write_text(json.dumps(snap), encoding="utf-8")
    moved = rs.replay(cfg, tmp_path / "out2")
    assert moved["fitted"] == base["fitted"]
    assert moved["metrics"]["executable"]["oos"]["net"] != base["metrics"]["executable"]["oos"]["net"]


def test_fees_and_margin_must_be_given_per_product(tmp_path):
    with pytest.raises(rs.ReplayInputError, match=r"config.assumptions.RB.*margin_rate"):
        rs.load_config(write_cfg(tmp_path, CAL, assumptions={"RB": {"fee_rate": 0.0001}}))
    with pytest.raises(rs.ReplayInputError, match=r"config.assumptions.HC.*fee_rate.*fee_per_lot"):
        rs.load_config(write_cfg(tmp_path, CROSS, assumptions={"RB": COSTS["RB"], "HC": {"margin_rate": 0.1}}))


def test_check_zeus_reports_required_and_optional_capabilities(full_zeus, daily_only_zeus):
    ok = rs.check_zeus(full_zeus, zf.TOKEN, "RB2205.SHF", START, END)
    assert ok["ok"] and all(v == "ok" for v in ok["optional"].values())
    partial = rs.check_zeus(daily_only_zeus, zf.TOKEN, "RB2205.SHF", START, END)
    assert partial["ok"] and all(v.startswith("absent") for v in partial["optional"].values())


def test_two_legs_resolving_to_the_same_contract_are_rejected(tmp_path):
    legs = [{"name": "x", "product": "RB", "exchange": "SHF", "select": "dominant"},
            {"name": "y", "product": "RB", "exchange": "SHF", "select": "dominant"}]
    with pytest.raises(rs.ReplayInputError, match="同一"):
        rs.load_config(write_cfg(tmp_path, {"legs": legs}))


def test_fixed_legs_still_fetch_contract_specs_from_fut_basic(tmp_path, full_zeus, daily_only_zeus):
    fixed = {"legs": ["HC2205.SHF", "RB2205.SHF"], "rationale": CROSS["rationale"]}
    cfg = write_cfg(tmp_path, fixed)
    snap = rs.fetch_snapshot(cfg, full_zeus, zf.TOKEN)
    assert "fut_basic" in [c["tool"] for c in snap["calls"]]
    cfg2 = write_cfg(tmp_path, fixed, snapshot="s2.json")
    snap2 = rs.fetch_snapshot(cfg2, daily_only_zeus, zf.TOKEN)
    assert "fut_basic" in {m["tool"] for m in snap2["missing_capabilities"]}


def test_start_after_end_is_rejected(tmp_path):
    with pytest.raises(rs.ReplayInputError, match="start_date"):
        rs.load_config(write_cfg(tmp_path, CAL, start_date="20230101", end_date="20220101"))


def test_non_json_tool_text_is_a_zeus_error(market):
    url, srv = zf.serve({**market, "fut_daily": [{"not": "used"}]})
    try:
        c = rs.ZeusClient(url, zf.TOKEN)
        c._rpc = lambda *a, **k: {"content": [{"type": "text", "text": "暂无数据"}]}
        with pytest.raises(rs.ZeusError, match="暂无数据"):
            c.call("fut_daily", {"ts_code": "RB2205.SHF"})
    finally:
        srv.shutdown()
