"""报告增强：图表（PNG，嵌入 report.md / 自包含 report.html）、首页关键数字、自动对账、敏感性分析、滚动稳定性。"""
import json

import numpy as np
import pandas as pd
import pytest

import run_statarb as rs
import zeus_fixtures as zf

START, END = zf.window_of()
COSTS = {p: {"fee_rate": 0.0001, "fee_per_lot": 0, "margin_rate": 0.1} for p in ("RB", "HC")}
CROSS = {"legs": [{"name": "HC", "product": "HC", "exchange": "SHF", "select": "dominant"},
                  {"name": "RB", "product": "RB", "exchange": "SHF", "select": "dominant"}],
         "rationale": "热卷与螺纹同为钢坯下游成材", "assumptions": COSTS}
FIGS = ("overview", "zscore_signals", "equity_drawdown", "roll_timeline", "research_vs_exec",
        "cost_waterfall", "rolling_stability")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    import os
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    tmp = tmp_path_factory.mktemp("rep")
    url, srv = zf.serve(zf.market())
    cfg = tmp / "config.json"
    cfg.write_text(json.dumps({"snapshot": "snapshot.json", "start_date": START, "end_date": END, **CROSS},
                              ensure_ascii=False), encoding="utf-8")
    rs.fetch_snapshot(cfg, url, zf.TOKEN)
    srv.shutdown()
    man = rs.replay(cfg, tmp / "out")
    return cfg, tmp / "out", man


def test_figures_are_rendered_and_embedded_with_reading_notes(run):
    _, out, man = run
    text = (out / "report.md").read_text(encoding="utf-8")
    for f in FIGS:
        png = out / "figures" / f"{f}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f
        assert f"](figures/{f}.png)" in text, f
        assert man["figures"][f"figures/{f}.png"]
    assert text.count("看图要点") >= len(FIGS) - 1           # 总览图之外每张都有看图要点


def test_overview_key_numbers_come_before_chapter_one(run):
    _, out, _ = run
    text = (out / "report.md").read_text(encoding="utf-8")
    head = text[:text.index("## 1.")]
    for s in ("统计证据", "交易可行性", "样本外净 PnL", "最大回撤", "figures/overview.png", "自动对账", "敏感性"):
        assert s in head, s


def test_html_report_is_self_contained(run):
    _, out, man = run
    html = (out / "report.html").read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == len(FIGS)
    assert 'src="figures/' not in html and "<table" in html and "不构成任何投资建议" in html
    assert man["figures"]["report.html"]


def test_reconciliation_passes_and_detects_tampering(run):
    cfg, out, man = run
    assert man["checks"] and all(c["ok"] for c in man["checks"]), man["checks"]
    names = {c["check"] for c in man["checks"]}
    assert {"净PnL勾稽", "区间净PnL与报告一致", "手续费复算", "滑点复算", "毛PnL复算", "换月决策日",
            "训练窗β复算", "训练窗EG复算", "训练窗半衰期/ADF/KPSS复算"} <= names
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "## 12. 自动对账" in text and "全部通过" in text
    trades = pd.read_csv(out / "trades.csv")
    trades.loc[0, "fee"] += 1.0
    trades.to_csv(out / "trades.csv", index=False)
    bad = {c["check"]: c["ok"] for c in rs.reconcile_run(cfg, out)}
    assert bad["手续费复算"] is False


def test_sensitivity_table_covers_the_planned_variants(run):
    _, out, man = run
    names = [r["variant"] for r in man["sensitivity"]]
    assert names[0] == "基准" and len(names) >= 7
    for n in ("手续费×2", "滑点+1跳", "训练窗60%", "训练窗80%", "A腿20手", "交割前2个月换出"):
        assert n in names, n
    for r in man["sensitivity"]:
        assert {"stat_ok", "exe_ok", "eg_p", "oos_net", "oos_sign", "oos_t", "round_trips", "same_as_base"} <= set(r)
        assert r["same_as_base"] == ((r["stat_ok"], r["exe_ok"], r["oos_sign"]) ==
                                     (names and man["sensitivity"][0]["stat_ok"], man["sensitivity"][0]["exe_ok"],
                                      man["sensitivity"][0]["oos_sign"]))
    assert "## 11. 敏感性分析" in (out / "report.md").read_text(encoding="utf-8")


def test_rolling_stability_is_computed_on_trailing_windows():
    idx = pd.bdate_range("2021-01-04", periods=400)
    rng = np.random.default_rng(1)
    b = np.cumsum(rng.normal(0, 0.01, 400))
    sp = np.zeros(400)
    for t in range(1, 400):
        sp[t] = 0.8 * sp[t - 1] + rng.normal(0, 0.005)
    research = pd.DataFrame({"a": 1.2 * b + sp, "b": b}, index=idx)
    roll = rs.rolling_stability(research, window=120, step=5)
    assert list(roll.columns) == ["beta", "half_life", "adf_p"]
    assert roll.index[0] == idx[119] and roll.index.is_monotonic_increasing
    assert np.allclose(roll.beta, 1.2, atol=0.15) and (roll.adf_p < 0.05).mean() > 0.8


def test_report_extras_can_be_switched_off(tmp_path):
    url, srv = zf.serve(zf.market())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"snapshot": "snapshot.json", "start_date": START, "end_date": END,
                               "report": {"charts": False, "sensitivity": False}, **CROSS},
                              ensure_ascii=False), encoding="utf-8")
    rs.fetch_snapshot(cfg, url, zf.TOKEN)
    srv.shutdown()
    man = rs.replay(cfg, tmp_path / "out")
    assert not (tmp_path / "out" / "figures").exists() and (tmp_path / "out" / "report.html").exists()
    assert man["sensitivity"] == [] and man["checks"]
    assert "未运行" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_every_markdown_table_renders_as_html_table(run):
    _, out, _ = run
    md = (out / "report.md").read_text(encoding="utf-8")
    n_tables = sum(1 for i, l in enumerate(md.splitlines()) if l.startswith("|---"))
    assert (out / "report.html").read_text(encoding="utf-8").count("<table") == n_tables


def test_cli_verify_reruns_reconciliation(run, monkeypatch, capsys):
    cfg, out, _ = run
    monkeypatch.setattr("sys.argv", ["run_statarb.py", "--config", str(cfg), "--verify", str(out)])
    with pytest.raises(SystemExit) as e:
        rs.main()
    out_text = capsys.readouterr().out          # 同模块的篡改测试可能已改过 trades.csv，这里只验证能跑且逐项输出
    assert "净PnL勾稽" in out_text and "训练窗半衰期/ADF/KPSS复算" in out_text and e.value.code in (0, 1)


def test_fill_markers_are_placed_on_the_signal_day():
    import charts
    idx = pd.bdate_range("2024-01-01", periods=6)
    daily = pd.DataFrame({"spread_pos": [0, 1, 1, 0, -1, 0]}, index=idx)
    long_in, short_in, exits = charts._fills(daily)
    assert list(long_in) == [idx[0]] and list(short_in) == [idx[3]] and list(exits) == [idx[2], idx[4]]


def test_manifest_verdicts_are_real_booleans(run):
    _, out, _ = run
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for r in man["sensitivity"]:
        if "error" not in r and not r.get("skipped"):
            assert isinstance(r["stat_ok"], bool) and isinstance(r["exe_ok"], bool)


def test_table_cells_escape_pipes(tmp_path):
    """单元格里的 |t|、|z| 等竖线必须转义，否则 Markdown/HTML 表格会错列丢数据。"""
    import re as _re
    flags = [("🟡 中", "Sharpe不显著", "样本外Sharpe的|t|<1.96", "t≈1.2 |z|≥2")]
    s = rs.run_stats(np.exp(pd.DataFrame({"A": np.cumsum(np.random.default_rng(0).normal(0, .01, 300)),
                                          "B": np.cumsum(np.random.default_rng(1).normal(0, .01, 300))},
                                         index=pd.bdate_range("2021-01-04", periods=300))),
                     {k: dict(rs.SYNTH_LEG, tick_size=0.01) for k in "AB"})
    px = np.exp(s["bt"]["spread"].to_frame("A").assign(B=1.0))
    rs.write_report("A", "B", "t", px, s, flags, tmp_path / "r.md")
    for line in (tmp_path / "r.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| 🟡 中 | Sharpe不显著"):
            assert len(_re.findall(r"(?<!\\)\|", line)) == 5, line      # 4 列 → 5 个未转义分隔符


def test_inapplicable_sensitivity_variants_are_not_counted_as_consistent(tmp_path):
    url, srv = zf.serve(zf.market())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"snapshot": "snapshot.json", "start_date": START, "end_date": END,
                               "legs": ["HC2205.SHF", "RB2205.SHF"], "rationale": "钢材链", "assumptions": COSTS,
                               "train_frac": 0.6, "execution": {"base_lots": 20},
                               "report": {"charts": True, "sensitivity": True}}, ensure_ascii=False), encoding="utf-8")
    rs.fetch_snapshot(cfg, url, zf.TOKEN)
    srv.shutdown()
    man = rs.replay(cfg, tmp_path / "out")
    by = {r["variant"]: r for r in man["sensitivity"]}
    for name in ("训练窗60%", "A腿20手", "交割前2个月换出"):         # 与基准相同或对固定合约不起作用
        assert by[name].get("skipped"), name
    assert "训练窗80%" in by and not by["训练窗80%"].get("skipped")
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "不适用" in text and (tmp_path / "out" / "figures" / "roll_timeline.png").exists()


def test_calendar_spread_renders_all_figures(tmp_path):
    url, srv = zf.serve(zf.market())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"snapshot": "snapshot.json", "start_date": START, "end_date": END,
                               "legs": [{"name": "近月", "product": "RB", "exchange": "SHF", "select": "dominant"},
                                        {"name": "远月", "product": "RB", "exchange": "SHF", "select": "second"}],
                               "assumptions": COSTS, "report": {"charts": True, "sensitivity": False}},
                              ensure_ascii=False), encoding="utf-8")
    rs.fetch_snapshot(cfg, url, zf.TOKEN)
    srv.shutdown()
    man = rs.replay(cfg, tmp_path / "out")
    assert len([f for f in man["figures"] if f.endswith(".png")]) == len(FIGS)
