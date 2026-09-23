"""合成数据自测：每种模式必须演示它所对应的裁决分支（文档里承诺的行为）。"""

import pytest

import run_statarb as rs


@pytest.mark.parametrize("mode, signal", [
    ("strong", "未触发高/中规则"),
    ("nocoint", "不协整"),
    ("inversion", "样本外优势缺乏样本内支撑"),
    ("leaked", "价差非市场中性"),
    ("drift", "对冲比率结构漂移"),
])
def test_synthetic_mode_demonstrates_its_verdict(tmp_path, mode, signal):
    px = rs._synthetic_pair(mode=mode)
    leg = dict(rs.SYNTH_LEG)
    s = rs.analyze(px, "A", "B", "synthetic", rs.load_market(px), tmp_path / "r.md", {"A": leg, "B": leg})
    flags = rs.robustness_flags(s["adf_tr"][1], s["kpss_tr"][1], s["hl_tr"], s["bt"], s["betastab"],
                                s["attr"], s["attr_sp"])
    assert signal in [f[1] for f in flags]
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "statsmodels" in text and "Engle-Granger" in text
