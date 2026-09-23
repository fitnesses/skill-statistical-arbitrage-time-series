"""Fetch seam: zeus MCP (Streamable HTTP) -> snapshot JSON -> replay. Uses the shared local fake server."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
import run_statarb as rs
import zeus_fixtures as zf

ROWS = [r for c in json.loads((FIX / "mcp_snapshot_hc_rb.json").read_text(encoding="utf-8"))["calls"]
        for r in c["rows"]]
BASE = json.loads((FIX / "replay_config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def zeus_url():
    url, srv = zf.serve({"fut_daily": ROWS}, tools=("fut_daily",))
    yield url
    srv.shutdown()



@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({**BASE, "snapshot": "snap.json", "start_date": "20230103", "end_date": "20240325"},
                            ensure_ascii=False), encoding="utf-8")
    return p


def edit(path, **kw):
    d = json.loads(path.read_text(encoding="utf-8")); d.update(kw)
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def test_fetch_writes_verbatim_snapshot_then_replays(cfg, zeus_url):
    snap = rs.fetch_snapshot(cfg, zeus_url, zf.TOKEN)
    on_disk = json.loads((cfg.parent / "snap.json").read_text(encoding="utf-8"))
    assert on_disk == snap
    assert snap["server"] == "ZeusMCP" and snap["server_version"] == "1.0.0"
    assert snap["tools_available"] == ["fut_daily"]
    assert {m["tool"] for m in snap["missing_capabilities"]} == {"fut_basic", "fut_settle", "ft_limit"}
    assert [c["params"] for c in snap["calls"]] == [
        {"ts_code": leg, "start_date": "20230103", "end_date": "20240325"}
        for leg in ("HC2405.SHF", "RB2405.SHF")]
    assert snap["calls"][0]["rows"] == [r for r in ROWS if r["ts_code"] == "HC2405.SHF"]
    assert all(c["retrieved_at"].endswith("Z") for c in snap["calls"])
    assert zf.TOKEN not in json.dumps(snap)

    man = rs.replay(cfg, cfg.parent / "out")
    assert man["snapshot"]["server_version"] == "1.0.0"
    assert man["data_window"]["n_research_days"] == 319


def test_fetch_never_overwrites_a_snapshot(cfg, zeus_url):
    (cfg.parent / "snap.json").write_text("{}", encoding="utf-8")
    with pytest.raises(rs.ReplayInputError, match="already exists"):
        rs.fetch_snapshot(cfg, zeus_url, zf.TOKEN)
    assert (cfg.parent / "snap.json").read_text(encoding="utf-8") == "{}"


def test_tool_error_names_the_leg(cfg, zeus_url):
    edit(cfg, legs=["HC2405.SHF", "BAD.SHF"])
    with pytest.raises(rs.ZeusError, match="BAD.SHF.*boom"):
        rs.fetch_snapshot(cfg, zeus_url, zf.TOKEN)
    assert not (cfg.parent / "snap.json").exists()


def test_empty_leg_is_reported_not_written(cfg, zeus_url):
    edit(cfg, legs=["HC2405.SHF", "I2405.DCE"])
    with pytest.raises(rs.ZeusError, match="I2405.DCE.*no rows"):
        rs.fetch_snapshot(cfg, zeus_url, zf.TOKEN)
    assert not (cfg.parent / "snap.json").exists()


def test_bad_token_is_reported(cfg, zeus_url):
    with pytest.raises(rs.ZeusError, match="401"):
        rs.fetch_snapshot(cfg, zeus_url, "wrong")


def test_fetch_requires_date_window(cfg, zeus_url):
    edit(cfg, start_date="2023-01-03")
    with pytest.raises(rs.ReplayInputError, match="config: 'start_date' must be YYYYMMDD"):
        rs.fetch_snapshot(cfg, zeus_url, zf.TOKEN)
