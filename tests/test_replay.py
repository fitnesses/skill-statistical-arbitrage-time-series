"""Replay seam: config + fixed MCP snapshot -> report + manifest, no live data."""
import hashlib, json, shutil, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
import run_statarb as rs  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Copy fixtures into tmp so tests can mutate them; block live data libs."""
    for name in ("akshare", "yfinance"):
        monkeypatch.setitem(sys.modules, name, None)   # any import -> ImportError
    for f in FIX.iterdir():
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


def run(case, out="out"):
    return rs.replay(case / "replay_config.json", case / out)


def edit_json(path, fn):
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d), encoding="utf-8")


def test_replay_writes_report_and_auditable_manifest(case):
    m = run(case)
    out = case / "out"
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man == m

    snap = man["snapshot"]
    assert snap["sha256"] == sha256(case / "mcp_snapshot_hc_rb.json")
    assert snap["server"] == "zeus" and "server_version" in snap
    assert "synthetic" in snap["provenance"]
    assert [c["tool"] for c in snap["calls"]] == ["fut_daily", "fut_daily"]
    assert snap["calls"][0]["params"]["ts_code"] == "HC2405.SHF"
    assert snap["calls"][0]["retrieved_at"] and snap["calls"][0]["n_rows"] == 320

    assert man["config"]["legs"] == ["HC2405.SHF", "RB2405.SHF"]
    assert man["config_sha256"] == sha256(case / "replay_config.json")
    assert man["code"]["script_sha256"] == sha256(ROOT / "scripts" / "run_statarb.py")
    assert man["run_id"] and man["created_at"]
    assert snap["path"] == "mcp_snapshot_hc_rb.json" and snap["snapshot_version"] == 1
    assert man["costs_used"] == {"commission_bps": 1.0, "stamp_duty_bps": 0.0, "borrow_annual_bps": 0.0}
    assert set(man["code"]["libs"]) >= {"python", "numpy", "pandas"}
    assert man["data_window"]["rows_per_leg"] == {"HC2405.SHF": 320, "RB2405.SHF": 320}

    report = out / "report.md"
    assert man["outputs"]["report.md"] == sha256(report)
    text = report.read_text(encoding="utf-8")
    assert "HC2405.SHF" in text and "`mcp`" in text


def test_replay_is_deterministic(case):
    a, b = run(case, "o1"), run(case, "o2")
    assert a["run_id"] == b["run_id"]
    assert a["outputs"] == b["outputs"]


def test_changed_snapshot_changes_integrity_and_run_id(case):
    before = run(case, "o1")
    edit_json(case / "mcp_snapshot_hc_rb.json",
              lambda d: d["calls"][0]["rows"][5].update(close=9999))
    after = run(case, "o2")
    assert before["snapshot"]["sha256"] != after["snapshot"]["sha256"]
    assert before["run_id"] != after["run_id"]


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["calls"][1]["rows"][3].pop("close"), "calls[1].rows[3]: missing 'close'"),
    (lambda d: d["calls"][0]["rows"][0].update(trade_date="2023-01-03"),
     "calls[0].rows[0]: trade_date '2023-01-03' is not YYYYMMDD"),
    (lambda d: d["calls"][0].pop("retrieved_at"), "calls[0]: missing 'retrieved_at'"),
    (lambda d: d.pop("calls"), "snapshot: missing 'calls'"),
    (lambda d: d["calls"][0]["rows"][2].update(close=True), "calls[0].rows[2]: 'close' is not numeric"),
    (lambda d: d["calls"][0]["rows"][2].update(close=float("nan")), "calls[0].rows[2]: 'close' is not numeric"),
    (lambda d: d["calls"][0]["rows"].append(dict(d["calls"][0]["rows"][4])),
     "calls[0].rows[320]: duplicate HC2405.SHF 20230109"),
])
def test_malformed_snapshot_fails_with_location(case, mutate, needle):
    edit_json(case / "mcp_snapshot_hc_rb.json", mutate)
    with pytest.raises(rs.ReplayInputError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        run(case)


@pytest.mark.parametrize("patch, needle", [
    ({"window": "20"}, "config: 'window' must be an integer"),
    ({"costs": {"commission_bps": "1"}}, "config.costs: 'commission_bps' must be a number"),
    ({"costs": {"fee": 1}}, "config.costs: unknown keys ['fee']"),
    ({"price_field": 3}, "config: 'price_field' must be a string"),
])
def test_malformed_config_fails_with_location(case, patch, needle):
    edit_json(case / "replay_config.json", lambda d: d.update(patch))
    with pytest.raises(rs.ReplayInputError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        run(case)


def test_changed_config_changes_run_id(case):
    before = run(case, "o1")
    edit_json(case / "replay_config.json", lambda d: d["costs"].update(commission_bps=3))
    after = run(case, "o2")
    assert before["config_sha256"] != after["config_sha256"]
    assert before["run_id"] != after["run_id"]
    assert after["costs_used"]["commission_bps"] == 3


def test_leg_absent_from_snapshot_is_named(case):
    edit_json(case / "replay_config.json", lambda d: d.update(legs=["HC2405.SHF", "I2405.DCE"]))
    with pytest.raises(rs.ReplayInputError, match="I2405.DCE"):
        run(case)


def test_missing_config_key_is_named(case):
    edit_json(case / "replay_config.json", lambda d: d.pop("legs"))
    with pytest.raises(rs.ReplayInputError, match="config: missing 'legs'"):
        run(case)
