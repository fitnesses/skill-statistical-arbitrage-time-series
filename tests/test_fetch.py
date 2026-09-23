"""Fetch seam: zeus MCP (Streamable HTTP) -> snapshot JSON -> replay. Uses a local fake server."""
import json, shutil, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
import run_statarb as rs  # noqa: E402

ROWS = [r for c in json.loads((FIX / "mcp_snapshot_hc_rb.json").read_text(encoding="utf-8"))["calls"]
        for r in c["rows"]]
TOKEN = "t0k"


class FakeZeus(BaseHTTPRequestHandler):
    """Mimics FastMCP streamable HTTP: SSE replies, Mcp-Session-Id, one text item per row."""
    def log_message(self, *a):
        pass

    def _sse(self, obj, headers=()):
        body = f"event: message\ndata: {json.dumps(obj)}\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_response(401); self.end_headers(); return
        if msg["method"] == "initialize":
            return self._sse({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2025-03-26", "capabilities": {},
                "serverInfo": {"name": "ZeusMCP", "version": "1.0.0"}}},
                headers=[("Mcp-Session-Id", "s1")])
        assert self.headers.get("Mcp-Session-Id") == "s1"
        if msg["method"] == "notifications/initialized":
            self.send_response(202); self.end_headers(); return
        args = msg["params"]["arguments"]
        if args["ts_code"] == "BAD.SHF":
            result = {"content": [{"type": "text", "text": "boom"}], "isError": True}
        else:
            rows = [r for r in ROWS if r["ts_code"] == args["ts_code"]
                    and args["start_date"] <= r["trade_date"] <= args["end_date"]]
            result = {"content": [{"type": "text", "text": json.dumps(r)} for r in rows],
                      "isError": False}
        self._sse({"jsonrpc": "2.0", "id": msg["id"], "result": result})


@pytest.fixture(scope="module")
def zeus_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeZeus)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/mcp"
    srv.shutdown()


@pytest.fixture(autouse=True)
def no_proxy_for_localhost(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")   # 本机代理会把 127.0.0.1 转成 502


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"snapshot": "snap.json", "legs": ["HC2405.SHF", "RB2405.SHF"],
                             "start_date": "20230103", "end_date": "20240325",
                             "costs": {"commission_bps": 1.0, "stamp_duty_bps": 0.0,
                                       "borrow_annual_bps": 0.0}}), encoding="utf-8")
    return p


def edit(path, **kw):
    d = json.loads(path.read_text(encoding="utf-8")); d.update(kw)
    path.write_text(json.dumps(d), encoding="utf-8")


def test_fetch_writes_verbatim_snapshot_then_replays(cfg, zeus_url):
    snap = rs.fetch_snapshot(cfg, zeus_url, TOKEN)
    on_disk = json.loads((cfg.parent / "snap.json").read_text(encoding="utf-8"))
    assert on_disk == snap
    assert snap["server"] == "ZeusMCP" and snap["server_version"] == "1.0.0"
    assert [c["params"] for c in snap["calls"]] == [
        {"ts_code": leg, "start_date": "20230103", "end_date": "20240325"}
        for leg in ("HC2405.SHF", "RB2405.SHF")]
    assert snap["calls"][0]["rows"] == [r for r in ROWS if r["ts_code"] == "HC2405.SHF"]
    assert all(c["retrieved_at"].endswith("Z") for c in snap["calls"])
    assert TOKEN not in json.dumps(snap)

    man = rs.replay(cfg, cfg.parent / "out")
    assert man["snapshot"]["server_version"] == "1.0.0"
    assert man["data_window"]["n_aligned"] == 320


def test_fetch_never_overwrites_a_snapshot(cfg, zeus_url):
    (cfg.parent / "snap.json").write_text("{}", encoding="utf-8")
    with pytest.raises(rs.ReplayInputError, match="already exists"):
        rs.fetch_snapshot(cfg, zeus_url, TOKEN)
    assert (cfg.parent / "snap.json").read_text(encoding="utf-8") == "{}"


def test_tool_error_names_the_leg(cfg, zeus_url):
    edit(cfg, legs=["HC2405.SHF", "BAD.SHF"])
    with pytest.raises(rs.ZeusError, match="BAD.SHF.*boom"):
        rs.fetch_snapshot(cfg, zeus_url, TOKEN)
    assert not (cfg.parent / "snap.json").exists()


def test_empty_leg_is_reported_not_written(cfg, zeus_url):
    edit(cfg, legs=["HC2405.SHF", "I2405.DCE"])
    with pytest.raises(rs.ZeusError, match="I2405.DCE.*no rows"):
        rs.fetch_snapshot(cfg, zeus_url, TOKEN)
    assert not (cfg.parent / "snap.json").exists()


def test_bad_token_is_reported(cfg, zeus_url):
    with pytest.raises(rs.ZeusError, match="401"):
        rs.fetch_snapshot(cfg, zeus_url, "wrong")


def test_fetch_requires_date_window(cfg, zeus_url):
    edit(cfg, start_date="2023-01-03")
    with pytest.raises(rs.ReplayInputError, match="config: 'start_date' must be YYYYMMDD"):
        rs.fetch_snapshot(cfg, zeus_url, TOKEN)
