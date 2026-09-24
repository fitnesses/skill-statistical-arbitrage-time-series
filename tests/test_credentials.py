"""zeus 连接信息的查找顺序：进程环境变量 → 工作目录 .env → skill 目录 .env → Claude Code 的 zeus MCP 配置。"""
import json

import pytest

import run_statarb as rs


def resolve(tmp_path, environ=None, **kw):
    dirs = {k: tmp_path / k for k in ("cwd", "skill", "home")}
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    return rs.resolve_zeus(cwd=dirs["cwd"], skill_dir=dirs["skill"], home=dirs["home"],
                           environ=environ or {}, **kw), dirs


def test_dotenv_in_workspace(tmp_path):
    (tmp_path / "cwd").mkdir()
    (tmp_path / "cwd" / ".env").write_text(
        '# zeus\nexport ZEUS_MCP_URL="http://h:8000/mcp"\nZEUS_MCP_TOKEN=\'abc\'  # 注释\nOTHER=1\n', encoding="utf-8")
    (url, headers, source), dirs = resolve(tmp_path)
    assert url == "http://h:8000/mcp" and headers == {"Authorization": "Bearer abc"}
    assert source.endswith(".env") and "abc" not in source


def test_process_env_wins_and_skill_dir_env_is_fallback(tmp_path):
    (tmp_path / "skill").mkdir()
    (tmp_path / "skill" / ".env").write_text("ZEUS_MCP_URL=http://skill/mcp\nZEUS_MCP_TOKEN=s\n", encoding="utf-8")
    (url, headers, _), _ = resolve(tmp_path)
    assert url == "http://skill/mcp" and headers["Authorization"] == "Bearer s"
    (url, headers, source), _ = resolve(tmp_path, environ={"ZEUS_MCP_URL": "http://env/mcp"})
    assert url == "http://env/mcp" and headers == {} and source == "环境变量"


def test_falls_back_to_claude_code_mcp_config(tmp_path):
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / ".claude.json").write_text(json.dumps({"mcpServers": {"zeus": {
        "type": "http", "url": "http://cc/mcp", "headers": {"Authorization": "Bearer cc-token"}}}}), encoding="utf-8")
    (url, headers, source), _ = resolve(tmp_path)
    assert url == "http://cc/mcp" and headers == {"Authorization": "Bearer cc-token"}
    assert ".claude.json" in source and "cc-token" not in source


def test_project_mcp_json_and_project_scoped_claude_config(tmp_path):
    (tmp_path / "cwd").mkdir()
    (tmp_path / "home").mkdir()
    cwd = tmp_path / "cwd"
    (tmp_path / "home" / ".claude.json").write_text(json.dumps({"projects": {str(cwd): {"mcpServers": {"zeus": {
        "url": "http://proj/mcp", "headers": {"Authorization": "Bearer p"}}}}}}), encoding="utf-8")
    (url, _, _), _ = resolve(tmp_path)
    assert url == "http://proj/mcp"
    (cwd / ".mcp.json").write_text(json.dumps({"mcpServers": {"zeus": {"url": "http://dot/mcp"}}}), encoding="utf-8")
    (url, _, source), _ = resolve(tmp_path)
    assert url == "http://dot/mcp" and ".mcp.json" in source


def test_nothing_found_tells_user_to_create_dotenv(tmp_path):
    with pytest.raises(rs.ZeusError, match=r"\.env\.example"):
        resolve(tmp_path)


def test_zeus_client_sends_configured_headers(monkeypatch):
    import zeus_fixtures as zf
    url, srv = zf.serve(zf.market())
    try:
        assert rs.ZeusClient(url, headers={"Authorization": f"Bearer {zf.TOKEN}"}).list_tools()
    finally:
        srv.shutdown()
