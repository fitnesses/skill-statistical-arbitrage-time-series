"""测试公共设置：脚本与测试夹具目录入 sys.path；本机代理不拦截 127.0.0.1（假 zeus 服务）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]


@pytest.fixture(autouse=True)
def no_proxy_for_localhost(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")   # 本机代理会把 127.0.0.1 转成 502
