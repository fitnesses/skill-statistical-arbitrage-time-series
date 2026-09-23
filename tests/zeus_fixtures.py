"""确定性的合约级期货数据生成器 + 本地假 zeus MCP 服务（FastMCP 风格 streamable HTTP）。

生成的数据形状与 zeus fut_daily 一致；fut_basic / fut_settle / ft_limit 按
references/zeus-mcp-interface.md 定义的接口生成，用于验证"zeus 尚未实现"的工具。
"""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

MONTHS = (1, 5, 10)          # 螺纹/热卷式的主力月份
TOKEN = "t0k"


def _dates(n, start="2021-01-04"):
    return pd.bdate_range(start, periods=n)


def spot_paths(n, seed=3):
    """RB 现货对数价随机游走；HC = RB + 常数 + 平稳 OU 价差（跨品种协整）。"""
    rng = np.random.default_rng(seed)
    rb = np.log(4000) + np.cumsum(rng.normal(0, 0.012, n))
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.85 * sp[t - 1] + rng.normal(0, 0.008)
    return {"RB": rb, "HC": rb + 0.05 + sp}


def product_rows(product, exch, log_spot, dates, seed, carry=0.03, tick=1):
    """每个合约：上市于交割前 12 个月，最后交易日=交割月 15 日；价格=现货×期限 carry+噪声；
    持仓量在距交割约 150 天时达到峰值 → 主力自然迁移。"""
    rng = np.random.default_rng(seed)
    rows, basic = [], []
    for y in range(dates[0].year, dates[-1].year + 2):
        for m in MONTHS:
            code = f"{product}{y % 100:02d}{m:02d}.{exch}"
            deliv = pd.Timestamp(y, m, 15)
            listed = deliv - pd.DateOffset(months=12)
            idx = np.where((dates >= listed) & (dates <= deliv))[0]
            if not len(idx):
                continue
            basic.append({"ts_code": code, "fut_code": product, "exchange": exch, "multiplier": 10,
                          "price_tick": tick, "list_date": listed.strftime("%Y%m%d"),
                          "delist_date": deliv.strftime("%Y%m%d"), "d_month": f"{y}{m:02d}"})
            prev_settle = None
            for i in idx:
                d = dates[i]
                dtd = (deliv - d).days
                px = float(np.exp(log_spot[i] + carry * dtd / 365 + rng.normal(0, 0.002)))
                close = round(px)
                settle = round(px * (1 + rng.normal(0, 0.0005)))
                opn = round(px * (1 + rng.normal(0, 0.002)))
                oi = round(1e6 * np.exp(-((dtd - 150) / 75) ** 2)) + 1000
                rows.append({"ts_code": code, "trade_date": d.strftime("%Y%m%d"),
                             "pre_settle": float(prev_settle if prev_settle else settle),
                             "open": float(opn), "high": float(max(opn, close) + 5 * tick),
                             "low": float(min(opn, close) - 5 * tick), "close": float(close),
                             "settle": float(settle), "vol": float(oi // 2), "oi": float(oi)})
                prev_settle = settle
    return rows, basic


def market(n=520, seed=3):
    """{ 'fut_daily': rows, 'fut_basic': rows, 'fut_settle': rows, 'ft_limit': rows }，RB 与 HC 两个品种。"""
    dates = _dates(n)
    spots = spot_paths(n, seed)
    daily, basic = [], []
    for k, prod in enumerate(("RB", "HC")):
        r, b = product_rows(prod, "SHF", spots[prod], dates, seed + 10 + k)
        daily += r
        basic += b
    settle, limit = [], []
    for r in daily:
        # 保证金在 2022-01-01 起由 10% 调到 12%：验证按日生效（effective-dated）
        m = 0.12 if r["trade_date"] >= "20220101" else 0.10
        settle.append({"ts_code": r["ts_code"], "trade_date": r["trade_date"], "settle": r["settle"],
                       "trading_fee_rate": 0.0001, "trading_fee": 0.0, "offset_today_fee": 0.0,
                       "long_margin_rate": m, "short_margin_rate": m, "exchange": "SHF"})
        limit.append({"ts_code": r["ts_code"], "trade_date": r["trade_date"],
                      "pre_settle": r["pre_settle"],
                      "up_limit": round(r["pre_settle"] * 1.07), "down_limit": round(r["pre_settle"] * 0.93)})
    return {"fut_daily": daily, "fut_basic": basic, "fut_settle": settle, "ft_limit": limit}


def window_of(n=520):
    d = _dates(n)
    return d[0].strftime("%Y%m%d"), d[-1].strftime("%Y%m%d")


# ---------------- 假 zeus MCP 服务 ----------------
def _filter(rows, args):
    out = rows
    for k in ("ts_code", "fut_code", "exchange", "trade_date"):
        if args.get(k):
            out = [r for r in out if r.get(k) == args[k]]
    if args.get("start_date"):
        out = [r for r in out if r["trade_date"] >= args["start_date"]]
    if args.get("end_date"):
        out = [r for r in out if r["trade_date"] <= args["end_date"]]
    return out


def serve(data, tools=("fut_daily", "fut_basic", "fut_settle", "ft_limit"), token=TOKEN, name="ZeusMCP"):
    """启动假 zeus（后台线程），返回 (url, server)。tools 控制 tools/list 暴露哪些工具。"""
    class H(BaseHTTPRequestHandler):
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
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401); self.end_headers(); return
            if msg["method"] == "initialize":
                return self._sse({"jsonrpc": "2.0", "id": msg["id"], "result": {
                    "protocolVersion": "2025-03-26", "capabilities": {},
                    "serverInfo": {"name": name, "version": "1.0.0"}}},
                    headers=[("Mcp-Session-Id", "s1")])
            if msg["method"] == "notifications/initialized":
                self.send_response(202); self.end_headers(); return
            if msg["method"] == "tools/list":
                return self._sse({"jsonrpc": "2.0", "id": msg["id"],
                                  "result": {"tools": [{"name": t} for t in tools]}})
            tool, args = msg["params"]["name"], msg["params"]["arguments"]
            if tool not in tools:
                return self._sse({"jsonrpc": "2.0", "id": msg["id"],
                                  "error": {"code": -32602, "message": f"Unknown tool: {tool}"}})
            if args.get("ts_code") == "BAD.SHF":
                result = {"content": [{"type": "text", "text": "boom"}], "isError": True}
            else:
                rows = _filter(data[tool], args)
                result = {"content": [{"type": "text", "text": json.dumps(r)} for r in rows],
                          "isError": False}
            self._sse({"jsonrpc": "2.0", "id": msg["id"], "result": result})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/mcp", srv
