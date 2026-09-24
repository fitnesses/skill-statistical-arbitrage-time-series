# zeus MCP 接口约定（期货统计套利 Skill 所需）

本 Skill **只**通过 zeus MCP 取数（不接第三方、不直连 DolphinDB）。脚本会调用两个工具：`fut_daily` 与 `fut_basic`，均已上线（2026-09-23 已核对真实返回与本约定一致）。脚本启动时用 `tools/list` 探测，缺失的工具会被记为“能力缺口”，并按下文的替代办法运行，报告里逐条披露。

字段命名与 tushare 同名接口保持兼容（现有 `fut_daily` 即为此口径），便于 zeus 侧直接映射 DolphinDB 表。

**不经 zeus 的数据**：手续费率/每手手续费、保证金率由用户在研究配置 `assumptions.<品种>` 中指定（这些数据难以获得可靠的逐日历史）；涨跌停暂不建模。

## 通用约定

- 传输：MCP Streamable HTTP（`initialize` → `notifications/initialized` → `tools/list` / `tools/call`），Bearer token 鉴权。
- 返回：行对象列表（FastMCP 可把 list 拆成多条 `text` 内容，每条一个 JSON 对象；也可返回一条 JSON 数组）。无数据返回空列表，不要报错。
- 日期：`YYYYMMDD` 字符串；数值字段为数字（不是字符串）；缺失值用 `null`。
- 合约代码：`<品种><YYMM>.<交易所>`，年月**必须 4 位**（郑商所也用 4 位，如 `SR2501.ZCE`），交易所后缀 `SHF/DCE/ZCE/INE/GFE/CFX`。连续合约代码 `RB.SHF` 可存在，但 Skill 的映射只使用具体合约。
- 工具出错时返回 `isError: true` 及可读原因；未知工具返回 JSON-RPC error。

## 1. `fut_daily`（已有，必需）

| 参数 | 说明 |
|---|---|
| `ts_code` | 合约代码（Skill 每次只查一张合约） |
| `start_date` / `end_date` | 区间 |
| `trade_date` / `exchange` | 可选 |

| 字段 | 必需 | 说明 |
|---|---|---|
| `ts_code`, `trade_date` | ✔ | 主键；同一 (ts_code, trade_date) 不得重复 |
| `close`, `settle` | ✔ | 收盘价、结算价 |
| `open`, `high`, `low` | ✔（可为 null） | 无成交日（`vol=0`）可为 null；Skill 视该日为不可成交，两腿顺延 |
| `vol`, `oi` | ✔ | 成交量、持仓量（手）；**`oi` 用于按时点安全的主力/次主力映射** |
| `pre_settle` | 可选 | 昨结算；Skill 暂不使用 |
| `pre_close`, `change1`, `change2`, `amount`, `oi_chg` | 可选 | Skill 不使用 |

## 2. `fut_basic`（已有）— 合约列表与静态合约参数

| 参数 | 说明 |
|---|---|
| `exchange` | 交易所后缀，如 `SHF`（必填） |
| `fut_code` | 品种代码，如 `RB`（Skill 只用 `exchange` + `fut_code` 查询） |

| 字段 | 必需 | 说明 |
|---|---|---|
| `ts_code` | ✔ | 具体合约代码 |
| `fut_code`, `exchange` | ✔ | |
| `multiplier` | ✔ | 合约乘数（吨/手等），按合约记录（合约存续期内不变） |
| `price_tick` | ✔ | 最小变动价位 |
| `list_date`, `delist_date` | ✔ | 上市日、最后交易日（用于筛选区间内存续的合约） |
| `d_month` | 推荐 | 交割月 `YYYYMM` |
| `last_ddate` | 可选 | 最后交割日 |

**缺失时的替代**：按“品种+YYMM”枚举区间起始月到结束月后 13 个月的合约代码，逐个用 `fut_daily` 探测；乘数/最小变动价位取配置 `assumptions.<品种>.multiplier / price_tick`，并在报告标注“合约参数使用配置假设”。

## 兼容性检查

```bash
uv run scripts/run_statarb.py --check-zeus RB2501.SHF --start 20240102 --end 20240131
```

输出 JSON：`required.fut_daily` 必须为 `ok`（否则退出码 1）；`optional.*` 为 `ok`、`absent（替代：…）`、`missing fields [...]` 或 `no rows …`。这是真实 MCP 的集成检查，普通测试（`pytest`）只用本地假服务，不依赖 zeus 在线。
