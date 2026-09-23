# zeus MCP 接口约定（期货统计套利 Skill 所需）

本 Skill **只**通过 zeus MCP 取数（不接第三方、不直连 DolphinDB）。下表是脚本会调用的工具：`fut_daily` 已上线；其余三个是**待 zeus 实现的接口**。脚本启动时用 `tools/list` 探测，缺失的工具会被记为“能力缺口”，并按下文的替代办法运行，报告里逐条披露。zeus 按本约定实现后，无需改动 Skill 即自动启用。

字段命名与 tushare 同名接口保持兼容（现有 `fut_daily` 即为此口径），便于 zeus 侧直接映射 DolphinDB 表。

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
| `open`, `high`, `low`, `close`, `settle` | ✔ | 价格（元/计价单位） |
| `vol`, `oi` | ✔ | 成交量、持仓量（手）；**`oi` 用于按时点安全的主力/次主力映射** |
| `pre_settle` | 推荐 | 昨结算；缺 `ft_limit` 时用于推断一字板方向 |
| `pre_close`, `change1`, `change2`, `amount`, `oi_chg` | 可选 | Skill 不使用 |

## 2. `fut_basic`（待实现）— 合约列表与静态合约参数

| 参数 | 说明 |
|---|---|
| `exchange` | 交易所后缀，如 `SHF` |
| `fut_code` | 品种代码，如 `RB` |

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

## 3. `fut_settle`（待实现）— 按日生效的手续费与保证金

| 参数 | 说明 |
|---|---|
| `ts_code` | 合约代码 |
| `start_date` / `end_date` 或 `trade_date` | 区间 |

| 字段 | 必需 | 说明 |
|---|---|---|
| `ts_code`, `trade_date` | ✔ | 该行参数在 `trade_date` 当日生效 |
| `trading_fee_rate` | ✔ | 按成交额收取的交易所手续费率（小数，万分之一 = 0.0001）；按手收取时填 0 |
| `trading_fee` | ✔ | 按手收取的交易所手续费（元/手）；按成交额收取时填 0 |
| `long_margin_rate`, `short_margin_rate` | ✔ | 交易所多/空投机保证金率（小数） |
| `offset_today_fee` | 可选 | 平今手续费（日线回测暂不使用，保留） |
| `settle`, `delivery_fee`, `b_hedging_margin_rate`, `s_hedging_margin_rate` | 可选 | |

Skill 在交易日 d 取该合约 `trade_date ≤ d` 的最近一行（as-of，按时点安全），再乘以配置的经纪商手续费倍数、加上经纪商保证金加成。

**缺失时的替代**：配置 `assumptions.<品种>.fee_rate / fee_per_lot / margin_rate`（整段常数），报告标注“非逐日历史”；配置也没有时直接报错，并指明缺哪个工具、哪个配置键，不会默认按 0 计。

## 4. `ft_limit`（待实现）— 每日涨跌停价

| 参数 | 说明 |
|---|---|
| `ts_code` | 合约代码 |
| `start_date` / `end_date` 或 `trade_date` | 区间 |

| 字段 | 必需 | 说明 |
|---|---|---|
| `ts_code`, `trade_date` | ✔ | |
| `up_limit`, `down_limit` | ✔ | 当日涨停价、跌停价 |
| `pre_settle` | 可选 | |

用途：买单在 `low ≥ up_limit`（全天封涨停）时判定不能成交，卖单在 `high ≤ down_limit` 时判定不能成交；任一腿不能成交则两腿都不成交，次日重试。

**缺失时的替代**：按 OHLC 一字板（`high == low`）结合 `pre_settle` 推断封板方向，报告标注“涨跌停为推断”。

## 兼容性检查

```bash
export ZEUS_MCP_URL=http://<host>:8000/mcp ZEUS_MCP_TOKEN=<token>
python scripts/run_statarb.py --check-zeus RB2501.SHF --start 20240102 --end 20240131
```

输出 JSON：`required.fut_daily` 必须为 `ok`（否则退出码 1）；`optional.*` 为 `ok`、`absent（替代：…）`、`missing fields [...]` 或 `no rows …`。这是真实 MCP 的集成检查，普通测试（`pytest`）只用本地假服务，不依赖 zeus 在线。
