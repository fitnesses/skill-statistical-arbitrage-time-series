# statistical-arbitrage-time-series

统计套利与时间序列建模 Agent Skill（中文优先），面向国内商品期货的跨期与跨品种价差。数据只经 zeus MCP 获取并存为不可变快照，可离线复跑；覆盖按时点安全的合约映射与换月、训练窗 ADF/KPSS/Engle-Granger、价差建模与均值回归、真实合约整手回测（逐腿手续费、滑点、用户指定的保证金率与流动性约束）以及稳健性风险。统计证据与交易可行性分开判定。

本项目只用于研究与方法验证，不构成投资建议，不承诺收益，也不代表 QuantSkills 或任何平台的官方背书。

## 使用

- 技能定义：[`SKILL.md`](SKILL.md)
- 研究指南：[`references/statarb-guide.md`](references/statarb-guide.md)
- zeus MCP 接口约定：[`references/zeus-mcp-interface.md`](references/zeus-mcp-interface.md)
- 可执行脚本：[`scripts/run_statarb.py`](scripts/run_statarb.py)、[`scripts/futures.py`](scripts/futures.py)
- English: [`README.en.md`](README.en.md)

```bash
# 依赖由 uv 按脚本内联声明自动安装（首次约 30 秒）；zeus 连接信息自动复用 Claude Code 的 zeus MCP 配置，
# 或把 .env.example 复制为工作目录下的 .env 填写
uv run scripts/run_statarb.py --check-zeus RB2501.SHF
uv run scripts/run_statarb.py --config statarb_runs/hc_rb/config.json --fetch --out-dir statarb_runs/hc_rb/out
uv run scripts/run_statarb.py --source synthetic --mode strong   # 离线自测
uv run --with pytest --with-requirements requirements.txt pytest tests -q   # 确定性测试
```

依赖安装与 Claude Code、Codex、Cursor、Hermes、OpenClaw 的入口说明见 [`agents/portable-loader.md`](agents/portable-loader.md)。

## 许可证

GPL-3.0-only，详见 [`LICENSE`](LICENSE)。
