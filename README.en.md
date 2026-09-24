# statistical-arbitrage-time-series

An Agent Skill for reproducible, sourced statistical-arbitrage research on Chinese commodity futures (calendar and cross-commodity spreads). Data comes only from the zeus MCP and is stored as an immutable, replayable snapshot. It covers point-in-time contract mapping and rolls, training-window ADF/KPSS/Engle-Granger tests, spread and hedge-ratio modeling, mean reversion, an integer-lot backtest on actual contracts with futures costs (per-leg fees, slippage, user-specified margin, and liquidity constraints), significance checks, and robustness risks. Statistical evidence and trading feasibility get separate verdicts.

This project is for research and method validation only. It makes no return claims, is not investment advice, and does not imply official endorsement by QuantSkills or any platform.

Start with [`SKILL.md`](SKILL.md), the [research guide](references/statarb-guide.md), the [zeus MCP interface](references/zeus-mcp-interface.md), and the scripts ([`run_statarb.py`](scripts/run_statarb.py), [`futures.py`](scripts/futures.py)):

```bash
# uv installs the inline-declared dependencies on first run (~30 s); zeus connection details are reused from
# Claude Code's zeus MCP config, or copy .env.example to .env in your workspace
uv run scripts/run_statarb.py --check-zeus RB2501.SHF
uv run scripts/run_statarb.py --config statarb_runs/hc_rb/config.json --fetch --out-dir statarb_runs/hc_rb/out
uv run scripts/run_statarb.py --source synthetic --mode strong   # offline self-test
uv run --with pytest --with-requirements requirements.txt pytest tests -q   # deterministic tests
```

Runtime entrypoints for Claude Code, Codex, Cursor, Hermes, and OpenClaw are documented in [`agents/portable-loader.md`](agents/portable-loader.md).

## License

GPL-3.0-only. See [`LICENSE`](LICENSE).
