# statistical-arbitrage-time-series

An Agent Skill for reproducible, sourced statistical-arbitrage research on Chinese commodity futures (calendar and cross-commodity spreads). Data comes only from the zeus MCP and is stored as an immutable, replayable snapshot. It covers point-in-time contract mapping and rolls, training-window ADF/KPSS/Engle-Granger tests, spread and hedge-ratio modeling, mean reversion, an integer-lot backtest on actual contracts with futures costs (per-leg fees, slippage, margin, limit-price and liquidity constraints), significance checks, and robustness risks. Statistical evidence and trading feasibility get separate verdicts.

This project is for research and method validation only. It makes no return claims, is not investment advice, and does not imply official endorsement by QuantSkills or any platform.

Start with [`SKILL.md`](SKILL.md), the [research guide](references/statarb-guide.md), the [zeus MCP interface](references/zeus-mcp-interface.md), and the scripts ([`run_statarb.py`](scripts/run_statarb.py), [`futures.py`](scripts/futures.py)):

```bash
pip install -r requirements.txt            # Python >= 3.10; statsmodels is mandatory
export ZEUS_MCP_URL=http://<host>:8000/mcp ZEUS_MCP_TOKEN=<token>
python scripts/run_statarb.py --config run1/config.json --fetch --out-dir run1
python scripts/run_statarb.py --source synthetic --mode strong   # offline self-test
python -m pytest tests -q                                        # deterministic tests
```

Runtime entrypoints for Claude Code, Codex, Cursor, Hermes, and OpenClaw are documented in [`agents/portable-loader.md`](agents/portable-loader.md).

## License

GPL-3.0-only. See [`LICENSE`](LICENSE).
