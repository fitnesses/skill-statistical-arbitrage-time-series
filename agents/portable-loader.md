# Portable Runtime Loader

This skill uses `SKILL.md` as its canonical instruction set. A runtime adapter should load that file and resolve paths relative to the repository root.

## Runtime mapping

- **Codex:** install under `.agents/skills/statistical-arbitrage-time-series` or `~/.agents/skills/...`; load `SKILL.md`.
- **Claude Code:** install under `.claude/skills/statistical-arbitrage-time-series` or `~/.claude/skills/...`; load `SKILL.md`.
- **Cursor:** use [`cursor-rule.mdc`](cursor-rule.mdc) and load `SKILL.md` for the active workspace.
- **Hermes:** load this file, then `SKILL.md`.
- **OpenClaw:** load [`openai.yaml`](openai.yaml), then `SKILL.md`.

Install dependencies once with `pip install -r requirements.txt` (Python ≥ 3.10; `statsmodels` is mandatory so every p-value is real). Data access goes only through the zeus MCP (`ZEUS_MCP_URL`, `ZEUS_MCP_TOKEN`); see `references/zeus-mcp-interface.md`.

The bundled scripts are research tooling; it must not be treated as financial advice or as a guarantee of research validity.
