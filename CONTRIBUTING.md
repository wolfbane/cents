# Contributing

Thanks for your interest in Cents.

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. The `[broker]` extra adds Alpaca trading support if you need it.

## Tests

```bash
pytest                        # Full suite
pytest tests/test_agents.py   # Single file
pytest -k "test_research"     # Pattern match
```

Add tests for any new behavior. The codebase uses in-memory SQLite fixtures (see `tests/conftest.py`) — no external API calls in tests.

## Style

Ruff and Black are on the roadmap but not yet wired up. For now, please match the existing code style:

- Type hints on public APIs
- Small, focused modules (~200–400 lines)
- Immutable patterns where reasonable
- Clear errors over silent fallback

## Docs

Per-command reference pages under `website/src/content/docs/commands/` are **auto-generated** from Click introspection by `scripts/generate_docs.py`. Hand-edits to those `*.mdx` files will be wiped on regeneration. To add narrative prose for a command, drop a sibling `_<command>.intro.mdx` next to the generated file — its content is prepended into the page on regen. The underscore prefix keeps Astro from rendering the intro source as a doc page of its own.

```bash
python scripts/generate_docs.py    # regenerate after CLI changes
```

Non-command docs (`architecture.mdx`, `principles.mdx`, `scheduling.mdx`, etc.) are hand-written and edited directly.

## Anthropic SDK usage

Every Anthropic client constructed in cents must pass an explicit
`timeout=settings.anthropic_timeout_sec` (default 30s). The SDK default
is a 600s read-timeout; combined with the SDK's two automatic retries
plus exponential backoff, a single hung call can burn ~30+ minutes. We
hit exactly this in the wild (cents-87v: an MCD sentiment call lost 38
minutes mid-symbol), so the pattern is now non-negotiable.

```python
from anthropic import Anthropic
from cents.config import settings

client = Anthropic(
    api_key=settings.anthropic_api_key,
    timeout=settings.anthropic_timeout_sec,
)
```

Current call sites — keep them passing `timeout=...`, and add the same
pattern to any new ones:

- `src/cents/agents/sentiment.py`
- `src/cents/agents/event.py`
- `src/cents/factory/premise.py`
- `src/cents/eval/runner.py`
- `src/cents/llm_usage.py`

The worst-case wall-clock per LLM call with this setting is ~106s
(30s × 2 retries + exponential backoff up to 8s). Override the default
via `CENTS_ANTHROPIC_TIMEOUT_SEC` or `anthropic_timeout_sec` in
`~/.cents/config.toml` if you have a documented reason; do not remove
the keyword argument.

## Issues and PRs

File issues and pull requests via [GitHub](https://github.com/wolfbane/cents). For PRs, include a brief description of the change and confirm `pytest` passes locally.

For security concerns, see [SECURITY.md](SECURITY.md) — please do not open public issues for vulnerabilities.
