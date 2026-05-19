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

## Issues and PRs

File issues and pull requests via [GitHub](https://github.com/wolfbane/cents). For PRs, include a brief description of the change and confirm `pytest` passes locally.

For security concerns, see [SECURITY.md](SECURITY.md) — please do not open public issues for vulnerabilities.
