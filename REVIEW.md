# Codebase Review and Next-Step Suggestions

## What exists today
- **CLI surface area**: `src/cents/cli.py` exposes commands for research (agent execution + optional evidence persistence), thesis lifecycle, paper positions, outcomes, watchlist scans/alerts, and Alpaca broker helpers. Evidence can be saved and thesis conviction auto-adjusted when a thesis is provided during research runs. Alert flows include watchlist scanning plus read/unread management. 
- **Persistence layer**: A lightweight SQLite schema (auto-created in `data/cents.db`) tracks theses, evidence, positions, outcomes, watchlist entries, and alerts, with repository classes for CRUD and simple filters.
- **Research agents**: Fundamentals, technicals, macro, sentiment, and an orchestrator aggregate signals. Each agent relies on external data (yfinance, FRED API, NewsAPI) and adjusts conviction via heuristic thresholds before synthesis.
- **Broker integration**: Optional Alpaca wrapper exposes account/position queries, syncing broker positions into the local store, and paper market orders when alpaca-py and API keys are available.

## Gaps and risks observed
- **Network fragility & rate limits**: Agents call third-party APIs synchronously without retries, rate-limit handling, or caching; failures return neutral results and may hide transient issues. News and macro agents silently degrade when API keys are missing, which makes unattended scans produce little value without surfacing configuration gaps.
- **Data quality & reproducibility**: Evidence stores only text plus lightweight metadata; no raw payloads or hashes are persisted, making follow-up audits or re-computation difficult. Position sync stamps the current date rather than broker entry timestamps, impacting P&L accuracy.
- **User experience**: The CLI emits mixed formatting and does not offer JSON output or a config file, which makes automation and integration harder. Scan alerts hinge solely on conviction deltas with a single numeric threshold and lack per-symbol settings.
- **Testing coverage**: Tests exist but primarily exercise happy-path flows; there are no integration tests for actual API interactions, configuration failure cases, or alert/notification behavior.

## Recommended next steps
- **Improve resilience**: Add retry/backoff and simple response caching for yfinance/NewsAPI/FRED calls; surface explicit warnings when required API keys are absent during scans instead of returning neutral output.
- **Richer data retention**: Persist raw API payload excerpts or hashes alongside evidence metadata, and store broker entry timestamps/side explicitly when syncing to keep P&L math trustworthy.
- **Config & UX**: Introduce a config file/ENV loader for API keys, thresholds, and defaults; add `--output json` or `--quiet` flags so research/scan results can be scripted; allow per-symbol scan thresholds and alert destinations.
- **Testing focus**: Add offline fixtures for agent inputs to validate conviction deltas, and unit tests around alert generation/notification fallbacks and broker sync edge cases (duplicates, missing credentials). Consider contract tests for repository serialization to guard schema changes.
