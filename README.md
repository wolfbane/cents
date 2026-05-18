# Cents

A research experiment in agent-orchestrated investment hypothesis tracking. **Not an investing tool.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#development)
[![Docs](https://img.shields.io/badge/docs-dollars-and-cents.ai-blueviolet.svg)](https://dollars-and-cents.ai)

> ⚠️ **Research tool, not an investing tool — and not financial advice.**
> Cents is an open-source experiment in multi-agent LLM orchestration applied
> to investment research. It is **not** an investment adviser, broker, or
> recommendation engine. Outputs (conviction scores, premise tags, model
> signals) are model-generated, uncalibrated, and may be wrong. There is no
> KYC, no suitability check, no portfolio risk controls, no slippage/borrow
> modeling, no reconciliation or audit trail. **Real-money trading is
> technically possible but explicitly out of scope** — the autonomous loop is
> hard-coded to paper. If you point this at a live account, you are doing so
> against the documented intent of the project. Read the
> [scope statement](https://dollars-and-cents.ai/scope/) before going further.
>
> The authors and contributors of this software do not provide investment
> advice, have no advisory relationship with any user, accept no fiduciary
> duty, and make no warranty as to the accuracy, completeness, or
> suitability of any output for any purpose. See the [LICENSE](LICENSE)
> rider for the full disclaimer.

## What is Cents?

Cents is a research pipeline for studying whether multi-agent LLM orchestration produces a calibrated signal on forward equity returns. You write down a thesis ("NVDA's data-center growth will drive earnings beats over 12 months"), and a set of specialised agents — fundamentals, technical, macro, sentiment, moat, insider, plus an orchestrator — gather evidence and adjust a conviction score. The autonomous factory loop walks a universe of symbols, opens paired paper theses where the orchestrator clears an entry threshold, and closes them on target / stop / horizon / premise-invalidation. **The engine records outcomes, it does not gate on trading-style controls.** Sizing, costs, hedging, drawdown, and liquidity utilities live in `cents/finance/` but are opt-in — the default research mode opens everything that clears the threshold so the resulting dataset isn't censored.

A matched control arm (`--orchestrator random`) and a pre-registered experiments workflow (`cents experiment register`) make the pipeline falsifiable: every thesis carries `orchestrator_label` (`"llm"` | `"random"`) and `experiment_id`, so cohort analytics can ask whether the LLM arm beats the random arm under a hypothesis written down before any theses opened.

Everything lives in local SQLite; data flows from FMP, Alpaca (paper), FRED, and NewsAPI.

## Install

Cents is not yet published on PyPI — install from source:

```bash
git clone https://github.com/wolfbane/cents.git
cd cents
pip install .                  # Basic install
pip install ".[dev]"           # With test dependencies
pip install ".[broker]"        # With Alpaca trading integration
```

For local development, swap `pip install` for `pip install -e` to get an editable install.

Requires Python 3.11+.

## Quickstart

```bash
# 1. Create a thesis
cents thesis create --symbol NVDA --title "NVDA AI dominance continues" \
  --hypothesis "Data center growth will drive earnings beats" \
  --valuation undervalued --time-horizon medium

# 2. Add to watchlist
cents watch add NVDA --thesis <thesis-id> --threshold 5.0

# 3. Run research
cents research NVDA --thesis <thesis-id>

# 4. Scan watchlist for alerts
cents scan

# 5. View alerts
cents alert list
```

## Commands

| Command | Description |
|---------|-------------|
| `thesis` | Create, list, update, close investment theses |
| `research` | Run research agents on a symbol |
| `evidence` | Manage research evidence |
| `watch` | Manage watchlist (add, list, remove) |
| `scan` | Scan watchlist and generate alerts |
| `alert` | View and manage alerts |
| `position` | Track positions linked to theses |
| `outcome` | Record and review investment outcomes |
| `recommend` | Emit model signals (bullish/bearish/neutral) for open theses |
| `portfolio` | Manage portfolios (separate database files) |
| `broker` | Alpaca broker integration (positions, orders) |
| `backtest` | Run and analyze agent backtests |
| `status` | Show current configuration and database status |

## Research Agents

| Agent | Source | Data |
|-------|--------|------|
| **Fundamentals** | FMP | P/E, margins, ROE, debt ratios, analyst ratings |
| **Technical** | Alpaca | Price momentum, moving averages, volume, 52w range |
| **Macro** | FRED | Fed funds rate, yield curve, VIX, unemployment |
| **Sentiment** | NewsAPI | Recent news sentiment analysis |
| **Moat** | FMP | Margin stability, ROIC trends, competitive advantages |
| **Insider** | FMP | Insider buying/selling patterns, cluster activity |
| **Orchestrator** | All | Synthesizes all agents into weighted conviction |

Each agent returns evidence items (supporting/contradicting/neutral), a conviction delta, and dimension scores (valuation, quality, moat, technical, risk).

## Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Create    │────▶│   Research  │────▶│    Track    │
│   Thesis    │     │   & Watch   │     │   Position  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    Scan     │     │   Record    │
                    │   Alerts    │     │   Outcome   │
                    └─────────────┘     └─────────────┘
```

## Configuration

`~/.cents/config.toml`:

```toml
fmp_api_key = "..."            # financialmodelingprep.com
alpaca_api_key = "..."         # alpaca.markets
alpaca_secret_key = "..."
news_api_key = "..."           # newsapi.org (optional)
fred_api_key = "..."           # fred.stlouisfed.org (optional)
default_scan_threshold = 5.0
```

Env vars override config: `FMP_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `NEWS_API_KEY`, `FRED_API_KEY`, `CENTS_DB_PATH`, `CENTS_OUTPUT_FORMAT`, `CENTS_SCAN_THRESHOLD`, `CENTS_WEBHOOK_URL`.

Database stored at `~/.cents/data/cents.db` (SQLite, created automatically).

## Documentation

Full docs at [dollars-and-cents.ai](https://dollars-and-cents.ai) — quickstart walkthrough, command reference for all 13 groups, agent internals, architecture, and roadmap.

## Development

```bash
pytest                        # Run all tests
pytest tests/test_agents.py   # Run specific test file
pytest -k "test_research"     # Run tests matching pattern
```

## License

MIT — see [LICENSE](LICENSE).
