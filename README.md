# Cents

Thesis-driven investment research, agent-orchestrated.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#development)
[![Docs](https://img.shields.io/badge/docs-dollarsandcents.ai-blueviolet.svg)](https://dollarsandcents.ai)

> ⚠️ **Not financial advice.** Cents is an educational and research tool for tracking your own investment theses. It does not provide investment advice, recommendations, or solicitations. Outputs are model-generated and may be inaccurate. You are solely responsible for your own investment decisions. Consult a licensed financial advisor before trading.

## What is Cents?

Cents is a CLI for the AI/finance crowd that treats investing as a hypothesis-testing exercise. You write down a thesis ("NVDA's data-center growth will drive earnings beats over 12 months"), and Cents orchestrates seven specialised research agents — fundamentals, technical, macro, sentiment, moat, insider, plus an orchestrator — that gather evidence and adjust a conviction score. Positions and outcomes are tracked against the original thesis so you measure thesis accuracy, not just P&L. Everything lives in local SQLite; data flows from FMP, Alpaca, FRED, and NewsAPI.

## Install

```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # With test dependencies
pip install -e ".[broker]"    # With Alpaca trading integration
```

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
| `recommend` | Generate buy/sell/hold recommendations from thesis rules |
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

Full docs at [dollarsandcents.ai](https://dollarsandcents.ai) — quickstart walkthrough, command reference for all 13 groups, agent internals, architecture, and roadmap.

## Development

```bash
pytest                        # Run all tests
pytest tests/test_agents.py   # Run specific test file
pytest -k "test_research"     # Run tests matching pattern
```

## License

MIT — see [LICENSE](LICENSE).
