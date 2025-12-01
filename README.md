# Cents

Agentic investing guidance with thesis-driven research.

Cents helps you track investment theses, gather evidence from multiple research agents, and measure outcomes based on thesis accuracy—not just P&L.

## Installation

```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # With test dependencies
pip install -e ".[broker]"    # With Alpaca trading integration
```

Requires Python 3.11+.

## Quick Start

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

## Configuration

Create `~/.cents/config.toml`:

```toml
news_api_key = "..."           # newsapi.org
fred_api_key = "..."           # fred.stlouisfed.org
fmp_api_key = "..."            # financialmodelingprep.com
alpaca_api_key = "..."         # alpaca.markets (data + trading)
alpaca_secret_key = "..."
default_scan_threshold = 5.0   # conviction delta for alerts
default_output = "text"        # "text" or "json"
```

Environment variables override config: `NEWS_API_KEY`, `FRED_API_KEY`, `FMP_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`.

## Commands

| Command | Description |
|---------|-------------|
| `thesis` | Create, list, update, close investment theses |
| `research` | Run research agents on a symbol |
| `watch` | Manage watchlist (add, list, remove) |
| `scan` | Scan watchlist and generate alerts |
| `alert` | View and manage alerts |
| `position` | Track positions linked to theses |
| `outcome` | Record and review investment outcomes |
| `broker` | Alpaca broker integration (positions, orders) |

## Research Agents

Research runs multiple agents that gather evidence and calculate conviction deltas:

| Agent | Source | Data |
|-------|--------|------|
| **Fundamentals** | FMP | P/E, margins, ROE, debt ratios |
| **Technical** | Alpaca | Price momentum, moving averages, volume |
| **Macro** | FRED | Fed funds rate, yield curve, VIX, unemployment |
| **Sentiment** | NewsAPI | Recent news sentiment analysis |
| **Orchestrator** | All | Synthesizes all agents into overall conviction |

Each agent returns:
- Evidence items (supporting/contradicting/neutral)
- Conviction delta (how much to adjust thesis, e.g., +3.0 or -2.0)
- Dimension scores (valuation, quality, moat, technical, risk)

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

1. **Create thesis** with hypothesis, valuation view, time horizon
2. **Research** gathers evidence from agents, updates conviction
3. **Scan** watchlist periodically, get alerts when conviction changes significantly
4. **Track position** when you enter a trade
5. **Record outcome** when thesis resolves—measure thesis accuracy, not just returns

## Development

```bash
pytest                        # Run all tests
pytest tests/test_agents.py   # Run specific test file
pytest -k "test_research"     # Run tests matching pattern
```

Database stored at `./data/cents.db` (SQLite, created automatically).
