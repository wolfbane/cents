# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
pip install -e .              # Install for development
pip install -e ".[dev]"       # With test dependencies
pip install -e ".[broker]"    # With Alpaca trading

pytest                        # Run all tests
pytest tests/test_agents.py   # Run specific test file
pytest -k "test_research"     # Run tests matching pattern
pytest -v --tb=short          # Verbose with short tracebacks

cents --help                  # CLI usage
```

## Architecture

**Thesis-Driven Investment Tracking**: Users create investment theses, research agents gather evidence, positions are tracked against theses, and outcomes measure thesis accuracy (not just P&L).

### System Layers
```
┌─────────────────────────────────────────────────────────────┐
│  CLI (cli.py)                                               │
│  Commands: thesis, position, watch, scan, alert, broker,    │
│            outcome, research                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Agents (agents/)                    Broker (broker/)       │
│  ┌─────────────┐ ┌─────────────┐    ┌─────────────────┐    │
│  │ Fundamentals│ │  Technical  │    │  AlpacaClient   │    │
│  │   (FMP)     │ │  (Alpaca)   │    │  (alpaca-py)    │    │
│  └─────────────┘ └─────────────┘    └─────────────────┘    │
│  ┌─────────────┐ ┌─────────────┐                           │
│  │   Macro     │ │  Sentiment  │    All return structured  │
│  │   (FRED)    │ │  (NewsAPI)  │    AgentResult or         │
│  └─────────────┘ └─────────────┘    BrokerPosition          │
│  ┌─────────────────────────────┐                           │
│  │      Orchestrator           │ ← Aggregates all agents   │
│  └─────────────────────────────┘                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Models (models/)              Config (config.py)           │
│  Thesis, Evidence, Position,   Settings dataclass           │
│  Outcome, WatchlistItem, Alert ~/.cents/config.toml         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Persistence (db/)                                          │
│  schema.py: SQLite DDL + _migrate_schema()                  │
│  repository.py: ThesisRepo, PositionRepo, EvidenceRepo...  │
└─────────────────────────────────────────────────────────────┘
```

### Key Abstractions

**BaseAgent** (`agents/base.py`): Abstract class all agents inherit. Provides:
- `research(symbol, thesis) → AgentResult` - Abstract method each agent implements
- `_with_retries(func)` - Exponential backoff for API calls
- `create_evidence()` - Factory for Evidence objects with agent name

**AgentResult**: Dataclass returned by all agents:
- `evidence: list[Evidence]` - Supporting/contradicting/neutral findings
- `conviction_delta: float` - How much to adjust thesis (e.g., +3.0 or -2.0)
- `summary: str` - Human-readable one-liner
- `dimension_scores: dict[str, float]` - Per-dimension deltas (valuation, quality, moat, etc.)

**Repository Pattern** (`db/repository.py`): Each model has a repository:
- Accepts optional `conn` for testing with in-memory DBs
- Handles serialization (JSON for tags, ISO for dates)
- `_row_to_*` methods convert sqlite3.Row to dataclass

**Data Providers** (`data/`): Abstraction layer for market data:
- `PriceDataProvider` protocol → `AlpacaPriceProvider` (price/volume data)
- `FundamentalsDataProvider` protocol → `FMPFundamentalsProvider` (P/E, margins, etc.)
- FMP uses stable API (`/stable/` endpoints with query params, not legacy `/api/v3/`)
- Agents accept optional providers for dependency injection in tests

### Data Flow for Research
1. CLI calls agent's `research(symbol, thesis)`
2. Agent fetches external data with retries
3. Agent creates Evidence objects and calculates conviction delta
4. CLI saves evidence and updates thesis conviction
5. Orchestrator synthesizes across all agents

### Scan & Alert Flow
1. `cents scan` iterates watchlist
2. For each symbol, runs OrchestratorAgent
3. Compares conviction delta to threshold (per-symbol or default)
4. Creates Alert if threshold exceeded
5. Calls `notify()` for terminal output and optional webhook

## Configuration

Config file at `~/.cents/config.toml`:
```toml
news_api_key = "..."           # newsapi.org
fred_api_key = "..."           # fred.stlouisfed.org
fmp_api_key = "..."            # financialmodelingprep.com
alpaca_api_key = "..."         # alpaca.markets (data + trading)
alpaca_secret_key = "..."
default_scan_threshold = 5.0   # conviction delta for alerts
default_output = "text"        # "text" or "json"
```

Environment variables override config: `NEWS_API_KEY`, `FRED_API_KEY`, `FMP_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `CENTS_OUTPUT_FORMAT`, `CENTS_SCAN_THRESHOLD`, `CENTS_WEBHOOK_URL`, `CENTS_DB_PATH`.

Database stored at `~/.cents/data/cents.db` (created automatically). Override with `CENTS_DB_PATH` env var.

## Testing Notes

Tests mock `get_settings()` to avoid reading real config:
```python
@patch("cents.agents.sentiment.get_settings")
def test_no_api_key(self, mock_settings):
    mock_settings.return_value.news_api_key = None
```

Database tests use fresh in-memory connections via `conftest.py` fixtures.
