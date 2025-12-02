# CLAUDE.md

## Build & Test

```bash
pip install -e ".[dev]"       # Install with test deps
pip install -e ".[broker]"    # Add Alpaca trading
pytest                        # Run all tests
cents --help                  # CLI usage
```

## Architecture

**Thesis-Driven Investment Tracking**: Create theses → agents gather evidence → track positions → measure thesis accuracy.

```
CLI (thesis, position, watch, scan, alert, broker, outcome, research)
  │
Agents: Fundamentals, Technical, Macro, Sentiment, Moat, Insider → Orchestrator
  │     (FMP)         (Alpaca)   (FRED) (NewsAPI)  (FMP) (FMP)
  │
Broker: AlpacaClient (alpaca-py) ─── All return AgentResult or BrokerPosition
  │
Models: Thesis, Evidence, Position, Outcome, WatchlistItem, Alert
  │
DB: SQLite at ~/.cents/data/cents.db (schema.py, repository.py)
```

**Key Patterns**:
- `BaseAgent.research(symbol, thesis) → AgentResult` with retry logic
- `AgentResult`: evidence list, conviction_delta, summary, dimension_scores
- Repository pattern with optional `conn` for test injection
- Data providers: `PriceDataProvider` (Alpaca), `FundamentalsDataProvider` (FMP stable API)

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

## Testing

Tests mock `get_settings()` and use in-memory DB fixtures from `conftest.py`.
