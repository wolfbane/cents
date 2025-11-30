## cents - Agentic Investing Guidance

CLI-first thesis-driven investment tracking with AI research agents.

### Quick Start
```bash
pip install -e .
cents thesis create "Your thesis here"
cents position open AAPL 100 --price 150.00 --thesis <id>
cents position close <id> 160.00
cents outcome record <id> --accuracy correct
```

### Architecture
- **models/**: Domain objects (Thesis, Evidence, Position, Outcome)
- **db/**: SQLite persistence (schema.py, repository.py)
- **agents/**: Research agents (Phase 2)
- **broker/**: Alpaca integration (Phase 5)

### Config
- `~/.cents/config.toml` or `CENTS_CONFIG` env var
- API keys: `NEWS_API_KEY`, `FRED_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- `--output json` and `--quiet` flags for scripting
- Per-symbol thresholds on watchlist

### Beads tracked (`bd help`)
Non-traditional project focused on real outcomes
