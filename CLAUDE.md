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

### Known Gaps
- **Network fragility**: Agents call APIs without retries/caching; missing API keys silently degrade
- **Data quality**: Evidence lacks raw payloads; broker sync uses current date vs actual entry timestamps
- **UX**: No JSON output or config file; single global threshold for alerts

### Next Priorities
1. Config file for API keys + `--output json` flag
2. Retry/backoff for external APIs + explicit warnings when keys missing
3. Per-symbol alert thresholds

### Beads tracked (`bd help`)
Non-traditional project focused on real outcomes
