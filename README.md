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
anthropic_api_key = "..."      # anthropic.com (LLM tagging + premise classification)
default_scan_threshold = 5.0
```

Database stored at `~/.cents/data/cents.db` (SQLite, created automatically).

### Environment variables

Environment variables override config file values. Authoritative source for everything below is `src/cents/config.py` plus a handful of module-level reads (`cache.py`, `llm_usage.py`, `broker/alpaca.py`, `factory/universe_resolver.py`).

**API keys** — required for the upstream the agent depends on; cents fails soft when a key is missing (the affected agent contributes zero signal):

| Variable | Default | When to override |
|---|---|---|
| `FMP_API_KEY` | unset | Required for fundamentals, moat, insider, FMP-screener-sourced universes. |
| `ALPACA_API_KEY` | unset | Required for technical agent + paper broker integration. |
| `ALPACA_SECRET_KEY` | unset | Paired with `ALPACA_API_KEY`. |
| `NEWS_API_KEY` | unset | Required for sentiment agent. Without it, sentiment scoring is skipped with a `WARNING`. |
| `FRED_API_KEY` | unset | Required for macro agent. Without it, macro context is limited to a degraded set. |
| `ANTHROPIC_API_KEY` | unset | Required for sentiment scoring, premise classification, event tagging, eval harness. Without it the LLM features are skipped with a clear message. |

**Storage paths** — where cents writes its state:

| Variable | Default | When to override |
|---|---|---|
| `CENTS_DB_PATH` | `~/.cents/data/cents.db` | Point at a separate DB for portfolio isolation, dry-runs, or per-experiment sandboxes. |
| `CENTS_LLM_BLOB_DIR` | `~/.cents/data/llm_calls/` | Move LLM call provenance blobs (used by `cents evidence trace`) to a different filesystem. |
| `CENTS_CONFIG` | `~/.cents/config.toml` | Point at an alternate config file (useful for layered configs in CI). |
| `CENTS_FACTORY_CONFIG` | `~/.cents/factory.toml` | Point at an alternate factory config (e.g. `experiments/pilot.toml`). |

**Tuning** — knobs you'll touch when something hangs, costs more than expected, or returns noisy data:

| Variable | Default | Unit | When to override |
|---|---|---|---|
| `CENTS_ANTHROPIC_TIMEOUT_SEC` | `30` | seconds | Lower for chattier UIs; raise if you hit timeouts on very long premise classifications. SDK default is 600s — that 600s combined with retries can burn 30+ minutes on a single hung call, so don't go back to that. See `CONTRIBUTING.md`. |
| `CENTS_PER_SYMBOL_DEADLINE_SEC` | `90` | seconds | Hard watchdog on the entire orchestrator-research call. Raise for universes where individual symbols pull a lot of evidence; lower to make hung upstreams fail faster. |
| `CENTS_API_TIMEOUT` | `10` | seconds | Per-request timeout on FMP, Alpaca, FRED, NewsAPI HTTP calls. |
| `CENTS_SCAN_THRESHOLD` | `5.0` | conviction-delta points | Threshold for `cents scan` alerts. |
| `CENTS_OUTPUT_FORMAT` | `text` | `text` \| `json` | Switch all CLI output to JSON for machine consumption. |

**Caps** — pre-call enforcement against runaway spend (see [scheduling docs](https://dollars-and-cents.ai/scheduling/#cost-cap-discipline) for the daily-vs-per-run split):

| Variable | Default | Unit | When to override |
|---|---|---|---|
| `CENTS_MAX_LLM_SPEND_USD_PER_DAY` | unset (disabled) | USD | Daily ceiling across ALL cents processes. Pre-flight estimate sums today's `llm_usage` rows + the projected next call; raises `CostCapExceeded` before the API call. Pair with the per-run `--max-cost-usd` CLI flag on `cents factory run`. |

**Behaviour flags** — change what cents does, not just how fast:

| Variable | Default | When to override |
|---|---|---|
| `CENTS_DISABLE_CACHE` | unset (cache on) | Set to `1` / `true` to bypass the `api_cache` table entirely. Useful when debugging stale upstream responses. |
| `CENTS_WEBHOOK_URL` | unset | URL to POST alerts to (Slack-compatible payload). |
| `CENTS_SCREENER_ALLOW_FULL_UNIVERSE` | unset (denied) | Set to `1` to allow a SCREENER-sourced universe without `--over <parent>`. Off by default to prevent accidental 5,000-symbol scans. |
| `CENTS_FETCH_FORWARD_ESTIMATES` | unset (off) | Set to `1` to enable forward P/E lookups via FMP analyst-estimates. Adds API cost; off by default for repro stability. |
| `CENTS_ALLOW_LIVE_TRADING` | unset (denied) | Required (set to `1`) to even attempt non-paper Alpaca trading. **Real-money trading is explicitly out of scope** — see [scope](https://dollars-and-cents.ai/scope/). |
| `CENTS_LIVE_TRADING_ACK` | unset | Must contain the verbatim acknowledgement phrase to pair with `CENTS_ALLOW_LIVE_TRADING`. Both must be set; either alone is rejected. |

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
