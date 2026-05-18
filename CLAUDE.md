# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Using the cents CLI

When the user asks about investing, stock research, theses, positions, watchlists, screens, or autonomous operation, **use the `cents` CLI directly rather than writing Python**. The CLI is the supported surface; the library underneath is intentionally not stable for direct consumption.

```bash
# Manual research / tracking
cents research NVDA --suggest-thesis
cents thesis create --title "..." --from-research NVDA
cents thesis twin <id> --hedge-with SOXX        # paired-neutral twin of an existing thesis
cents position open NVDA --size 100 --price 135 --thesis <id>
cents watch add NVDA --thesis <id>
cents scan
cents alert list

# Regime-aware substrate
cents event refresh                              # pull Federal Register events, fire PREMISE_INVALIDATION alerts
cents event list --tag tariffs.china
cents cohort                                     # per-cohort spread P&L

# Autonomous loop
cents universe create my_value --source screener --strategy value --over sp500
cents universe set-default my_value
cents factory init
cents factory run --dry-run
cents factory run
cents factory status
cents factory analyze --by discovery,cohort,regime

# Cost tracking
cents usage summary --by agent
```

Use `--output json` for machine-readable output. Run `cents --help` or `cents <command> --help` for full options.

## Build & Test

```bash
pip install -e ".[dev]"       # Install with test deps
pip install -e ".[broker]"    # Add Alpaca trading
pytest                        # Full suite (~560+ tests, ~20s)
pytest tests/test_factory.py  # Single file
pytest -k "premise"           # By keyword
pytest --lf                   # Re-run last failures
```

A reinstall (`pip install -e .`) is required after switching the working tree between a worktree-built feature and the main checkout, otherwise `cents` may dispatch to stale installed code.

## High-level architecture

cents is structured as **four cooperating layers**. The discovery → evaluation → invalidation → analytics path is the load-bearing flow; everything else hangs off it.

```
1. Discovery
   Screeners (cents/screeners/{value,growth,momentum,mean_reversion,insider_cluster}.py)
   → Universes (cents/models/universe.py, cents/factory/universe_resolver.py)
                pluggable sources: STATIC, WATCHLIST, FMP_INDEX, SCREENER

2. Evaluation
   Orchestrator + 7 agents (cents/agents/*.py):
     Fundamentals (FMP) · Technical (Alpaca) · Macro (FRED) ·
     Sentiment (NewsAPI + Anthropic) · Moat (FMP) · Insider (FMP) ·
     Event (Federal Register + Anthropic)
   All agents return AgentResult(evidence, conviction_delta, summary, dimension_scores, aggregate)

3. Invalidation
   EventAgent.refresh() ingests policy events tagged against EVENT_TAGS
   (cents/models/event.py) and fires AlertType.PREMISE_INVALIDATION when an
   event's tags intersect an open thesis's premise_tags.

4. Autonomous loop + analytics
   FactoryEngine (cents/factory/engine.py) walks a universe, runs the
   close phase (target/stop/expiry/INVALIDATED/PREEMPTED), then open phase
   (entry threshold → premise classification → concentration cap →
   budget / conviction-weighted preemption). Records discovery_source
   + regime_snapshot on every thesis.
   cents factory analyze --by {cohort,discovery,regime} stratifies outcomes.
```

### Things that aren't obvious from a single file

- **The orchestrator's `AgentResult` has a different clamp from individual agents.** Per-agent `conviction_delta` clamps to ±10 (`MAX_CONVICTION_DELTA`); the orchestrator's aggregate (constructed with `aggregate=True`) clamps to ±30 (`MAX_AGGREGATE_CONVICTION_DELTA`) so strong consensus isn't quantized to ±10. See `cents/agents/base.py`.
- **Premise tags are a controlled vocabulary** (`EVENT_TAGS` in `cents/models/event.py`). Both the EventAgent (tagging fetched events) and the premise classifier (`cents/factory/premise.py`) draw from this single list — that's what makes intersection matching work. **Adding tags is safe; renaming them is not.**
- **A neutral-cohort thesis owns BOTH legs** as two `Position` rows on the same `thesis_id` (one LONG on `symbol`, one SHORT on `hedge_symbol`). It is NOT two linked theses. Closing the thesis closes both legs naturally.
- **Direction follows signal sign.** Bullish `conviction_delta` opens LONG underlying (+ SHORT hedge in paired mode); bearish opens SHORT underlying (+ LONG hedge). Target/stop semantics flip — short theses' target sits *below* entry, stop above.
- **`cohort_mode=paired` is the intended factory default** for measurement reasons (the neutral cohort is the control group for separating skill from regime beta). The scaffolded `factory init` config currently lands `directional_only`; that's not the "right" default, it's the safe one for a fresh install. Most users should switch to `paired` after running `cents factory init`.
- **`Thesis.discovery_source = <universe_name>`** is the link between the discovery layer and outcome analytics. Without it, `cents factory analyze --by discovery` has nothing to stratify on. The factory engine sets it automatically; manually-created theses leave it `None`.
- **The api_cache table has no TTL.** Daily-mutable endpoints (FMP TTM ratios, profile) use `daily_key=True` in `_fetch_json` to inject today's date into the cache key. Alpaca `get_history` is keyed by `today` (or supplied `as_of`) for the same reason. Don't cache TTM data with a stable key — it will go stale.

### Repository + persistence

- SQLite at `~/.cents/data/cents.db` (override via `CENTS_DB_PATH`). Schema lives in `cents/db/schema.py` — additions must go into BOTH the `SCHEMA` constant AND `_migrate_schema` so test fixtures (which execute `SCHEMA` directly) and existing DBs stay in sync.
- Every repository accepts an optional `conn` so tests can inject an in-memory SQLite connection. See `tests/conftest.py` — the autouse `isolate_api_cache` fixture points each test at a throwaway tmp DB so cached external responses can't leak across tests.
- Every Anthropic call routes through `cents.llm_usage.record_llm_usage()`, persisting a row to `llm_usage`. `cents usage summary` reads from there. `cents.pricing.estimate_cost_usd` prices known models; unknown models return `None`.

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

Env vars override config: `FMP_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `NEWS_API_KEY`, `FRED_API_KEY`, `ANTHROPIC_API_KEY`, `CENTS_DB_PATH`, `CENTS_OUTPUT_FORMAT`, `CENTS_SCAN_THRESHOLD`, `CENTS_WEBHOOK_URL`.

Factory-specific config: `~/.cents/factory.toml` (override via `CENTS_FACTORY_CONFIG`). Scaffold via `cents factory init`.

Screener safety: a SCREENER-sourced universe without an `--over <parent>` errors out unless `CENTS_SCREENER_ALLOW_FULL_UNIVERSE=1` is set. This is intentional — prevents accidental 5,000-symbol scans.

## Setting price targets

When creating theses with `--target-price` or `--stop-price`, anchor to real data:

1. **Web-search analyst consensus** before setting targets (e.g., "SYMBOL analyst price target consensus")
2. **Use available anchors**: analyst targets, P/E × forward EPS, technical levels (52W range, MAs)
3. **When multiple values exist, present options to the user**:
   ```
   | Target | Basis |
   | $55    | Analyst consensus |
   | $70    | Analyst high |
   | $75    | 52W midpoint |

   Which target should we use?
   ```
4. **Never guess** — if no data available, ask or omit the field. Always include the current price for context.

## Beads (multi-session task tracking)

`bd` CLI runs from the repo root with `BEADS_DIR=/Users/matthew/Projects/cents/.beads` (auto-discovery is broken on this checkout; set the env var). Common operations:

```bash
bd list                              # open issues
bd create "title" -t feature -p 2 --body "..."
bd close <id> -m "fixed in <commit>"
bd dep add A B                       # A is blocked by B (B must complete first)
```

Use beads for follow-ups that span sessions or are explicitly deferred from the current work. Embed the full content (not a summary) when creating.

## Website

`website/` is an Astro Starlight site that deploys to `dollars-and-cents.ai` via a GitHub Actions workflow on push to `master`. Build locally with `cd website && bun run build`. The sample report iframe at `/agents/` is sacred — don't regenerate the NVDA demo HTML unless explicitly asked ("refresh cents" means scan only, not re-export).

The homepage "video" is actually an asciinema cast at `website/public/demo.cast` (plain-text terminal recording, not video). To re-record: run `bash scripts/demo-setup.sh` to pre-populate `/tmp/cents-demo.db`, then `asciinema rec website/public/demo.cast --command 'bash scripts/demo.sh'`. The demo script simulates typing + executes each command against the pre-populated DB so the recording stays fast.
