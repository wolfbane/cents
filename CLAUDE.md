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
cents alert list --since today              # also accepts ISO date or Nh/Nd
cents alert digest --since 24h              # per-type summary for scheduled-run logs (launchd: alert-digest.plist)

# Regime-aware substrate
cents event refresh                              # pull Federal Register events, fire PREMISE_INVALIDATION alerts
cents event list --tag tariffs.china
cents cohort                                     # per-cohort spread P&L

# Autonomous loop
cents universe create my_value --source screener --strategy value --over sp500
cents universe set-default my_value
cents factory init
cents factory run --dry-run
cents factory run                              # LLM arm (the real multi-agent stack)
cents factory run --orchestrator random        # control arm (uniform conviction_delta)
cents factory run --orchestrator random --orchestrator-seed 42   # reproducible control arm
cents factory run --max-cost-usd 5.00          # abort if cumulative LLM spend would exceed this
cents factory status
cents factory analyze --by discovery,cohort,regime,orchestrator,hedge_basis
cents factory funnel --since-days 30   # per-arm rejection funnel + cross-arm tag-cap crowding (run shadow backfill first)

# Pre-registered experiments (makes the pipeline falsifiable)
cents experiment register <spec.yaml>   # freezes factory.toml SHA; stamps every opened thesis
cents experiment list
cents experiment status pilot_v1        # progress against minimum_n_per_arm (NAME also accepts --name)
cents experiment finalize <name> --verdict verdict.json

# Cost tracking + reproducibility
cents usage summary --by agent
cents evidence trace <evidence_id>      # reconstruct the original LLM call from prompt/output hashes

# Calibration (Layer 2 #3)
cents calibration refit                          # fit logistic regression on closed-thesis outcomes
cents calibration refit --holdout-pct 0.2        # honest generalisation metrics via held-out split
cents calibration report                         # coefficients + Brier + AUC + reliability buckets

# Evals (Layer 2 #4)
cents eval golden show --set premise
cents eval run --set all                # runs the live API against golden fixtures
cents eval run --persist-history        # append today's metrics to ~/.cents/data/eval_history/YYYY-MM-DD.jsonl
cents eval run --gate --baseline-f1 0.85 --baseline-brier 0.18 --tolerance-pp 3   # CI gate
cents eval drift-check                  # fires MODEL_DRIFT alert if F1 falls >5pp below trailing-7 median

# Signal output (NOT advice — see /scope/)
cents recommend NVDA                    # emits bullish_signal / bearish_signal / neutral_signal
```

Use `--output json` for machine-readable output. Run `cents --help` or `cents <command> --help` for full options.

For running the factory + event refresh + shadow backfill on a daily cadence (cron / launchd recipes, cost-cap discipline, editable-install drift), see the [Scheduling page](https://dollars-and-cents.ai/scheduling/) — the docs surface for the 90-day forward test.

## Deployment topology

Pilots and registered experiments run on a **dedicated Mac mini** (macOS 26,
always plugged in), not on the development laptop. The mini owns the
single source of truth for the pilot's labeled outcomes dataset; the
laptop is used for code edits, worktrees, ad-hoc CLI invocations against a
scratch portfolio, and eval baseline locking. Solo developer, same
`~/.cents/config.toml` (same API keys) on both machines.

**The laptop must never run `cents factory run` against a registered
experiment.** Two machines ticking the same experiment_id produces two
disjoint cohorts that merge silently into hit-rate analytics — unrecoverable
contamination. Use `cents portfolio` to swap to a scratch dataset on the
laptop before any ad-hoc factory experimentation.

`OPERATIONS.md` covers first-time mini setup, the launchd recipe, daily
cadence, health checks, and failure-mode recovery. Read it before standing
up a new pilot machine or diagnosing a missed scheduled run.

## Build & Test

```bash
pip install -e ".[dev]"       # Install with test deps
pip install -e ".[broker]"    # Add Alpaca trading (paper only — see /scope/)
pytest                        # Full suite (~990 tests, ~65s)
pytest tests/test_factory.py  # Single file
pytest -k "premise"           # By keyword
pytest --lf                   # Re-run last failures
```

A reinstall (`pip install -e .`) is required after switching the working tree between a worktree-built feature and the main checkout, otherwise `cents` may dispatch to stale installed code (symptom + diagnosis below in "things that aren't obvious").

## High-level architecture

cents is a **research pipeline** for studying whether multi-agent LLM orchestration produces a calibrated signal on forward equity returns. **It is not a trading tool.** The factory engine RECORDS what happens to a labeled outcomes dataset; it does not FILTER or GATE on trading-style controls. `cents/finance/*` modules exist as utilities for callers writing their own analytics — the engine's default behaviour doesn't use them as gates.

cents is structured as **four cooperating layers** + a transversal **finance substrate** + an **experiments registry** that pre-registers hypotheses against a frozen factory.toml SHA. The discovery → evaluation → invalidation → analytics path is the load-bearing flow.

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
   event's tags intersect an open thesis's premise_tags AND the event's
   polarity opposes the thesis's premise_direction on the shared tag.

4. Autonomous loop + analytics
   FactoryEngine (cents/factory/engine.py) walks a universe, runs the
   close phase (target/stop/expiry/INVALIDATED/PREEMPTED), then open phase
   (entry threshold → premise classification (with direction) → per-tag
   concentration cap → budget / conviction-weighted preemption → open).
   Records discovery_source + regime_snapshot + calibrated_p_correct +
   premise_direction on every thesis. **The engine records, it does not
   gate** — drawdown, liquidity, borrow, and calibrated-p are computed
   and stored but never block an open. The point is the labeled outcomes
   dataset, not trading controls.
   cents factory analyze --by {cohort,discovery,regime} stratifies outcomes.

Transversal: cents/finance/ — UTILITIES, not gates
   These modules exist so analytics can stratify outcomes and so callers
   writing their own pipelines can opt into trading-shaped behaviour.
   The default engine config does not use them as gating decisions.
   - sizing.py: vol_scaled_shares (opt-in via sizing_mode="vol_scaled";
     default is equal_dollar)
   - costs.py: apply_open_cost / apply_close_cost — applied so cohort
     numbers are net of realistic frictions (research honesty, not gating)
   - hedging.py: estimate_beta + beta_match_ratio — opt-in via
     beta_match_hedge=true; default is equal-dollar hedge match
   - liquidity.py: passes_liquidity_gate, passes_borrow_gate — utilities;
     the engine never skips on them
   - portfolio.py: compute_drawdown + check_kill_switch — utilities;
     the engine never halts on them
   - calibration.py: CalibrationModel + fit_calibration — used to record
     calibrated_p_correct on every thesis; engine never skips on it
```

### Things that aren't obvious from a single file

- **The pipeline has a two-arm control design.** `cents factory run` defaults to `--orchestrator llm` (the real multi-agent stack). `--orchestrator random` (`cents/agents/random_orchestrator.py`) is the matched-cadence control — uniform `conviction_delta` in `[-30, +30]`, no agent-stack LLM calls, `orchestrator_label = "random"` on every opened thesis. The paired-neutral cohort is a control for *regime beta*; the random arm is a control for *signal value*. `Thesis.orchestrator_label` + `Thesis.experiment_id` are the cohort-analytics columns that make this falsifiable. (The factory open phase still issues one small `classify_premise` LLM call per opened random-arm thesis — see the PREMISE_INVALIDATION note below. Sentiment / fundamentals / macro / etc. agents do NOT run for random theses.)
- **Experiments are pre-registered, not post-hoc.** `cents experiment register <spec.yaml>` writes an Experiment row with the current factory.toml SHA + body, frozen at registration time. Spec is `{name, hypothesis, primary_metric, minimum_n_per_arm, stopping_rule}`. The factory engine stamps `experiment_id` on every thesis opened while the experiment is active. `cents experiment status` shows progress against `minimum_n_per_arm`; `cents experiment finalize <name>` locks it. See `cents/experiments/registry.py`.
- **The orchestrator's `AgentResult` has a different clamp from individual agents.** Per-agent `conviction_delta` clamps to ±10 (`MAX_CONVICTION_DELTA`); the orchestrator's aggregate (constructed with `aggregate=True`) clamps to ±30 (`MAX_AGGREGATE_CONVICTION_DELTA`) so strong consensus isn't quantized to ±10. See `cents/agents/base.py`.
- **Premise tags are a controlled vocabulary** (`EVENT_TAGS` in `cents/models/event.py`). Both the EventAgent (tagging fetched events) and the premise classifier (`cents/factory/premise.py`) draw from this single list. **Adding tags is safe; renaming them is not.**
- **Premise invalidation is polarity-aware.** `Event.matches_premise(tags, direction)` requires tag overlap, and for BULLISH/BEARISH events the polarity must oppose the thesis's `premise_direction` on a shared tag (a bullish event on a "positive"-direction thesis confirms, doesn't invalidate). NEUTRAL/UNCLEAR events fall back to legacy unsigned intersection — they DO invalidate when tags overlap. The fail-open choice is deliberate: an ambiguous-polarity tariff event with a shared tag should not silently fail to alert a tariff-dependent thesis. Empty `premise_direction` (legacy theses) also falls back to unsigned intersection.
- **`classify_premise_tags` returns a 2-tuple `(tags, direction)`** — `direction` is `{tag: "positive"|"negative"}`. The factory engine uses `_coerce_premise_classification` to accept legacy bare-list stubs from older tests.
- **A neutral-cohort thesis owns BOTH legs** as two `Position` rows on the same `thesis_id` (one LONG on `symbol`, one SHORT on `hedge_symbol`). It is NOT two linked theses. Closing the thesis closes both legs naturally.
- **The hedge leg is dollar-matched by default.** `beta_match_hedge=false` in the scaffolded TOML and the FactoryConfig default. When opted in, `cents/finance/hedging.py:estimate_beta` does 60-day OLS of log returns vs the hedge ETF, clamps to `[beta_min, beta_max]` (default `[0.10, 5.0]`), and refuses estimation when R² is below `beta_min_r_squared` (default 0.5) — in which case the engine **refuses to open the thesis at all** (shadow reason `hedge_beta_rejected`) rather than putting a non-beta-neutral pair into the NEUTRAL cohort. Likewise, a paired candidate whose hedge ETF has no price is skipped entirely (shadow reason `no_hedge_price`) — a one-legged thesis must never wear the NEUTRAL label.
- **`Thesis.hedge_basis` records how the neutral leg was sized.** `HedgeBasis.BETA` (genuine beta-matched), `DOLLAR_FALLBACK` (beta_match_hedge=true but price history was unavailable so the leg was dollar-matched — low-R² fits no longer fall back, they refuse the open), or `DOLLAR` (equal-dollar by config). Directional theses are `None` and bucket as "directional" in `factory analyze --by hedge_basis`. Stratifying neutral cohorts by basis is the only way to tell whether a "neutral" result reflects skill or a hedge that was never actually beta-neutral.
- **`Thesis.premise_classification_source` records which path produced the tags.** `PremiseSource.LLM` (classifier returned ≥1 vocabulary-mapped tag), `FALLBACK_SECTOR` (LLM produced nothing usable, sector-derived tags applied), or `FALLBACK_EMPTY` (neither path produced tags; also the default for legacy / manually-created theses). Stratify with `factory analyze` — a sustained high FALLBACK_SECTOR share on the LLM arm means the classifier is underperforming, not that the signal is bad.
- **Repository (de)serialize round-trips str-Enums as enums, not raw strings.** `hedge_basis`, `premise_classification_source`, `valuation`, `time_horizon`, `outcome`, `cohort` all bind as TEXT on write but must come back as their Enum type on read. Pattern-match callers (`match t.hedge_basis: case HedgeBasis.BETA:`) will silently miss every branch if a new Enum field skips this — string equality keeps working, masking the bug. When adding a new str-Enum field on `Thesis`, mirror the existing pattern in `ThesisRepository` (see commit `7eae145`).
- **Direction follows signal sign.** Bullish `conviction_delta` opens LONG underlying (+ SHORT hedge in paired mode); bearish opens SHORT underlying (+ LONG hedge). Target/stop semantics flip — short theses' target sits *below* entry, stop above.
- **`cohort_mode=paired` is the factory default** for measurement reasons (the neutral cohort is the control group for separating skill from regime beta). Both the scaffolded `factory init` config and `FactoryConfig` ship `paired`. The hedge leg is dollar-matched by default; flip `beta_match_hedge=true` to opt into beta-matched sizing.
- **Sizing is equal-dollar by default** — `budget_usd / target_positions` shared between primary and hedge legs. Flip `sizing_mode = "vol_scaled"` to opt into inverse-vol sizing targeting `target_vol_pct_per_position` of annualized $-volatility, capped at `max_position_pct` of budget.
- **Calibration is RECORDED, never gates an open.** `calibrated_p_correct` lands on every Thesis row when a model exists, but the engine deliberately doesn't skip opens at low p. The point is to study what actually happened at every p value — that's the research question. `cents/finance/calibration.py` is a utility.
- **`Position.pnl` is NET of costs.** `Position.gross_pnl` is the pre-cost figure. `costs_applied_usd` accumulates commission + slippage + short borrow + gap penalty across both open and close. Cohort analytics should always use `pnl`, not `gross_pnl`.
- **Stop fills are gap-aware.** When closing on a stop trigger (`ThesisOutcome.INCORRECT`), `realized_exit_price = min/max(mark, stop_price)` (worst-for-position direction) plus `gap_slippage_bps`. Position stores both the signal `exit_price` and the modeled `realized_exit_price`.
- **Sentiment LLM calls are cached per (symbol, article-set, thesis, model, day).** `cents/agents/sentiment.py:_sentiment_cache_params` keys filter + score calls on a SHA256 of sorted article URLs (`_article_set_hash`), the thesis-hypothesis hash, the model constant, and today's date. Same corpus + same thesis on the same day → cache hit, no LLM call. Add/remove one article or change the thesis hypothesis → different hash → miss. The `_day` field deliberately scopes hits to a single trading day; cross-day reuse is intentionally not allowed because article relevance decays.
- **Anthropic per-request timeout is capped at 30s.** The SDK default is 600s read-timeout which, combined with 2 retries × exponential backoff, can burn 30+ minutes on a single hung call (cents-87v repro: MCD lost 38 min mid-symbol). Every Anthropic client constructed by cents (`sentiment.py`, `event.py`, `premise.py`, `eval/runner.py`) passes `timeout=settings.anthropic_timeout_sec` (default 30s). Override via `CENTS_ANTHROPIC_TIMEOUT_SEC` env var or `anthropic_timeout_sec` in `~/.cents/config.toml`. Worst-case bound per LLM call is now ~106s (30s × 2 retries + exponential backoff up to 8s).
- **Open phase is first-fit on a per-run shuffled universe, not best-fit.** The loop in `cents/factory/engine.py:_open_phase` walks `universe_symbols` and opens the first `max_new_per_run` symbols that exceed `entry_threshold` and pass gates — it does NOT pre-score every symbol and pick the top-N. To remove systematic ordering bias (e.g. always opening A-side symbols on alphabetical static universes), the engine shuffles the universe in place using `random.Random(run_id).shuffle(...)` before iterating: each run gets a different (reproducible) order, both LLM and random arms see the same order within a run. Raising `max_new_per_run` widens coverage proportionally but does not re-rank.
- **Live (non-paper) trading is hard-gated.** `AlpacaClient(paper=False)` raises `BrokerError` unless BOTH `CENTS_ALLOW_LIVE_TRADING=1` AND `CENTS_LIVE_TRADING_ACK` matches `LIVE_TRADING_ACK_PHRASE` verbatim (27 words). All factory + CLI broker callers hard-code `paper=True`. See `/scope/` — real-money trading is explicitly out of scope.
- **LLM call provenance is reproducibility, not audit-grade.** Every Anthropic call writes a gzipped JSONL blob to `~/.cents/data/llm_calls/YYYYMMDD/<call_id>.json.gz` plus a row in `llm_usage`. Evidence rows persist `llm_call_id` + model + 3 SHA256 hashes (prompt/input/output). `cents evidence trace <id>` reconstructs the call. **Files are user-writable** — this is a research log, not Rule 204-2 recordkeeping. See `cents/llm_usage.py:persist_call_blob`.
- **Pre-flight LLM cost cap.** `cents factory run --max-cost-usd N` (per-run) and `max_llm_spend_usd_per_day` in **`~/.cents/config.toml`** (daily, NOT factory.toml — easy mistake) are enforced PRE-call via `check_cost_cap` in `cents/llm_usage.py`. The daily cap can also be set via the `CENTS_MAX_LLM_SPEND_USD_PER_DAY` env var. Estimate uses a 4-chars/token heuristic on `max_tokens` + message content; raises `CostCapExceeded` before the offending API call is made. `cents usage headroom` shows today's spend against the cap + trailing-window cap pressure.
- **Untrusted text is delimited with a per-call nonce.** `cents/agents/base.py:safe_delimit(text, tag)` wraps news article / Federal Register / thesis text in `<{tag}-{nonce}>...</{tag}-{nonce}>` and escapes literal `</{tag}` substrings. System prompts reference the nonce-tagged form.
- **`Thesis.discovery_source = <universe_name>`** is the link between the discovery layer and outcome analytics. Without it, `cents factory analyze --by discovery` has nothing to stratify on. The factory engine sets it automatically; manually-created theses leave it `None`.
- **`factory analyze` low-N flag gates on `judged`, not `opened`.** A cohort with 50 opened but only 2 closed-and-judged is still low-N. Threshold is `LOW_N_THRESHOLD = 30` in `cents/cli/_disclosures.py`.
- **The api_cache table has a per-(provider, endpoint) TTL policy.** See `TTL_DAYS_BY_ENDPOINT` in `cents/cache.py`: daily-keyed endpoints (FMP TTM ratios, profile, Alpaca `bars_split_v1`) get 7 days; quarterly historicals (FMP ratios, key-metrics) get 90; FRED observations get 365; the dead pre-split-adjust `alpaca/bars` namespace is marked TTL=0 (immediately stale on read). Endpoints with no entry never expire. Expired rows are dropped lazily on read AND in bulk by `cents cache prune` (CLI). Use `cents cache stats` to see row counts / size / age / TTL per endpoint. Daily-mutable endpoints still use `daily_key=True` in `_fetch_json` to inject today's date into the cache key — the TTL is a second layer so old day-keys age out instead of accumulating forever.
- **Alpaca bars are split-adjusted, NOT dividend-adjusted.** `cents/data/alpaca.py` passes `adjustment=Adjustment.SPLIT` so historical bars are comparable across split boundaries. Without this, NVDA's June 2024 10:1 split produced a phantom -87.7% 20d return in backtest reports. Dividends are intentionally not adjusted — that would shift absolute price levels relative to live quotes (MA20 / 52W range / target prices). The bars cache namespace is `bars_split_v1`; if the adjustment basis ever changes again, bump the namespace to invalidate stale entries rather than serving mixed-basis data.
- **Insider trades are deduplicated per-insider.** `_aggregate_by_insider` collapses rows by `reportingName` so one insider filing five 10b5-1 sale slices doesn't masquerade as cluster activity. When N>1 rows collapsed, the rendered evidence appends `(across N filings, likely 10b5-1)` so the contextual hint is visible. Without this, automatic scheduled sales programs would inflate the "cluster selling" signal.
- **EventAgent drops untagged events in no-thesis research.** Untagged events have no premise-tag intersection by construction, so they can't invalidate or score against any thesis. Returning them only adds noise to `cents research` output. They are still persisted (and may match later when theses are written), but they don't surface as evidence rows for a no-thesis research call.
- **Agent evidence rows carry fired-rule attribution.** Evidence content for `[+]` / `[-]` rows now suffixes the actual rule that fired (e.g. `Unemployment Rate 4.30 — low_level: UNRATE < 4.5%`). Without this, "falling unemployment is bullish" reads as the driver when the actual rule was the absolute level. The macro-level signal name is `low_level` (was `low_stable` — the rule never checked stability, only the level). Affects macro, technical, fundamentals, and moat agents — read the rule from the evidence text, not from the metric value.
- **Per-experiment `minimum_calendar_days`.** Experiments now carry a `minimum_calendar_days` field (default 14, back-compat alias `MINIMUM_ELAPSED_DAYS`) instead of using a module-level constant. The shipped specs are `experiments/pilot_v1.yaml` (30-day pilot, N=200/arm) and `experiments/hit_rate_delta_v1.yaml` (90-day full run, N=400/arm). Stopping rule is "later of" (N AND calendar) so neither gate can short-circuit the other. `cents experiment status` returns both `minimum_calendar_days` and the legacy alias.
- **`factory analyze --include-cost-per-outcome`** is opt-in. Adds `llm_cost_per_opened` / `llm_cost_per_judged` / `llm_cost_per_correct` to every cohort cell plus `llm_cost_total_usd`. Unattributable LLM calls (no `thesis_id` and no `symbol`-within-window match) accumulate at the top level as `unattributable_cost_usd`. Attribution rule: `llm_usage.context` matches `thesis.id` directly, OR matches `thesis.symbol` AND `called_at` falls within `thesis.created_at..closed_at`. Random-arm cells naturally pick up $0.
- **`factory analyze --by orchestrator`** stratifies by `orchestrator_label` (`llm` vs `random`). Combine with `--by cohort,orchestrator` for the 2×2 win-rate table the experiment's primary metric reads against.
- **Eval harness has a CI gate and a drift detector.** `cents eval run --gate --baseline-f1 N --baseline-brier N --tolerance-pp N` exits 2 on regression (distinct from skipped=1). `--persist-baseline` writes/updates `baseline.json` (locked_at defaults to null so a fresh install never fails a build). `cents eval run --persist-history` appends today's metrics to `~/.cents/data/eval_history/YYYY-MM-DD.jsonl`; `cents eval drift-check` reads the trailing 7 days and fires `AlertType.MODEL_DRIFT` if F1 falls >5pp below the trailing-7 median. Bootstrap CIs (1000 samples, seed=17) come for free in both text and JSON output.

- **PREMISE_INVALIDATION applies to both arms; the asymmetry is in tag-set size, not invalidatability.** Both arms hit `classify_premise_tags` in `cents/factory/engine.py:_open_phase` (lines ~858-869). The LLM arm gets 0–5 semantically-derived tags from the classifier. The random arm's thesis summary (`"random control: SYMBOL → delta=±NN.NN"`, ~35-45 chars) trips the `_SPARSE_SUMMARY_THRESHOLD = 50` check in `cents/factory/premise.py`, so the classifier returns `[]` and a sector-derived fallback (`_sector_fallback_tags`, capped at `_SECTOR_FALLBACK_TAG_CAP = 2`) kicks in — tags are looked up from the hedge ETF (e.g. `XLI → defense_spending, tariffs.universal`; `XLK → ai_capex, tariffs.china`), direction follows the thesis side. This was a deliberate fix (commit `331a26f`, 2026-05-18) for the selection-bias defect where random theses were left `premise_tags = []` and structurally un-invalidatable. The remaining asymmetry: LLM arm carries 0–5 tags vs. random's 0–2, so the LLM arm has more event-intersection surface area and a structurally higher expected per-thesis invalidation rate. Classifier *noise* (false positives) still cuts only against the LLM arm because random's tags are deterministic from sector. Worth tracking arm-by-arm INVALIDATED rates during pilots — a sustained LLM/random INVALIDATED ratio much higher than the tag-count ratio (~2.5×) suggests classifier false positives are eating disproportionate LLM-arm N. Current locked baseline (`src/cents/eval/baseline.json`) is premise F1 = 0.66 — both over- and under-predicts tags in roughly equal measure. **Cost note:** the random-arm `classify_premise` call costs ~$0.001 (sparse prompt → small token counts) and the LLM almost always returns `[]` before falling back; it's not free, but it's a rounding error against the LLM arm's per-symbol cost.
- **Model selection is a code change, not a config change.** Every Anthropic call (sentiment, event, premise classifier, eval harness) routes through a single constant in `cents/llm_models.py` (`HAIKU_TAGGING = "claude-haiku-4-5-20251001"`). There is no env var or config switch — swapping models for a new experiment means editing the constant and re-registering with a fresh factory.toml SHA. Deliberate: the per-call provenance hash chain (`prompt_sha256` / `input_sha256` / `output_sha256`) is meaningful only because the model is held constant across all evidence rows in an experiment. Mixing models silently breaks that audit.
- **Eval `baseline.json` travels with the git clone; `eval_history/*.jsonl` is per-machine user state.** `src/cents/eval/baseline.json` is a packaged (git-tracked) file — the locked F1 + Brier reference for the drift gate. A fresh clone on a new machine gets the baseline for free, which is what makes the laptop→Mac-mini handoff clean. `~/.cents/data/eval_history/YYYY-MM-DD.jsonl` is user state per machine — `cents eval drift-check` reads this trailing-7-day local history. Drift detection therefore needs ≥3 days of accumulated history on the *target* machine before it has anything to compare against; seed it with `cents eval run --persist-history` on first deploy.
- **The `cents` CLI doesn't read from a fixed DB path.** `~/.cents/datasets.toml` has an `active` field selecting which entry from the `[datasets]` map the CLI uses; managed via `cents portfolio {list,add,use,current}`. The active path is also printed by `cents status` — always copy-paste from there for ad-hoc sqlite work, because raw `sqlite3 ~/.cents/data/cents.db "..."` will silently hit the wrong DB if the active dataset points elsewhere (e.g. `test.db`, `marc.db`). The fresh-install default is `active = "default"` → `~/.cents/data/cents.db`; named portfolios are opt-in.
- **`pip install -e .` stale-worktree drift, symptom + diagnosis.** Symptom is `ModuleNotFoundError: No module named 'cents.cli'` even though `cents --version` worked earlier in the session. Cause: `pip show cents` shows an `Editable project location` that no longer exists on disk (typically a removed worktree). Cure: `pip install -e .` from the canonical checkout. The diagnostic: `pip show cents | grep "Editable project location"` should match the directory you actually want.
- **launchd plists are PT-translated from the ET schedule.** The pilot Mac mini's system TZ is Pacific (`PT`); `StartCalendarInterval` fires in system-local time, NOT in `TZ=America/New_York` (the wrapper's `TZ` env var only affects how cents *reads* time inside the process, not when launchd fires). So every `Hour` value in `scripts/launchd/*.plist` is PT = ET − 3 (e.g. `Hour=3` = 03:00 PT = 06:00 ET). If the mini's system TZ ever changes to `America/New_York`, re-translate the plists back to ET values or the schedule will be 3 hours off. US ET and PT observe DST on the same Sundays, so the 3-hour offset is constant year-round — there is no recurring drift, just the two transition Sundays per year to watch.
- **DST transition Sundays to spot-check after.** Both arms run Mon-Fri so the transitions themselves don't fire factory jobs, but `delistings` runs at 01:00 PT Sundays and could be affected: on **fall-back Sundays (1st Sunday of November)**, 02:00 PDT rewinds to 01:00 PST, so 01:00 PT happens *twice* — modern launchd usually fires once at the wall-clock match, but worth checking `~/.cents/logs/<date>/delistings.log` for a duplicate entry (the job is idempotent — it overwrites the delistings table — so two fires are safe, just noisy). On **spring-forward Sundays (2nd Sunday of March)**, 02:00 PST jumps instantly to 03:00 PDT, so 02:00–02:59 PT doesn't exist that day; nothing in our schedule falls in that window. Transition Sundays to spot-check: **2026-11-01, 2027-03-14, 2027-11-07, 2028-03-12, 2028-11-05**. On the Monday following each, eyeball that the 03:00 PT event-refresh fired (`~/.cents/logs/<monday>/event-refresh.log`).

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
