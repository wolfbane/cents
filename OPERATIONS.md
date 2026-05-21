# OPERATIONS.md

Operational runbook for running cents pilots and experiments. Read this when
you are (a) setting up a Mac mini for the first time, (b) running daily
health checks, or (c) diagnosing a missed run. Architecture and agent
internals live in `CLAUDE.md`; pre-registered experiment specs live in
`experiments/`; the website's `scheduling.mdx` covers the public-facing
deployment recipe. This doc is the *operator's* view — what to run, in what
order, on which machine, and what to do when something breaks.

## Deployment topology

| Machine | Role | What lives here |
|---|---|---|
| Laptop | Development | Source edits, worktrees, ad-hoc `cents` invocations against a scratch dataset, eval baseline locking. Pilot factory runs **never** tick here. |
| Mac mini (macOS 26, always plugged in) | Pilot host | The single source of truth for the pilot's labeled outcomes. launchd ticks the factory + event refresh + eval drift jobs. SQLite database, evidence trace, llm_call blobs, and eval history all live on the mini. |

Solo developer: same `~/.cents/config.toml` (same API keys) on both machines.
Same git remote. The mini gets a fresh `git clone` per deployment (not a
worktree copy from the laptop — that path resolution is brittle).

**Why the split matters.** If both machines tick the factory on the same
experiment, two databases accumulate two disjoint cohorts of theses under
the same `experiment_id`. Hit-rate analytics merge them silently and the
result is unrecoverable contamination. The discipline is: laptop **never**
runs `cents factory run` against a registered experiment.

## First-time Mac mini setup

Prerequisites: Python 3.12+, pyenv or a usable system Python with `venv`,
git, and at least one **interactive GUI login** before running launchd jobs
(TCC consent prompts only surface in an active GUI session, never via SSH).

### 1. Clone and install

```bash
git clone <repo-url> ~/Projects/cents
cd ~/Projects/cents
python -m venv ~/.venvs/cents
source ~/.venvs/cents/bin/activate
pip install -e .
cents --version  # smoke test
```

Verify the editable install points at the cloned tree:
```bash
pip show cents | grep "Editable project location"
# expect: /Users/<you>/Projects/cents
```

A mismatch here is the source of every "ModuleNotFoundError: No module named
'cents.cli'" failure. Re-run `pip install -e .` from the right directory if
it points anywhere else.

### 2. API keys

Copy `~/.cents/config.toml` from the laptop. Confirm with:
```bash
cents status
```

You should see `✓ FMP`, `✓ Alpaca`, `✓ Anthropic`, `✓ NewsAPI`, `✓ FRED`.
The output also prints the active dataset path — useful for confirming step
4.

### 2b. (Optional) If you bulk-copied `~/.cents/` from a development machine

A development laptop's `~/.cents/` tree picks up scratch state the pilot
host doesn't want. If the whole folder came over (not just `config.toml`
+ `factory.toml`), clean it up before continuing.

**The laptop's `test.db` is the most valuable thing in the copy.** It
typically has user-research tables already cleared but external-ingested
tables (`events`, `api_cache`, `delistings`, `universes`) preserved.
Renaming it to the pilot's dataset saves hours of re-ingestion on day
one. Confirm with `sqlite3 ~/.cents/data/test.db "SELECT
(SELECT COUNT(*) FROM theses), (SELECT COUNT(*) FROM events),
(SELECT COUNT(*) FROM universes);"` — you want theses=0,
events>>0, universes>0.

```bash
# Repurpose the laptop's pre-cleaned DB as the pilot dataset.
mv ~/.cents/data/test.db ~/.cents/data/pilot_v1.db

# Drop laptop-specific scratch.
rm -f  ~/.cents/data/*.bak-*               # session backups
rm -f  ~/.cents/data/marc.db               # other laptop portfolios
rm -f  ~/.cents/data/cents.db              # default portfolio, usually empty stub
rm -rf ~/.cents/data/eval_history          # mini accumulates its own from t=0
rm -rf ~/.cents/data/llm_calls             # laptop's provenance blobs aren't pilot-relevant

# Reset the dataset selector — laptop's active="test" doesn't translate.
rm ~/.cents/datasets.toml
cents portfolio add pilot_v1 ~/.cents/data/pilot_v1.db
cents portfolio use pilot_v1

# Sanity check: the renamed DB still has its preserved external data.
cents factory status   # expect 0 open theses, schema intact
cents universe list    # expect sp100_static_v1 + any other laptop-side universes
```

If the bulk copy didn't include a pre-cleaned `test.db` (or if you'd
rather start truly fresh), skip the rename and fall through to step 4
for a green-field dataset.

### 3. Factory config

If `~/.cents/factory.toml` doesn't exist on the mini:
```bash
cents factory init
```

Then confirm the four pre-flight invariants (matched against the active
experiment spec's checklist):
- `cohort_mode = "paired"`
- `sizing_mode = "equal_dollar"`
- `beta_match_hedge = false`
- The cost cap matches the spec's daily cap (e.g. `max_llm_spend_usd_per_day = 10.0`)

### 4. Pilot dataset

The `cents` CLI doesn't read from a fixed DB path. It reads from `active`
in `~/.cents/datasets.toml`, which maps names → DB paths. **Always create a
named dataset for each pilot so the pilot's data never co-mingles with
unrelated work.**

```bash
cents portfolio add pilot_v1 ~/.cents/data/pilot_v1.db
cents portfolio use pilot_v1
cents status  # confirm "Active: pilot_v1"
```

Run any command (e.g. `cents factory status`) to trigger schema creation in
the new DB.

### 5. Universe + delistings

Either import the pilot universe from a CSV or recreate it via screener.
For the current pilot (`sp100_static_v1`), the symbol list is a
hand-curated ~100 large caps:
```bash
cents universe create sp100_static_v1 --source static --file <path-to-symbol-list>
cents universe set-default sp100_static_v1
cents universe ingest-delistings
```

`ingest-delistings` should be re-run weekly (Sunday cron handles this — see
step 7).

### 6. Eval baseline

The locked baseline travels with git at `src/cents/eval/baseline.json` —
the Mac mini's clone already has it. Seed the **history** on the mini with
a single run so drift-check has a t=0 row to compare against on day one:
```bash
cents eval run --set all --persist-history
```

Do **not** pass `--persist-baseline` here. That would overwrite the
committed baseline.json with the mini's run, which is wasteful (the
baseline is supposed to be a frozen reference). Only re-baseline after an
intentional model snapshot bump.

### 7. Scheduling (launchd, macOS 26)

Two artifacts in the repo: a wrapper script (`scripts/cents-wrap`) and
eight plist templates (`scripts/launchd/*.plist`).

Copy and adjust the wrapper:
```bash
cp scripts/cents-wrap ~/.cents/bin/cents-wrap
chmod +x ~/.cents/bin/cents-wrap
# Edit the CENTS_BIN line to point at ~/.venvs/cents/bin/cents
```

For each plist in `scripts/launchd/`:
```bash
sed "s|\$HOME|$HOME|g" scripts/launchd/<job>.plist > ~/Library/LaunchAgents/<job>.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<job>.plist
launchctl enable gui/$(id -u)/<job-label>
```

`launchctl load -w` still works on macOS 26 but is deprecated; `bootstrap`
+ `enable` is the supported path.

**Eight jobs** (matching `website/src/content/docs/scheduling.mdx`):

| Time (ET) | Job | Plist | Command |
|---|---|---|---|
| 06:00 Mon-Fri | event-refresh | `event-refresh.plist` | `event refresh` |
| 06:30 Mon-Fri | factory-llm | `factory-llm.plist` | `factory run --max-cost-usd 10` |
| 06:35 Mon-Fri | factory-random | `factory-random.plist` | `factory run --orchestrator random --orchestrator-seed __DATE_SECONDS__ --max-cost-usd 10` |
| 07:00 Mon-Fri | shadow | `shadow.plist` | `shadow backfill --horizon 30` |
| 04:00 Sun | delistings | `delistings.plist` | `universe ingest-delistings` |
| 18:00 Mon-Fri | status | `status.plist` | `experiment status --output json` |
| 18:30 Mon-Fri | eval-run | `eval-run.plist` | `eval run --persist-history` |
| 18:35 Mon-Fri | eval-drift | `eval-drift.plist` | `eval drift-check` |

`__DATE_SECONDS__` is a sentinel the wrapper substitutes with
`$(date +%s)` at exec time. Launchd's `ProgramArguments` array doesn't
run shell expansion — the substitution has to happen inside the wrapper.
Fresh seed per day = fresh universe shuffle per day, matching the LLM
arm's `Random(run_id)` shuffle. Don't replace this with a fixed seed: a
fixed seed produces the same symbol order every day, a subtle but real
non-match between arms.

### 8. Smoke test

Run each wrapper interactively before letting launchd tick. This surfaces
macOS 26 TCC / Background Task consent prompts in the GUI; SSH sessions
hide them and the first launchd-triggered run will silently fail with a
sandbox denial otherwise.

```bash
~/.cents/bin/cents-wrap factory-llm-smoke factory run --dry-run --max-cost-usd 1
~/.cents/bin/cents-wrap event-refresh-smoke event refresh
~/.cents/bin/cents-wrap eval-smoke eval run --set premise --limit 10
```

Approve any prompts in System Settings → Privacy & Security. If a job
needs Full Disk Access (the wrapper reads `~/.cents/config.toml`), grant
it to `/bin/bash` and to the cents binary.

### 9. Register the experiment

Walk the 12-item pre-flight checklist in `experiments/<experiment>.yaml`
end-to-end. Then:
```bash
cents experiment register experiments/pilot_v1.yaml
cents experiment list  # confirm registered
cents experiment status pilot_v1  # confirm 0 opens, frozen factory.toml SHA
```

From this moment forward, every thesis opened gets stamped with
`experiment_id = pilot_v1`. The factory.toml SHA is frozen — do not edit
factory.toml during the pilot (the engine writes a `CONFIG_DRIFT` alert if
the SHA changes between registration and a run).

## Daily cadence

Once launchd is ticking and the experiment is registered, the operator's
day-to-day responsibility is light: spot-check that the cron is firing,
that costs are tracking, and that no alerts have fired.

### Five-second health check
```bash
cents experiment status                # N per arm, calendar days
cents usage headroom                   # cap headroom for today
cents factory status                   # open theses, last run
cents alert list --since today         # invalidations, drift, cost trips
```

### Per-day log archaeology
Logs land in `~/.cents/logs/YYYY-MM-DD/<job>.log`. Each scheduled run
writes to its own file; an empty log file for a scheduled hour means the
job didn't fire (sleep, TCC denial, launchd unloaded, etc.). Cross-check
against:
```bash
launchctl print gui/$(id -u)/ai.dollars-and-cents.factory-llm
# look for State, LastExitStatus, LastRunCompletedAt
```

### Weekly verification
- Friday end-of-week: run `cents factory analyze --by orchestrator
  --include-cost-per-outcome` against the in-flight cohort to eyeball the
  trajectory. **Do not finalize early** based on this — the pre-registered
  stopping rule is "later of N AND calendar days," and the decision tree
  is pre-committed in the spec.
- Sunday morning: confirm `delistings` cron fired (`ls
  ~/.cents/logs/$(date +%Y-%m-%d)/delistings.log`).

## Failure modes

### Editable install drift
- **Symptom:** `ModuleNotFoundError: No module named 'cents.cli'` when running any subcommand, despite `cents --version` working earlier.
- **Cause:** `pip show cents` reports an `Editable project location` that no longer exists (typically a removed worktree).
- **Fix:** `cd ~/Projects/cents && pip install -e .` from the canonical checkout.
- **Verify:** `pip show cents | grep "Editable project location"` matches the current `pwd`.

### Active-dataset surprise
- **Symptom:** `cents universe list` shows expected universes, but `sqlite3 ~/.cents/data/cents.db "SELECT ... FROM universes"` returns nothing.
- **Cause:** `~/.cents/datasets.toml` `active` field points at a different DB. CLI reads from the active dataset; raw sqlite queries hit whatever path you typed.
- **Fix:** `cents status` shows the active path. Always copy-paste from there for ad-hoc sqlite work.

### TCC sandbox denial
- **Symptom:** launchd job exits silently or with sandbox errors visible in `Console.app` (filter for `cents` or the wrapper path).
- **Cause:** macOS 26 demands explicit consent for background tasks reading user data; SSH-triggered first runs can't surface the prompt.
- **Fix:** Log into the mini's GUI, run the wrapper interactively (`~/.cents/bin/cents-wrap <job> <command>`), approve System Settings prompts. Then `launchctl kickstart -k gui/$(id -u)/<Label>` to re-trigger.

### Missed run after sleep
- **Symptom:** No entry in `~/.cents/logs/$(date +%Y-%m-%d)/<job>.log` for a scheduled time.
- **Cause:** Machine slept past the schedule, external display disconnected and triggered suspend, or the machine restarted and launchd hasn't bootstrapped the job.
- **Fix:** System Settings → Energy → "Prevent automatic sleeping when the display is off" → on. After reboot, confirm `launchctl print gui/$(id -u)/<Label>` shows the job loaded.

### Cost cap trip mid-run
- **Symptom:** `factory_runs.error = "cost_cap_exceeded"` for the day. Partial open phase may have persisted; close phase usually completed before the trip.
- **Cause:** Actual token usage outpaced the per-call estimate, typically due to a long tail of large sentiment payloads.
- **Fix:** **Do not raise the cap mid-pilot** — the spec's early-stop guards forbid it. Investigate token distribution via `cents usage summary --by agent`; if the estimate is systematically wrong, document and abort per the cost-overrun stopping rule.

### Model drift alert
- **Symptom:** `cents alert list` shows `MODEL_DRIFT`, indicating today's eval F1 fell >5pp below the trailing-7 median.
- **Cause:** Anthropic snapshot rolled, the eval saw an anomalous fixture set, or genuine model regression.
- **Fix:** **Do not auto-rebaseline.** Pause the pilot, inspect the per-fixture drop (`cents eval run --output json --limit 20`), and decide whether to (a) wait it out, (b) intentionally re-baseline against a documented snapshot bump, or (c) abort. The drift detector exists so this decision is conscious, not silent.

### Config drift mid-pilot
- **Symptom:** `cents alert list` shows `CONFIG_DRIFT`.
- **Cause:** `factory.toml` was edited after experiment registration, so the SHA recorded on the experiment row no longer matches the current file.
- **Fix:** **Revert factory.toml** to the registered SHA (recoverable from the experiment row's `factory_toml_body` field). The engine refuses to open new theses until the SHA matches again. If the edit was intentional and necessary, abort the experiment and re-register with the new SHA.

## End of pilot

Once `cents experiment status pilot_v1` reports both gates cleared (N ≥
spec minimum AND elapsed days ≥ spec floor):

1. **Pull the verdict table:**
   ```bash
   cents factory analyze --by cohort,orchestrator --include-cost-per-outcome > verdict.json
   ```
2. **Apply the decision tree** from the spec's footer. Pre-committed before
   data was seen — do not improvise.
3. **Lock the outcome:**
   ```bash
   cents experiment finalize pilot_v1 --verdict verdict.json
   ```
4. **Disable launchd jobs** (don't delete; the next experiment will reuse
   them):
   ```bash
   for label in factory-llm factory-random event-refresh shadow delistings status eval-run eval-drift; do
     launchctl disable gui/$(id -u)/ai.dollars-and-cents.$label
   done
   ```
5. **Archive the pilot dataset:**
   ```bash
   cp ~/.cents/data/pilot_v1.db ~/.cents/data/archive/pilot_v1-$(date +%Y%m%d).db
   ```

If the verdict is **PROCEED**, register the follow-on experiment, switch
the active portfolio to its dataset, re-enable launchd jobs, and resume.

## See also

- `CLAUDE.md` — architecture, agents, premise tags, factory.toml semantics, the things that aren't obvious from a single file.
- `experiments/pilot_v1.yaml` — current pre-registered spec, pre-flight checklist, decision tree.
- `website/src/content/docs/scheduling.mdx` — public-facing scheduling recipe (the `scripts/launchd/` plists derive from this).
- `scripts/cents-wrap` and `scripts/launchd/*.plist` — versioned templates, the source of truth for the Mac mini setup.
