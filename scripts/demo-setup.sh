#!/usr/bin/env bash
# Pre-populate a clean demo database with real factory output, so the
# asciinema recording (scripts/demo.sh) only has to *display* results
# instead of waiting on real API calls. Run this once before recording.
#
#   bash scripts/demo-setup.sh
#
# Produces /tmp/cents-demo.db + /tmp/cents-demo.toml. The recording
# script sources the same env vars.

set -euo pipefail

export CENTS_DB_PATH=/tmp/cents-demo.db
export CENTS_FACTORY_CONFIG=/tmp/cents-demo.toml

rm -f "$CENTS_DB_PATH" "$CENTS_FACTORY_CONFIG"

echo "[setup] Writing factory config..."
cents factory init >/dev/null
cat > "$CENTS_FACTORY_CONFIG" <<'EOF'
universe = "default"
budget_usd = 100000.0
target_positions = 20
entry_threshold = 3.0
preemption_margin = 5.0
cohort_mode = "paired"
default_horizon_days = 30
default_stop_pct = -5.0
default_target_pct = 10.0
max_new_per_run = 5
max_per_premise_tag = 2
EOF

echo "[setup] Creating universe..."
cents universe create demo --source static \
  --symbols NVDA,AMD,JPM,LLY,XOM >/dev/null
cents universe set-default demo >/dev/null

echo "[setup] Refreshing events (Federal Register + LLM tagging)..."
cents event refresh

echo "[setup] Running factory (real orchestrator + premise classification)..."
cents factory run

echo ""
echo "[setup] Demo DB ready at $CENTS_DB_PATH"
echo ""
echo "Now record the asciinema cast with:"
echo ""
echo "  asciinema rec website/public/demo.cast --command 'bash scripts/demo.sh'"
echo ""
echo "Or run the demo script live during a regular asciinema rec session."
