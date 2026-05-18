#!/usr/bin/env bash
# The asciinema recording script. Pre-populated state comes from
# scripts/demo-setup.sh — run that once first.
#
# Recording:
#   asciinema rec website/public/demo.cast --command 'bash scripts/demo.sh'
#
# Then commit website/public/demo.cast and push — the homepage player
# will pick it up automatically.

set -euo pipefail

export CENTS_DB_PATH=/tmp/cents-demo.db
export CENTS_FACTORY_CONFIG=/tmp/cents-demo.toml

# Clean prompt + colors off so the recording is portable
export PS1='$ '
export TERM=xterm-256color
export NO_COLOR=1

# --- timing knobs (tune for your typing-speed aesthetic) -----------------
TYPE_DELAY=0.035    # seconds between characters when "typing"
PAUSE_BEFORE_RUN=0.4    # pause after finishing a typed command, before exec
PAUSE_AFTER_OUTPUT=1.8  # pause after output, before next command
SECTION_PAUSE=2.5       # longer pause between narrative beats

# Print a command character-by-character then execute it. Mimics live typing.
type_cmd() {
  local cmd="$1"
  printf '$ '
  local i
  for ((i=0; i<${#cmd}; i++)); do
    printf "%s" "${cmd:$i:1}"
    sleep "$TYPE_DELAY"
  done
  printf '\n'
  sleep "$PAUSE_BEFORE_RUN"
  eval "$cmd" || true
  echo ''
  sleep "$PAUSE_AFTER_OUTPUT"
}

# Print a "comment" line — looks like the recorder is narrating in-terminal.
say() {
  printf '\033[2m# %s\033[0m\n' "$1"
  sleep 1.0
}

clear

# --- Act 1: the universe ---------------------------------------------------
say "What's in scope — five names across sectors."
type_cmd "cents universe show demo"

sleep "$SECTION_PAUSE"

# --- Act 2: research a single symbol --------------------------------------
say "Research fans out across seven specialised agents and synthesises a single conviction."
type_cmd "cents research NVDA 2>&1 | head -5"

sleep "$SECTION_PAUSE"

# --- Act 3: the factory's rules + what they produced ----------------------
say "The factory runs under a config you control..."
type_cmd "cat \$CENTS_FACTORY_CONFIG"

say "...and opens paper theses across the universe wherever signal clears your threshold."
type_cmd "cents factory status"

sleep "$SECTION_PAUSE"

# --- Act 4: what a thesis looks like --------------------------------------
say "Each thesis is paired-neutral, premise-tagged, regime-snapshotted."
# Pick the highest-conviction thesis so the regime / premise output is meaningful.
DEMO_THESIS=$(sqlite3 "$CENTS_DB_PATH" "SELECT id FROM theses ORDER BY conviction DESC LIMIT 1;")
type_cmd "cents thesis show $DEMO_THESIS"

sleep "$SECTION_PAUSE"

# --- Act 5: regime invalidation -------------------------------------------
say "EventAgent ingests Federal Register events; tags intersect with open theses to fire premise-invalidation alerts."
type_cmd "cents event list --since-days 7 --limit 5"

sleep "$SECTION_PAUSE"

# --- Act 6: cost is visible -----------------------------------------------
say "Every Anthropic call is recorded — autonomy with a visible bill."
type_cmd "cents usage summary --by operation"

sleep "$SECTION_PAUSE"

say "github.com/wolfbane/cents"
sleep 3
