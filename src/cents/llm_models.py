"""Centralised Anthropic model snapshot identifiers.

Three agents (sentiment, event, premise) plus the eval harness all run the
same Haiku snapshot. Hardcoding the dated string in each module turns
snapshot retirement into a 4-place grep-and-replace, and the eval harness
was already importing the constant across packages with a `noqa: WPS437`
to dedup. Pinning the string here gives the snapshot a single source of
truth and keeps the per-call provenance hash stable across the codebase.

Adding a new snapshot is one line here; rolling forward (e.g. Haiku 4.6)
becomes a controlled, reviewable change instead of an inconsistent edit
spread across several agents.
"""

from __future__ import annotations

# Light-weight tagging / classification model used by the LLM-tagged
# sentiment, event, and premise-classifier paths. All three are dated
# snapshots — Anthropic retires snapshots periodically, and the per-call
# blob log captures whichever model_snapshot was used at call time.
HAIKU_TAGGING: str = "claude-haiku-4-5-20251001"
