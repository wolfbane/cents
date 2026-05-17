"""Anthropic pricing table — derives USD cost from token counts at report time.

Storing tokens raw (in `llm_usage`) and converting at read time avoids the need
to backfill rows when Anthropic adjusts rates.

Pricing source: Anthropic published rates, captured 2026-Q1. Rates are
expressed per million tokens. Revisit if Anthropic changes pricing tiers or
introduces new caching dimensions.
"""

from __future__ import annotations


# USD per 1M tokens. Keys are canonical model family identifiers; the lookup
# matches by prefix so dated snapshots like "claude-haiku-4-5-20251001" resolve
# to the same family.
_PRICES_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
}


def _resolve_family(model: str) -> str | None:
    """Map a model identifier (possibly with a date suffix) to a canonical family key."""
    if not model:
        return None
    # Prefer exact match first.
    if model in _PRICES_PER_MILLION:
        return model
    # Fall back to longest prefix match — `claude-haiku-4-5-20251001` should
    # resolve to `claude-haiku-4-5`, not to a hypothetical `claude-haiku-4`.
    candidates = [key for key in _PRICES_PER_MILLION if model.startswith(key)]
    if not candidates:
        return None
    return max(candidates, key=len)


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float | None:
    """Estimate USD cost for an Anthropic API call.

    Returns None if `model` isn't recognized — callers can render "-" in tables
    rather than silently treating it as $0.
    """
    family = _resolve_family(model)
    if family is None:
        return None
    rates = _PRICES_PER_MILLION[family]
    per_million = 1_000_000.0
    return (
        (input_tokens * rates["input"]) / per_million
        + (output_tokens * rates["output"]) / per_million
        + (cache_read * rates["cache_read"]) / per_million
        + (cache_write * rates["cache_write"]) / per_million
    )
