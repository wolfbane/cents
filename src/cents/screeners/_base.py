"""Shared helpers for screener implementations."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from cents.agents.base import RECOVERABLE_EXCEPTIONS

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 30

T = TypeVar("T")


def safe_per_symbol(
    fn: Callable[[str], T | None], symbol: str
) -> T | None:
    """Run ``fn(symbol)`` and swallow recoverable exceptions so one bad symbol can't fail a screen.

    Returns the function's result, or None on a recoverable exception (the
    screener treats None as "skip this symbol"). Programming errors
    (AssertionError, etc.) propagate so they surface in tests.
    """
    try:
        return fn(symbol)
    except RECOVERABLE_EXCEPTIONS as exc:
        logger.debug("Screener skipping %s: %s", symbol, exc)
        return None


def run_per_symbol_screen(
    score_fn: Callable[[str], float | None],
    candidates: list[str] | None,
    limit: int,
) -> list[str]:
    """Boilerplate-free per-symbol screen: score each, rank, truncate.

    Identical across every screener — the differentiation is in ``score_fn``.
    ``candidates=None`` is treated the same as an empty list (no symbols to
    score), which lets each screener define its own default-candidates path.
    """
    scored: list[tuple[str, float]] = []
    for symbol in candidates or []:
        score = safe_per_symbol(score_fn, symbol)
        if score is not None:
            scored.append((symbol, score))
    return rank_and_limit(scored, limit)


def _get_fundamentals_provider():
    """Lazy import — keeps Alpaca/FMP optional at module import time."""
    from cents.data.fmp import get_fundamentals_provider
    return get_fundamentals_provider()


def _get_price_provider():
    """Lazy import — keeps Alpaca optional at module import time."""
    from cents.data.alpaca import get_price_provider
    return get_price_provider()


def rank_and_limit(
    scored: list[tuple[str, float]],
    limit: int,
) -> list[str]:
    """Sort (symbol, score) tuples by score desc (then symbol asc), keep top ``limit``."""
    ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
    return [sym for sym, _ in ranked[:limit]]
