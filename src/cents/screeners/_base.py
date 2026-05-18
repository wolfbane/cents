"""Shared helpers for screener implementations."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 30

T = TypeVar("T")


def safe_per_symbol(
    fn: Callable[[str], T | None], symbol: str
) -> T | None:
    """Run ``fn(symbol)`` and swallow exceptions so one bad symbol can't fail a screen.

    Returns the function's result, or None on any exception (the screener
    treats None as "skip this symbol"). Errors are logged at debug because
    a noisy WARN per skipped symbol drowns out signal.
    """
    try:
        return fn(symbol)
    except Exception as exc:
        logger.debug("Screener skipping %s: %s", symbol, exc)
        return None


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
