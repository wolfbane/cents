"""Forward-return backfill for shadow_opens.

Walks the shadow_opens table looking for rows older than `horizon_days` whose
`forward_return_*` field is NULL, then fills it in using the supplied price
provider's `get_history`. Designed to be invoked off the hot path of a factory
run (e.g., as a separate CLI invocation or scheduled job).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from cents.db import ShadowOpenRepository
from cents.models import ShadowOpen

logger = logging.getLogger(__name__)


class _PriceHistoryProvider(Protocol):
    def get_history(self, symbol: str, days: int = ..., as_of: date | None = ...): ...


@dataclass
class BackfillResult:
    """Counts surfaced by a backfill run."""

    scanned: int = 0
    filled: int = 0
    skipped_no_history: int = 0
    skipped_no_entry_price: int = 0
    skipped_too_young: int = 0


def _forward_return_column(horizon_days: int) -> str:
    """Choose which forward_return_* column to populate based on horizon."""
    return "forward_return_60d" if horizon_days >= 60 else "forward_return_30d"


def _close_on_or_after(history, target: date) -> float | None:
    """Find the earliest close on or after `target` in a PriceHistory.

    PriceHistory bars are timestamp-ordered; this returns the first close whose
    bar timestamp's date is >= target. None if no such bar exists.
    """
    if history is None:
        return None
    for bar in getattr(history, "bars", []) or []:
        ts = getattr(bar, "timestamp", None)
        if ts is None:
            continue
        bar_date = ts.date() if isinstance(ts, datetime) else ts
        if bar_date >= target:
            return float(bar.close)
    return None


def backfill_forward_returns(
    price_provider: _PriceHistoryProvider,
    *,
    horizon_days: int = 30,
    shadow_repo: ShadowOpenRepository | None = None,
    now: datetime | None = None,
) -> BackfillResult:
    """Fill forward_return_<horizon> for shadow_opens past their horizon.

    For each candidate row:
      1. Skip if created_at + horizon_days is still in the future (too young).
      2. Skip if would_be_entry_price is missing — there's nothing to compare to.
      3. Query price_provider.get_history(symbol, days=horizon_days + buffer,
         as_of=created_at + horizon_days), then pick the first close on/after
         that date.
      4. Compute (forward_price - entry) / entry, write back the column and
         `backfilled_at = now`.

    Args:
        price_provider: Object exposing PriceDataProvider.get_history.
        horizon_days: Which horizon to fill (30 or 60).
        shadow_repo: Override for testing.
        now: Override for testing.

    Returns:
        BackfillResult with per-class counts.
    """
    repo = shadow_repo or ShadowOpenRepository()
    current = now or datetime.now()

    column = _forward_return_column(horizon_days)
    pending = repo.list_pending_backfill(horizon_days=horizon_days)
    result = BackfillResult(scanned=len(pending))

    for row in pending:
        target_dt = row.created_at + timedelta(days=horizon_days)
        if target_dt > current:
            result.skipped_too_young += 1
            continue
        entry = row.would_be_entry_price
        if entry is None or entry <= 0:
            result.skipped_no_entry_price += 1
            continue

        try:
            history = price_provider.get_history(
                row.symbol,
                days=horizon_days + 10,
                as_of=target_dt.date(),
            )
        except Exception:
            logger.exception(
                "Failed to fetch history for shadow_open %s (%s)", row.id, row.symbol
            )
            history = None

        forward_price = _close_on_or_after(history, target_dt.date())
        if forward_price is None:
            result.skipped_no_history += 1
            continue

        forward_return = (forward_price - entry) / entry
        if column == "forward_return_60d":
            row.forward_return_60d = forward_return
        else:
            row.forward_return_30d = forward_return
        row.backfilled_at = current
        repo.update(row)
        result.filled += 1

    return result


__all__ = ["BackfillResult", "backfill_forward_returns"]
