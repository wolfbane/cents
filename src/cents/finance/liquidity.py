"""Liquidity + short-borrow gates for the factory open phase.

The Risk reviewer flagged that the factory will open ``position_size_usd``
of a $20M-cap name as cheerfully as AAPL, and that shorts are opened with
no borrow check at all. This module supplies a minimum-ADV gate and a
synthetic borrow gate. The borrow gate always passes in paper mode but
attaches a synthetic borrow_rate_pa figure so the cost model can apply
borrow carry — which is the part that actually shows up in P&L.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiquidityCheck:
    """Outcome of a liquidity probe."""

    symbol: str
    passes: bool
    adv_dollars: float | None
    required_dollars: float
    reason: str | None = None


@dataclass(frozen=True)
class BorrowCheck:
    """Outcome of a (synthetic) borrow probe for a short open."""

    symbol: str
    passes: bool
    borrow_rate_pa_pct: float
    reason: str | None = None


def average_daily_volume(closes: list[float], volumes: list[int], *, lookback: int = 20) -> float | None:
    """Return median(close * volume) over the lookback window in dollars.

    Median rather than mean to dampen single-day blowoffs that overstate
    ongoing liquidity. None when insufficient data.
    """
    if len(closes) < lookback or len(volumes) < lookback:
        return None
    series = list(zip(closes[-lookback:], volumes[-lookback:]))
    dollar_volume = [c * v for c, v in series if c > 0 and v > 0]
    if not dollar_volume:
        return None
    return float(median(dollar_volume))


def passes_liquidity_gate(
    *,
    symbol: str,
    position_size_usd: float,
    closes: list[float] | None,
    volumes: list[int] | None,
    adv_multiple: float,
    lookback: int = 20,
) -> LiquidityCheck:
    """True when position size is small relative to median dollar ADV.

    Conservative posture: if we can't compute ADV, gate FAILS — better to
    skip a symbol than open into something we can't characterize.
    """
    required = position_size_usd * adv_multiple
    if closes is None or volumes is None:
        return LiquidityCheck(symbol, False, None, required, "no price history available")
    adv = average_daily_volume(closes, volumes, lookback=lookback)
    if adv is None:
        return LiquidityCheck(symbol, False, None, required, "insufficient bars for ADV")
    if adv < required:
        return LiquidityCheck(
            symbol, False, adv, required,
            f"ADV ${adv:,.0f} < required {adv_multiple}x position = ${required:,.0f}",
        )
    return LiquidityCheck(symbol, True, adv, required)


def passes_borrow_gate(
    *,
    symbol: str,
    side: str,
    default_borrow_rate_pa_pct: float,
) -> BorrowCheck:
    """Synthetic borrow check for a short open.

    In paper mode this always passes — but the returned ``borrow_rate_pa_pct``
    flows into the cost model so realized P&L for shorts includes borrow carry.
    A future live-mode implementation would replace this with a real locate
    + HTB-rate lookup; for now it documents the assumption explicitly.
    """
    if side != "short":
        return BorrowCheck(symbol, True, 0.0)
    # Future: real Alpaca / IBKR locate check goes here. Paper: pass.
    logger.debug(
        "Synthetic borrow check for %s @ %.2f%%pa (paper mode)",
        symbol, default_borrow_rate_pa_pct,
    )
    return BorrowCheck(symbol, True, default_borrow_rate_pa_pct)
