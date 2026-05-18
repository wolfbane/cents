"""Transaction cost model — commission, slippage, and short borrow carry.

The PM/Risk reviewers flagged that cents/factory/engine.py closes at
get_latest_price() with no bid/ask, no commission, no impact, no slippage,
and no borrow on shorts. For a 30-day horizon with 5% stop/10% target,
unmodeled costs of 30-80bps round-trip will flip the sign of any small
edge. This module gives the engine an honest cost figure to subtract.

None of this is production-grade. Slippage is a flat bps haircut, not a
market-impact curve. Borrow is a synthetic flat rate (not real HTB lookup).
But it's enough to stop the cohort tables from being optimistic by default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cost:
    """All-in cost applied to a single side of a trade.

    Dollars, signed positive (i.e. always subtracts from P&L). Components are
    broken out so analytics can decompose alpha vs friction.
    """

    commission: float = 0.0
    slippage: float = 0.0
    borrow_carry: float = 0.0
    gap_penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.slippage + self.borrow_carry + self.gap_penalty


def apply_open_cost(
    *,
    shares: float,
    price: float,
    commission_per_share_usd: float,
    slippage_bps: float,
) -> Cost:
    """Cost paid at position open: commission + entry slippage.

    Slippage is a flat bps haircut on notional; the model treats this as the
    expected market-impact + crossing-the-spread loss for a small order.
    """
    if shares <= 0 or price <= 0:
        return Cost()
    notional = shares * price
    commission = commission_per_share_usd * shares
    slippage = notional * (slippage_bps / 10_000.0)
    return Cost(commission=commission, slippage=slippage)


def apply_close_cost(
    *,
    side: str,
    shares: float,
    entry_price: float,
    exit_price: float,
    days_held: int,
    commission_per_share_usd: float,
    slippage_bps: float,
    borrow_rate_pa_pct: float,
    gap_penalty_bps: float = 0.0,
) -> Cost:
    """Cost paid at position close: commission + exit slippage + (short) borrow + optional gap penalty.

    Args:
        side: "long" or "short". Borrow carry only applies to shorts.
        days_held: Used to amortize the annual borrow rate.
        gap_penalty_bps: Additional slippage applied when the close was a
            stop-trigger that gapped through. Caller decides when to apply.
    """
    if shares <= 0 or exit_price <= 0:
        return Cost()
    exit_notional = shares * exit_price
    commission = commission_per_share_usd * shares
    slippage = exit_notional * (slippage_bps / 10_000.0)
    gap = exit_notional * (gap_penalty_bps / 10_000.0)

    borrow = 0.0
    if side == "short" and entry_price > 0 and days_held > 0:
        # Short borrow accrued on average notional over the holding period.
        avg_notional = shares * (entry_price + exit_price) / 2.0
        borrow = avg_notional * (borrow_rate_pa_pct / 100.0) * (days_held / 365.0)

    return Cost(
        commission=commission, slippage=slippage, borrow_carry=borrow, gap_penalty=gap
    )
