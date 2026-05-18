"""Vol-scaled position sizing.

Replaces equal-dollar sizing (``budget_usd / target_positions``) with
inverse-vol scaling toward a target per-position daily $-volatility, with a
hard cap on single-position weight as a fraction of the budget.

The PM/Risk/CFO critique converged on this one: equal-dollar sizing means a
50% IV name carries 3-5x the dollar risk of a 15% IV name, with no awareness
in either the entry threshold or the analytics. Inverse-vol sizing is the
crudest fix that turns this from "fiction" into "honest."
"""

from __future__ import annotations

import logging
import math
from statistics import pstdev

logger = logging.getLogger(__name__)


def realized_vol_pct(closes: list[float], *, lookback: int = 20) -> float | None:
    """Annualized realized volatility (%) from a series of daily closes.

    Uses pstdev of log returns over the last ``lookback`` bars * sqrt(252) * 100.
    Returns None when fewer than ``lookback`` returns are available — caller
    decides whether to fall back to equal-dollar or skip the symbol.
    """
    if len(closes) < lookback + 1:
        return None
    series = closes[-(lookback + 1):]
    returns = []
    for prev, curr in zip(series[:-1], series[1:]):
        if prev <= 0 or curr <= 0:
            return None
        returns.append(math.log(curr / prev))
    if not returns:
        return None
    daily_std = pstdev(returns)
    return daily_std * math.sqrt(252) * 100.0


def vol_scaled_shares(
    *,
    price: float,
    annual_vol_pct: float | None,
    budget_usd: float,
    target_vol_pct_per_position: float,
    max_position_pct: float,
    fallback_position_usd: float | None = None,
) -> tuple[float, str]:
    """Return (shares, sizing_method) for a single-leg position.

    Args:
        price: Current price per share.
        annual_vol_pct: Annualized vol of the underlying as a percent (e.g. 35.0).
            None falls back to the equal-dollar ``fallback_position_usd``.
        budget_usd: Total factory budget.
        target_vol_pct_per_position: Target per-position annual $-vol as a fraction
            of budget (e.g. 0.5 = 0.5%).
        max_position_pct: Hard cap on per-position dollar weight as a % of budget.
        fallback_position_usd: Used when vol is unavailable.

    Returns:
        (shares, method) where method is "vol_scaled", "max_capped", or "equal_dollar".
    """
    if price <= 0 or budget_usd <= 0:
        return 0.0, "equal_dollar"

    max_dollar = budget_usd * (max_position_pct / 100.0)

    if annual_vol_pct is None or annual_vol_pct <= 0:
        # No vol data — fall back to equal-dollar or zero out.
        if fallback_position_usd is None:
            return 0.0, "equal_dollar"
        dollar = min(fallback_position_usd, max_dollar)
        return dollar / price, "equal_dollar"

    # Target per-position daily $-vol = budget * target_vol_pct / 100.
    # Position vol per share per year = annual_vol_pct / 100 * price.
    # So shares = (target_dollar_vol_annual) / (annual_vol_pct/100 * price).
    target_dollar_vol = budget_usd * (target_vol_pct_per_position / 100.0)
    vol_dollar_per_share = (annual_vol_pct / 100.0) * price
    if vol_dollar_per_share <= 0:
        return 0.0, "equal_dollar"
    shares = target_dollar_vol / vol_dollar_per_share

    # Cap at max position weight.
    if shares * price > max_dollar:
        return max_dollar / price, "max_capped"
    return shares, "vol_scaled"
