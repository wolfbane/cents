"""Beta-matched hedge ratio for the factory's paired-cohort twins.

The PM + Risk reviewers flagged that the current paired-neutral twin is
dollar-matched, not beta-matched: ``shares = position_size_usd / hedge_price``
implicitly assumes beta = 1. NVDA's beta to XLK is ~1.6-1.8, so the
"neutral" book has been running net-long high-beta and net-short the index.

This module computes a 60-day historical beta of the underlying versus the
chosen hedge ETF and returns a beta-scaled hedge notional. When beta cannot
be computed (insufficient history), it returns ``default_beta`` (usually 1.0)
so behavior degrades gracefully to the previous equal-dollar match.
"""

from __future__ import annotations

import logging
import math
from statistics import mean

logger = logging.getLogger(__name__)


def estimate_beta(
    underlying_closes: list[float],
    hedge_closes: list[float],
    *,
    lookback: int = 60,
) -> float | None:
    """OLS beta of log returns: underlying ~ hedge over the last ``lookback`` bars.

    Returns None when fewer than lookback+1 paired bars are available.
    """
    n = min(len(underlying_closes), len(hedge_closes))
    if n < lookback + 1:
        return None
    u = underlying_closes[-(lookback + 1):]
    h = hedge_closes[-(lookback + 1):]
    ur, hr = [], []
    for i in range(1, len(u)):
        if u[i - 1] <= 0 or u[i] <= 0 or h[i - 1] <= 0 or h[i] <= 0:
            return None
        ur.append(math.log(u[i] / u[i - 1]))
        hr.append(math.log(h[i] / h[i - 1]))
    if len(ur) < 2:
        return None
    mu_h = mean(hr)
    mu_u = mean(ur)
    cov = sum((x - mu_h) * (y - mu_u) for x, y in zip(hr, ur)) / len(ur)
    var = sum((x - mu_h) ** 2 for x in hr) / len(hr)
    if var <= 0:
        return None
    return cov / var


def beta_match_ratio(
    *,
    beta: float | None,
    default_beta: float,
    min_beta: float = 0.25,
    max_beta: float = 3.0,
) -> float:
    """Return a sanitized hedge-ratio multiplier from an estimated beta.

    Clamps to [min_beta, max_beta] to bound the hedge-leg notional. Falls back
    to ``default_beta`` when beta is None. The clamp is deliberately wide —
    too aggressive a clamp would just reintroduce the dollar-matched bug.
    """
    if beta is None or not math.isfinite(beta):
        return max(min_beta, min(max_beta, default_beta))
    return max(min_beta, min(max_beta, beta))
