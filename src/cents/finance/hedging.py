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


def estimate_beta_fit(
    underlying_closes: list[float],
    hedge_closes: list[float],
    *,
    lookback: int = 60,
) -> tuple[float, float | None] | None:
    """OLS ``(beta, r_squared)`` of log returns: underlying ~ hedge over the
    last ``lookback`` bars.

    Returns None when fewer than lookback+1 paired bars are available or the
    regression is degenerate (non-positive closes, flat hedge). ``r_squared``
    is None when the underlying itself is flat (corr² is undefined) — gate
    callers should treat that as a failing fit.

    Exposing the fit quality (not just the beta) lets the factory engine
    persist ``hedge_fit_r2`` per thesis, so "neutral"-cohort analytics can
    stratify by how genuinely beta-neutral each hedge actually was (v0.13).
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
    var_h = sum((x - mu_h) ** 2 for x in hr) / len(hr)
    var_u = sum((y - mu_u) ** 2 for y in ur) / len(ur)
    if var_h <= 0:
        return None
    beta = cov / var_h
    # R² of underlying-on-hedge regression = corr² = cov² / (var_h * var_u).
    r_squared = (cov * cov) / (var_h * var_u) if var_u > 0 else None
    return beta, r_squared


def estimate_beta(
    underlying_closes: list[float],
    hedge_closes: list[float],
    *,
    lookback: int = 60,
    min_r_squared: float | None = None,
) -> float | None:
    """OLS beta of log returns: underlying ~ hedge over the last ``lookback`` bars.

    Returns None when fewer than lookback+1 paired bars are available, when
    the regression is degenerate, or when ``min_r_squared`` is supplied and
    the fit R² falls below it (the relationship is too weak to hedge with).
    Thin wrapper over ``estimate_beta_fit`` for callers that don't need the
    fit quality.
    """
    fit = estimate_beta_fit(underlying_closes, hedge_closes, lookback=lookback)
    if fit is None:
        return None
    beta, r_squared = fit
    if min_r_squared is not None:
        # When the underlying is itself flat (r_squared is None), the
        # regression carries no information — treat as failing the gate.
        if r_squared is None or r_squared < min_r_squared:
            return None
    return beta


def beta_match_ratio(
    *,
    beta: float | None,
    default_beta: float,
    min_beta: float = 0.10,
    max_beta: float = 5.0,
) -> float:
    """Return a sanitized hedge-ratio multiplier from an estimated beta.

    Clamps to [min_beta, max_beta] to bound the hedge-leg notional. Falls back
    to ``default_beta`` when beta is None. The defaults are deliberately wide
    — a tight clamp re-introduces dollar-mismatch on low-correlation single
    names (forcing fake net-short hedge) and leaves residual net-long beta on
    high-vol names.
    """
    if beta is None or not math.isfinite(beta):
        return max(min_beta, min(max_beta, default_beta))
    return max(min_beta, min(max_beta, beta))
