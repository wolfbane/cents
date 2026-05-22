"""Static sector → SPDR sector ETF map used by the factory's paired-mode twins.

Source: SPDR sector ETF lineup, snapshot date 2026-01-01. Keys match FMP's
`sector` field on the company profile endpoint.
"""

import logging

from cents.config import get_settings

logger = logging.getLogger(__name__)


SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

FALLBACK_ETF = "SPY"


class TransientSectorLookupError(Exception):
    """FMP fundamentals fetch failed transiently; caller should skip + retry next run.

    Distinct from "symbol has no sector entry" (returns None), which is a
    terminal answer and legitimately routes to the SPY fallback. The
    transient case must NOT fall back to SPY: a network-degraded sector
    lookup silently produces a "neutral"-cohort thesis hedged against SPY
    (not against the symbol's actual sector), contaminating the cohort
    comparison the experiment is meant to measure.
    """


def hedge_etf_for(symbol: str) -> str | None:
    """Return the SPDR sector ETF for a symbol's sector, falling back to SPY.

    Resolution order:
    1. FMP profile lookup → sector → SECTOR_ETF_MAP
    2. If FMP returned successfully but the sector is unknown / unmapped,
       return SPY (legitimate broad-market fallback).

    Raises:
        TransientSectorLookupError: when the FMP fetch was degraded (network
            failure, 5xx, timeout) AND no sector was resolvable. The caller
            should skip the symbol rather than silently hedge against SPY.
    """
    settings = get_settings()
    if not settings.fmp_api_key:
        return FALLBACK_ETF

    sector = _lookup_sector(symbol)
    if not sector:
        return FALLBACK_ETF
    return SECTOR_ETF_MAP.get(sector, FALLBACK_ETF)


def _lookup_sector(symbol: str) -> str | None:
    """Fetch the sector for a symbol via the FMP fundamentals provider.

    Returns:
        The sector string (e.g. "Technology") when FMP responded with one.
        None when FMP responded but the symbol has no sector entry — a
        terminal "no answer" that legitimately falls through to FALLBACK_ETF.

    Raises:
        TransientSectorLookupError: when FMP marked the fetch as degraded
            (network failure / 5xx / timeout, via FundamentalsData.degraded)
            AND no sector was resolved. Distinguishes "we don't know the
            sector" (terminal) from "we couldn't reach FMP" (transient).
    """
    try:
        from cents.data import get_fundamentals_provider

        provider = get_fundamentals_provider()
        data = provider.get_fundamentals(symbol)
    except Exception as exc:
        # Unexpected provider error — surface as transient so the caller
        # skips the symbol rather than degrading to SPY. A truly terminal
        # provider error (auth, malformed symbol) will recur next run and
        # surface in the logs there; better to skip than to contaminate.
        logger.debug("Sector lookup raised for %s: %s", symbol, exc)
        raise TransientSectorLookupError(str(exc)) from exc

    if data.sector:
        return data.sector
    if data.degraded:
        # FMP returned but at least one of profile/ratios/metrics failed —
        # we can't trust that sector is genuinely absent vs. that the
        # profile fetch is the one that failed.
        raise TransientSectorLookupError(
            f"FMP fundamentals degraded for {symbol}; sector unresolved"
        )
    return None
