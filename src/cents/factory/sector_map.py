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


def hedge_etf_for(symbol: str) -> str | None:
    """Return the SPDR sector ETF for a symbol's sector, falling back to SPY.

    Resolution order:
    1. FMP profile lookup → sector → SECTOR_ETF_MAP
    2. If FMP unavailable or sector unknown, return SPY
    """
    settings = get_settings()
    if not settings.fmp_api_key:
        return FALLBACK_ETF

    sector = _lookup_sector(symbol)
    if not sector:
        return FALLBACK_ETF
    return SECTOR_ETF_MAP.get(sector, FALLBACK_ETF)


def _lookup_sector(symbol: str) -> str | None:
    """Fetch the sector for a symbol via the FMP fundamentals provider."""
    try:
        from cents.data import get_fundamentals_provider

        provider = get_fundamentals_provider()
        data = provider.get_fundamentals(symbol)
        return data.sector
    except Exception as exc:
        logger.debug("Sector lookup failed for %s: %s", symbol, exc)
        return None
