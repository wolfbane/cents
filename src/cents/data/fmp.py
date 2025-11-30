"""Financial Modeling Prep (FMP) fundamentals data provider."""

import urllib.request
import urllib.error
import json
from typing import Optional

from cents.config import get_settings
from cents.data.providers import FundamentalsData, FundamentalsDataProvider

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPFundamentalsProvider:
    """Fundamentals data provider using Financial Modeling Prep API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key (defaults to config/env)
        """
        settings = get_settings()
        self._api_key = api_key or settings.fmp_api_key

        if not self._api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY environment variable "
                "or fmp_api_key in ~/.cents/config.toml"
            )

    def _fetch_json(self, endpoint: str) -> Optional[dict | list]:
        """Fetch JSON from FMP API."""
        url = f"{FMP_BASE_URL}/{endpoint}?apikey={self._api_key}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data
        except (urllib.error.URLError, json.JSONDecodeError):
            return None

    def get_fundamentals(self, symbol: str) -> FundamentalsData:
        """
        Get fundamental data for a symbol from FMP.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")

        Returns:
            FundamentalsData with available metrics
        """
        # Fetch company profile for basic info and some metrics
        profile_data = self._fetch_json(f"profile/{symbol}")
        profile = profile_data[0] if profile_data and len(profile_data) > 0 else {}

        # Fetch TTM ratios for detailed financial ratios
        ratios_data = self._fetch_json(f"ratios-ttm/{symbol}")
        ratios = ratios_data[0] if ratios_data and len(ratios_data) > 0 else {}

        # Fetch key metrics for growth data
        metrics_data = self._fetch_json(f"key-metrics-ttm/{symbol}")
        metrics = metrics_data[0] if metrics_data and len(metrics_data) > 0 else {}

        # Fetch analyst rating/recommendation
        rating_data = self._fetch_json(f"rating/{symbol}")
        rating = rating_data[0] if rating_data and len(rating_data) > 0 else {}

        # Map FMP fields to our FundamentalsData
        return FundamentalsData(
            symbol=symbol,
            name=profile.get("companyName"),
            # Valuation
            pe_ratio=profile.get("peRatioTTM") or ratios.get("peRatioTTM"),
            forward_pe=None,  # FMP doesn't have forward P/E in standard endpoints
            peg_ratio=ratios.get("pegRatioTTM"),
            # Growth - FMP provides these as decimals
            revenue_growth=metrics.get("revenuePerShareTTM"),  # Use as proxy
            earnings_growth=None,  # Would need historical comparison
            # Profitability
            profit_margin=ratios.get("netProfitMarginTTM"),
            return_on_equity=ratios.get("returnOnEquityTTM"),
            # Balance sheet
            debt_to_equity=ratios.get("debtEquityRatioTTM"),
            current_ratio=ratios.get("currentRatioTTM"),
            # Analyst - map FMP rating to simple recommendation
            recommendation=self._map_rating(rating.get("ratingRecommendation")),
            # Store raw data for extensibility
            raw={
                "profile": profile,
                "ratios": ratios,
                "metrics": metrics,
                "rating": rating,
            },
        )

    def _map_rating(self, fmp_rating: Optional[str]) -> Optional[str]:
        """Map FMP rating to standard recommendation string."""
        if not fmp_rating:
            return None
        rating_lower = fmp_rating.lower()
        if "strong buy" in rating_lower:
            return "strong_buy"
        elif "buy" in rating_lower:
            return "buy"
        elif "hold" in rating_lower or "neutral" in rating_lower:
            return "hold"
        elif "strong sell" in rating_lower:
            return "strong_sell"
        elif "sell" in rating_lower:
            return "sell"
        return rating_lower.replace(" ", "_")


# Singleton instance for convenience
_default_provider: Optional[FMPFundamentalsProvider] = None


def get_fundamentals_provider() -> FMPFundamentalsProvider:
    """Get or create the default FMP fundamentals provider."""
    global _default_provider
    if _default_provider is None:
        _default_provider = FMPFundamentalsProvider()
    return _default_provider
