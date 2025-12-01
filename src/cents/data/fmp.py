"""Financial Modeling Prep (FMP) fundamentals data provider."""

import json
import logging
import urllib.error
import urllib.request
from datetime import date
from typing import Optional

from cents.config import get_settings
from cents.data.providers import FundamentalsData, FundamentalsDataProvider
from cents.exceptions import ConfigurationError, DataFetchError

logger = logging.getLogger(__name__)

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
            raise ConfigurationError(
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
        except urllib.error.URLError as e:
            logger.warning("FMP API request failed for %s: %s", endpoint, e)
            return None
        except json.JSONDecodeError as e:
            logger.warning("FMP API returned invalid JSON for %s: %s", endpoint, e)
            return None

    def get_fundamentals(
        self, symbol: str, as_of: Optional[date] = None
    ) -> FundamentalsData:
        """
        Get fundamental data for a symbol from FMP.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            as_of: Date to get fundamentals for (default: latest TTM)

        Returns:
            FundamentalsData with available metrics
        """
        if as_of:
            return self._get_historical_fundamentals(symbol, as_of)

        return self._get_current_fundamentals(symbol)

    def _get_current_fundamentals(self, symbol: str) -> FundamentalsData:
        """Get current/TTM fundamentals."""
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

    def _get_historical_fundamentals(
        self, symbol: str, as_of: date
    ) -> FundamentalsData:
        """Get fundamentals as of a historical date using quarterly data."""
        # Fetch company profile (static info)
        profile_data = self._fetch_json(f"profile/{symbol}")
        profile = profile_data[0] if profile_data and len(profile_data) > 0 else {}

        # Fetch historical quarterly ratios
        ratios_data = self._fetch_json(f"ratios/{symbol}?period=quarter&limit=40")
        ratios = self._find_quarter_data(ratios_data, as_of)

        # Fetch historical quarterly key metrics
        metrics_data = self._fetch_json(f"key-metrics/{symbol}?period=quarter&limit=40")
        metrics = self._find_quarter_data(metrics_data, as_of)

        return FundamentalsData(
            symbol=symbol,
            name=profile.get("companyName"),
            # Valuation
            pe_ratio=ratios.get("priceEarningsRatio"),
            forward_pe=None,
            peg_ratio=ratios.get("priceEarningsToGrowthRatio"),
            # Growth
            revenue_growth=metrics.get("revenuePerShare"),
            earnings_growth=None,
            # Profitability
            profit_margin=ratios.get("netProfitMargin"),
            return_on_equity=ratios.get("returnOnEquity"),
            # Balance sheet
            debt_to_equity=ratios.get("debtEquityRatio"),
            current_ratio=ratios.get("currentRatio"),
            # No historical recommendations available
            recommendation=None,
            raw={
                "profile": profile,
                "ratios": ratios,
                "metrics": metrics,
                "as_of": as_of.isoformat(),
            },
        )

    def _find_quarter_data(
        self, data: Optional[list], as_of: date
    ) -> dict:
        """Find the most recent quarterly data before as_of date."""
        if not data:
            return {}

        as_of_str = as_of.isoformat()
        for item in data:
            # FMP quarterly data has 'date' field in YYYY-MM-DD format
            item_date = item.get("date", "")
            if item_date <= as_of_str:
                return item

        # If no data before as_of, return empty
        return {}

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
