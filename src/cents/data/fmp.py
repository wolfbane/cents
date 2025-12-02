"""Financial Modeling Prep (FMP) fundamentals data provider."""

import functools
import json
import logging
import urllib.error
import urllib.request
from datetime import date

from cents.config import get_settings
from cents.data.providers import FundamentalsData, FundamentalsDataProvider
from cents.exceptions import ConfigurationError, DataFetchError

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def _sanitize_url(url: str) -> str:
    """Remove API keys from URL for safe logging."""
    import re
    return re.sub(r"(apikey=)[^&]+", r"\1***", url, flags=re.IGNORECASE)


class FMPFundamentalsProvider:
    """Fundamentals data provider using Financial Modeling Prep API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key (defaults to config/env)
        """
        settings = get_settings()
        self._api_key = api_key or settings.fmp_api_key
        self._timeout = settings.default_api_timeout

        if not self._api_key:
            raise ConfigurationError(
                "FMP API key required. Set FMP_API_KEY environment variable "
                "or fmp_api_key in ~/.cents/config.toml"
            )

    def _fetch_json(self, endpoint: str, **params) -> dict | list | None:
        """Fetch JSON from FMP API.

        Returns None on network/API errors, logs warnings for debugging.
        Callers should handle None gracefully (e.g., return empty dict for missing data).
        """
        params["apikey"] = self._api_key
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{FMP_BASE_URL}/{endpoint}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as response:
                data = json.loads(response.read().decode())
                # FMP returns error messages as {"Error Message": "..."} on some endpoints
                if isinstance(data, dict) and "Error Message" in data:
                    logger.warning("FMP API error for %s: %s", endpoint, data["Error Message"])
                    return None
                return data
        except urllib.error.URLError as e:
            logger.warning("FMP API request failed for %s: %s", endpoint, e)
            logger.debug("Failed URL: %s", _sanitize_url(url))
            return None
        except json.JSONDecodeError as e:
            logger.warning("FMP API returned invalid JSON for %s: %s", endpoint, e)
            return None

    def _fetch_analyst_estimates(self, symbol: str) -> dict | None:
        """Fetch analyst earnings estimates for forward P/E calculation.

        Only called when fetch_forward_estimates is enabled in config.
        Returns the most recent annual estimate.
        """
        if not get_settings().fetch_forward_estimates:
            return None

        data = self._fetch_json("analyst-estimates", symbol=symbol, period="annual")
        if data and len(data) > 0:
            return data[0]
        return None

    def get_fundamentals(
        self, symbol: str, as_of: date | None = None
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
        profile_data = self._fetch_json("profile", symbol=symbol)
        profile = profile_data[0] if profile_data and len(profile_data) > 0 else {}

        # Fetch TTM ratios for detailed financial ratios
        ratios_data = self._fetch_json("ratios-ttm", symbol=symbol)
        ratios = ratios_data[0] if ratios_data and len(ratios_data) > 0 else {}

        # Fetch key metrics for growth data
        metrics_data = self._fetch_json("key-metrics-ttm", symbol=symbol)
        metrics = metrics_data[0] if metrics_data and len(metrics_data) > 0 else {}

        # Fetch analyst estimates for forward metrics (if enabled)
        estimates = self._fetch_analyst_estimates(symbol)

        # Calculate forward P/E and earnings growth if estimates available
        forward_pe = None
        earnings_growth = None

        if estimates:
            estimated_eps = estimates.get("epsAvg")  # FMP stable API field name
            current_price = profile.get("price")
            trailing_eps = metrics.get("netIncomePerShareTTM")

            # Forward P/E = Current Price / Estimated EPS
            if estimated_eps and current_price and estimated_eps > 0:
                forward_pe = current_price / estimated_eps

            # Earnings Growth = (Estimated - Trailing) / Trailing
            if estimated_eps and trailing_eps and trailing_eps > 0:
                earnings_growth = (estimated_eps - trailing_eps) / trailing_eps

        # Map FMP fields to our FundamentalsData (stable API field names)
        return FundamentalsData(
            symbol=symbol,
            name=profile.get("companyName"),
            sector=profile.get("sector"),  # e.g., "Technology", "Healthcare"
            # Valuation
            pe_ratio=ratios.get("priceToEarningsRatioTTM"),
            forward_pe=forward_pe,
            peg_ratio=ratios.get("priceToEarningsGrowthRatioTTM"),
            # Growth - FMP provides these as decimals
            # Note: FMP key-metrics-ttm doesn't provide revenue growth rate directly
            # Revenue growth requires comparing historical income statements
            revenue_growth=None,
            earnings_growth=earnings_growth,
            # Profitability
            profit_margin=ratios.get("netProfitMarginTTM"),
            return_on_equity=metrics.get("returnOnEquityTTM"),
            # Balance sheet
            debt_to_equity=ratios.get("debtToEquityRatioTTM"),
            current_ratio=ratios.get("currentRatioTTM"),
            # Analyst - rating endpoint deprecated in stable API
            recommendation=None,
            # Store raw data for extensibility
            raw={
                "profile": profile,
                "ratios": ratios,
                "metrics": metrics,
                "estimates": estimates,
            },
        )

    def _get_historical_fundamentals(
        self, symbol: str, as_of: date
    ) -> FundamentalsData:
        """Get fundamentals as of a historical date using quarterly data."""
        # Fetch company profile (static info)
        profile_data = self._fetch_json("profile", symbol=symbol)
        profile = profile_data[0] if profile_data and len(profile_data) > 0 else {}

        # Fetch historical quarterly ratios
        ratios_data = self._fetch_json("ratios", symbol=symbol, period="quarter", limit=40)
        ratios = self._find_quarter_data(ratios_data, as_of)

        # Fetch historical quarterly key metrics
        metrics_data = self._fetch_json("key-metrics", symbol=symbol, period="quarter", limit=40)
        metrics = self._find_quarter_data(metrics_data, as_of)

        return FundamentalsData(
            symbol=symbol,
            name=profile.get("companyName"),
            sector=profile.get("sector"),
            # Valuation
            pe_ratio=ratios.get("priceEarningsRatio"),
            forward_pe=None,
            peg_ratio=ratios.get("priceEarningsToGrowthRatio"),
            # Growth
            # Note: Historical endpoint doesn't provide revenue growth rate directly
            revenue_growth=None,
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
        self, data: list | None, as_of: date
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

    def _map_rating(self, fmp_rating: str | None) -> str | None:
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

    def get_historical_ratios(self, symbol: str, years: int = 5) -> list[dict]:
        """Fetch historical annual ratios for moat analysis.

        Returns list of dicts with: date, grossProfitMargin, operatingProfitMargin,
        returnOnEquity, and ROIC (from key-metrics).

        Args:
            symbol: Ticker symbol
            years: Number of years of history (default 5)

        Returns:
            List of annual ratio records, most recent first
        """
        # Fetch historical ratios (annual)
        ratios_data = self._fetch_json(
            "ratios", symbol=symbol, period="annual", limit=years
        )
        ratios_list = ratios_data if ratios_data else []

        # Fetch historical key metrics for ROIC
        metrics_data = self._fetch_json(
            "key-metrics", symbol=symbol, period="annual", limit=years
        )
        metrics_list = metrics_data if metrics_data else []

        # Build lookup of metrics by date
        metrics_by_date = {m.get("date"): m for m in metrics_list}

        # Merge ratios with ROIC/ROE from key-metrics
        result = []
        for ratio in ratios_list:
            record_date = ratio.get("date")
            metrics = metrics_by_date.get(record_date, {})

            result.append({
                "date": record_date,
                "grossProfitMargin": ratio.get("grossProfitMargin"),
                "operatingProfitMargin": ratio.get("operatingProfitMargin"),
                "netProfitMargin": ratio.get("netProfitMargin"),
                # ROE and ROIC come from key-metrics endpoint
                "returnOnEquity": metrics.get("returnOnEquity"),
                "roic": metrics.get("returnOnInvestedCapital"),
            })

        return result

    def get_insider_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        """Fetch insider trading transactions for a symbol.

        Args:
            symbol: Ticker symbol
            limit: Maximum number of transactions to fetch (default 100)

        Returns:
            List of insider trade records with fields:
            - transactionDate: Date of transaction
            - transactionType: Type (S-Sale, P-Purchase, G-Gift, etc.)
            - reportingName: Name of insider
            - typeOfOwner: Role (e.g., "officer: CEO")
            - securitiesTransacted: Number of shares
            - price: Transaction price (0 for non-market trades)
            - acquisitionOrDisposition: A=acquire, D=dispose
        """
        data = self._fetch_json(
            "insider-trading/search", symbol=symbol, limit=limit
        )
        return data if data else []


@functools.lru_cache(maxsize=1)
def get_fundamentals_provider() -> FMPFundamentalsProvider:
    """Get or create the default FMP fundamentals provider (thread-safe singleton)."""
    return FMPFundamentalsProvider()


def clear_fundamentals_provider_cache() -> None:
    """Clear the cached fundamentals provider.

    Call this if settings change and you need a fresh provider instance.
    """
    get_fundamentals_provider.cache_clear()
