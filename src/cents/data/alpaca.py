"""Alpaca market data provider."""

import functools
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from cents.config import get_settings
from cents.data.providers import PriceBar, PriceHistory, PriceDataProvider
from cents.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame

    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False


class AlpacaPriceProvider:
    """Price data provider using Alpaca Market Data API."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Alpaca data client.

        Args:
            api_key: Alpaca API key (defaults to config/env)
            secret_key: Alpaca secret key (defaults to config/env)
        """
        if not ALPACA_DATA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Install with: pip install cents[broker]"
            )

        settings = get_settings()
        self._api_key = api_key or settings.alpaca_api_key
        self._secret_key = secret_key or settings.alpaca_secret_key

        if not self._api_key or not self._secret_key:
            raise ConfigurationError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or in ~/.cents/config.toml"
            )

        self._client = StockHistoricalDataClient(self._api_key, self._secret_key)

    def get_history(
        self, symbol: str, days: int = 180, as_of: Optional[date] = None
    ) -> PriceHistory:
        """
        Get historical daily bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            days: Number of days of history
            as_of: End date for history (default: today)

        Returns:
            PriceHistory with daily OHLCV bars
        """
        if as_of:
            end = datetime.combine(as_of, datetime.max.time())
        else:
            end = datetime.now()
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )

        bars_response = self._client.get_stock_bars(request)

        # Convert Alpaca bars to our format
        bars = []
        if symbol in bars_response.data:
            for bar in bars_response.data[symbol]:
                bars.append(
                    PriceBar(
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                    )
                )

        return PriceHistory(symbol=symbol, bars=bars)

    def get_latest_price(
        self, symbol: str, as_of: Optional[date] = None
    ) -> Optional[float]:
        """
        Get latest quote midpoint for a symbol.

        Args:
            symbol: Ticker symbol
            as_of: Date to get closing price for (default: current quote)

        Returns:
            Latest price (bid/ask midpoint) or close price for as_of date
        """
        try:
            if as_of:
                # For historical date, fetch the bar and return close
                history = self.get_history(symbol, days=5, as_of=as_of)
                return history.latest_close()

            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                # Use midpoint of bid/ask, fallback to ask or bid
                if quote.bid_price and quote.ask_price:
                    return (float(quote.bid_price) + float(quote.ask_price)) / 2
                return float(quote.ask_price or quote.bid_price or 0) or None
            return None
        except Exception as e:
            logger.warning("Failed to get latest price for %s: %s", symbol, e)
            return None


@functools.lru_cache(maxsize=1)
def get_price_provider() -> AlpacaPriceProvider:
    """Get or create the default Alpaca price provider (thread-safe singleton)."""
    return AlpacaPriceProvider()
