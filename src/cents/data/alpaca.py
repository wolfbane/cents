"""Alpaca market data provider."""

import functools
import logging
from datetime import date, datetime, timedelta

from cents.cache import cached_request
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

    # Max bid/ask spread as percentage of midpoint before falling back to close
    MAX_SPREAD_PCT = 0.02  # 2%

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
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
        self, symbol: str, days: int = 180, as_of: date | None = None
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
        # Cache key params — keying current-day requests by today's date so
        # same-day repeats hit the cache without serving stale bars tomorrow.
        cache_params = {
            "symbol": symbol,
            "days": days,
            "as_of": (as_of or date.today()).isoformat(),
        }

        def do_fetch():
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

            # Convert Alpaca bars to JSON-serializable format for caching
            bars_data = []
            if symbol in bars_response.data:
                for bar in bars_response.data[symbol]:
                    bars_data.append({
                        "timestamp": bar.timestamp.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    })
            return bars_data

        # Always cache — same-day key prevents stale data across days, and
        # repeated current-day requests (screeners, factory runs) hit cache.
        bars_data = cached_request("alpaca", "bars", cache_params, do_fetch)

        # Convert cached data back to PriceBar objects
        bars = [
            PriceBar(
                timestamp=datetime.fromisoformat(b["timestamp"]),
                open=b["open"],
                high=b["high"],
                low=b["low"],
                close=b["close"],
                volume=b["volume"],
            )
            for b in (bars_data or [])
        ]

        return PriceHistory(symbol=symbol, bars=bars)

    def _get_last_closes(self, symbols: list[str]) -> dict[str, float]:
        """Fetch last daily close for multiple symbols."""
        closes = {}
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=5),
                end=datetime.now(),
            )
            bars = self._client.get_stock_bars(request)
            for sym in symbols:
                if sym in bars.data and bars.data[sym]:
                    closes[sym] = float(bars.data[sym][-1].close)
        except Exception as e:
            logger.debug("Failed to get closes: %s", e)
        return closes

    def get_latest_prices(
        self, symbols: list[str]
    ) -> dict[str, float]:
        """
        Get latest prices for multiple symbols (batch).

        Uses quote midpoint if spread is tight, otherwise falls back to last close.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to price (missing symbols excluded)
        """
        if not symbols:
            return {}
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._client.get_stock_latest_quote(request)

            prices = {}
            need_close = []

            for sym in symbols:
                if sym in quotes:
                    quote = quotes[sym]
                    bid = float(quote.bid_price) if quote.bid_price else 0
                    ask = float(quote.ask_price) if quote.ask_price else 0

                    if bid and ask:
                        mid = (bid + ask) / 2
                        spread_pct = (ask - bid) / mid if mid else 1
                        if spread_pct <= self.MAX_SPREAD_PCT:
                            prices[sym] = mid
                        else:
                            need_close.append(sym)
                    elif ask or bid:
                        need_close.append(sym)

            # Fall back to last close for wide spreads
            if need_close:
                closes = self._get_last_closes(need_close)
                prices.update(closes)

            return prices
        except Exception as e:
            logger.warning("Failed to get batch prices: %s", e)
            return {}

    def get_latest_price(
        self, symbol: str, as_of: date | None = None
    ) -> float | None:
        """
        Get latest price for a symbol.

        Uses quote midpoint if spread is tight, otherwise falls back to last close.

        Args:
            symbol: Ticker symbol
            as_of: Date to get closing price for (default: current quote)

        Returns:
            Latest price or close price for as_of date
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
                bid = float(quote.bid_price) if quote.bid_price else 0
                ask = float(quote.ask_price) if quote.ask_price else 0

                if bid and ask:
                    mid = (bid + ask) / 2
                    spread_pct = (ask - bid) / mid if mid else 1
                    if spread_pct <= self.MAX_SPREAD_PCT:
                        return mid
                    # Wide spread - fall back to last close
                    closes = self._get_last_closes([symbol])
                    return closes.get(symbol)
                elif ask or bid:
                    closes = self._get_last_closes([symbol])
                    return closes.get(symbol)
            return None
        except Exception as e:
            logger.warning("Failed to get latest price for %s: %s", symbol, e)
            return None


@functools.lru_cache(maxsize=1)
def get_price_provider() -> AlpacaPriceProvider:
    """Get or create the default Alpaca price provider (thread-safe singleton)."""
    return AlpacaPriceProvider()


def clear_price_provider_cache() -> None:
    """Clear the cached price provider.

    Call this if settings change and you need a fresh provider instance.
    """
    get_price_provider.cache_clear()
