"""Tests for data providers."""

from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from cents.data.providers import (
    PriceBar,
    PriceHistory,
    FundamentalsData,
)
from cents.data.fmp import FMPFundamentalsProvider
from cents.data.alpaca import AlpacaPriceProvider, ALPACA_DATA_AVAILABLE


class TestPriceBar:
    """Tests for PriceBar dataclass."""

    def test_create_price_bar(self):
        """Create a price bar."""
        bar = PriceBar(
            timestamp=datetime(2024, 1, 15, 9, 30),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
        )
        assert bar.open == 150.0
        assert bar.close == 154.0
        assert bar.volume == 1000000


class TestPriceHistory:
    """Tests for PriceHistory dataclass."""

    @pytest.fixture
    def sample_history(self):
        """Create sample price history."""
        bars = [
            PriceBar(datetime(2024, 1, 1), 100.0, 105.0, 99.0, 103.0, 1000000),
            PriceBar(datetime(2024, 1, 2), 103.0, 108.0, 102.0, 107.0, 1200000),
            PriceBar(datetime(2024, 1, 3), 107.0, 110.0, 106.0, 109.0, 900000),
        ]
        return PriceHistory(symbol="AAPL", bars=bars)

    def test_closes(self, sample_history):
        """Get close prices."""
        assert sample_history.closes == [103.0, 107.0, 109.0]

    def test_volumes(self, sample_history):
        """Get volumes."""
        assert sample_history.volumes == [1000000, 1200000, 900000]

    def test_highs(self, sample_history):
        """Get high prices."""
        assert sample_history.highs == [105.0, 108.0, 110.0]

    def test_lows(self, sample_history):
        """Get low prices."""
        assert sample_history.lows == [99.0, 102.0, 106.0]

    def test_latest_close(self, sample_history):
        """Get most recent close."""
        assert sample_history.latest_close() == 109.0

    def test_latest_close_empty(self):
        """Returns None for empty history."""
        history = PriceHistory(symbol="AAPL", bars=[])
        assert history.latest_close() is None

    def test_close_at(self, sample_history):
        """Get close at specific day."""
        assert sample_history.close_at(0) == 109.0  # Most recent
        assert sample_history.close_at(1) == 107.0
        assert sample_history.close_at(2) == 103.0

    def test_close_at_out_of_range(self, sample_history):
        """Returns None for out of range."""
        assert sample_history.close_at(10) is None
        assert sample_history.close_at(-1) is None


class TestFundamentalsData:
    """Tests for FundamentalsData dataclass."""

    def test_create_with_all_fields(self):
        """Create fundamentals data with all fields."""
        data = FundamentalsData(
            symbol="AAPL",
            name="Apple Inc.",
            pe_ratio=28.5,
            forward_pe=25.0,
            peg_ratio=1.5,
            revenue_growth=0.15,
            earnings_growth=0.20,
            profit_margin=0.25,
            return_on_equity=0.40,
            debt_to_equity=1.5,
            current_ratio=1.2,
            recommendation="buy",
            raw={"source": "test"},
        )
        assert data.symbol == "AAPL"
        assert data.pe_ratio == 28.5
        assert data.recommendation == "buy"

    def test_create_with_defaults(self):
        """Create fundamentals data with defaults."""
        data = FundamentalsData(symbol="AAPL")
        assert data.name is None
        assert data.pe_ratio is None
        assert data.raw == {}


class TestFMPFundamentalsProvider:
    """Tests for FMP fundamentals provider."""

    @patch("cents.data.fmp.get_settings")
    def test_init_missing_api_key(self, mock_settings):
        """Raises ConfigurationError when API key missing."""
        from cents.exceptions import ConfigurationError
        mock_settings.return_value.fmp_api_key = None

        with pytest.raises(ConfigurationError, match="FMP API key required"):
            FMPFundamentalsProvider()

    @patch("cents.data.fmp.get_settings")
    def test_init_with_api_key(self, mock_settings):
        """Initialize with API key from settings."""
        mock_settings.return_value.fmp_api_key = "test_key"

        provider = FMPFundamentalsProvider()
        assert provider._api_key == "test_key"

    @patch("cents.data.fmp.get_settings")
    def test_init_with_explicit_api_key(self, mock_settings):
        """Initialize with explicit API key."""
        mock_settings.return_value.fmp_api_key = "from_settings"

        provider = FMPFundamentalsProvider(api_key="explicit_key")
        assert provider._api_key == "explicit_key"

    @patch("cents.data.fmp.urllib.request.urlopen")
    @patch("cents.data.fmp.get_settings")
    def test_get_fundamentals_success(self, mock_settings, mock_urlopen):
        """Successfully fetch fundamentals."""
        mock_settings.return_value.fmp_api_key = "test_key"
        mock_settings.return_value.fetch_forward_estimates = False

        # Mock responses for each endpoint (stable API field names)
        responses = [
            [{"companyName": "Apple Inc."}],  # profile
            [{"priceToEarningsRatioTTM": 28.5, "netProfitMarginTTM": 0.25, "debtToEquityRatioTTM": 1.5}],  # ratios
            [{"revenuePerShareTTM": 10.5, "returnOnEquityTTM": 0.35}],  # metrics
        ]
        call_count = [0]

        def mock_response(*args, **kwargs):
            response = MagicMock()
            response.read.return_value = __import__("json").dumps(responses[call_count[0]]).encode()
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            call_count[0] += 1
            return response

        mock_urlopen.side_effect = mock_response

        provider = FMPFundamentalsProvider()
        data = provider.get_fundamentals("AAPL")

        assert data.symbol == "AAPL"
        assert data.name == "Apple Inc."
        assert data.pe_ratio == 28.5
        assert data.profit_margin == 0.25
        assert data.return_on_equity == 0.35

    @patch("cents.data.fmp.urllib.request.urlopen")
    @patch("cents.data.fmp.get_settings")
    def test_get_fundamentals_empty_response(self, mock_settings, mock_urlopen):
        """Handle empty API response."""
        mock_settings.return_value.fmp_api_key = "test_key"

        def mock_response(*args, **kwargs):
            response = MagicMock()
            response.read.return_value = b"[]"
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            return response

        mock_urlopen.side_effect = mock_response

        provider = FMPFundamentalsProvider()
        data = provider.get_fundamentals("INVALID")

        assert data.symbol == "INVALID"
        assert data.name is None
        assert data.pe_ratio is None

    @patch("cents.data.fmp.urllib.request.urlopen")
    @patch("cents.data.fmp.get_settings")
    def test_get_fundamentals_api_error(self, mock_settings, mock_urlopen):
        """Handle API error gracefully."""
        import urllib.error

        mock_settings.return_value.fmp_api_key = "test_key"
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        provider = FMPFundamentalsProvider()
        data = provider.get_fundamentals("AAPL")

        # Should return data with None values
        assert data.symbol == "AAPL"
        assert data.name is None

    def test_map_rating_strong_buy(self):
        """Map strong buy rating."""
        with patch("cents.data.fmp.get_settings") as mock_settings:
            mock_settings.return_value.fmp_api_key = "test_key"
            provider = FMPFundamentalsProvider()

            assert provider._map_rating("Strong Buy") == "strong_buy"
            assert provider._map_rating("STRONG BUY") == "strong_buy"

    def test_map_rating_buy(self):
        """Map buy rating."""
        with patch("cents.data.fmp.get_settings") as mock_settings:
            mock_settings.return_value.fmp_api_key = "test_key"
            provider = FMPFundamentalsProvider()

            assert provider._map_rating("Buy") == "buy"

    def test_map_rating_hold(self):
        """Map hold rating."""
        with patch("cents.data.fmp.get_settings") as mock_settings:
            mock_settings.return_value.fmp_api_key = "test_key"
            provider = FMPFundamentalsProvider()

            assert provider._map_rating("Hold") == "hold"
            assert provider._map_rating("Neutral") == "hold"

    def test_map_rating_sell(self):
        """Map sell ratings."""
        with patch("cents.data.fmp.get_settings") as mock_settings:
            mock_settings.return_value.fmp_api_key = "test_key"
            provider = FMPFundamentalsProvider()

            assert provider._map_rating("Sell") == "sell"
            assert provider._map_rating("Strong Sell") == "strong_sell"

    def test_map_rating_none(self):
        """Map None rating."""
        with patch("cents.data.fmp.get_settings") as mock_settings:
            mock_settings.return_value.fmp_api_key = "test_key"
            provider = FMPFundamentalsProvider()

            assert provider._map_rating(None) is None

    @patch("cents.data.fmp.urllib.request.urlopen")
    @patch("cents.data.fmp.get_settings")
    def test_get_fundamentals_with_as_of(self, mock_settings, mock_urlopen):
        """Get historical fundamentals with as_of date."""
        mock_settings.return_value.fmp_api_key = "test_key"

        # Mock responses for historical queries. FMP Premium quarterly ratios
        # endpoint uses `priceToEarningsRatio`/`debtToEquityRatio` names — NOT
        # the older `priceEarningsRatio` form which the agent used to read
        # (see cents-tjr).
        responses = [
            [{"companyName": "Apple Inc."}],  # profile
            [
                {"date": "2024-06-30", "priceToEarningsRatio": 30.0, "netProfitMargin": 0.26},
                {"date": "2024-03-31", "priceToEarningsRatio": 28.0, "netProfitMargin": 0.25},
                {"date": "2023-12-31", "priceToEarningsRatio": 27.0, "netProfitMargin": 0.24},
            ],  # ratios (quarterly)
            [
                {"date": "2024-06-30", "revenuePerShare": 12.0},
                {"date": "2024-03-31", "revenuePerShare": 11.5},
                {"date": "2023-12-31", "revenuePerShare": 11.0},
            ],  # metrics (quarterly)
        ]
        call_count = [0]

        def mock_response(*args, **kwargs):
            response = MagicMock()
            response.read.return_value = __import__("json").dumps(responses[call_count[0]]).encode()
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            call_count[0] += 1
            return response

        mock_urlopen.side_effect = mock_response

        provider = FMPFundamentalsProvider()
        # Request data as of April 15, 2024 - should get Q1 2024 data
        data = provider.get_fundamentals("AAPL", as_of=date(2024, 4, 15))

        assert data.symbol == "AAPL"
        assert data.name == "Apple Inc."
        assert data.pe_ratio == 28.0  # Q1 2024 data
        assert data.profit_margin == 0.25
        assert data.raw["as_of"] == "2024-04-15"

    @patch("cents.data.fmp.get_settings")
    def test_find_quarter_data(self, mock_settings):
        """Test quarter data lookup."""
        mock_settings.return_value.fmp_api_key = "test_key"
        provider = FMPFundamentalsProvider()

        data = [
            {"date": "2024-06-30", "value": 3},
            {"date": "2024-03-31", "value": 2},
            {"date": "2023-12-31", "value": 1},
        ]

        # Should find Q1 2024 (most recent before May 1)
        result = provider._find_quarter_data(data, date(2024, 5, 1))
        assert result["value"] == 2
        assert result["date"] == "2024-03-31"

        # Should find Q2 2024 (exact match on date)
        result = provider._find_quarter_data(data, date(2024, 6, 30))
        assert result["value"] == 3

        # Should return empty for date before all data
        result = provider._find_quarter_data(data, date(2023, 1, 1))
        assert result == {}

    @patch("cents.data.fmp.get_settings")
    def test_find_quarter_data_empty(self, mock_settings):
        """Test quarter data lookup with empty data."""
        mock_settings.return_value.fmp_api_key = "test_key"
        provider = FMPFundamentalsProvider()

        result = provider._find_quarter_data(None, date(2024, 5, 1))
        assert result == {}

        result = provider._find_quarter_data([], date(2024, 5, 1))
        assert result == {}


@pytest.mark.skipif(not ALPACA_DATA_AVAILABLE, reason="alpaca-py not installed")
class TestAlpacaPriceProvider:
    """Tests for Alpaca price provider."""

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_init_success(self, mock_settings, mock_client):
        """Initialize with API credentials."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        provider = AlpacaPriceProvider()

        mock_client.assert_called_once_with("test_key", "test_secret")

    @patch("cents.data.alpaca.get_settings")
    def test_init_missing_api_key(self, mock_settings):
        """Raises ConfigurationError when API key missing."""
        from cents.exceptions import ConfigurationError
        mock_settings.return_value.alpaca_api_key = None
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        with pytest.raises(ConfigurationError, match="Alpaca API credentials required"):
            AlpacaPriceProvider()

    @patch("cents.data.alpaca.get_settings")
    def test_init_missing_secret_key(self, mock_settings):
        """Raises ConfigurationError when secret key missing."""
        from cents.exceptions import ConfigurationError
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = None

        with pytest.raises(ConfigurationError, match="Alpaca API credentials required"):
            AlpacaPriceProvider()

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_history_success(self, mock_settings, mock_client_class):
        """Successfully fetch price history."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        # Mock bar data
        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2024, 1, 15, 16, 0)
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 149.0
        mock_bar.close = 154.0
        mock_bar.volume = 1000000

        mock_response = MagicMock()
        mock_response.data = {"AAPL": [mock_bar]}

        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        history = provider.get_history("AAPL", days=30)

        assert history.symbol == "AAPL"
        assert len(history.bars) == 1
        assert history.bars[0].close == 154.0
        assert history.bars[0].volume == 1000000

    @patch("cents.data.alpaca.StockBarsRequest")
    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_history_requests_split_adjusted_bars(
        self, mock_settings, mock_client_class, mock_request_class
    ):
        """Regression: bars must be requested with adjustment=SPLIT.

        Without split adjustment, tickers that split during the requested
        window (e.g. NVDA Jun-2024 10:1) produce nominal ~-90% bar-to-bar
        discontinuities that contaminate forward returns, MA20/50 levels,
        52W range, and Position P&L. See bug report 2026-05-18.
        """
        from alpaca.data.enums import Adjustment

        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_response = MagicMock()
        mock_response.data = {}
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        provider.get_history("NVDA", days=30, as_of=date(2024, 6, 30))

        assert mock_request_class.call_count >= 1
        kwargs = mock_request_class.call_args.kwargs
        assert kwargs.get("adjustment") == Adjustment.SPLIT, (
            "get_history must request split-adjusted bars; "
            f"got adjustment={kwargs.get('adjustment')!r}"
        )

    @patch("cents.data.alpaca.StockBarsRequest")
    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_last_closes_requests_split_adjusted_bars(
        self, mock_settings, mock_client_class, mock_request_class
    ):
        """Regression: last-close fallback path must also request split-adjusted bars."""
        from alpaca.data.enums import Adjustment

        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_bars_response = MagicMock()
        mock_bars_response.data = {}
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_bars_response
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        provider._get_last_closes(["NVDA"])

        assert mock_request_class.call_count >= 1
        kwargs = mock_request_class.call_args.kwargs
        assert kwargs.get("adjustment") == Adjustment.SPLIT


class TestForwardReturnsContinuity:
    """Regression: forward-return math must not show ~-90% artifacts
    when prices are consistently adjusted across a synthetic split window.

    This is the consumer-side guard for the NVDA Jun-2024 10:1 bug.
    With split-adjusted bars, the bar at signal_date and the bar at
    signal_date + 20d share the same adjustment basis, so the computed
    return reflects real market move, not the split ratio."""

    def _build_bars(self, signal_date: date, *, split_factor: float | None) -> "PriceHistory":
        """Build 70 daily bars across the signal window.

        ``split_factor=None`` returns continuous adjusted bars (the
        SPLIT-adjusted shape Alpaca returns post-fix). ``split_factor=10``
        returns RAW unadjusted bars with a 10:1 split at index 30 — pre-split
        bars stay at the nominal $1130 level, post-split bars drop to $113.
        That is the literal bug shape that produced the -87.7% artifact.
        """
        from cents.data.providers import PriceBar, PriceHistory

        bars = []
        # Index 0..29 is pre-split, 30..69 is post-split. Signal_date is at
        # index 10 (10 days before is index 0); 20-day-forward lands at
        # index 30 — i.e. right after the split bar lands.
        for i in range(70):
            d = signal_date - timedelta(days=10) + timedelta(days=i)
            if split_factor and i < 30:
                # RAW pre-split bars sit at the unadjusted $1130 level.
                close = 1130.0 + (i / 30.0) * 100.0
            elif split_factor:
                # RAW post-split bars drop to $113 — the unadjusted discontinuity.
                close = 113.0 + ((i - 30) / 40.0) * 5.0
            else:
                # Continuous SPLIT-adjusted ramp at the post-split scale.
                close = 113.0 + (i / 70.0) * 5.0
            bars.append(
                PriceBar(
                    timestamp=datetime.combine(d, datetime.min.time()),
                    open=close, high=close, low=close, close=close,
                    volume=10_000_000,
                )
            )
        return PriceHistory(symbol="NVDA", bars=bars)

    def test_unadjusted_bars_reproduce_phantom_drop(self):
        """Without split adjustment, the 10:1 NVDA-shaped discontinuity
        produces the documented ~-90% artifact. This pins the bug shape so
        a future regression in the consumer is detectable from the test.
        """
        from cents.cli.backtest import _calculate_forward_returns

        signal_date = date(2024, 5, 26)
        provider = MagicMock()
        provider.get_history.return_value = self._build_bars(signal_date, split_factor=10)

        returns = _calculate_forward_returns("NVDA", signal_date, provider)

        # The bug surfaced as -87.7%; assert we're firmly in the phantom-drop
        # regime so this test would have failed *before* the bc6c140 fix.
        assert "20d" in returns
        assert returns["20d"] < -0.80

    def test_split_adjusted_bars_produce_continuous_return(self):
        """With consistently SPLIT-adjusted bars, the same window returns a
        near-zero forward move — the property the fix delivers."""
        from cents.cli.backtest import _calculate_forward_returns

        signal_date = date(2024, 5, 26)
        provider = MagicMock()
        provider.get_history.return_value = self._build_bars(signal_date, split_factor=None)

        returns = _calculate_forward_returns("NVDA", signal_date, provider)

        assert "20d" in returns
        assert abs(returns["20d"]) < 0.10, (
            f"20d return {returns['20d']:.2%} suggests a split-adjustment "
            "regression — adjusted bars should never show ~-90% artifacts"
        )

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_history_no_data(self, mock_settings, mock_client_class):
        """Handle empty response."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_response = MagicMock()
        mock_response.data = {}

        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        history = provider.get_history("INVALID")

        assert history.symbol == "INVALID"
        assert len(history.bars) == 0

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_latest_price_success(self, mock_settings, mock_client_class):
        """Get latest price successfully."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_quote = MagicMock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.10

        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        price = provider.get_latest_price("AAPL")

        assert price == 150.05  # Midpoint

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_latest_price_no_data(self, mock_settings, mock_client_class):
        """Returns None when no quote available."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {}
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        price = provider.get_latest_price("INVALID")

        assert price is None

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_latest_price_exception(self, mock_settings, mock_client_class):
        """Returns None on exception."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        price = provider.get_latest_price("AAPL")

        assert price is None

    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_latest_price_ask_only_falls_back_to_close(self, mock_settings, mock_client_class):
        """Fall back to last close when only ask available (no bid)."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_quote = MagicMock()
        mock_quote.bid_price = None
        mock_quote.ask_price = 150.10

        # Mock bar data for fallback
        mock_bar = MagicMock()
        mock_bar.close = 149.50

        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_client.get_stock_bars.return_value.data = {"AAPL": [mock_bar]}
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        price = provider.get_latest_price("AAPL")

        # Should use last close, not ask
        assert price == 149.50

    @patch("cents.data.alpaca.cached_request", side_effect=lambda p, e, pa, fn, **kw: fn())
    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_history_with_as_of(self, mock_settings, mock_client_class, mock_cache):
        """Get historical price data with as_of date."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2024, 1, 15, 16, 0)
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 149.0
        mock_bar.close = 154.0
        mock_bar.volume = 1000000

        mock_response = MagicMock()
        mock_response.data = {"AAPL": [mock_bar]}

        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        history = provider.get_history("AAPL", days=30, as_of=date(2024, 1, 15))

        assert history.symbol == "AAPL"
        assert len(history.bars) == 1
        assert history.bars[0].close == 154.0

        # Verify the request used the as_of date
        call_args = mock_client.get_stock_bars.call_args
        request = call_args[0][0]
        # End date should be based on as_of
        assert request.end.date() == date(2024, 1, 15)

    @patch("cents.data.alpaca.cached_request", side_effect=lambda p, e, pa, fn, **kw: fn())
    @patch("cents.data.alpaca.StockHistoricalDataClient")
    @patch("cents.data.alpaca.get_settings")
    def test_get_latest_price_with_as_of(self, mock_settings, mock_client_class, mock_cache):
        """Get historical price using as_of date."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        # Mock bar data for historical lookup
        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2024, 1, 15, 16, 0)
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 149.0
        mock_bar.close = 154.0
        mock_bar.volume = 1000000

        mock_response = MagicMock()
        mock_response.data = {"AAPL": [mock_bar]}

        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AlpacaPriceProvider()
        price = provider.get_latest_price("AAPL", as_of=date(2024, 1, 15))

        # Should return the close price from historical data
        assert price == 154.0
        # Should NOT call get_stock_latest_quote
        mock_client.get_stock_latest_quote.assert_not_called()
        # Should call get_stock_bars instead
        mock_client.get_stock_bars.assert_called_once()
