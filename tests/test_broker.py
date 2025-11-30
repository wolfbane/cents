"""Tests for broker integration."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from cents.broker.alpaca import BrokerPosition, OrderResult, AlpacaClient, ALPACA_AVAILABLE
from cents.models import PositionSide, PositionStatus


class TestBrokerPosition:
    """Tests for BrokerPosition dataclass."""

    def test_create_long_position(self):
        """Create a long position."""
        pos = BrokerPosition(
            symbol="AAPL",
            qty=10.0,
            side="long",
            avg_entry_price=150.0,
            current_price=155.0,
            unrealized_pl=50.0,
            unrealized_plpc=3.33,
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 10.0
        assert pos.side == "long"
        assert pos.unrealized_pl == 50.0

    def test_create_short_position(self):
        """Create a short position."""
        pos = BrokerPosition(
            symbol="TSLA",
            qty=5.0,
            side="short",
            avg_entry_price=200.0,
            current_price=190.0,
            unrealized_pl=50.0,
            unrealized_plpc=5.0,
        )
        assert pos.side == "short"
        assert pos.unrealized_pl == 50.0


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_create_filled_order(self):
        """Create a filled order result."""
        order = OrderResult(
            order_id="abc123",
            symbol="AAPL",
            qty=10.0,
            side="buy",
            status="filled",
            filled_avg_price=150.50,
        )
        assert order.order_id == "abc123"
        assert order.status == "filled"
        assert order.filled_avg_price == 150.50

    def test_create_pending_order(self):
        """Create a pending order with no fill price."""
        order = OrderResult(
            order_id="def456",
            symbol="TSLA",
            qty=5.0,
            side="sell",
            status="pending",
        )
        assert order.filled_avg_price is None


@pytest.mark.skipif(not ALPACA_AVAILABLE, reason="alpaca-py not installed")
class TestAlpacaClient:
    """Tests for AlpacaClient wrapper."""

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_init_paper_trading(self, mock_settings, mock_client):
        """Initialize client for paper trading."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        client = AlpacaClient(paper=True)

        assert client.paper is True
        mock_client.assert_called_once_with("test_key", "test_secret", paper=True)

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_init_live_trading(self, mock_settings, mock_client):
        """Initialize client for live trading."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        client = AlpacaClient(paper=False)

        assert client.paper is False
        mock_client.assert_called_once_with("test_key", "test_secret", paper=False)

    @patch("cents.broker.alpaca.get_settings")
    def test_init_missing_api_key(self, mock_settings):
        """Raises ValueError when API key is missing."""
        mock_settings.return_value.alpaca_api_key = None
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            AlpacaClient()

    @patch("cents.broker.alpaca.get_settings")
    def test_init_missing_secret_key(self, mock_settings):
        """Raises ValueError when secret key is missing."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = None

        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            AlpacaClient()

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_get_account(self, mock_settings, mock_client_class):
        """Get account information."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_account = MagicMock()
        mock_account.buying_power = "10000.00"
        mock_account.cash = "5000.00"
        mock_account.portfolio_value = "15000.00"
        mock_account.equity = "15000.00"

        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        account = client.get_account()

        assert account["buying_power"] == 10000.00
        assert account["cash"] == 5000.00
        assert account["portfolio_value"] == 15000.00

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_get_positions(self, mock_settings, mock_client_class):
        """Get all open positions."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.avg_entry_price = "150.00"
        mock_pos.current_price = "155.00"
        mock_pos.unrealized_pl = "50.00"
        mock_pos.unrealized_plpc = "0.0333"

        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [mock_pos]
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        positions = client.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10.0
        assert positions[0].side == "long"
        assert abs(positions[0].unrealized_plpc - 3.33) < 0.01  # Converted from decimal

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_get_position_found(self, mock_settings, mock_client_class):
        """Get specific position by symbol."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.avg_entry_price = "150.00"
        mock_pos.current_price = "155.00"
        mock_pos.unrealized_pl = "50.00"
        mock_pos.unrealized_plpc = "0.0333"

        mock_client = MagicMock()
        mock_client.get_open_position.return_value = mock_pos
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        position = client.get_position("AAPL")

        assert position is not None
        assert position.symbol == "AAPL"

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_get_position_not_found(self, mock_settings, mock_client_class):
        """Returns None when position not found."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_client = MagicMock()
        mock_client.get_open_position.side_effect = Exception("No position")
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        position = client.get_position("AAPL")

        assert position is None

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_submit_buy_order(self, mock_settings, mock_client_class):
        """Submit a buy order."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.side.value = "buy"
        mock_order.status.value = "filled"
        mock_order.filled_avg_price = "150.50"

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        result = client.submit_order("AAPL", 10, "buy")

        assert result.order_id == "order123"
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.status == "filled"
        assert result.filled_avg_price == 150.50

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_submit_sell_order(self, mock_settings, mock_client_class):
        """Submit a sell order."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_order = MagicMock()
        mock_order.id = "order456"
        mock_order.symbol = "TSLA"
        mock_order.qty = "5"
        mock_order.side.value = "sell"
        mock_order.status.value = "pending"
        mock_order.filled_avg_price = None

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        result = client.submit_order("TSLA", 5, "sell")

        assert result.side == "sell"
        assert result.status == "pending"
        assert result.filled_avg_price is None

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_close_position(self, mock_settings, mock_client_class):
        """Close an entire position."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"

        mock_order = MagicMock()
        mock_order.id = "close123"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.side.value = "sell"
        mock_order.status.value = "filled"
        mock_order.filled_avg_price = "155.00"

        mock_client = MagicMock()
        mock_client.close_position.return_value = mock_order
        mock_client_class.return_value = mock_client

        client = AlpacaClient()
        result = client.close_position("AAPL")

        assert result.order_id == "close123"
        assert result.status == "filled"
        mock_client.close_position.assert_called_once_with("AAPL")

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_to_cents_position_long(self, mock_settings, mock_client_class):
        """Convert broker position to cents Position model."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"
        mock_client_class.return_value = MagicMock()

        client = AlpacaClient(paper=True)

        bp = BrokerPosition(
            symbol="AAPL",
            qty=10.0,
            side="long",
            avg_entry_price=150.0,
            current_price=155.0,
            unrealized_pl=50.0,
            unrealized_plpc=3.33,
        )

        position = client.to_cents_position(bp, thesis_id="test123")

        assert position.symbol == "AAPL"
        assert position.side == PositionSide.LONG
        assert position.entry_price == 150.0
        assert position.size == 10.0
        assert position.thesis_id == "test123"
        assert position.status == PositionStatus.OPEN
        assert position.paper is True
        assert "Synced from Alpaca" in position.notes

    @patch("cents.broker.alpaca.TradingClient")
    @patch("cents.broker.alpaca.get_settings")
    def test_to_cents_position_short(self, mock_settings, mock_client_class):
        """Convert short broker position to cents Position."""
        mock_settings.return_value.alpaca_api_key = "test_key"
        mock_settings.return_value.alpaca_secret_key = "test_secret"
        mock_client_class.return_value = MagicMock()

        client = AlpacaClient(paper=False)

        bp = BrokerPosition(
            symbol="TSLA",
            qty=-5.0,  # Negative for short
            side="short",
            avg_entry_price=200.0,
            current_price=190.0,
            unrealized_pl=50.0,
            unrealized_plpc=5.0,
        )

        position = client.to_cents_position(bp)

        assert position.side == PositionSide.SHORT
        assert position.size == 5.0  # abs value
        assert position.paper is False
        assert position.thesis_id is None


class TestAlpacaClientNoImport:
    """Tests for AlpacaClient when alpaca-py is not installed."""

    def test_init_raises_import_error_when_unavailable(self):
        """Raises ImportError when alpaca-py not available."""
        # Temporarily patch ALPACA_AVAILABLE to False
        with patch("cents.broker.alpaca.ALPACA_AVAILABLE", False):
            # Need to reload to pick up the patched value
            with pytest.raises(ImportError, match="alpaca-py not installed"):
                # Create a new instance that checks availability
                from cents.broker.alpaca import AlpacaClient as AC
                # The check happens at runtime in __init__
                ac = AC.__new__(AC)
                ac.__init__()
