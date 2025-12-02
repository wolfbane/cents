"""Alpaca broker integration."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from cents.config import get_settings
from cents.exceptions import BrokerError, ConfigurationError
from cents.models import Position, PositionSide, PositionStatus

logger = logging.getLogger(__name__)


@dataclass
class BrokerPosition:
    """Position from broker."""
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class OrderResult:
    """Result of an order execution."""
    order_id: str
    symbol: str
    qty: float
    side: str
    status: str
    filled_avg_price: Optional[float] = None


class AlpacaClient:
    """Wrapper for Alpaca Trading API."""

    def __init__(self, paper: bool = True):
        """Initialize Alpaca client.

        Args:
            paper: Use paper trading (default True for safety)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Install with: pip install cents[broker]"
            )

        settings = get_settings()
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key

        if not api_key or not secret_key:
            raise ConfigurationError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required"
            )

        self.paper = paper
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account(self) -> dict:
        """Get account information."""
        account = self.client.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
        }

    def get_positions(self) -> list[BrokerPosition]:
        """Get all open positions."""
        positions = self.client.get_all_positions()
        return [
            BrokerPosition(
                symbol=p.symbol,
                qty=float(p.qty),
                side="long" if float(p.qty) > 0 else "short",
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc) * 100,
            )
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a specific symbol.

        Returns None if position not found, raises BrokerError on API failure.
        """
        try:
            p = self.client.get_open_position(symbol)
            return BrokerPosition(
                symbol=p.symbol,
                qty=float(p.qty),
                side="long" if float(p.qty) > 0 else "short",
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc) * 100,
            )
        except Exception as e:
            # Alpaca raises APIError with 404 when position not found
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                return None
            logger.warning("Failed to get position for %s: %s", symbol, e)
            return None

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" or "sell"
    ) -> OrderResult:
        """Submit a market order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = self.client.submit_order(request)

        return OrderResult(
            order_id=str(order.id),
            symbol=order.symbol,
            qty=float(order.qty),
            side=order.side.value,
            status=order.status.value,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        )

    def close_position(self, symbol: str) -> OrderResult:
        """Close an entire position."""
        order = self.client.close_position(symbol)
        return OrderResult(
            order_id=str(order.id),
            symbol=order.symbol,
            qty=float(order.qty),
            side=order.side.value,
            status=order.status.value,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        )

    def to_cents_position(self, bp: BrokerPosition, thesis_id: Optional[str] = None) -> Position:
        """Convert broker position to cents Position model.

        Note: Alpaca's position API doesn't expose the original entry date.
        The entry_date is set to today's date, which affects P&L duration
        calculations. For accurate tracking, manually update entry_date
        after syncing or open positions through cents first.
        """
        return Position(
            symbol=bp.symbol,
            side=PositionSide.LONG if bp.side == "long" else PositionSide.SHORT,
            entry_price=bp.avg_entry_price,
            size=abs(bp.qty),
            entry_date=date.today(),  # Alpaca API doesn't expose original entry date
            thesis_id=thesis_id,
            status=PositionStatus.OPEN,
            paper=self.paper,
            notes=f"Synced from Alpaca on {date.today()} (entry date unknown)",
        )
