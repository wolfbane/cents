"""Market data provider protocols and data classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Protocol, runtime_checkable


@dataclass
class PriceBar:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class PriceHistory:
    """Historical price data for a symbol."""

    symbol: str
    bars: list[PriceBar]

    @property
    def closes(self) -> list[float]:
        """Get close prices."""
        return [b.close for b in self.bars]

    @property
    def volumes(self) -> list[int]:
        """Get volumes."""
        return [b.volume for b in self.bars]

    @property
    def highs(self) -> list[float]:
        """Get high prices."""
        return [b.high for b in self.bars]

    @property
    def lows(self) -> list[float]:
        """Get low prices."""
        return [b.low for b in self.bars]

    def latest_close(self) -> Optional[float]:
        """Get most recent close price."""
        return self.bars[-1].close if self.bars else None

    def close_at(self, days_ago: int) -> Optional[float]:
        """Get close price N days ago (0 = most recent)."""
        idx = len(self.bars) - 1 - days_ago
        return self.bars[idx].close if 0 <= idx < len(self.bars) else None


@dataclass
class FundamentalsData:
    """Company fundamental data."""

    symbol: str
    name: Optional[str] = None

    # Valuation
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None  # As decimal (0.20 = 20%)
    earnings_growth: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None  # As decimal
    return_on_equity: Optional[float] = None

    # Balance sheet
    debt_to_equity: Optional[float] = None  # As percentage
    current_ratio: Optional[float] = None

    # Analyst
    recommendation: Optional[str] = None  # "buy", "hold", "sell", etc.

    # Raw data for extensibility
    raw: dict = field(default_factory=dict)


@runtime_checkable
class PriceDataProvider(Protocol):
    """Protocol for price/volume data providers."""

    def get_history(
        self,
        symbol: str,
        days: int = 180,
        as_of: Optional[date] = None,
    ) -> PriceHistory:
        """
        Get historical price data.

        Args:
            symbol: Ticker symbol
            days: Number of days of history (default 180 = ~6 months)
            as_of: End date for history (default: today). For backtesting.

        Returns:
            PriceHistory with daily bars

        Raises:
            Exception on API errors after retries
        """
        ...

    def get_latest_price(
        self, symbol: str, as_of: Optional[date] = None
    ) -> Optional[float]:
        """
        Get current/latest price for symbol.

        Args:
            symbol: Ticker symbol
            as_of: Date to get price for (default: current quote)
        """
        ...


@runtime_checkable
class FundamentalsDataProvider(Protocol):
    """Protocol for company fundamentals data providers."""

    def get_fundamentals(
        self, symbol: str, as_of: Optional[date] = None
    ) -> FundamentalsData:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Ticker symbol
            as_of: Date to get fundamentals for (default: latest).
                   For backtesting, returns most recent quarterly data
                   as of that date.

        Returns:
            FundamentalsData (fields may be None if unavailable)

        Raises:
            Exception on API errors after retries
        """
        ...
