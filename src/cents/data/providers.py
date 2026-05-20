"""Market data provider protocols and data classes."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, runtime_checkable


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

    def latest_close(self) -> float | None:
        """Get most recent close price."""
        return self.bars[-1].close if self.bars else None

    def close_at(self, days_ago: int) -> float | None:
        """Get close price N days ago (0 = most recent)."""
        idx = len(self.bars) - 1 - days_ago
        return self.bars[idx].close if 0 <= idx < len(self.bars) else None


@dataclass
class FundamentalsData:
    """Company fundamental data."""

    symbol: str
    name: str | None = None
    sector: str | None = None  # e.g., "Technology", "Healthcare", "Utilities"

    # Valuation
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None

    # Growth
    revenue_growth: float | None = None  # As decimal (0.20 = 20%)
    earnings_growth: float | None = None

    # Profitability
    profit_margin: float | None = None  # As decimal
    return_on_equity: float | None = None

    # Balance sheet
    debt_to_equity: float | None = None  # As percentage
    current_ratio: float | None = None

    # Analyst
    recommendation: str | None = None  # "buy", "hold", "sell", etc.

    # cents-dfx: True when the provider couldn't fetch one or more of the
    # core ratio endpoints (plan limit, 402/403, network) and the resulting
    # FundamentalsData has None values that would otherwise look identical
    # to "field genuinely null." Consumers can stratify outcomes by this
    # flag to detect quiet data degradation across cohorts.
    degraded: bool = False

    # Raw data for extensibility
    raw: dict = field(default_factory=dict)


@runtime_checkable
class PriceDataProvider(Protocol):
    """Protocol for price/volume data providers."""

    def get_history(
        self,
        symbol: str,
        days: int = 180,
        as_of: date | None = None,
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
        self, symbol: str, as_of: date | None = None
    ) -> float | None:
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
        self, symbol: str, as_of: date | None = None
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
