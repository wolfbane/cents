"""Position domain model."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from uuid import uuid4


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Position:
    """A paper or real position linked to a thesis."""

    symbol: str
    side: PositionSide
    entry_price: float
    size: float  # shares or dollar amount
    entry_date: date = field(default_factory=date.today)
    thesis_id: str | None = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float | None = None
    exit_date: date | None = None
    paper: bool = True
    notes: str = ""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {self.entry_price}")
        if self.exit_price is not None and self.exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {self.exit_price}")

    def close(self, exit_price: float, exit_date: date | None = None) -> None:
        """Close the position with an exit price."""
        if exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {exit_price}")
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.exit_date = exit_date or date.today()

    @property
    def pnl(self) -> float | None:
        """Calculate P&L if position is closed."""
        if self.exit_price is None:
            return None
        diff = self.exit_price - self.entry_price
        if self.side == PositionSide.SHORT:
            diff = -diff
        return diff * self.size

    @property
    def pnl_pct(self) -> float | None:
        """Calculate P&L percentage if position is closed."""
        if self.exit_price is None:
            return None
        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L given current market price."""
        diff = current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            diff = -diff
        return diff * self.size

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage given current market price."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100

    def market_value(self, current_price: float) -> float:
        """Calculate current market value of the position."""
        return current_price * self.size

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis of the position."""
        return self.entry_price * self.size
