"""Position domain model."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional
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
    thesis_id: Optional[str] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_date: Optional[date] = None
    paper: bool = True
    notes: str = ""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    def close(self, exit_price: float, exit_date: Optional[date] = None) -> None:
        """Close the position with an exit price."""
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.exit_date = exit_date or date.today()

    @property
    def pnl(self) -> Optional[float]:
        """Calculate P&L if position is closed."""
        if self.exit_price is None:
            return None
        diff = self.exit_price - self.entry_price
        if self.side == PositionSide.SHORT:
            diff = -diff
        return diff * self.size

    @property
    def pnl_pct(self) -> Optional[float]:
        """Calculate P&L percentage if position is closed."""
        if self.exit_price is None:
            return None
        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100
