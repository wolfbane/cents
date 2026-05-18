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
    """A paper or real position linked to a thesis.

    Extended fields support honest cost accounting and post-hoc analytics:

    - ``costs_applied_usd``: sum of commission + slippage + borrow + gap penalty
      across open and close, in dollars. P&L subtracts this; ``gross_pnl`` does not.
    - ``realized_exit_price``: the modeled fill price (after slippage / gap
      penalty). ``exit_price`` is the signal/trigger price; the two differ when
      a stop is breached by a gap. Both are preserved so analytics can decompose
      "stop logic" from "stop fill quality".
    - ``sizing_method``: how shares were chosen ("vol_scaled", "max_capped",
      "equal_dollar", "beta_matched_hedge"). Lets cohort analytics stratify on it.
    - ``borrow_rate_pa_pct``: the (synthetic, for paper) annual borrow rate at
      which this short was opened. None for longs.
    """

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
    # Cost-aware accounting fields (additive — None / 0.0 preserves prior behavior).
    costs_applied_usd: float = 0.0
    realized_exit_price: float | None = None
    sizing_method: str | None = None
    borrow_rate_pa_pct: float | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {self.entry_price}")
        if self.exit_price is not None and self.exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {self.exit_price}")
        if self.realized_exit_price is not None and self.realized_exit_price <= 0:
            raise ValueError(
                f"realized_exit_price must be positive, got {self.realized_exit_price}"
            )
        if self.costs_applied_usd < 0:
            raise ValueError(
                f"costs_applied_usd must be non-negative, got {self.costs_applied_usd}"
            )

    def close(
        self,
        exit_price: float,
        exit_date: date | None = None,
        *,
        realized_exit_price: float | None = None,
        costs_applied_usd: float | None = None,
    ) -> None:
        """Close the position with an exit price.

        Args:
            exit_price: The signal/trigger price (target, stop, premise close).
            realized_exit_price: Modeled fill price after slippage / gap penalty.
                Defaults to exit_price when not provided.
            costs_applied_usd: Cumulative open + close costs in dollars.
        """
        if exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {exit_price}")
        if realized_exit_price is not None and realized_exit_price <= 0:
            raise ValueError(
                f"realized_exit_price must be positive, got {realized_exit_price}"
            )
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.exit_date = exit_date or date.today()
        if realized_exit_price is not None:
            self.realized_exit_price = realized_exit_price
        if costs_applied_usd is not None:
            self.costs_applied_usd = max(0.0, costs_applied_usd)

    @property
    def fill_price(self) -> float | None:
        """Modeled fill price — realized_exit_price if set, else exit_price.

        Use this in analytics; ``exit_price`` is the trigger, ``fill_price`` is
        the dollar that actually settled.
        """
        if self.realized_exit_price is not None:
            return self.realized_exit_price
        return self.exit_price

    @property
    def gross_pnl(self) -> float | None:
        """P&L before any modeled costs (slippage, commission, borrow, gap)."""
        fill = self.fill_price
        if fill is None:
            return None
        diff = fill - self.entry_price
        if self.side == PositionSide.SHORT:
            diff = -diff
        return diff * self.size

    @property
    def pnl(self) -> float | None:
        """Net P&L after costs (slippage, commission, borrow, gap)."""
        gross = self.gross_pnl
        if gross is None:
            return None
        return gross - (self.costs_applied_usd or 0.0)

    @property
    def pnl_pct(self) -> float | None:
        """Net P&L percentage (against entry notional) after costs."""
        net = self.pnl
        if net is None or self.entry_price <= 0 or self.size <= 0:
            return None
        return net / (self.entry_price * self.size) * 100.0

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
