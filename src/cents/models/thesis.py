"""Thesis domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class ThesisStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    INVALIDATED = "invalidated"


class Valuation(str, Enum):
    UNDERVALUED = "undervalued"
    FAIR = "fair"
    OVERVALUED = "overvalued"


class TimeHorizon(str, Enum):
    SHORT = "short"      # < 3 months
    MEDIUM = "medium"    # 3-12 months
    LONG = "long"        # > 12 months


class ThesisOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    UNCLEAR = "unclear"


@dataclass
class Thesis:
    """An investment thesis - a testable hypothesis about an investment."""

    title: str
    hypothesis: str = ""
    status: ThesisStatus = ThesisStatus.OPEN
    conviction: float = 50.0  # 0-100
    tags: list[str] = field(default_factory=list)
    # Structured thesis fields
    symbol: str | None = None
    business_quality: str | None = None
    valuation: Valuation | None = None
    moat: str | None = None
    time_horizon: TimeHorizon | None = None
    horizon_end: datetime | None = None
    key_risks: list[str] = field(default_factory=list)
    # Resolution triggers
    target_price: float | None = None
    stop_price: float | None = None
    outcome: ThesisOutcome | None = None
    closed_at: datetime | None = None
    # Metadata
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0.0 <= self.conviction <= 100.0:
            raise ValueError(f"conviction must be between 0 and 100, got {self.conviction}")
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError(f"target_price must be positive, got {self.target_price}")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError(f"stop_price must be positive, got {self.stop_price}")

    def update_conviction(self, delta: float) -> None:
        """Adjust conviction score, clamping to [0, 100]."""
        self.conviction = max(0.0, min(100.0, self.conviction + delta))
        self.updated_at = datetime.now()

    def close(self, outcome: ThesisOutcome | None = None) -> None:
        """Mark thesis as closed with optional outcome."""
        self.status = ThesisStatus.CLOSED
        self.outcome = outcome
        self.closed_at = datetime.now()
        self.updated_at = datetime.now()

    def invalidate(self) -> None:
        """Mark thesis as invalidated."""
        self.status = ThesisStatus.INVALIDATED
        self.updated_at = datetime.now()
