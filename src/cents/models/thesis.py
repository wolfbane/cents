"""Thesis domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class ThesisStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    INVALIDATED = "invalidated"


@dataclass
class Thesis:
    """An investment thesis - a testable hypothesis about an investment."""

    title: str
    hypothesis: str = ""
    status: ThesisStatus = ThesisStatus.OPEN
    conviction: float = 50.0  # 0-100
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update_conviction(self, delta: float) -> None:
        """Adjust conviction score, clamping to [0, 100]."""
        self.conviction = max(0.0, min(100.0, self.conviction + delta))
        self.updated_at = datetime.now()

    def close(self) -> None:
        """Mark thesis as closed."""
        self.status = ThesisStatus.CLOSED
        self.updated_at = datetime.now()

    def invalidate(self) -> None:
        """Mark thesis as invalidated."""
        self.status = ThesisStatus.INVALIDATED
        self.updated_at = datetime.now()
