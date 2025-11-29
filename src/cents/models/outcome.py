"""Outcome domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class ThesisAccuracy(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    UNCLEAR = "unclear"


@dataclass
class Outcome:
    """Recorded outcome of a closed position with retrospective analysis."""

    position_id: str
    pnl: float
    pnl_pct: float
    thesis_accuracy: ThesisAccuracy = ThesisAccuracy.UNCLEAR
    agent_performance: dict[str, float] = field(default_factory=dict)  # agent -> score
    retrospective: str = ""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    recorded_at: datetime = field(default_factory=datetime.now)
