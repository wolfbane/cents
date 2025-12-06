"""Evidence domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EvidenceType(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class ThesisDimension(str, Enum):
    """Thesis dimensions that evidence can relate to."""

    VALUATION = "valuation"  # Price vs intrinsic value
    QUALITY = "quality"  # Business quality, margins, growth
    MOAT = "moat"  # Competitive advantage durability
    TECHNICAL = "technical"  # Price action, momentum
    MACRO = "macro"  # Economic/sector factors
    SENTIMENT = "sentiment"  # News, analyst opinions
    RISK = "risk"  # Key risks to the thesis


@dataclass
class Evidence:
    """A piece of evidence produced by an agent for a thesis."""

    agent: str
    content: str
    source: str
    thesis_id: str | None = None  # None for standalone research
    symbol: str | None = None  # For standalone evidence without thesis
    type: EvidenceType = EvidenceType.NEUTRAL
    confidence: float = 0.5  # 0-1, agent's confidence in this evidence
    dimension: ThesisDimension | None = None  # Which thesis aspect this relates to
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
