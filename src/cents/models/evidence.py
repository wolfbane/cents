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
    # Provenance link to the LLM call that produced this evidence (if any).
    # Optional dict with keys: llm_call_id, model_snapshot, prompt_sha256,
    # input_sha256, output_sha256 — see cents.agents.base.make_provenance.
    # `None` means non-LLM evidence (e.g. keyword sentiment, FMP fundamentals).
    provenance: dict[str, str] | None = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    # cents-ct9k: when the agent's source data was MEASURED (vs when this
    # Evidence row was created). The orchestrator's age-weighted conviction
    # prefers this field if set so a technical agent reading 20-day-old bars
    # is weighted as "20 days old" rather than "today". Agents that source
    # from live APIs (sentiment NewsAPI, event Federal Register fetch-now)
    # can leave this None — defaulting to timestamp is correct for them.
    data_as_of: datetime | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
