"""Evidence domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class EvidenceType(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


@dataclass
class Evidence:
    """A piece of evidence produced by an agent for a thesis."""

    thesis_id: str
    agent: str
    content: str
    source: str
    type: EvidenceType = EvidenceType.NEUTRAL
    confidence: float = 0.5  # 0-1, agent's confidence in this evidence
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
