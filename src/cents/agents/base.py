"""Base agent class for research agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from cents.models import Evidence, EvidenceType, Thesis
from cents.db import EvidenceRepository, ThesisRepository


@dataclass
class AgentResult:
    """Result from an agent's research."""

    evidence: list[Evidence]
    conviction_delta: float  # How much to adjust thesis conviction
    summary: str  # Human-readable summary


class BaseAgent(ABC):
    """Abstract base class for research agents."""

    name: str = "base"

    def __init__(self):
        self.evidence_repo = EvidenceRepository()
        self.thesis_repo = ThesisRepository()

    @abstractmethod
    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """
        Perform research on a symbol.

        Args:
            symbol: Stock ticker symbol
            thesis: Optional thesis to evaluate against

        Returns:
            AgentResult with evidence and conviction adjustment
        """
        pass

    def save_evidence(self, evidence: list[Evidence]) -> None:
        """Persist evidence to database."""
        for e in evidence:
            self.evidence_repo.create(e)

    def update_thesis_conviction(self, thesis: Thesis, delta: float) -> None:
        """Update thesis conviction based on research."""
        thesis.update_conviction(delta)
        self.thesis_repo.update(thesis)

    def create_evidence(
        self,
        thesis_id: str,
        content: str,
        source: str,
        evidence_type: EvidenceType = EvidenceType.NEUTRAL,
        confidence: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> Evidence:
        """Helper to create evidence with this agent's name."""
        return Evidence(
            thesis_id=thesis_id,
            agent=self.name,
            content=content,
            source=source,
            type=evidence_type,
            confidence=confidence,
            metadata=metadata or {},
        )
