"""Base agent class for research agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Callable, Optional, TypeVar


_T = TypeVar("_T")

from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension
from cents.db import EvidenceRepository, ThesisRepository

# Maximum conviction delta any single agent can return (prevents wild swings)
MAX_CONVICTION_DELTA = 10.0


def clamp_conviction_delta(delta: float) -> float:
    """Clamp conviction delta to prevent extreme swings from a single agent."""
    return max(-MAX_CONVICTION_DELTA, min(MAX_CONVICTION_DELTA, delta))


@dataclass
class AgentResult:
    """Result from an agent's research."""

    evidence: list[Evidence]
    conviction_delta: float  # How much to adjust thesis conviction
    summary: str  # Human-readable summary
    dimension_scores: dict[str, float] = field(default_factory=dict)  # Per-dimension conviction deltas

    def __post_init__(self):
        """Clamp conviction delta to prevent extreme values."""
        self.conviction_delta = clamp_conviction_delta(self.conviction_delta)


class BaseAgent(ABC):
    """Abstract base class for research agents."""

    name: str = "base"

    def __init__(self):
        # Lazy-initialized repositories (only created when needed for persistence)
        self._evidence_repo: Optional[EvidenceRepository] = None
        self._thesis_repo: Optional[ThesisRepository] = None

    @property
    def evidence_repo(self) -> EvidenceRepository:
        """Get evidence repository, creating it lazily."""
        if self._evidence_repo is None:
            self._evidence_repo = EvidenceRepository()
        return self._evidence_repo

    @property
    def thesis_repo(self) -> ThesisRepository:
        """Get thesis repository, creating it lazily."""
        if self._thesis_repo is None:
            self._thesis_repo = ThesisRepository()
        return self._thesis_repo

    def _with_retries(
        self,
        func: Callable[[], "_T"],
        retries: int = 3,
        backoff: float = 0.5,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> "_T":
        """Execute a callable with simple exponential backoff retries."""

        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return func()
            except exceptions as exc:
                last_exc = exc
                if attempt == retries - 1:
                    break
                time.sleep(backoff * (2**attempt))

        if last_exc:
            raise last_exc
        raise RuntimeError("Retry helper exited without executing")

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
        dimension: Optional[ThesisDimension] = None,
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
            dimension=dimension,
            metadata=metadata or {},
        )

    def _error_result(self, symbol: str, error: Exception) -> AgentResult:
        """Create a standardized error result when research fails.

        Args:
            symbol: The symbol that was being researched
            error: The exception that occurred

        Returns:
            AgentResult with empty evidence, zero conviction delta, and error summary
        """
        return AgentResult(
            evidence=[],
            conviction_delta=0,
            summary=f"{symbol}: {self.name} failed - {error}",
        )
