"""Base agent class for research agents."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
import time
from typing import Callable, TypeVar
from urllib.error import URLError


_T = TypeVar("_T")


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response.

    Handles common malformations (trailing commas before `}` or `]`) that
    LLMs occasionally emit. Returns None if no recoverable object is found.
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*}", "}", candidate)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension
from cents.db import EvidenceRepository, ThesisRepository

# Maximum conviction delta any single agent can return (prevents wild swings)
MAX_CONVICTION_DELTA = 10.0

# Standard exceptions that agents should catch and handle gracefully.
# These are "recoverable" errors that shouldn't crash the CLI.
RECOVERABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    # Data structure errors
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    AttributeError,
    # JSON parsing errors
    json.JSONDecodeError,
    # Network errors
    URLError,
    TimeoutError,
    ConnectionError,
    OSError,  # Covers socket errors and other IO issues
)


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
    metadata: dict = field(default_factory=dict)  # Additional info (signal mode, etc.)

    def __post_init__(self):
        """Clamp conviction delta to prevent extreme values."""
        self.conviction_delta = clamp_conviction_delta(self.conviction_delta)


class BaseAgent(ABC):
    """Abstract base class for research agents."""

    name: str = "base"

    def __init__(self):
        # Lazy-initialized repositories (only created when needed for persistence)
        self._evidence_repo: EvidenceRepository | None = None
        self._thesis_repo: ThesisRepository | None = None

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

        last_exc: Exception | None = None
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
    def research(
        self, symbol: str, thesis: Thesis | None = None, as_of: date | None = None
    ) -> AgentResult:
        """
        Perform research on a symbol.

        Args:
            symbol: Stock ticker symbol
            thesis: Optional thesis to evaluate against
            as_of: Optional date for historical analysis (backtesting)

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
        thesis_id: str | None,
        content: str,
        source: str,
        evidence_type: EvidenceType = EvidenceType.NEUTRAL,
        confidence: float = 0.5,
        dimension: ThesisDimension | None = None,
        metadata: dict | None = None,
        symbol: str | None = None,
    ) -> Evidence:
        """Helper to create evidence with this agent's name."""
        return Evidence(
            thesis_id=thesis_id,
            symbol=symbol,
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

    def _accumulate_dimension(
        self, scores: dict[str, float], dimension: str, delta: float
    ) -> None:
        """Accumulate a delta into a dimension score.

        Args:
            scores: The dimension_scores dict to update
            dimension: The dimension key (e.g., "valuation", "quality", "risk")
            delta: The conviction delta to add
        """
        scores[dimension] = scores.get(dimension, 0) + delta
