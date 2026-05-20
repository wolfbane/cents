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
    INVALIDATED = "invalidated"
    PREEMPTED = "preempted"


class ThesisCohort(str, Enum):
    DIRECTIONAL = "directional"
    NEUTRAL = "neutral"


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
    # Cohort pairing (policy-neutral control group)
    cohort: ThesisCohort = ThesisCohort.DIRECTIONAL
    hedge_symbol: str | None = None
    paired_thesis_id: str | None = None
    # Hedge basis flag (cents-931f): how the hedge leg was sized.
    #   "beta"            — beta_match_hedge=true and the R² gate passed
    #   "dollar_fallback" — beta_match_hedge=true but estimation failed
    #                       (no history or R² below gate); the "neutral"
    #                       cohort is contaminated by a directional bet
    #                       on the hedge ETF and analytics should treat it
    #                       as such
    #   "dollar"          — beta_match_hedge=false (no estimation attempted)
    #   None              — directional thesis with no hedge
    hedge_basis: str | None = None
    # Regime / premise tracking
    premise_tags: list[str] = field(default_factory=list)
    # Per-tag polarity (Layer 2 #1): "positive" = thesis benefits when this
    # tag's events are bullish; "negative" = thesis benefits when bearish.
    # Tags not in this dict fall back to legacy unsigned-intersection matching.
    premise_direction: dict[str, str] = field(default_factory=dict)
    # Which classifier path produced premise_tags (cents-83xl): "llm" =
    # LLM produced ≥1 mapped tag; "fallback_sector" = sector defaults used
    # because the LLM was unavailable or returned nothing usable;
    # "fallback_empty" = neither path produced tags (also the default for
    # legacy/manually-created theses). Stratifier for `factory analyze`.
    premise_classification_source: str = "fallback_empty"
    regime_snapshot: dict = field(default_factory=dict)
    # Discovery (e.g. universe name or screener strategy that surfaced this symbol)
    discovery_source: str | None = None
    # Calibration (Layer 2 #3): logistic-regression-fit P(target hit before stop).
    # None when no calibration model existed at thesis-open time.
    calibrated_p_correct: float | None = None
    # Calibration model vintage — stamp the model's fit_at on every thesis it
    # touched so `factory analyze` can stratify outcomes by which calibration
    # produced them and surface model-staleness regressions. ISO timestamp.
    calibration_fit_at: str | None = None
    # Which orchestrator opened this thesis ("llm" default, "random" for the
    # control arm). Lets cohort analytics ask: does the LLM-arm cohort beat
    # the random-arm cohort? See cents/agents/random_orchestrator.py.
    orchestrator_label: str = "llm"
    # Which registered experiment was active when this thesis was opened.
    # None when no experiment is registered. See cents/experiments/.
    experiment_id: str | None = None
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
        if self.cohort == ThesisCohort.NEUTRAL and not self.hedge_symbol:
            raise ValueError("neutral cohort theses require a hedge_symbol")
        if self.calibrated_p_correct is not None and not 0.0 <= self.calibrated_p_correct <= 1.0:
            raise ValueError(
                f"calibrated_p_correct must be in [0, 1], got {self.calibrated_p_correct}"
            )

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
