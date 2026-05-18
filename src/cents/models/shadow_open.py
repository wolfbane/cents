"""ShadowOpen domain model — would-be theses the factory rejected.

When the factory's open phase rejects a candidate (below the entry threshold,
premise-tag concentration cap, budget locked with no preemption available),
we still record what would have happened: the orchestrator's signal, the
prevailing regime, and (later) the forward return of the symbol over the
default horizon. Used to measure whether the rejection rule itself adds value.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class ShadowOpen:
    """A rejected factory candidate, logged for later forward-return analysis.

    `reason` is one of:
      - "below_threshold": |conviction_delta| < entry_threshold
      - "concentration_cap": premise tag already at max_per_premise_tag
      - "budget_locked": notional + new position > budget and no preemption qualified
      - "no_price": no live price available, candidate dropped
    """

    symbol: str
    conviction_delta: float
    reason: str
    run_id: str | None = None
    would_be_entry_price: float | None = None
    primary_side: str | None = None
    premise_tags: list[str] = field(default_factory=list)
    premise_direction: dict[str, Any] = field(default_factory=dict)
    regime_snapshot: dict[str, Any] = field(default_factory=dict)
    orchestrator_label: str = "llm"
    experiment_id: str | None = None
    discovery_source: str | None = None
    horizon_days: int | None = None
    forward_return_30d: float | None = None
    forward_return_60d: float | None = None
    backfilled_at: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("ShadowOpen.symbol must be non-empty")
        self.symbol = self.symbol.strip().upper()
        if not self.reason or not self.reason.strip():
            raise ValueError("ShadowOpen.reason must be non-empty")
