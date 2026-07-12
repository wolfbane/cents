"""Experiment model — a pre-registered hypothesis the factory is running against.

The point of registering an experiment is to make `cents factory analyze`
falsifiable. Once registered, the experiment carries:

- a frozen SHA of the factory.toml that was in effect at registration time;
- a pre-stated hypothesis + primary metric + minimum N per arm;
- a started_at timestamp.

The engine warns when the live factory.toml has drifted from the frozen
SHA — that's the discipline that prevents iterating on parameters
mid-experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


_ACTIVE = "active"
_FINALIZED = "finalized"
_ABANDONED = "abandoned"


@dataclass
class Experiment:
    """A pre-registered research experiment."""

    name: str
    hypothesis: str
    primary_metric: str
    minimum_n_per_arm: int
    frozen_config_sha: str
    frozen_config_json: str
    # JSON-serialised list of symbols resolved from the universe at registration
    # time. Without this, SCREENER-sourced universes drift daily (FMP TTM data
    # is daily_key-cached) so cohorts at week 4 are over a different population
    # than cohorts at week 1 — confounding any between-arm comparison. When set,
    # the engine uses this list verbatim for the experiment's duration. Empty
    # string means "no freeze; resolve live" (legacy + non-screener universes).
    frozen_universe_json: str = ""
    # Canonical behavioural-payload JSON the frozen_config_sha was computed
    # over (v0.13). Kept so drift reporting can diff the current payload
    # against the frozen one and name exactly which keys moved, instead of
    # showing two opaque hashes. Empty string = registered before this field
    # existed (SHA-only drift reporting).
    frozen_payload_json: str = ""
    stopping_rule: str = ""
    # Minimum calendar days before verdict_ready can fire. Set per-experiment
    # so pilots can use a shorter floor (e.g. 30) than full runs (e.g. 90).
    # Defaults to 14 for back-compat with experiments registered before this
    # field existed.
    minimum_calendar_days: int = 14
    started_at: datetime = field(default_factory=datetime.now)
    finalized_at: datetime | None = None
    verdict_json: str | None = None  # filled at finalize time
    status: str = _ACTIVE
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name is required")
        if self.minimum_n_per_arm <= 0:
            raise ValueError("minimum_n_per_arm must be positive")
        if self.minimum_calendar_days < 0:
            raise ValueError("minimum_calendar_days must be non-negative")
        if self.status not in {_ACTIVE, _FINALIZED, _ABANDONED}:
            raise ValueError(
                f"status must be one of {{active, finalized, abandoned}}, got {self.status!r}"
            )

    @property
    def is_active(self) -> bool:
        return self.status == _ACTIVE
