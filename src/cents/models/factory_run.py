"""Factory run log domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class FactoryRun:
    """A single execution of the factory engine over a universe."""

    universe_name: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    theses_opened: int = 0
    theses_closed: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    preemptions: int = 0
    events_refreshed: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float | None = None
    dry_run: bool = False
    summary_json: dict = field(default_factory=dict)
    error: str | None = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def __post_init__(self) -> None:
        if not self.universe_name or not self.universe_name.strip():
            raise ValueError("FactoryRun.universe_name must be non-empty")
        self.universe_name = self.universe_name.strip().lower()
