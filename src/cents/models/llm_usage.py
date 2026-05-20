"""LLMUsage model — captures one row per Anthropic API call for cost/usage reporting.

Tokens are stored raw; cost is derived at report time via `cents.pricing` so
historical rows don't need backfilling when Anthropic adjusts rates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class LLMUsage:
    """A single Anthropic API call's token consumption."""

    model: str
    agent: str  # e.g. "sentiment", "event"
    operation: str  # e.g. "filter_articles", "score_article", "tag_event"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    context: str | None = None  # free-form short attribution (symbol, thesis_id, event_id)
    called_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def __post_init__(self) -> None:
        # cents-48ua: defend against string called_at sneaking in. SQL queries
        # do ISO-string comparison on this column and would silently misreport
        # cost metrics if any row had a non-ISO value. The dataclass default
        # is a real datetime, but external constructors (tests, migrations,
        # manual edits) could otherwise pass through bad values.
        if not isinstance(self.called_at, datetime):
            raise TypeError(
                f"LLMUsage.called_at must be a datetime, got "
                f"{type(self.called_at).__name__}: {self.called_at!r}"
            )
