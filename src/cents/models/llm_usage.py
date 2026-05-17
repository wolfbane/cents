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
