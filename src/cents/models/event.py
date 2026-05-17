"""Event domain model — discrete, time-stamped market-relevant events.

Unlike per-symbol news (handled by SentimentAgent), Events are durable records
of policy/macro/regulatory actions sourced from authoritative feeds (Federal
Register, SEC EDGAR, court calendars, etc.) and cross-referenced against open
theses via their `premise_tags`.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EventPolarity(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


# Controlled vocabulary for event/premise tags.
# Both EventAgent (when tagging fetched events) and humans (when authoring
# thesis premise_tags) draw from this set. Matching is a string intersection,
# so the vocabulary must stay stable — additions are safe, renames are not.
EVENT_TAGS: frozenset[str] = frozenset({
    # Trade / tariffs
    "tariffs.universal",
    "tariffs.china",
    "tariffs.eu",
    "tariffs.mexico_canada",
    "tariffs.sectoral",
    "export_controls",
    "sanctions",
    # Fiscal / tax
    "tax_policy",
    "fiscal_spending",
    "debt_ceiling",
    "shutdown",
    # Monetary / FX
    "fed_policy",
    "rates",
    "dollar",
    # Sectoral policy
    "energy_policy",
    "energy_permitting",
    "clean_energy_credits",
    "semis_policy",
    "ai_policy",
    "healthcare_policy",
    "drug_pricing",
    "antitrust",
    "financial_regulation",
    "crypto_policy",
    "defense_spending",
    "labor_policy",
    "immigration_policy",
    # Legal / regulatory
    "scotus_ruling",
    "executive_order",
    # Macro themes (for theses that depend on these)
    "ai_capex",
    "reshoring",
    "deglobalization",
    "geopolitical_conflict",
})


@dataclass
class Event:
    """A discrete market-relevant event from an authoritative source."""

    source: str  # e.g. "federal_register", "sec_edgar", "scotus"
    source_id: str  # External stable ID, used with `source` for dedupe
    event_type: str  # e.g. "executive_order", "proposed_rule", "8-K", "opinion"
    title: str
    occurred_at: datetime
    summary: str = ""
    url: str = ""
    affected_symbols: list[str] = field(default_factory=list)
    affected_sectors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    polarity: EventPolarity = EventPolarity.UNCLEAR
    confidence: float = 0.5
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    ingested_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        # Normalize symbols to uppercase
        self.affected_symbols = [s.upper() for s in self.affected_symbols]

    def matches_premise(self, premise_tags: list[str]) -> bool:
        """True if any premise tag appears in this event's tags."""
        if not premise_tags or not self.tags:
            return False
        return bool(set(self.tags) & set(premise_tags))
