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


class EventTagStatus(str, Enum):
    """How the event's ``tags`` list was populated.

    ``no_relevance`` and ``tagger_failed`` both produce ``tags == []`` and were
    previously indistinguishable — the no-thesis research path could silently
    suppress events that lost their tags to an LLM outage as if they were
    genuinely irrelevant. The status lets downstream filtering distinguish.
    """

    TAGGED = "tagged"           # Tagger ran, assigned >=1 vocabulary tag.
    NO_RELEVANCE = "no_relevance"  # Tagger ran, assigned no tags by design.
    TAGGER_FAILED = "tagger_failed"  # LLM raised / response unparseable.
    TAGGER_SKIPPED = "tagger_skipped"  # No Anthropic client configured.


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
    # Defaults to TAGGER_SKIPPED so events constructed without going through
    # the LLM tagger (tests, manual seeding, partial fixtures) don't lie about
    # having been tagged. The agent overwrites this when it successfully or
    # unsuccessfully runs the tagger.
    tag_status: EventTagStatus = EventTagStatus.TAGGER_SKIPPED
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    ingested_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        # Normalize symbols to uppercase
        self.affected_symbols = [s.upper() for s in self.affected_symbols]

    def matches_premise(
        self,
        thesis_premise_tags: list[str],
        thesis_premise_direction: dict[str, str] | None = None,
    ) -> bool:
        """True if this event invalidates a thesis with the given premise.

        Requires at least one shared tag, and:

        * BULLISH or BEARISH events match only when the event polarity is
          *opposite* the thesis's direction on at least one overlapping tag.
        * NEUTRAL or UNCLEAR events fall back to legacy unsigned set-
          intersection — an ambiguous-polarity tariff event that shares a
          tag with a tariff-dependent thesis should NOT silently fail to
          alert. The inline comment below documents this fail-open choice.

        If ``thesis_premise_direction`` is None or empty (e.g. legacy theses
        from before per-tag direction was stamped), the function also falls
        back to unsigned set-intersection so older callers keep working.
        """
        if not thesis_premise_tags or not self.tags:
            return False
        overlap = set(self.tags) & set(thesis_premise_tags)
        if not overlap:
            return False

        # Legacy behaviour when no direction info is available.
        if not thesis_premise_direction:
            return True

        # When the event's polarity is ambiguous (NEUTRAL/UNCLEAR), we cannot
        # tell whether it confirms or contradicts the thesis. The conservative
        # choice — and the choice that preserves prior alerting behaviour — is
        # to fall back to legacy unsigned-intersection matching. Failing
        # closed here means a tariff-ambiguity event silently fails to alert
        # a thesis that depends on tariff policy; that's the wrong default
        # for an invalidation surface.
        if self.polarity not in (EventPolarity.BULLISH, EventPolarity.BEARISH):
            return True

        # Polarised matching: only material (BULLISH/BEARISH) events that
        # oppose the thesis direction on a shared tag invalidate.
        opposite = "negative" if self.polarity == EventPolarity.BULLISH else "positive"
        for tag in overlap:
            if thesis_premise_direction.get(tag) == opposite:
                return True
        return False
