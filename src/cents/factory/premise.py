"""Classify a thesis's regime/policy dependencies and capture regime context.

The factory creates Theses autonomously; without populating premise_tags and
regime_snapshot at creation time, the regime-aware substrate (EventAgent's
PREMISE_INVALIDATION alerts, regime-stratified outcome analytics) has nothing
to bite on. This module fills that gap.

Two responsibilities:

- `classify_premise_tags(...)`: one LLM call per thesis against the EVENT_TAGS
  controlled vocabulary, asking which regime variables the thesis depends on.
  Falls back to [] when no anthropic key is configured.
- `capture_regime_snapshot(...)`: pure DB-read summary of recent event activity.
  Stored on every thesis so future analytics can stratify outcomes by regime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from cents.agents.base import extract_json_object
from cents.config import get_settings
from cents.db import EventRepository
from cents.llm_usage import record_llm_usage
from cents.models import EVENT_TAGS, EventPolarity


logger = logging.getLogger(__name__)

_LLM_MODEL = "claude-haiku-4-5"
_RECENT_EVENT_WINDOW_DAYS = 14
_MAX_PREMISE_TAGS = 5


def classify_premise_tags(
    symbol: str,
    summary: str,
    evidence_texts: list[str] | None = None,
    *,
    anthropic_client=None,
) -> list[str]:
    """Return 0-5 EVENT_TAGS that represent regime dependencies of the thesis.

    Returns [] when no anthropic client is configured or the call fails.
    """
    client = anthropic_client or _build_anthropic_client()
    if client is None:
        return []

    vocab = sorted(EVENT_TAGS)
    evidence_blob = "\n".join(f"- {e[:200]}" for e in (evidence_texts or [])[:5]) or "(no evidence)"
    prompt = (
        "Identify which regime variables this US-equities investment thesis depends on.\n\n"
        f"Symbol: {symbol}\n"
        f"Agent summary: {summary[:600]}\n"
        f"Evidence:\n{evidence_blob}\n\n"
        f"Choose 0-{_MAX_PREMISE_TAGS} tags from this controlled vocabulary — a tag belongs\n"
        "only if a federal action affecting that regime variable would materially shift the\n"
        "thesis's expected outcome (i.e., the premise could be invalidated):\n"
        f"{', '.join(vocab)}\n\n"
        'Return ONLY a JSON object: {"tags": [...]}\n'
        "Tags must come from the vocabulary verbatim. Return fewer tags rather than stretching."
    )

    try:
        response = client.messages.create(
            model=_LLM_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        record_llm_usage(response, agent="factory", operation="classify_premise", context=symbol)
        text = response.content[0].text.strip()
    except Exception as e:
        logger.debug("classify_premise_tags LLM call failed: %s", e)
        return []

    parsed = extract_json_object(text)
    if not parsed:
        return []
    raw = parsed.get("tags") or []
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, str) and t in EVENT_TAGS][:_MAX_PREMISE_TAGS]


def capture_regime_snapshot(*, event_repo: EventRepository | None = None, now: datetime | None = None) -> dict:
    """Snapshot of recent event activity for later regime-stratified analytics.

    Pure DB read — no external API calls. Captures top event tags + net polarity
    over a fixed lookback window so outcomes can later be stratified by the
    regime conditions the thesis was born into.
    """
    repo = event_repo or EventRepository()
    anchor = now or datetime.now()
    since = anchor - timedelta(days=_RECENT_EVENT_WINDOW_DAYS)
    events = repo.list_recent(since=since, limit=500)

    tag_counts: dict[str, int] = {}
    net_polarity = 0
    for event in events:
        for tag in event.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if event.polarity == EventPolarity.BULLISH:
            net_polarity += 1
        elif event.polarity == EventPolarity.BEARISH:
            net_polarity -= 1

    top_tags = dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])[:10])
    return {
        "captured_at": anchor.isoformat(),
        "recent_window_days": _RECENT_EVENT_WINDOW_DAYS,
        "recent_event_count": len(events),
        "top_event_tags": top_tags,
        "net_polarity": net_polarity,
    }


def _build_anthropic_client():
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    except ImportError:
        logger.warning("anthropic package not installed; premise classification disabled")
        return None
