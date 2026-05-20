"""Tests for the factory's premise-tag classifier and regime snapshot."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from cents.db import EventRepository
from cents.factory.premise import (
    SECTOR_FALLBACK_TAGS,
    capture_regime_snapshot,
    classify_premise_tags,
)
from cents.models import EVENT_TAGS, Event, EventPolarity


class _FakeAnthropic:
    """Stand-in matching the anthropic.Anthropic().messages.create() shape."""

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = MagicMock()
        msg.content = [MagicMock(text=self._response_text)]
        msg.model = "claude-haiku-4-5"
        msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        return msg


class TestClassifyPremiseTags:
    def test_returns_controlled_vocab_tags(self, db_conn, monkeypatch):
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        client = _FakeAnthropic(
            '{"tags": ["tariffs.china", "ai_capex", "not_in_vocab"],'
            ' "directions": {"tariffs.china": "negative", "ai_capex": "positive"}}'
        )
        tags, directions = classify_premise_tags(
            "NVDA", "Bullish on AI capex", ["positive earnings"], anthropic_client=client
        )
        assert tags == ["tariffs.china", "ai_capex"]
        assert directions == {"tariffs.china": "negative", "ai_capex": "positive"}

    def test_falls_back_to_empty_without_client(self):
        # No anthropic_client passed and (in test env) no api key. Should return ([], {}).
        result = classify_premise_tags(
            "NVDA", "Bullish", [], anthropic_client=None
        )
        # Contract: must not raise; must be a 2-tuple of (list, dict).
        assert isinstance(result, tuple)
        tags, directions = result
        assert isinstance(tags, list)
        assert isinstance(directions, dict)

    def test_caps_at_five_tags(self, db_conn, monkeypatch):
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        # LLM returns 8 valid tags; classifier should cap at 5.
        many = [
            "tariffs.china", "ai_capex", "fed_policy", "energy_policy",
            "semis_policy", "tax_policy", "healthcare_policy", "crypto_policy",
        ]
        import json
        client = _FakeAnthropic(json.dumps({"tags": many}))
        tags, directions = classify_premise_tags("NVDA", "", [], anthropic_client=client)
        assert len(tags) == 5
        assert all(t in EVENT_TAGS for t in tags)
        assert directions == {}

    def test_handles_malformed_llm_response(self, db_conn, monkeypatch):
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        client = _FakeAnthropic("LLM hallucinated some random text with no JSON")
        tags, directions = classify_premise_tags("NVDA", "", [], anthropic_client=client)
        assert tags == []
        assert directions == {}

    def test_filters_directions_to_surviving_tags_only(self, db_conn, monkeypatch):
        """Directions whose tag failed vocab validation must be dropped."""
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        client = _FakeAnthropic(
            '{"tags": ["tariffs.china", "not_in_vocab"],'
            ' "directions": {"tariffs.china": "negative",'
            ' "not_in_vocab": "positive", "also_bad": "negative"}}'
        )
        tags, directions = classify_premise_tags(
            "NVDA", "", [], anthropic_client=client
        )
        assert tags == ["tariffs.china"]
        assert directions == {"tariffs.china": "negative"}

    def test_malformed_directions_degrades_gracefully(self, db_conn, monkeypatch):
        """Garbage in the 'directions' field shouldn't crash — return empty dict."""
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        client = _FakeAnthropic(
            '{"tags": ["tariffs.china"], "directions": "this is not a dict"}'
        )
        tags, directions = classify_premise_tags(
            "NVDA", "", [], anthropic_client=client
        )
        assert tags == ["tariffs.china"]
        assert directions == {}

    def test_invalid_direction_values_are_dropped(self, db_conn, monkeypatch):
        """Direction values outside {positive, negative} are silently dropped."""
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        client = _FakeAnthropic(
            '{"tags": ["tariffs.china", "ai_capex"],'
            ' "directions": {"tariffs.china": "bullish",'
            ' "ai_capex": "positive"}}'
        )
        tags, directions = classify_premise_tags(
            "NVDA", "", [], anthropic_client=client
        )
        assert tags == ["tariffs.china", "ai_capex"]
        assert directions == {"ai_capex": "positive"}

    def test_prompt_asks_for_directions(self, db_conn, monkeypatch):
        """Regression: the prompt should explicitly request per-tag directions."""
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        client = _FakeAnthropic('{"tags": []}')
        classify_premise_tags("NVDA", "", [], anthropic_client=client)
        assert client.calls, "expected the classifier to invoke the LLM"
        prompt = client.calls[0]["messages"][0]["content"]
        assert "directions" in prompt
        assert "positive" in prompt
        assert "negative" in prompt

    def test_records_llm_usage(self, db_conn, monkeypatch):
        from cents.db import LLMUsageRepository
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        # record_llm_usage uses LLMUsageRepository() — point it at our test conn.
        monkeypatch.setattr(
            "cents.llm_usage.LLMUsageRepository", lambda: LLMUsageRepository(db_conn)
        )
        client = _FakeAnthropic('{"tags": ["fed_policy"]}')
        classify_premise_tags("NVDA", "", [], anthropic_client=client)
        usage = LLMUsageRepository(db_conn).list_recent(limit=10)
        assert len(usage) == 1
        assert usage[0].agent == "factory"
        assert usage[0].operation == "classify_premise"
        assert usage[0].context == "NVDA"


class TestSectorFallback:
    """Sector-tag fallback when thesis text is too thin for the LLM (cents-heo).

    Without this, control-arm theses (random orchestrator, summary is one
    boilerplate line) silently get premise_tags=[], so EventAgent can never
    invalidate them — asymmetrically inflating the random arm's hit rate.
    """

    def test_sector_tags_used_when_summary_is_empty_and_no_client(
        self, db_conn, monkeypatch
    ):
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        # JPM is Financial Services → XLF. No client, sparse summary, side=long.
        monkeypatch.setattr(
            "cents.factory.sector_map.hedge_etf_for", lambda sym: "XLF"
        )
        tags, directions = classify_premise_tags(
            "JPM", "", [], anthropic_client=None, side="long"
        )
        assert tags == SECTOR_FALLBACK_TAGS["XLF"]
        # Long thesis → "positive" on every tag (BEARISH event invalidates).
        assert directions == {t: "positive" for t in tags}
        assert all(t in EVENT_TAGS for t in tags)

    def test_short_side_emits_negative_direction(self, db_conn, monkeypatch):
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        monkeypatch.setattr(
            "cents.factory.sector_map.hedge_etf_for", lambda sym: "XLK"
        )
        tags, directions = classify_premise_tags(
            "NVDA", "x", [], anthropic_client=None, side="short"
        )
        assert tags == SECTOR_FALLBACK_TAGS["XLK"]
        # Short thesis → "negative" on every tag (BULLISH event invalidates).
        assert directions == {t: "negative" for t in tags}

    def test_falls_back_when_llm_returns_empty_on_sparse_text(
        self, db_conn, monkeypatch
    ):
        """The LLM-arm path also falls back to sector tags on sparse input.

        Real-world case: a synthetic summary that happens to still go through
        the LLM (e.g. a tiny custom orchestrator with anthropic client wired
        in) should still get sector tags rather than nothing.
        """
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        monkeypatch.setattr(
            "cents.factory.sector_map.hedge_etf_for", lambda sym: "XLF"
        )
        client = _FakeAnthropic('{"tags": [], "directions": {}}')
        tags, directions = classify_premise_tags(
            "JPM", "short signal", [], anthropic_client=client, side="long"
        )
        assert tags == SECTOR_FALLBACK_TAGS["XLF"]
        assert directions == {t: "positive" for t in tags}

    def test_no_fallback_when_summary_has_real_content(
        self, db_conn, monkeypatch
    ):
        """LLM's "no regime dependency" answer on real text is respected.

        Above the sparseness threshold, an empty LLM result is meaningful —
        the thesis genuinely doesn't depend on any of the policy tags. Don't
        paper over that with a sector default.
        """
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        long_summary = "A " * 200  # well above the sparse threshold
        client = _FakeAnthropic('{"tags": [], "directions": {}}')
        tags, directions = classify_premise_tags(
            "JPM", long_summary, [], anthropic_client=client, side="long"
        )
        assert tags == []
        assert directions == {}

    def test_no_fallback_without_side_hint(self, db_conn, monkeypatch):
        """Back-compat: callers that don't pass `side` keep legacy ([], {})."""
        monkeypatch.setattr(
            "cents.factory.premise.EventRepository", lambda: EventRepository(db_conn)
        )
        monkeypatch.setattr(
            "cents.factory.premise._build_anthropic_client", lambda: None
        )
        tags, directions = classify_premise_tags(
            "JPM", "", [], anthropic_client=None
        )
        assert tags == []
        assert directions == {}

    def test_sector_tags_remain_a_subset_of_event_tags(self):
        """Regression: every fallback tag must exist in EVENT_TAGS verbatim.

        EVENT_TAGS is the controlled vocabulary that EventAgent tags events
        against. A typo here would silently produce premise_tags that never
        match any event — exactly the bug this fix is solving.
        """
        for etf, tags in SECTOR_FALLBACK_TAGS.items():
            for tag in tags:
                assert tag in EVENT_TAGS, (
                    f"SECTOR_FALLBACK_TAGS[{etf!r}] contains {tag!r} "
                    f"which is not in EVENT_TAGS"
                )


class TestCaptureRegimeSnapshot:
    def test_empty_event_store_returns_zero_counts(self, db_conn):
        snap = capture_regime_snapshot(event_repo=EventRepository(db_conn))
        assert snap["recent_event_count"] == 0
        assert snap["top_event_tags"] == {}
        assert snap["net_polarity"] == 0
        assert snap["recent_window_days"] == 14
        assert "captured_at" in snap

    def test_aggregates_recent_events_by_tag(self, db_conn):
        repo = EventRepository(db_conn)
        now = datetime.now()
        repo.create(Event(
            source="federal_register", source_id="a", event_type="EO",
            title="t1", occurred_at=now - timedelta(days=2),
            tags=["tariffs.china", "semis_policy"], polarity=EventPolarity.BEARISH,
        ))
        repo.create(Event(
            source="federal_register", source_id="b", event_type="EO",
            title="t2", occurred_at=now - timedelta(days=5),
            tags=["tariffs.china"], polarity=EventPolarity.BEARISH,
        ))
        repo.create(Event(
            source="federal_register", source_id="c", event_type="Rule",
            title="t3", occurred_at=now - timedelta(days=1),
            tags=["energy_policy"], polarity=EventPolarity.BULLISH,
        ))

        snap = capture_regime_snapshot(event_repo=repo, now=now)
        assert snap["recent_event_count"] == 3
        assert snap["top_event_tags"]["tariffs.china"] == 2
        assert snap["top_event_tags"]["semis_policy"] == 1
        assert snap["top_event_tags"]["energy_policy"] == 1
        # 1 bullish, 2 bearish → net = -1
        assert snap["net_polarity"] == -1

    def test_ignores_events_outside_window(self, db_conn):
        repo = EventRepository(db_conn)
        now = datetime.now()
        repo.create(Event(
            source="federal_register", source_id="old", event_type="EO",
            title="ancient", occurred_at=now - timedelta(days=60),
            tags=["tariffs.china"], polarity=EventPolarity.BEARISH,
        ))
        snap = capture_regime_snapshot(event_repo=repo, now=now)
        assert snap["recent_event_count"] == 0
        assert snap["top_event_tags"] == {}
