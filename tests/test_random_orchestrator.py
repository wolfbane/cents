"""Tests for the random-orchestrator control arm (cents-k7j)."""

from __future__ import annotations

import sqlite3

import pytest
from click.testing import CliRunner

from cents.agents.random_orchestrator import RandomOrchestrator
from cents.agents.base import MAX_AGGREGATE_CONVICTION_DELTA
from cents.db import ThesisRepository, UniverseRepository
from cents.db.schema import SCHEMA
from cents.factory.config import FactoryConfig
from cents.factory.engine import FactoryEngine, TAG_FACTORY
from cents.models import Universe


# ---- unit tests for the orchestrator itself ----------------------------


class TestRandomOrchestratorUnit:
    def test_research_returns_aggregate_clamped_delta(self):
        orch = RandomOrchestrator(seed=42)
        result = orch.research("AAPL")
        assert -MAX_AGGREGATE_CONVICTION_DELTA <= result.conviction_delta <= MAX_AGGREGATE_CONVICTION_DELTA
        assert result.aggregate is True
        assert result.evidence == []
        assert "random control" in result.summary

    def test_seed_is_deterministic(self):
        a = RandomOrchestrator(seed=1234)
        b = RandomOrchestrator(seed=1234)
        for sym in ("AAPL", "NVDA", "TSLA"):
            assert a.research(sym).conviction_delta == b.research(sym).conviction_delta

    def test_different_seeds_diverge(self):
        a = RandomOrchestrator(seed=1)
        b = RandomOrchestrator(seed=2)
        # With these two seeds the first call must differ.
        assert a.research("AAPL").conviction_delta != b.research("AAPL").conviction_delta

    def test_orchestrator_label_is_random(self):
        assert RandomOrchestrator.orchestrator_label == "random"


# ---- engine integration ------------------------------------------------


@pytest.fixture
def factory_db(tmp_path, monkeypatch):
    db_path = tmp_path / "factory.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return db_path


@pytest.fixture(autouse=True)
def _stub_event_agent(monkeypatch):
    from unittest.mock import MagicMock
    import cents.agents
    fake = MagicMock()
    fake.refresh.return_value = {"fetched": 0, "new": 0, "alerts_fired": 0}
    monkeypatch.setattr(cents.agents, "EventAgent", lambda: fake)


@pytest.fixture(autouse=True)
def _stub_premise_classifier(monkeypatch):
    monkeypatch.setattr(
        "cents.factory.engine.classify_premise_tags",
        lambda *args, **kwargs: ([], {}),
    )


def _seed_universe(symbols: list[str]) -> None:
    UniverseRepository().create(Universe(name="test", symbols=symbols, is_default=True))


def _config(**kwargs) -> FactoryConfig:
    defaults = dict(
        universe="default",
        budget_usd=10000.0,
        target_positions=10,
        entry_threshold=5.0,
        cohort_mode="directional_only",
        default_horizon_days=30,
        default_stop_pct=-10.0,
        default_target_pct=10.0,
        max_new_per_run=10,
    )
    defaults.update(kwargs)
    return FactoryConfig(**defaults)


def _price_provider(prices: dict[str, float]):
    from unittest.mock import MagicMock
    m = MagicMock()
    m.get_latest_price.side_effect = lambda sym: prices.get(sym)
    return m


class TestRandomOrchestratorEngineIntegration:
    def test_random_arm_labels_opened_theses_as_random(self, factory_db):
        _seed_universe(["AAPL"])
        # Seed=0 → deterministic. With universe size 1 and ±30 range,
        # ~5/6 of seeds produce |delta| > 5 which clears entry_threshold.
        # Just retry seeds until we land on an open for the test.
        for seed in range(10):
            engine = FactoryEngine(
                config=_config(entry_threshold=5.0),
                orchestrator=RandomOrchestrator(seed=seed),
                price_provider=_price_provider({"AAPL": 100.0}),
            )
            engine.run()
            theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
            if theses:
                assert theses[0].orchestrator_label == "random"
                return
            # Reset for next attempt — clear opened-by-prior-iteration state.
            ThesisRepository().delete(theses[0].id) if theses else None
        pytest.fail("No random-arm seed produced an open in 10 tries — extremely unlikely")

    def test_random_arm_theses_get_sector_premise_tags(
        self, factory_db, monkeypatch
    ):
        """Regression for cents-heo selection bias.

        Random-orchestrator theses have one-line synthetic summaries, so the
        LLM classifier (when wired in) has nothing to anchor on and used to
        return premise_tags=[]. Without premise_tags, EventAgent's
        PREMISE_INVALIDATION never fires on the random arm — asymmetrically
        favouring its hit rate vs the LLM arm in the two-arm forward test.

        Fix: when the thesis summary is sparse, fall back to sector-derived
        tags so the random arm is invalidatable by events on the same footing
        as the LLM arm. This test asserts the engine path now records
        non-empty premise_tags for random-arm opens.
        """
        # Undo the autouse classifier stub — we want the real fallback path.
        # The classifier has no LLM client (no anthropic_api_key in test env)
        # so it goes straight to the sparse-summary fallback branch.
        from cents.factory.premise import (
            SECTOR_FALLBACK_TAGS,
            classify_premise_tags,
        )
        monkeypatch.setattr(
            "cents.factory.engine.classify_premise_tags", classify_premise_tags
        )
        # Pin sector lookup so we don't hit FMP — JPM as a financial.
        monkeypatch.setattr(
            "cents.factory.sector_map.hedge_etf_for", lambda sym: "XLF"
        )

        _seed_universe(["JPM"])
        # Iterate seeds until one clears the entry threshold so we can
        # actually inspect the opened thesis. Same pattern as the test above.
        for seed in range(10):
            engine = FactoryEngine(
                config=_config(entry_threshold=5.0),
                orchestrator=RandomOrchestrator(seed=seed),
                price_provider=_price_provider({"JPM": 100.0}),
            )
            engine.run()
            theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
            if theses:
                t = theses[0]
                assert t.orchestrator_label == "random"
                # cents-2xd4: sector fallback is capped at top-2 relevant tags
                # so the random arm's tag-set size matches LLM's typical 1-3.
                # Verify against the leading slice of the canonical list.
                from cents.factory.premise import _SECTOR_FALLBACK_TAG_CAP
                assert t.premise_tags == SECTOR_FALLBACK_TAGS["XLF"][:_SECTOR_FALLBACK_TAG_CAP]
                assert t.premise_tags_count == len(t.premise_tags)
                # Direction polarity must match the side the random arm took.
                # Either all "positive" (long) or all "negative" (short).
                assert t.premise_direction
                polarities = set(t.premise_direction.values())
                assert polarities in ({"positive"}, {"negative"})
                return
        pytest.fail("No random-arm seed produced an open in 10 tries")

    def test_default_engine_labels_theses_as_llm(self, factory_db):
        """Sanity: when no random orchestrator is injected, the label defaults to 'llm'."""
        from unittest.mock import MagicMock

        _seed_universe(["AAPL"])
        m = MagicMock()
        m.research.return_value = type(
            "AR",
            (),
            {
                "conviction_delta": 7.0,
                "evidence": [],
                "summary": "x",
                "dimension_scores": {},
            },
        )()
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=m,
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        engine.run()
        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].orchestrator_label == "llm"


# ---- CLI flag ----------------------------------------------------------


class TestRandomOrchestratorCLIFlag:
    def test_factory_run_random_flag_smoke(self, factory_db, monkeypatch, tmp_path):
        """--orchestrator random doesn't crash and produces a random-labeled run."""
        from cents.cli import cli

        # Point factory config at a tmp path so we don't read user state.
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        _seed_universe(["AAPL", "NVDA"])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "factory", "run",
                "--orchestrator", "random",
                "--orchestrator-seed", "42",
                "--output", "json",
            ],
        )
        assert result.exit_code == 0, result.output
        # Any theses that did open should be labeled 'random'.
        random_theses = [
            t for t in ThesisRepository().list()
            if TAG_FACTORY in t.tags and t.orchestrator_label == "random"
        ]
        # Don't require any specific number — depends on seed + entry threshold —
        # but if anything opened, it must be labeled correctly.
        llm_theses = [
            t for t in ThesisRepository().list()
            if TAG_FACTORY in t.tags and t.orchestrator_label == "llm"
        ]
        assert llm_theses == [], f"Random-arm run produced LLM-labeled theses: {llm_theses}"
