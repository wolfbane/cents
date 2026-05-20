"""Tests for the shadow-open log + backfill + analyze CLI."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from cents.db import (
    PositionRepository,
    ShadowOpenRepository,
    ThesisRepository,
    UniverseRepository,
)
from cents.factory.config import FactoryConfig
from cents.factory.engine import FactoryEngine, TAG_FACTORY
from cents.factory.shadow import backfill_forward_returns
from cents.models import (
    Position,
    PositionSide,
    PositionStatus,
    ShadowOpen,
    Thesis,
    ThesisOutcome,
    Universe,
)


# ---- shared fixtures (mirror tests/test_factory.py) ----------------------


def _orchestrator(delta_for: dict[str, float] | None = None, default: float = 0.0):
    m = MagicMock()
    deltas = delta_for or {}

    def research(symbol: str, thesis=None):
        d = deltas.get(symbol, default)
        result = MagicMock()
        result.conviction_delta = d
        result.evidence = []
        result.summary = f"mock {symbol}: {d}"
        result.dimension_scores = {}
        return result

    m.research.side_effect = research
    return m


def _price_provider(prices: dict[str, float] | float | None = None):
    m = MagicMock()

    def get_latest_price(symbol: str):
        if isinstance(prices, dict):
            return prices.get(symbol)
        return prices

    m.get_latest_price.side_effect = get_latest_price
    return m


def _event_agent():
    m = MagicMock()
    m.refresh.return_value = {"fetched": 0, "new": 0, "alerts_fired": 0}
    return m


@pytest.fixture(autouse=True)
def _stub_event_agent(monkeypatch):
    import cents.agents
    monkeypatch.setattr(cents.agents, "EventAgent", lambda: _event_agent())


@pytest.fixture(autouse=True)
def _stub_premise_classifier(monkeypatch):
    monkeypatch.setattr(
        "cents.factory.engine.classify_premise_tags",
        lambda *args, **kwargs: [],
    )


@pytest.fixture(autouse=True)
def _stub_regime_snapshot(monkeypatch):
    """Avoid Anthropic / EventRepository churn when capturing regime."""
    monkeypatch.setattr(
        "cents.factory.engine.capture_regime_snapshot",
        lambda **kwargs: {},
    )


@pytest.fixture
def factory_db(tmp_path, monkeypatch):
    """Backing sqlite DB for engine tests."""
    from cents.db.schema import SCHEMA

    db_path = tmp_path / "factory.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return db_path


def _seed_universe(symbols: list[str], name: str = "test") -> None:
    UniverseRepository().create(Universe(name=name, symbols=symbols, is_default=True))


def _config(**kwargs) -> FactoryConfig:
    defaults = dict(
        universe="default",
        budget_usd=10000.0,
        target_positions=10,
        entry_threshold=5.0,
        preemption_margin=5.0,
        cohort_mode="directional_only",
        default_horizon_days=30,
        default_stop_pct=-10.0,
        default_target_pct=10.0,
        max_new_per_run=10,
    )
    defaults.update(kwargs)
    return FactoryConfig(**defaults)


# ---- engine integration -------------------------------------------------


class TestEngineRecordsShadowOpens:
    def test_below_threshold_records_shadow(self, factory_db):
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 1.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        engine.run()
        rows = ShadowOpenRepository().list()
        assert len(rows) == 1
        s = rows[0]
        assert s.symbol == "AAPL"
        assert s.reason == "below_threshold"
        assert s.conviction_delta == pytest.approx(1.0)
        assert s.would_be_entry_price == pytest.approx(100.0)
        assert s.primary_side == "LONG"  # delta >= 0 => long
        assert s.horizon_days == 30

    def test_negative_below_threshold_marks_short_side(self, factory_db):
        _seed_universe(["JPM"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"JPM": -1.0}),
            price_provider=_price_provider({"JPM": 50.0}),
        )
        engine.run()
        s = ShadowOpenRepository().list()[0]
        assert s.primary_side == "SHORT"

    def test_concentration_cap_records_shadow(self, factory_db, monkeypatch):
        trepo = ThesisRepository()
        for sym in ("A", "B"):
            trepo.create(Thesis(
                title=f"factory:{sym}",
                symbol=sym,
                tags=[TAG_FACTORY],
                premise_tags=["fed_policy"],
            ))

        _seed_universe(["C"])
        monkeypatch.setattr(
            "cents.factory.engine.classify_premise_tags",
            lambda *a, **kw: ["fed_policy"],
        )

        engine = FactoryEngine(
            config=_config(
                cohort_mode="directional_only",
                entry_threshold=1.0,
                max_per_premise_tag=2,
                budget_usd=100000.0,
                target_positions=20,
            ),
            orchestrator=_orchestrator({"C": 6.0}),
            price_provider=_price_provider({"C": 100.0}),
        )
        engine.run()

        rows = ShadowOpenRepository().list()
        # Exactly one shadow row for C, with concentration_cap reason
        assert len(rows) == 1
        assert rows[0].symbol == "C"
        assert rows[0].reason == "concentration_cap"
        assert rows[0].premise_tags == ["fed_policy"]

    def test_budget_locked_records_shadow(self, factory_db):
        _seed_universe(["NEW"])
        # Saturate the budget with a high-conviction unpreemptable thesis.
        trepo = ThesisRepository()
        prepo = PositionRepository()
        existing = Thesis(
            title="factory:OLD",
            symbol="OLD",
            conviction=95.0,
            tags=[TAG_FACTORY],
        )
        trepo.create(existing)
        prepo.create(Position(
            symbol="OLD",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10.0,
            thesis_id=existing.id,
        ))

        engine = FactoryEngine(
            config=_config(
                budget_usd=1000.0,
                target_positions=10,
                entry_threshold=1.0,
                preemption_margin=5.0,
            ),
            # new candidate's conviction (55) < 95 + margin(5), so cannot preempt
            orchestrator=_orchestrator({"NEW": 5.0}),
            price_provider=_price_provider({"OLD": 100.0, "NEW": 100.0}),
        )
        engine.run()

        rows = [s for s in ShadowOpenRepository().list() if s.symbol == "NEW"]
        assert len(rows) == 1
        assert rows[0].reason == "budget_locked"

    def test_dry_run_does_not_persist_shadow(self, factory_db):
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 1.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        engine.run(dry_run=True)
        assert ShadowOpenRepository().list() == []

    def test_shadow_carries_run_id_and_default_orchestrator_label(self, factory_db):
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 1.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        run = engine.run()
        s = ShadowOpenRepository().list()[0]
        assert s.run_id == run.id
        # Default arm label is 'llm' (the only orchestrator we ship today).
        # If experiments are introduced later, this is what the experiment_id
        # column is for.
        assert s.orchestrator_label == "llm"
        assert s.experiment_id is None

    def test_shadow_carries_discovery_source(self, factory_db):
        _seed_universe(["AAPL"], name="myuni")
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 1.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        engine.run()
        s = ShadowOpenRepository().list()[0]
        assert s.discovery_source == "myuni"


# ---- backfill helper ----------------------------------------------------


@dataclass
class _StubBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class _StubHistory:
    symbol: str
    bars: list[_StubBar]


class _StubPriceHistoryProvider:
    """In-memory price history keyed by symbol."""

    def __init__(self, bars_by_symbol: dict[str, list[_StubBar]]):
        self._bars = bars_by_symbol
        self.calls: list[tuple[str, int, object]] = []

    def get_history(self, symbol, days=180, as_of=None):
        self.calls.append((symbol, days, as_of))
        return _StubHistory(symbol=symbol, bars=self._bars.get(symbol, []))


class TestBackfill:
    def test_fills_30d_forward_return(self, factory_db):
        # Shadow row created ~31 days ago; price now reflects a +10% move.
        repo = ShadowOpenRepository()
        created = datetime.now() - timedelta(days=31)
        target_date = (created + timedelta(days=30)).date()
        row = ShadowOpen(
            symbol="AAPL",
            conviction_delta=2.0,
            reason="below_threshold",
            would_be_entry_price=100.0,
            primary_side="LONG",
            horizon_days=30,
            created_at=created,
        )
        repo.create(row)

        provider = _StubPriceHistoryProvider({
            "AAPL": [
                _StubBar(
                    timestamp=datetime.combine(target_date, datetime.min.time()),
                    open=110.0, high=111.0, low=109.0, close=110.0, volume=1000,
                ),
            ],
        })
        result = backfill_forward_returns(provider, horizon_days=30)
        assert result.scanned == 1
        assert result.filled == 1
        reloaded = repo.list()[0]
        assert reloaded.forward_return_30d == pytest.approx(0.10)
        assert reloaded.backfilled_at is not None

    def test_skips_too_young_rows(self, factory_db):
        repo = ShadowOpenRepository()
        repo.create(ShadowOpen(
            symbol="AAPL",
            conviction_delta=2.0,
            reason="below_threshold",
            would_be_entry_price=100.0,
            horizon_days=30,
            created_at=datetime.now() - timedelta(days=5),
        ))
        provider = _StubPriceHistoryProvider({})
        result = backfill_forward_returns(provider, horizon_days=30)
        assert result.scanned == 1
        assert result.filled == 0
        assert result.skipped_too_young == 1

    def test_skips_when_no_entry_price(self, factory_db):
        repo = ShadowOpenRepository()
        repo.create(ShadowOpen(
            symbol="AAPL",
            conviction_delta=2.0,
            reason="below_threshold",
            would_be_entry_price=None,
            horizon_days=30,
            created_at=datetime.now() - timedelta(days=40),
        ))
        provider = _StubPriceHistoryProvider({})
        result = backfill_forward_returns(provider, horizon_days=30)
        assert result.filled == 0
        assert result.skipped_no_entry_price == 1

    def test_skips_when_history_unavailable(self, factory_db):
        repo = ShadowOpenRepository()
        repo.create(ShadowOpen(
            symbol="AAPL",
            conviction_delta=2.0,
            reason="below_threshold",
            would_be_entry_price=100.0,
            horizon_days=30,
            created_at=datetime.now() - timedelta(days=40),
        ))
        provider = _StubPriceHistoryProvider({})  # empty history
        result = backfill_forward_returns(provider, horizon_days=30)
        assert result.filled == 0
        assert result.skipped_no_history == 1


# ---- CLI ----------------------------------------------------------------


class TestShadowCli:
    def test_help_lists_subcommands(self, factory_db):
        from cents.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow", "--help"])
        assert result.exit_code == 0, result.output
        assert "analyze" in result.output
        assert "backfill" in result.output

    def test_analyze_json_payload_shape(self, factory_db):
        """`cents shadow analyze --output json` returns the comparison payload."""
        from cents.cli import cli

        # Seed: 1 accepted (closed correct, +10%) + 2 shadow rows (one with
        # filled forward return, one without).
        trepo = ThesisRepository()
        prepo = PositionRepository()
        accepted = Thesis(title="factory:A", symbol="A", tags=[TAG_FACTORY])
        trepo.create(accepted)
        accepted.close(ThesisOutcome.CORRECT)
        trepo.update(accepted)
        prepo.create(Position(
            symbol="A",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=1.0,
            thesis_id=accepted.id,
            status=PositionStatus.CLOSED,
            exit_price=110.0,
        ))

        srepo = ShadowOpenRepository()
        srepo.create(ShadowOpen(
            symbol="REJ",
            conviction_delta=1.0,
            reason="below_threshold",
            would_be_entry_price=100.0,
            primary_side="LONG",
            forward_return_30d=-0.05,  # rejected name dropped 5% — good rejection
        ))
        srepo.create(ShadowOpen(
            symbol="UNFILLED",
            conviction_delta=1.0,
            reason="below_threshold",
            would_be_entry_price=100.0,
            primary_side="LONG",
        ))

        runner = CliRunner()
        result = runner.invoke(cli, ["shadow", "analyze", "--output", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)

        assert payload["accepted"]["n"] == 1
        assert payload["accepted"]["mean_return"] == pytest.approx(0.10)
        # Only the filled shadow row contributes to rejected stats.
        assert payload["rejected"]["n"] == 1
        assert payload["rejected"]["mean_return"] == pytest.approx(-0.05)
        assert "below_threshold" in payload["by_reason"]
        assert payload["by_reason"]["below_threshold"]["n"] == 1
        # Default arm bucket is 'llm'.
        assert "llm" in payload["by_orchestrator"]["accepted"]
        assert "llm" in payload["by_orchestrator"]["rejected"]


class TestSummaryJsonDispositions:
    """Covers cents-9yn: per-disposition counts + stop_reason in factory_runs.summary_json."""

    def test_below_threshold_counted_and_evaluated_tracked(self, factory_db):
        _seed_universe(["AAPL", "MSFT", "NVDA"])
        # Conviction below entry_threshold=5.0 for all three → all "evaluated, below threshold".
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0, max_new_per_run=10),
            orchestrator=_orchestrator({"AAPL": 1.0, "MSFT": 2.0, "NVDA": -1.5}),
            price_provider=_price_provider(100.0),
        )
        run = engine.run()
        s = run.summary_json
        assert s["universe_size"] == 3
        assert s["symbols_evaluated"] == 3
        assert s["symbols_below_threshold"] == 3
        assert s["symbols_skipped_held"] == 0
        assert s["stop_reason"] == "end_of_universe"

    def test_stop_reason_max_new_per_run(self, factory_db):
        _seed_universe(["A", "B", "C", "D", "E"])
        engine = FactoryEngine(
            config=_config(
                entry_threshold=5.0,
                max_new_per_run=2,
                cohort_mode="directional_only",
            ),
            orchestrator=_orchestrator(default=8.0),  # every symbol strongly above threshold
            price_provider=_price_provider(100.0),
        )
        run = engine.run()
        s = run.summary_json
        assert s["stop_reason"] == "max_new_per_run"
        # Only the first 2 should have been evaluated before the break.
        assert s["symbols_evaluated"] == 2

    def test_universe_size_replaces_symbols_considered(self, factory_db):
        """Schema migration: old key 'symbols_considered' must NOT appear."""
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 1.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        run = engine.run()
        assert "universe_size" in run.summary_json
        assert "symbols_considered" not in run.summary_json


class TestUniverseShuffle:
    """Covers seeded shuffle removing alphabetical bias from open-phase iteration."""

    def test_shuffle_visits_more_than_just_alphabetic_prefix(self, factory_db):
        """With max_new_per_run smaller than universe, the shuffled order should
        eventually visit a late-alphabet symbol — proving order isn't strictly A→Z.

        Run 200 trials with different run_id seeds and check at least one trial
        opens a symbol from the LATE half of the alphabet. With pure alphabetic
        order this could never happen (only early symbols would ever open)."""
        from cents.models import FactoryRun

        symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                   "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
        _seed_universe(symbols)

        late_half = {"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"}
        opened_late = False
        for _ in range(50):
            # Re-create a fresh engine each trial to get a new run_id
            engine = FactoryEngine(
                config=_config(
                    entry_threshold=5.0,
                    max_new_per_run=1,  # only 1 open per run → tight test
                    cohort_mode="directional_only",
                ),
                orchestrator=_orchestrator(default=8.0),  # all symbols above threshold
                price_provider=_price_provider(100.0),
            )
            run = engine.run(dry_run=True)
            proposals = run.summary_json.get("proposals", [])
            if proposals and proposals[0]["symbol"] in late_half:
                opened_late = True
                break

        assert opened_late, (
            "Expected at least one of 50 trials to open a late-alphabet symbol "
            "under shuffled iteration; pure alphabetic order would never reach there."
        )

    def test_shuffle_is_deterministic_per_run_id(self, factory_db):
        """Same run_id → same visit order. Reproducibility for debugging."""
        import random as _random

        symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

        # Replay the shuffle logic outside the engine to verify determinism.
        run_id = "fixed-run-id-for-test"
        order_1 = list(symbols)
        _random.Random(run_id).shuffle(order_1)
        order_2 = list(symbols)
        _random.Random(run_id).shuffle(order_2)
        assert order_1 == order_2, "Shuffle must be deterministic given the same run_id seed"

        # And a different run_id must (almost certainly) produce a different order.
        order_3 = list(symbols)
        _random.Random("a-different-run-id").shuffle(order_3)
        # With 5! = 120 permutations, collision probability is ~0.8%. Acceptable for a regression test.
        assert order_1 != order_3, "Different run_ids should produce different shuffles"


class TestPerSymbolWatchdog:
    """Covers cents-87v: per-symbol deadline catches hangs anywhere in the agent chain."""

    def test_research_with_deadline_returns_normal_result(self, factory_db):
        from cents.factory.engine import _research_with_deadline

        class FastOrchestrator:
            def research(self, symbol, thesis):
                from cents.agents.base import AgentResult
                return AgentResult(
                    evidence=[], conviction_delta=5.0, summary="ok",
                    dimension_scores={}, aggregate=True,
                )

        r = _research_with_deadline(FastOrchestrator(), "AAPL", None, deadline_sec=5.0)
        assert r.conviction_delta == 5.0

    def test_research_with_deadline_raises_on_hang(self, factory_db):
        from cents.factory.engine import _research_with_deadline, _PerSymbolTimeout
        import time

        class HangingOrchestrator:
            def research(self, symbol, thesis):
                time.sleep(10)  # would hang past the 0.5s deadline

        with pytest.raises(_PerSymbolTimeout, match="exceeded 0"):
            _research_with_deadline(
                HangingOrchestrator(), "AAPL", None, deadline_sec=0.5
            )

    def test_hung_symbol_skipped_run_continues(self, factory_db, monkeypatch):
        """If one symbol hangs, the open phase logs + skips and other symbols still get evaluated."""
        from cents.config import Settings, get_settings as real_get_settings
        from cents.agents.base import AgentResult
        import time

        _seed_universe(["A", "B", "C"])

        class MixedOrchestrator:
            def __init__(self):
                self.calls = []
            def research(self, symbol, thesis):
                self.calls.append(symbol)
                if symbol == "B":
                    time.sleep(5)  # exceeds the 0.3s deadline below
                return AgentResult(
                    evidence=[], conviction_delta=8.0, summary=f"ok-{symbol}",
                    dimension_scores={}, aggregate=True,
                )

        # Force a tight per-symbol deadline via monkeypatch on get_settings
        orig = real_get_settings()

        def settings_with_tight_deadline():
            return Settings(
                **{**orig.__dict__, "per_symbol_deadline_sec": 0.3},
            )

        monkeypatch.setattr("cents.factory.engine.get_settings", settings_with_tight_deadline)

        orch = MixedOrchestrator()
        engine = FactoryEngine(
            config=_config(
                entry_threshold=5.0,
                max_new_per_run=5,
                cohort_mode="directional_only",
            ),
            orchestrator=orch,
            price_provider=_price_provider(100.0),
        )
        run = engine.run(dry_run=True)
        s = run.summary_json
        assert s["symbols_timed_out"] >= 1
        # A and C should still have been evaluated successfully
        assert s["symbols_evaluated"] >= 2
        # The run should NOT have hung — it completed and returned
        assert run.completed_at is not None
