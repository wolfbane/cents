"""Tests for the factory engine + CLI."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cents.db import (
    AlertRepository,
    FactoryRunRepository,
    PositionRepository,
    ThesisRepository,
    UniverseRepository,
)
from cents.factory.config import FactoryConfig, load_factory_config, scaffold_factory_config
from cents.factory.engine import (
    FactoryEngine,
    TAG_FACTORY,
)
from cents.models import (
    Alert,
    AlertType,
    Position,
    PositionSide,
    PositionStatus,
    Thesis,
    ThesisCohort,
    ThesisOutcome,
    ThesisStatus,
    Universe,
    UniverseSource,
)


# ---- helpers ----------------------------------------------------------


def _orchestrator(delta_for: dict[str, float] | None = None, default: float = 0.0):
    """Build a mock orchestrator that returns a configured delta per symbol."""
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
    """Build a mock price provider. Pass a dict for per-symbol prices or a scalar."""
    m = MagicMock()

    def get_latest_price(symbol: str):
        if isinstance(prices, dict):
            return prices.get(symbol)
        return prices

    m.get_latest_price.side_effect = get_latest_price
    return m


def _event_agent(new: int = 0):
    """No-op EventAgent stand-in so tests don't touch the network."""
    m = MagicMock()
    m.refresh.return_value = {"fetched": 0, "new": new, "alerts_fired": 0}
    return m


@pytest.fixture(autouse=True)
def _stub_event_agent(monkeypatch):
    """Prevent EventAgent.refresh from hitting Federal Register during tests."""
    import cents.agents
    fake = _event_agent()
    monkeypatch.setattr(cents.agents, "EventAgent", lambda: fake)


@pytest.fixture(autouse=True)
def _stub_premise_classifier(monkeypatch):
    """Prevent the factory's per-thesis premise classifier from hitting Anthropic."""
    monkeypatch.setattr(
        "cents.factory.engine.classify_premise_tags",
        lambda *args, **kwargs: [],
    )


@pytest.fixture
def factory_db(tmp_path, monkeypatch):
    """Backing sqlite DB for engine tests with a default universe configured."""
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
    repo = UniverseRepository()
    repo.create(Universe(name=name, symbols=symbols, is_default=True))


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


# ---- tests ------------------------------------------------------------


class TestFactoryConfig:
    def test_invalid_cohort_mode_rejected(self):
        with pytest.raises(ValueError):
            FactoryConfig(cohort_mode="bogus")

    def test_position_size_derived(self):
        cfg = FactoryConfig(budget_usd=10000.0, target_positions=10)
        assert cfg.position_size_usd == 1000.0

    def test_scaffold_writes_default_toml(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "factory.toml"
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(cfg_path))
        scaffold_factory_config()
        assert cfg_path.exists()
        loaded = load_factory_config()
        assert loaded.cohort_mode in {"paired", "directional_only"}

    def test_scaffold_refuses_overwrite(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "factory.toml"
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(cfg_path))
        scaffold_factory_config()
        with pytest.raises(FileExistsError):
            scaffold_factory_config()


class TestEntryThreshold:
    def test_no_thesis_when_below_threshold(self, factory_db):
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 1.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 0
        assert ThesisRepository().list() == []

    def test_opens_when_at_or_above_threshold(self, factory_db):
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 1
        thesis = ThesisRepository().list()[0]
        assert TAG_FACTORY in thesis.tags
        assert thesis.symbol == "AAPL"


class TestPremiseConcentration:
    def test_caps_open_theses_per_premise_tag(self, factory_db, monkeypatch):
        """When a tag has hit the cap, new candidates with that tag must be skipped."""
        # Two existing factory theses, both tagged 'fed_policy'
        trepo = ThesisRepository()
        for sym in ("A", "B"):
            trepo.create(Thesis(
                title=f"factory:{sym}",
                symbol=sym,
                tags=[TAG_FACTORY],
                premise_tags=["fed_policy"],
            ))

        _seed_universe(["C", "D"])
        # Make the premise classifier return 'fed_policy' for C and an unrelated tag for D
        def fake_classify(symbol, summary, evidence_texts, **kwargs):
            return ["fed_policy"] if symbol == "C" else ["semis_policy"]
        monkeypatch.setattr(
            "cents.factory.engine.classify_premise_tags", fake_classify
        )

        engine = FactoryEngine(
            config=_config(
                cohort_mode="directional_only",
                entry_threshold=1.0,
                max_per_premise_tag=2,
                budget_usd=100000.0,
                target_positions=20,
            ),
            orchestrator=_orchestrator({"C": 6.0, "D": 6.0}),
            price_provider=_price_provider({"C": 100.0, "D": 100.0}),
        )
        run = engine.run()

        # Only D opens — C would be the 3rd fed_policy thesis (cap=2)
        opened = [t for t in ThesisRepository().list() if t.tags == [TAG_FACTORY] and t.symbol in {"C", "D"}]
        assert {t.symbol for t in opened} == {"D"}
        assert run.theses_opened == 1

    def test_cap_zero_disables_check(self, factory_db, monkeypatch):
        """max_per_premise_tag=0 means the check is off."""
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
                max_per_premise_tag=0,
                budget_usd=100000.0,
                target_positions=20,
            ),
            orchestrator=_orchestrator({"C": 6.0}),
            price_provider=_price_provider({"C": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 1


class TestDirectionAwareOpening:
    def test_bearish_signal_opens_short_directional(self, factory_db):
        _seed_universe(["JPM"])
        engine = FactoryEngine(
            config=_config(
                cohort_mode="directional_only",
                entry_threshold=1.0,
                default_target_pct=10.0,
                default_stop_pct=-5.0,
            ),
            orchestrator=_orchestrator({"JPM": -8.0}),
            price_provider=_price_provider({"JPM": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 1
        thesis = ThesisRepository().list()[0]
        assert thesis.target_price == pytest.approx(90.0)   # 10% drop wins a short
        assert thesis.stop_price == pytest.approx(105.0)    # 5% rise stops the short
        positions = PositionRepository().list()
        assert len(positions) == 1
        assert positions[0].side == PositionSide.SHORT
        assert positions[0].symbol == "JPM"

    def test_bearish_signal_opens_flipped_paired_legs(self, factory_db):
        _seed_universe(["JPM"])
        with patch("cents.factory.engine.hedge_etf_for", return_value="XLF"):
            engine = FactoryEngine(
                config=_config(
                    cohort_mode="paired",
                    entry_threshold=1.0,
                    budget_usd=100000.0,
                ),
                orchestrator=_orchestrator({"JPM": -8.0}),
                price_provider=_price_provider({"JPM": 100.0, "XLF": 50.0}),
            )
            engine.run()
        positions = PositionRepository().list()
        by_symbol = {p.symbol: p for p in positions}
        # Bearish paired: short underlying, long the hedge ETF.
        assert by_symbol["JPM"].side == PositionSide.SHORT
        assert by_symbol["XLF"].side == PositionSide.LONG

    def test_short_thesis_closes_correct_on_price_drop(self, factory_db):
        _seed_universe(["JPM"])
        engine = FactoryEngine(
            config=_config(
                cohort_mode="directional_only",
                entry_threshold=1.0,
                default_target_pct=10.0,
                default_stop_pct=-5.0,
            ),
            orchestrator=_orchestrator({"JPM": -8.0}),
            price_provider=_price_provider({"JPM": 100.0}),
        )
        engine.run()
        thesis = ThesisRepository().list()[0]
        # Now price drops 12% — short thesis should resolve CORRECT.
        engine2 = FactoryEngine(
            config=_config(entry_threshold=999.0),  # no new opens
            orchestrator=_orchestrator(),
            price_provider=_price_provider({"JPM": 88.0}),
        )
        engine2.run()
        assert ThesisRepository().get(thesis.id).outcome == ThesisOutcome.CORRECT

    def test_short_thesis_closes_incorrect_on_price_rise(self, factory_db):
        _seed_universe(["JPM"])
        engine = FactoryEngine(
            config=_config(
                cohort_mode="directional_only",
                entry_threshold=1.0,
                default_stop_pct=-5.0,
            ),
            orchestrator=_orchestrator({"JPM": -8.0}),
            price_provider=_price_provider({"JPM": 100.0}),
        )
        engine.run()
        thesis = ThesisRepository().list()[0]
        # Price climbs past stop (5% rise) — short thesis resolves INCORRECT.
        engine2 = FactoryEngine(
            config=_config(entry_threshold=999.0),
            orchestrator=_orchestrator(),
            price_provider=_price_provider({"JPM": 107.0}),
        )
        engine2.run()
        assert ThesisRepository().get(thesis.id).outcome == ThesisOutcome.INCORRECT


class TestMaxNewPerRun:
    def test_rate_limit_respected(self, factory_db):
        _seed_universe(["A", "B", "C"])
        engine = FactoryEngine(
            config=_config(entry_threshold=1.0, max_new_per_run=2, target_positions=100),
            orchestrator=_orchestrator(default=5.0),
            price_provider=_price_provider(100.0),
        )
        run = engine.run()
        assert run.theses_opened == 2


class TestBudgetAndPreemption:
    def test_opens_when_within_budget(self, factory_db):
        _seed_universe(["A"])
        engine = FactoryEngine(
            config=_config(budget_usd=1000.0, target_positions=10, entry_threshold=1.0),
            orchestrator=_orchestrator({"A": 5.0}),
            price_provider=_price_provider({"A": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 1
        assert run.positions_opened == 1

    def test_preempts_when_margin_exceeded(self, factory_db):
        _seed_universe(["B"])
        # Pre-seed a low-conviction factory-managed open thesis at full notional
        trepo = ThesisRepository()
        prepo = PositionRepository()
        existing = Thesis(title="factory:OLD", symbol="OLD", conviction=40.0, tags=[TAG_FACTORY])
        trepo.create(existing)
        prepo.create(Position(
            symbol="OLD", side=PositionSide.LONG, entry_price=100.0, size=10.0,
            thesis_id=existing.id,
        ))

        engine = FactoryEngine(
            config=_config(
                budget_usd=1000.0,
                target_positions=10,
                entry_threshold=1.0,
                preemption_margin=5.0,
            ),
            orchestrator=_orchestrator({"B": 10.0}),  # new conviction 60
            price_provider=_price_provider({"OLD": 100.0, "B": 100.0}),
        )
        run = engine.run()
        assert run.preemptions == 1
        assert run.theses_opened == 1
        # Old thesis was closed as PREEMPTED
        reloaded = trepo.get(existing.id)
        assert reloaded.status == ThesisStatus.CLOSED
        assert reloaded.outcome == ThesisOutcome.PREEMPTED
        assert "preempted" in reloaded.hypothesis.lower()

    def test_no_preemption_when_margin_not_met(self, factory_db):
        _seed_universe(["B"])
        trepo = ThesisRepository()
        prepo = PositionRepository()
        existing = Thesis(title="factory:OLD", symbol="OLD", conviction=58.0, tags=[TAG_FACTORY])
        trepo.create(existing)
        prepo.create(Position(
            symbol="OLD", side=PositionSide.LONG, entry_price=100.0, size=10.0,
            thesis_id=existing.id,
        ))

        engine = FactoryEngine(
            config=_config(
                budget_usd=1000.0,
                target_positions=10,
                entry_threshold=1.0,
                preemption_margin=5.0,
            ),
            orchestrator=_orchestrator({"B": 5.0}),  # new conviction 55 (NOT > 58 + 5)
            price_provider=_price_provider({"OLD": 100.0, "B": 100.0}),
        )
        run = engine.run()
        assert run.preemptions == 0
        assert run.theses_opened == 0
        assert trepo.get(existing.id).status == ThesisStatus.OPEN


class TestPairedMode:
    def test_paired_open_creates_both_legs(self, factory_db):
        _seed_universe(["NVDA"])
        with patch("cents.factory.engine.hedge_etf_for", return_value="XLK"):
            engine = FactoryEngine(
                config=_config(
                    cohort_mode="paired",
                    entry_threshold=1.0,
                    budget_usd=10000.0,
                    target_positions=10,
                ),
                orchestrator=_orchestrator({"NVDA": 5.0}),
                price_provider=_price_provider({"NVDA": 100.0, "XLK": 200.0}),
            )
            run = engine.run()

        assert run.theses_opened == 1
        assert run.positions_opened == 2
        theses = ThesisRepository().list()
        neutral = next(t for t in theses if t.cohort == ThesisCohort.NEUTRAL)
        assert neutral.symbol == "NVDA"
        assert neutral.hedge_symbol == "XLK"
        positions = PositionRepository().list()
        legs = [p for p in positions if p.thesis_id == neutral.id]
        assert {p.symbol for p in legs} == {"NVDA", "XLK"}
        assert {p.side for p in legs} == {PositionSide.LONG, PositionSide.SHORT}

    def test_paired_skipped_when_pair_wont_fit(self, factory_db):
        _seed_universe(["NVDA"])
        trepo = ThesisRepository()
        prepo = PositionRepository()
        # Saturate budget with a high-conviction position that can't be preempted
        existing = Thesis(title="factory:OLD", symbol="OLD", conviction=95.0, tags=[TAG_FACTORY])
        trepo.create(existing)
        prepo.create(Position(
            symbol="OLD", side=PositionSide.LONG, entry_price=100.0, size=10.0,
            thesis_id=existing.id,
        ))

        with patch("cents.factory.engine.hedge_etf_for", return_value="XLK"):
            engine = FactoryEngine(
                config=_config(
                    cohort_mode="paired",
                    entry_threshold=1.0,
                    budget_usd=1000.0,
                    target_positions=10,
                    preemption_margin=5.0,
                ),
                orchestrator=_orchestrator({"NVDA": 5.0}),  # 55 < 95+5
                price_provider=_price_provider({"OLD": 100.0, "NVDA": 100.0, "XLK": 200.0}),
            )
            run = engine.run()
        assert run.theses_opened == 0
        # The existing thesis is untouched
        assert trepo.get(existing.id).status == ThesisStatus.OPEN


class TestCloseTriggers:
    def _seed_open_thesis(self, **kwargs) -> Thesis:
        trepo = ThesisRepository()
        prepo = PositionRepository()
        t = Thesis(
            title="factory:T",
            symbol=kwargs.get("symbol", "T"),
            tags=[TAG_FACTORY],
            target_price=kwargs.get("target_price"),
            stop_price=kwargs.get("stop_price"),
            horizon_end=kwargs.get("horizon_end"),
        )
        trepo.create(t)
        prepo.create(Position(
            symbol=t.symbol, side=PositionSide.LONG, entry_price=100.0, size=1.0,
            thesis_id=t.id,
        ))
        return t

    def test_target_hit_closes_as_correct(self, factory_db):
        _seed_universe([])
        t = self._seed_open_thesis(target_price=110.0)
        engine = FactoryEngine(
            config=_config(entry_threshold=99.0),
            orchestrator=_orchestrator(),
            price_provider=_price_provider({"T": 120.0}),
        )
        engine.run()
        reloaded = ThesisRepository().get(t.id)
        assert reloaded.outcome == ThesisOutcome.CORRECT
        assert reloaded.status == ThesisStatus.CLOSED

    def test_stop_hit_closes_as_incorrect(self, factory_db):
        _seed_universe([])
        t = self._seed_open_thesis(stop_price=90.0)
        engine = FactoryEngine(
            config=_config(entry_threshold=99.0),
            orchestrator=_orchestrator(),
            price_provider=_price_provider({"T": 85.0}),
        )
        engine.run()
        reloaded = ThesisRepository().get(t.id)
        assert reloaded.outcome == ThesisOutcome.INCORRECT

    def test_horizon_expired_closes_as_unclear(self, factory_db):
        _seed_universe([])
        t = self._seed_open_thesis(
            horizon_end=datetime.now() - timedelta(days=1)
        )
        engine = FactoryEngine(
            config=_config(entry_threshold=99.0),
            orchestrator=_orchestrator(),
            price_provider=_price_provider({"T": 100.0}),
        )
        engine.run()
        reloaded = ThesisRepository().get(t.id)
        assert reloaded.outcome == ThesisOutcome.UNCLEAR

    def test_invalidation_alert_stale_by_one_day_is_ignored(self, factory_db):
        """The engine's 1-day staleness window says an alert older than
        ``thesis.updated_at - 1d`` should not invalidate. With wall-clock
        ``datetime.now()`` the boundary is untestable; freeze the clock
        so the relationship is pinned by elapsed time, not test runtime.
        """
        from freezegun import freeze_time

        _seed_universe([])

        # Set the clock so the thesis updated_at is exactly the reference
        # moment, and the alert is 25h old — outside the 1-day window.
        with freeze_time("2026-05-18 12:00:00"):
            t = self._seed_open_thesis()
            trepo = ThesisRepository()
            reloaded = trepo.get(t.id)
            reloaded.updated_at = datetime.now()
            trepo.update(reloaded)

        # Step back to seed the older alert.
        with freeze_time("2026-05-17 11:00:00"):
            AlertRepository().create(Alert(
                symbol="T",
                alert_type=AlertType.PREMISE_INVALIDATION,
                message="stale",
                data={"thesis_id": t.id},
                created_at=datetime.now(),
            ))

        # Re-enter the post-update window — the alert is now ~25h before
        # the thesis updated_at, outside the 1-day window.
        with freeze_time("2026-05-18 12:00:00"):
            engine = FactoryEngine(
                config=_config(entry_threshold=99.0),
                orchestrator=_orchestrator(),
                price_provider=_price_provider({"T": 100.0}),
            )
            engine.run()

        # Outcome should not be INVALIDATED — the stale alert must not fire.
        assert ThesisRepository().get(t.id).outcome != ThesisOutcome.INVALIDATED

    def test_invalidation_alert_closes_as_invalidated(self, factory_db):
        _seed_universe([])
        t = self._seed_open_thesis()
        AlertRepository().create(Alert(
            symbol="T",
            alert_type=AlertType.PREMISE_INVALIDATION,
            message="premise broken",
            data={"thesis_id": t.id},
            created_at=datetime.now(),
        ))
        # Ensure the thesis's updated_at is also recent
        trepo = ThesisRepository()
        reloaded = trepo.get(t.id)
        reloaded.updated_at = datetime.now()
        trepo.update(reloaded)

        engine = FactoryEngine(
            config=_config(entry_threshold=99.0),
            orchestrator=_orchestrator(),
            price_provider=_price_provider({"T": 100.0}),
        )
        engine.run()
        assert ThesisRepository().get(t.id).outcome == ThesisOutcome.INVALIDATED

    def test_invalidated_symbol_not_reopened_in_same_run(self, factory_db):
        """After close-as-invalidated, the same-run open phase must skip the symbol."""
        _seed_universe(["T"])
        t = self._seed_open_thesis()
        AlertRepository().create(Alert(
            symbol="T",
            alert_type=AlertType.PREMISE_INVALIDATION,
            message="premise broken",
            data={"thesis_id": t.id},
            created_at=datetime.now(),
        ))
        trepo = ThesisRepository()
        reloaded = trepo.get(t.id)
        reloaded.updated_at = datetime.now()
        trepo.update(reloaded)

        engine = FactoryEngine(
            config=_config(entry_threshold=1.0),  # would otherwise open T trivially
            orchestrator=_orchestrator({"T": 9.0}),
            price_provider=_price_provider({"T": 100.0}),
        )
        run = engine.run()
        assert run.theses_closed == 1
        assert run.theses_opened == 0, (
            "factory must not reopen a symbol invalidated earlier in the same run"
        )
        # T thesis is closed; no new open T thesis exists.
        open_for_T = [
            x for x in ThesisRepository().list(status=ThesisStatus.OPEN)
            if x.symbol == "T"
        ]
        assert open_for_T == []

    def test_invalidated_symbol_stays_skipped_after_other_opens_same_run(self, factory_db):
        """Round-5 regression: skip_symbols was dropped from held_symbols after
        every successful open. If a universe contains the same invalidated
        symbol twice (screener+watchlist overlap), the engine would reopen it
        on its second appearance. Universe here: T (invalidated), A (clean),
        T (duplicate). After closing+invalidating T and opening A, the second
        T must remain skipped.
        """
        _seed_universe(["T", "A", "T"])
        t = self._seed_open_thesis()
        AlertRepository().create(Alert(
            symbol="T",
            alert_type=AlertType.PREMISE_INVALIDATION,
            message="premise broken",
            data={"thesis_id": t.id},
            created_at=datetime.now(),
        ))
        trepo = ThesisRepository()
        reloaded = trepo.get(t.id)
        reloaded.updated_at = datetime.now()
        trepo.update(reloaded)

        engine = FactoryEngine(
            config=_config(entry_threshold=1.0, max_new_per_run=10),
            orchestrator=_orchestrator({"T": 9.0, "A": 9.0}),
            price_provider=_price_provider({"T": 100.0, "A": 50.0}),
        )
        run = engine.run()
        assert run.theses_closed == 1
        # A opens (1 thesis); T's second appearance must still be skipped.
        open_for_T = [
            x for x in ThesisRepository().list(status=ThesisStatus.OPEN)
            if x.symbol == "T"
        ]
        assert open_for_T == [], (
            "skip_symbols must persist across successful opens in the same run"
        )

    def test_invalidated_hedge_symbol_also_skipped(self, factory_db):
        """When a paired thesis is invalidated, its hedge symbol is also off-limits this run."""
        _seed_universe(["T", "XLK"])
        # Pre-open a paired (neutral) factory thesis for symbol T hedged with XLK,
        # then mark it invalidated via a PREMISE_INVALIDATION alert.
        trepo = ThesisRepository()
        from cents.models import ThesisCohort
        t = Thesis(
            title="factory:T/hedge:XLK",
            symbol="T",
            tags=[TAG_FACTORY],
            cohort=ThesisCohort.NEUTRAL,
            hedge_symbol="XLK",
        )
        trepo.create(t)
        AlertRepository().create(Alert(
            symbol="T",
            alert_type=AlertType.PREMISE_INVALIDATION,
            message="premise broken",
            data={"thesis_id": t.id},
            created_at=datetime.now(),
        ))
        reloaded = trepo.get(t.id)
        reloaded.updated_at = datetime.now()
        trepo.update(reloaded)

        engine = FactoryEngine(
            config=_config(entry_threshold=1.0, cohort_mode="directional_only"),
            orchestrator=_orchestrator({"T": 9.0, "XLK": 9.0}),
            price_provider=_price_provider({"T": 100.0, "XLK": 200.0}),
        )
        run = engine.run()
        assert run.theses_opened == 0, (
            "neither the invalidated symbol nor its hedge may reopen in the same run"
        )


class TestDryRun:
    def test_dry_run_writes_run_record_only(self, factory_db):
        _seed_universe(["A"])
        engine = FactoryEngine(
            config=_config(entry_threshold=1.0),
            orchestrator=_orchestrator({"A": 5.0}),
            price_provider=_price_provider({"A": 100.0}),
        )
        run = engine.run(dry_run=True)
        # No theses were persisted
        assert ThesisRepository().list() == []
        # A run row was written with dry_run=1
        runs = FactoryRunRepository().list()
        assert len(runs) == 1
        assert runs[0].dry_run is True
        # The proposed action is captured in summary_json
        proposals = runs[0].summary_json.get("proposals", [])
        assert any(p["symbol"] == "A" for p in proposals)


class TestIlliquidNameOpensInResearchMode:
    """Research mode: illiquid names are NOT filtered out — the engine
    records them so analytics can study whether illiquidity correlates with
    outcomes. (Pre-research-mode this gated; the test now documents that
    the gate is intentionally absent.)
    """

    def test_illiquid_name_opens(self, factory_db):
        from datetime import datetime as _dt, timedelta as _td

        from cents.data.providers import PriceBar, PriceHistory

        # Equal-dollar would be $100k / 10 = $10k. Required ADV at 5x = $50k.
        # ADV in the stub is ~$30k → equal-dollar gate FAILS.
        # Vol-scaled (huge annualized vol) collapses sized notional to <$1k
        # → required ADV < $5k. Gate PASSES.
        class _Provider:
            def get_latest_price(self, symbol):
                return 100.0

            def get_history(self, symbol, days):
                base = _dt.now()
                bars = []
                price = 100.0
                # 40 bars with ~10% daily swings → enormous annualized vol →
                # vol-scaled sizing collapses to far below equal-dollar.
                for i in range(40):
                    price = price * (1.10 if i % 2 == 0 else 1 / 1.10)
                    bars.append(PriceBar(
                        timestamp=base - _td(days=40 - i),
                        open=price, high=price * 1.01, low=price * 0.99,
                        close=price, volume=300,  # ADV ≈ $30k (below pre-fix req)
                    ))
                return PriceHistory(symbol=symbol, bars=bars)

        _seed_universe(["ILLIQ"])
        engine = FactoryEngine(
            config=_config(
                entry_threshold=1.0,
                cohort_mode="directional_only",
                budget_usd=100_000.0,
                target_positions=10,
                sizing_mode="vol_scaled",
                target_vol_pct_per_position=0.5,
                max_position_pct=5.0,
                min_adv_multiple=5.0,  # require ADV >= 5x sized notional
                liquidity_lookback_days=20,
            ),
            orchestrator=_orchestrator({"ILLIQ": 7.0}),
            price_provider=_Provider(),
        )
        run = engine.run()
        # Pre-fix this would have been gated out (sized vs $10k equal-dollar).
        # Post-fix the actual sized notional dwarfs the ADV requirement.
        assert run.theses_opened == 1, run.summary_json


class TestSkipHeldSymbols:
    def test_does_not_reopen_symbol_already_open(self, factory_db):
        _seed_universe(["AAPL"])
        trepo = ThesisRepository()
        trepo.create(Thesis(title="factory:AAPL", symbol="AAPL", tags=[TAG_FACTORY]))
        engine = FactoryEngine(
            config=_config(entry_threshold=1.0),
            orchestrator=_orchestrator({"AAPL": 5.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 0


class TestDiscoverySourceLabeling:
    def test_factory_thesis_records_universe_name_as_discovery_source(self, factory_db):
        _seed_universe(["AAPL"], name="myuni")
        engine = FactoryEngine(
            config=_config(entry_threshold=1.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        run = engine.run()
        assert run.theses_opened == 1
        thesis = ThesisRepository().list()[0]
        assert thesis.discovery_source == "myuni"

    def test_manually_created_thesis_has_no_discovery_source(self, factory_db):
        trepo = ThesisRepository()
        t = Thesis(title="hand-made", symbol="GOOG")
        trepo.create(t)
        retrieved = trepo.get(t.id)
        assert retrieved.discovery_source is None


class TestFactoryCli:
    def test_help_lists_subcommands(self, factory_db):
        from cents.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["factory", "--help"])
        assert "init" in result.output
        assert "run" in result.output
        assert "status" in result.output
        assert "analyze" in result.output

    def test_init_writes_config(self, tmp_path, factory_db, monkeypatch):
        from cents.cli import cli

        cfg_path = tmp_path / "factory.toml"
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(cfg_path))
        runner = CliRunner()
        result = runner.invoke(cli, ["factory", "init"])
        assert result.exit_code == 0, result.output
        assert cfg_path.exists()

    def test_status_runs(self, factory_db, monkeypatch, tmp_path):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        runner = CliRunner()
        # No runs yet; status should still succeed
        result = runner.invoke(cli, ["factory", "status"])
        assert result.exit_code == 0, result.output

    def test_analyze_separates_cohorts_and_excludes_preempted(self, factory_db, monkeypatch, tmp_path):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        # Directional: 1 correct, 1 preempted
        good = Thesis(title="factory:A", symbol="A", tags=[TAG_FACTORY])
        trepo.create(good)
        good.close(ThesisOutcome.CORRECT)
        trepo.update(good)

        preempted = Thesis(title="factory:B", symbol="B", tags=[TAG_FACTORY])
        trepo.create(preempted)
        preempted.close(ThesisOutcome.PREEMPTED)
        trepo.update(preempted)

        # Neutral cohort: 1 incorrect
        paired = Thesis(
            title="factory:C",
            symbol="C",
            tags=[TAG_FACTORY],
            cohort=ThesisCohort.NEUTRAL,
            hedge_symbol="SPY",
        )
        trepo.create(paired)
        paired.close(ThesisOutcome.INCORRECT)
        trepo.update(paired)

        runner = CliRunner()
        result = runner.invoke(cli, ["factory", "analyze", "--output", "json"])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        # win_rate ignores preempted: directional has 1 correct / 1 judged = 1.0
        assert payload["directional"]["win_rate"] == 1.0
        assert payload["directional"]["preempted"] == 1
        # neutral has 1 incorrect / 1 judged = 0.0
        assert payload["neutral"]["win_rate"] == 0.0

    def test_analyze_by_discovery(self, factory_db, monkeypatch, tmp_path):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        # Two theses from `value` screener: 1 correct + 1 incorrect.
        t1 = Thesis(title="factory:A", symbol="A", tags=[TAG_FACTORY], discovery_source="value")
        trepo.create(t1)
        t1.close(ThesisOutcome.CORRECT)
        trepo.update(t1)

        t2 = Thesis(title="factory:B", symbol="B", tags=[TAG_FACTORY], discovery_source="value")
        trepo.create(t2)
        t2.close(ThesisOutcome.INCORRECT)
        trepo.update(t2)

        # One thesis from `momentum` screener (low-N).
        t3 = Thesis(title="factory:C", symbol="C", tags=[TAG_FACTORY], discovery_source="momentum")
        trepo.create(t3)
        t3.close(ThesisOutcome.CORRECT)
        trepo.update(t3)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze", "--by", "discovery", "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        assert payload["by"] == ["discovery"]
        # Find the value bucket
        value_cell = next(c for c in payload["cells"] if c["discovery"] == "value")
        assert value_cell["metrics"]["opened"] == 2
        assert value_cell["metrics"]["win_rate"] == 0.5
        # low_n gates on `judged` (win_rate denominator), not `opened`.
        # Both theses are judged here (2 < 5 threshold).
        assert value_cell["metrics"]["judged"] == 2
        assert value_cell["metrics"]["low_n"] is True
        momentum_cell = next(c for c in payload["cells"] if c["discovery"] == "momentum")
        assert momentum_cell["metrics"]["opened"] == 1
        assert momentum_cell["metrics"]["judged"] == 1
        assert momentum_cell["metrics"]["low_n"] is True

    def test_analyze_cross_tab_discovery_cohort(self, factory_db, monkeypatch, tmp_path):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        # 1 directional value, 1 neutral value
        d = Thesis(title="factory:A", symbol="A", tags=[TAG_FACTORY], discovery_source="value")
        trepo.create(d)
        n = Thesis(
            title="factory:B",
            symbol="B",
            tags=[TAG_FACTORY],
            discovery_source="value",
            cohort=ThesisCohort.NEUTRAL,
            hedge_symbol="SPY",
        )
        trepo.create(n)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze", "--by", "discovery,cohort", "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        assert payload["by"] == ["discovery", "cohort"]
        # Two cells expected: value/directional and value/neutral
        cells = payload["cells"]
        assert len(cells) == 2
        keys = {(c["discovery"], c["cohort"]) for c in cells}
        assert keys == {("value", "directional"), ("value", "neutral")}

    def test_analyze_rejects_unknown_axis(self, factory_db, monkeypatch, tmp_path):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        runner = CliRunner()
        result = runner.invoke(cli, ["factory", "analyze", "--by", "garbage"])
        assert result.exit_code != 0
        assert "axis" in result.output.lower() or "garbage" in result.output.lower()

    def test_analyze_low_n_flagged(self, factory_db, monkeypatch, tmp_path):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        for i in range(2):
            trepo.create(Thesis(
                title=f"factory:S{i}",
                symbol=f"S{i}",
                tags=[TAG_FACTORY],
                discovery_source="rare",
            ))
        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze", "--by", "discovery", "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        rare = next(c for c in payload["cells"] if c["discovery"] == "rare")
        assert rare["metrics"]["low_n"] is True

    def test_analyze_cost_flag_off_by_default(self, factory_db, monkeypatch, tmp_path):
        """Without --include-cost-per-outcome, no new fields appear (back-compat)."""
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        trepo.create(Thesis(
            title="factory:A",
            symbol="A",
            tags=[TAG_FACTORY],
            discovery_source="value",
        ))
        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze", "--by", "discovery", "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        cell = payload["cells"][0]
        assert "llm_cost_per_opened" not in cell["metrics"]
        assert "llm_cost_per_judged" not in cell["metrics"]
        assert "llm_cost_per_correct" not in cell["metrics"]
        assert "unattributable_cost_usd" not in payload
        # And the cohort-default path is unchanged too.
        result2 = runner.invoke(cli, ["factory", "analyze", "--output", "json"])
        payload2 = json.loads(result2.output)
        assert "llm_cost_per_opened" not in payload2["directional"]
        assert "unattributable_cost_usd" not in payload2

    def test_analyze_cost_per_outcome_attribution(self, factory_db, monkeypatch, tmp_path):
        """LLM spend on a thesis's symbol during its open window is attributed
        to that thesis; ratios divide by opened/judged/correct."""
        from cents.cli import cli
        from cents.db import LLMUsageRepository
        from cents.models import LLMUsage

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        usage_repo = LLMUsageRepository()
        now = datetime.now()

        # Two LLM-arm theses on symbol A: 1 correct, 1 incorrect → 2 opened, 2 judged, 1 correct.
        t_correct = Thesis(
            title="factory:A1",
            symbol="A",
            tags=[TAG_FACTORY],
            discovery_source="value",
            orchestrator_label="llm",
        )
        # Force the lifetime so we can pin LLM-call timestamps inside it.
        t_correct.created_at = now - timedelta(days=10)
        t_correct.closed_at = now - timedelta(days=8)
        trepo.create(t_correct)
        t_correct.close(ThesisOutcome.CORRECT)
        # Re-set closed_at after close() overwrites it with `now`.
        t_correct.closed_at = now - timedelta(days=8)
        trepo.update(t_correct)

        t_incorrect = Thesis(
            title="factory:A2",
            symbol="A",
            tags=[TAG_FACTORY],
            discovery_source="value",
            orchestrator_label="llm",
        )
        t_incorrect.created_at = now - timedelta(days=5)
        t_incorrect.closed_at = now - timedelta(days=3)
        trepo.create(t_incorrect)
        t_incorrect.close(ThesisOutcome.INCORRECT)
        t_incorrect.closed_at = now - timedelta(days=3)
        trepo.update(t_incorrect)

        # 1M input + 1M output tokens on haiku-4-5 = $1 + $5 = $6 per call.
        # Stamp each call inside the corresponding thesis's lifetime.
        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context="A",
            called_at=now - timedelta(days=9),  # inside t_correct's lifetime
        ))
        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context="A",
            called_at=now - timedelta(days=4),  # inside t_incorrect's lifetime
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze",
            "--by", "discovery",
            "--include-cost-per-outcome",
            "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        cell = next(c for c in payload["cells"] if c["discovery"] == "value")
        m = cell["metrics"]
        # 2 opened, 2 judged, 1 correct, total spend = $12.
        assert m["opened"] == 2
        assert m["judged"] == 2
        assert m["llm_cost_per_opened"] == pytest.approx(6.0, rel=1e-6)
        assert m["llm_cost_per_judged"] == pytest.approx(6.0, rel=1e-6)
        assert m["llm_cost_per_correct"] == pytest.approx(12.0, rel=1e-6)
        # Top-level unattributable bucket exists and is zero in this fixture.
        assert payload["unattributable_cost_usd"] == 0.0

    def test_analyze_cost_random_arm_is_zero(self, factory_db, monkeypatch, tmp_path):
        """Random-arm cells accrue $0 cost because the random orchestrator emits
        no LLM calls (no row in llm_usage has the random thesis's symbol)."""
        from cents.cli import cli
        from cents.db import LLMUsageRepository
        from cents.models import LLMUsage

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        usage_repo = LLMUsageRepository()
        now = datetime.now()

        # LLM arm thesis on AAA.
        llm_t = Thesis(
            title="factory:AAA",
            symbol="AAA",
            tags=[TAG_FACTORY],
            orchestrator_label="llm",
        )
        llm_t.created_at = now - timedelta(days=10)
        trepo.create(llm_t)

        # Random arm thesis on BBB. Symbol differs; no LLM rows ever target it.
        rand_t = Thesis(
            title="factory:BBB",
            symbol="BBB",
            tags=[TAG_FACTORY],
            orchestrator_label="random",
        )
        rand_t.created_at = now - timedelta(days=10)
        trepo.create(rand_t)

        # Only the LLM arm's symbol has any usage rows.
        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context="AAA",
            called_at=now - timedelta(days=5),
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze",
            "--by", "orchestrator",
            "--include-cost-per-outcome",
            "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        cells = {c["orchestrator"]: c["metrics"] for c in payload["cells"]}
        assert cells["random"]["llm_cost_per_opened"] == 0.0
        assert cells["random"]["llm_cost_per_judged"] is None  # no judged in fixture
        assert cells["random"]["llm_cost_per_correct"] is None
        assert cells["llm"]["llm_cost_per_opened"] == pytest.approx(6.0, rel=1e-6)

    def test_analyze_unattributable_cost_surfaced(self, factory_db, monkeypatch, tmp_path):
        """LLM calls with no thesis_id/symbol match show up in `unattributable_cost_usd`,
        NOT in any cell's per-cost figures."""
        from cents.cli import cli
        from cents.db import LLMUsageRepository
        from cents.models import LLMUsage

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        usage_repo = LLMUsageRepository()
        now = datetime.now()

        # One factory thesis on symbol X.
        t = Thesis(
            title="factory:X",
            symbol="X",
            tags=[TAG_FACTORY],
            discovery_source="value",
            orchestrator_label="llm",
        )
        t.created_at = now - timedelta(days=10)
        trepo.create(t)

        # Ad-hoc call: context is a symbol no factory thesis owns.
        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context="UNRELATED",
            called_at=now - timedelta(days=5),
        ))
        # No-context call (e.g., a tagging sweep) — also unattributable.
        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="event",
            operation="tag_event",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context=None,
            called_at=now - timedelta(days=5),
        ))
        # And one legitimately attributable call on X for contrast.
        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context="X",
            called_at=now - timedelta(days=5),
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze",
            "--by", "discovery",
            "--include-cost-per-outcome",
            "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        # Two of three rows are unattributable: 2 × $6 = $12.
        assert payload["unattributable_cost_usd"] == pytest.approx(12.0, rel=1e-6)
        # The attributable row lands on the value cell.
        value_cell = next(c for c in payload["cells"] if c["discovery"] == "value")
        assert value_cell["metrics"]["llm_cost_per_opened"] == pytest.approx(6.0, rel=1e-6)

    def test_analyze_cost_legacy_cohort_path(self, factory_db, monkeypatch, tmp_path):
        """The single-axis cohort path (no `--by`) also emits cost fields when the flag is set."""
        from cents.cli import cli
        from cents.db import LLMUsageRepository
        from cents.models import LLMUsage

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        trepo = ThesisRepository()
        usage_repo = LLMUsageRepository()
        now = datetime.now()

        t = Thesis(
            title="factory:Q",
            symbol="Q",
            tags=[TAG_FACTORY],
            orchestrator_label="llm",
        )
        t.created_at = now - timedelta(days=10)
        trepo.create(t)

        usage_repo.create(LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            context="Q",
            called_at=now - timedelta(days=5),
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "factory", "analyze",
            "--include-cost-per-outcome",
            "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        assert payload["directional"]["llm_cost_per_opened"] == pytest.approx(6.0, rel=1e-6)
        # No neutral-cohort theses → opened=0 → ratio None.
        assert payload["neutral"]["llm_cost_per_opened"] is None
        assert payload["unattributable_cost_usd"] == 0.0


class TestPremiseDirectionPersistence:
    """Layer 2 #1 — engine threads premise_direction onto the Thesis."""

    def test_engine_persists_premise_direction(self, factory_db, monkeypatch):
        _seed_universe(["AAPL"])
        # Stub the classifier to return the new 2-tuple shape.
        monkeypatch.setattr(
            "cents.factory.engine.classify_premise_tags",
            lambda *args, **kwargs: (["ai_capex"], {"ai_capex": "positive"}),
        )
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        engine.run()
        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].premise_tags == ["ai_capex"]
        assert theses[0].premise_direction == {"ai_capex": "positive"}

    def test_engine_accepts_legacy_list_stub(self, factory_db, monkeypatch):
        """A test stub returning a bare list (no direction) must still work."""
        _seed_universe(["AAPL"])
        monkeypatch.setattr(
            "cents.factory.engine.classify_premise_tags",
            lambda *args, **kwargs: ["ai_capex"],
        )
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
        )
        engine.run()
        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].premise_tags == ["ai_capex"]
        assert theses[0].premise_direction == {}


class TestCalibratedPrediction:
    """Layer 2 #3 (research-mode): calibrated_p_correct is recorded, never gates.

    The engine no longer skips opens based on calibrated probability. The
    cohort table will show what happened at every p value — that's the
    research question (does the LLM signal beat regime beta?), not a
    trading-control question (should I open this?).
    """

    def test_no_calibration_model_records_none(self, factory_db):
        _seed_universe(["AAPL"])
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
            calibration_model=None,
        )
        engine.run()
        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].calibrated_p_correct is None
        positions = [p for p in PositionRepository().list() if p.thesis_id == theses[0].id]
        assert positions and all(p.size > 0 for p in positions)

    def test_high_p_records_prediction_and_opens(self, factory_db):
        """p=0.75 → recorded on thesis; positions open normally."""
        _seed_universe(["AAPL"])
        from unittest.mock import MagicMock
        model = MagicMock()
        model.predict.return_value = 0.75
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
            calibration_model=model,
        )
        engine.run()
        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].calibrated_p_correct == 0.75
        assert model.predict.called
        positions = [p for p in PositionRepository().list() if p.thesis_id == theses[0].id]
        assert positions and all(p.size > 0 for p in positions)

    def test_low_p_still_opens(self, factory_db):
        """Research-mode: low p is RECORDED, not used to skip the open.

        Previously (Layer 2 + round-3 fixes) a p below the payoff-adjusted
        break-even skipped the open entirely. In research mode every thesis
        opens so the cohort table sees how low-p signals actually performed.
        """
        _seed_universe(["AAPL"])
        from unittest.mock import MagicMock
        model = MagicMock()
        model.predict.return_value = 0.3
        engine = FactoryEngine(
            config=_config(entry_threshold=5.0),
            orchestrator=_orchestrator({"AAPL": 7.0}),
            price_provider=_price_provider({"AAPL": 100.0}),
            calibration_model=model,
        )
        engine.run()
        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].calibrated_p_correct == 0.3
        positions = [p for p in PositionRepository().list() if p.thesis_id == theses[0].id]
        assert positions and all(p.size > 0 for p in positions)
