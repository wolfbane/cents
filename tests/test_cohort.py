"""Tests for policy-neutral cohort primitives."""

import json
import sqlite3
from datetime import date, timedelta

import pytest
from click.testing import CliRunner

from cents.cli import cli
from cents.db import PositionRepository, ThesisRepository
from cents.db.schema import SCHEMA
from cents.models import (
    Position,
    PositionSide,
    PositionStatus,
    Thesis,
    ThesisCohort,
    TimeHorizon,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_db(tmp_path, monkeypatch):
    """Temp DB the CLI will use via CENTS_DB_PATH."""
    db_path = tmp_path / "data" / "cents.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return tmp_path


class TestThesisCohortModel:
    def test_default_cohort_is_directional(self):
        t = Thesis(title="x")
        assert t.cohort == ThesisCohort.DIRECTIONAL
        assert t.hedge_symbol is None
        assert t.paired_thesis_id is None

    def test_neutral_requires_hedge_symbol(self):
        with pytest.raises(ValueError, match="hedge_symbol"):
            Thesis(title="x", cohort=ThesisCohort.NEUTRAL)

    def test_neutral_with_hedge_symbol_ok(self):
        t = Thesis(title="x", symbol="NVDA", cohort=ThesisCohort.NEUTRAL, hedge_symbol="SPY")
        assert t.cohort == ThesisCohort.NEUTRAL
        assert t.hedge_symbol == "SPY"

    def test_cohort_enum_string_values(self):
        assert ThesisCohort.DIRECTIONAL.value == "directional"
        assert ThesisCohort.NEUTRAL.value == "neutral"


class TestThesisRepositoryCohort:
    def test_persists_and_rehydrates_cohort(self, db_conn):
        repo = ThesisRepository(db_conn)
        t = Thesis(
            title="NVDA neutral twin",
            symbol="NVDA",
            cohort=ThesisCohort.NEUTRAL,
            hedge_symbol="spy",
            paired_thesis_id="parent01",
        )
        repo.create(t)

        retrieved = repo.get(t.id)
        assert retrieved is not None
        assert retrieved.cohort == ThesisCohort.NEUTRAL
        assert retrieved.hedge_symbol == "SPY"
        assert retrieved.paired_thesis_id == "parent01"

    def test_directional_thesis_round_trip(self, db_conn):
        repo = ThesisRepository(db_conn)
        t = Thesis(title="directional", symbol="AAPL")
        repo.create(t)

        retrieved = repo.get(t.id)
        assert retrieved.cohort == ThesisCohort.DIRECTIONAL
        assert retrieved.hedge_symbol is None
        assert retrieved.paired_thesis_id is None


class TestThesisCreateCLI:
    def test_create_neutral_requires_hedge_with(self, runner, mock_db):
        result = runner.invoke(
            cli,
            ["thesis", "create", "--title", "x", "--cohort", "neutral"],
        )
        assert result.exit_code != 0
        assert "--hedge-with" in result.output

    def test_create_neutral_with_hedge(self, runner, mock_db):
        result = runner.invoke(
            cli,
            [
                "thesis",
                "create",
                "--title",
                "long NVDA / short SPY",
                "--symbol",
                "NVDA",
                "--cohort",
                "neutral",
                "--hedge-with",
                "spy",
            ],
        )
        assert result.exit_code == 0
        repo = ThesisRepository()
        all_t = repo.list()
        assert len(all_t) == 1
        assert all_t[0].cohort == ThesisCohort.NEUTRAL
        assert all_t[0].hedge_symbol == "SPY"


class TestTwinCommand:
    def _create_parent(self, runner) -> str:
        result = runner.invoke(
            cli,
            [
                "thesis",
                "create",
                "--title",
                "Bull NVDA",
                "--symbol",
                "NVDA",
                "--time-horizon",
                "medium",
                "--target-price",
                "200",
                "--stop-price",
                "100",
            ],
        )
        assert result.exit_code == 0
        return ThesisRepository().list()[0].id

    def test_creates_linked_twin(self, runner, mock_db):
        parent_id = self._create_parent(runner)
        result = runner.invoke(cli, ["thesis", "twin", parent_id, "--hedge-with", "SPY"])
        assert result.exit_code == 0, result.output

        repo = ThesisRepository()
        parent = repo.get(parent_id)
        assert parent.paired_thesis_id is not None
        twin = repo.get(parent.paired_thesis_id)
        assert twin is not None
        assert twin.cohort == ThesisCohort.NEUTRAL
        assert twin.hedge_symbol == "SPY"
        assert twin.paired_thesis_id == parent_id
        assert twin.symbol == "NVDA"
        assert twin.title.startswith("[NEUTRAL] ")
        assert twin.time_horizon == TimeHorizon.MEDIUM
        assert twin.target_price == 200.0
        assert twin.stop_price == 100.0
        assert "long NVDA" in twin.hypothesis and "short SPY" in twin.hypothesis

    def test_twinning_neutral_parent_rejected(self, runner, mock_db):
        runner.invoke(
            cli,
            [
                "thesis",
                "create",
                "--title",
                "n",
                "--symbol",
                "NVDA",
                "--cohort",
                "neutral",
                "--hedge-with",
                "SPY",
            ],
        )
        neutral_id = ThesisRepository().list()[0].id
        result = runner.invoke(cli, ["thesis", "twin", neutral_id, "--hedge-with", "QQQ"])
        assert result.exit_code != 0
        assert "already neutral" in result.output

    def test_re_twinning_rejected(self, runner, mock_db):
        parent_id = self._create_parent(runner)
        first = runner.invoke(cli, ["thesis", "twin", parent_id, "--hedge-with", "SPY"])
        assert first.exit_code == 0
        second = runner.invoke(cli, ["thesis", "twin", parent_id, "--hedge-with", "QQQ"])
        assert second.exit_code != 0
        assert "already has a paired twin" in second.output

    def test_twin_missing_parent(self, runner, mock_db):
        result = runner.invoke(cli, ["thesis", "twin", "nope", "--hedge-with", "SPY"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestPositionOpenNeutral:
    def _setup_neutral(self, runner) -> str:
        runner.invoke(
            cli,
            [
                "thesis",
                "create",
                "--title",
                "twin",
                "--symbol",
                "NVDA",
                "--cohort",
                "neutral",
                "--hedge-with",
                "SPY",
            ],
        )
        return ThesisRepository().list()[0].id

    def test_opens_both_legs_dollar_matched(self, runner, mock_db):
        thesis_id = self._setup_neutral(runner)
        result = runner.invoke(
            cli,
            [
                "position",
                "open",
                "NVDA",
                "--size",
                "10",
                "--price",
                "100",
                "--thesis",
                thesis_id,
                "--hedge-price",
                "400",
            ],
        )
        assert result.exit_code == 0, result.output

        positions = PositionRepository().list()
        assert len(positions) == 2
        by_side = {p.side: p for p in positions}
        long_leg = by_side[PositionSide.LONG]
        short_leg = by_side[PositionSide.SHORT]
        assert long_leg.symbol == "NVDA"
        assert long_leg.size == 10
        assert long_leg.entry_price == 100
        assert long_leg.thesis_id == thesis_id
        assert short_leg.symbol == "SPY"
        assert short_leg.entry_price == 400
        # Dollar-matched: 10 * 100 / 400 = 2.5
        assert short_leg.size == pytest.approx(2.5)
        assert short_leg.thesis_id == thesis_id

    def test_hedge_size_override(self, runner, mock_db):
        thesis_id = self._setup_neutral(runner)
        result = runner.invoke(
            cli,
            [
                "position",
                "open",
                "NVDA",
                "--size",
                "10",
                "--price",
                "100",
                "--thesis",
                thesis_id,
                "--hedge-price",
                "400",
                "--hedge-size",
                "3",
            ],
        )
        assert result.exit_code == 0
        short_leg = next(p for p in PositionRepository().list() if p.side == PositionSide.SHORT)
        assert short_leg.size == 3

    def test_neutral_requires_hedge_price(self, runner, mock_db):
        thesis_id = self._setup_neutral(runner)
        result = runner.invoke(
            cli,
            [
                "position",
                "open",
                "NVDA",
                "--size",
                "10",
                "--price",
                "100",
                "--thesis",
                thesis_id,
            ],
        )
        assert result.exit_code != 0
        assert "--hedge-price" in result.output

    def test_directional_rejects_hedge_price(self, runner, mock_db):
        runner.invoke(
            cli,
            ["thesis", "create", "--title", "d", "--symbol", "AAPL"],
        )
        d_id = ThesisRepository().list()[0].id
        result = runner.invoke(
            cli,
            [
                "position",
                "open",
                "AAPL",
                "--size",
                "10",
                "--price",
                "100",
                "--thesis",
                d_id,
                "--hedge-price",
                "50",
            ],
        )
        assert result.exit_code != 0
        assert "only apply to neutral" in result.output


class TestCohortReport:
    def _seed_mixed_portfolio(self, db_conn):
        """Build: 1 directional (closed profitable) + 1 neutral (spread positive)."""
        trepo = ThesisRepository(db_conn)
        prepo = PositionRepository(db_conn)

        directional = Thesis(title="d", symbol="AAPL")
        trepo.create(directional)
        dp = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100,
            size=10,
            entry_date=date(2024, 1, 1),
            thesis_id=directional.id,
        )
        dp.close(120, exit_date=date(2024, 1, 11))
        prepo.create(dp)

        neutral = Thesis(
            title="n",
            symbol="NVDA",
            cohort=ThesisCohort.NEUTRAL,
            hedge_symbol="SPY",
        )
        trepo.create(neutral)
        long_leg = Position(
            symbol="NVDA",
            side=PositionSide.LONG,
            entry_price=100,
            size=10,
            entry_date=date(2024, 1, 1),
            thesis_id=neutral.id,
        )
        long_leg.close(130, exit_date=date(2024, 1, 21))  # +300
        prepo.create(long_leg)
        short_leg = Position(
            symbol="SPY",
            side=PositionSide.SHORT,
            entry_price=400,
            size=2.5,
            entry_date=date(2024, 1, 1),
            thesis_id=neutral.id,
        )
        short_leg.close(420, exit_date=date(2024, 1, 21))  # -50
        prepo.create(short_leg)

        return directional, neutral

    def test_spread_pnl_aggregation(self, db_conn):
        from cents.cli.cohort import _aggregate_cohort
        from collections import defaultdict

        directional, neutral = self._seed_mixed_portfolio(db_conn)
        trepo = ThesisRepository(db_conn)
        prepo = PositionRepository(db_conn)
        positions_by_thesis = defaultdict(list)
        for p in prepo.list():
            if p.thesis_id:
                positions_by_thesis[p.thesis_id].append(p)

        buckets = _aggregate_cohort(trepo.list(), positions_by_thesis)
        d = buckets["directional"]
        n = buckets["neutral"]

        assert d["thesis_count"] == 1
        assert d["position_count"] == 1
        assert d["realized_pnl"] == pytest.approx(200.0)
        assert d["win_rate"] == pytest.approx(1.0)
        assert d["avg_held_days"] == pytest.approx(10.0)

        assert n["thesis_count"] == 1
        assert n["position_count"] == 2
        # Long leg +300, short leg (400-420)*2.5 = -50; spread +250
        assert n["realized_pnl"] == pytest.approx(250.0)
        assert n["win_rate"] == pytest.approx(1.0)
        assert n["avg_held_days"] == pytest.approx(20.0)


class TestCohortCLI:
    def test_text_output(self, runner, mock_db):
        thesis_repo = ThesisRepository()
        pos_repo = PositionRepository()

        directional = Thesis(title="d", symbol="AAPL")
        thesis_repo.create(directional)
        dp = Position(
            symbol="AAPL", side=PositionSide.LONG, entry_price=100, size=10,
            entry_date=date(2024, 1, 1), thesis_id=directional.id,
        )
        dp.close(120, exit_date=date(2024, 1, 11))
        pos_repo.create(dp)

        neutral = Thesis(
            title="n", symbol="NVDA", cohort=ThesisCohort.NEUTRAL, hedge_symbol="SPY",
        )
        thesis_repo.create(neutral)

        result = runner.invoke(cli, ["cohort"])
        assert result.exit_code == 0, result.output
        assert "directional" in result.output
        assert "neutral" in result.output
        assert "200.00" in result.output

    def test_json_output(self, runner, mock_db):
        thesis_repo = ThesisRepository()
        thesis_repo.create(Thesis(title="d", symbol="AAPL"))
        thesis_repo.create(
            Thesis(title="n", symbol="NVDA", cohort=ThesisCohort.NEUTRAL, hedge_symbol="SPY")
        )

        result = runner.invoke(cli, ["cohort", "--output", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        cohorts = {entry["cohort"]: entry for entry in data["cohorts"]}
        assert cohorts["directional"]["thesis_count"] == 1
        assert cohorts["neutral"]["thesis_count"] == 1
        assert cohorts["directional"]["realized_pnl"] == 0
        assert cohorts["directional"]["win_rate"] is None
        assert "_disclosure" in data
        assert "_low_n" in data
