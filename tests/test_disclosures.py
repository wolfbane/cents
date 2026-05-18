"""Tests for the shared CLI disclosure footer.

These cover three concerns:
1. The text helpers (`disclosure_text`, `low_n_warning`) have the wording
   and threshold semantics the rest of the codebase relies on.
2. The low-N threshold has been raised to N<30 (was N<5) so the warning
   actually fires on samples that aren't statistically meaningful.
3. Every performance-emitting CLI command (`cohort`, `backtest analyze`,
   `factory analyze`) actually surfaces the footer in its text output.
"""

from __future__ import annotations

import sqlite3

import pytest
from click.testing import CliRunner

from cents.cli import cli
from cents.cli._disclosures import (
    LOW_N_THRESHOLD,
    disclosure_text,
    low_n_warning,
)
from cents.db import BacktestRepository, ThesisRepository
from cents.db.schema import SCHEMA
from cents.models import (
    Backtest,
    BacktestSignal,
    Thesis,
    ThesisCohort,
)


# ---------------------------------------------------------------------------
# Unit tests for the helpers themselves.
# ---------------------------------------------------------------------------


class TestDisclosureText:
    def test_includes_research_tool_framing(self):
        text = disclosure_text()
        assert "research tool" in text

    def test_includes_out_of_scope_framing(self):
        text = disclosure_text()
        assert "out of scope" in text

    def test_default_is_gross_of_costs(self):
        text = disclosure_text()
        assert "gross of costs" in text.lower()

    def test_costs_applied_flips_wording(self):
        text = disclosure_text(costs_applied=True)
        assert "net of modeled costs" in text.lower()
        # And it no longer claims gross.
        assert "gross of costs" not in text.lower()

    def test_mentions_past_performance(self):
        # Regression: the user-facing block must remind readers that past
        # performance doesn't predict future returns.
        text = disclosure_text()
        assert "past performance" in text.lower()

    def test_audience_argument_is_accepted(self):
        # Reserved-for-future-variants argument must not crash today.
        assert disclosure_text(audience="personal") == disclosure_text()


class TestLowNWarning:
    def test_below_threshold_returns_warning(self):
        warning = low_n_warning(5)
        assert warning is not None
        assert "5" in warning
        assert "low sample size" in warning.lower()

    def test_at_threshold_returns_none(self):
        assert low_n_warning(LOW_N_THRESHOLD) is None

    def test_above_threshold_returns_none(self):
        assert low_n_warning(31) is None

    def test_threshold_raised_to_thirty(self):
        # Regression: threshold used to be N<5, which silently passed many
        # statistically meaningless samples. New threshold is N<30.
        assert LOW_N_THRESHOLD == 30
        # And concrete: a sample of 20 (formerly fine) now warns.
        assert low_n_warning(20) is not None

    def test_custom_threshold_override(self):
        assert low_n_warning(10, threshold=5) is None
        assert low_n_warning(4, threshold=5) is not None


# ---------------------------------------------------------------------------
# Smoke tests: every perf CLI surfaces the footer in text output.
# ---------------------------------------------------------------------------


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


def _footer_snippet() -> str:
    """A stable substring of the disclosure footer to assert against."""
    return "research tool"


class TestCohortFooter:
    def test_cohort_text_includes_disclosure(self, runner, mock_db):
        # Seed one thesis so the table renders cleanly.
        ThesisRepository().create(Thesis(title="d", symbol="AAPL"))
        result = runner.invoke(cli, ["cohort"])
        assert result.exit_code == 0, result.output
        assert _footer_snippet() in result.output

    def test_cohort_json_includes_disclosure_field(self, runner, mock_db):
        import json

        ThesisRepository().create(Thesis(title="d", symbol="AAPL"))
        result = runner.invoke(cli, ["cohort", "--output", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "_disclosure" in payload
        assert _footer_snippet() in payload["_disclosure"]
        assert "_low_n" in payload


class TestFactoryAnalyzeFooter:
    def test_factory_analyze_text_includes_disclosure(
        self, runner, mock_db, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        # No factory theses needed — the footer must render even when the
        # body says "(no factory theses in window)".
        result = runner.invoke(cli, ["factory", "analyze", "--by", "discovery"])
        assert result.exit_code == 0, result.output
        assert _footer_snippet() in result.output

    def test_factory_analyze_cohort_default_text_includes_disclosure(
        self, runner, mock_db, monkeypatch, tmp_path
    ):
        # The legacy "--by cohort" path goes through a different printer
        # and used to skip the footer entirely.
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        result = runner.invoke(cli, ["factory", "analyze"])
        assert result.exit_code == 0, result.output
        assert _footer_snippet() in result.output

    def test_factory_analyze_json_includes_disclosure_field(
        self, runner, mock_db, monkeypatch, tmp_path
    ):
        import json

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "f.toml"))
        result = runner.invoke(
            cli,
            ["factory", "analyze", "--by", "discovery", "--output", "json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "_disclosure" in payload
        assert "_low_n" in payload


class TestBacktestAnalyzeFooter:
    def _seed_backtest(self) -> str:
        from datetime import date

        repo = BacktestRepository()
        bt = Backtest(
            symbol="NVDA",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
        )
        repo.create(bt)
        # Two signals so the analyze command has something to work with.
        repo.add_signal(
            BacktestSignal(
                backtest_id=bt.id,
                date=date(2024, 1, 5),
                agent_name="fundamentals",
                conviction_delta=5.0,
                dimension_scores={"valuation": 5.0},
                forward_returns={"5d": 0.02, "20d": 0.05, "60d": 0.10},
            )
        )
        repo.add_signal(
            BacktestSignal(
                backtest_id=bt.id,
                date=date(2024, 1, 12),
                agent_name="fundamentals",
                conviction_delta=-3.0,
                dimension_scores={"valuation": -3.0},
                forward_returns={"5d": -0.01, "20d": -0.03, "60d": -0.05},
            )
        )
        return bt.id

    def test_backtest_analyze_text_includes_disclosure(self, runner, mock_db):
        bt_id = self._seed_backtest()
        result = runner.invoke(cli, ["backtest", "analyze", bt_id])
        assert result.exit_code == 0, result.output
        assert _footer_snippet() in result.output

    def test_backtest_analyze_low_n_warning_fires(self, runner, mock_db):
        # Two signals is well below the 30-signal threshold; warning must
        # appear in text output and the JSON _low_n flag must be True.
        bt_id = self._seed_backtest()
        result = runner.invoke(cli, ["backtest", "analyze", bt_id])
        assert result.exit_code == 0, result.output
        assert "low sample size" in result.output.lower()

    def test_backtest_analyze_json_includes_disclosure_field(
        self, runner, mock_db
    ):
        import json

        bt_id = self._seed_backtest()
        result = runner.invoke(
            cli, ["backtest", "analyze", bt_id, "--output", "json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "_disclosure" in payload
        assert payload["_low_n"] is True


# ---------------------------------------------------------------------------
# Neutral-cohort theses need a hedge symbol; covered above by directional
# only. This guard documents the dependency we rely on in fixtures.
# ---------------------------------------------------------------------------


def test_neutral_thesis_constructor_still_requires_hedge():
    with pytest.raises(ValueError):
        Thesis(title="x", cohort=ThesisCohort.NEUTRAL)
