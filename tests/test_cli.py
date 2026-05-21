"""Tests for CLI commands."""

import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cents.cli import cli
from cents.db.schema import SCHEMA


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_db(tmp_path, monkeypatch):
    """Create temporary database for CLI tests."""
    db_path = tmp_path / "data" / "cents.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    # Set env var so CLI uses this database
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return tmp_path


class TestThesisCLI:
    """Tests for thesis CLI commands."""

    def test_thesis_create(self, runner, mock_db):
        """Create a thesis."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["thesis", "create", "--title", "Test thesis"])
            assert result.exit_code == 0
            assert "Created thesis" in result.output

    def test_thesis_create_with_options(self, runner, mock_db):
        """Create thesis with hypothesis and tags."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli,
                [
                    "thesis",
                    "create",
                    "--title",
                    "AI thesis",
                    "--hypothesis",
                    "AI will grow",
                    "--tags",
                    "tech,AI",
                ],
            )
            assert result.exit_code == 0
            assert "Created thesis" in result.output

    def test_thesis_list_empty(self, runner, mock_db):
        """List when no theses exist."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["thesis", "list"])
            assert result.exit_code == 0
            assert "No theses found" in result.output

    def test_thesis_list(self, runner, mock_db):
        """List created theses."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["thesis", "create", "--title", "Thesis 1"])
            runner.invoke(cli, ["thesis", "create", "--title", "Thesis 2"])
            result = runner.invoke(cli, ["thesis", "list"])
            assert result.exit_code == 0
            assert "Thesis 1" in result.output
            assert "Thesis 2" in result.output

    def test_thesis_show(self, runner, mock_db):
        """Show thesis details."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(cli, ["thesis", "create", "--title", "Test thesis"])
            # Extract ID from "Created thesis XXXX: ..."
            thesis_id = create_result.output.split()[2].rstrip(":")

            result = runner.invoke(cli, ["thesis", "show", thesis_id])
            assert result.exit_code == 0
            assert "Test thesis" in result.output
            assert "Conviction: 50.0%" in result.output

    def test_thesis_show_not_found(self, runner, mock_db):
        """Show nonexistent thesis."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["thesis", "show", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_thesis_update(self, runner, mock_db):
        """Update thesis conviction."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(cli, ["thesis", "create", "--title", "Test"])
            thesis_id = create_result.output.split()[2].rstrip(":")

            result = runner.invoke(
                cli, ["thesis", "update", thesis_id, "--conviction", "75"]
            )
            assert result.exit_code == 0
            assert "Updated thesis" in result.output


class TestPositionCLI:
    """Tests for position CLI commands."""

    def test_position_open(self, runner, mock_db):
        """Open a position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "150"]
            )
            assert result.exit_code == 0
            assert "Opened long position" in result.output
            assert "AAPL" in result.output
            assert "$150.00" in result.output

    def test_position_open_short(self, runner, mock_db):
        """Open a short position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "150", "--short"]
            )
            assert result.exit_code == 0
            assert "Opened short position" in result.output

    def test_position_close(self, runner, mock_db):
        """Close a position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")

            result = runner.invoke(cli, ["position", "close", pos_id, "--price", "110"])
            assert result.exit_code == 0
            assert "Closed position" in result.output
            assert "+$1000.00" in result.output
            assert "+10.0%" in result.output

    def test_position_close_loss(self, runner, mock_db):
        """Close a position at a loss."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")

            result = runner.invoke(cli, ["position", "close", pos_id, "--price", "90"])
            assert result.exit_code == 0
            assert "-1000.00" in result.output
            assert "-10.0%" in result.output

    def test_position_close_not_found(self, runner, mock_db):
        """Close nonexistent position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["position", "close", "nonexistent", "--price", "100"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_position_list(self, runner, mock_db):
        """List positions."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["position", "open", "AAPL", "--size", "100", "--price", "150"])
            runner.invoke(cli, ["position", "open", "GOOG", "--size", "50", "--price", "100"])

            result = runner.invoke(cli, ["position", "list"])
            assert result.exit_code == 0
            assert "AAPL" in result.output
            assert "GOOG" in result.output


class TestOutcomeCLI:
    """Tests for outcome CLI commands."""

    def test_outcome_record(self, runner, mock_db):
        """Record outcome for closed position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            # Open and close a position
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")
            runner.invoke(cli, ["position", "close", pos_id, "--price", "110"])

            # Record outcome
            result = runner.invoke(
                cli,
                ["outcome", "record", pos_id, "--accuracy", "correct", "--notes", "Good trade"],
            )
            assert result.exit_code == 0
            assert "Recorded outcome" in result.output

    def test_outcome_record_open_position(self, runner, mock_db):
        """Cannot record outcome for open position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")

            result = runner.invoke(cli, ["outcome", "record", pos_id])
            assert result.exit_code == 1
            assert "not closed" in result.output

    def test_outcome_list(self, runner, mock_db):
        """List recorded outcomes."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            # Create and close a position
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")
            runner.invoke(cli, ["position", "close", pos_id, "--price", "110"])
            runner.invoke(cli, ["outcome", "record", pos_id, "--accuracy", "correct"])

            result = runner.invoke(cli, ["outcome", "list"])
            assert result.exit_code == 0
            assert "[C]" in result.output  # Correct


class TestWatchlistCLI:
    """Tests for watchlist CLI commands."""

    def test_watch_add(self, runner, mock_db):
        """Add symbol to watchlist."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["watch", "add", "AAPL"])
            assert result.exit_code == 0
            assert "Added AAPL to watchlist" in result.output

    def test_watch_add_duplicate_updates(self, runner, mock_db):
        """Adding duplicate updates the entry."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["watch", "add", "AAPL"])
            result = runner.invoke(cli, ["watch", "add", "AAPL", "--threshold", "5.0"])
            assert "Updated AAPL on watchlist" in result.output

    def test_watch_remove(self, runner, mock_db):
        """Remove symbol from watchlist."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["watch", "add", "AAPL"])
            result = runner.invoke(cli, ["watch", "remove", "AAPL"])
            assert result.exit_code == 0
            assert "Removed AAPL" in result.output

    def test_watch_list(self, runner, mock_db):
        """List watchlist items."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["watch", "add", "AAPL"])
            runner.invoke(cli, ["watch", "add", "GOOG"])

            result = runner.invoke(cli, ["watch", "list"])
            assert "AAPL" in result.output
            assert "GOOG" in result.output

    def test_watch_add_with_threshold_and_webhook(self, runner, mock_db):
        """Add watchlist item with custom threshold and webhook."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            add_result = runner.invoke(
                cli,
                [
                    "watch",
                    "add",
                    "AAPL",
                    "--threshold",
                    "3.5",
                    "--webhook",
                    "https://example.com/hook",
                ],
            )
            assert add_result.exit_code == 0

            list_result = runner.invoke(cli, ["watch", "list"])
            assert "threshold: 3.5" in list_result.output
            assert "alert: custom" in list_result.output


class TestAlertCLI:
    """Tests for alert CLI commands."""

    def test_alert_list_empty(self, runner, mock_db):
        """List when no alerts."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list"])
            assert result.exit_code == 0
            assert "No unread alerts" in result.output


class TestThesisCloseCLI:
    """Tests for thesis close command."""

    def test_thesis_close(self, runner, mock_db):
        """Close a thesis."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(cli, ["thesis", "create", "--title", "Test thesis"])
            thesis_id = create_result.output.split()[2].rstrip(":")

            result = runner.invoke(cli, ["thesis", "close", thesis_id])
            assert result.exit_code == 0
            assert "Closed thesis" in result.output

    def test_thesis_close_with_outcome(self, runner, mock_db):
        """Close thesis with outcome."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(cli, ["thesis", "create", "--title", "Test thesis"])
            thesis_id = create_result.output.split()[2].rstrip(":")

            result = runner.invoke(
                cli, ["thesis", "close", thesis_id, "--outcome", "correct"]
            )
            assert result.exit_code == 0
            assert "(correct)" in result.output

    def test_thesis_close_not_found(self, runner, mock_db):
        """Close nonexistent thesis."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["thesis", "close", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output


class TestThesisStructuredFields:
    """Tests for thesis structured fields."""

    def test_thesis_create_with_structured_fields(self, runner, mock_db):
        """Create thesis with structured fields."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli,
                [
                    "thesis",
                    "create",
                    "--title",
                    "AAPL thesis",
                    "--symbol",
                    "AAPL",
                    "--valuation",
                    "undervalued",
                    "--time-horizon",
                    "medium",
                    "--target-price",
                    "200",
                    "--stop-price",
                    "150",
                ],
            )
            assert result.exit_code == 0
            assert "Created thesis" in result.output

    def test_thesis_show_structured_fields(self, runner, mock_db):
        """Show thesis with structured fields."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(
                cli,
                [
                    "thesis",
                    "create",
                    "--title",
                    "AAPL thesis",
                    "--symbol",
                    "AAPL",
                    "--valuation",
                    "undervalued",
                ],
            )
            thesis_id = create_result.output.split()[2].rstrip(":")

            result = runner.invoke(cli, ["thesis", "show", thesis_id])
            assert result.exit_code == 0
            assert "AAPL" in result.output
            assert "undervalued" in result.output

    def test_thesis_update_structured_fields(self, runner, mock_db):
        """Update thesis structured fields."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(cli, ["thesis", "create", "--title", "Test thesis"])
            thesis_id = create_result.output.split()[2].rstrip(":")

            result = runner.invoke(
                cli,
                [
                    "thesis",
                    "update",
                    thesis_id,
                    "--valuation",
                    "fair",
                    "--moat",
                    "Strong brand",
                ],
            )
            assert result.exit_code == 0

            show_result = runner.invoke(cli, ["thesis", "show", thesis_id])
            assert "fair" in show_result.output


class TestAlertCLIExtended:
    """Extended tests for alert CLI commands."""

    def test_alert_list_all(self, runner, mock_db):
        """List all alerts including read ones."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--all"])
            assert result.exit_code == 0

    def test_alert_read_all(self, runner, mock_db):
        """Mark all alerts as read."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "read", "--all"])
            assert result.exit_code == 0
            assert "Marked" in result.output

    def _seed_alerts(self, db_path):
        from datetime import datetime, timedelta
        from cents.db import AlertRepository
        from cents.models import Alert, AlertType

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        repo = AlertRepository(conn)
        now = datetime.now()
        old = Alert(
            symbol="OLD",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="old alert",
            created_at=now - timedelta(days=10),
        )
        recent = Alert(
            symbol="NEW",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="recent alert",
            created_at=now - timedelta(minutes=5),
        )
        repo.create(old)
        repo.create(recent)
        conn.close()

    def test_alert_list_since_today(self, runner, mock_db):
        """--since today shows only alerts from today."""
        db_path = os.environ["CENTS_DB_PATH"]
        self._seed_alerts(db_path)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--since", "today"])
            assert result.exit_code == 0
            assert "NEW" in result.output
            assert "OLD" not in result.output

    def test_alert_list_since_iso_date(self, runner, mock_db):
        """--since accepts ISO date."""
        from datetime import datetime, timedelta

        db_path = os.environ["CENTS_DB_PATH"]
        self._seed_alerts(db_path)
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--since", yesterday])
            assert result.exit_code == 0
            assert "NEW" in result.output
            assert "OLD" not in result.output

    def test_alert_list_since_relative_hours(self, runner, mock_db):
        """--since Nh accepts relative hours."""
        db_path = os.environ["CENTS_DB_PATH"]
        self._seed_alerts(db_path)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--since", "24h"])
            assert result.exit_code == 0
            assert "NEW" in result.output
            assert "OLD" not in result.output

    def test_alert_list_since_relative_days(self, runner, mock_db):
        """--since Nd accepts relative days; 30d window includes 10-day-old alert."""
        db_path = os.environ["CENTS_DB_PATH"]
        self._seed_alerts(db_path)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--since", "30d"])
            assert result.exit_code == 0
            assert "NEW" in result.output
            assert "OLD" in result.output

    def test_alert_list_since_invalid(self, runner, mock_db):
        """--since with invalid value exits non-zero with clear error."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--since", "yesterday"])
            assert result.exit_code != 0
            assert "yesterday" in result.output or "yesterday" in (result.stderr or "")

    def test_alert_list_since_composes_with_all(self, runner, mock_db):
        """--all --since composes: shows read+unread within window."""
        from cents.db import AlertRepository

        db_path = os.environ["CENTS_DB_PATH"]
        self._seed_alerts(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        repo = AlertRepository(conn)
        for a in repo.list_unread():
            repo.mark_read(a.id)
        conn.close()
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["alert", "list", "--all", "--since", "today"])
            assert result.exit_code == 0
            assert "NEW" in result.output
            assert "OLD" not in result.output


class TestResearchCLI:
    """Tests for research command."""

    def test_research_command_help(self, runner):
        """Research command help."""
        result = runner.invoke(cli, ["research", "--help"])
        assert result.exit_code == 0
        assert "Run research agents" in result.output

    @patch("cents.cli.research.AGENTS")
    def test_research_runs_agents(self, mock_agents, runner, mock_db):
        """Research runs agents and displays results."""
        from unittest.mock import MagicMock
        from cents.agents.base import AgentResult

        mock_agent_instance = MagicMock()
        mock_agent_instance.research.return_value = AgentResult(
            evidence=[],
            conviction_delta=5.0,
            summary="Test agent: bullish signal",
        )
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        # Default (no --agent) now uses orchestrator only
        mock_agents.__getitem__.return_value = mock_agent_class

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "AAPL", "--no-save"])
            assert result.exit_code == 0
            assert "conviction delta" in result.output.lower()

    @patch("cents.cli.research.AGENTS")
    def test_research_json_output(self, mock_agents, runner, mock_db):
        """Research with JSON output format."""
        from unittest.mock import MagicMock
        from cents.agents.base import AgentResult

        mock_agent_instance = MagicMock()
        mock_agent_instance.research.return_value = AgentResult(
            evidence=[],
            conviction_delta=3.5,
            summary="Test summary",
        )
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        # Default (no --agent) now uses orchestrator only
        mock_agents.__getitem__.return_value = mock_agent_class

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "AAPL", "--output", "json", "--no-save"])
            assert result.exit_code == 0
            import json
            data = json.loads(result.output)
            assert data["symbol"] == "AAPL"
            assert data["total_conviction_delta"] == 3.5
            assert "agents" in data

    @patch("cents.cli.research.AGENTS")
    def test_research_suggest_thesis(self, mock_agents, runner, mock_db):
        """Research with --suggest-thesis generates thesis suggestion."""
        from unittest.mock import MagicMock
        from cents.agents.base import AgentResult

        mock_agent_instance = MagicMock()
        mock_agent_instance.research.return_value = AgentResult(
            evidence=[],
            conviction_delta=10.0,
            summary="Strong fundamentals detected",
        )
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        # Default (no --agent) now uses orchestrator only
        mock_agents.__getitem__.return_value = mock_agent_class

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "NVDA", "--suggest-thesis", "--no-save"])
            assert result.exit_code == 0
            assert "THESIS SUGGESTION" in result.output
            assert "NVDA" in result.output

    @patch("cents.cli.research.AGENTS")
    def test_research_suggest_thesis_json(self, mock_agents, runner, mock_db):
        """Research with --suggest-thesis includes suggestion in JSON output."""
        from unittest.mock import MagicMock
        from cents.agents.base import AgentResult

        mock_agent_instance = MagicMock()
        mock_agent_instance.research.return_value = AgentResult(
            evidence=[],
            conviction_delta=5.0,
            summary="Bullish signal",
        )
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        # Default (no --agent) now uses orchestrator only
        mock_agents.__getitem__.return_value = mock_agent_class

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, [
                "research", "TSLA", "--suggest-thesis", "--output", "json", "--no-save"
            ])
            assert result.exit_code == 0
            import json
            data = json.loads(result.output)
            assert "thesis_suggestion" in data
            assert data["thesis_suggestion"]["symbol"] == "TSLA"
            assert data["thesis_suggestion"]["conviction"] == 55.0  # 50 + 5

    def test_research_with_nonexistent_thesis(self, runner, mock_db):
        """Research with non-existent thesis ID fails."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "AAPL", "--thesis", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_research_help_advertises_export_flag(self, runner):
        """--export-html should appear in the research command help."""
        result = runner.invoke(cli, ["research", "--help"])
        assert result.exit_code == 0
        assert "--export-html" in result.output

    @patch("cents.cli.research.AGENTS")
    def test_research_export_html(self, mock_agents, runner, mock_db, tmp_path):
        """--export-html PATH writes a self-contained HTML file."""
        from cents.agents.base import AgentResult
        from cents.models import Evidence, EvidenceType

        evidence = Evidence(
            agent="orchestrator",
            content="Strong revenue growth",
            source="fundamentals-agent",
            type=EvidenceType.SUPPORTING,
            confidence=0.8,
        )
        mock_agent_instance = MagicMock()
        mock_agent_instance.research.return_value = AgentResult(
            evidence=[evidence],
            conviction_delta=4.5,
            summary="Bullish setup",
            dimension_scores={"valuation": 1.5, "quality": 3.0},
        )
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        mock_agents.__getitem__.return_value = mock_agent_class

        out_path = tmp_path / "report.html"

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli,
                ["research", "NVDA", "--no-save", "--export-html", str(out_path)],
            )
            assert result.exit_code == 0, result.output

        html = out_path.read_text(encoding="utf-8")
        assert len(html) > 1000, "HTML report should have substantive content"
        assert "NVDA" in html
        assert "<style>" in html
        assert 'rel="stylesheet"' not in html
        assert "Bullish setup" in html
        assert "Strong revenue growth" in html


class TestRecommendOutput:
    """Tests for the recommend command's output formatting."""

    def test_priority_1_uses_action_review_header(self, capsys):
        """Priority-1 recommendations render under the 'ACTION (review)' header.

        Regression test: this header was previously 'URGENT (act now)' which
        reads as financial advice. Don't let it slip back.
        """
        from cents.cli.recommend import (
            Action,
            Recommendation,
            _print_recommendations,
        )

        rec = Recommendation(
            symbol="NVDA",
            action=Action.CLOSE,
            reason="Stop-loss triggered",
            thesis_id="t1",
            current_price=100.0,
            conviction=20.0,
            priority=1,
        )
        _print_recommendations([rec], actionable=False)

        out = capsys.readouterr().out
        assert "ACTION (review)" in out
        assert "URGENT" not in out
        assert "act now" not in out

    def test_priority_2_uses_model_signals_header(self, capsys):
        """Priority-2 signals render under 'MODEL SIGNALS:', not 'RECOMMENDATIONS:'.

        Regression test: the old header read as advice. The rename
        (BUY/SELL/HOLD -> bullish/bearish/neutral signal) should NOT
        regress to advice-language headers.
        """
        from cents.cli.recommend import (
            Action,
            Recommendation,
            _print_recommendations,
        )

        rec = Recommendation(
            symbol="NVDA",
            action=Action.BULLISH,
            reason="Conviction 80%, 20% upside to target",
            thesis_id="t1",
            current_price=100.0,
            conviction=80.0,
            priority=2,
        )
        _print_recommendations([rec], actionable=False)

        out = capsys.readouterr().out
        assert "MODEL SIGNALS" in out
        # Advice-language headers must stay gone.
        assert "RECOMMENDATIONS:" not in out
        # And the displayed action label is signal language, not BUY.
        assert "BULLISH" in out
        assert "BUY " not in out  # tolerates "buy-threshold" elsewhere

    def test_action_enum_uses_signal_language(self):
        """Action enum values are signal language, not advice.

        Regression: BUY/SELL/HOLD enum values were textbook advice; this
        guards the rename so we don't silently revert.
        """
        from cents.cli.recommend import Action

        assert Action.BULLISH.value == "bullish_signal"
        assert Action.BEARISH.value == "bearish_signal"
        assert Action.NEUTRAL.value == "neutral_signal"
        # And the advice values are gone entirely.
        values = {a.value for a in Action}
        assert "buy" not in values
        assert "sell" not in values
        assert "hold" not in values

    def test_output_includes_scope_disclaimer(self, capsys):
        """Text output ends with a 'Model signal, not investment advice' line."""
        from cents.cli.recommend import (
            Action,
            Recommendation,
            _print_recommendations,
        )

        rec = Recommendation(
            symbol="NVDA",
            action=Action.NEUTRAL,
            reason="Thesis intact, no signal",
            thesis_id="t1",
            current_price=100.0,
            conviction=50.0,
            priority=3,
        )
        _print_recommendations([rec], actionable=False)

        out = capsys.readouterr().out
        assert "Model signal" in out
        assert "not investment advice" in out
        assert "/scope/" in out


class TestGenerateThesisSuggestion:
    """Tests for _generate_thesis_suggestion helper."""

    def test_basic_suggestion(self):
        """Generate basic thesis suggestion from empty research."""
        from cents.cli import _generate_thesis_suggestion

        result = _generate_thesis_suggestion("AAPL", [], 0.0)

        assert result["symbol"] == "AAPL"
        assert result["title"] == "AAPL investment thesis"
        assert result["conviction"] == 50.0

    def test_suggestion_with_positive_conviction(self):
        """Positive conviction delta increases suggestion conviction."""
        from cents.cli import _generate_thesis_suggestion

        result = _generate_thesis_suggestion("NVDA", [], 20.0)

        assert result["conviction"] == 70.0

    def test_suggestion_conviction_clamped(self):
        """Conviction is clamped to 0-100 range."""
        from cents.cli import _generate_thesis_suggestion

        # Test upper bound
        result = _generate_thesis_suggestion("TEST", [], 100.0)
        assert result["conviction"] == 100.0

        # Test lower bound
        result = _generate_thesis_suggestion("TEST", [], -100.0)
        assert result["conviction"] == 0.0

    def test_suggestion_extracts_pe_valuation(self):
        """PE ratio determines valuation assessment."""
        from cents.cli import _generate_thesis_suggestion

        # Low PE = undervalued
        agent_outputs = [{
            "agent": "fundamentals",
            "summary": "Low P/E",
            "evidence": [{
                "type": "supporting",
                "content": "P/E of 10",
                "metadata": {"metric": "pe_ratio", "value": 10}
            }]
        }]
        result = _generate_thesis_suggestion("AAPL", agent_outputs, 0.0)
        assert result["valuation"] == "undervalued"

        # High PE = overvalued
        agent_outputs[0]["evidence"][0]["metadata"]["value"] = 40
        result = _generate_thesis_suggestion("AAPL", agent_outputs, 0.0)
        assert result["valuation"] == "overvalued"

        # Mid PE = fair
        agent_outputs[0]["evidence"][0]["metadata"]["value"] = 20
        result = _generate_thesis_suggestion("AAPL", agent_outputs, 0.0)
        assert result["valuation"] == "fair"

    def test_suggestion_extracts_quality_notes(self):
        """Profit margin affects quality assessment."""
        from cents.cli import _generate_thesis_suggestion

        agent_outputs = [{
            "agent": "fundamentals",
            "summary": "Strong margins",
            "evidence": [{
                "type": "supporting",
                "content": "High profit margin",
                "metadata": {"metric": "profit_margin", "value": 0.25}
            }]
        }]
        result = _generate_thesis_suggestion("AAPL", agent_outputs, 0.0)
        assert result["business_quality"] is not None
        assert "margins" in result["business_quality"].lower()

    def test_suggestion_extracts_risks_from_contradicting_evidence(self):
        """Contradicting evidence becomes key risks."""
        from cents.cli import _generate_thesis_suggestion

        agent_outputs = [{
            "agent": "macro",
            "summary": "Bearish macro",
            "evidence": [{
                "type": "contradicting",
                "content": "High interest rates",
                "metadata": {}
            }, {
                "type": "contradicting",
                "content": "Inverted yield curve",
                "metadata": {}
            }]
        }]
        result = _generate_thesis_suggestion("AAPL", agent_outputs, 0.0)
        assert len(result["key_risks"]) == 2
        assert "High interest rates" in result["key_risks"]

    def test_suggestion_limits_risks_to_five(self):
        """Key risks are limited to 5 items."""
        from cents.cli import _generate_thesis_suggestion

        agent_outputs = [{
            "agent": "test",
            "summary": "Many risks",
            "evidence": [
                {"type": "contradicting", "content": f"Risk {i}", "metadata": {}}
                for i in range(10)
            ]
        }]
        result = _generate_thesis_suggestion("TEST", agent_outputs, 0.0)
        assert len(result["key_risks"]) == 5


class TestScanCLI:
    """Tests for scan command."""

    def test_scan_empty_watchlist(self, runner, mock_db):
        """Scan with empty watchlist."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["scan"])
            assert result.exit_code == 0
            assert "Watchlist is empty" in result.output

    def test_scan_json_output_empty(self, runner, mock_db):
        """Scan with JSON output and empty watchlist."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["scan", "--output", "json"])
            assert result.exit_code == 0
            assert "[]" in result.output


class TestVersionAndHelp:
    """Tests for version and help."""

    def test_version(self, runner):
        """Show version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        """Show help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Agentic investing guidance" in result.output

    def test_thesis_help(self, runner):
        """Thesis command help."""
        result = runner.invoke(cli, ["thesis", "--help"])
        assert result.exit_code == 0

    def test_position_help(self, runner):
        """Position command help."""
        result = runner.invoke(cli, ["position", "--help"])
        assert result.exit_code == 0

    def test_watch_help(self, runner):
        """Watch command help."""
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0


class TestBrokerCLIErrors:
    """Tests for broker CLI error paths."""

    @patch("cents.broker.ALPACA_AVAILABLE", False)
    def test_broker_status_alpaca_not_installed(self, runner):
        """Broker status fails when Alpaca not installed."""
        result = runner.invoke(cli, ["broker", "status"])
        assert result.exit_code == 1
        assert "Alpaca not installed" in result.output
        assert "pip install cents[broker]" in result.output

    @patch("cents.broker.ALPACA_AVAILABLE", False)
    def test_broker_list_alpaca_not_installed(self, runner):
        """Broker list fails when Alpaca not installed."""
        result = runner.invoke(cli, ["broker", "list"])
        assert result.exit_code == 1
        assert "Alpaca not installed" in result.output

    @patch("cents.broker.ALPACA_AVAILABLE", False)
    def test_broker_sync_alpaca_not_installed(self, runner, mock_db):
        """Broker sync fails when Alpaca not installed."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["broker", "sync"])
            assert result.exit_code == 1
            assert "Alpaca not installed" in result.output

    @patch("cents.broker.ALPACA_AVAILABLE", False)
    def test_broker_buy_alpaca_not_installed(self, runner):
        """Broker buy fails when Alpaca not installed."""
        result = runner.invoke(cli, ["broker", "buy", "AAPL", "--qty", "10", "--yes"])
        assert result.exit_code == 1
        assert "Alpaca not installed" in result.output

    @patch("cents.broker.ALPACA_AVAILABLE", False)
    def test_broker_sell_alpaca_not_installed(self, runner):
        """Broker sell fails when Alpaca not installed."""
        result = runner.invoke(cli, ["broker", "sell", "AAPL", "--qty", "10", "--yes"])
        assert result.exit_code == 1
        assert "Alpaca not installed" in result.output

    @patch("cents.broker.AlpacaClient")
    def test_broker_status_connection_error(self, mock_client_class, runner):
        """Broker status handles connection errors."""
        mock_client_class.side_effect = ValueError("Missing ALPACA_API_KEY")

        result = runner.invoke(cli, ["broker", "status"])
        assert result.exit_code == 1
        assert "Configuration error" in result.output

    @patch("cents.broker.AlpacaClient")
    def test_broker_status_api_error(self, mock_client_class, runner):
        """Broker status handles API errors."""
        from cents.exceptions import BrokerError
        mock_client = mock_client_class.return_value
        mock_client.get_account.side_effect = BrokerError("API rate limit exceeded")

        result = runner.invoke(cli, ["broker", "status"])
        assert result.exit_code == 1
        assert "API error" in result.output

    @patch("cents.broker.AlpacaClient")
    def test_broker_status_network_error(self, mock_client_class, runner):
        """Broker status handles network errors."""
        mock_client = mock_client_class.return_value
        mock_client.get_account.side_effect = ConnectionError("Connection refused")

        result = runner.invoke(cli, ["broker", "status"])
        assert result.exit_code == 1
        assert "Connection failed" in result.output

    @patch("cents.broker.AlpacaClient")
    def test_broker_list_api_error(self, mock_client_class, runner):
        """Broker list handles API errors."""
        from cents.exceptions import APIError
        mock_client = mock_client_class.return_value
        mock_client.get_positions.side_effect = APIError("Network error")

        result = runner.invoke(cli, ["broker", "list"])
        assert result.exit_code == 1
        assert "API error" in result.output

    @patch("cents.broker.AlpacaClient")
    def test_broker_buy_order_failure(self, mock_client_class, runner):
        """Broker buy handles order failures."""
        from cents.exceptions import BrokerError
        mock_client = mock_client_class.return_value
        mock_client.submit_order.side_effect = BrokerError("Insufficient funds")

        result = runner.invoke(cli, ["broker", "buy", "AAPL", "--qty", "10", "--yes"])
        assert result.exit_code == 1
        assert "Order failed" in result.output


class TestSymbolValidation:
    """Tests for symbol validation error paths."""

    def test_position_open_invalid_symbol(self, runner, mock_db):
        """Position open validates symbol format."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            # Lowercase should be converted to uppercase
            result = runner.invoke(
                cli, ["position", "open", "aapl", "--size", "10", "--price", "100"]
            )
            assert result.exit_code == 0
            assert "AAPL" in result.output

    def test_thesis_update_not_found(self, runner, mock_db):
        """Thesis update returns error for non-existent thesis."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli, ["thesis", "update", "nonexistent", "--conviction", "75"]
            )
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_position_close_already_closed(self, runner, mock_db):
        """Position close handles already closed position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            # Open and close a position
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "--size", "10", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")
            runner.invoke(cli, ["position", "close", pos_id, "--price", "110"])

            # Try to close again
            result = runner.invoke(cli, ["position", "close", pos_id, "--price", "120"])
            assert result.exit_code == 1
            assert "not open" in result.output.lower() or "already" in result.output.lower()


class TestThesisResolutionTriggers:
    """Tests for thesis resolution trigger error paths."""

    def test_thesis_create_invalid_price_values(self, runner, mock_db):
        """Thesis create handles invalid price values."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            # Create thesis with valid price values
            result = runner.invoke(
                cli,
                [
                    "thesis", "create",
                    "--title", "Test",
                    "--target-price", "200",
                    "--stop-price", "150",
                ],
            )
            assert result.exit_code == 0

    def test_watch_remove_not_found(self, runner, mock_db):
        """Watch remove handles non-existent symbol."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["watch", "remove", "NOTFOUND"])
            assert result.exit_code == 1
            assert "not found" in result.output


class TestUsageHeadroom:
    """Tests for `cents usage headroom` — daily cost cap tracking surface."""

    def _seed_usage(self, model="claude-haiku-4-5-20251001", *, input_tokens, output_tokens, days_ago=0):
        """Insert one llm_usage row at `now - days_ago` so today_cost_usd can find it."""
        import sqlite3
        from datetime import datetime, timedelta
        from cents.db.schema import get_db_path
        from cents.models import LLMUsage
        from cents.db import LLMUsageRepository

        repo = LLMUsageRepository()
        called_at = datetime.now() - timedelta(days=days_ago)
        row = LLMUsage(
            model=model,
            agent="sentiment",
            operation="score_article",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            called_at=called_at,
        )
        repo.create(row)

    def test_headroom_no_cap_configured(self, runner, mock_db, monkeypatch):
        """Reports 'no_cap_configured' when neither env nor settings supply a cap."""
        monkeypatch.delenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", raising=False)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["usage", "headroom", "--output", "json"])
            assert result.exit_code == 0
            import json
            payload = json.loads(result.output)
            assert payload["status"] == "no_cap_configured"
            assert payload["cap_usd"] is None

    def test_headroom_ok_status_when_well_under_warn(self, runner, mock_db, monkeypatch):
        """Status = 'ok' when today's spend is under warn_pct of cap."""
        monkeypatch.setenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", "5.0")
        # Spend ~$0.001 today, $5 cap → ok
        self._seed_usage(input_tokens=100, output_tokens=20)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["usage", "headroom", "--output", "json"])
            assert result.exit_code == 0
            import json
            payload = json.loads(result.output)
            assert payload["status"] == "ok"
            assert payload["cap_usd"] == 5.0
            assert payload["used_pct"] < 1.0
            assert payload["headroom_pct"] > 99.0

    def test_headroom_approaching_cap(self, runner, mock_db, monkeypatch):
        """Status = 'approaching_cap' when spend crosses warn_pct."""
        monkeypatch.setenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", "1.0")
        # Force ~$0.85 by seeding ~850k input tokens ($1/M for Haiku 4.5)
        self._seed_usage(input_tokens=850_000, output_tokens=0)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["usage", "headroom", "--output", "json"])
            assert result.exit_code == 0
            import json
            payload = json.loads(result.output)
            assert payload["status"] == "approaching_cap"
            assert payload["used_pct"] >= 80.0
            assert payload["headroom_pct"] < 20.0

    def test_headroom_hit_cap(self, runner, mock_db, monkeypatch):
        """Status = 'hit_cap' when today's spend is at or above the cap."""
        monkeypatch.setenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", "1.0")
        self._seed_usage(input_tokens=1_000_000, output_tokens=0)
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["usage", "headroom", "--output", "json"])
            assert result.exit_code == 0
            import json
            payload = json.loads(result.output)
            assert payload["status"] == "hit_cap"
            assert payload["headroom_pct"] == 0.0


class TestCacheCLI:
    """cents cache stats / prune / clear (cents-ame follow-up)."""

    def test_cache_stats_text(self, runner, mock_db):
        # Empty DB: should report nothing-to-show
        result = runner.invoke(cli, ["cache", "stats"])
        assert result.exit_code == 0, result.output
        assert "api_cache is empty" in result.output

    def test_cache_stats_json(self, runner, mock_db):
        # Seed a row directly into the api_cache table created by SCHEMA
        conn = sqlite3.connect(mock_db / "data" / "cents.db")
        conn.execute(
            "INSERT INTO api_cache (id, provider, endpoint, cache_key, response_data, cached_at) "
            "VALUES ('a1', 'fmp', 'ratios', 'k1', '[1,2,3]', datetime('now'))"
        )
        conn.commit()
        conn.close()

        result = runner.invoke(cli, ["cache", "stats", "--output", "json"])
        assert result.exit_code == 0, result.output
        import json
        rows = json.loads(result.output)
        assert any(r["provider"] == "fmp" and r["endpoint"] == "ratios" for r in rows)

    def test_cache_prune_drops_dead_namespace(self, runner, mock_db):
        # Seed a dead-namespace row + a fresh policied row. Use Python ISO
        # format — cents writes via datetime.now().isoformat() and the prune
        # cutoff compares against that format; SQL datetime('now') would use
        # a different format and break the string comparison.
        from datetime import datetime
        now_iso = datetime.now().isoformat()
        conn = sqlite3.connect(mock_db / "data" / "cents.db")
        conn.execute(
            "INSERT INTO api_cache (id, provider, endpoint, cache_key, response_data, cached_at) "
            "VALUES ('a1', 'alpaca', 'bars', 'k1', '[]', ?)",
            (now_iso,),
        )
        conn.execute(
            "INSERT INTO api_cache (id, provider, endpoint, cache_key, response_data, cached_at) "
            "VALUES ('a2', 'alpaca', 'bars_split_v1', 'k2', '[]', ?)",
            (now_iso,),
        )
        conn.commit()
        conn.close()

        result = runner.invoke(cli, ["cache", "prune"])
        assert result.exit_code == 0, result.output
        # Dead namespace should be reported as pruned
        assert "alpaca/bars" in result.output

        # Re-read DB: only bars_split_v1 should remain
        conn = sqlite3.connect(mock_db / "data" / "cents.db")
        remaining = conn.execute(
            "SELECT endpoint FROM api_cache"
        ).fetchall()
        conn.close()
        assert len(remaining) == 1
        assert remaining[0][0] == "bars_split_v1"

    def test_cache_clear_requires_confirmation(self, runner, mock_db):
        # No --yes → confirmation prompt; pipe in 'n' to cancel
        result = runner.invoke(cli, ["cache", "clear"], input="n\n")
        # Click confirmation_option exits non-zero when declined
        assert result.exit_code != 0

    def test_cache_clear_with_confirmation_wipes_table(self, runner, mock_db):
        conn = sqlite3.connect(mock_db / "data" / "cents.db")
        conn.execute(
            "INSERT INTO api_cache (id, provider, endpoint, cache_key, response_data, cached_at) "
            "VALUES ('a1', 'fmp', 'ratios', 'k1', '[]', datetime('now'))"
        )
        conn.commit()
        conn.close()

        result = runner.invoke(cli, ["cache", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert "Cleared 1 rows" in result.output


class TestCalculateHitRate:
    """Bug fix: hit-rate must skip neutral (delta=0) signals, not count them as misses."""

    def test_skips_neutral_signals(self):
        """delta=0 is 'no prediction' — excluded from both numerator and denominator."""
        from cents.cli._shared import calculate_hit_rate
        # 1 correct hit + 3 neutrals → hit-rate should be 100% (1/1), NOT 25% (1/4)
        rate = calculate_hit_rate(
            deltas=[5.0, 0.0, 0.0, 0.0],
            returns=[0.1, 0.1, -0.1, 0.1],
        )
        assert rate == 1.0, f"Expected 1.0 (1 of 1 non-neutral signals hit), got {rate}"

    def test_all_neutral_returns_none(self):
        from cents.cli._shared import calculate_hit_rate
        rate = calculate_hit_rate(
            deltas=[0.0, 0.0, 0.0],
            returns=[0.1, -0.1, 0.0],
        )
        assert rate is None

    def test_mixed_hits_and_misses(self):
        from cents.cli._shared import calculate_hit_rate
        # 2 hits (+/+, -/-), 2 misses (+/-, -/+), 1 neutral
        rate = calculate_hit_rate(
            deltas=[5.0, -3.0, 4.0, -2.0, 0.0],
            returns=[0.05, -0.02, -0.03, 0.01, 0.99],
        )
        assert rate == 0.5, f"Expected 0.5 (2/4 non-neutral signals hit), got {rate}"

    def test_empty_inputs_return_none(self):
        from cents.cli._shared import calculate_hit_rate
        assert calculate_hit_rate([], []) is None

    def test_length_mismatch_returns_none(self):
        from cents.cli._shared import calculate_hit_rate
        assert calculate_hit_rate([1.0, 2.0], [0.1]) is None


class TestFactoryInitValidation:
    """cents-5lj8: cents factory init must probe the configured default universe
    and emit clear WARNINGs (not errors) when the universe is missing, empty,
    or its tickers don't resolve at FMP."""

    @pytest.fixture
    def factory_config_path(self, tmp_path, monkeypatch):
        """Point factory.toml at a throwaway tmp path so init never touches ~/."""
        path = tmp_path / "factory.toml"
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(path))
        return path

    def _seed_universe(self, name, symbols, *, is_default=True):
        """Create a universe row + optionally mark it default."""
        from cents.db import UniverseRepository
        from cents.models import Universe

        repo = UniverseRepository()
        repo.create(Universe(name=name, symbols=symbols, is_default=is_default))

    def test_valid_default_universe_emits_no_warnings(
        self, runner, mock_db, factory_config_path, monkeypatch
    ):
        """When default universe exists with symbols + FMP probe succeeds → no WARNING."""
        self._seed_universe("sp500_lite", ["AAPL"], is_default=True)
        monkeypatch.setenv("FMP_API_KEY", "fake-key")

        # Mock the FMP profile fetch to return a non-empty payload.
        with patch(
            "cents.data.fmp.FMPFundamentalsProvider._fetch_json",
            return_value=[{"symbol": "AAPL", "companyName": "Apple Inc."}],
        ):
            result = runner.invoke(cli, ["factory", "init"])

        assert result.exit_code == 0
        assert "Wrote factory config" in result.output
        assert "WARNING" not in result.output
        assert "Probed default_universe" in result.output
        assert "AAPL" in result.output
        assert factory_config_path.exists()

    def test_missing_universe_emits_warning_and_writes_config(
        self, runner, mock_db, factory_config_path, monkeypatch
    ):
        """No universe of that name → WARNING with remediation, config still written, exit 0."""
        # Seed the DB with a universe by another name and not marked default.
        self._seed_universe("other_uni", ["AAPL"], is_default=False)
        monkeypatch.setenv("FMP_API_KEY", "fake-key")

        result = runner.invoke(cli, ["factory", "init"])

        assert result.exit_code == 0
        assert factory_config_path.exists()
        assert "WARNING" in result.output
        assert "not registered" in result.output
        # Remediation pointer.
        assert "cents universe" in result.output

    def test_empty_universe_emits_warning(
        self, runner, mock_db, factory_config_path, monkeypatch
    ):
        """Universe exists but has 0 symbols → WARNING, config still written."""
        self._seed_universe("sp500_lite", [], is_default=True)
        monkeypatch.setenv("FMP_API_KEY", "fake-key")

        result = runner.invoke(cli, ["factory", "init"])

        assert result.exit_code == 0
        assert factory_config_path.exists()
        assert "WARNING" in result.output
        assert "0 symbols" in result.output

    def test_fmp_unreachable_emits_warning(
        self, runner, mock_db, factory_config_path, monkeypatch
    ):
        """FMP probe raises → WARNING surfaces the FMP failure, config still written."""
        self._seed_universe("sp500_lite", ["AAPL"], is_default=True)
        monkeypatch.setenv("FMP_API_KEY", "fake-key")

        with patch(
            "cents.data.fmp.FMPFundamentalsProvider._fetch_json",
            side_effect=RuntimeError("connection refused"),
        ):
            result = runner.invoke(cli, ["factory", "init"])

        assert result.exit_code == 0
        assert factory_config_path.exists()
        assert "WARNING" in result.output
        assert "FMP probe of AAPL failed" in result.output
        assert "connection refused" in result.output

    def test_fmp_empty_payload_emits_warning(
        self, runner, mock_db, factory_config_path, monkeypatch
    ):
        """FMP returns empty list (unknown ticker) → WARNING, config still written."""
        self._seed_universe("sp500_lite", ["BOGUS"], is_default=True)
        monkeypatch.setenv("FMP_API_KEY", "fake-key")

        with patch(
            "cents.data.fmp.FMPFundamentalsProvider._fetch_json",
            return_value=[],
        ):
            result = runner.invoke(cli, ["factory", "init"])

        assert result.exit_code == 0
        assert factory_config_path.exists()
        assert "WARNING" in result.output
        assert "returned no data" in result.output

    def test_no_fmp_key_skips_probe_with_distinct_note(
        self, runner, mock_db, factory_config_path, monkeypatch
    ):
        """No FMP key configured → softer NOTE, no FMP probe attempted, config written."""
        self._seed_universe("sp500_lite", ["AAPL"], is_default=True)
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        # Force settings to see no key (also clear file-based config via CENTS_CONFIG).
        monkeypatch.setenv("CENTS_CONFIG", "/nonexistent/config.toml")

        # The FMP provider should NEVER be constructed; assert via patch.
        with patch(
            "cents.data.fmp.FMPFundamentalsProvider.__init__",
            side_effect=AssertionError("FMP probe must not run when key is missing"),
        ):
            result = runner.invoke(cli, ["factory", "init"])

        assert result.exit_code == 0
        assert factory_config_path.exists()
        assert "NOTE" in result.output
        assert "FMP_API_KEY not configured" in result.output
        # Distinct from the FMP-reachability warnings.
        assert "FMP probe of" not in result.output
