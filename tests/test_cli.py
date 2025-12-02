"""Tests for CLI commands."""

import sqlite3
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cents.cli import cli
from cents.db.schema import SCHEMA


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_db(tmp_path):
    """Create temporary database for CLI tests."""
    db_path = tmp_path / "data" / "cents.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
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


class TestResearchCLI:
    """Tests for research command."""

    def test_research_command_help(self, runner):
        """Research command help."""
        result = runner.invoke(cli, ["research", "--help"])
        assert result.exit_code == 0
        assert "Run research agents" in result.output

    @patch("cents.cli.AGENTS")
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
        mock_agents.items.return_value = [("test", mock_agent_class)]

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "AAPL", "--no-save"])
            assert result.exit_code == 0
            assert "conviction delta" in result.output.lower()

    @patch("cents.cli.AGENTS")
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
        mock_agents.items.return_value = [("test", mock_agent_class)]

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "AAPL", "--output", "json", "--no-save"])
            assert result.exit_code == 0
            import json
            data = json.loads(result.output)
            assert data["symbol"] == "AAPL"
            assert data["total_conviction_delta"] == 3.5
            assert "agents" in data

    @patch("cents.cli.AGENTS")
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
        mock_agents.items.return_value = [("fundamentals", mock_agent_class)]

        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["research", "NVDA", "--suggest-thesis", "--no-save"])
            assert result.exit_code == 0
            assert "THESIS SUGGESTION" in result.output
            assert "NVDA" in result.output

    @patch("cents.cli.AGENTS")
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
        mock_agents.items.return_value = [("test", mock_agent_class)]

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
