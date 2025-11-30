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
            result = runner.invoke(cli, ["thesis", "create", "Test thesis"])
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
            runner.invoke(cli, ["thesis", "create", "Thesis 1"])
            runner.invoke(cli, ["thesis", "create", "Thesis 2"])
            result = runner.invoke(cli, ["thesis", "list"])
            assert result.exit_code == 0
            assert "Thesis 1" in result.output
            assert "Thesis 2" in result.output

    def test_thesis_show(self, runner, mock_db):
        """Show thesis details."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            create_result = runner.invoke(cli, ["thesis", "create", "Test thesis"])
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
            create_result = runner.invoke(cli, ["thesis", "create", "Test"])
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
                cli, ["position", "open", "AAPL", "100", "--price", "150"]
            )
            assert result.exit_code == 0
            assert "Opened long position" in result.output
            assert "AAPL" in result.output
            assert "$150.00" in result.output

    def test_position_open_short(self, runner, mock_db):
        """Open a short position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(
                cli, ["position", "open", "AAPL", "100", "--price", "150", "--short"]
            )
            assert result.exit_code == 0
            assert "Opened short position" in result.output

    def test_position_close(self, runner, mock_db):
        """Close a position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")

            result = runner.invoke(cli, ["position", "close", pos_id, "110"])
            assert result.exit_code == 0
            assert "Closed position" in result.output
            assert "+$1000.00" in result.output
            assert "+10.0%" in result.output

    def test_position_close_loss(self, runner, mock_db):
        """Close a position at a loss."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            open_result = runner.invoke(
                cli, ["position", "open", "AAPL", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")

            result = runner.invoke(cli, ["position", "close", pos_id, "90"])
            assert result.exit_code == 0
            assert "-1000.00" in result.output
            assert "-10.0%" in result.output

    def test_position_close_not_found(self, runner, mock_db):
        """Close nonexistent position."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            result = runner.invoke(cli, ["position", "close", "nonexistent", "100"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_position_list(self, runner, mock_db):
        """List positions."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["position", "open", "AAPL", "100", "--price", "150"])
            runner.invoke(cli, ["position", "open", "GOOG", "50", "--price", "100"])

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
                cli, ["position", "open", "AAPL", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")
            runner.invoke(cli, ["position", "close", pos_id, "110"])

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
                cli, ["position", "open", "AAPL", "100", "--price", "100"]
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
                cli, ["position", "open", "AAPL", "100", "--price", "100"]
            )
            pos_id = open_result.output.split()[3].rstrip(":")
            runner.invoke(cli, ["position", "close", pos_id, "110"])
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

    def test_watch_add_duplicate(self, runner, mock_db):
        """Adding duplicate shows warning."""
        with runner.isolated_filesystem(temp_dir=mock_db):
            runner.invoke(cli, ["watch", "add", "AAPL"])
            result = runner.invoke(cli, ["watch", "add", "AAPL"])
            assert "already on watchlist" in result.output

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
