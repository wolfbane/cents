"""Tests for repository layer."""

import pytest

from cents.db import (
    ThesisRepository,
    PositionRepository,
    OutcomeRepository,
    EvidenceRepository,
    WatchlistRepository,
    AlertRepository,
)
from cents.models import (
    Thesis,
    ThesisStatus,
    Position,
    PositionSide,
    PositionStatus,
    Outcome,
    ThesisAccuracy,
    Evidence,
    EvidenceType,
    WatchlistItem,
    Alert,
    AlertType,
)


class TestThesisRepository:
    """Tests for ThesisRepository."""

    def test_create_and_get(self, db_conn):
        """Create and retrieve a thesis."""
        repo = ThesisRepository(db_conn)
        thesis = Thesis(title="Test thesis", hypothesis="Test hypothesis")
        repo.create(thesis)

        retrieved = repo.get(thesis.id)
        assert retrieved is not None
        assert retrieved.title == "Test thesis"
        assert retrieved.hypothesis == "Test hypothesis"
        assert retrieved.conviction == 50.0

    def test_get_nonexistent(self, db_conn):
        """Get returns None for missing thesis."""
        repo = ThesisRepository(db_conn)
        assert repo.get("nonexistent") is None

    def test_list_all(self, db_conn):
        """List all theses."""
        repo = ThesisRepository(db_conn)
        repo.create(Thesis(title="Thesis 1"))
        repo.create(Thesis(title="Thesis 2"))

        theses = repo.list()
        assert len(theses) == 2

    def test_list_by_status(self, db_conn):
        """List theses filtered by status."""
        repo = ThesisRepository(db_conn)
        t1 = Thesis(title="Open thesis")
        t2 = Thesis(title="Closed thesis", status=ThesisStatus.CLOSED)
        repo.create(t1)
        repo.create(t2)

        open_theses = repo.list(status=ThesisStatus.OPEN)
        assert len(open_theses) == 1
        assert open_theses[0].title == "Open thesis"

    def test_update(self, db_conn):
        """Update thesis fields."""
        repo = ThesisRepository(db_conn)
        thesis = Thesis(title="Original")
        repo.create(thesis)

        thesis.title = "Updated"
        thesis.conviction = 75.0
        repo.update(thesis)

        retrieved = repo.get(thesis.id)
        assert retrieved.title == "Updated"
        assert retrieved.conviction == 75.0

    def test_delete(self, db_conn):
        """Delete a thesis."""
        repo = ThesisRepository(db_conn)
        thesis = Thesis(title="To delete")
        repo.create(thesis)

        assert repo.delete(thesis.id) is True
        assert repo.get(thesis.id) is None

    def test_delete_nonexistent(self, db_conn):
        """Delete returns False for missing thesis."""
        repo = ThesisRepository(db_conn)
        assert repo.delete("nonexistent") is False

    def test_tags_serialization(self, db_conn):
        """Tags are stored and retrieved correctly."""
        repo = ThesisRepository(db_conn)
        thesis = Thesis(title="Tagged", tags=["tech", "AI", "growth"])
        repo.create(thesis)

        retrieved = repo.get(thesis.id)
        assert retrieved.tags == ["tech", "AI", "growth"]


class TestPositionRepository:
    """Tests for PositionRepository."""

    def test_create_and_get(self, db_conn):
        """Create and retrieve a position."""
        repo = PositionRepository(db_conn)
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=150.0,
            size=100,
        )
        repo.create(pos)

        retrieved = repo.get(pos.id)
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.entry_price == 150.0

    def test_list_by_status(self, db_conn):
        """List positions by status."""
        repo = PositionRepository(db_conn)
        p1 = Position(symbol="AAPL", side=PositionSide.LONG, entry_price=100, size=10)
        p2 = Position(symbol="GOOG", side=PositionSide.LONG, entry_price=100, size=10)
        p2.close(110)
        repo.create(p1)
        repo.create(p2)

        open_positions = repo.list(status=PositionStatus.OPEN)
        assert len(open_positions) == 1
        assert open_positions[0].symbol == "AAPL"

    def test_update_after_close(self, db_conn):
        """Update position after closing."""
        repo = PositionRepository(db_conn)
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10,
        )
        repo.create(pos)

        pos.close(120.0)
        repo.update(pos)

        retrieved = repo.get(pos.id)
        assert retrieved.status == PositionStatus.CLOSED
        assert retrieved.exit_price == 120.0


class TestOutcomeRepository:
    """Tests for OutcomeRepository."""

    def test_create_and_get_for_position(self, db_conn):
        """Create outcome and retrieve by position."""
        # Create parent position first (FK constraint)
        pos_repo = PositionRepository(db_conn)
        position = Position(
            id="pos123",
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10,
        )
        pos_repo.create(position)

        repo = OutcomeRepository(db_conn)
        outcome = Outcome(
            position_id="pos123",
            pnl=100.0,
            pnl_pct=10.0,
            thesis_accuracy=ThesisAccuracy.CORRECT,
        )
        repo.create(outcome)

        retrieved = repo.get_for_position("pos123")
        assert retrieved is not None
        assert retrieved.pnl == 100.0
        assert retrieved.thesis_accuracy == ThesisAccuracy.CORRECT

    def test_get_for_nonexistent_position(self, db_conn):
        """Returns None for position without outcome."""
        repo = OutcomeRepository(db_conn)
        assert repo.get_for_position("nonexistent") is None

    def test_list(self, db_conn):
        """List all outcomes."""
        # Create parent positions first (FK constraint)
        pos_repo = PositionRepository(db_conn)
        pos_repo.create(Position(id="p1", symbol="AAPL", side=PositionSide.LONG, entry_price=100, size=10))
        pos_repo.create(Position(id="p2", symbol="MSFT", side=PositionSide.LONG, entry_price=200, size=5))

        repo = OutcomeRepository(db_conn)
        repo.create(Outcome(position_id="p1", pnl=100, pnl_pct=10))
        repo.create(Outcome(position_id="p2", pnl=-50, pnl_pct=-5))

        outcomes = repo.list()
        assert len(outcomes) == 2


class TestEvidenceRepository:
    """Tests for EvidenceRepository."""

    def test_create_and_list_for_thesis(self, db_conn):
        """Create evidence and list by thesis."""
        # Create parent thesis first (FK constraint)
        thesis_repo = ThesisRepository(db_conn)
        thesis = Thesis(id="t123", title="Test thesis")
        thesis_repo.create(thesis)

        repo = EvidenceRepository(db_conn)
        e1 = Evidence(
            thesis_id="t123",
            agent="fundamentals",
            type=EvidenceType.SUPPORTING,
            content="Strong earnings",
            source="yfinance",
        )
        e2 = Evidence(
            thesis_id="t123",
            agent="technical",
            type=EvidenceType.NEUTRAL,
            content="Sideways trend",
            source="price_analysis",
        )
        repo.create(e1)
        repo.create(e2)

        evidence = repo.list_for_thesis("t123")
        assert len(evidence) == 2

    def test_list_for_nonexistent_thesis(self, db_conn):
        """Empty list for thesis without evidence."""
        repo = EvidenceRepository(db_conn)
        assert repo.list_for_thesis("nonexistent") == []


class TestWatchlistRepository:
    """Tests for WatchlistRepository."""

    def test_add_and_get(self, db_conn):
        """Add and retrieve watchlist item."""
        repo = WatchlistRepository(db_conn)
        item = WatchlistItem(
            symbol="AAPL",
            notes="Watch for earnings",
            threshold=4.5,
            alert_destination="https://example.com/webhook",
        )
        repo.add(item)

        retrieved = repo.get("AAPL")
        assert retrieved is not None
        assert retrieved.notes == "Watch for earnings"
        assert retrieved.threshold == 4.5
        assert retrieved.alert_destination == "https://example.com/webhook"

    def test_symbol_normalized_to_uppercase(self, db_conn):
        """Symbols are stored uppercase."""
        repo = WatchlistRepository(db_conn)
        item = WatchlistItem(symbol="aapl")
        repo.add(item)

        assert repo.get("AAPL") is not None
        assert repo.get("aapl") is not None

    def test_remove(self, db_conn):
        """Remove item from watchlist."""
        repo = WatchlistRepository(db_conn)
        repo.add(WatchlistItem(symbol="AAPL"))

        assert repo.remove("AAPL") is True
        assert repo.get("AAPL") is None

    def test_remove_nonexistent(self, db_conn):
        """Remove returns False for missing item."""
        repo = WatchlistRepository(db_conn)
        assert repo.remove("NONEXISTENT") is False

    def test_list(self, db_conn):
        """List all watchlist items."""
        repo = WatchlistRepository(db_conn)
        repo.add(WatchlistItem(symbol="AAPL"))
        repo.add(WatchlistItem(symbol="GOOG"))

        items = repo.list()
        assert len(items) == 2


class TestAlertRepository:
    """Tests for AlertRepository."""

    def test_create_and_list_unread(self, db_conn):
        """Create alert and list unread."""
        repo = AlertRepository(db_conn)
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Significant movement",
        )
        repo.create(alert)

        unread = repo.list_unread()
        assert len(unread) == 1
        assert unread[0].read is False

    def test_mark_read(self, db_conn):
        """Mark alert as read."""
        repo = AlertRepository(db_conn)
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test",
        )
        repo.create(alert)

        repo.mark_read(alert.id)
        unread = repo.list_unread()
        assert len(unread) == 0

    def test_mark_all_read(self, db_conn):
        """Mark all alerts as read."""
        repo = AlertRepository(db_conn)
        repo.create(Alert(symbol="AAPL", alert_type=AlertType.CONVICTION_CHANGE, message="1"))
        repo.create(Alert(symbol="GOOG", alert_type=AlertType.CONVICTION_CHANGE, message="2"))

        count = repo.mark_all_read()
        assert count == 2
        assert repo.list_unread() == []
