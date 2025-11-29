"""Repository layer for CRUD operations."""

import json
import sqlite3
from datetime import date, datetime
from typing import Optional

from cents.models import (
    Alert,
    AlertType,
    Evidence,
    EvidenceType,
    Outcome,
    Position,
    PositionSide,
    PositionStatus,
    Thesis,
    ThesisAccuracy,
    ThesisStatus,
    WatchlistItem,
)
from cents.db.schema import get_connection


class ThesisRepository:
    """CRUD operations for theses."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    def create(self, thesis: Thesis) -> Thesis:
        """Insert a new thesis."""
        self.conn.execute(
            """
            INSERT INTO theses (id, title, hypothesis, status, conviction, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thesis.id,
                thesis.title,
                thesis.hypothesis,
                thesis.status.value,
                thesis.conviction,
                json.dumps(thesis.tags),
                thesis.created_at.isoformat(),
                thesis.updated_at.isoformat(),
            ),
        )
        self.conn.commit()
        return thesis

    def get(self, thesis_id: str) -> Optional[Thesis]:
        """Get thesis by ID."""
        row = self.conn.execute(
            "SELECT * FROM theses WHERE id = ?", (thesis_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_thesis(row)

    def list(self, status: Optional[ThesisStatus] = None) -> list[Thesis]:
        """List theses, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM theses WHERE status = ? ORDER BY updated_at DESC",
                (status.value,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM theses ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_thesis(row) for row in rows]

    def update(self, thesis: Thesis) -> Thesis:
        """Update an existing thesis."""
        thesis.updated_at = datetime.now()
        self.conn.execute(
            """
            UPDATE theses
            SET title = ?, hypothesis = ?, status = ?, conviction = ?, tags = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                thesis.title,
                thesis.hypothesis,
                thesis.status.value,
                thesis.conviction,
                json.dumps(thesis.tags),
                thesis.updated_at.isoformat(),
                thesis.id,
            ),
        )
        self.conn.commit()
        return thesis

    def delete(self, thesis_id: str) -> bool:
        """Delete a thesis by ID."""
        cursor = self.conn.execute("DELETE FROM theses WHERE id = ?", (thesis_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def _row_to_thesis(self, row: sqlite3.Row) -> Thesis:
        return Thesis(
            id=row["id"],
            title=row["title"],
            hypothesis=row["hypothesis"],
            status=ThesisStatus(row["status"]),
            conviction=row["conviction"],
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


class PositionRepository:
    """CRUD operations for positions."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    def create(self, position: Position) -> Position:
        """Insert a new position."""
        self.conn.execute(
            """
            INSERT INTO positions
            (id, thesis_id, symbol, side, entry_price, entry_date, size, status, exit_price, exit_date, paper, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.id,
                position.thesis_id,
                position.symbol,
                position.side.value,
                position.entry_price,
                position.entry_date.isoformat(),
                position.size,
                position.status.value,
                position.exit_price,
                position.exit_date.isoformat() if position.exit_date else None,
                1 if position.paper else 0,
                position.notes,
                position.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return position

    def get(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        row = self.conn.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_position(row)

    def list(self, status: Optional[PositionStatus] = None) -> list[Position]:
        """List positions, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM positions ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_position(row) for row in rows]

    def update(self, position: Position) -> Position:
        """Update an existing position."""
        self.conn.execute(
            """
            UPDATE positions
            SET thesis_id = ?, symbol = ?, side = ?, entry_price = ?, entry_date = ?,
                size = ?, status = ?, exit_price = ?, exit_date = ?, paper = ?, notes = ?
            WHERE id = ?
            """,
            (
                position.thesis_id,
                position.symbol,
                position.side.value,
                position.entry_price,
                position.entry_date.isoformat(),
                position.size,
                position.status.value,
                position.exit_price,
                position.exit_date.isoformat() if position.exit_date else None,
                1 if position.paper else 0,
                position.notes,
                position.id,
            ),
        )
        self.conn.commit()
        return position

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        return Position(
            id=row["id"],
            thesis_id=row["thesis_id"],
            symbol=row["symbol"],
            side=PositionSide(row["side"]),
            entry_price=row["entry_price"],
            entry_date=date.fromisoformat(row["entry_date"]),
            size=row["size"],
            status=PositionStatus(row["status"]),
            exit_price=row["exit_price"],
            exit_date=date.fromisoformat(row["exit_date"]) if row["exit_date"] else None,
            paper=bool(row["paper"]),
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


class EvidenceRepository:
    """CRUD operations for evidence."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    def create(self, evidence: Evidence) -> Evidence:
        """Insert new evidence."""
        self.conn.execute(
            """
            INSERT INTO evidence (id, thesis_id, agent, type, content, source, confidence, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.thesis_id,
                evidence.agent,
                evidence.type.value,
                evidence.content,
                evidence.source,
                evidence.confidence,
                json.dumps(evidence.metadata),
                evidence.timestamp.isoformat(),
            ),
        )
        self.conn.commit()
        return evidence

    def list_for_thesis(self, thesis_id: str) -> list[Evidence]:
        """List all evidence for a thesis."""
        rows = self.conn.execute(
            "SELECT * FROM evidence WHERE thesis_id = ? ORDER BY timestamp DESC",
            (thesis_id,),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def _row_to_evidence(self, row: sqlite3.Row) -> Evidence:
        return Evidence(
            id=row["id"],
            thesis_id=row["thesis_id"],
            agent=row["agent"],
            type=EvidenceType(row["type"]),
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )


class OutcomeRepository:
    """CRUD operations for outcomes."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    def create(self, outcome: Outcome) -> Outcome:
        """Insert a new outcome."""
        self.conn.execute(
            """
            INSERT INTO outcomes (id, position_id, pnl, pnl_pct, thesis_accuracy, agent_performance, retrospective, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.id,
                outcome.position_id,
                outcome.pnl,
                outcome.pnl_pct,
                outcome.thesis_accuracy.value,
                json.dumps(outcome.agent_performance),
                outcome.retrospective,
                outcome.recorded_at.isoformat(),
            ),
        )
        self.conn.commit()
        return outcome

    def get_for_position(self, position_id: str) -> Optional[Outcome]:
        """Get outcome for a position."""
        row = self.conn.execute(
            "SELECT * FROM outcomes WHERE position_id = ?", (position_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_outcome(row)

    def list(self) -> list[Outcome]:
        """List all outcomes."""
        rows = self.conn.execute(
            "SELECT * FROM outcomes ORDER BY recorded_at DESC"
        ).fetchall()
        return [self._row_to_outcome(row) for row in rows]

    def _row_to_outcome(self, row: sqlite3.Row) -> Outcome:
        return Outcome(
            id=row["id"],
            position_id=row["position_id"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            thesis_accuracy=ThesisAccuracy(row["thesis_accuracy"]),
            agent_performance=json.loads(row["agent_performance"]),
            retrospective=row["retrospective"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
        )


class WatchlistRepository:
    """CRUD operations for watchlist."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    def add(self, item: WatchlistItem) -> WatchlistItem:
        """Add a symbol to watchlist."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO watchlist (id, symbol, notes, thesis_id, last_scanned, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.symbol.upper(),
                item.notes,
                item.thesis_id,
                item.last_scanned.isoformat() if item.last_scanned else None,
                item.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return item

    def remove(self, symbol: str) -> bool:
        """Remove a symbol from watchlist."""
        cursor = self.conn.execute(
            "DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get(self, symbol: str) -> Optional[WatchlistItem]:
        """Get watchlist item by symbol."""
        row = self.conn.execute(
            "SELECT * FROM watchlist WHERE symbol = ?", (symbol.upper(),)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def list(self) -> list[WatchlistItem]:
        """List all watchlist items."""
        rows = self.conn.execute(
            "SELECT * FROM watchlist ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def update_scanned(self, symbol: str) -> None:
        """Update last_scanned timestamp."""
        self.conn.execute(
            "UPDATE watchlist SET last_scanned = ? WHERE symbol = ?",
            (datetime.now().isoformat(), symbol.upper()),
        )
        self.conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> WatchlistItem:
        return WatchlistItem(
            id=row["id"],
            symbol=row["symbol"],
            notes=row["notes"],
            thesis_id=row["thesis_id"],
            last_scanned=datetime.fromisoformat(row["last_scanned"]) if row["last_scanned"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )


class AlertRepository:
    """CRUD operations for alerts."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    def create(self, alert: Alert) -> Alert:
        """Create a new alert."""
        self.conn.execute(
            """
            INSERT INTO alerts (id, symbol, alert_type, message, data, read, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.id,
                alert.symbol,
                alert.alert_type.value,
                alert.message,
                json.dumps(alert.data),
                1 if alert.read else 0,
                alert.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return alert

    def list_unread(self) -> list[Alert]:
        """List unread alerts."""
        rows = self.conn.execute(
            "SELECT * FROM alerts WHERE read = 0 ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def list_all(self, limit: int = 50) -> list[Alert]:
        """List all alerts."""
        rows = self.conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def mark_read(self, alert_id: str) -> bool:
        """Mark an alert as read."""
        cursor = self.conn.execute(
            "UPDATE alerts SET read = 1 WHERE id = ?", (alert_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def mark_all_read(self) -> int:
        """Mark all alerts as read."""
        cursor = self.conn.execute("UPDATE alerts SET read = 1 WHERE read = 0")
        self.conn.commit()
        return cursor.rowcount

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            symbol=row["symbol"],
            alert_type=AlertType(row["alert_type"]),
            message=row["message"],
            data=json.loads(row["data"]),
            read=bool(row["read"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
