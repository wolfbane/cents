"""SQLite schema and database initialization."""

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS theses (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    hypothesis TEXT DEFAULT '',
    status TEXT DEFAULT 'open',
    conviction REAL DEFAULT 50.0,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    thesis_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    type TEXT DEFAULT 'neutral',
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    metadata TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (thesis_id) REFERENCES theses(id)
);

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    thesis_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    size REAL NOT NULL,
    status TEXT DEFAULT 'open',
    exit_price REAL,
    exit_date TEXT,
    paper INTEGER DEFAULT 1,
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (thesis_id) REFERENCES theses(id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    pnl REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    thesis_accuracy TEXT DEFAULT 'unclear',
    agent_performance TEXT DEFAULT '{}',
    retrospective TEXT DEFAULT '',
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

CREATE TABLE IF NOT EXISTS watchlist (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    notes TEXT DEFAULT '',
    thesis_id TEXT,
    threshold REAL,
    alert_destination TEXT,
    last_scanned TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (thesis_id) REFERENCES theses(id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT DEFAULT '{}',
    read INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_theses_status ON theses(status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_evidence_thesis ON evidence(thesis_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(read);
"""


def get_db_path() -> Path:
    """Get the default database path."""
    # Look for data dir relative to working directory
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "cents.db"


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply schema migrations for existing databases."""
    migrations = [
        # Add threshold and alert_destination to watchlist (added in v0.2)
        ("watchlist", "threshold", "ALTER TABLE watchlist ADD COLUMN threshold REAL"),
        ("watchlist", "alert_destination", "ALTER TABLE watchlist ADD COLUMN alert_destination TEXT"),
    ]

    for table, column, sql in migrations:
        # Check if column exists
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            conn.execute(sql)


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize database with schema and return connection."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get database connection, creating schema if needed."""
    return init_db(db_path)
