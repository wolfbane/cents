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

CREATE INDEX IF NOT EXISTS idx_theses_status ON theses(status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_evidence_thesis ON evidence(thesis_id);
"""


def get_db_path() -> Path:
    """Get the default database path."""
    # Look for data dir relative to working directory
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "cents.db"


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize database with schema and return connection."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get database connection, creating schema if needed."""
    return init_db(db_path)
