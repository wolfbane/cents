"""SQLite schema and database initialization."""

import sqlite3
from pathlib import Path

# Singleton connection for the application
_connection: sqlite3.Connection | None = None
_db_path: Path | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS theses (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    hypothesis TEXT DEFAULT '',
    status TEXT DEFAULT 'open',
    conviction REAL DEFAULT 50.0,
    tags TEXT DEFAULT '[]',
    symbol TEXT,
    business_quality TEXT,
    valuation TEXT,
    moat TEXT,
    time_horizon TEXT,
    horizon_end TEXT,
    key_risks TEXT DEFAULT '[]',
    target_price REAL,
    stop_price REAL,
    outcome TEXT,
    closed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    thesis_id TEXT,
    symbol TEXT,
    agent TEXT NOT NULL,
    type TEXT DEFAULT 'neutral',
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    dimension TEXT,
    metadata TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
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
    FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
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
    FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE CASCADE
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
    FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
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

CREATE TABLE IF NOT EXISTS backtests (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_signals (
    id TEXT PRIMARY KEY,
    backtest_id TEXT NOT NULL,
    date TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    conviction_delta REAL NOT NULL,
    dimension_scores TEXT DEFAULT '{}',
    forward_returns TEXT DEFAULT '{}',
    FOREIGN KEY (backtest_id) REFERENCES backtests(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_theses_status ON theses(status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_evidence_thesis ON evidence(thesis_id);
CREATE INDEX IF NOT EXISTS idx_evidence_thesis_timestamp ON evidence(thesis_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(read);
CREATE INDEX IF NOT EXISTS idx_backtests_symbol ON backtests(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_signals_backtest ON backtest_signals(backtest_id);

CREATE TABLE IF NOT EXISTS api_cache (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    response_data TEXT NOT NULL,
    cached_at TEXT NOT NULL,
    UNIQUE(provider, endpoint, cache_key)
);

CREATE INDEX IF NOT EXISTS idx_api_cache_lookup ON api_cache(provider, endpoint, cache_key);
"""


def get_db_path() -> Path:
    """Get the database path based on configuration.

    Priority:
    1. CENTS_DB_PATH environment variable (highest)
    2. Active dataset from ~/.cents/datasets.toml
    3. Default: ~/.cents/data/cents.db
    """
    import os

    # 1. Allow override via environment variable (highest priority)
    if env_path := os.environ.get("CENTS_DB_PATH"):
        path = Path(env_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # 2. Check for active dataset
    from cents.datasets import get_active_dataset

    _name, path = get_active_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply schema migrations for existing databases."""
    column_migrations = [
        # Add threshold and alert_destination to watchlist (added in v0.2)
        ("watchlist", "threshold", "ALTER TABLE watchlist ADD COLUMN threshold REAL"),
        ("watchlist", "alert_destination", "ALTER TABLE watchlist ADD COLUMN alert_destination TEXT"),
        # Add structured thesis fields (added in v0.3)
        ("theses", "symbol", "ALTER TABLE theses ADD COLUMN symbol TEXT"),
        ("theses", "business_quality", "ALTER TABLE theses ADD COLUMN business_quality TEXT"),
        ("theses", "valuation", "ALTER TABLE theses ADD COLUMN valuation TEXT"),
        ("theses", "moat", "ALTER TABLE theses ADD COLUMN moat TEXT"),
        ("theses", "time_horizon", "ALTER TABLE theses ADD COLUMN time_horizon TEXT"),
        ("theses", "horizon_end", "ALTER TABLE theses ADD COLUMN horizon_end TEXT"),
        ("theses", "key_risks", "ALTER TABLE theses ADD COLUMN key_risks TEXT DEFAULT '[]'"),
        # Add resolution trigger fields (added in v0.4)
        ("theses", "target_price", "ALTER TABLE theses ADD COLUMN target_price REAL"),
        ("theses", "stop_price", "ALTER TABLE theses ADD COLUMN stop_price REAL"),
        ("theses", "outcome", "ALTER TABLE theses ADD COLUMN outcome TEXT"),
        ("theses", "closed_at", "ALTER TABLE theses ADD COLUMN closed_at TEXT"),
        # Add dimension to evidence (added in v0.5)
        ("evidence", "dimension", "ALTER TABLE evidence ADD COLUMN dimension TEXT"),
        # Add symbol to evidence for standalone research (added in v0.7)
        ("evidence", "symbol", "ALTER TABLE evidence ADD COLUMN symbol TEXT"),
    ]

    for table, column, sql in column_migrations:
        # Check if column exists
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            conn.execute(sql)

    # Migrate foreign keys to include ON DELETE actions (added in v0.6)
    _migrate_foreign_keys(conn)


def _migrate_foreign_keys(conn: sqlite3.Connection) -> None:
    """Migrate tables to use ON DELETE CASCADE/SET NULL for foreign keys.

    SQLite doesn't support altering foreign key constraints, so we must
    recreate the affected tables. This migration is idempotent.
    """
    # Check if all required tables exist (some tests create partial schemas)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('evidence', 'positions', 'outcomes', 'watchlist')"
    )
    existing_tables = {row[0] for row in cursor.fetchall()}
    required_tables = {"evidence", "positions", "outcomes", "watchlist"}
    if not required_tables.issubset(existing_tables):
        return  # Not all tables exist yet, skip migration

    # Check if migration is needed by inspecting foreign_key_list
    # Evidence should have SET NULL, others should have CASCADE/SET NULL
    cursor = conn.execute("PRAGMA foreign_key_list(evidence)")
    fk_info = cursor.fetchall()
    if fk_info and fk_info[0][6] == "SET NULL":  # on_delete is column 6
        return  # Already migrated to latest (SET NULL for evidence)

    # Temporarily disable foreign keys for the migration
    conn.execute("PRAGMA foreign_keys = OFF")

    try:
        # Migrate evidence table (ON DELETE SET NULL, nullable thesis_id, add symbol)
        conn.execute("ALTER TABLE evidence RENAME TO evidence_old")
        conn.execute("""
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY,
                thesis_id TEXT,
                symbol TEXT,
                agent TEXT NOT NULL,
                type TEXT DEFAULT 'neutral',
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                dimension TEXT,
                metadata TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
            )
        """)
        conn.execute("""
            INSERT INTO evidence (id, thesis_id, agent, type, content, source, confidence, dimension, metadata, timestamp)
            SELECT id, thesis_id, agent, type, content, source, confidence, dimension, metadata, timestamp
            FROM evidence_old
        """)
        conn.execute("DROP TABLE evidence_old")

        # Migrate positions table (ON DELETE SET NULL)
        conn.execute("ALTER TABLE positions RENAME TO positions_old")
        conn.execute("""
            CREATE TABLE positions (
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
                FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
            )
        """)
        conn.execute("""
            INSERT INTO positions SELECT * FROM positions_old
        """)
        conn.execute("DROP TABLE positions_old")

        # Migrate outcomes table (ON DELETE CASCADE)
        conn.execute("ALTER TABLE outcomes RENAME TO outcomes_old")
        conn.execute("""
            CREATE TABLE outcomes (
                id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                thesis_accuracy TEXT DEFAULT 'unclear',
                agent_performance TEXT DEFAULT '{}',
                retrospective TEXT DEFAULT '',
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            INSERT INTO outcomes SELECT * FROM outcomes_old
        """)
        conn.execute("DROP TABLE outcomes_old")

        # Migrate watchlist table (ON DELETE SET NULL)
        conn.execute("ALTER TABLE watchlist RENAME TO watchlist_old")
        conn.execute("""
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                notes TEXT DEFAULT '',
                thesis_id TEXT,
                threshold REAL,
                alert_destination TEXT,
                last_scanned TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
            )
        """)
        conn.execute("""
            INSERT INTO watchlist SELECT * FROM watchlist_old
        """)
        conn.execute("DROP TABLE watchlist_old")

        # Recreate indexes that were dropped with the tables
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_thesis ON evidence(thesis_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol)")

        conn.commit()
    finally:
        # Re-enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize database with schema and return connection."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get or create the singleton database connection.

    The connection is cached and reused across all repositories.
    Use close_connection() to explicitly close it (e.g., for testing).

    Args:
        db_path: Optional path to database file. Only used on first call.

    Returns:
        The shared database connection.
    """
    global _connection, _db_path

    if _connection is not None:
        return _connection

    _db_path = db_path
    _connection = init_db(db_path)
    return _connection


def close_connection() -> None:
    """Close the singleton database connection.

    Call this when shutting down the application or between tests.
    The next call to get_connection() will create a new connection.
    """
    global _connection, _db_path

    if _connection is not None:
        _connection.close()
        _connection = None
        _db_path = None


def reset_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Close existing connection and create a new one.

    Useful for testing or when switching databases.

    Args:
        db_path: Optional path to database file.

    Returns:
        A fresh database connection.
    """
    close_connection()
    return get_connection(db_path)
