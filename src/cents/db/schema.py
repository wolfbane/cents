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
    cohort TEXT DEFAULT 'directional',
    hedge_symbol TEXT,
    paired_thesis_id TEXT,
    premise_tags TEXT DEFAULT '[]',
    premise_direction TEXT DEFAULT '{}',
    regime_snapshot TEXT DEFAULT '{}',
    discovery_source TEXT,
    calibrated_p_correct REAL,
    calibration_fit_at TEXT,
    orchestrator_label TEXT DEFAULT 'llm',
    experiment_id TEXT,
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
    -- Provenance fields linking an Evidence row to the LLM call that produced
    -- it. NULL for non-LLM evidence (keyword sentiment, FMP fundamentals, …).
    -- `llm_call_id` references `llm_usage.id` but the FK is intentionally not
    -- enforced so evidence survives `llm_usage` pruning.
    llm_call_id TEXT,
    model_snapshot TEXT,
    prompt_sha256 TEXT,
    input_sha256 TEXT,
    output_sha256 TEXT,
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
    costs_applied_usd REAL DEFAULT 0.0,
    realized_exit_price REAL,
    sizing_method TEXT,
    borrow_rate_pa_pct REAL,
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
CREATE INDEX IF NOT EXISTS idx_alerts_type_created ON alerts(alert_type, created_at DESC);
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

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT DEFAULT '',
    url TEXT DEFAULT '',
    occurred_at TEXT NOT NULL,
    affected_symbols TEXT DEFAULT '[]',
    affected_sectors TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    polarity TEXT DEFAULT 'unclear',
    confidence REAL DEFAULT 0.5,
    raw_text TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    tag_status TEXT DEFAULT 'tagger_skipped',
    ingested_at TEXT NOT NULL,
    UNIQUE(source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_events_occurred ON events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);

CREATE TABLE IF NOT EXISTS llm_usage (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    agent TEXT NOT NULL,
    operation TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    context TEXT,
    called_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_called_at ON llm_usage(called_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_agent ON llm_usage(agent);

CREATE TABLE IF NOT EXISTS universes (
    name TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    source TEXT NOT NULL DEFAULT 'static',
    source_config TEXT NOT NULL DEFAULT '{}',
    symbols TEXT NOT NULL DEFAULT '[]',
    is_default INTEGER NOT NULL DEFAULT 0,
    id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_universes_default ON universes(is_default);

CREATE TABLE IF NOT EXISTS factory_runs (
    id TEXT PRIMARY KEY,
    universe_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    theses_opened INTEGER NOT NULL DEFAULT 0,
    theses_closed INTEGER NOT NULL DEFAULT 0,
    positions_opened INTEGER NOT NULL DEFAULT 0,
    positions_closed INTEGER NOT NULL DEFAULT 0,
    preemptions INTEGER NOT NULL DEFAULT 0,
    events_refreshed INTEGER NOT NULL DEFAULT 0,
    llm_input_tokens INTEGER NOT NULL DEFAULT 0,
    llm_output_tokens INTEGER NOT NULL DEFAULT 0,
    llm_cost_usd REAL,
    dry_run INTEGER NOT NULL DEFAULT 0,
    summary_json TEXT NOT NULL DEFAULT '{}',
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_factory_runs_started ON factory_runs(started_at);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    hypothesis TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    minimum_n_per_arm INTEGER NOT NULL,
    stopping_rule TEXT,
    minimum_calendar_days INTEGER NOT NULL DEFAULT 14,
    frozen_config_sha TEXT NOT NULL,
    frozen_config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finalized_at TEXT,
    verdict_json TEXT,
    status TEXT DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

CREATE TABLE IF NOT EXISTS delistings (
    symbol TEXT PRIMARY KEY,
    delisted_on TEXT NOT NULL,
    last_close REAL,
    source TEXT DEFAULT 'fmp',
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_delistings_delisted_on ON delistings(delisted_on);

CREATE TABLE IF NOT EXISTS shadow_opens (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    symbol TEXT NOT NULL,
    would_be_entry_price REAL,
    conviction_delta REAL NOT NULL,
    primary_side TEXT,
    premise_tags TEXT DEFAULT '[]',
    premise_direction TEXT DEFAULT '{}',
    regime_snapshot TEXT DEFAULT '{}',
    reason TEXT NOT NULL,
    orchestrator_label TEXT DEFAULT 'llm',
    experiment_id TEXT,
    discovery_source TEXT,
    horizon_days INTEGER,
    forward_return_30d REAL,
    forward_return_60d REAL,
    backfilled_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_opens_created ON shadow_opens(created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_opens_experiment ON shadow_opens(experiment_id);
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
        # Add regime-aware thesis fields (added in v0.8)
        ("theses", "premise_tags", "ALTER TABLE theses ADD COLUMN premise_tags TEXT DEFAULT '[]'"),
        ("theses", "regime_snapshot", "ALTER TABLE theses ADD COLUMN regime_snapshot TEXT DEFAULT '{}'"),
        # Add cohort fields to theses for policy-neutral pairing (added in v0.8)
        ("theses", "cohort", "ALTER TABLE theses ADD COLUMN cohort TEXT DEFAULT 'directional'"),
        ("theses", "hedge_symbol", "ALTER TABLE theses ADD COLUMN hedge_symbol TEXT"),
        ("theses", "paired_thesis_id", "ALTER TABLE theses ADD COLUMN paired_thesis_id TEXT"),
        # Add discovery_source to label which universe/screener surfaced the symbol (added in v0.9)
        ("theses", "discovery_source", "ALTER TABLE theses ADD COLUMN discovery_source TEXT"),
        # Cost-aware position accounting (added in v0.10)
        ("positions", "costs_applied_usd", "ALTER TABLE positions ADD COLUMN costs_applied_usd REAL DEFAULT 0.0"),
        ("positions", "realized_exit_price", "ALTER TABLE positions ADD COLUMN realized_exit_price REAL"),
        ("positions", "sizing_method", "ALTER TABLE positions ADD COLUMN sizing_method TEXT"),
        ("positions", "borrow_rate_pa_pct", "ALTER TABLE positions ADD COLUMN borrow_rate_pa_pct REAL"),
        # Per-tag polarity (Layer 2 #1) — JSON-serialized dict.
        ("theses", "premise_direction", "ALTER TABLE theses ADD COLUMN premise_direction TEXT DEFAULT '{}'"),
        # Calibrated P(correct) at thesis-open time (Layer 2 #3).
        ("theses", "calibrated_p_correct", "ALTER TABLE theses ADD COLUMN calibrated_p_correct REAL"),
        # Calibration model vintage (Bug E, r3) — ISO fit_at of the model used.
        ("theses", "calibration_fit_at", "ALTER TABLE theses ADD COLUMN calibration_fit_at TEXT"),
        # Orchestrator label (research experiment) — "llm" default, "random" for control arm.
        ("theses", "orchestrator_label", "ALTER TABLE theses ADD COLUMN orchestrator_label TEXT DEFAULT 'llm'"),
        # Experiment registration (cents-hvz) — which active experiment a thesis was opened under.
        ("theses", "experiment_id", "ALTER TABLE theses ADD COLUMN experiment_id TEXT"),
        # Per-experiment calendar-day floor on verdict_ready (pilot uses 30, full uses 90).
        # Default 14 preserves back-compat with experiments registered before this column existed.
        ("experiments", "minimum_calendar_days", "ALTER TABLE experiments ADD COLUMN minimum_calendar_days INTEGER NOT NULL DEFAULT 14"),
        # Evidence provenance columns linking to the LLM call (added in v0.10).
        # Run BEFORE and AFTER the FK migration since that migration may recreate
        # the evidence table with the legacy column set.
        ("evidence", "llm_call_id", "ALTER TABLE evidence ADD COLUMN llm_call_id TEXT"),
        ("evidence", "model_snapshot", "ALTER TABLE evidence ADD COLUMN model_snapshot TEXT"),
        ("evidence", "prompt_sha256", "ALTER TABLE evidence ADD COLUMN prompt_sha256 TEXT"),
        ("evidence", "input_sha256", "ALTER TABLE evidence ADD COLUMN input_sha256 TEXT"),
        ("evidence", "output_sha256", "ALTER TABLE evidence ADD COLUMN output_sha256 TEXT"),
        # Event tag_status distinguishes "tagger ran, no relevance" from
        # "tagger failed" — previously both produced tags=[] and the
        # no-thesis research path silently suppressed failures as if they
        # were genuinely irrelevant. Default 'tagger_skipped' is the safest
        # back-compat for rows that predate the column.
        ("events", "tag_status", "ALTER TABLE events ADD COLUMN tag_status TEXT DEFAULT 'tagger_skipped'"),
    ]

    def _apply_column_migrations() -> None:
        for table, column, sql in column_migrations:
            # Skip missing tables — some test fixtures create partial schemas.
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if not columns:
                continue
            if column not in columns:
                conn.execute(sql)

    _apply_column_migrations()

    # Migrate foreign keys to include ON DELETE actions (added in v0.6)
    _migrate_foreign_keys(conn)

    # Re-apply column adds — the FK migration may have recreated tables
    # without the v0.10 columns.
    _apply_column_migrations()

    # Idempotent CREATE TABLE for tables added after the FK migration was
    # established. New tables go here rather than in column_migrations.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delistings (
            symbol TEXT PRIMARY KEY,
            delisted_on TEXT NOT NULL,
            last_close REAL,
            source TEXT DEFAULT 'fmp',
            ingested_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_delistings_delisted_on ON delistings(delisted_on)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_opens (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            symbol TEXT NOT NULL,
            would_be_entry_price REAL,
            conviction_delta REAL NOT NULL,
            primary_side TEXT,
            premise_tags TEXT DEFAULT '[]',
            premise_direction TEXT DEFAULT '{}',
            regime_snapshot TEXT DEFAULT '{}',
            reason TEXT NOT NULL,
            orchestrator_label TEXT DEFAULT 'llm',
            experiment_id TEXT,
            discovery_source TEXT,
            horizon_days INTEGER,
            forward_return_30d REAL,
            forward_return_60d REAL,
            backfilled_at TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_opens_created ON shadow_opens(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_opens_experiment ON shadow_opens(experiment_id)")

    # AlertRepository.find_invalidation_for filters by alert_type and created_at
    # window; without this composite index it scans the full alerts table and
    # runs json_extract on every row. Some test fixtures create partial schemas
    # without alerts, so guard the table existence.
    has_alerts = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='alerts'"
    ).fetchone()
    if has_alerts:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_type_created "
            "ON alerts(alert_type, created_at DESC)"
        )


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
        # Migrate evidence table (ON DELETE SET NULL, nullable thesis_id, add
        # symbol + v0.10 provenance columns). The CREATE includes the v0.10
        # provenance columns so any data already populated in evidence_old for
        # those columns is preserved by the dynamic-column INSERT below; the
        # previous static 10-column INSERT silently dropped them.
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
                llm_call_id TEXT,
                model_snapshot TEXT,
                prompt_sha256 TEXT,
                input_sha256 TEXT,
                output_sha256 TEXT,
                FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
            )
        """)
        # Introspect evidence_old's column set so any v0.10 provenance columns
        # added by earlier column-migrations carry over. Mirrors the positions
        # migration pattern above.
        cursor = conn.execute("PRAGMA table_info(evidence_old)")
        old_cols = [row[1] for row in cursor.fetchall()]
        col_csv = ", ".join(old_cols)
        conn.execute(f"INSERT INTO evidence ({col_csv}) SELECT {col_csv} FROM evidence_old")
        conn.execute("DROP TABLE evidence_old")

        # Migrate positions table (ON DELETE SET NULL)
        # Column list includes v0.10 cost-aware accounting fields so a DB that
        # picked those up via column_migrations before reaching here still copies cleanly.
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
                costs_applied_usd REAL DEFAULT 0.0,
                realized_exit_price REAL,
                sizing_method TEXT,
                borrow_rate_pa_pct REAL,
                FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
            )
        """)
        # positions_old may or may not have the v0.10 columns depending on whether
        # column_migrations ran first. Pull the present column set and insert by name.
        cursor = conn.execute("PRAGMA table_info(positions_old)")
        old_cols = [row[1] for row in cursor.fetchall()]
        col_csv = ", ".join(old_cols)
        conn.execute(f"INSERT INTO positions ({col_csv}) SELECT {col_csv} FROM positions_old")
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

        # Recreate indexes that were dropped with the tables — must mirror
        # every CREATE INDEX in the SCHEMA constant for the affected tables,
        # or DBs that went through this migration permanently lose them.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_thesis ON evidence(thesis_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evidence_thesis_timestamp "
            "ON evidence(thesis_id, timestamp)"
        )
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
