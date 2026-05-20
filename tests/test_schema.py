"""Tests for database schema module."""

import sqlite3
from pathlib import Path

import pytest

from cents.db.schema import (
    init_db,
    get_db_path,
    _migrate_schema,
    _migrate_foreign_keys,
    SCHEMA,
)


class TestGetDbPath:
    """Tests for get_db_path."""

    def test_returns_path_in_home_directory(self, tmp_path, monkeypatch):
        """Returns path in ~/.cents/data/ by default."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CENTS_DB_PATH", raising=False)
        result = get_db_path()
        assert result == tmp_path / ".cents" / "data" / "cents.db"

    def test_creates_data_directory(self, tmp_path, monkeypatch):
        """Creates ~/.cents/data/ directory if it doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CENTS_DB_PATH", raising=False)
        assert not (tmp_path / ".cents" / "data").exists()
        get_db_path()
        assert (tmp_path / ".cents" / "data").exists()

    def test_respects_env_var_override(self, tmp_path, monkeypatch):
        """CENTS_DB_PATH env var overrides default path."""
        custom_path = tmp_path / "custom" / "my.db"
        monkeypatch.setenv("CENTS_DB_PATH", str(custom_path))
        result = get_db_path()
        assert result == custom_path
        assert custom_path.parent.exists()

    def test_env_var_creates_parent_directories(self, tmp_path, monkeypatch):
        """CENTS_DB_PATH creates parent directories if needed."""
        custom_path = tmp_path / "deep" / "nested" / "path" / "my.db"
        monkeypatch.setenv("CENTS_DB_PATH", str(custom_path))
        assert not custom_path.parent.exists()
        get_db_path()
        assert custom_path.parent.exists()


class TestInitDb:
    """Tests for init_db."""

    def test_creates_all_tables(self, tmp_path):
        """All expected tables are created."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "theses", "evidence", "positions", "outcomes", "watchlist", "alerts",
            "universes", "factory_runs",
        }
        assert expected_tables.issubset(tables)
        conn.close()

    def test_creates_all_indexes(self, tmp_path):
        """All expected indexes are created."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            "idx_theses_status",
            "idx_positions_status",
            "idx_evidence_thesis",
            "idx_evidence_thesis_timestamp",
            "idx_watchlist_symbol",
            "idx_alerts_read",
            "idx_alerts_type_created",
            "idx_backtests_symbol",
            "idx_backtest_signals_backtest",
            "idx_api_cache_lookup",
            "idx_events_occurred",
            "idx_events_source",
            "idx_llm_usage_called_at",
            "idx_llm_usage_agent",
            "idx_universes_default",
            "idx_factory_runs_started",
            "idx_experiments_status",
            "idx_delistings_delisted_on",
            "idx_shadow_opens_created",
            "idx_shadow_opens_experiment",
        }
        assert expected_indexes == indexes
        conn.close()

    def test_row_factory_is_set(self, tmp_path):
        """Connection has Row factory set."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_idempotent(self, tmp_path):
        """Calling init_db multiple times is safe."""
        db_path = tmp_path / "test.db"
        conn1 = init_db(db_path)
        conn1.close()

        # Should not raise
        conn2 = init_db(db_path)
        cursor = conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = cursor.fetchall()
        assert len(tables) > 0
        conn2.close()


class TestMigrateSchema:
    """Tests for _migrate_schema."""

    def test_adds_threshold_column_to_watchlist(self, tmp_path):
        """Migration adds threshold column to watchlist table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create minimal schema with old watchlist (missing threshold columns)
        conn.executescript("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT,
                symbol TEXT, business_quality TEXT, valuation TEXT, moat TEXT,
                time_horizon TEXT, horizon_end TEXT, key_risks TEXT,
                target_price REAL, stop_price REAL, outcome TEXT, closed_at TEXT
            );
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY, thesis_id TEXT, agent TEXT, type TEXT,
                content TEXT, source TEXT, confidence REAL, metadata TEXT,
                timestamp TEXT, dimension TEXT
            );
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                notes TEXT DEFAULT '',
                thesis_id TEXT,
                last_scanned TEXT,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()

        # Run migration
        _migrate_schema(conn)

        # Verify column was added
        cursor = conn.execute("PRAGMA table_info(watchlist)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "threshold" in columns
        assert "alert_destination" in columns
        conn.close()

    def test_adds_structured_fields_to_theses(self, tmp_path):
        """Migration adds structured thesis fields."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create schema with old theses (missing structured fields)
        conn.executescript("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                hypothesis TEXT DEFAULT '',
                status TEXT DEFAULT 'open',
                conviction REAL DEFAULT 50.0,
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY, thesis_id TEXT, agent TEXT, type TEXT,
                content TEXT, source TEXT, confidence REAL, metadata TEXT,
                timestamp TEXT, dimension TEXT
            );
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY, symbol TEXT UNIQUE, notes TEXT,
                thesis_id TEXT, threshold REAL, alert_destination TEXT,
                last_scanned TEXT, created_at TEXT
            );
        """)
        conn.commit()

        # Run migration
        _migrate_schema(conn)

        # Verify columns were added
        cursor = conn.execute("PRAGMA table_info(theses)")
        columns = [row[1] for row in cursor.fetchall()]

        expected_new_columns = [
            "symbol", "business_quality", "valuation", "moat",
            "time_horizon", "horizon_end", "key_risks",
            "target_price", "stop_price", "outcome", "closed_at"
        ]
        for col in expected_new_columns:
            assert col in columns, f"Column {col} not found"
        conn.close()

    def test_adds_dimension_to_evidence(self, tmp_path):
        """Migration adds dimension column to evidence table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create schema with old evidence (missing dimension column)
        conn.executescript("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT,
                symbol TEXT, business_quality TEXT, valuation TEXT, moat TEXT,
                time_horizon TEXT, horizon_end TEXT, key_risks TEXT,
                target_price REAL, stop_price REAL, outcome TEXT, closed_at TEXT
            );
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY,
                thesis_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                type TEXT DEFAULT 'neutral',
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                metadata TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL
            );
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY, symbol TEXT UNIQUE, notes TEXT,
                thesis_id TEXT, threshold REAL, alert_destination TEXT,
                last_scanned TEXT, created_at TEXT
            );
        """)
        conn.commit()

        # Run migration
        _migrate_schema(conn)

        # Verify column was added
        cursor = conn.execute("PRAGMA table_info(evidence)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "dimension" in columns
        conn.close()

    def test_migration_is_idempotent(self, tmp_path):
        """Running migrations twice doesn't cause errors."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create schema with executescript
        conn.executescript(SCHEMA)
        conn.commit()

        # Run migration multiple times - should not raise
        _migrate_schema(conn)
        _migrate_schema(conn)
        _migrate_schema(conn)

        # Verify schema is still valid
        cursor = conn.execute("PRAGMA table_info(theses)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "symbol" in columns
        conn.close()

    def test_preserves_existing_data(self, tmp_path):
        """Migration preserves existing data in tables."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create schema with old theses (missing structured fields)
        conn.executescript("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                hypothesis TEXT DEFAULT '',
                status TEXT DEFAULT 'open',
                conviction REAL DEFAULT 50.0,
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY, thesis_id TEXT, agent TEXT, type TEXT,
                content TEXT, source TEXT, confidence REAL, metadata TEXT,
                timestamp TEXT, dimension TEXT
            );
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY, symbol TEXT UNIQUE, notes TEXT,
                thesis_id TEXT, threshold REAL, alert_destination TEXT,
                last_scanned TEXT, created_at TEXT
            );
        """)

        # Insert test data
        conn.execute("""
            INSERT INTO theses (id, title, created_at, updated_at)
            VALUES ('test1', 'Test Thesis', '2024-01-01', '2024-01-01')
        """)
        conn.commit()

        # Run migration
        _migrate_schema(conn)

        # Verify data is preserved
        cursor = conn.execute("SELECT id, title FROM theses WHERE id = 'test1'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test1"
        assert row[1] == "Test Thesis"
        conn.close()

    def test_preserves_v010_evidence_provenance_through_fk_migration(self, tmp_path):
        """The FK migration's evidence table rebuild must preserve v0.10
        provenance columns (llm_call_id, model_snapshot, *_sha256). Previously
        the static 10-column INSERT silently dropped them; populated provenance
        data would survive _apply_column_migrations but vanish on the next FK
        migration pass.
        """
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Pre-FK-migration schema with evidence already carrying provenance
        # columns (simulating a DB that ran v0.10 column-adds before reaching
        # this migration). thesis_id is NOT NULL — the pre-FK shape.
        conn.executescript("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT,
                symbol TEXT, business_quality TEXT, valuation TEXT, moat TEXT,
                time_horizon TEXT, horizon_end TEXT, key_risks TEXT,
                target_price REAL, stop_price REAL, outcome TEXT, closed_at TEXT
            );
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY,
                thesis_id TEXT NOT NULL,
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
                output_sha256 TEXT
            );
            CREATE TABLE positions (
                id TEXT PRIMARY KEY, thesis_id TEXT NOT NULL, symbol TEXT,
                side TEXT, entry_price REAL, entry_date TEXT, size REAL,
                status TEXT, exit_price REAL, exit_date TEXT,
                paper INTEGER, notes TEXT, created_at TEXT
            );
            CREATE TABLE outcomes (
                id TEXT PRIMARY KEY, position_id TEXT NOT NULL, pnl REAL,
                pnl_pct REAL, thesis_accuracy TEXT, agent_performance TEXT,
                retrospective TEXT, recorded_at TEXT
            );
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY, symbol TEXT UNIQUE, notes TEXT,
                thesis_id TEXT, threshold REAL, alert_destination TEXT,
                last_scanned TEXT, created_at TEXT
            );
            INSERT INTO theses (id, title, created_at, updated_at)
                VALUES ('t1', 'T', '2024-01-01', '2024-01-01');
            INSERT INTO evidence (
                id, thesis_id, agent, type, content, source, confidence,
                dimension, metadata, timestamp,
                llm_call_id, model_snapshot, prompt_sha256, input_sha256, output_sha256
            ) VALUES (
                'e1', 't1', 'sentiment', 'supporting', 'body', 'newsapi', 0.7,
                'sentiment', '{}', '2024-01-01',
                'call-abc', 'claude-haiku-4-5-20251001',
                'p_hash', 'i_hash', 'o_hash'
            );
        """)
        conn.commit()

        _migrate_schema(conn)

        # Verify both schema (the columns exist post-migration) and data
        # (the row we inserted survived with provenance intact).
        cursor = conn.execute("PRAGMA table_info(evidence)")
        cols = {row[1] for row in cursor.fetchall()}
        assert {
            "llm_call_id", "model_snapshot",
            "prompt_sha256", "input_sha256", "output_sha256",
        } <= cols
        row = conn.execute(
            "SELECT llm_call_id, model_snapshot, prompt_sha256, "
            "input_sha256, output_sha256 FROM evidence WHERE id = 'e1'"
        ).fetchone()
        assert row == ("call-abc", "claude-haiku-4-5-20251001",
                       "p_hash", "i_hash", "o_hash")
        conn.close()


class TestSchemaIntegrity:
    """Tests for overall schema integrity."""

    def test_foreign_keys_are_defined(self, tmp_path):
        """Foreign key relationships are properly defined in schema."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Insert a thesis
        conn.execute("""
            INSERT INTO theses (id, title, created_at, updated_at)
            VALUES ('t1', 'Test', '2024-01-01', '2024-01-01')
        """)

        # Insert evidence referencing that thesis - should work
        conn.execute("""
            INSERT INTO evidence (id, thesis_id, agent, content, source, timestamp)
            VALUES ('e1', 't1', 'test', 'content', 'source', '2024-01-01')
        """)
        conn.commit()
        conn.close()

    def test_watchlist_symbol_unique(self, tmp_path):
        """Watchlist symbol has unique constraint."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)

        conn.execute("""
            INSERT INTO watchlist (id, symbol, created_at)
            VALUES ('w1', 'AAPL', '2024-01-01')
        """)
        conn.commit()

        # Duplicate symbol should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO watchlist (id, symbol, created_at)
                VALUES ('w2', 'AAPL', '2024-01-01')
            """)
        conn.close()


class TestMigrateForeignKeys:
    """Tests for _migrate_foreign_keys."""

    def _create_old_schema(self, conn: sqlite3.Connection) -> None:
        """Create schema without ON DELETE actions (pre-v0.6)."""
        conn.executescript("""
            CREATE TABLE theses (
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

            CREATE TABLE evidence (
                id TEXT PRIMARY KEY,
                thesis_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                type TEXT DEFAULT 'neutral',
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                dimension TEXT,
                metadata TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (thesis_id) REFERENCES theses(id)
            );

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
                FOREIGN KEY (thesis_id) REFERENCES theses(id)
            );

            CREATE TABLE outcomes (
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

            CREATE TABLE watchlist (
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

            CREATE TABLE alerts (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT DEFAULT '{}',
                read INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX idx_evidence_thesis ON evidence(thesis_id);
            CREATE INDEX idx_positions_status ON positions(status);
            CREATE INDEX idx_watchlist_symbol ON watchlist(symbol);
        """)
        conn.commit()

    def test_adds_set_null_to_evidence(self, tmp_path):
        """Migration adds ON DELETE SET NULL to evidence foreign key."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Run migration
        _migrate_foreign_keys(conn)

        # Verify SET NULL is set (evidence thesis_id is nullable)
        cursor = conn.execute("PRAGMA foreign_key_list(evidence)")
        fk_info = cursor.fetchall()
        assert len(fk_info) == 1
        assert fk_info[0][6] == "SET NULL"  # on_delete column
        conn.close()

    def test_adds_set_null_to_positions(self, tmp_path):
        """Migration adds ON DELETE SET NULL to positions foreign key."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Run migration
        _migrate_foreign_keys(conn)

        # Verify SET NULL is set
        cursor = conn.execute("PRAGMA foreign_key_list(positions)")
        fk_info = cursor.fetchall()
        assert len(fk_info) == 1
        assert fk_info[0][6] == "SET NULL"  # on_delete column
        conn.close()

    def test_adds_cascade_to_outcomes(self, tmp_path):
        """Migration adds ON DELETE CASCADE to outcomes foreign key."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Run migration
        _migrate_foreign_keys(conn)

        # Verify CASCADE is set
        cursor = conn.execute("PRAGMA foreign_key_list(outcomes)")
        fk_info = cursor.fetchall()
        assert len(fk_info) == 1
        assert fk_info[0][6] == "CASCADE"  # on_delete column
        conn.close()

    def test_adds_set_null_to_watchlist(self, tmp_path):
        """Migration adds ON DELETE SET NULL to watchlist foreign key."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Run migration
        _migrate_foreign_keys(conn)

        # Verify SET NULL is set
        cursor = conn.execute("PRAGMA foreign_key_list(watchlist)")
        fk_info = cursor.fetchall()
        assert len(fk_info) == 1
        assert fk_info[0][6] == "SET NULL"  # on_delete column
        conn.close()

    def test_preserves_existing_data(self, tmp_path):
        """Migration preserves all existing data."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Insert test data
        conn.execute("""
            INSERT INTO theses (id, title, created_at, updated_at)
            VALUES ('t1', 'Test Thesis', '2024-01-01', '2024-01-01')
        """)
        conn.execute("""
            INSERT INTO evidence (id, thesis_id, agent, content, source, timestamp)
            VALUES ('e1', 't1', 'test', 'Test content', 'test', '2024-01-01')
        """)
        conn.execute("""
            INSERT INTO positions (id, thesis_id, symbol, side, entry_price, entry_date, size, created_at)
            VALUES ('p1', 't1', 'AAPL', 'long', 100, '2024-01-01', 10, '2024-01-01')
        """)
        conn.execute("""
            INSERT INTO outcomes (id, position_id, pnl, pnl_pct, recorded_at)
            VALUES ('o1', 'p1', 100, 10.0, '2024-01-01')
        """)
        conn.execute("""
            INSERT INTO watchlist (id, symbol, thesis_id, created_at)
            VALUES ('w1', 'GOOG', 't1', '2024-01-01')
        """)
        conn.commit()

        # Run migration
        _migrate_foreign_keys(conn)

        # Verify data preserved
        cursor = conn.execute("SELECT id, title FROM theses")
        assert cursor.fetchone() == ('t1', 'Test Thesis')

        cursor = conn.execute("SELECT id, thesis_id, content FROM evidence")
        row = cursor.fetchone()
        assert row == ('e1', 't1', 'Test content')

        cursor = conn.execute("SELECT id, thesis_id, symbol FROM positions")
        row = cursor.fetchone()
        assert row == ('p1', 't1', 'AAPL')

        cursor = conn.execute("SELECT id, position_id, pnl FROM outcomes")
        row = cursor.fetchone()
        assert row == ('o1', 'p1', 100)

        cursor = conn.execute("SELECT id, symbol, thesis_id FROM watchlist")
        row = cursor.fetchone()
        assert row == ('w1', 'GOOG', 't1')

        conn.close()

    def test_is_idempotent(self, tmp_path):
        """Running migration multiple times is safe."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Run migration multiple times
        _migrate_foreign_keys(conn)
        _migrate_foreign_keys(conn)
        _migrate_foreign_keys(conn)

        # Should still have correct FK settings
        cursor = conn.execute("PRAGMA foreign_key_list(evidence)")
        fk_info = cursor.fetchall()
        assert len(fk_info) == 1
        assert fk_info[0][6] == "SET NULL"
        conn.close()

    def test_skips_if_tables_missing(self, tmp_path):
        """Migration skips if required tables don't exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create only theses table
        conn.execute("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

        # Should not raise
        _migrate_foreign_keys(conn)
        conn.close()

    def test_recreates_indexes_after_migration(self, tmp_path):
        """Migration recreates indexes that were dropped with tables."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        self._create_old_schema(conn)

        # Run migration
        _migrate_foreign_keys(conn)

        # Check indexes exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_evidence_thesis" in indexes
        assert "idx_positions_status" in indexes
        assert "idx_watchlist_symbol" in indexes
        conn.close()


class TestMigrationPath:
    """cents-s047: schema migration must actually exercise _migrate_schema.

    Test fixtures use executescript(SCHEMA), which creates fresh tables with
    all columns. The _migrate_schema path was never tested, so adding a
    column there but forgetting to add it to SCHEMA (or vice versa) would
    silently pass tests and fail at upgrade time.
    """

    def test_migrate_schema_adds_every_declared_column(self, tmp_path):
        """Build a minimal-shape DB (legacy schema), run _migrate_schema, verify
        every column listed in column_migrations is present."""
        import sqlite3
        from cents.db.schema import _migrate_schema

        # Read the column_migrations list by introspecting the function so we
        # don't have to hard-code it here — keeps the test in sync.
        import inspect
        from cents.db import schema as schema_mod
        src = inspect.getsource(schema_mod._migrate_schema)

        # Parse the (table, column, sql) tuples by scanning for "ADD COLUMN".
        # Each migration is of the form ("table", "column", "ALTER TABLE ...").
        # Robust enough for the current shape.
        import re
        declared: list[tuple[str, str]] = []
        for m in re.finditer(
            r'\(\s*"([^"]+)",\s*"([^"]+)",\s*"ALTER TABLE',
            src,
        ):
            declared.append((m.group(1), m.group(2)))
        assert declared, "Failed to introspect column_migrations — test brittle"

        # Build a minimal "old-style" DB containing every table mentioned in
        # any migration, with just an id PK so the ALTER TABLE statements work.
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        tables_needed = sorted({t for t, _ in declared})
        for table in tables_needed:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY)")
        conn.commit()

        # Run the migration. Should add every declared column to its table.
        _migrate_schema(conn)
        conn.commit()

        # Verify each declared column is present
        for table, column in declared:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert column in cols, (
                f"_migrate_schema did NOT add {table}.{column} — "
                f"either the ALTER is broken or the column name in the migration "
                f"declaration doesn't match what's added."
            )
        conn.close()

    def test_premise_classification_source_round_trips(self, tmp_path):
        """cents-83xl: column lands on theses via migration AND round-trips a value.

        Without this, a Thesis stamped with premise_classification_source="llm"
        could silently revert to "fallback_empty" on read — exactly the
        stratification blocker the bead is trying to prevent.
        """
        import sqlite3
        from cents.db.schema import _migrate_schema

        # Start with a minimal pre-cents-83xl theses table (no source column).
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE theses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                hypothesis TEXT DEFAULT '',
                status TEXT DEFAULT 'open',
                conviction REAL DEFAULT 50.0,
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY, thesis_id TEXT, agent TEXT, type TEXT,
                content TEXT, source TEXT, confidence REAL, metadata TEXT,
                timestamp TEXT, dimension TEXT
            );
            CREATE TABLE watchlist (
                id TEXT PRIMARY KEY, symbol TEXT UNIQUE, notes TEXT,
                thesis_id TEXT, threshold REAL, alert_destination TEXT,
                last_scanned TEXT, created_at TEXT
            );
        """)
        conn.commit()

        _migrate_schema(conn)

        cols = {row[1] for row in conn.execute("PRAGMA table_info(theses)")}
        assert "premise_classification_source" in cols

        # Round-trip a non-default value.
        conn.execute(
            "INSERT INTO theses (id, title, created_at, updated_at, "
            "premise_classification_source) "
            "VALUES ('t1', 'T', '2024-01-01', '2024-01-01', 'fallback_sector')"
        )
        conn.commit()
        row = conn.execute(
            "SELECT premise_classification_source FROM theses WHERE id='t1'"
        ).fetchone()
        assert row[0] == "fallback_sector"

        # And legacy rows (no value) get the schema default.
        conn.execute(
            "INSERT INTO theses (id, title, created_at, updated_at) "
            "VALUES ('t2', 'T2', '2024-01-01', '2024-01-01')"
        )
        conn.commit()
        row = conn.execute(
            "SELECT premise_classification_source FROM theses WHERE id='t2'"
        ).fetchone()
        assert row[0] == "fallback_empty"
        conn.close()

    def test_migrate_schema_is_idempotent(self, tmp_path):
        """Running _migrate_schema twice must not fail (re-add) or change shape."""
        import sqlite3
        from cents.db.schema import _migrate_schema, SCHEMA

        db_path = tmp_path / "idempotent.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Start with the full schema (so all columns present)
        conn.executescript(SCHEMA)
        conn.commit()

        # First migration pass — should be a no-op (all columns already present)
        _migrate_schema(conn)
        cols_pass_1 = {
            t: sorted(r[1] for r in conn.execute(f"PRAGMA table_info({t})"))
            for t in ("theses", "evidence", "watchlist", "positions", "events", "experiments")
        }

        # Second pass — must not raise and must produce identical column sets
        _migrate_schema(conn)
        cols_pass_2 = {
            t: sorted(r[1] for r in conn.execute(f"PRAGMA table_info({t})"))
            for t in ("theses", "evidence", "watchlist", "positions", "events", "experiments")
        }

        assert cols_pass_1 == cols_pass_2
        conn.close()
