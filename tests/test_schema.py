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
            "idx_backtests_symbol",
            "idx_backtest_signals_backtest",
            "idx_api_cache_lookup",
            "idx_events_occurred",
            "idx_events_source",
            "idx_llm_usage_called_at",
            "idx_llm_usage_agent",
            "idx_universes_default",
            "idx_factory_runs_started",
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
