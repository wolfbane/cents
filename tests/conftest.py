"""Shared test fixtures."""

import sqlite3
import pytest

from cents.db.schema import init_db, close_connection, SCHEMA


@pytest.fixture(autouse=True)
def reset_db_connection():
    """Reset the singleton database connection before each test.

    This ensures tests don't share database state and the singleton
    doesn't interfere with tests using custom in-memory connections.
    """
    close_connection()
    yield
    close_connection()


@pytest.fixture
def db_conn():
    """In-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    yield conn
    conn.close()
