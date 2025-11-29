"""Shared test fixtures."""

import sqlite3
import pytest

from cents.db.schema import init_db


@pytest.fixture
def db_conn():
    """In-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Initialize schema
    from cents.db.schema import SCHEMA
    conn.executescript(SCHEMA)
    conn.commit()
    yield conn
    conn.close()
