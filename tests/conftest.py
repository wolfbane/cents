"""Shared test fixtures."""

import sqlite3
import pytest

from cents.db.schema import init_db, close_connection, SCHEMA
from cents.llm_usage import reset_cost_cap_state


def pytest_addoption(parser):
    """Add custom command-line flags.

    ``--runlookahead`` opts into the lookahead-leak audit (cents-ekd),
    which hits the live Anthropic API. See ``tests/test_lookahead_audit.py``.
    """
    parser.addoption(
        "--runlookahead",
        action="store_true",
        default=False,
        help="Run the live lookahead-leak audit (requires ANTHROPIC_API_KEY).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lookahead_audit: lookahead-leak audit on the sentiment agent (cents-ekd).",
    )


@pytest.fixture(autouse=True)
def reset_db_connection():
    """Reset the singleton database connection before each test.

    This ensures tests don't share database state and the singleton
    doesn't interfere with tests using custom in-memory connections.
    """
    close_connection()
    yield
    close_connection()


@pytest.fixture(autouse=True)
def reset_llm_cost_cap():
    """Clear any per-run LLM cost cap set by a prior test."""
    reset_cost_cap_state()
    yield
    reset_cost_cap_state()


@pytest.fixture(autouse=True)
def isolate_api_cache(tmp_path, monkeypatch):
    """Point every test at a throwaway DB so cached external responses
    (FMP/Alpaca/etc.) from prior runs can't leak across tests."""
    monkeypatch.setenv("CENTS_DB_PATH", str(tmp_path / "test.db"))
    # Also isolate the LLM blob store so persist_call_blob doesn't write
    # under the developer's real ~/.cents/data tree during tests.
    monkeypatch.setenv("CENTS_LLM_BLOB_DIR", str(tmp_path / "llm_calls"))
    # Clear the daily cap env var so leaked CI/dev settings don't trip caps
    # in tests that don't explicitly opt in.
    monkeypatch.delenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", raising=False)


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
