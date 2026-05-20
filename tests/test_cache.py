"""Tests for the api_cache TTL + prune behavior (cents-ame follow-up)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from cents.cache import APICache, TTL_DAYS_BY_ENDPOINT, _is_expired, _ttl_for
from cents.db.schema import SCHEMA


@pytest.fixture
def cache_conn(tmp_path):
    """Fresh in-memory-ish SQLite with just the api_cache table."""
    db_path = tmp_path / "cache_test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _insert(conn, provider, endpoint, key, data="{}", cached_at=None):
    cached_at = cached_at or datetime.now().isoformat()
    conn.execute(
        "INSERT INTO api_cache (id, provider, endpoint, cache_key, response_data, cached_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"id-{key}", provider, endpoint, key, data, cached_at),
    )
    conn.commit()


class TestTTLPolicy:
    """Policy table sanity + lazy expiry on read."""

    def test_known_endpoints_have_sensible_ttls(self):
        assert _ttl_for("alpaca", "bars_split_v1") == 7
        assert _ttl_for("fmp", "ratios-ttm") == 7
        assert _ttl_for("fmp", "ratios") == 90
        assert _ttl_for("fred", "observations") == 365
        # Dead namespace is immediately-stale (TTL=0)
        assert _ttl_for("alpaca", "bars") == 0
        # Unknown endpoints are not policy'd — never expire
        assert _ttl_for("fmp", "made-up-endpoint") is None

    def test_is_expired_threshold(self):
        # ttl=None → never expire
        assert _is_expired((datetime.now() - timedelta(days=10_000)).isoformat(), None) is False
        # ttl=0 → always expired
        assert _is_expired(datetime.now().isoformat(), 0) is True
        # ttl=N → expired iff older than N days
        ts_old = (datetime.now() - timedelta(days=8)).isoformat()
        ts_fresh = (datetime.now() - timedelta(days=2)).isoformat()
        assert _is_expired(ts_old, 7) is True
        assert _is_expired(ts_fresh, 7) is False

    def test_get_returns_miss_for_expired_row_and_deletes_it(self, cache_conn):
        # 9 days old, TTL=7 → should be treated as miss + deleted
        old_ts = (datetime.now() - timedelta(days=9)).isoformat()
        _insert(cache_conn, "alpaca", "bars_split_v1", "abc", data='"stale"', cached_at=old_ts)
        cache = APICache(conn=cache_conn)

        # Trick: use the same param set the row was inserted with — but get()
        # computes the key from params, so we need to insert with the right key.
        # Reach in and replace the row with a deterministic key.
        cache_conn.execute("DELETE FROM api_cache")
        params = {"symbol": "INTC", "_day": "2026-05-01"}
        key = cache._make_cache_key(params)
        _insert(cache_conn, "alpaca", "bars_split_v1", key, data='"stale"', cached_at=old_ts)

        # Confirm row exists
        assert cache_conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0] == 1

        # Read should miss + delete
        result = cache.get("alpaca", "bars_split_v1", params)
        assert result is None
        assert cache_conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0] == 0

    def test_get_returns_value_for_fresh_row(self, cache_conn):
        cache = APICache(conn=cache_conn)
        params = {"symbol": "INTC", "_day": "2026-05-19"}
        key = cache._make_cache_key(params)
        # Insert a fresh row
        _insert(cache_conn, "alpaca", "bars_split_v1", key, data='"fresh"', cached_at=datetime.now().isoformat())

        result = cache.get("alpaca", "bars_split_v1", params)
        assert result == "fresh"

    def test_get_no_policy_endpoint_never_expires(self, cache_conn):
        cache = APICache(conn=cache_conn)
        params = {"symbol": "ANYTHING"}
        key = cache._make_cache_key(params)
        # Insert an ancient row for an endpoint with no TTL policy
        ancient = (datetime.now() - timedelta(days=10_000)).isoformat()
        _insert(cache_conn, "fmp", "no-policy-here", key, data='"still-here"', cached_at=ancient)
        result = cache.get("fmp", "no-policy-here", params)
        assert result == "still-here"


class TestPrune:
    def test_prune_deletes_dead_namespace(self, cache_conn):
        _insert(cache_conn, "alpaca", "bars", "k1", cached_at=datetime.now().isoformat())
        _insert(cache_conn, "alpaca", "bars", "k2", cached_at=datetime.now().isoformat())
        _insert(cache_conn, "alpaca", "bars_split_v1", "k3", cached_at=datetime.now().isoformat())

        cache = APICache(conn=cache_conn)
        deleted = cache.prune()

        assert deleted.get(("alpaca", "bars"), 0) == 2
        # bars_split_v1 row is fresh — shouldn't be touched
        assert ("alpaca", "bars_split_v1") not in deleted
        remaining = cache_conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
        assert remaining == 1

    def test_prune_respects_ttl_per_endpoint(self, cache_conn):
        # 90-day-policy endpoint with 30-day-old row → keep (under TTL)
        ts_30d = (datetime.now() - timedelta(days=30)).isoformat()
        _insert(cache_conn, "fmp", "ratios", "k1", cached_at=ts_30d)
        # 90-day-policy endpoint with 100-day-old row → delete
        ts_100d = (datetime.now() - timedelta(days=100)).isoformat()
        _insert(cache_conn, "fmp", "ratios", "k2", cached_at=ts_100d)
        # 7-day endpoint with 10-day-old row → delete
        ts_10d = (datetime.now() - timedelta(days=10)).isoformat()
        _insert(cache_conn, "fmp", "ratios-ttm", "k3", cached_at=ts_10d)
        # No-policy endpoint with 1000-day-old row → keep (no TTL configured)
        ts_1000d = (datetime.now() - timedelta(days=1000)).isoformat()
        _insert(cache_conn, "fmp", "no-policy", "k4", cached_at=ts_1000d)

        cache = APICache(conn=cache_conn)
        deleted = cache.prune()

        assert deleted.get(("fmp", "ratios"), 0) == 1
        assert deleted.get(("fmp", "ratios-ttm"), 0) == 1
        assert ("fmp", "no-policy") not in deleted

        remaining_endpoints = {
            (r["provider"], r["endpoint"])
            for r in cache_conn.execute("SELECT provider, endpoint FROM api_cache").fetchall()
        }
        assert ("fmp", "ratios") in remaining_endpoints  # the 30-day-old one
        assert ("fmp", "no-policy") in remaining_endpoints
        assert ("fmp", "ratios-ttm") not in remaining_endpoints


class TestDetailedStats:
    def test_detailed_stats_groups_and_attaches_ttl(self, cache_conn):
        _insert(cache_conn, "alpaca", "bars_split_v1", "k1", data="x" * 1000)
        _insert(cache_conn, "alpaca", "bars_split_v1", "k2", data="x" * 2000)
        _insert(cache_conn, "fmp", "ratios", "k3", data="x" * 500)

        cache = APICache(conn=cache_conn)
        rows = cache.detailed_stats()
        by_key = {(r["provider"], r["endpoint"]): r for r in rows}
        assert by_key[("alpaca", "bars_split_v1")]["rows"] == 2
        assert by_key[("alpaca", "bars_split_v1")]["bytes"] >= 3000
        assert by_key[("alpaca", "bars_split_v1")]["ttl_days"] == 7
        assert by_key[("fmp", "ratios")]["ttl_days"] == 90

    def test_detailed_stats_sorted_by_size_desc(self, cache_conn):
        _insert(cache_conn, "fmp", "ratios", "k1", data="x" * 500)
        _insert(cache_conn, "alpaca", "bars_split_v1", "k2", data="x" * 2000)
        cache = APICache(conn=cache_conn)
        rows = cache.detailed_stats()
        # Bigger-byte endpoint should come first
        assert rows[0]["endpoint"] == "bars_split_v1"
        assert rows[1]["endpoint"] == "ratios"


class TestDeadNamespaceWriteBlock:
    """cents-lqqw: writes to TTL=0 (dead) namespaces are refused."""

    def test_write_to_dead_namespace_is_refused(self, cache_conn):
        """A TTL=0 namespace = no-op write + warning. Verify no row created."""
        cache = APICache(conn=cache_conn)
        cache.set("alpaca", "bars", {"symbol": "X"}, {"any": "data"})
        rows = cache_conn.execute(
            "SELECT COUNT(*) FROM api_cache WHERE provider='alpaca' AND endpoint='bars'"
        ).fetchone()[0]
        assert rows == 0, "Dead-namespace write should be refused"

    def test_write_to_live_namespace_still_works(self, cache_conn):
        """Sanity: TTL>0 namespaces still accept writes."""
        cache = APICache(conn=cache_conn)
        cache.set("alpaca", "bars_split_v1", {"symbol": "X"}, {"bars": [1, 2, 3]})
        rows = cache_conn.execute(
            "SELECT COUNT(*) FROM api_cache WHERE provider='alpaca' AND endpoint='bars_split_v1'"
        ).fetchone()[0]
        assert rows == 1
