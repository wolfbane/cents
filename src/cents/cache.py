"""API response caching for historical data.

Historical data (fundamentals, ratios, FRED observations) doesn't change,
so we cache it to avoid redundant API calls and rate limits.
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from cents.db.schema import get_connection

logger = logging.getLogger(__name__)

# Allow disabling cache via environment variable (useful for tests)
CACHE_DISABLED = os.environ.get("CENTS_DISABLE_CACHE", "").lower() in ("1", "true", "yes")


class APICache:
    """Cache for API responses stored in SQLite."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()
        self._table_exists = self._check_table_exists()

    def _check_table_exists(self) -> bool:
        """Check if api_cache table exists."""
        try:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='api_cache'"
            )
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _make_cache_key(self, params: dict[str, Any]) -> str:
        """Create a stable cache key from parameters."""
        # Sort keys for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]

    def get(
        self, provider: str, endpoint: str, params: dict[str, Any]
    ) -> Any | None:
        """Get cached response if available.

        Args:
            provider: API provider name (e.g., "fmp", "fred", "alpaca")
            endpoint: API endpoint (e.g., "ratios", "observations")
            params: Request parameters used to generate cache key

        Returns:
            Cached response data or None if not cached
        """
        if CACHE_DISABLED or not self._table_exists:
            return None

        cache_key = self._make_cache_key(params)
        try:
            row = self.conn.execute(
                """
                SELECT response_data FROM api_cache
                WHERE provider = ? AND endpoint = ? AND cache_key = ?
                """,
                (provider, endpoint, cache_key),
            ).fetchone()

            if row:
                logger.debug("Cache hit: %s/%s %s", provider, endpoint, cache_key[:8])
                return json.loads(row[0])

            logger.debug("Cache miss: %s/%s %s", provider, endpoint, cache_key[:8])
        except Exception as e:
            # Handle closed connections or missing tables gracefully
            logger.debug("Cache lookup failed: %s", e)
        return None

    def set(
        self, provider: str, endpoint: str, params: dict[str, Any], data: Any
    ) -> None:
        """Store response in cache.

        Args:
            provider: API provider name
            endpoint: API endpoint
            params: Request parameters used to generate cache key
            data: Response data to cache (must be JSON-serializable)
        """
        if CACHE_DISABLED or not self._table_exists:
            return

        cache_key = self._make_cache_key(params)
        now = datetime.now().isoformat()

        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO api_cache
                (id, provider, endpoint, cache_key, response_data, cached_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4())[:8], provider, endpoint, cache_key, json.dumps(data), now),
            )
            self.conn.commit()
            logger.debug("Cached: %s/%s %s", provider, endpoint, cache_key[:8])
        except Exception as e:
            # Handle closed connections or missing tables gracefully
            logger.debug("Cache write failed: %s", e)

    def clear(self, provider: str | None = None) -> int:
        """Clear cache entries.

        Args:
            provider: If specified, only clear entries for this provider

        Returns:
            Number of entries cleared
        """
        if provider:
            cursor = self.conn.execute(
                "DELETE FROM api_cache WHERE provider = ?", (provider,)
            )
        else:
            cursor = self.conn.execute("DELETE FROM api_cache")
        self.conn.commit()
        return cursor.rowcount

    def stats(self) -> dict[str, int]:
        """Get cache statistics by provider."""
        rows = self.conn.execute(
            """
            SELECT provider, COUNT(*) as count
            FROM api_cache
            GROUP BY provider
            """
        ).fetchall()
        return {row[0]: row[1] for row in rows}


def get_cache(conn: sqlite3.Connection | None = None) -> APICache:
    """Get a cache instance using the provided or default connection."""
    return APICache(conn)


def cached_request(
    provider: str,
    endpoint: str,
    params: dict[str, Any],
    fetch_fn: Callable[[], Any],
    conn: sqlite3.Connection | None = None,
) -> Any:
    """Execute a request with caching.

    Args:
        provider: API provider name (e.g., "fmp", "fred")
        endpoint: API endpoint
        params: Request parameters (used as cache key)
        fetch_fn: Function to call if cache miss
        conn: Optional database connection

    Returns:
        Response data (from cache or fresh fetch)
    """
    cache = get_cache(conn)

    # Check cache first
    cached = cache.get(provider, endpoint, params)
    if cached is not None:
        return cached

    # Fetch fresh data
    data = fetch_fn()

    # Cache the response
    if data is not None:
        cache.set(provider, endpoint, params, data)

    return data
