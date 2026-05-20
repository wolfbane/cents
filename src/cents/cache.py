"""API response caching for historical data.

Historical data (fundamentals, ratios, FRED observations) doesn't change,
so we cache it to avoid redundant API calls and rate limits.

TTL policy (cents-ame follow-up): daily-keyed endpoints get short TTLs so
their cache entries are recycled instead of growing unboundedly. Stable
endpoints (quarterly fundamentals, FRED) get longer TTLs because the data
is genuinely historical and immutable. Entries with no TTL configured are
never expired. Use `cents cache prune` to apply the policy retroactively.
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

from cents.db.schema import get_connection

logger = logging.getLogger(__name__)

# Allow disabling cache via environment variable (useful for tests)
CACHE_DISABLED = os.environ.get("CENTS_DISABLE_CACHE", "").lower() in ("1", "true", "yes")

# TTL policy in days, keyed by (provider, endpoint). Missing entries → never
# expire. Set to 0 to make a (provider, endpoint) effectively dead-on-read.
#
# Daily-keyed endpoints (those whose cache_params include `_day=today` or an
# `as_of` date) write a fresh row per calendar day; once today rolls over the
# row is never read again, so a short TTL recycles them aggressively.
#
# Stable-keyed endpoints (quarterly historicals, FRED observations) describe
# data that doesn't change after publication — they get long TTLs so the cache
# stays warm across runs, but old entries still age out instead of accumulating
# forever.
TTL_DAYS_BY_ENDPOINT: dict[tuple[str, str], int] = {
    # Dead namespace — pre-split-adjust bars superseded by `bars_split_v1`.
    # TTL=0 makes any leftover rows immediately stale on read + pruneable.
    ("alpaca", "bars"): 0,
    # Daily-keyed
    ("alpaca", "bars_split_v1"): 7,
    ("fmp", "ratios-ttm"): 7,
    ("fmp", "key-metrics-ttm"): 7,
    ("fmp", "profile"): 7,
    # Stable-keyed
    ("fmp", "ratios"): 90,
    ("fmp", "key-metrics"): 90,
    ("fmp", "insider-trading/search"): 30,
    ("fmp", "delisted-companies"): 30,
    ("fred", "observations"): 365,
}


def _ttl_for(provider: str, endpoint: str) -> int | None:
    """Return TTL in days for (provider, endpoint), or None for never-expire."""
    return TTL_DAYS_BY_ENDPOINT.get((provider, endpoint))


def _is_expired(cached_at: str, ttl_days: int | None) -> bool:
    """Check whether a row with the given cached_at iso-string is past its TTL."""
    if ttl_days is None:
        return False
    if ttl_days <= 0:
        return True
    try:
        ts = datetime.fromisoformat(cached_at)
    except (TypeError, ValueError):
        return False
    return datetime.now() - ts > timedelta(days=ttl_days)


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
                SELECT response_data, cached_at FROM api_cache
                WHERE provider = ? AND endpoint = ? AND cache_key = ?
                """,
                (provider, endpoint, cache_key),
            ).fetchone()

            if row:
                ttl_days = _ttl_for(provider, endpoint)
                if _is_expired(row[1], ttl_days):
                    # Lazy GC: drop the expired row + treat as miss so the
                    # caller re-fetches with fresh data.
                    self.conn.execute(
                        "DELETE FROM api_cache WHERE provider = ? AND endpoint = ? AND cache_key = ?",
                        (provider, endpoint, cache_key),
                    )
                    self.conn.commit()
                    logger.debug(
                        "Cache expired (TTL %sd): %s/%s %s",
                        ttl_days, provider, endpoint, cache_key[:8],
                    )
                    return None
                logger.debug("Cache hit: %s/%s %s", provider, endpoint, cache_key[:8])
                return json.loads(row[0])

            logger.debug("Cache miss: %s/%s %s", provider, endpoint, cache_key[:8])
        except Exception as e:
            # cents-0zfx: WARNING (not DEBUG) so cache health issues are visible.
            # A closed connection / missing table silently returning None can
            # mask outcome drift for hours — surface them at runtime instead.
            logger.warning("Cache lookup failed for %s/%s: %s", provider, endpoint, e)
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

        # cents-lqqw: refuse to write to namespaces marked TTL=0 (dead). The
        # TTL=0 sentinel exists to evict left-over pre-fix rows on read; if
        # the live code path still WRITES to that namespace we'd be growing
        # dead rows forever until prune() catches up. Better to fail loud at
        # the source — a stack trace pointing at the writer is the only way
        # to find a still-wired-up dead caller.
        ttl_days = _ttl_for(provider, endpoint)
        if ttl_days == 0:
            logger.warning(
                "Refusing cache write to dead namespace %s/%s — TTL=0 means this "
                "endpoint was superseded (e.g. bars → bars_split_v1). Update the "
                "caller to use the live namespace.",
                provider, endpoint,
            )
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
            # cents-0zfx: WARNING — a silent cache-write failure means the next
            # call re-fetches over the network, eroding the cache's value
            # invisibly. Surface so operators can investigate.
            logger.warning("Cache write failed for %s/%s: %s", provider, endpoint, e)

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

    def detailed_stats(self) -> list[dict]:
        """Per-(provider, endpoint) breakdown with row count + bytes + TTL policy.

        Returns a list of dicts sorted by total bytes descending so the heaviest
        endpoints surface first.
        """
        rows = self.conn.execute(
            """
            SELECT provider, endpoint,
                   COUNT(*) AS rows,
                   SUM(length(response_data)) AS bytes,
                   MIN(cached_at) AS oldest,
                   MAX(cached_at) AS newest
            FROM api_cache
            GROUP BY provider, endpoint
            """
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "provider": r[0],
                "endpoint": r[1],
                "rows": r[2],
                "bytes": r[3] or 0,
                "oldest": r[4],
                "newest": r[5],
                "ttl_days": _ttl_for(r[0], r[1]),
            })
        out.sort(key=lambda d: d["bytes"], reverse=True)
        return out

    def prune(self) -> dict[tuple[str, str], int]:
        """Delete every row whose (provider, endpoint, cached_at) is past TTL.

        Walks the policy table; for each (provider, endpoint) with a TTL set,
        deletes rows older than the TTL. Endpoints with no policy are left alone.

        Returns a {(provider, endpoint): rows_deleted} dict so callers can
        report what changed.
        """
        deleted: dict[tuple[str, str], int] = {}
        for (provider, endpoint), ttl_days in TTL_DAYS_BY_ENDPOINT.items():
            if ttl_days is None:
                continue
            cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
            cur = self.conn.execute(
                """
                DELETE FROM api_cache
                WHERE provider = ? AND endpoint = ? AND cached_at < ?
                """,
                (provider, endpoint, cutoff),
            )
            if cur.rowcount:
                deleted[(provider, endpoint)] = cur.rowcount
        self.conn.commit()
        return deleted


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
