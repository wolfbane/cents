"""Resolve a Universe into a concrete list of symbols."""

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import date

from cents.config import get_settings
from cents.db import DelistingsRepository, UniverseRepository, WatchlistRepository
from cents.exceptions import ConfigurationError, DataFetchError
from cents.models import Universe, UniverseSource

logger = logging.getLogger(__name__)

FMP_INDEX_BASE_URL = "https://financialmodelingprep.com/stable"

# Map our canonical index keys to FMP's constituent endpoints
FMP_INDEX_ENDPOINTS: dict[str, str] = {
    "sp500": "sp500-constituent",
    "nasdaq": "nasdaq-constituent",
    "dowjones": "dowjones-constituent",
}

# A screener universe without an `over` parent would scan every US-listed symbol.
# That's expensive and easy to trigger by accident, so it's gated behind an
# explicit env flag. Users who really want full-universe screens set this once.
FULL_UNIVERSE_ENV = "CENTS_SCREENER_ALLOW_FULL_UNIVERSE"


def resolve_symbols(
    universe: Universe,
    _visited: frozenset[str] = frozenset(),
    asof_date: date | None = None,
) -> list[str]:
    """Resolve a universe to its current symbol list.

    ``_visited`` carries the names of universes currently being resolved on
    this call stack so screener parent-chains can't form a cycle.

    ``asof_date``, when provided, layers survivorship-bias correction onto
    SCREENER-sourced universes: the resolved member list is the screener's
    current output PLUS any tracked delistings whose ``delisted_on`` is
    on/after ``asof_date`` (those symbols were members of the screened
    universe as of that date even though they have since fallen out).
    Other sources are unaffected — STATIC/WATCHLIST/FMP_INDEX members are
    already specified explicitly and don't need reconstruction.
    """
    if universe.source == UniverseSource.STATIC:
        return list(universe.symbols)

    if universe.source == UniverseSource.WATCHLIST:
        items = WatchlistRepository().list()
        return [item.symbol for item in items]

    if universe.source == UniverseSource.FMP_INDEX:
        return _resolve_fmp_index(universe)

    if universe.source == UniverseSource.SCREENER:
        return _resolve_screener(universe, _visited, asof_date=asof_date)

    raise ValueError(f"Unsupported universe source: {universe.source}")


def _resolve_screener(
    universe: Universe,
    visited: frozenset[str],
    asof_date: date | None = None,
) -> list[str]:
    from cents.screeners import get_screener

    cfg = universe.source_config
    strategy = cfg.get("strategy")
    if not strategy:
        raise ValueError(
            f"Screener universe '{universe.name}' is missing source_config['strategy']"
        )

    screener = get_screener(strategy)
    limit = int(cfg.get("limit", 30))

    over = cfg.get("over")
    if over:
        if over in visited:
            raise ValueError(
                f"Universe resolution cycle: {' → '.join(visited)} → {over}"
            )
        parent = UniverseRepository().get(over)
        if parent is None:
            raise ValueError(
                f"Screener universe '{universe.name}' references unknown parent universe '{over}'"
            )
        if parent.name == universe.name:
            raise ValueError(
                f"Screener universe '{universe.name}' cannot reference itself as parent"
            )
        candidates = resolve_symbols(parent, visited | {universe.name})
        if not candidates:
            return []
    else:
        if os.environ.get(FULL_UNIVERSE_ENV, "").lower() not in ("1", "true", "yes"):
            raise ConfigurationError(
                f"Screener universe '{universe.name}' has no `over` parent. "
                f"Full-universe screens are gated by {FULL_UNIVERSE_ENV}=1 to avoid "
                "accidental wide scans. Set the env var or add `--over <parent>` "
                "when creating the universe."
            )
        candidates = None

    symbols = screener.screen(candidate_symbols=candidates)
    symbols = symbols[:limit]

    if asof_date is not None:
        # Layer in tracked delistings so the resolved member list reflects
        # point-in-time membership rather than current survivors. A symbol
        # that was delisted on/after asof_date was still listed on that day
        # and so was screenable in principle.
        try:
            delistings = DelistingsRepository().list_since(asof_date)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Delistings lookup failed for asof %s: %s", asof_date, exc)
            delistings = []
        seen = set(symbols)
        for d in delistings:
            if d.symbol not in seen:
                symbols.append(d.symbol)
                seen.add(d.symbol)

    return symbols


def _resolve_fmp_index(universe: Universe) -> list[str]:
    settings = get_settings()
    if not settings.fmp_api_key:
        raise ConfigurationError(
            "Universe '{}' uses FMP_INDEX source but FMP_API_KEY is not configured.".format(universe.name)
        )

    index_key = universe.source_config.get("index", "").strip().lower()
    endpoint = FMP_INDEX_ENDPOINTS.get(index_key)
    if not endpoint:
        raise ValueError(
            f"Unknown FMP index '{index_key}' for universe '{universe.name}'. "
            f"Supported: {', '.join(sorted(FMP_INDEX_ENDPOINTS))}"
        )

    url = f"{FMP_INDEX_BASE_URL}/{endpoint}?apikey={settings.fmp_api_key}"
    try:
        with urllib.request.urlopen(url, timeout=settings.default_api_timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise DataFetchError(f"FMP index fetch failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DataFetchError(f"FMP index returned invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise DataFetchError(f"FMP index returned unexpected payload: {type(data).__name__}")

    symbols: list[str] = []
    for row in data:
        sym = row.get("symbol") if isinstance(row, dict) else None
        if sym:
            symbols.append(sym.strip().upper())
    return symbols
