"""Resolve a Universe into a concrete list of symbols."""

import json
import logging
import urllib.error
import urllib.request

from cents.config import get_settings
from cents.db import WatchlistRepository
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


def resolve_symbols(universe: Universe) -> list[str]:
    """Resolve a universe to its current symbol list."""
    if universe.source == UniverseSource.STATIC:
        return list(universe.symbols)

    if universe.source == UniverseSource.WATCHLIST:
        items = WatchlistRepository().list()
        return [item.symbol for item in items]

    if universe.source == UniverseSource.FMP_INDEX:
        return _resolve_fmp_index(universe)

    raise ValueError(f"Unsupported universe source: {universe.source}")


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
