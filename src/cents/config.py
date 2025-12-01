"""Configuration loader for cents.

Loads settings from a config file (TOML) and environment variables.
Environment variables take precedence over config file values.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tomllib

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    news_api_key: str | None = None
    fred_api_key: str | None = None
    fmp_api_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    default_scan_threshold: float = 5.0
    default_webhook: str | None = None
    default_output: str = "text"


def _load_config_file(config_path: Path) -> dict:
    """Load TOML config file if it exists."""

    if not config_path.exists():
        return {}

    try:
        data = tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
    except OSError as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        return {}

    # Allow values either under [cents] or at top-level
    return data.get("cents", data)


def get_settings(config_path: str | None = None) -> Settings:
    """Load settings from config file and environment variables."""

    env_path = os.environ.get("CENTS_CONFIG")
    path = Path(config_path or env_path or Path.home() / ".cents" / "config.toml")
    file_config = _load_config_file(path)

    def _get(key: str, env_var: str, default):
        value = os.environ.get(env_var, None)
        if value is not None:
            return value
        return file_config.get(key, default)

    default_output = _get("default_output", "CENTS_OUTPUT_FORMAT", "text")
    if default_output not in {"text", "json"}:
        default_output = "text"

    threshold_raw = _get("default_scan_threshold", "CENTS_SCAN_THRESHOLD", 5.0)
    try:
        threshold_value = float(threshold_raw)
    except (TypeError, ValueError):
        threshold_value = 5.0

    return Settings(
        news_api_key=_get("news_api_key", "NEWS_API_KEY", None),
        fred_api_key=_get("fred_api_key", "FRED_API_KEY", None),
        fmp_api_key=_get("fmp_api_key", "FMP_API_KEY", None),
        alpaca_api_key=_get("alpaca_api_key", "ALPACA_API_KEY", None),
        alpaca_secret_key=_get("alpaca_secret_key", "ALPACA_SECRET_KEY", None),
        default_scan_threshold=threshold_value,
        default_webhook=_get("default_webhook", "CENTS_WEBHOOK_URL", None),
        default_output=default_output,
    )

