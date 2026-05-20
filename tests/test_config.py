"""Tests for config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cents.config import Settings, get_settings, _load_config_file


class TestSettings:
    """Tests for Settings dataclass."""

    def test_default_values(self):
        """Settings has expected defaults."""
        s = Settings()
        assert s.news_api_key is None
        assert s.fred_api_key is None
        assert s.fmp_api_key is None
        assert s.alpaca_api_key is None
        assert s.alpaca_secret_key is None
        assert s.anthropic_api_key is None
        assert s.default_scan_threshold == 5.0
        assert s.default_webhook is None
        assert s.default_output == "text"
        # cents-87v: Anthropic per-request timeout default (SDK default is 600s, much too long)
        assert s.anthropic_timeout_sec == 30.0


class TestLoadConfigFile:
    """Tests for _load_config_file."""

    def test_missing_file_returns_empty(self, tmp_path):
        """Non-existent file returns empty dict."""
        result = _load_config_file(tmp_path / "nonexistent.toml")
        assert result == {}

    def test_valid_toml_top_level(self, tmp_path):
        """Valid TOML with top-level keys is loaded."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('news_api_key = "test123"\ndefault_scan_threshold = 10.0')
        result = _load_config_file(config_file)
        assert result["news_api_key"] == "test123"
        assert result["default_scan_threshold"] == 10.0

    def test_valid_toml_cents_section(self, tmp_path):
        """Valid TOML with [cents] section is loaded."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[cents]\nfmp_api_key = "fmp_key"')
        result = _load_config_file(config_file)
        assert result["fmp_api_key"] == "fmp_key"

    def test_malformed_toml_returns_empty(self, tmp_path):
        """Malformed TOML returns empty dict."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml [[[")
        result = _load_config_file(config_file)
        assert result == {}


class TestGetSettings:
    """Tests for get_settings."""

    def test_defaults_when_no_config(self, tmp_path):
        """Returns defaults when config file doesn't exist."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings(str(tmp_path / "nonexistent.toml"))
        assert settings.news_api_key is None
        assert settings.default_scan_threshold == 5.0
        assert settings.default_output == "text"

    def test_loads_from_config_file(self, tmp_path):
        """Settings loaded from config file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
news_api_key = "news_test"
fred_api_key = "fred_test"
default_scan_threshold = 7.5
default_output = "json"
""")
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings(str(config_file))
        assert settings.news_api_key == "news_test"
        assert settings.fred_api_key == "fred_test"
        assert settings.default_scan_threshold == 7.5
        assert settings.default_output == "json"

    def test_env_vars_override_config(self, tmp_path):
        """Environment variables override config file values."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('news_api_key = "from_file"')

        env = {"NEWS_API_KEY": "from_env"}
        with patch.dict(os.environ, env, clear=True):
            settings = get_settings(str(config_file))
        assert settings.news_api_key == "from_env"

    def test_all_env_vars(self, tmp_path):
        """All environment variables are respected."""
        env = {
            "NEWS_API_KEY": "news",
            "FRED_API_KEY": "fred",
            "FMP_API_KEY": "fmp",
            "ALPACA_API_KEY": "alpaca",
            "ALPACA_SECRET_KEY": "secret",
            "ANTHROPIC_API_KEY": "anthropic",
            "CENTS_SCAN_THRESHOLD": "8.0",
            "CENTS_WEBHOOK_URL": "https://hook.test",
            "CENTS_OUTPUT_FORMAT": "json",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = get_settings(str(tmp_path / "nonexistent.toml"))

        assert settings.news_api_key == "news"
        assert settings.fred_api_key == "fred"
        assert settings.fmp_api_key == "fmp"
        assert settings.alpaca_api_key == "alpaca"
        assert settings.alpaca_secret_key == "secret"
        assert settings.anthropic_api_key == "anthropic"
        assert settings.default_scan_threshold == 8.0
        assert settings.default_webhook == "https://hook.test"
        assert settings.default_output == "json"

    def test_invalid_threshold_defaults_to_5(self, tmp_path):
        """Invalid threshold value defaults to 5.0."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('default_scan_threshold = "not_a_number"')
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings(str(config_file))
        assert settings.default_scan_threshold == 5.0

    def test_anthropic_timeout_default_30s(self, tmp_path):
        """cents-87v: default anthropic_timeout_sec is 30 (NOT the SDK's 600s)."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings(str(tmp_path / "nonexistent.toml"))
        assert settings.anthropic_timeout_sec == 30.0

    def test_anthropic_timeout_env_override(self, tmp_path):
        """CENTS_ANTHROPIC_TIMEOUT_SEC env var overrides default."""
        with patch.dict(os.environ, {"CENTS_ANTHROPIC_TIMEOUT_SEC": "15"}, clear=True):
            settings = get_settings(str(tmp_path / "nonexistent.toml"))
        assert settings.anthropic_timeout_sec == 15.0

    def test_anthropic_timeout_config_file(self, tmp_path):
        """anthropic_timeout_sec in config file is honored."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("anthropic_timeout_sec = 45.0")
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings(str(config_file))
        assert settings.anthropic_timeout_sec == 45.0

    def test_anthropic_timeout_invalid_falls_back_to_30(self, tmp_path):
        """Malformed value falls back to the safe 30s default."""
        with patch.dict(os.environ, {"CENTS_ANTHROPIC_TIMEOUT_SEC": "garbage"}, clear=True):
            settings = get_settings(str(tmp_path / "nonexistent.toml"))
        assert settings.anthropic_timeout_sec == 30.0

    def test_anthropic_timeout_negative_falls_back_to_30(self, tmp_path):
        """Negative or zero value falls back to the safe 30s default."""
        with patch.dict(os.environ, {"CENTS_ANTHROPIC_TIMEOUT_SEC": "-5"}, clear=True):
            settings = get_settings(str(tmp_path / "nonexistent.toml"))
        assert settings.anthropic_timeout_sec == 30.0

    def test_invalid_output_format_defaults_to_text(self, tmp_path):
        """Invalid output format defaults to 'text'."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('default_output = "invalid_format"')
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings(str(config_file))
        assert settings.default_output == "text"

    def test_cents_config_env_var(self, tmp_path):
        """CENTS_CONFIG environment variable specifies config path."""
        config_file = tmp_path / "custom_config.toml"
        config_file.write_text('fmp_api_key = "custom_fmp"')

        env = {"CENTS_CONFIG": str(config_file)}
        with patch.dict(os.environ, env, clear=True):
            settings = get_settings()  # No explicit path
        assert settings.fmp_api_key == "custom_fmp"

    def test_explicit_path_overrides_env_var(self, tmp_path):
        """Explicit path parameter overrides CENTS_CONFIG env var."""
        env_config = tmp_path / "env_config.toml"
        env_config.write_text('news_api_key = "from_env_config"')

        explicit_config = tmp_path / "explicit_config.toml"
        explicit_config.write_text('news_api_key = "from_explicit"')

        env = {"CENTS_CONFIG": str(env_config)}
        with patch.dict(os.environ, env, clear=True):
            settings = get_settings(str(explicit_config))
        assert settings.news_api_key == "from_explicit"
