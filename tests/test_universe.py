"""Tests for universe model, repository, resolver, and CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cents.db import UniverseRepository, WatchlistRepository
from cents.factory.universe_resolver import resolve_symbols
from cents.models import Universe, UniverseSource, WatchlistItem


class TestUniverseModel:
    def test_validates_non_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            Universe(name="")

    def test_name_normalized_lowercase(self):
        uni = Universe(name="  SP500  ")
        assert uni.name == "sp500"

    def test_symbols_normalized_uppercase(self):
        uni = Universe(name="x", symbols=["aapl", "msft "])
        assert uni.symbols == ["AAPL", "MSFT"]

    def test_fmp_index_requires_index_in_source_config(self):
        with pytest.raises(ValueError, match="source_config"):
            Universe(name="bad", source=UniverseSource.FMP_INDEX)

    def test_fmp_index_with_index_succeeds(self):
        uni = Universe(
            name="sp500",
            source=UniverseSource.FMP_INDEX,
            source_config={"index": "sp500"},
        )
        assert uni.source_config["index"] == "sp500"


class TestUniverseRepository:
    def test_create_and_get(self, db_conn):
        repo = UniverseRepository(db_conn)
        uni = Universe(name="test", symbols=["NVDA", "AAPL"])
        repo.create(uni)
        retrieved = repo.get("test")
        assert retrieved is not None
        assert retrieved.symbols == ["NVDA", "AAPL"]
        assert retrieved.source == UniverseSource.STATIC

    def test_get_case_insensitive(self, db_conn):
        repo = UniverseRepository(db_conn)
        repo.create(Universe(name="Mega"))
        assert repo.get("MEGA") is not None
        assert repo.get("mega") is not None

    def test_list(self, db_conn):
        repo = UniverseRepository(db_conn)
        repo.create(Universe(name="alpha"))
        repo.create(Universe(name="beta"))
        names = [u.name for u in repo.list()]
        assert set(names) == {"alpha", "beta"}

    def test_update(self, db_conn):
        repo = UniverseRepository(db_conn)
        uni = Universe(name="u1", description="old")
        repo.create(uni)
        uni.description = "new"
        repo.update(uni)
        assert repo.get("u1").description == "new"

    def test_delete(self, db_conn):
        repo = UniverseRepository(db_conn)
        repo.create(Universe(name="gone"))
        assert repo.delete("gone") is True
        assert repo.get("gone") is None

    def test_set_default_atomically_clears_previous(self, db_conn):
        repo = UniverseRepository(db_conn)
        repo.create(Universe(name="a", is_default=True))
        repo.create(Universe(name="b"))
        repo.set_default("b")
        assert repo.get("a").is_default is False
        assert repo.get("b").is_default is True
        assert repo.get_default().name == "b"

    def test_set_default_unknown_returns_none(self, db_conn):
        repo = UniverseRepository(db_conn)
        assert repo.set_default("ghost") is None


class TestUniverseResolver:
    def test_static_returns_symbols_verbatim(self):
        uni = Universe(name="t", source=UniverseSource.STATIC, symbols=["AAPL", "NVDA"])
        assert resolve_symbols(uni) == ["AAPL", "NVDA"]

    def test_watchlist_mirrors_repo(self, db_conn):
        wrepo = WatchlistRepository(db_conn)
        wrepo.add(WatchlistItem(symbol="NVDA"))
        wrepo.add(WatchlistItem(symbol="AAPL"))

        uni = Universe(name="w", source=UniverseSource.WATCHLIST)
        with patch("cents.factory.universe_resolver.WatchlistRepository") as MockRepo:
            MockRepo.return_value = wrepo
            symbols = resolve_symbols(uni)
        assert set(symbols) == {"NVDA", "AAPL"}

    def test_fmp_index_raises_without_api_key(self, monkeypatch):
        from cents.exceptions import ConfigurationError

        monkeypatch.delenv("FMP_API_KEY", raising=False)
        with patch("cents.factory.universe_resolver.get_settings") as gs:
            gs.return_value.fmp_api_key = None
            gs.return_value.default_api_timeout = 5
            uni = Universe(
                name="sp",
                source=UniverseSource.FMP_INDEX,
                source_config={"index": "sp500"},
            )
            with pytest.raises(ConfigurationError, match="FMP_API_KEY"):
                resolve_symbols(uni)


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    """Provision a real sqlite db at tmp_path and point CENTS_DB_PATH at it."""
    import sqlite3
    from cents.db.schema import SCHEMA

    db_path = tmp_path / "data" / "cents.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return db_path


class TestUniverseCli:
    def test_create_static(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "universe", "create", "test",
            "--source", "static",
            "--symbols", "NVDA,AAPL",
        ])
        assert result.exit_code == 0, result.output
        uni = UniverseRepository().get("test")
        assert uni is not None
        assert uni.symbols == ["NVDA", "AAPL"]

    def test_create_from_file(self, tmp_path, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        sym_file = tmp_path / "syms.txt"
        sym_file.write_text("AAPL\n# comment\nMSFT\n")
        result = runner.invoke(cli, [
            "universe", "create", "fromfile",
            "--from-file", str(sym_file),
        ])
        assert result.exit_code == 0, result.output
        assert UniverseRepository().get("fromfile").symbols == ["AAPL", "MSFT"]

    def test_create_fmp_index_requires_index(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "universe", "create", "idx",
            "--source", "fmp_index",
        ])
        assert result.exit_code != 0

    def test_create_watchlist_source(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "universe", "create", "watch",
            "--source", "watchlist",
        ])
        assert result.exit_code == 0, result.output
        assert UniverseRepository().get("watch").source == UniverseSource.WATCHLIST

    def test_list_and_show(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        UniverseRepository().create(Universe(name="alpha", symbols=["A", "B"]))
        result_list = runner.invoke(cli, ["universe", "list"])
        assert "alpha" in result_list.output
        result_show = runner.invoke(cli, ["universe", "show", "alpha"])
        assert "alpha" in result_show.output
        assert "A" in result_show.output

    def test_set_default(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        repo = UniverseRepository()
        repo.create(Universe(name="a"))
        repo.create(Universe(name="b"))
        result = runner.invoke(cli, ["universe", "set-default", "b"])
        assert result.exit_code == 0, result.output
        assert UniverseRepository().get_default().name == "b"

    def test_refresh_static_is_noop(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        UniverseRepository().create(Universe(name="static1", symbols=["X"]))
        result = runner.invoke(cli, ["universe", "refresh", "static1"])
        assert result.exit_code == 0, result.output
        assert UniverseRepository().get("static1").symbols == ["X"]

    def test_delete_with_force(self, cli_db):
        from cents.cli import cli

        runner = CliRunner()
        UniverseRepository().create(Universe(name="bye"))
        result = runner.invoke(cli, ["universe", "delete", "bye", "--force"])
        assert result.exit_code == 0, result.output
        assert UniverseRepository().get("bye") is None
