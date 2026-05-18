"""Tests for survivorship-aware universes (delistings table + asof resolver)."""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cents.db import DelistingsRepository, UniverseRepository
from cents.db.schema import SCHEMA
from cents.models import Delisting, Universe, UniverseSource


class TestDelistingModel:
    def test_symbol_normalized_uppercase(self):
        d = Delisting(symbol=" aapl ", delisted_on=date(2025, 6, 1))
        assert d.symbol == "AAPL"

    def test_rejects_empty_symbol(self):
        with pytest.raises(ValueError, match="non-empty"):
            Delisting(symbol="", delisted_on=date(2025, 6, 1))

    def test_default_source_is_fmp(self):
        d = Delisting(symbol="X", delisted_on=date(2025, 1, 1))
        assert d.source == "fmp"

    def test_default_ingested_at_is_now(self):
        before = datetime.now()
        d = Delisting(symbol="X", delisted_on=date(2025, 1, 1))
        after = datetime.now()
        assert before <= d.ingested_at <= after


class TestDelistingsRepository:
    def test_upsert_and_get(self, db_conn):
        repo = DelistingsRepository(db_conn)
        d = Delisting(symbol="OLD", delisted_on=date(2025, 6, 1), last_close=12.34)
        repo.upsert(d)
        got = repo.get("old")
        assert got is not None
        assert got.symbol == "OLD"
        assert got.delisted_on == date(2025, 6, 1)
        assert got.last_close == pytest.approx(12.34)
        assert got.source == "fmp"

    def test_upsert_replaces_existing(self, db_conn):
        repo = DelistingsRepository(db_conn)
        repo.upsert(Delisting(symbol="X", delisted_on=date(2025, 1, 1), last_close=1.0))
        repo.upsert(Delisting(symbol="X", delisted_on=date(2025, 3, 1), last_close=2.0))
        got = repo.get("X")
        assert got.delisted_on == date(2025, 3, 1)
        assert got.last_close == pytest.approx(2.0)

    def test_list_since_filters_by_date(self, db_conn):
        repo = DelistingsRepository(db_conn)
        repo.upsert(Delisting(symbol="OLD", delisted_on=date(2024, 1, 1)))
        repo.upsert(Delisting(symbol="NEW", delisted_on=date(2025, 6, 1)))
        repo.upsert(Delisting(symbol="MID", delisted_on=date(2025, 1, 1)))
        results = repo.list_since(date(2025, 1, 1))
        symbols = {d.symbol for d in results}
        assert symbols == {"NEW", "MID"}

    def test_list_all(self, db_conn):
        repo = DelistingsRepository(db_conn)
        repo.upsert(Delisting(symbol="A", delisted_on=date(2025, 1, 1)))
        repo.upsert(Delisting(symbol="B", delisted_on=date(2025, 2, 1)))
        assert {d.symbol for d in repo.list_all()} == {"A", "B"}

    def test_delete(self, db_conn):
        repo = DelistingsRepository(db_conn)
        repo.upsert(Delisting(symbol="GONE", delisted_on=date(2025, 1, 1)))
        assert repo.delete("gone") is True
        assert repo.get("GONE") is None

    def test_delete_missing_returns_false(self, db_conn):
        repo = DelistingsRepository(db_conn)
        assert repo.delete("ghost") is False


class TestAsofUniverseResolver:
    """The asof_date parameter on resolve_symbols layers delistings into
    screener output so point-in-time membership is reconstructable."""

    def _seed_universe(self, db_conn):
        urepo = UniverseRepository(db_conn)
        urepo.create(Universe(name="parent", symbols=["AAA", "BBB", "CCC"]))
        urepo.create(
            Universe(
                name="screened",
                source=UniverseSource.SCREENER,
                source_config={"strategy": "fake_d", "over": "parent", "limit": 10},
            )
        )
        return urepo

    def _install_fake_screener(self, monkeypatch, returned: list[str]):
        from cents import screeners as screener_mod

        class _Fake:
            name = "fake_d"

            def describe(self):
                return {"description": "", "rules": []}

            def screen(self, candidate_symbols=None):
                return list(returned)

        monkeypatch.setitem(screener_mod.SCREENERS, "fake_d", _Fake())

    def test_default_resolution_excludes_delistings(self, db_conn, monkeypatch):
        """Without asof_date, resolution behaves exactly as before — no
        delistings layered in. This is the live forward-test path."""
        from cents.factory import universe_resolver as resolver_mod

        urepo = self._seed_universe(db_conn)
        DelistingsRepository(db_conn).upsert(
            Delisting(symbol="GONE", delisted_on=date(2025, 6, 1))
        )
        self._install_fake_screener(monkeypatch, ["AAA", "BBB"])
        monkeypatch.setattr(resolver_mod, "UniverseRepository", lambda: urepo)

        symbols = resolver_mod.resolve_symbols(urepo.get("screened"))
        assert symbols == ["AAA", "BBB"]
        assert "GONE" not in symbols

    def test_asof_includes_delistings_on_or_after_date(self, db_conn, monkeypatch):
        from cents.factory import universe_resolver as resolver_mod

        urepo = self._seed_universe(db_conn)
        drepo = DelistingsRepository(db_conn)
        drepo.upsert(Delisting(symbol="GONE_NEW", delisted_on=date(2025, 7, 1)))
        drepo.upsert(Delisting(symbol="GONE_OLD", delisted_on=date(2024, 1, 1)))
        self._install_fake_screener(monkeypatch, ["AAA", "BBB"])
        monkeypatch.setattr(resolver_mod, "UniverseRepository", lambda: urepo)
        monkeypatch.setattr(resolver_mod, "DelistingsRepository", lambda: drepo)

        symbols = resolver_mod.resolve_symbols(
            urepo.get("screened"),
            asof_date=date(2025, 1, 1),
        )
        # GONE_NEW (delisted after asof) is added; GONE_OLD (delisted before) is not.
        assert set(symbols) == {"AAA", "BBB", "GONE_NEW"}

    def test_asof_does_not_duplicate_symbols(self, db_conn, monkeypatch):
        """If a delisted symbol is somehow still in the current screener
        output, it must not appear twice in the resolved list."""
        from cents.factory import universe_resolver as resolver_mod

        urepo = self._seed_universe(db_conn)
        drepo = DelistingsRepository(db_conn)
        drepo.upsert(Delisting(symbol="BBB", delisted_on=date(2025, 6, 1)))
        self._install_fake_screener(monkeypatch, ["AAA", "BBB"])
        monkeypatch.setattr(resolver_mod, "UniverseRepository", lambda: urepo)
        monkeypatch.setattr(resolver_mod, "DelistingsRepository", lambda: drepo)

        symbols = resolver_mod.resolve_symbols(
            urepo.get("screened"),
            asof_date=date(2025, 1, 1),
        )
        assert symbols.count("BBB") == 1

    def test_asof_no_effect_on_static_source(self, db_conn, monkeypatch):
        """STATIC universes already specify members explicitly. asof_date
        is a no-op there — we only need point-in-time reconstruction for
        live-screened universes."""
        from cents.factory import universe_resolver as resolver_mod

        drepo = DelistingsRepository(db_conn)
        drepo.upsert(Delisting(symbol="GONE", delisted_on=date(2025, 6, 1)))
        monkeypatch.setattr(resolver_mod, "DelistingsRepository", lambda: drepo)

        uni = Universe(name="s", source=UniverseSource.STATIC, symbols=["AAPL", "NVDA"])
        symbols = resolver_mod.resolve_symbols(uni, asof_date=date(2025, 1, 1))
        assert symbols == ["AAPL", "NVDA"]


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    """Provision a real sqlite db at tmp_path and point CENTS_DB_PATH at it."""
    db_path = tmp_path / "data" / "cents.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return db_path


class TestIngestDelistingsCli:
    def test_ingest_dry_run_writes_nothing(self, cli_db, monkeypatch):
        from cents.cli import cli

        fake_delistings = [
            Delisting(symbol="ABC", delisted_on=date(2025, 6, 1)),
            Delisting(symbol="XYZ", delisted_on=date(2025, 7, 1)),
        ]

        class _FakeProvider:
            def get_delistings(self, since):
                return list(fake_delistings)

        # Patch the provider class used by the CLI command — it imports
        # FMPFundamentalsProvider inside the function body.
        monkeypatch.setattr(
            "cents.data.fmp.FMPFundamentalsProvider",
            lambda: _FakeProvider(),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["universe", "ingest-delistings", "--dry-run", "--since", "2024-01-01"],
        )
        assert result.exit_code == 0, result.output
        # Nothing should have been persisted.
        assert DelistingsRepository().list_all() == []

    def test_ingest_persists_delistings(self, cli_db, monkeypatch):
        from cents.cli import cli

        fake = [
            Delisting(symbol="ABC", delisted_on=date(2025, 6, 1), last_close=3.14),
            Delisting(symbol="XYZ", delisted_on=date(2025, 7, 1)),
        ]

        class _FakeProvider:
            def get_delistings(self, since):
                assert since == date(2024, 1, 1)
                return list(fake)

        monkeypatch.setattr(
            "cents.data.fmp.FMPFundamentalsProvider",
            lambda: _FakeProvider(),
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["universe", "ingest-delistings", "--since", "2024-01-01"],
        )
        assert result.exit_code == 0, result.output
        symbols = {d.symbol for d in DelistingsRepository().list_all()}
        assert symbols == {"ABC", "XYZ"}

    def test_ingest_without_api_key_no_ops_cleanly(self, cli_db, monkeypatch):
        """No FMP API key → command surfaces a friendly message and exits 0."""
        from cents.cli import cli

        def _raise(*_a, **_kw):
            from cents.exceptions import ConfigurationError
            raise ConfigurationError("FMP_API_KEY not configured")

        monkeypatch.setattr("cents.data.fmp.FMPFundamentalsProvider", _raise)
        runner = CliRunner()
        result = runner.invoke(cli, ["universe", "ingest-delistings"])
        assert result.exit_code == 0, result.output
        assert "FMP_API_KEY" in result.output


class TestShowAsofCli:
    def test_show_with_asof_includes_delistings(self, cli_db, monkeypatch):
        from cents.cli import cli
        from cents import screeners as screener_mod

        UniverseRepository().create(Universe(name="parent", symbols=["AAA", "BBB"]))
        UniverseRepository().create(
            Universe(
                name="screened",
                source=UniverseSource.SCREENER,
                source_config={"strategy": "fake_show", "over": "parent", "limit": 10},
            )
        )
        DelistingsRepository().upsert(
            Delisting(symbol="GONE", delisted_on=date(2025, 6, 1))
        )

        class _Fake:
            name = "fake_show"

            def describe(self):
                return {"description": "", "rules": []}

            def screen(self, candidate_symbols=None):
                return ["AAA", "BBB"]

        monkeypatch.setitem(screener_mod.SCREENERS, "fake_show", _Fake())

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["universe", "show", "screened", "--as-of", "2025-01-01", "--output", "json"],
        )
        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.output)
        assert payload["as_of"] == "2025-01-01"
        assert set(payload["resolved_symbols"]) == {"AAA", "BBB", "GONE"}

    def test_show_with_asof_rejects_bad_date(self, cli_db):
        from cents.cli import cli

        UniverseRepository().create(Universe(name="u", symbols=["X"]))
        runner = CliRunner()
        result = runner.invoke(cli, ["universe", "show", "u", "--as-of", "notadate"])
        assert result.exit_code != 0
