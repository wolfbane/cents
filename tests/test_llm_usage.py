"""Tests for the LLMUsage model, LLMUsageRepository, recording helper,
pricing module, and `cents usage` CLI."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from cents import llm_usage as llm_usage_mod
from cents.cli import cli
from cents.db import LLMUsageRepository
from cents.db.schema import SCHEMA
from cents.models import LLMUsage
from cents.pricing import estimate_cost_usd


# --- Model + repo round-trip ---


class TestLLMUsageRepository:
    def test_create_and_get(self, db_conn):
        repo = LLMUsageRepository(db_conn)
        row = LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            input_tokens=1500,
            output_tokens=200,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
            context="NVDA",
        )
        repo.create(row)
        got = repo.get(row.id)
        assert got is not None
        assert got.model == "claude-haiku-4-5"
        assert got.agent == "sentiment"
        assert got.operation == "score_article"
        assert got.input_tokens == 1500
        assert got.output_tokens == 200
        assert got.cache_read_input_tokens == 100
        assert got.cache_creation_input_tokens == 50
        assert got.context == "NVDA"
        assert isinstance(got.called_at, datetime)

    def test_list_recent_with_since(self, db_conn):
        repo = LLMUsageRepository(db_conn)
        old = LLMUsage(
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
            called_at=datetime.now() - timedelta(days=10),
        )
        recent = LLMUsage(
            model="claude-haiku-4-5",
            agent="event",
            operation="tag_event",
            called_at=datetime.now() - timedelta(hours=1),
        )
        repo.create(old)
        repo.create(recent)

        rows = repo.list_recent(since=datetime.now() - timedelta(days=1))
        assert len(rows) == 1
        assert rows[0].id == recent.id

        all_rows = repo.list_recent()
        assert len(all_rows) == 2

    def test_aggregate_by_agent(self, db_conn):
        repo = LLMUsageRepository(db_conn)
        for _ in range(3):
            repo.create(
                LLMUsage(
                    model="claude-haiku-4-5",
                    agent="sentiment",
                    operation="score_article",
                    input_tokens=100,
                    output_tokens=20,
                )
            )
        repo.create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="event",
                operation="tag_event",
                input_tokens=200,
                output_tokens=30,
            )
        )

        rows = repo.aggregate("agent")
        by_bucket = {r["bucket"]: r for r in rows}
        assert by_bucket["sentiment"]["calls"] == 3
        assert by_bucket["sentiment"]["input_tokens"] == 300
        assert by_bucket["sentiment"]["output_tokens"] == 60
        assert by_bucket["event"]["calls"] == 1
        assert by_bucket["event"]["input_tokens"] == 200

    def test_aggregate_unsupported_dimension_raises(self, db_conn):
        repo = LLMUsageRepository(db_conn)
        with pytest.raises(ValueError):
            repo.aggregate("nonsense")


# --- record_llm_usage helper ---


def _stub_response(
    model: str = "claude-haiku-4-5",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read=None,
    cache_write=None,
):
    """Mimic an anthropic Message — just `.model` and `.usage.*` are read."""
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


class TestRecordLLMUsage:
    def test_records_happy_path(self, db_conn, monkeypatch):
        monkeypatch.setattr(
            "cents.llm_usage.LLMUsageRepository", lambda: LLMUsageRepository(db_conn)
        )

        response = _stub_response(
            input_tokens=500,
            output_tokens=75,
            cache_read=10,
            cache_write=5,
        )
        llm_usage_mod.record_llm_usage(
            response, agent="sentiment", operation="filter_articles", context="NVDA"
        )

        rows = LLMUsageRepository(db_conn).list_recent()
        assert len(rows) == 1
        r = rows[0]
        assert r.model == "claude-haiku-4-5"
        assert r.agent == "sentiment"
        assert r.operation == "filter_articles"
        assert r.context == "NVDA"
        assert r.input_tokens == 500
        assert r.output_tokens == 75
        assert r.cache_read_input_tokens == 10
        assert r.cache_creation_input_tokens == 5

    def test_swallows_db_failure(self, monkeypatch, caplog):
        class _Boom:
            def create(self, _row):
                raise sqlite3.OperationalError("table missing")

        monkeypatch.setattr("cents.llm_usage.LLMUsageRepository", lambda: _Boom())

        response = _stub_response()
        with caplog.at_level(logging.DEBUG, logger="cents.llm_usage"):
            # Must not raise.
            llm_usage_mod.record_llm_usage(
                response, agent="sentiment", operation="score_article"
            )
        # And must have logged at debug — bookkeeping failure is observable
        # without polluting normal output.
        assert any("record_llm_usage failed" in m for m in caplog.messages)

    def test_handles_none_cache_fields(self, db_conn, monkeypatch):
        """Anthropic returns None for cache fields when caching wasn't used."""
        monkeypatch.setattr(
            "cents.llm_usage.LLMUsageRepository", lambda: LLMUsageRepository(db_conn)
        )

        response = _stub_response(
            input_tokens=10,
            output_tokens=5,
            cache_read=None,
            cache_write=None,
        )
        llm_usage_mod.record_llm_usage(
            response, agent="event", operation="tag_event", context=None
        )

        rows = LLMUsageRepository(db_conn).list_recent()
        assert len(rows) == 1
        assert rows[0].cache_read_input_tokens == 0
        assert rows[0].cache_creation_input_tokens == 0

# --- Pricing ---


class TestPricing:
    def test_haiku_4_5_math(self):
        # 1M input @ $1, 1M output @ $5 → $6.00
        assert estimate_cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.00)

        # Small call: 1000 input, 200 output → 1000 * 1/1M + 200 * 5/1M
        cost = estimate_cost_usd("claude-haiku-4-5", 1000, 200)
        assert cost == pytest.approx(0.001 + 0.001)

    def test_haiku_cache_dimensions(self):
        # 10_000 cache reads @ $0.10/M = $0.001
        # 10_000 cache writes @ $1.25/M = $0.0125
        cost = estimate_cost_usd(
            "claude-haiku-4-5", 0, 0, cache_read=10_000, cache_write=10_000
        )
        assert cost == pytest.approx(0.001 + 0.0125)

    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd("imaginary-llm", 100, 50) is None

    def test_dated_snapshot_resolves_to_family(self):
        # Future-proofing: when Anthropic publishes a dated snapshot like
        # claude-haiku-4-5-20251001, it should price the same.
        cost = estimate_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        assert cost == pytest.approx(6.00)


# --- CLI ---


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    """Real on-disk SQLite DB for CLI tests, wired via CENTS_DB_PATH."""
    db_path = tmp_path / "data" / "cents.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return db_path


class TestUsageCLI:
    def test_summary_empty(self, cli_db):
        runner = CliRunner()
        result = runner.invoke(cli, ["usage", "summary"])
        assert result.exit_code == 0
        assert "No usage recorded" in result.output

    def test_summary_text(self, cli_db):
        # Seed two rows: 2 sentiment, 1 event.
        conn = sqlite3.connect(cli_db)
        conn.row_factory = sqlite3.Row
        repo = LLMUsageRepository(conn)
        repo.create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="sentiment",
                operation="score_article",
                input_tokens=100,
                output_tokens=20,
            )
        )
        repo.create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="sentiment",
                operation="filter_articles",
                input_tokens=300,
                output_tokens=10,
            )
        )
        repo.create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="event",
                operation="tag_event",
                input_tokens=200,
                output_tokens=50,
            )
        )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["usage", "summary", "--by", "agent"])
        assert result.exit_code == 0, result.output
        assert "sentiment" in result.output
        assert "event" in result.output
        # Should include a $ cost since we know haiku-4-5 pricing.
        assert "$" in result.output

    def test_summary_json(self, cli_db):
        conn = sqlite3.connect(cli_db)
        conn.row_factory = sqlite3.Row
        LLMUsageRepository(conn).create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="sentiment",
                operation="score_article",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            )
        )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["usage", "summary", "--by", "agent", "--output", "json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert payload[0]["agent"] == "sentiment"
        assert payload[0]["calls"] == 1
        assert payload[0]["input_tokens"] == 1_000_000
        assert payload[0]["est_cost_usd"] == pytest.approx(6.00, rel=1e-3)

    def test_list_empty(self, cli_db):
        runner = CliRunner()
        result = runner.invoke(cli, ["usage", "list"])
        assert result.exit_code == 0
        assert "No usage recorded" in result.output

    def test_list_shows_recent_calls(self, cli_db):
        conn = sqlite3.connect(cli_db)
        conn.row_factory = sqlite3.Row
        LLMUsageRepository(conn).create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="sentiment",
                operation="score_article",
                input_tokens=100,
                output_tokens=20,
                context="NVDA",
            )
        )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["usage", "list"])
        assert result.exit_code == 0, result.output
        assert "sentiment.score_article" in result.output
        assert "NVDA" in result.output

    def test_list_json(self, cli_db):
        conn = sqlite3.connect(cli_db)
        conn.row_factory = sqlite3.Row
        LLMUsageRepository(conn).create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="event",
                operation="tag_event",
                input_tokens=200,
                output_tokens=30,
                context="2026-99999",
            )
        )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["usage", "list", "--output", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload) == 1
        assert payload[0]["agent"] == "event"
        assert payload[0]["operation"] == "tag_event"
        assert payload[0]["context"] == "2026-99999"
        assert payload[0]["est_cost_usd"] is not None
