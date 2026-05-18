"""Tests for evidence ↔ LLM-call provenance + blob store + trace CLI (bead cents-dzg)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from cents.agents.base import make_provenance
from cents.agents.sentiment import SentimentAgent, clear_sentiment_cache
from cents.cli import cli
from cents.db import EvidenceRepository, LLMUsageRepository
from cents.db.schema import SCHEMA, _migrate_schema
from cents.llm_usage import blob_path_for, persist_call_blob
from cents.models import Evidence, EvidenceType, LLMUsage


# --- make_provenance --------------------------------------------------------


class TestMakeProvenance:
    def test_returns_stable_hashes(self):
        a = make_provenance(
            prompt="hello",
            input_text="world",
            output_text="42",
            model="claude-haiku-4-5",
            llm_call_id="abc12345",
        )
        b = make_provenance(
            prompt="hello",
            input_text="world",
            output_text="42",
            model="claude-haiku-4-5",
            llm_call_id="abc12345",
        )
        assert a == b
        assert a["llm_call_id"] == "abc12345"
        assert a["model_snapshot"] == "claude-haiku-4-5"
        assert len(a["prompt_sha256"]) == 64
        # Different source strings → different hashes.
        assert a["prompt_sha256"] != a["input_sha256"]

    def test_distinct_inputs_distinct_hashes(self):
        a = make_provenance(
            prompt="foo", input_text="x", output_text="x",
            model="m", llm_call_id="i",
        )
        b = make_provenance(
            prompt="bar", input_text="x", output_text="x",
            model="m", llm_call_id="i",
        )
        assert a["prompt_sha256"] != b["prompt_sha256"]
        assert a["input_sha256"] == b["input_sha256"]


# --- Evidence persistence with provenance ----------------------------------


class TestEvidenceProvenancePersistence:
    def test_create_and_round_trip_with_provenance(self, db_conn):
        repo = EvidenceRepository(db_conn)
        prov = make_provenance(
            prompt="P", input_text="I", output_text="O",
            model="claude-haiku-4-5", llm_call_id="callid01",
        )
        ev = Evidence(
            agent="sentiment",
            content="Earnings beat",
            source="newsapi",
            symbol="NVDA",
            type=EvidenceType.SUPPORTING,
            confidence=0.8,
            provenance=prov,
        )
        repo.create(ev)
        got = repo.get(ev.id)
        assert got is not None
        assert got.provenance is not None
        assert got.provenance["llm_call_id"] == "callid01"
        assert got.provenance["model_snapshot"] == "claude-haiku-4-5"
        assert got.provenance["prompt_sha256"] == prov["prompt_sha256"]
        assert got.provenance["input_sha256"] == prov["input_sha256"]
        assert got.provenance["output_sha256"] == prov["output_sha256"]

    def test_evidence_without_provenance_is_none(self, db_conn):
        repo = EvidenceRepository(db_conn)
        ev = Evidence(agent="fundamentals", content="P/E 15", source="fmp", symbol="MSFT")
        repo.create(ev)
        got = repo.get(ev.id)
        assert got is not None
        assert got.provenance is None


# --- Happy-path LLM call writes provenance ---------------------------------


class TestSentimentAgentLLMCallWritesProvenance:
    def setup_method(self):
        clear_sentiment_cache()

    def test_llm_evidence_has_provenance(self, db_conn, monkeypatch):
        """A SentimentAgent LLM scoring call should produce evidence with provenance."""
        # Wire the LLMUsageRepository to the in-memory db so record_llm_usage
        # round-trips cleanly.
        monkeypatch.setattr(
            "cents.llm_usage.LLMUsageRepository", lambda: LLMUsageRepository(db_conn)
        )

        # Mock anthropic client returning a known JSON score.
        class _Content:
            def __init__(self, text):
                self.text = text

        class _Response:
            model = "claude-haiku-4-5"
            content = [_Content('{"score": 0.8, "reasoning": "bullish"}')]
            usage = SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _Response()

        # Patch settings so the agent thinks it has API keys.
        monkeypatch.setattr(
            "cents.agents.sentiment.get_settings",
            lambda: SimpleNamespace(
                news_api_key="x", anthropic_api_key="y", default_api_timeout=10
            ),
        )

        agent = SentimentAgent(anthropic_client=mock_client)
        article = {
            "title": "Stock surges",
            "description": "Strong gains",
            "url": "https://example.com/p1",
        }
        ev_type, score, confidence, metadata, provenance = agent._score_with_llm(
            article, "NVDA", None
        )

        assert provenance is not None
        assert len(provenance["llm_call_id"]) > 0
        assert provenance["model_snapshot"].startswith("claude-haiku-4-5")
        assert len(provenance["prompt_sha256"]) == 64
        assert len(provenance["output_sha256"]) == 64


# --- Blob store + cents evidence trace --------------------------------------


class TestEvidenceTraceCLI:
    def test_trace_reconstructs_prompt_and_output(self, tmp_path, monkeypatch):
        # Use a real on-disk DB for the CLI.
        db_path = tmp_path / "data" / "cents.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        # Insert one piece of evidence with provenance via the repo.
        repo = EvidenceRepository(conn)
        prov = make_provenance(
            prompt="What is the sentiment?",
            input_text="What is the sentiment?",
            output_text='{"score": 0.5}',
            model="claude-haiku-4-5",
            llm_call_id="traceid1",
        )
        ev = Evidence(
            agent="sentiment",
            content="bullish article",
            source="newsapi",
            symbol="NVDA",
            type=EvidenceType.SUPPORTING,
            provenance=prov,
        )
        repo.create(ev)
        conn.close()

        # Persist a matching blob.
        monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
        blob_root = tmp_path / "llm_calls"
        monkeypatch.setenv("CENTS_LLM_BLOB_DIR", str(blob_root))
        path = persist_call_blob(
            "traceid1",
            prompt="What is the sentiment?",
            input_text="What is the sentiment?",
            output_text='{"score": 0.5}',
            model="claude-haiku-4-5",
            agent="sentiment",
            operation="score_article",
        )
        assert path is not None
        assert path.exists()

        runner = CliRunner()
        result = runner.invoke(cli, ["evidence", "trace", ev.id])
        assert result.exit_code == 0, result.output
        assert "What is the sentiment?" in result.output
        assert '"score": 0.5' in result.output
        assert "traceid1" in result.output

    def test_trace_fails_when_no_provenance(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "cents.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        repo = EvidenceRepository(conn)
        ev = Evidence(agent="fundamentals", content="P/E 15", source="fmp", symbol="MSFT")
        repo.create(ev)
        conn.close()

        monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
        runner = CliRunner()
        result = runner.invoke(cli, ["evidence", "trace", ev.id])
        assert result.exit_code == 1
        assert "no llm provenance" in result.output.lower()


# --- Schema migration idempotence ------------------------------------------


class TestMigrationIdempotence:
    def test_migrate_schema_runs_twice(self, tmp_path):
        """Running the migration on a fresh DB twice in a row must not raise."""
        db_path = tmp_path / "mig.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        _migrate_schema(conn)
        # Second call should be a no-op.
        _migrate_schema(conn)
        # Sanity: all provenance columns are present.
        cursor = conn.execute("PRAGMA table_info(evidence)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "llm_call_id" in cols
        assert "model_snapshot" in cols
        assert "prompt_sha256" in cols
        assert "input_sha256" in cols
        assert "output_sha256" in cols
        conn.close()

    def test_migrate_schema_on_legacy_evidence_table(self, tmp_path):
        """A DB created from a pre-provenance schema should pick up the new columns."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Build a pre-provenance evidence table by hand.
        conn.execute(
            """
            CREATE TABLE evidence (
                id TEXT PRIMARY KEY,
                thesis_id TEXT,
                symbol TEXT,
                agent TEXT NOT NULL,
                type TEXT DEFAULT 'neutral',
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                dimension TEXT,
                metadata TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL
            )
            """
        )
        # Other tables that the FK migration touches need to exist too.
        conn.executescript(SCHEMA)
        conn.commit()
        _migrate_schema(conn)
        cursor = conn.execute("PRAGMA table_info(evidence)")
        cols = {row[1] for row in cursor.fetchall()}
        for col in (
            "llm_call_id",
            "model_snapshot",
            "prompt_sha256",
            "input_sha256",
            "output_sha256",
        ):
            assert col in cols, f"missing column: {col}"
        # Idempotency
        _migrate_schema(conn)
        conn.close()


# --- blob path resolution ---------------------------------------------------


class TestBlobPathHelpers:
    def test_blob_path_includes_date_and_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CENTS_LLM_BLOB_DIR", str(tmp_path / "x"))
        when = datetime(2026, 5, 17)
        p = blob_path_for("abc123", when=when)
        assert "20260517" in str(p)
        assert p.name == "abc123.json.gz"
