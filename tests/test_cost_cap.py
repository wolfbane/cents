"""Tests for the pre-flight LLM cost cap + kill switch (bead cents-ujb)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from cents import llm_usage as llm_usage_mod
from cents.cli import cli
from cents.db import LLMUsageRepository
from cents.db.schema import SCHEMA
from cents.exceptions import CostCapExceeded
from cents.llm_usage import (
    check_cost_cap,
    cost_cap,
    current_run_spend_usd,
    peek_cost_usd,
    reset_cost_cap_state,
    today_cost_usd,
)
from cents.models import LLMUsage


# --- peek_cost_usd -----------------------------------------------------------


class TestPeekCostUsd:
    def test_returns_positive_estimate_for_typical_call(self):
        call_kwargs = {
            "model": "claude-haiku-4-5",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Score this article: " + ("x" * 800)}],
        }
        cost = peek_cost_usd(call_kwargs)
        assert cost > 0
        # Sanity: ~200 input tokens (800/4) @ $1/M + 200 output @ $5/M ~ $0.0012
        assert 0.0005 < cost < 0.01

    def test_unknown_model_is_charged_conservatively(self):
        """Unknown model should still consume cap budget so a runaway is bounded."""
        call_kwargs = {
            "model": "imaginary-llm",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "x" * 4000}],
        }
        cost = peek_cost_usd(call_kwargs)
        assert cost > 0

    def test_handles_missing_fields_safely(self):
        # No model/max_tokens/messages — should still return a non-negative cost.
        cost = peek_cost_usd({})
        assert cost >= 0


# --- check_cost_cap ----------------------------------------------------------


class TestCheckCostCap:
    def setup_method(self):
        reset_cost_cap_state()

    def teardown_method(self):
        reset_cost_cap_state()

    def test_no_cap_set_allows_calls(self):
        # No per-run cap, no daily cap → no exception.
        check_cost_cap(
            {"model": "claude-haiku-4-5", "max_tokens": 100, "messages": []},
            agent="sentiment",
            operation="score_article",
        )

    def test_per_run_cap_blocks_overshoot(self):
        with cost_cap(0.0001):  # one-hundredth of a cent
            with pytest.raises(CostCapExceeded) as exc_info:
                check_cost_cap(
                    {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    agent="sentiment",
                    operation="score_article",
                )
            assert exc_info.value.cap_kind == "run"
            assert exc_info.value.cap_usd == 0.0001
            assert exc_info.value.next_call_estimate_usd > 0.0001

    def test_per_run_cap_allows_calls_under_cap(self):
        with cost_cap(10.0):
            check_cost_cap(
                {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                agent="sentiment",
                operation="score_article",
            )


# --- daily cap from config ---------------------------------------------------


class TestDailyCap:
    def setup_method(self):
        reset_cost_cap_state()

    def teardown_method(self):
        reset_cost_cap_state()

    def test_daily_cap_blocks_when_today_total_exceeds(self, monkeypatch, tmp_path):
        """If today's spend (from llm_usage) already exceeds the cap, calls block."""
        db_path = tmp_path / "data" / "cents.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        # Seed a large call from today — 1M input + 1M output @ haiku = $6.
        LLMUsageRepository(conn).create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="sentiment",
                operation="score_article",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                called_at=datetime.now(),
            )
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
        monkeypatch.setenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", "1.00")

        with pytest.raises(CostCapExceeded) as exc_info:
            check_cost_cap(
                {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                agent="sentiment",
                operation="score_article",
            )
        assert exc_info.value.cap_kind == "daily"
        assert exc_info.value.cap_usd == 1.00
        assert exc_info.value.current_usd >= 6.00

    def test_daily_cap_unset_allows_calls(self, monkeypatch):
        monkeypatch.delenv("CENTS_MAX_LLM_SPEND_USD_PER_DAY", raising=False)
        check_cost_cap(
            {"model": "claude-haiku-4-5", "max_tokens": 100, "messages": []},
            agent="sentiment",
            operation="score_article",
        )


# --- today_cost_usd ----------------------------------------------------------


class TestTodayCostUsd:
    def test_today_cost_sums_priced_rows(self, tmp_path, monkeypatch):
        db_path = tmp_path / "today.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        LLMUsageRepository(conn).create(
            LLMUsage(
                model="claude-haiku-4-5",
                agent="x",
                operation="y",
                input_tokens=1_000_000,
                output_tokens=0,
                called_at=datetime.now(),
            )
        )
        conn.commit()
        conn.close()
        monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
        # 1M input @ $1/M = $1.00
        assert today_cost_usd() == pytest.approx(1.00, rel=1e-3)


# --- record_llm_usage updates the cap accumulator ---------------------------


class TestActualCostAccumulator:
    def setup_method(self):
        reset_cost_cap_state()

    def test_record_adds_to_running_spend(self, db_conn, monkeypatch):
        monkeypatch.setattr(
            "cents.llm_usage.LLMUsageRepository", lambda: LLMUsageRepository(db_conn)
        )

        response = SimpleNamespace(
            model="claude-haiku-4-5",
            usage=SimpleNamespace(
                input_tokens=1_000_000,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )
        with cost_cap(10.0):
            row_id = llm_usage_mod.record_llm_usage(
                response, agent="sentiment", operation="score_article"
            )
            assert row_id  # cents-dzg: id is returned
            # 1M input @ $1/M = $1.00
            assert current_run_spend_usd() == pytest.approx(1.00, rel=1e-3)


# --- CLI: cents factory run --max-cost-usd ----------------------------------


class TestFactoryRunMaxCostUsd:
    def test_max_cost_aborts_early(self, tmp_path, monkeypatch):
        """`cents factory run --max-cost-usd 0.001` should abort before any expensive call."""
        db_path = tmp_path / "data" / "cents.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        monkeypatch.setenv("CENTS_DB_PATH", str(db_path))

        # Stub the FactoryEngine.run to simulate one LLM call that trips the cap.
        # NB: the `cents.cli` package re-exports the `factory` click Group under
        # the same attribute name as the submodule, so we resolve via sys.modules.
        import sys
        import cents.cli.factory  # noqa: F401 — ensure the module is registered
        factory_cli_mod = sys.modules["cents.cli.factory"]

        class _BoomEngine:
            def __init__(self, *a, **kw):
                pass

            def run(self, *, dry_run=False, universe_override=None, allow_frozen_drift=False):
                # Simulate an in-loop LLM call check that exceeds the cap.
                check_cost_cap(
                    {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": "x" * 200}],
                    },
                    agent="factory",
                    operation="classify_premise",
                )
                # If we got past the check, the cap wasn't enforced.
                raise AssertionError("cap was not enforced")

        monkeypatch.setattr(factory_cli_mod, "FactoryEngine", _BoomEngine)
        # Avoid pulling a real factory config from disk.
        from cents.factory.config import FactoryConfig

        monkeypatch.setattr(
            factory_cli_mod, "load_factory_config", lambda: FactoryConfig(universe="default")
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["factory", "run", "--max-cost-usd", "0.0000001"])
        assert result.exit_code == 1, result.output
        assert "cap" in result.output.lower()


# --- CLI: cents research --max-cost-usd -------------------------------------


class TestResearchMaxCostUsd:
    def test_research_aborts_under_cap(self, tmp_path, monkeypatch):
        """`cents research --max-cost-usd 0.0000001` aborts in the agent loop."""
        db_path = tmp_path / "data" / "cents.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        monkeypatch.setenv("CENTS_DB_PATH", str(db_path))

        # Stub the orchestrator agent so it tries to make an LLM call.
        class _CapAgent:
            def __init__(self, *a, **kw):
                pass

            def research(self, symbol, thesis=None, as_of=None):
                check_cost_cap(
                    {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 1000,
                        "messages": [{"role": "user", "content": "x" * 1000}],
                    },
                    agent="orchestrator",
                    operation="evaluate",
                )
                raise AssertionError("cap was not enforced")

        import sys
        import cents.cli.research  # noqa: F401
        research_cli_mod = sys.modules["cents.cli.research"]
        from cents import agents as agents_mod

        monkeypatch.setitem(agents_mod.AGENTS, "orchestrator", _CapAgent)
        # Suppress price provider lookup.
        monkeypatch.setattr(
            research_cli_mod, "get_price_provider", lambda: MagicMock(get_latest_price=lambda *a, **kw: None)
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["research", "NVDA", "--max-cost-usd", "0.0000001"])
        assert result.exit_code == 1, result.output
        assert "cap" in result.output.lower()
