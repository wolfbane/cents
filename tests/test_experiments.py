"""Tests for the experiment scaffold (cents-hvz)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from cents.db import ExperimentRepository, ThesisRepository
from cents.db.schema import SCHEMA
from cents.experiments import (
    compute_factory_config_sha,
    finalize_experiment,
    get_active_experiment,
    load_experiment_spec,
    register_experiment,
    status_snapshot,
)
from cents.experiments.registry import ExperimentSpecError
from cents.models import Experiment, Thesis, ThesisStatus


@pytest.fixture
def db_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("CENTS_DB_PATH", str(db_path))
    return db_path


def _yaml_spec(name: str = "exp1") -> str:
    return (
        f"experiment: {name}\n"
        'hypothesis: "LLM-arm beats random-arm by 10pp"\n'
        "primary_metric: win_rate_delta\n"
        "minimum_n_per_arm: 50\n"
        'stopping_rule: "90 days or N=50 per arm"\n'
    )


class TestSpecLoading:
    def test_load_yaml_spec(self, tmp_path: Path):
        path = tmp_path / "exp.yaml"
        path.write_text(_yaml_spec())
        spec = load_experiment_spec(path)
        assert spec["name"] == "exp1"
        assert spec["primary_metric"] == "win_rate_delta"
        assert spec["minimum_n_per_arm"] == 50

    def test_load_json_spec(self, tmp_path: Path):
        path = tmp_path / "exp.json"
        path.write_text(json.dumps({
            "name": "exp2",
            "hypothesis": "h",
            "primary_metric": "m",
            "minimum_n_per_arm": 100,
        }))
        spec = load_experiment_spec(path)
        assert spec["name"] == "exp2"

    def test_missing_required_field_raises(self, tmp_path: Path):
        path = tmp_path / "exp.yaml"
        path.write_text("name: x\nhypothesis: y\nprimary_metric: z\n")  # missing minimum_n
        with pytest.raises(ExperimentSpecError, match="missing required"):
            load_experiment_spec(path)


class TestRegistration:
    def test_register_freezes_config_sha(self, db_conn, tmp_path: Path, monkeypatch):
        # Point factory config at a known file so SHA is deterministic.
        cfg = tmp_path / "factory.toml"
        cfg.write_text("budget_usd = 10000\n")
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(cfg))

        path = tmp_path / "exp.yaml"
        path.write_text(_yaml_spec("e1"))
        exp = register_experiment(spec_path=path)

        expected_sha, _ = compute_factory_config_sha(cfg)
        assert exp.frozen_config_sha == expected_sha
        assert exp.is_active

    def test_cannot_register_duplicate_active_name(self, db_conn, tmp_path: Path):
        path = tmp_path / "exp.yaml"
        path.write_text(_yaml_spec("dup"))
        register_experiment(spec_path=path)
        with pytest.raises(ValueError, match="already active"):
            register_experiment(spec_path=path)


class TestActiveLookup:
    def test_get_active_returns_latest(self, db_conn, tmp_path: Path):
        for i, name in enumerate(("a", "b", "c")):
            path = tmp_path / f"{name}.yaml"
            path.write_text(_yaml_spec(name))
            register_experiment(spec_path=path)
        active = get_active_experiment()
        # All three are 'active' — we return the most recently started.
        assert active is not None
        assert active.name == "c"

    def test_get_active_returns_none_when_empty(self, db_conn):
        assert get_active_experiment() is None


class TestStatusSnapshot:
    def test_status_with_zero_theses(self, db_conn, tmp_path: Path):
        path = tmp_path / "e.yaml"
        path.write_text(_yaml_spec("e"))
        exp = register_experiment(spec_path=path)
        snap = status_snapshot(exp)
        assert snap["minimum_n_per_arm_reached"] is False
        assert snap["cadence_per_day"] == 0.0
        assert snap["opened_by_arm"] == {}

    def test_status_counts_by_arm(self, db_conn, tmp_path: Path):
        path = tmp_path / "e.yaml"
        path.write_text(_yaml_spec("e"))
        exp = register_experiment(spec_path=path)
        # Seed theses across both arms.
        trepo = ThesisRepository()
        for i in range(3):
            trepo.create(Thesis(
                title=f"llm-{i}", symbol=f"L{i}",
                experiment_id=exp.id, orchestrator_label="llm",
            ))
        for i in range(2):
            trepo.create(Thesis(
                title=f"rnd-{i}", symbol=f"R{i}",
                experiment_id=exp.id, orchestrator_label="random",
            ))
        snap = status_snapshot(exp)
        assert snap["opened_by_arm"] == {"llm": 3, "random": 2}
        assert snap["minimum_n_per_arm_reached"] is False  # 50 needed per arm

    def test_minimum_n_reached_flag(self, db_conn, tmp_path: Path):
        path = tmp_path / "e.yaml"
        # Small target so we can hit it in the test.
        path.write_text(
            "experiment: e\nhypothesis: h\nprimary_metric: m\nminimum_n_per_arm: 2\n"
        )
        exp = register_experiment(spec_path=path)
        trepo = ThesisRepository()
        for arm in ("llm", "random"):
            for i in range(3):
                t = Thesis(
                    title=f"{arm}-{i}", symbol=f"{arm[0].upper()}{i}",
                    experiment_id=exp.id, orchestrator_label=arm,
                )
                trepo.create(t)
                # Mark closed so it counts toward "decided" outcomes.
                t.status = ThesisStatus.CLOSED
                trepo.update(t)
        snap = status_snapshot(exp)
        assert snap["minimum_n_per_arm_reached"] is True


class TestFinalize:
    def test_finalize_locks_status_and_records_verdict(self, db_conn, tmp_path: Path):
        path = tmp_path / "e.yaml"
        path.write_text(_yaml_spec("e"))
        exp = register_experiment(spec_path=path)
        verdict = {"primary_metric_value": 0.12, "decision": "supported"}
        exp = finalize_experiment(exp, verdict=verdict)
        assert exp.status == "finalized"
        assert exp.finalized_at is not None
        assert json.loads(exp.verdict_json) == verdict


class TestCLI:
    def test_register_status_finalize_flow(self, db_conn, tmp_path: Path, monkeypatch):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        spec = tmp_path / "exp.yaml"
        spec.write_text(_yaml_spec("flow-test"))

        runner = CliRunner()
        # register
        r = runner.invoke(cli, ["experiment", "register", str(spec)])
        assert r.exit_code == 0, r.output
        assert "flow-test" in r.output

        # list
        r = runner.invoke(cli, ["experiment", "list"])
        assert r.exit_code == 0
        assert "flow-test" in r.output

        # status
        r = runner.invoke(cli, ["experiment", "status", "--output", "json"])
        assert r.exit_code == 0
        payload = json.loads(r.output)
        assert payload["name"] == "flow-test"
        assert payload["minimum_n_per_arm"] == 50

        # finalize
        verdict_path = tmp_path / "v.json"
        verdict_path.write_text(json.dumps({"primary_metric_value": 0.0}))
        r = runner.invoke(cli, [
            "experiment", "finalize", "flow-test",
            "--verdict", str(verdict_path),
        ])
        assert r.exit_code == 0, r.output
        assert "Finalized" in r.output

        # second status now reports finalized
        r = runner.invoke(cli, ["experiment", "status", "--name", "flow-test", "--output", "json"])
        payload = json.loads(r.output)
        assert payload["status"] == "finalized"

    def test_register_duplicate_active_errors(self, db_conn, tmp_path: Path, monkeypatch):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        spec = tmp_path / "exp.yaml"
        spec.write_text(_yaml_spec("dup-cli"))
        runner = CliRunner()
        r = runner.invoke(cli, ["experiment", "register", str(spec)])
        assert r.exit_code == 0
        r = runner.invoke(cli, ["experiment", "register", str(spec)])
        assert r.exit_code != 0
        assert "already active" in r.output


class TestEngineIntegration:
    def test_engine_stamps_experiment_id_on_opened_theses(self, db_conn, tmp_path: Path, monkeypatch):
        """Active experiment → opened theses carry experiment_id."""
        from unittest.mock import MagicMock
        from cents.factory.config import FactoryConfig
        from cents.factory.engine import FactoryEngine, TAG_FACTORY
        from cents.db import UniverseRepository
        from cents.models import Universe

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        # Register an active experiment.
        spec = tmp_path / "exp.yaml"
        spec.write_text(_yaml_spec("engine-test"))
        exp = register_experiment(spec_path=spec)

        # Stub orchestrator + premise + event so the engine just opens.
        monkeypatch.setattr(
            "cents.factory.engine.classify_premise_tags",
            lambda *a, **k: ([], {}),
        )
        import cents.agents
        fake_event = MagicMock()
        fake_event.refresh.return_value = {"fetched": 0, "new": 0, "alerts_fired": 0}
        monkeypatch.setattr(cents.agents, "EventAgent", lambda: fake_event)

        UniverseRepository().create(Universe(name="test", symbols=["AAPL"], is_default=True))

        orch = MagicMock()
        orch.research.return_value = type(
            "AR", (), {
                "conviction_delta": 7.0, "evidence": [],
                "summary": "x", "dimension_scores": {},
            }
        )()

        provider = MagicMock()
        provider.get_latest_price.return_value = 100.0

        cfg = FactoryConfig(
            budget_usd=10_000, target_positions=10, entry_threshold=5.0,
            cohort_mode="directional_only",
        )
        engine = FactoryEngine(config=cfg, orchestrator=orch, price_provider=provider)
        engine.run()

        theses = [t for t in ThesisRepository().list() if TAG_FACTORY in t.tags]
        assert len(theses) == 1
        assert theses[0].experiment_id == exp.id
