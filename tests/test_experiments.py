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

        # finalize — flow has zero opened theses, so verdict_ready is
        # False; --force is required to abandon early (cents-1qp).
        verdict_path = tmp_path / "v.json"
        verdict_path.write_text(json.dumps({"primary_metric_value": 0.0}))
        r = runner.invoke(cli, [
            "experiment", "finalize", "flow-test",
            "--verdict", str(verdict_path),
            "--force",
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


class TestVerdictReady:
    """Sample-size refusal-to-conclude (cents-1qp).

    ``verdict_ready`` gates on three discipline checks: minimum N per arm,
    a minimum elapsed-days floor (default 14), and no factory.toml SHA
    drift since registration.
    """

    def _small_n_spec(self, name: str = "vr") -> str:
        # Tiny N so we can saturate it in a test without seeding hundreds
        # of rows. Real experiments use N >= 50.
        return (
            f"experiment: {name}\nhypothesis: h\nprimary_metric: m\n"
            f"minimum_n_per_arm: 2\n"
        )

    def test_status_snapshot_includes_verdict_ready_field(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        path = tmp_path / "e.yaml"
        path.write_text(self._small_n_spec("vr-1"))
        exp = register_experiment(spec_path=path)

        snap = status_snapshot(exp)
        assert "verdict_ready" in snap
        assert "verdict_ready_reason" in snap
        assert isinstance(snap["verdict_ready"], bool)
        assert isinstance(snap["verdict_ready_reason"], str)

    def test_verdict_ready_false_below_minimum_n(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        path = tmp_path / "e.yaml"
        path.write_text(self._small_n_spec("vr-n"))
        exp = register_experiment(spec_path=path)

        # Seed a single closed thesis on llm arm; target is 2 per arm.
        trepo = ThesisRepository()
        t = Thesis(title="t", symbol="A", experiment_id=exp.id, orchestrator_label="llm")
        trepo.create(t)
        t.status = ThesisStatus.CLOSED
        trepo.update(t)

        snap = status_snapshot(exp)
        assert snap["verdict_ready"] is False
        assert "1/2" in snap["verdict_ready_reason"]
        assert "llm" in snap["verdict_ready_reason"]

    def test_verdict_ready_false_below_elapsed_days_floor(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        from cents.experiments.registry import MINIMUM_ELAPSED_DAYS

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        path = tmp_path / "e.yaml"
        path.write_text(self._small_n_spec("vr-e"))
        exp = register_experiment(spec_path=path)

        # Saturate minimum N per arm on both arms.
        trepo = ThesisRepository()
        for arm in ("llm", "random"):
            for i in range(2):
                t = Thesis(
                    title=f"{arm}-{i}", symbol=f"{arm[0].upper()}{i}",
                    experiment_id=exp.id, orchestrator_label=arm,
                )
                trepo.create(t)
                t.status = ThesisStatus.CLOSED
                trepo.update(t)

        # N is satisfied, but elapsed_days is 0 < MINIMUM_ELAPSED_DAYS.
        snap = status_snapshot(exp, now=exp.started_at + timedelta(days=1))
        assert snap["minimum_n_per_arm_reached"] is True
        assert snap["verdict_ready"] is False
        assert f"{MINIMUM_ELAPSED_DAYS}" in snap["verdict_ready_reason"]

    def test_per_experiment_minimum_calendar_days_overrides_default(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        """Spec field `minimum_calendar_days` overrides the global default."""
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))

        # Spec a 30-day floor (pilot) instead of the default 14.
        path = tmp_path / "e.yaml"
        path.write_text(
            "experiment: pilot-30\nhypothesis: h\nprimary_metric: m\n"
            "minimum_n_per_arm: 2\nminimum_calendar_days: 30\n"
        )
        exp = register_experiment(spec_path=path)
        assert exp.minimum_calendar_days == 30

        # Saturate N on both arms.
        trepo = ThesisRepository()
        for arm in ("llm", "random"):
            for i in range(2):
                t = Thesis(
                    title=f"{arm}-{i}", symbol=f"{arm[0].upper()}{i}",
                    experiment_id=exp.id, orchestrator_label=arm,
                )
                trepo.create(t)
                t.status = ThesisStatus.CLOSED
                trepo.update(t)

        # Day 15 — past the legacy 14-day default, but under the 30-day floor.
        snap = status_snapshot(exp, now=exp.started_at + timedelta(days=15))
        assert snap["verdict_ready"] is False
        assert "30" in snap["verdict_ready_reason"]
        assert snap["minimum_calendar_days"] == 30

        # Day 31 — past the per-experiment floor. Should fire.
        snap = status_snapshot(exp, now=exp.started_at + timedelta(days=31))
        assert snap["verdict_ready"] is True

    def test_verdict_ready_false_on_sha_drift(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        # Point factory config at a real file with content A.
        cfg = tmp_path / "factory.toml"
        cfg.write_text("budget_usd = 10000\n")
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(cfg))

        path = tmp_path / "e.yaml"
        path.write_text(self._small_n_spec("vr-s"))
        exp = register_experiment(spec_path=path)

        # Saturate N on both arms.
        trepo = ThesisRepository()
        for arm in ("llm", "random"):
            for i in range(2):
                t = Thesis(
                    title=f"{arm}-{i}", symbol=f"{arm[0].upper()}{i}",
                    experiment_id=exp.id, orchestrator_label=arm,
                )
                trepo.create(t)
                t.status = ThesisStatus.CLOSED
                trepo.update(t)

        # Drift the config — same path, different SHA.
        cfg.write_text("budget_usd = 99999\n# drifted\n")

        from cents.experiments.registry import MINIMUM_ELAPSED_DAYS

        # Push the clock past the elapsed-days floor so SHA drift is the
        # only remaining blocker.
        snap = status_snapshot(
            exp, now=exp.started_at + timedelta(days=MINIMUM_ELAPSED_DAYS + 1)
        )
        assert snap["config_sha_drift"] is True
        assert snap["verdict_ready"] is False
        assert "drift" in snap["verdict_ready_reason"].lower()

    def test_verdict_ready_true_when_all_gates_pass(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        cfg = tmp_path / "factory.toml"
        cfg.write_text("budget_usd = 10000\n")
        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(cfg))

        path = tmp_path / "e.yaml"
        path.write_text(self._small_n_spec("vr-ok"))
        exp = register_experiment(spec_path=path)

        trepo = ThesisRepository()
        for arm in ("llm", "random"):
            for i in range(2):
                t = Thesis(
                    title=f"{arm}-{i}", symbol=f"{arm[0].upper()}{i}",
                    experiment_id=exp.id, orchestrator_label=arm,
                )
                trepo.create(t)
                t.status = ThesisStatus.CLOSED
                trepo.update(t)

        from cents.experiments.registry import MINIMUM_ELAPSED_DAYS

        snap = status_snapshot(
            exp, now=exp.started_at + timedelta(days=MINIMUM_ELAPSED_DAYS + 1)
        )
        assert snap["verdict_ready"] is True
        assert snap["config_sha_drift"] is False


class TestFinalizeGating:
    """Finalize is gated on verdict_ready unless --force is supplied (cents-1qp)."""

    def test_finalize_without_force_errors_when_not_ready(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        from cents.cli import cli

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        spec = tmp_path / "exp.yaml"
        spec.write_text(_yaml_spec("gate-block"))
        runner = CliRunner()
        r = runner.invoke(cli, ["experiment", "register", str(spec)])
        assert r.exit_code == 0

        r = runner.invoke(cli, ["experiment", "finalize", "gate-block"])
        assert r.exit_code != 0
        assert "not verdict-ready" in r.output.lower()
        assert "--force" in r.output

    def test_finalize_force_succeeds_and_records_forced_flag(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        from cents.cli import cli
        from cents.db import ExperimentRepository

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        spec = tmp_path / "exp.yaml"
        spec.write_text(_yaml_spec("gate-force"))
        runner = CliRunner()
        r = runner.invoke(cli, ["experiment", "register", str(spec)])
        assert r.exit_code == 0

        r = runner.invoke(cli, ["experiment", "finalize", "gate-force", "--force"])
        assert r.exit_code == 0, r.output

        exp = ExperimentRepository().get_by_name("gate-force")
        assert exp is not None
        assert exp.status == "finalized"
        verdict = json.loads(exp.verdict_json)
        assert verdict["forced"] is True
        # The recorded forced_reason should explain WHY the verdict wasn't
        # ready — that's the audit trail for a discipline-violating close.
        assert "forced_reason" in verdict
        assert verdict["forced_reason"]  # non-empty

    def test_finalize_force_merges_with_verdict_file(
        self, db_conn, tmp_path: Path, monkeypatch
    ):
        from cents.cli import cli
        from cents.db import ExperimentRepository

        monkeypatch.setenv("CENTS_FACTORY_CONFIG", str(tmp_path / "factory.toml"))
        spec = tmp_path / "exp.yaml"
        spec.write_text(_yaml_spec("gate-merge"))
        runner = CliRunner()
        runner.invoke(cli, ["experiment", "register", str(spec)])

        verdict_file = tmp_path / "v.json"
        verdict_file.write_text(json.dumps({"primary_metric_value": 0.05}))
        r = runner.invoke(cli, [
            "experiment", "finalize", "gate-merge",
            "--verdict", str(verdict_file), "--force",
        ])
        assert r.exit_code == 0, r.output

        exp = ExperimentRepository().get_by_name("gate-merge")
        verdict = json.loads(exp.verdict_json)
        assert verdict["forced"] is True
        assert verdict["primary_metric_value"] == 0.05
