"""Registry operations for pre-registered experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from cents.db import ExperimentRepository, ThesisRepository
from cents.factory.config import get_factory_config_path
from cents.models import Experiment, ThesisStatus


REQUIRED_FIELDS = ("name", "hypothesis", "primary_metric", "minimum_n_per_arm")


class ExperimentSpecError(ValueError):
    """Raised when an experiment YAML/dict is malformed."""


def compute_factory_config_sha(config_path: Path | None = None) -> tuple[str, str]:
    """Return ``(sha256, raw_text)`` of the factory config file.

    Used to freeze the config in effect at experiment registration time.
    """
    path = config_path or get_factory_config_path()
    if not path.exists():
        # An experiment can still be registered without a written config —
        # the engine's defaults are the frozen baseline.
        raw = "# (no factory.toml present — defaults in effect)"
    else:
        raw = path.read_text()
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return sha, raw


def load_experiment_spec(spec_path: Path) -> dict:
    """Load + validate an experiment spec file (YAML or JSON).

    YAML support is soft — falls back to a minimal parser when PyYAML is
    unavailable. The fixture surface stays small (top-level scalars only).
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Experiment spec not found: {spec_path}")

    raw = spec_path.read_text()
    data: dict
    suffix = spec_path.suffix.lower()
    if suffix in (".json",):
        data = json.loads(raw)
    else:
        # Try PyYAML, then fall back to a tiny parser sufficient for the
        # documented schema (top-level scalar key: value pairs).
        try:
            import yaml  # type: ignore[import-untyped]
            data = yaml.safe_load(raw) or {}
        except ImportError:
            data = _parse_simple_yaml(raw)

    if not isinstance(data, dict):
        raise ExperimentSpecError(
            f"Experiment spec must be a mapping, got {type(data).__name__}"
        )

    # Some specs use 'experiment:' as the name key; accept both.
    if "name" not in data and "experiment" in data:
        data["name"] = data["experiment"]

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ExperimentSpecError(
            f"Experiment spec missing required fields: {', '.join(missing)}"
        )

    return data


def _parse_simple_yaml(raw: str) -> dict:
    """Minimal YAML parser — top-level `key: value` pairs only.

    Sufficient for the experiment spec schema; avoids the PyYAML soft-dep.
    Strips inline ``#`` comments and trims quotes.
    """
    result: dict = {}
    for line in raw.splitlines():
        # Strip line comments and trailing whitespace.
        if "#" in line:
            line = line[: line.index("#")]
        line = line.rstrip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        # Type-coerce common primitives.
        if value.isdigit():
            result[key] = int(value)
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def register_experiment(
    spec_path: Path | None = None,
    *,
    spec: dict | None = None,
    config_path: Path | None = None,
    repo: ExperimentRepository | None = None,
) -> Experiment:
    """Register a new experiment and freeze the factory.toml SHA + body.

    Either ``spec_path`` (load from disk) or ``spec`` (pre-loaded dict)
    must be supplied. Raises ExperimentSpecError on schema violations,
    and ValueError if an experiment of the same name is already active.
    """
    if spec is None:
        if spec_path is None:
            raise ValueError("must pass either spec_path or spec")
        spec = load_experiment_spec(spec_path)

    repo = repo or ExperimentRepository()
    name = spec["name"]
    existing = repo.get_by_name(name)
    if existing is not None and existing.is_active:
        raise ValueError(
            f"Experiment {name!r} is already active "
            f"(started {existing.started_at.isoformat()}, "
            f"finalize or abandon it first)."
        )

    sha, raw = compute_factory_config_sha(config_path=config_path)
    exp = Experiment(
        name=name,
        hypothesis=spec["hypothesis"],
        primary_metric=spec["primary_metric"],
        minimum_n_per_arm=int(spec["minimum_n_per_arm"]),
        stopping_rule=str(spec.get("stopping_rule", "")),
        frozen_config_sha=sha,
        frozen_config_json=raw,
    )
    return repo.create(exp)


def get_active_experiment(
    repo: ExperimentRepository | None = None,
) -> Experiment | None:
    """Return the single active experiment, or None.

    If multiple are active, returns the most recently started one. The
    engine treats this single experiment as the registration context for
    any theses it opens.
    """
    repo = repo or ExperimentRepository()
    active = repo.list_active()
    if not active:
        return None
    return sorted(active, key=lambda e: e.started_at, reverse=True)[0]


def status_snapshot(
    exp: Experiment,
    *,
    thesis_repo: ThesisRepository | None = None,
    now: datetime | None = None,
) -> dict:
    """Return a dict summarising progress against the experiment's targets."""
    thesis_repo = thesis_repo or ThesisRepository()
    theses = thesis_repo.list()
    in_exp = [t for t in theses if getattr(t, "experiment_id", None) == exp.id]
    by_arm: dict[str, int] = {}
    closed_by_arm: dict[str, int] = {}
    for t in in_exp:
        label = t.orchestrator_label or "llm"
        by_arm[label] = by_arm.get(label, 0) + 1
        if t.status == ThesisStatus.CLOSED:
            closed_by_arm[label] = closed_by_arm.get(label, 0) + 1

    anchor = now or datetime.now()
    elapsed_days = max(0, (anchor - exp.started_at).days)
    closed_total = sum(closed_by_arm.values())
    cadence_per_day = closed_total / elapsed_days if elapsed_days > 0 else 0.0
    target_total = exp.minimum_n_per_arm * max(1, len(by_arm) or 1)
    days_to_target = (
        (target_total - closed_total) / cadence_per_day
        if cadence_per_day > 0 and closed_total < target_total
        else None
    )

    target_reached = all(
        closed_by_arm.get(arm, 0) >= exp.minimum_n_per_arm
        for arm in (by_arm.keys() or ["llm"])
    )
    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "status": exp.status,
        "hypothesis": exp.hypothesis,
        "primary_metric": exp.primary_metric,
        "minimum_n_per_arm": exp.minimum_n_per_arm,
        "started_at": exp.started_at.isoformat(),
        "elapsed_days": elapsed_days,
        "opened_by_arm": by_arm,
        "closed_by_arm": closed_by_arm,
        "cadence_per_day": round(cadence_per_day, 2),
        "projected_days_to_target": (
            round(days_to_target, 1) if days_to_target is not None else None
        ),
        "minimum_n_per_arm_reached": target_reached,
    }


def finalize_experiment(
    exp: Experiment,
    *,
    verdict: dict | None = None,
    repo: ExperimentRepository | None = None,
    now: datetime | None = None,
) -> Experiment:
    """Mark experiment finalized + optionally record a verdict dict."""
    repo = repo or ExperimentRepository()
    exp.status = "finalized"
    exp.finalized_at = now or datetime.now()
    if verdict is not None:
        exp.verdict_json = json.dumps(verdict)
    return repo.update(exp)
