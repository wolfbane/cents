"""Registry operations for pre-registered experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
from pathlib import Path

from cents.db import ExperimentRepository, ThesisRepository, UniverseRepository
from cents.factory.config import get_factory_config_path, load_factory_config
from cents.factory.universe_resolver import resolve_symbols
from cents.models import Experiment, ThesisStatus, UniverseSource


REQUIRED_FIELDS = ("name", "hypothesis", "primary_metric", "minimum_n_per_arm")

# Default calendar-day floor (cents-1qp). Per-experiment overridable via the
# `minimum_calendar_days` spec field — pilots use 30, full runs use 90.
# Kept short by default so it doesn't fight against legitimate small-N
# hypothesis tests where 14 days is enough.
DEFAULT_MINIMUM_CALENDAR_DAYS = 14
# Back-compat alias for tests that imported the original constant name.
MINIMUM_ELAPSED_DAYS = DEFAULT_MINIMUM_CALENDAR_DAYS


class ExperimentSpecError(ValueError):
    """Raised when an experiment YAML/dict is malformed."""


def _gather_behavioural_inputs(cfg) -> dict:
    """Return the full set of inputs that determine engine behaviour.

    Pre-Batch-J the experiment SHA hashed only ``FactoryConfig``. That left
    several real behaviour-shifters invisible to the audit field: a Haiku
    snapshot rollover, a one-line prompt-template edit, a screener-config
    change, or an addition to ``EVENT_TAGS`` could all change what the
    pipeline does mid-experiment without budging the SHA. Hashing all of
    them as a single canonical payload makes the audit honest.
    """
    from cents.llm_models import HAIKU_TAGGING
    from cents.models import EVENT_TAGS
    from cents.agents.sentiment import _SYSTEM_PROMPT as SENTIMENT_PROMPT
    from cents.agents.event import _SYSTEM_PROMPT as EVENT_PROMPT
    from cents.factory.premise import _SYSTEM_PROMPT as PREMISE_PROMPT

    payload: dict = {
        "factory_config": dataclasses.asdict(cfg),
        "model_snapshot": HAIKU_TAGGING,
        "prompt_sha256": {
            "sentiment": hashlib.sha256(SENTIMENT_PROMPT.encode("utf-8")).hexdigest(),
            "event": hashlib.sha256(EVENT_PROMPT.encode("utf-8")).hexdigest(),
            "premise": hashlib.sha256(PREMISE_PROMPT.encode("utf-8")).hexdigest(),
        },
        "event_tags": sorted(EVENT_TAGS),
    }
    # Universe screener config (when the resolved universe is screener-sourced).
    try:
        from cents.db import UniverseRepository
        urepo = UniverseRepository()
        universe = urepo.get_default() if cfg.universe == "default" else urepo.get_by_name(cfg.universe)
        if universe is not None:
            payload["universe"] = {
                "name": universe.name,
                "source": universe.source.value if hasattr(universe.source, "value") else str(universe.source),
                "source_config": universe.source_config,
            }
    except Exception:  # noqa: BLE001 — best-effort
        payload["universe"] = None
    return payload


def compute_factory_config_sha(config_path: Path | None = None) -> tuple[str, str]:
    """Return ``(sha256, raw_text)`` capturing the *effective* factory config
    AND adjacent behaviour-shifters (model snapshot, prompt templates,
    screener config, EVENT_TAGS vocabulary).

    The audit field is the SHA of the whole behavioural payload, not just
    the toml file text. Hand-editing the toml, rolling a model snapshot,
    edting a prompt, or adding a vocabulary tag all bump the SHA — keeping
    the experiment-registry "frozen" claim honest.

    ``raw_text`` is kept (toml when present, JSON snapshot otherwise) so
    the experiments table can show a human-readable view of what was
    registered.
    """
    path = config_path or get_factory_config_path()
    cfg = load_factory_config(path) if path.exists() else load_factory_config()
    payload = _gather_behavioural_inputs(cfg)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if path.exists():
        raw = path.read_text()
    else:
        raw = (
            "# (no factory.toml present — defaults in effect)\n"
            f"# behavioural_payload_sha256 = {sha}\n"
        )
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
    # Resolve and stamp the universe member list. Without this, SCREENER
    # universes drift daily (FMP TTM is daily_key-cached) and cohorts at
    # week 4 are over a different population than cohorts at week 1, which
    # confounds any between-arm comparison.
    frozen_universe_json = _resolve_frozen_universe(config_path=config_path)
    exp = Experiment(
        name=name,
        hypothesis=spec["hypothesis"],
        primary_metric=spec["primary_metric"],
        minimum_n_per_arm=int(spec["minimum_n_per_arm"]),
        stopping_rule=str(spec.get("stopping_rule", "")),
        minimum_calendar_days=int(
            spec.get("minimum_calendar_days", DEFAULT_MINIMUM_CALENDAR_DAYS)
        ),
        frozen_config_sha=sha,
        frozen_config_json=raw,
        frozen_universe_json=frozen_universe_json,
    )
    return repo.create(exp)


def _resolve_frozen_universe(config_path: Path | None = None) -> str:
    """Resolve the universe the experiment will run against and return a
    JSON-serialised symbol list.

    Returns an empty string when resolution fails or the universe doesn't
    exist — engine falls back to live resolution in that case. We do NOT
    raise on a missing universe because experiments can legitimately be
    registered before the universe is created (test fixtures, etc.).
    """
    try:
        cfg = load_factory_config(config_path) if config_path is None else load_factory_config(config_path)
        universe_name = cfg.universe
        urepo = UniverseRepository()
        if universe_name == "default":
            universe = urepo.get_default()
        else:
            universe = urepo.get_by_name(universe_name)
        if universe is None:
            return ""
        symbols = resolve_symbols(universe)
        return json.dumps(sorted(set(symbols)))
    except Exception:  # noqa: BLE001 — universe freeze is best-effort
        return ""


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
    config_path: Path | None = None,
) -> dict:
    """Return a dict summarising progress against the experiment's targets.

    The snapshot includes a ``verdict_ready`` flag (cents-1qp) — True only
    when the experiment has enough N per arm AND has been running long
    enough to call AND the factory.toml SHA hasn't drifted from the frozen
    registration-time SHA. ``verdict_ready_reason`` explains why.
    """
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

    # SHA drift: compute the SHA of the current factory.toml and compare
    # to the SHA frozen at registration time.
    current_sha, _ = compute_factory_config_sha(config_path=config_path)
    config_sha_drift = current_sha != exp.frozen_config_sha

    verdict_ready, verdict_ready_reason = _evaluate_verdict_ready(
        target_reached=target_reached,
        closed_by_arm=closed_by_arm,
        by_arm=by_arm,
        minimum_n_per_arm=exp.minimum_n_per_arm,
        elapsed_days=elapsed_days,
        minimum_calendar_days=exp.minimum_calendar_days,
        config_sha_drift=config_sha_drift,
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
        "minimum_elapsed_days": exp.minimum_calendar_days,
        "minimum_calendar_days": exp.minimum_calendar_days,
        "opened_by_arm": by_arm,
        "closed_by_arm": closed_by_arm,
        "cadence_per_day": round(cadence_per_day, 2),
        "projected_days_to_target": (
            round(days_to_target, 1) if days_to_target is not None else None
        ),
        "minimum_n_per_arm_reached": target_reached,
        "frozen_config_sha": exp.frozen_config_sha,
        "current_config_sha": current_sha,
        "config_sha_drift": config_sha_drift,
        "verdict_ready": verdict_ready,
        "verdict_ready_reason": verdict_ready_reason,
    }


def _evaluate_verdict_ready(
    *,
    target_reached: bool,
    closed_by_arm: dict[str, int],
    by_arm: dict[str, int],
    minimum_n_per_arm: int,
    elapsed_days: int,
    minimum_calendar_days: int,
    config_sha_drift: bool,
) -> tuple[bool, str]:
    """Return ``(verdict_ready, reason)`` for a status snapshot (cents-1qp).

    Verdict is only "ready" when all three discipline gates pass:
      1. Minimum N per arm reached on every arm with theses
      2. At least ``minimum_calendar_days`` have passed since registration
      3. No SHA drift from the frozen factory.toml

    Reason is human-readable and points at the next blocker.
    """
    if not target_reached:
        # Find the most-behind arm to make the message actionable.
        arms = by_arm.keys() or ["llm"]
        worst = min(
            arms,
            key=lambda a: closed_by_arm.get(a, 0),
        )
        n = closed_by_arm.get(worst, 0)
        return (
            False,
            f"n={n}/{minimum_n_per_arm} on {worst} arm; "
            f"reach {minimum_n_per_arm} to enable.",
        )
    if elapsed_days < minimum_calendar_days:
        return (
            False,
            f"only {elapsed_days} days elapsed; "
            f"wait at least {minimum_calendar_days} since registration.",
        )
    if config_sha_drift:
        return (
            False,
            "factory.toml SHA has drifted since registration — "
            "this is a discipline violation and invalidates the verdict.",
        )
    return True, "minimum N reached, elapsed-day floor cleared, no SHA drift."


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
