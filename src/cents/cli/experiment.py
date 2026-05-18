"""Experiment registration CLI (cents-hvz).

Pre-registers a hypothesis + frozen factory.toml so the factory analytics
become falsifiable rather than post-hoc storytelling. See
``cents/experiments/registry.py`` for the schema.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from cents.db import ExperimentRepository
from cents.experiments import (
    ExperimentSpecError,
    finalize_experiment,
    get_active_experiment,
    load_experiment_spec,
    register_experiment,
    status_snapshot,
)
from cents.serialization import serialize

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


# The registry module raises ExperimentSpecError on bad specs; re-export so
# the CLI module can catch it.
from cents.experiments.registry import ExperimentSpecError as _SpecError  # noqa: F401


@default_subcommand("list")
def experiment(ctx):
    """Register and inspect pre-registered research experiments."""


@experiment.command("register")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def experiment_register(spec_path: Path, output: str | None):
    """Register a new experiment, freezing the current factory.toml SHA."""
    output = resolve_output_format(output)
    try:
        spec = load_experiment_spec(spec_path)
        exp = register_experiment(spec=spec)
    except ExperimentSpecError as exc:
        exit_with_error(f"Invalid experiment spec: {exc}")
    except ValueError as exc:
        exit_with_error(str(exc))

    payload = serialize(exp)
    respond_with_output(output, payload, lambda: _print_register(exp))


def _print_register(exp) -> None:
    click.echo(f"Registered experiment {exp.name!r} (id={exp.id})")
    click.echo(f"  Hypothesis:       {exp.hypothesis}")
    click.echo(f"  Primary metric:   {exp.primary_metric}")
    click.echo(f"  Min N per arm:    {exp.minimum_n_per_arm}")
    click.echo(f"  Frozen SHA:       {exp.frozen_config_sha[:12]}…")
    click.echo(f"  Started at:       {exp.started_at.isoformat()}")
    click.echo()
    click.echo(
        "  The engine will warn when ~/.cents/factory.toml drifts from the\n"
        "  frozen SHA. Treat that as a discipline violation — don't iterate\n"
        "  on parameters mid-experiment."
    )


@experiment.command("list")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def experiment_list(output: str | None):
    """List registered experiments."""
    output = resolve_output_format(output)
    repo = ExperimentRepository()
    experiments = repo.list()
    payload = [serialize(e) for e in experiments]
    respond_with_output(
        output, payload,
        lambda: _print_list(experiments),
    )


def _print_list(experiments) -> None:
    if not experiments:
        click.echo("(no experiments registered)")
        return
    for e in experiments:
        marker = "●" if e.is_active else "○"
        click.echo(f"  {marker} {e.name}  [{e.status}]  id={e.id}")
        click.echo(f"      Hypothesis: {e.hypothesis}")
        click.echo(f"      Started:    {e.started_at.isoformat()}")


@experiment.command("status")
@click.option("--name", help="Experiment name (defaults to the active one)")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def experiment_status(name: str | None, output: str | None):
    """Show progress of an active experiment against its targets."""
    output = resolve_output_format(output)
    repo = ExperimentRepository()
    if name:
        exp = repo.get_by_name(name)
        if exp is None:
            exit_with_error(f"No experiment named {name!r}")
    else:
        exp = get_active_experiment(repo=repo)
        if exp is None:
            exit_with_error(
                "No active experiment. Register one with "
                "`cents experiment register <spec.yaml>`."
            )

    snap = status_snapshot(exp)
    respond_with_output(output, snap, lambda: _print_status(snap))


def _print_status(snap: dict) -> None:
    click.echo(f"Experiment: {snap['name']}  (id={snap['experiment_id']}, status={snap['status']})")
    click.echo(f"  Hypothesis:      {snap['hypothesis']}")
    click.echo(f"  Primary metric:  {snap['primary_metric']}")
    click.echo(f"  Min N per arm:   {snap['minimum_n_per_arm']}")
    click.echo(f"  Started:         {snap['started_at']}  ({snap['elapsed_days']} days elapsed)")
    click.echo(f"  Cadence:         {snap['cadence_per_day']} closed theses/day")
    click.echo()
    click.echo("  Opened by arm:")
    for arm, n in (snap["opened_by_arm"] or {}).items():
        click.echo(f"    {arm:>8s}: {n}")
    click.echo("  Closed by arm:")
    for arm, n in (snap["closed_by_arm"] or {}).items():
        click.echo(f"    {arm:>8s}: {n}")
    click.echo()
    if snap["minimum_n_per_arm_reached"]:
        click.echo("  ✓ Minimum N reached on all arms — primary metric is now meaningful.")
    elif snap["projected_days_to_target"] is not None:
        click.echo(f"  Projected days to target N: {snap['projected_days_to_target']}")
    else:
        click.echo("  No closed theses yet — can't project time-to-target.")


@experiment.command("finalize")
@click.argument("name")
@click.option(
    "--verdict",
    "verdict_path",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with the verdict on the primary metric.",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def experiment_finalize(name: str, verdict_path: Path | None, output: str | None):
    """Finalize an experiment (lock its status and optionally record a verdict)."""
    output = resolve_output_format(output)
    repo = ExperimentRepository()
    exp = repo.get_by_name(name)
    if exp is None:
        exit_with_error(f"No experiment named {name!r}")
    if not exp.is_active:
        exit_with_error(f"Experiment {name!r} is already {exp.status}.")

    verdict: dict | None = None
    if verdict_path is not None:
        try:
            verdict = json.loads(verdict_path.read_text())
        except json.JSONDecodeError as exc:
            exit_with_error(f"Verdict file is not valid JSON: {exc}")

    exp = finalize_experiment(exp, verdict=verdict, repo=repo, now=datetime.now())
    payload = serialize(exp)
    respond_with_output(
        output, payload,
        lambda: click.echo(f"Finalized experiment {exp.name!r} at {exp.finalized_at.isoformat()}."),
    )
