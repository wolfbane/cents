"""Factory CLI — the autonomous open/close loop over a symbol universe."""

from __future__ import annotations

from datetime import datetime, timedelta

import click

from cents.db import (
    FactoryRunRepository,
    PositionRepository,
    ThesisRepository,
    UniverseRepository,
)
from cents.factory.config import (
    get_factory_config_path,
    load_factory_config,
    scaffold_factory_config,
)
from cents.factory.engine import FactoryEngine, TAG_FACTORY
from cents.models import PositionStatus, ThesisCohort, ThesisOutcome, ThesisStatus
from cents.serialization import serialize

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


@default_subcommand("status")
def factory(ctx):
    """Run and inspect the autonomous factory loop."""


@factory.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing config")
def factory_init(force: bool):
    """Scaffold ~/.cents/factory.toml with sensible defaults."""
    try:
        path = scaffold_factory_config(force=force)
    except FileExistsError as exc:
        exit_with_error(str(exc))
    click.echo(f"Wrote factory config to {path}")


@factory.command("run")
@click.option("--dry-run", is_flag=True, help="Plan actions without mutating state")
@click.option("--universe", "universe_name", help="Universe name (defaults to config / default)")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_run(dry_run: bool, universe_name: str | None, output: str | None):
    """Run the factory engine once."""
    output = resolve_output_format(output)
    config = load_factory_config()
    engine = FactoryEngine(config=config)
    run = engine.run(dry_run=dry_run, universe_override=universe_name)

    respond_with_output(
        output,
        serialize(run),
        lambda: _print_run(run, dry_run=dry_run),
    )


def _print_run(run, *, dry_run: bool) -> None:
    label = "[dry-run] " if dry_run else ""
    click.echo(f"{label}Factory run {run.id} on universe '{run.universe_name}'")
    click.echo(f"  Theses opened:   {run.theses_opened}")
    click.echo(f"  Theses closed:   {run.theses_closed}")
    click.echo(f"  Preemptions:     {run.preemptions}")
    click.echo(f"  Positions:       {run.positions_opened} opened, {run.positions_closed} closed")
    if run.error:
        click.echo(f"  Error: {run.error}")
    proposals = run.summary_json.get("proposals", [])
    if proposals:
        click.echo("  Proposals:")
        for p in proposals:
            click.echo(f"    - {p['kind']}: {p['symbol']} ({p['detail']})")


@factory.command("status")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_status(output: str | None):
    """Summarize the factory's current state."""
    output = resolve_output_format(output)
    run_repo = FactoryRunRepository()
    thesis_repo = ThesisRepository()
    position_repo = PositionRepository()
    config = load_factory_config()

    open_theses = [t for t in thesis_repo.list(status=ThesisStatus.OPEN) if TAG_FACTORY in t.tags]
    open_positions = position_repo.list(status=PositionStatus.OPEN)
    factory_thesis_ids = {t.id for t in open_theses}
    factory_positions = [p for p in open_positions if p.thesis_id in factory_thesis_ids]
    notional = sum(p.entry_price * p.size for p in factory_positions)

    paired = sum(1 for t in open_theses if t.cohort == ThesisCohort.NEUTRAL)
    directional = len(open_theses) - paired

    latest = run_repo.latest()
    recent_runs = run_repo.list(limit=5)

    payload = {
        "config_path": str(get_factory_config_path()),
        "universe": config.universe,
        "open_theses_total": len(open_theses),
        "open_theses_directional": directional,
        "open_theses_paired": paired,
        "open_positions": len(factory_positions),
        "current_notional_usd": notional,
        "budget_usd": config.budget_usd,
        "latest_run": serialize(latest) if latest else None,
        "recent_runs": [serialize(r) for r in recent_runs],
    }

    respond_with_output(
        output,
        payload,
        lambda: _print_status(payload),
    )


def _print_status(payload: dict) -> None:
    click.echo(f"Config:        {payload['config_path']}")
    click.echo(f"Universe:      {payload['universe']}")
    click.echo(
        f"Open theses:   {payload['open_theses_total']} "
        f"(directional={payload['open_theses_directional']}, paired={payload['open_theses_paired']})"
    )
    click.echo(
        f"Notional:      ${payload['current_notional_usd']:,.2f} / "
        f"${payload['budget_usd']:,.2f}"
    )
    if payload["latest_run"]:
        run = payload["latest_run"]
        click.echo(f"Last run:      {run['id']} at {run['started_at']} (dry_run={run['dry_run']})")
    click.echo(f"Recent runs:   {len(payload['recent_runs'])}")


@factory.command("analyze")
@click.option("--since-days", type=int, default=90, help="Look-back window in days")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_analyze(since_days: int, output: str | None):
    """Outcomes stratified by cohort."""
    output = resolve_output_format(output)
    thesis_repo = ThesisRepository()
    cutoff = datetime.now() - timedelta(days=since_days)
    position_repo = PositionRepository()

    factory_theses = [t for t in thesis_repo.list() if TAG_FACTORY in t.tags]
    if cutoff:
        factory_theses = [t for t in factory_theses if t.created_at >= cutoff]

    directional = [t for t in factory_theses if t.cohort == ThesisCohort.DIRECTIONAL]
    paired = [t for t in factory_theses if t.cohort == ThesisCohort.NEUTRAL]

    payload = {
        "since_days": since_days,
        "directional": _cohort_metrics(directional, position_repo),
        "neutral": _cohort_metrics(paired, position_repo),
    }
    respond_with_output(
        output,
        payload,
        lambda: _print_analyze(payload),
    )


def _cohort_metrics(theses, position_repo) -> dict:
    opened = len(theses)
    closed = [t for t in theses if t.status == ThesisStatus.CLOSED]
    preempted = [t for t in closed if t.outcome == ThesisOutcome.PREEMPTED]
    judged = [t for t in closed if t.outcome != ThesisOutcome.PREEMPTED]
    wins = [t for t in judged if t.outcome == ThesisOutcome.CORRECT]
    win_rate = (len(wins) / len(judged)) if judged else None

    pnl_values: list[float] = []
    held_days_values: list[float] = []
    all_positions = position_repo.list()
    thesis_ids = {t.id for t in theses}
    for pos in all_positions:
        if pos.thesis_id not in thesis_ids:
            continue
        if pos.pnl is not None:
            pnl_values.append(pos.pnl)
        if pos.exit_date and pos.entry_date:
            held_days_values.append((pos.exit_date - pos.entry_date).days)

    avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else None
    avg_held_days = sum(held_days_values) / len(held_days_values) if held_days_values else None

    return {
        "opened": opened,
        "closed": len(closed),
        "preempted": len(preempted),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_held_days": avg_held_days,
    }


def _print_analyze(payload: dict) -> None:
    click.echo(f"Cohort analysis (last {payload['since_days']} days)")
    for cohort_name in ("directional", "neutral"):
        m = payload[cohort_name]
        click.echo(f"  {cohort_name}:")
        click.echo(f"    opened:    {m['opened']}")
        click.echo(f"    closed:    {m['closed']} (preempted: {m['preempted']})")
        win = "n/a" if m["win_rate"] is None else f"{m['win_rate'] * 100:.1f}%"
        avg_pnl = "n/a" if m["avg_pnl"] is None else f"${m['avg_pnl']:.2f}"
        avg_held = "n/a" if m["avg_held_days"] is None else f"{m['avg_held_days']:.1f}d"
        click.echo(f"    win_rate:  {win}")
        click.echo(f"    avg_pnl:   {avg_pnl}")
        click.echo(f"    avg_held:  {avg_held}")
