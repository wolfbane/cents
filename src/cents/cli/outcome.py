"""Outcome tracking CLI commands."""

import click

from cents.db import PositionRepository, OutcomeRepository
from cents.models import Outcome, PositionStatus, ThesisAccuracy


@click.group()
def outcome():
    """Track outcomes."""
    pass


@outcome.command("record")
@click.argument("position_id")
@click.option(
    "--accuracy",
    "-a",
    type=click.Choice(["correct", "incorrect", "partial", "unclear"]),
    default="unclear",
    help="Was the thesis correct?",
)
@click.option("--notes", "-n", default="", help="Retrospective notes")
def outcome_record(position_id: str, accuracy: str, notes: str):
    """Record outcome for a closed position."""
    pos_repo = PositionRepository()
    out_repo = OutcomeRepository()

    p = pos_repo.get(position_id)
    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    if p.status != PositionStatus.CLOSED:
        click.echo(f"Position {position_id} is not closed yet.", err=True)
        raise SystemExit(1)

    existing = out_repo.get_for_position(position_id)
    if existing:
        click.echo(f"Outcome already recorded for position {position_id}.", err=True)
        raise SystemExit(1)

    o = Outcome(
        position_id=position_id,
        pnl=p.pnl,
        pnl_pct=p.pnl_pct,
        thesis_accuracy=ThesisAccuracy(accuracy),
        retrospective=notes,
    )
    out_repo.create(o)
    click.echo(f"Recorded outcome {o.id} for position {position_id}")


@outcome.command("list")
def outcome_list():
    """List all recorded outcomes."""
    repo = OutcomeRepository()
    outcomes = repo.list()

    if not outcomes:
        click.echo("No outcomes recorded.")
        return

    for o in outcomes:
        sign = "+" if o.pnl >= 0 else ""
        acc = o.thesis_accuracy.value[:1].upper()
        click.echo(
            f"[{acc}] {o.id}: position {o.position_id} {sign}${o.pnl:.2f} ({sign}{o.pnl_pct:.1f}%)"
        )
