"""CLI entry point for cents."""

from datetime import date
from typing import Optional

import click

from cents.db import ThesisRepository, PositionRepository, OutcomeRepository, EvidenceRepository
from cents.models import (
    Thesis,
    ThesisStatus,
    Position,
    PositionSide,
    PositionStatus,
    Outcome,
    ThesisAccuracy,
    EvidenceType,
)
from cents.agents import AGENTS


@click.group()
@click.version_option()
def cli():
    """Cents: Agentic investing guidance."""
    pass


# --- Research command ---


@cli.command("research")
@click.argument("symbol")
@click.option("--thesis", "-t", "thesis_id", help="Thesis ID to evaluate against")
@click.option(
    "--agent",
    "-a",
    "agent_name",
    type=click.Choice(list(AGENTS.keys())),
    help="Run specific agent only",
)
@click.option("--save/--no-save", default=True, help="Save evidence to database")
def research(symbol: str, thesis_id: Optional[str], agent_name: Optional[str], save: bool):
    """Run research agents on a symbol."""
    # Get thesis if specified
    thesis = None
    if thesis_id:
        thesis_repo = ThesisRepository()
        thesis = thesis_repo.get(thesis_id)
        if thesis is None:
            click.echo(f"Thesis {thesis_id} not found.", err=True)
            raise SystemExit(1)
        click.echo(f"Evaluating against thesis: {thesis.title}\n")

    # Determine which agents to run
    if agent_name:
        agents_to_run = {agent_name: AGENTS[agent_name]}
    else:
        agents_to_run = AGENTS

    total_conviction_delta = 0.0
    all_evidence = []

    for name, agent_class in agents_to_run.items():
        click.echo(f"--- {name.upper()} ---")
        agent = agent_class()
        result = agent.research(symbol.upper(), thesis)

        click.echo(f"Summary: {result.summary}")
        click.echo(f"Conviction delta: {result.conviction_delta:+.1f}")

        if result.evidence:
            click.echo("Evidence:")
            for e in result.evidence:
                icon = {"supporting": "+", "contradicting": "-", "neutral": "~"}[e.type.value]
                click.echo(f"  [{icon}] {e.content}")

        total_conviction_delta += result.conviction_delta
        all_evidence.extend(result.evidence)
        click.echo()

    # Save evidence and update thesis if requested
    if save and all_evidence and thesis:
        evidence_repo = EvidenceRepository()
        for e in all_evidence:
            e.thesis_id = thesis.id
            evidence_repo.create(e)

        thesis_repo = ThesisRepository()
        thesis.update_conviction(total_conviction_delta)
        thesis_repo.update(thesis)

        click.echo(f"Saved {len(all_evidence)} evidence items")
        click.echo(f"Thesis conviction: {thesis.conviction:.1f}% ({total_conviction_delta:+.1f})")
    elif not thesis and all_evidence:
        click.echo(f"Generated {len(all_evidence)} evidence items (not saved - no thesis linked)")


# --- Thesis commands ---


@cli.group()
def thesis():
    """Manage investment theses."""
    pass


@thesis.command("create")
@click.argument("title")
@click.option("--hypothesis", "-h", default="", help="Detailed thesis statement")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
def thesis_create(title: str, hypothesis: str, tags: str):
    """Create a new investment thesis."""
    repo = ThesisRepository()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    t = Thesis(title=title, hypothesis=hypothesis, tags=tag_list)
    repo.create(t)
    click.echo(f"Created thesis {t.id}: {t.title}")


@thesis.command("list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["open", "closed", "invalidated"]),
    help="Filter by status",
)
def thesis_list(status: Optional[str]):
    """List theses."""
    repo = ThesisRepository()
    status_filter = ThesisStatus(status) if status else None
    theses = repo.list(status=status_filter)

    if not theses:
        click.echo("No theses found.")
        return

    for t in theses:
        status_icon = {"open": "+", "closed": "-", "invalidated": "x"}[t.status.value]
        click.echo(f"[{status_icon}] {t.id}: {t.title} (conviction: {t.conviction:.0f}%)")


@thesis.command("show")
@click.argument("thesis_id")
def thesis_show(thesis_id: str):
    """Show thesis details."""
    repo = ThesisRepository()
    t = repo.get(thesis_id)

    if t is None:
        click.echo(f"Thesis {thesis_id} not found.", err=True)
        raise SystemExit(1)

    click.echo(f"ID:         {t.id}")
    click.echo(f"Title:      {t.title}")
    click.echo(f"Status:     {t.status.value}")
    click.echo(f"Conviction: {t.conviction:.1f}%")
    if t.hypothesis:
        click.echo(f"Hypothesis: {t.hypothesis}")
    if t.tags:
        click.echo(f"Tags:       {', '.join(t.tags)}")
    click.echo(f"Created:    {t.created_at.strftime('%Y-%m-%d %H:%M')}")
    click.echo(f"Updated:    {t.updated_at.strftime('%Y-%m-%d %H:%M')}")


@thesis.command("update")
@click.argument("thesis_id")
@click.option("--conviction", "-c", type=float, help="Set conviction (0-100)")
@click.option("--status", "-s", type=click.Choice(["open", "closed", "invalidated"]))
def thesis_update(thesis_id: str, conviction: Optional[float], status: Optional[str]):
    """Update a thesis."""
    repo = ThesisRepository()
    t = repo.get(thesis_id)

    if t is None:
        click.echo(f"Thesis {thesis_id} not found.", err=True)
        raise SystemExit(1)

    if conviction is not None:
        t.conviction = max(0.0, min(100.0, conviction))
    if status:
        t.status = ThesisStatus(status)

    repo.update(t)
    click.echo(f"Updated thesis {t.id}")


# --- Position commands ---


@cli.group()
def position():
    """Manage positions."""
    pass


@position.command("open")
@click.argument("symbol")
@click.argument("size", type=float)
@click.option("--price", "-p", type=float, required=True, help="Entry price")
@click.option("--thesis", "-t", "thesis_id", help="Link to thesis ID")
@click.option("--short", is_flag=True, help="Short position (default is long)")
@click.option("--notes", "-n", default="", help="Position notes")
def position_open(
    symbol: str,
    size: float,
    price: float,
    thesis_id: Optional[str],
    short: bool,
    notes: str,
):
    """Open a new paper position."""
    repo = PositionRepository()
    side = PositionSide.SHORT if short else PositionSide.LONG
    p = Position(
        symbol=symbol.upper(),
        size=size,
        entry_price=price,
        side=side,
        thesis_id=thesis_id,
        notes=notes,
        paper=True,
    )
    repo.create(p)
    click.echo(f"Opened {side.value} position {p.id}: {size} {symbol.upper()} @ ${price:.2f}")


@position.command("close")
@click.argument("position_id")
@click.argument("exit_price", type=float)
def position_close(position_id: str, exit_price: float):
    """Close a position."""
    repo = PositionRepository()
    p = repo.get(position_id)

    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    if p.status == PositionStatus.CLOSED:
        click.echo(f"Position {position_id} is already closed.", err=True)
        raise SystemExit(1)

    p.close(exit_price)
    repo.update(p)

    pnl = p.pnl
    pnl_pct = p.pnl_pct
    sign = "+" if pnl >= 0 else ""
    click.echo(
        f"Closed position {p.id}: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
    )


@position.command("list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["open", "closed"]),
    help="Filter by status",
)
def position_list(status: Optional[str]):
    """List positions."""
    repo = PositionRepository()
    status_filter = PositionStatus(status) if status else None
    positions = repo.list(status=status_filter)

    if not positions:
        click.echo("No positions found.")
        return

    for p in positions:
        status_icon = "+" if p.status == PositionStatus.OPEN else "-"
        side_icon = "L" if p.side == PositionSide.LONG else "S"
        if p.status == PositionStatus.CLOSED and p.pnl is not None:
            pnl_str = f" P&L: {'+' if p.pnl >= 0 else ''}${p.pnl:.2f}"
        else:
            pnl_str = ""
        click.echo(
            f"[{status_icon}] {p.id}: {side_icon} {p.size:.0f} {p.symbol} @ ${p.entry_price:.2f}{pnl_str}"
        )


@position.command("show")
@click.argument("position_id")
def position_show(position_id: str):
    """Show position details."""
    repo = PositionRepository()
    p = repo.get(position_id)

    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    click.echo(f"ID:          {p.id}")
    click.echo(f"Symbol:      {p.symbol}")
    click.echo(f"Side:        {p.side.value}")
    click.echo(f"Size:        {p.size:.2f}")
    click.echo(f"Entry:       ${p.entry_price:.2f} on {p.entry_date}")
    click.echo(f"Status:      {p.status.value}")
    if p.exit_price:
        click.echo(f"Exit:        ${p.exit_price:.2f} on {p.exit_date}")
        click.echo(f"P&L:         ${p.pnl:.2f} ({p.pnl_pct:+.1f}%)")
    if p.thesis_id:
        click.echo(f"Thesis:      {p.thesis_id}")
    if p.notes:
        click.echo(f"Notes:       {p.notes}")
    click.echo(f"Paper:       {'Yes' if p.paper else 'No'}")


# --- Outcome commands ---


@cli.group()
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


if __name__ == "__main__":
    cli()
