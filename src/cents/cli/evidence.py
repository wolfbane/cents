"""Evidence CLI commands."""

import json

import click

from cents.db import EvidenceRepository, ThesisRepository
from cents.serialization import serialize

from ._shared import (
    default_subcommand,
    exit_with_error,
    respond_with_output,
    validate_symbol,
)


@default_subcommand("list")
def evidence(ctx):
    """Manage research evidence."""


@evidence.command("list")
@click.argument("symbol", required=False)
@click.option("--orphans", is_flag=True, help="Only show evidence without a thesis")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def evidence_list(symbol: str | None, orphans: bool, output: str):
    """List evidence, optionally filtered by symbol."""
    repo = EvidenceRepository()

    if orphans:
        items = repo.list_orphans(symbol.upper() if symbol else None)
    elif symbol:
        items = repo.list_for_symbol(symbol.upper())
    else:
        # No filter - show all orphans as that's the most useful default
        items = repo.list_orphans()

    respond_with_output(
        output,
        [serialize(e) for e in items],
        lambda: _print_evidence_items(items, symbol, orphans),
    )


def _print_evidence_items(items, symbol: str | None, orphans: bool) -> None:
    """Render evidence list in text mode."""
    if not items:
        if orphans:
            click.echo("No orphan evidence found.")
        elif symbol:
            click.echo(f"No evidence found for {symbol.upper()}.")
        else:
            click.echo("No orphan evidence found.")
        return

    click.echo(f"Found {len(items)} evidence items:\n")
    for e in items:
        icon = {
            "supporting": "+",
            "contradicting": "-",
            "neutral": "~",
        }[e.type.value]
        symbol_str = f" [{e.symbol}]" if e.symbol else ""
        thesis_str = f" (thesis: {e.thesis_id})" if e.thesis_id else " (orphan)"
        click.echo(f"  [{icon}] {e.id}{symbol_str}{thesis_str}")
        click.echo(f"      {e.agent}: {e.content[:80]}...")
        click.echo()


@evidence.command("link")
@click.argument("symbol")
@click.option("--thesis", "-t", "thesis_id", required=True, help="Thesis ID to link to")
def evidence_link(symbol: str, thesis_id: str):
    """Link orphan evidence for a symbol to a thesis."""
    symbol = validate_symbol(symbol)

    # Verify thesis exists
    thesis_repo = ThesisRepository()
    thesis = thesis_repo.get(thesis_id)
    if thesis is None:
        exit_with_error(f"Thesis {thesis_id} not found.")

    evidence_repo = EvidenceRepository()
    count = evidence_repo.link_symbol_to_thesis(symbol, thesis_id)

    if count == 0:
        click.echo(f"No orphan evidence found for {symbol.upper()}.")
    else:
        click.echo(f"Linked {count} evidence items to thesis '{thesis.title}'")


@evidence.command("delete")
@click.argument("evidence_id")
@click.confirmation_option(prompt="Are you sure you want to delete this evidence?")
def evidence_delete(evidence_id: str):
    """Delete a specific evidence item."""
    repo = EvidenceRepository()
    if repo.delete(evidence_id):
        click.echo(f"Deleted evidence {evidence_id}")
    else:
        exit_with_error(f"Evidence {evidence_id} not found.")


@evidence.command("prune")
@click.option(
    "--retention-days",
    "-d",
    default=30,
    type=int,
    help="Days to retain evidence after thesis closure (default: 30)",
)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
def evidence_prune(retention_days: int, dry_run: bool):
    """Delete evidence for theses closed more than N days ago.

    By default, removes evidence for theses closed more than 30 days ago.
    Use --retention-days to change the threshold.
    """
    repo = EvidenceRepository()

    if dry_run:
        # Count without deleting
        from cents.db import get_connection
        conn = get_connection()
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM evidence
            WHERE thesis_id IN (
                SELECT id FROM theses
                WHERE status = 'closed'
                AND closed_at IS NOT NULL
                AND date(closed_at) < date('now', ?)
            )
            """,
            (f"-{retention_days} days",),
        )
        count = cursor.fetchone()[0]
        click.echo(f"Would delete {count} evidence items (dry run)")
    else:
        count = repo.prune_for_closed_theses(retention_days)
        if count > 0:
            click.echo(f"Pruned {count} evidence items from closed theses")
        else:
            click.echo("No evidence to prune")
