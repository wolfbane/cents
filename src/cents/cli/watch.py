"""Watchlist management CLI commands."""

from typing import Optional

import click

from cents.db import WatchlistRepository
from cents.models import WatchlistItem

from ._shared import validate_symbol


@click.group()
def watch():
    """Manage watchlist."""
    pass


@watch.command("add")
@click.argument("symbol")
@click.option("--thesis", "-t", "thesis_id", help="Link to thesis ID")
@click.option("--notes", "-n", default="", help="Notes for this watch")
@click.option(
    "--threshold",
    type=float,
    help="Custom conviction delta threshold for this symbol",
)
@click.option("--webhook", help="Custom webhook/alert destination for this symbol")
def watch_add(
    symbol: str,
    thesis_id: Optional[str],
    notes: str,
    threshold: Optional[float],
    webhook: Optional[str],
):
    """Add or update a symbol on watchlist."""
    symbol = validate_symbol(symbol)
    repo = WatchlistRepository()
    existing = repo.get(symbol)

    if existing:
        # Update existing entry, preserving values not explicitly set
        item = WatchlistItem(
            id=existing.id,  # Keep same ID
            symbol=symbol,
            thesis_id=thesis_id if thesis_id else existing.thesis_id,
            notes=notes if notes else existing.notes,
            threshold=threshold if threshold is not None else existing.threshold,
            alert_destination=webhook if webhook else existing.alert_destination,
            last_scanned=existing.last_scanned,
            created_at=existing.created_at,
        )
        repo.add(item)  # INSERT OR REPLACE
        click.echo(f"Updated {symbol} on watchlist")
    else:
        item = WatchlistItem(
            symbol=symbol,
            thesis_id=thesis_id,
            notes=notes,
            threshold=threshold,
            alert_destination=webhook,
        )
        repo.add(item)
        click.echo(f"Added {symbol} to watchlist")


@watch.command("remove")
@click.argument("symbol")
def watch_remove(symbol: str):
    """Remove a symbol from watchlist."""
    symbol = validate_symbol(symbol)
    repo = WatchlistRepository()
    if repo.remove(symbol):
        click.echo(f"Removed {symbol} from watchlist")
    else:
        click.echo(f"{symbol} not found in watchlist.", err=True)
        raise SystemExit(1)


@watch.command("list")
def watch_list():
    """List watched symbols."""
    repo = WatchlistRepository()
    items = repo.list()

    if not items:
        click.echo("Watchlist is empty.")
        return

    for item in items:
        scanned = item.last_scanned.strftime("%Y-%m-%d %H:%M") if item.last_scanned else "never"
        thesis_str = f" (thesis: {item.thesis_id})" if item.thesis_id else ""
        extras = []
        if item.threshold is not None:
            extras.append(f"threshold: {item.threshold:.1f}")
        if item.alert_destination:
            extras.append("alert: custom")
        extras_str = f" | {'; '.join(extras)}" if extras else ""
        click.echo(f"  {item.symbol}{thesis_str} - last scanned: {scanned}{extras_str}")
