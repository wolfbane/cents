"""Event management CLI commands."""

from datetime import datetime, timedelta

import click

from cents.agents import EventAgent
from cents.db import EventRepository

from ._shared import default_subcommand, exit_with_error


@default_subcommand("list")
def event(ctx):
    """Manage policy/macro events."""


@event.command("refresh")
@click.option(
    "--lookback-days",
    type=int,
    default=None,
    help="Force a fixed lookback window (default: incremental since last fetch).",
)
def event_refresh(lookback_days: int | None):
    """Fetch new events and fire premise-invalidation alerts."""
    agent = EventAgent()
    summary = agent.refresh(lookback_days=lookback_days)
    if "error" in summary:
        exit_with_error(f"Refresh failed: {summary['error']}")
    click.echo(
        f"Fetched {summary['fetched']} documents; "
        f"stored {summary['new']} new event(s); "
        f"fired {summary['alerts_fired']} premise alert(s)."
    )


@event.command("list")
@click.option("--tag", multiple=True, help="Filter by tag (repeatable).")
@click.option(
    "--since-days",
    type=int,
    default=30,
    help="Window in days (default: 30).",
)
@click.option("--limit", type=int, default=20, show_default=True)
def event_list(tag: tuple[str, ...], since_days: int, limit: int):
    """List recently ingested events."""
    repo = EventRepository()
    since = datetime.now() - timedelta(days=since_days)
    events = repo.list_recent(since=since, tags=list(tag) or None, limit=limit)
    if not events:
        click.echo("No events in window.")
        return
    for e in events:
        when = e.occurred_at.strftime("%Y-%m-%d")
        tag_str = ",".join(e.tags) if e.tags else "-"
        click.echo(
            f"{e.id}  {when}  {e.polarity.value:<8}  [{tag_str}]  {e.title[:80]}"
        )


@event.command("show")
@click.argument("event_id")
def event_show(event_id: str):
    """Show full detail of a single event."""
    repo = EventRepository()
    e = repo.get(event_id)
    if not e:
        exit_with_error(f"Event {event_id} not found.")
    click.echo(f"ID:           {e.id}")
    click.echo(f"Source:       {e.source} ({e.source_id})")
    click.echo(f"Type:         {e.event_type}")
    click.echo(f"Occurred:     {e.occurred_at.isoformat()}")
    click.echo(f"Title:        {e.title}")
    click.echo(f"Polarity:     {e.polarity.value} (confidence {e.confidence:.2f})")
    click.echo(f"Tags:         {', '.join(e.tags) if e.tags else '-'}")
    click.echo(f"Sectors:      {', '.join(e.affected_sectors) if e.affected_sectors else '-'}")
    click.echo(f"URL:          {e.url or '-'}")
    if e.summary:
        click.echo(f"\nSummary:\n{e.summary}")
