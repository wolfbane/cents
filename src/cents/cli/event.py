"""Event management CLI commands."""

from datetime import date, datetime, timedelta

import click

from cents.agents import EventAgent
from cents.db import EventRepository

from ._shared import default_subcommand, exit_with_error, parse_date


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


@event.command("retag")
@click.option("--since", "since_str", default=None, help="Only retag events on/after YYYY-MM-DD")
@click.option("--until", "until_str", default=None, help="Only retag events on/before YYYY-MM-DD")
@click.option("--limit", type=int, default=None, help="Cap how many to retag")
def event_retag(since_str: str | None, until_str: str | None, limit: int | None):
    """LLM-tag events left as 'tagger_skipped' by an earlier --no-tag backfill.

    Examples:

      cents event retag --since 2026-02-20                   # last 3 months
      cents event retag --since 2026-02-20 --limit 50        # smoke test
    """
    since = parse_date(since_str, "since") if since_str else None
    until = parse_date(until_str, "until") if until_str else None
    agent = EventAgent()
    summary = agent.retag(since=since, until=until, limit=limit)
    click.echo(
        f"Retag: scanned {summary['scanned']}, tagged {summary['tagged']}, "
        f"failed {summary['failed']}."
    )


@event.command("backfill")
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_str", default=None, help="End date YYYY-MM-DD (default: today)")
@click.option(
    "--no-tag", is_flag=True, default=False,
    help="Skip LLM tagging (much faster + free; events stored with tagger_skipped status).",
)
@click.option(
    "--window-days", type=int, default=30, show_default=True,
    help="Date-window size per paginated request.",
)
def event_backfill(start_str: str, end_str: str | None, no_tag: bool, window_days: int):
    """Bulk-ingest historical Federal Register events.

    Used to populate the events table for historical agent backtests. Walks the
    date range in month-sized windows, paginating per_page=50 within each window.

    Examples:

      cents event backfill --start 2022-01-01 --end 2024-12-31         # tagged (slower, LLM cost)
      cents event backfill --start 2022-01-01 --no-tag                 # untagged (fast, free)
    """
    start_date = parse_date(start_str, "start")
    end_date = parse_date(end_str, "end") if end_str else date.today()
    agent = EventAgent()
    summary = agent.backfill(
        start_date=start_date, end_date=end_date, tag=not no_tag, window_days=window_days,
    )
    click.echo(
        f"Backfill {start_date} → {end_date}: "
        f"fetched {summary['fetched']}, new {summary['new']}, "
        f"tagged {summary['tagged']}, skipped_existing {summary['skipped_existing']}, "
        f"alerts_fired {summary['alerts_fired']}."
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
