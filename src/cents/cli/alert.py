"""Alert management CLI commands."""

import re
from datetime import datetime, timedelta

import click

from cents.db import AlertRepository


_RELATIVE_SINCE = re.compile(r"^(\d+)([hd])$")


def _parse_since(value: str) -> datetime:
    """Parse --since: 'today', ISO date, or 'Nh'/'Nd'."""
    raw = value.strip().lower()
    if raw == "today":
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    m = _RELATIVE_SINCE.match(raw)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = timedelta(hours=n) if unit == "h" else timedelta(days=n)
        return datetime.now() - delta
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise click.BadParameter(
            f"{value!r}: expected 'today', ISO date (YYYY-MM-DD), or relative duration (e.g. '24h', '7d')"
        ) from exc


@click.group(invoke_without_command=True)
@click.pass_context
def alert(ctx):
    """Manage alerts."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(alert_list)


@alert.command("list")
@click.option("--all", "show_all", is_flag=True, help="Show all alerts including read")
@click.option("--since", "since", default=None, help="Filter alerts since: 'today', ISO date, or duration ('24h', '7d')")
def alert_list(show_all: bool, since: str | None):
    """List alerts."""
    repo = AlertRepository()
    since_dt = _parse_since(since) if since else None
    alerts = repo.list_all(since=since_dt) if show_all else repo.list_unread(since=since_dt)

    if not alerts:
        click.echo("No alerts." if show_all else "No unread alerts.")
        return

    for a in alerts:
        icon = " " if a.read else "*"
        time = a.created_at.strftime("%m-%d %H:%M")
        click.echo(f"[{icon}] {a.id} {time} {a.symbol}: {a.message}")


@alert.command("read")
@click.argument("alert_id", required=False)
@click.option("--all", "mark_all", is_flag=True, help="Mark all as read")
def alert_read(alert_id: str | None, mark_all: bool):
    """Mark alert(s) as read."""
    repo = AlertRepository()
    if mark_all:
        count = repo.mark_all_read()
        click.echo(f"Marked {count} alerts as read")
    elif alert_id:
        if repo.mark_read(alert_id):
            click.echo(f"Marked alert {alert_id} as read")
        else:
            click.echo(f"Alert {alert_id} not found.", err=True)
            raise SystemExit(1)
    else:
        click.echo("Specify alert ID or use --all", err=True)
        raise SystemExit(1)
