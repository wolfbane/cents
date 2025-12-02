"""Alert management CLI commands."""

from typing import Optional

import click

from cents.db import AlertRepository


@click.group()
def alert():
    """Manage alerts."""
    pass


@alert.command("list")
@click.option("--all", "show_all", is_flag=True, help="Show all alerts including read")
def alert_list(show_all: bool):
    """List alerts."""
    repo = AlertRepository()
    alerts = repo.list_all() if show_all else repo.list_unread()

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
def alert_read(alert_id: Optional[str], mark_all: bool):
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
