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


@alert.command("digest")
@click.option(
    "--since", "since", default="24h",
    help="Window: 'today', ISO date, or duration ('24h', '7d'). Default 24h.",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
@click.option(
    "--quiet-if-empty", is_flag=True,
    help="Print nothing (text mode) when no alerts landed in the window — keeps scheduled-run logs clean.",
)
def alert_digest(since: str, output: str | None, quiet_if_empty: bool):
    """Compact per-type alert summary for scheduled-run logs.

    Designed to be appended to the wrapper log by a daily launchd/cron job
    so PREMISE_INVALIDATION / MODEL_DRIFT alerts stop landing silently in
    the DB. Always exits 0 — an empty digest is not an error.
    """
    from ._shared import resolve_output_format, respond_with_output

    output = resolve_output_format(output)
    since_dt = _parse_since(since)
    repo = AlertRepository()
    alerts = repo.list_all(limit=10000, since=since_dt)

    by_type: dict[str, dict] = {}
    for a in alerts:
        type_key = a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type)
        cell = by_type.setdefault(type_key, {
            "count": 0,
            "unread": 0,
            "symbols": [],
            "latest": None,
        })
        cell["count"] += 1
        if not a.read:
            cell["unread"] += 1
        if a.symbol and a.symbol not in cell["symbols"]:
            cell["symbols"].append(a.symbol)
        # list_all is created_at DESC — first row per type is the latest.
        if cell["latest"] is None:
            cell["latest"] = {
                "id": a.id,
                "symbol": a.symbol,
                "message": a.message,
                "created_at": a.created_at.isoformat(),
            }

    payload = {
        "since": since_dt.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "total": len(alerts),
        "unread_total": sum(c["unread"] for c in by_type.values()),
        "by_type": by_type,
    }

    def _print_digest() -> None:
        if not alerts and quiet_if_empty:
            return
        header = (
            f"[alert digest] {payload['generated_at'][:16]} — "
            f"{payload['total']} alert(s) since {payload['since'][:16]} "
            f"({payload['unread_total']} unread)"
        )
        click.echo(header)
        for type_key in sorted(by_type.keys()):
            cell = by_type[type_key]
            symbols = ", ".join(cell["symbols"][:6])
            if len(cell["symbols"]) > 6:
                symbols += f", +{len(cell['symbols']) - 6} more"
            click.echo(
                f"  {type_key:<22} n={cell['count']:<3} unread={cell['unread']:<3} [{symbols}]"
            )
            latest = cell["latest"]
            click.echo(
                f"    latest: {latest['created_at'][:16]} {latest['symbol']}: {latest['message']}"
            )

    respond_with_output(output, payload, _print_digest)


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
