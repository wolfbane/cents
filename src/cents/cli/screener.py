"""Screener CLI — inspect and preview discovery strategies."""

from __future__ import annotations

import click

from cents.db import UniverseRepository
from cents.factory.universe_resolver import resolve_symbols
from cents.models import Universe, UniverseSource
from cents.screeners import SCREENERS, get_screener

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


@default_subcommand("list")
def screener(ctx):
    """Inspect and preview screener-based discovery strategies."""


@screener.command("list")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def screener_list(output: str | None):
    """List registered screener strategies."""
    output = resolve_output_format(output)
    rows = sorted(
        ({"name": name, **screener.describe()} for name, screener in SCREENERS.items()),
        key=lambda r: r["name"],
    )

    respond_with_output(
        output,
        rows,
        lambda: _print_screeners(rows),
    )


def _print_screeners(rows: list[dict]) -> None:
    if not rows:
        click.echo("No screeners registered.")
        return
    for row in rows:
        click.echo(f"  {row['name']:<18} {row['description']}")


@screener.command("describe")
@click.argument("strategy")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def screener_describe(strategy: str, output: str | None):
    """Show the rule set for a screener strategy."""
    output = resolve_output_format(output)
    try:
        s = get_screener(strategy)
    except KeyError as exc:
        exit_with_error(str(exc))
    payload = {"name": s.name, **s.describe()}

    def _print():
        click.echo(f"Screener: {payload['name']}")
        click.echo(f"  {payload['description']}")
        for rule in payload.get("rules", []):
            click.echo(f"   - {rule}")

    respond_with_output(output, payload, _print)


@screener.command("preview")
@click.argument("strategy")
@click.option("--over", "over_universe", help="Parent universe to screen over")
@click.option("--limit", type=int, default=20, show_default=True, help="Max symbols to return")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def screener_preview(
    strategy: str,
    over_universe: str | None,
    limit: int,
    output: str | None,
):
    """Dry-run a screener and print the symbols it would return."""
    output = resolve_output_format(output)
    try:
        get_screener(strategy)
    except KeyError as exc:
        exit_with_error(str(exc))

    # Resolve through the universe path so the same gating + parent resolution
    # used by the factory applies here.
    source_config: dict = {"strategy": strategy, "limit": limit}
    if over_universe:
        if UniverseRepository().get(over_universe) is None:
            exit_with_error(f"Parent universe '{over_universe}' not found.")
        source_config["over"] = over_universe

    transient = Universe(
        name=f"preview-{strategy}",
        source=UniverseSource.SCREENER,
        source_config=source_config,
    )
    try:
        symbols = resolve_symbols(transient)
    except Exception as exc:
        exit_with_error(f"Preview failed: {exc}")

    payload = {
        "strategy": strategy,
        "over": over_universe,
        "limit": limit,
        "count": len(symbols),
        "symbols": symbols,
    }

    def _print():
        click.echo(f"{strategy}: {len(symbols)} symbol(s)" + (f" over '{over_universe}'" if over_universe else ""))
        if symbols:
            click.echo("  " + ", ".join(symbols))

    respond_with_output(output, payload, _print)
