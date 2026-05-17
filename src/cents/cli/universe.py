"""Universe management CLI commands."""

from pathlib import Path

import click

from cents.db import UniverseRepository
from cents.factory.universe_resolver import resolve_symbols
from cents.models import Universe, UniverseSource
from cents.serialization import serialize

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
    validate_symbol,
)


@default_subcommand("list")
def universe(ctx):
    """Manage symbol universes for the factory."""


@universe.command("create")
@click.argument("name")
@click.option(
    "--source",
    type=click.Choice(["static", "watchlist", "fmp_index"]),
    default="static",
)
@click.option("--symbols", help="Comma-separated symbols (for static)")
@click.option("--from-file", "from_file", type=click.Path(exists=True), help="Read symbols from file (one per line)")
@click.option("--index", help="Index key for fmp_index source (e.g. sp500)")
@click.option("--description", default="", help="Free-form description")
def universe_create(
    name: str,
    source: str,
    symbols: str | None,
    from_file: str | None,
    index: str | None,
    description: str,
):
    """Create a new universe."""
    repo = UniverseRepository()
    if repo.get(name) is not None:
        exit_with_error(f"Universe '{name}' already exists.")

    source_enum = UniverseSource(source)
    symbol_list: list[str] = []
    if symbols:
        symbol_list.extend(validate_symbol(s.strip()) for s in symbols.split(",") if s.strip())
    if from_file:
        for line in Path(from_file).read_text().splitlines():
            sym = line.strip()
            if sym and not sym.startswith("#"):
                symbol_list.append(validate_symbol(sym))

    source_config: dict = {}
    if source_enum == UniverseSource.FMP_INDEX:
        if not index:
            exit_with_error("--index is required when --source=fmp_index")
        source_config["index"] = index.strip().lower()

    try:
        uni = Universe(
            name=name,
            description=description,
            source=source_enum,
            source_config=source_config,
            symbols=symbol_list,
        )
    except ValueError as exc:
        exit_with_error(str(exc))

    repo.create(uni)
    click.echo(f"Created universe '{uni.name}' ({uni.source.value}, {len(uni.symbols)} symbols)")


@universe.command("list")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def universe_list(output: str | None):
    """List universes."""
    output = resolve_output_format(output)
    repo = UniverseRepository()
    universes = repo.list()

    respond_with_output(
        output,
        [serialize(u) for u in universes],
        lambda: _print_universes(universes),
    )


def _print_universes(universes: list[Universe]) -> None:
    if not universes:
        click.echo("No universes defined. Create one with: cents universe create <NAME>")
        return
    for u in universes:
        marker = " *" if u.is_default else "  "
        click.echo(
            f"{marker} {u.name} [{u.source.value}] — {len(u.symbols)} symbols"
        )


@universe.command("show")
@click.argument("name")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def universe_show(name: str, output: str | None):
    """Show a universe's details and resolved symbols."""
    output = resolve_output_format(output)
    repo = UniverseRepository()
    uni = repo.get(name)
    if uni is None:
        exit_with_error(f"Universe '{name}' not found.")

    respond_with_output(
        output,
        serialize(uni),
        lambda: _print_universe(uni),
    )


def _print_universe(uni: Universe) -> None:
    click.echo(f"Name:        {uni.name}")
    click.echo(f"Source:      {uni.source.value}")
    if uni.description:
        click.echo(f"Description: {uni.description}")
    if uni.source_config:
        click.echo(f"Config:      {uni.source_config}")
    click.echo(f"Default:     {'yes' if uni.is_default else 'no'}")
    click.echo(f"Symbols:     {len(uni.symbols)}")
    if uni.symbols:
        click.echo("  " + ", ".join(uni.symbols[:20]))
        if len(uni.symbols) > 20:
            click.echo(f"  ... and {len(uni.symbols) - 20} more")


@universe.command("refresh")
@click.argument("name")
def universe_refresh(name: str):
    """Re-resolve a universe's symbols (no-op for static unless symbols are provided)."""
    repo = UniverseRepository()
    uni = repo.get(name)
    if uni is None:
        exit_with_error(f"Universe '{name}' not found.")

    try:
        symbols = resolve_symbols(uni)
    except Exception as exc:
        exit_with_error(f"Refresh failed: {exc}")

    uni.symbols = symbols
    repo.update(uni)
    click.echo(f"Refreshed '{uni.name}' — {len(symbols)} symbols")


@universe.command("set-default")
@click.argument("name")
def universe_set_default(name: str):
    """Mark a universe as the default for `cents factory run`."""
    repo = UniverseRepository()
    result = repo.set_default(name)
    if result is None:
        exit_with_error(f"Universe '{name}' not found.")
    click.echo(f"Default universe set to '{result.name}'")


@universe.command("delete")
@click.argument("name")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def universe_delete(name: str, force: bool):
    """Delete a universe."""
    repo = UniverseRepository()
    uni = repo.get(name)
    if uni is None:
        exit_with_error(f"Universe '{name}' not found.")
    if not force:
        click.confirm(f"Delete universe '{uni.name}'?", abort=True)
    repo.delete(uni.name)
    click.echo(f"Deleted universe '{uni.name}'")
