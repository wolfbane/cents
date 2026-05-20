"""Universe management CLI commands."""

from datetime import date, timedelta
from pathlib import Path

import click

from cents.db import DelistingsRepository, UniverseRepository
from cents.factory.universe_resolver import resolve_symbols
from cents.models import Universe, UniverseSource
from cents.serialization import serialize

from ._shared import (
    default_subcommand,
    exit_with_error,
    parse_date,
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
    type=click.Choice(["static", "watchlist", "fmp_index", "screener"]),
    default="static",
)
@click.option("--symbols", help="Comma-separated symbols (for static)")
@click.option("--from-file", "from_file", type=click.Path(exists=True), help="Read symbols from file (one per line)")
@click.option("--index", help="Index key for fmp_index source (e.g. sp500)")
@click.option("--strategy", help="Screener strategy name (required when --source=screener)")
@click.option("--over", "over_universe", help="Parent universe for screener to filter")
@click.option("--limit", type=int, default=30, show_default=True, help="Max symbols a screener universe returns")
@click.option("--description", default="", help="Free-form description")
def universe_create(
    name: str,
    source: str,
    symbols: str | None,
    from_file: str | None,
    index: str | None,
    strategy: str | None,
    over_universe: str | None,
    limit: int,
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
        # cents-kme: validate against the resolver's known endpoints AT CREATE
        # TIME — previously bogus values like 'sp100' (FMP has 'sp500'/'nasdaq'/
        # 'dowjones') created a 0-symbol universe that silently failed on first
        # `refresh` or `factory run`.
        from cents.factory.universe_resolver import FMP_INDEX_ENDPOINTS
        idx_normalized = index.strip().lower()
        if idx_normalized not in FMP_INDEX_ENDPOINTS:
            exit_with_error(
                f"Unknown FMP index '{idx_normalized}'. "
                f"Supported: {', '.join(sorted(FMP_INDEX_ENDPOINTS))}"
            )
        source_config["index"] = idx_normalized

    if source_enum == UniverseSource.SCREENER:
        if not strategy:
            exit_with_error("--strategy is required when --source=screener")
        from cents.screeners import SCREENERS
        if strategy not in SCREENERS:
            exit_with_error(
                f"Unknown screener '{strategy}'. Available: {', '.join(sorted(SCREENERS))}"
            )
        source_config["strategy"] = strategy
        source_config["limit"] = limit
        if over_universe:
            if repo.get(over_universe) is None:
                exit_with_error(f"Parent universe '{over_universe}' not found.")
            source_config["over"] = over_universe

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
@click.option(
    "--as-of",
    "as_of",
    help="Resolve point-in-time membership as of YYYY-MM-DD (screener universes only).",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def universe_show(name: str, as_of: str | None, output: str | None):
    """Show a universe's details and resolved symbols.

    With ``--as-of YYYY-MM-DD``, screener universes are reconstructed at
    that point in time: the current screener output is augmented with
    symbols whose tracked delisting date is on/after the requested date
    (those were members on that day even though they aren't today).
    """
    output = resolve_output_format(output)
    repo = UniverseRepository()
    uni = repo.get(name)
    if uni is None:
        exit_with_error(f"Universe '{name}' not found.")

    asof_date = parse_date(as_of, "as-of") if as_of else None

    if asof_date is not None:
        try:
            resolved = resolve_symbols(uni, asof_date=asof_date)
        except Exception as exc:
            exit_with_error(f"Resolution failed: {exc}")
        payload = serialize(uni)
        payload["as_of"] = asof_date.isoformat()
        payload["resolved_symbols"] = resolved
        respond_with_output(
            output,
            payload,
            lambda: _print_universe_asof(uni, asof_date, resolved),
        )
        return

    respond_with_output(
        output,
        serialize(uni),
        lambda: _print_universe(uni),
    )


def _print_universe_asof(uni: Universe, asof_date: date, resolved: list[str]) -> None:
    click.echo(f"Name:        {uni.name}")
    click.echo(f"Source:      {uni.source.value}")
    click.echo(f"As-of:       {asof_date.isoformat()}")
    click.echo(f"Symbols:     {len(resolved)}")
    if resolved:
        click.echo("  " + ", ".join(resolved[:20]))
        if len(resolved) > 20:
            click.echo(f"  ... and {len(resolved) - 20} more")


def _print_universe(uni: Universe) -> None:
    click.echo(f"Name:        {uni.name}")
    click.echo(f"Source:      {uni.source.value}")
    if uni.description:
        click.echo(f"Description: {uni.description}")
    if uni.source_config:
        click.echo(f"Config:      {uni.source_config}")
    click.echo(f"Default:     {'yes' if uni.is_default else 'no'}")
    # Cached resolved-symbol count. Dynamic sources (watchlist, fmp_index,
    # screener) resolve on demand at factory run time, so an empty cache here
    # doesn't mean an empty universe — hint the user toward `refresh`.
    if uni.symbols:
        click.echo(f"Symbols:     {len(uni.symbols)}")
        click.echo("  " + ", ".join(uni.symbols[:20]))
        if len(uni.symbols) > 20:
            click.echo(f"  ... and {len(uni.symbols) - 20} more")
    elif uni.source == UniverseSource.STATIC:
        click.echo("Symbols:     0")
    else:
        click.echo(
            f"Symbols:     0 cached (resolves at run time; "
            f"`cents universe refresh {uni.name}` to populate)"
        )


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


@universe.command("ingest-delistings")
@click.option(
    "--since",
    help="Pull delistings dated on/after this YYYY-MM-DD (default: 1 year ago).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Fetch and report counts without writing to the delistings table.",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def universe_ingest_delistings(since: str | None, dry_run: bool, output: str | None):
    """Pull recent delistings from FMP and persist them.

    Stored delistings are used by ``cents universe show --as-of`` and by
    point-in-time backtests to reconstruct universes without survivorship
    bias. Without an FMP API key, the command no-ops with a clear message.
    """
    output = resolve_output_format(output)
    since_date = parse_date(since, "since") if since else (date.today() - timedelta(days=365))

    try:
        from cents.data.fmp import FMPFundamentalsProvider
        provider = FMPFundamentalsProvider()
    except Exception as exc:
        # No API key (ConfigurationError) or other init error — surface a
        # friendly note and exit cleanly. This keeps the command safe to
        # script even in environments without FMP access.
        payload = {
            "ingested": 0,
            "skipped": 0,
            "since": since_date.isoformat(),
            "reason": str(exc),
        }
        respond_with_output(
            output,
            payload,
            lambda: click.echo(f"Skipped: {exc}"),
        )
        return

    try:
        delistings = provider.get_delistings(since_date)
    except Exception as exc:
        exit_with_error(f"Fetch failed: {exc}")

    repo = DelistingsRepository()
    ingested = 0
    if not dry_run:
        for d in delistings:
            repo.upsert(d)
            ingested += 1

    payload = {
        "since": since_date.isoformat(),
        "fetched": len(delistings),
        "ingested": ingested,
        "dry_run": dry_run,
    }
    respond_with_output(
        output,
        payload,
        lambda: click.echo(
            f"Fetched {len(delistings)} delistings since {since_date.isoformat()}"
            + (f"; ingested {ingested}" if not dry_run else " (dry run, nothing written)")
        ),
    )
