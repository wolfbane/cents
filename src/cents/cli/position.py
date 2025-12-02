"""Position management CLI commands."""

import json
from typing import Optional

import click

from cents.db import PositionRepository, ThesisRepository
from cents.models import Position, PositionSide, PositionStatus

from ._shared import validate_symbol, get_settings_lazy


@click.group()
def position():
    """Manage positions."""
    pass


@position.command("open")
@click.argument("symbol")
@click.option("--size", "-s", type=float, required=True, help="Number of shares")
@click.option("--price", "-p", type=float, required=True, help="Entry price")
@click.option("--thesis", "-t", "thesis_id", help="Link to thesis ID")
@click.option("--short", is_flag=True, help="Short position (default is long)")
@click.option("--notes", "-n", default="", help="Position notes")
def position_open(
    symbol: str,
    size: float,
    price: float,
    thesis_id: Optional[str],
    short: bool,
    notes: str,
):
    """Open a new paper position.

    Example: cents position open NVDA --size 10 --price 137
    """
    symbol = validate_symbol(symbol)
    repo = PositionRepository()
    side = PositionSide.SHORT if short else PositionSide.LONG
    p = Position(
        symbol=symbol,
        size=size,
        entry_price=price,
        side=side,
        thesis_id=thesis_id,
        notes=notes,
        paper=True,
    )
    repo.create(p)
    click.echo(f"Opened {side.value} position {p.id}: {size} {symbol} @ ${price:.2f}")


@position.command("close")
@click.argument("position_id")
@click.option("--price", "-p", type=float, required=True, help="Exit price")
def position_close(position_id: str, price: float):
    """Close a position.

    Example: cents position close abc123 --price 177
    """
    repo = PositionRepository()
    p = repo.get(position_id)

    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    if p.status == PositionStatus.CLOSED:
        click.echo(f"Position {position_id} is already closed.", err=True)
        raise SystemExit(1)

    p.close(price)
    repo.update(p)

    pnl = p.pnl
    pnl_pct = p.pnl_pct
    sign = "+" if pnl >= 0 else ""
    click.echo(
        f"Closed position {p.id}: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
    )


@position.command("list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["open", "closed"]),
    help="Filter by status",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def position_list(status: Optional[str], output: Optional[str]):
    """List positions."""
    if output is None:
        output = get_settings_lazy().default_output

    repo = PositionRepository()
    status_filter = PositionStatus(status) if status else None
    positions = repo.list(status=status_filter)

    if output == "json":
        result = [
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side.value,
                "size": p.size,
                "entry_price": p.entry_price,
                "entry_date": p.entry_date.isoformat(),
                "status": p.status.value,
                "exit_price": p.exit_price,
                "exit_date": p.exit_date.isoformat() if p.exit_date else None,
                "pnl": p.pnl,
                "pnl_pct": p.pnl_pct,
                "thesis_id": p.thesis_id,
                "paper": p.paper,
            }
            for p in positions
        ]
        click.echo(json.dumps(result, indent=2))
        return

    if not positions:
        click.echo("No positions found.")
        return

    for p in positions:
        status_icon = "+" if p.status == PositionStatus.OPEN else "-"
        side_icon = "L" if p.side == PositionSide.LONG else "S"
        if p.status == PositionStatus.CLOSED and p.pnl is not None:
            pnl_str = f" P&L: {'+' if p.pnl >= 0 else ''}${p.pnl:.2f}"
        else:
            pnl_str = ""
        click.echo(
            f"[{status_icon}] {p.id}: {side_icon} {p.size:.0f} {p.symbol} @ ${p.entry_price:.2f}{pnl_str}"
        )


@position.command("show")
@click.argument("position_id")
def position_show(position_id: str):
    """Show position details."""
    repo = PositionRepository()
    p = repo.get(position_id)

    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    click.echo(f"ID:          {p.id}")
    click.echo(f"Symbol:      {p.symbol}")
    click.echo(f"Side:        {p.side.value}")
    click.echo(f"Size:        {p.size:.2f}")
    click.echo(f"Entry:       ${p.entry_price:.2f} on {p.entry_date}")
    click.echo(f"Status:      {p.status.value}")
    if p.exit_price:
        click.echo(f"Exit:        ${p.exit_price:.2f} on {p.exit_date}")
        click.echo(f"P&L:         ${p.pnl:.2f} ({p.pnl_pct:+.1f}%)")
    if p.thesis_id:
        click.echo(f"Thesis:      {p.thesis_id}")
    if p.notes:
        click.echo(f"Notes:       {p.notes}")
    click.echo(f"Paper:       {'Yes' if p.paper else 'No'}")


@position.command("link")
@click.argument("position_id")
@click.option("--thesis", "-t", required=True, help="Thesis ID to link to")
def position_link(position_id: str, thesis: str):
    """Link a position to a thesis.

    Example: cents position link abc123 --thesis def456
    """
    pos_repo = PositionRepository()
    thesis_repo = ThesisRepository()

    p = pos_repo.get(position_id)
    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    t = thesis_repo.get(thesis)
    if t is None:
        click.echo(f"Thesis {thesis} not found.", err=True)
        raise SystemExit(1)

    p.thesis_id = thesis
    pos_repo.update(p)
    click.echo(f"Linked position {position_id} ({p.symbol}) to thesis {thesis}")


@position.command("value")
@click.argument("symbol", required=False)
def position_value(symbol: Optional[str]):
    """Show current market value of open positions.

    Example: cents position value        # All open positions
             cents position value AAPL   # Specific symbol only
    """
    from cents.data import get_price_provider
    from cents.exceptions import ConfigurationError

    repo = PositionRepository()
    positions = repo.list(status=PositionStatus.OPEN)

    if symbol:
        symbol = symbol.upper()
        positions = [p for p in positions if p.symbol == symbol]

    if not positions:
        if symbol:
            click.echo(f"No open positions for {symbol}.")
        else:
            click.echo("No open positions.")
        return

    # Get price provider
    try:
        provider = get_price_provider()
    except ConfigurationError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for live prices.", err=True)
        raise SystemExit(1)

    # Fetch prices for all unique symbols
    symbols = list(set(p.symbol for p in positions))
    prices: dict[str, Optional[float]] = {}
    for sym in symbols:
        try:
            prices[sym] = provider.get_latest_price(sym)
        except Exception as e:
            click.echo(f"Warning: Could not fetch price for {sym}: {e}", err=True)
            prices[sym] = None

    # Display positions with current values
    total_cost = 0.0
    total_value = 0.0
    total_pnl = 0.0

    click.echo("")
    click.echo(f"{'Symbol':<8} {'Side':<6} {'Shares':>8} {'Entry':>10} {'Current':>10} {'Value':>12} {'P&L':>12} {'%':>8}")
    click.echo("-" * 82)

    for p in positions:
        current = prices.get(p.symbol)
        side_str = "LONG" if p.side == PositionSide.LONG else "SHORT"

        if current is None:
            click.echo(f"{p.symbol:<8} {side_str:<6} {p.size:>8.0f} ${p.entry_price:>9.2f} {'N/A':>10} {'N/A':>12} {'N/A':>12} {'N/A':>8}")
            continue

        value = p.market_value(current)
        pnl = p.unrealized_pnl(current)
        pnl_pct = p.unrealized_pnl_pct(current)
        cost = p.cost_basis

        total_cost += cost
        total_value += value
        total_pnl += pnl

        pnl_sign = "+" if pnl >= 0 else ""
        pct_sign = "+" if pnl_pct >= 0 else ""

        click.echo(
            f"{p.symbol:<8} {side_str:<6} {p.size:>8.0f} "
            f"${p.entry_price:>9.2f} ${current:>9.2f} "
            f"${value:>11.2f} {pnl_sign}${pnl:>10.2f} {pct_sign}{pnl_pct:>6.1f}%"
        )

    # Summary
    if len(positions) > 1 and total_cost > 0:
        click.echo("-" * 82)
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        pnl_sign = "+" if total_pnl >= 0 else ""
        pct_sign = "+" if total_pnl_pct >= 0 else ""
        click.echo(
            f"{'TOTAL':<8} {'':<6} {'':<8} "
            f"{'':>11} {'':>11} "
            f"${total_value:>11.2f} {pnl_sign}${total_pnl:>10.2f} {pct_sign}{total_pnl_pct:>6.1f}%"
        )
    click.echo("")
