"""Broker integration CLI commands."""

import logging

import click

from cents.db import PositionRepository
from cents.exceptions import ConfigurationError, BrokerError, APIError
from cents.models import PositionStatus

from ._shared import validate_symbol

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def broker(ctx):
    """Alpaca broker integration."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(broker_list)


@broker.command("status")
def broker_status():
    """Check broker connection and account status."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]", err=True)
        raise SystemExit(1)

    try:
        client = AlpacaClient(paper=True)
        account = client.get_account()
        click.echo("Connected to Alpaca (paper trading)")
        click.echo(f"  Buying Power: ${account['buying_power']:,.2f}")
        click.echo(f"  Cash: ${account['cash']:,.2f}")
        click.echo(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
    except (ConfigurationError, ValueError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except (BrokerError, APIError) as e:
        click.echo(f"API error: {e}", err=True)
        raise SystemExit(1)
    except (ConnectionError, TimeoutError, OSError) as e:
        click.echo(f"Connection failed: {e}", err=True)
        raise SystemExit(1)


@broker.command("list")
def broker_list():
    """List positions from broker."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]", err=True)
        raise SystemExit(1)

    try:
        client = AlpacaClient(paper=True)
        positions = client.get_positions()

        if not positions:
            click.echo("No open positions in broker.")
            return

        click.echo("Broker positions:")
        for p in positions:
            sign = "+" if p.unrealized_pl >= 0 else ""
            click.echo(
                f"  {p.symbol}: {p.qty:.0f} shares @ ${p.avg_entry_price:.2f} "
                f"| Now: ${p.current_price:.2f} | P&L: {sign}${p.unrealized_pl:.2f} ({sign}{p.unrealized_plpc:.1f}%)"
            )
    except (ConfigurationError, ValueError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except (BrokerError, APIError) as e:
        click.echo(f"API error: {e}", err=True)
        raise SystemExit(1)
    except (ConnectionError, TimeoutError, OSError) as e:
        click.echo(f"Connection failed: {e}", err=True)
        raise SystemExit(1)


@broker.command("sync")
@click.option("--thesis", "-t", "thesis_id", help="Link synced positions to thesis")
def broker_sync(thesis_id: str | None):
    """Sync positions from broker to cents."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]", err=True)
        raise SystemExit(1)

    try:
        client = AlpacaClient(paper=True)
        positions = client.get_positions()

        if not positions:
            click.echo("No positions to sync.")
            return

        repo = PositionRepository()
        synced = 0

        # Load existing positions once to avoid N+1 queries
        existing_symbols = {
            p.symbol for p in repo.list() if p.status == PositionStatus.OPEN
        }

        for bp in positions:
            # Check if already tracked
            if bp.symbol in existing_symbols:
                click.echo(f"  {bp.symbol}: already tracked, skipping")
                continue

            pos = client.to_cents_position(bp, thesis_id)
            repo.create(pos)
            synced += 1
            click.echo(f"  {bp.symbol}: synced {bp.qty:.0f} shares @ ${bp.avg_entry_price:.2f}")

        click.echo(f"\nSynced {synced} positions")
    except (ConfigurationError, ValueError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except (BrokerError, APIError) as e:
        click.echo(f"API error: {e}", err=True)
        raise SystemExit(1)
    except (ConnectionError, TimeoutError, OSError) as e:
        click.echo(f"Connection failed: {e}", err=True)
        raise SystemExit(1)


@broker.command("buy")
@click.argument("symbol")
@click.option("--qty", "-q", type=float, required=True, help="Number of shares")
@click.option("--thesis", "-t", "thesis_id", help="Link to thesis ID")
@click.confirmation_option(prompt="Are you sure you want to execute this trade?")
def broker_buy(symbol: str, qty: float, thesis_id: str | None):
    """Buy shares (paper trading).

    Example: cents broker buy NVDA --qty 10
    """
    symbol = validate_symbol(symbol)
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]", err=True)
        raise SystemExit(1)

    try:
        client = AlpacaClient(paper=True)
        result = client.submit_order(symbol, qty, "buy")
        click.echo(f"Order submitted: {result.status}")
        click.echo(f"  Order ID: {result.order_id}")
        if result.filled_avg_price:
            click.echo(f"  Filled @ ${result.filled_avg_price:.2f}")
    except (ConfigurationError, ValueError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except (BrokerError, APIError) as e:
        click.echo(f"Order failed: {e}", err=True)
        raise SystemExit(1)
    except (ConnectionError, TimeoutError, OSError) as e:
        click.echo(f"Connection failed: {e}", err=True)
        raise SystemExit(1)


@broker.command("sell")
@click.argument("symbol")
@click.option("--qty", "-q", type=float, required=True, help="Number of shares")
@click.confirmation_option(prompt="Are you sure you want to execute this trade?")
def broker_sell(symbol: str, qty: float):
    """Sell shares (paper trading).

    Example: cents broker sell NVDA --qty 10
    """
    symbol = validate_symbol(symbol)
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]", err=True)
        raise SystemExit(1)

    try:
        client = AlpacaClient(paper=True)
        result = client.submit_order(symbol, qty, "sell")
        click.echo(f"Order submitted: {result.status}")
        click.echo(f"  Order ID: {result.order_id}")
        if result.filled_avg_price:
            click.echo(f"  Filled @ ${result.filled_avg_price:.2f}")
    except (ConfigurationError, ValueError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except (BrokerError, APIError) as e:
        click.echo(f"Order failed: {e}", err=True)
        raise SystemExit(1)
    except (ConnectionError, TimeoutError, OSError) as e:
        click.echo(f"Connection failed: {e}", err=True)
        raise SystemExit(1)
