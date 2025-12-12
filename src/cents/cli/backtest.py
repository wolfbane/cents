"""Backtest CLI commands."""

import json
import logging
from datetime import date, datetime, timedelta

import click

from cents.agents import AGENTS
from cents.db import BacktestRepository
from cents.models import Backtest, BacktestSignal

from ._shared import validate_symbol

logger = logging.getLogger(__name__)

# Individual agents (exclude orchestrator for backtesting)
BACKTEST_AGENTS = {k: v for k, v in AGENTS.items() if k != "orchestrator"}


def _generate_dates(start: date, end: date, interval: str) -> list[date]:
    """Generate evaluation dates between start and end."""
    dates = []
    current = start

    if interval == "daily":
        delta = timedelta(days=1)
    elif interval == "weekly":
        delta = timedelta(weeks=1)
    elif interval == "monthly":
        delta = timedelta(days=30)  # Approximate
    else:
        delta = timedelta(weeks=1)  # Default to weekly

    while current <= end:
        dates.append(current)
        current += delta

    return dates


def _calculate_forward_returns(
    symbol: str, signal_date: date, provider
) -> dict[str, float]:
    """Calculate forward returns from signal date."""
    returns = {}

    # Get price history starting from signal date
    try:
        # Get enough history to cover our forward windows
        history = provider.get_history(symbol, days=100, as_of=signal_date + timedelta(days=70))
        if not history.bars:
            return returns

        # Find the signal date price (or closest prior)
        signal_price = None
        for bar in history.bars:
            bar_date = bar.timestamp.date() if hasattr(bar.timestamp, 'date') else bar.timestamp
            if bar_date <= signal_date:
                signal_price = bar.close
            else:
                break

        if signal_price is None:
            return returns

        # Calculate returns at different horizons
        horizons = {"5d": 5, "20d": 20, "60d": 60}

        for label, days in horizons.items():
            target_date = signal_date + timedelta(days=days)
            # Find price at or after target date
            for bar in history.bars:
                bar_date = bar.timestamp.date() if hasattr(bar.timestamp, 'date') else bar.timestamp
                if bar_date >= target_date:
                    returns[label] = (bar.close - signal_price) / signal_price
                    break

    except Exception as e:
        logger.debug("Could not calculate forward returns: %s", e)

    return returns


@click.group("backtest")
def backtest():
    """Run and analyze agent backtests."""
    pass


@backtest.command("run")
@click.argument("symbol")
@click.option(
    "--start", "-s", "start_str", required=True,
    help="Start date (YYYY-MM-DD)"
)
@click.option(
    "--end", "-e", "end_str", default=None,
    help="End date (YYYY-MM-DD, default: 60 days ago)"
)
@click.option(
    "--interval", "-i",
    type=click.Choice(["daily", "weekly", "monthly"]),
    default="weekly",
    help="Evaluation interval"
)
@click.option(
    "--agents", "-a", "agent_names",
    default=None,
    help="Comma-separated list of agents (default: all)"
)
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def run_backtest(
    symbol: str,
    start_str: str,
    end_str: str | None,
    interval: str,
    agent_names: str | None,
    output: str,
):
    """Run a backtest for a symbol over a date range.

    Example: cents backtest run NVDA --start 2023-01-01 --end 2024-01-01
    """
    symbol = validate_symbol(symbol)

    # Parse dates
    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid start date: {start_str}. Use YYYY-MM-DD.", err=True)
        raise SystemExit(1)

    # Default end: 60 days ago (need forward returns)
    if end_str:
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
        except ValueError:
            click.echo(f"Invalid end date: {end_str}. Use YYYY-MM-DD.", err=True)
            raise SystemExit(1)
    else:
        end_date = date.today() - timedelta(days=60)

    if start_date >= end_date:
        click.echo("Start date must be before end date.", err=True)
        raise SystemExit(1)

    # Parse agents
    if agent_names:
        selected_agents = {}
        for name in agent_names.split(","):
            name = name.strip()
            if name not in BACKTEST_AGENTS:
                click.echo(f"Unknown agent: {name}. Available: {', '.join(BACKTEST_AGENTS.keys())}", err=True)
                raise SystemExit(1)
            selected_agents[name] = BACKTEST_AGENTS[name]
    else:
        selected_agents = BACKTEST_AGENTS

    # Initialize price provider
    try:
        from cents.data.alpaca import get_price_provider
        provider = get_price_provider()
    except Exception as e:
        click.echo(f"Could not initialize price provider: {e}", err=True)
        raise SystemExit(1)

    # Create backtest record
    repo = BacktestRepository()
    bt = Backtest(symbol=symbol, start_date=start_date, end_date=end_date)
    repo.create(bt)

    if output == "text":
        click.echo(f"Backtest {bt.id}: {symbol} from {start_date} to {end_date}")
        click.echo(f"Interval: {interval}, Agents: {', '.join(selected_agents.keys())}")
        click.echo()

    # Generate evaluation dates
    eval_dates = _generate_dates(start_date, end_date, interval)

    if output == "text":
        click.echo(f"Running {len(eval_dates)} evaluations...")

    signal_count = 0
    errors = []

    for i, eval_date in enumerate(eval_dates):
        if output == "text":
            click.echo(f"  [{i+1}/{len(eval_dates)}] {eval_date}...", nl=False)

        date_signals = 0
        for agent_name, agent_class in selected_agents.items():
            try:
                agent = agent_class()
                result = agent.research(symbol, thesis=None, as_of=eval_date)

                # Calculate forward returns
                forward_returns = _calculate_forward_returns(symbol, eval_date, provider)

                # Store signal
                signal = BacktestSignal(
                    backtest_id=bt.id,
                    date=eval_date,
                    agent_name=agent_name,
                    conviction_delta=result.conviction_delta,
                    dimension_scores=result.dimension_scores,
                    forward_returns=forward_returns,
                )
                repo.add_signal(signal)
                signal_count += 1
                date_signals += 1

            except Exception as e:
                errors.append(f"{eval_date} {agent_name}: {e}")
                logger.debug("Agent error: %s %s: %s", eval_date, agent_name, e)

        if output == "text":
            click.echo(f" {date_signals} signals")

    if output == "text":
        click.echo()
        click.echo(f"Completed: {signal_count} signals recorded")
        if errors:
            click.echo(f"Errors: {len(errors)}")
        click.echo(f"\nView results: cents backtest show {bt.id}")
    else:
        click.echo(json.dumps({
            "backtest_id": bt.id,
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "interval": interval,
            "signal_count": signal_count,
            "error_count": len(errors),
        }, indent=2))


@backtest.command("list")
@click.option("--symbol", "-s", default=None, help="Filter by symbol")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def list_backtests(symbol: str | None, output: str):
    """List all backtests."""
    repo = BacktestRepository()
    backtests = repo.list(symbol=symbol.upper() if symbol else None)

    if output == "json":
        click.echo(json.dumps([
            {
                "id": bt.id,
                "symbol": bt.symbol,
                "start_date": bt.start_date.isoformat(),
                "end_date": bt.end_date.isoformat(),
                "created_at": bt.created_at.isoformat(),
            }
            for bt in backtests
        ], indent=2))
    else:
        if not backtests:
            click.echo("No backtests found.")
            return

        click.echo(f"{'ID':<10} {'Symbol':<8} {'Start':<12} {'End':<12} {'Created'}")
        click.echo("-" * 60)
        for bt in backtests:
            click.echo(f"{bt.id:<10} {bt.symbol:<8} {bt.start_date} {bt.end_date} {bt.created_at.strftime('%Y-%m-%d')}")


@backtest.command("show")
@click.argument("backtest_id")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def show_backtest(backtest_id: str, output: str):
    """Show backtest details and signals."""
    repo = BacktestRepository()
    bt = repo.get(backtest_id)

    if bt is None:
        click.echo(f"Backtest {backtest_id} not found.", err=True)
        raise SystemExit(1)

    signals = repo.get_signals(backtest_id)

    if output == "json":
        click.echo(json.dumps({
            "id": bt.id,
            "symbol": bt.symbol,
            "start_date": bt.start_date.isoformat(),
            "end_date": bt.end_date.isoformat(),
            "created_at": bt.created_at.isoformat(),
            "signals": [
                {
                    "date": s.date.isoformat(),
                    "agent_name": s.agent_name,
                    "conviction_delta": s.conviction_delta,
                    "dimension_scores": s.dimension_scores,
                    "forward_returns": s.forward_returns,
                }
                for s in signals
            ],
        }, indent=2))
    else:
        click.echo(f"Backtest {bt.id}")
        click.echo(f"  Symbol: {bt.symbol}")
        click.echo(f"  Period: {bt.start_date} to {bt.end_date}")
        click.echo(f"  Created: {bt.created_at.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"  Signals: {len(signals)}")
        click.echo()

        if signals:
            # Group by agent for summary
            agent_stats: dict[str, list[tuple[float, dict]]] = {}
            for s in signals:
                if s.agent_name not in agent_stats:
                    agent_stats[s.agent_name] = []
                agent_stats[s.agent_name].append((s.conviction_delta, s.forward_returns))

            click.echo("Agent Summary:")
            click.echo(f"  {'Agent':<14} {'Signals':<8} {'Avg Delta':<10} {'Correlation (20d)'}")
            click.echo("  " + "-" * 55)

            for agent_name, data in sorted(agent_stats.items()):
                deltas = [d[0] for d in data]
                returns_20d = [d[1].get("20d", 0) for d in data if "20d" in d[1]]

                avg_delta = sum(deltas) / len(deltas) if deltas else 0

                # Simple correlation calculation
                corr_str = "N/A"
                if len(returns_20d) >= 3:
                    deltas_with_ret = [d[0] for d in data if "20d" in d[1]]
                    if deltas_with_ret:
                        mean_d = sum(deltas_with_ret) / len(deltas_with_ret)
                        mean_r = sum(returns_20d) / len(returns_20d)

                        num = sum((d - mean_d) * (r - mean_r) for d, r in zip(deltas_with_ret, returns_20d))
                        den_d = sum((d - mean_d) ** 2 for d in deltas_with_ret) ** 0.5
                        den_r = sum((r - mean_r) ** 2 for r in returns_20d) ** 0.5

                        if den_d > 0 and den_r > 0:
                            corr = num / (den_d * den_r)
                            corr_str = f"{corr:+.2f}"

                click.echo(f"  {agent_name:<14} {len(data):<8} {avg_delta:+.2f}      {corr_str}")


@backtest.command("delete")
@click.argument("backtest_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete_backtest(backtest_id: str, yes: bool):
    """Delete a backtest and its signals."""
    repo = BacktestRepository()
    bt = repo.get(backtest_id)

    if bt is None:
        click.echo(f"Backtest {backtest_id} not found.", err=True)
        raise SystemExit(1)

    if not yes:
        signals = repo.get_signals(backtest_id)
        click.echo(f"Delete backtest {bt.id} ({bt.symbol}, {len(signals)} signals)?")
        if not click.confirm("Continue?"):
            click.echo("Cancelled.")
            return

    repo.delete(backtest_id)
    click.echo(f"Deleted backtest {backtest_id}")
