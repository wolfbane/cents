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


def _calculate_correlation(x: list[float], y: list[float]) -> float | None:
    """Calculate Pearson correlation between two lists."""
    if len(x) < 3 or len(x) != len(y):
        return None

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if den_x > 0 and den_y > 0:
        return num / (den_x * den_y)
    return None


def _calculate_hit_rate(deltas: list[float], returns: list[float]) -> float | None:
    """Calculate hit rate: % of times delta sign matches return sign."""
    if not deltas or len(deltas) != len(returns):
        return None

    hits = sum(1 for d, r in zip(deltas, returns) if (d > 0 and r > 0) or (d < 0 and r < 0))
    return hits / len(deltas)


@backtest.command("analyze")
@click.argument("backtest_id", required=False)
@click.option("--symbol", "-s", default=None, help="Analyze all backtests for symbol")
@click.option("--all", "analyze_all", is_flag=True, help="Analyze all backtests")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def analyze_backtest(backtest_id: str | None, symbol: str | None, analyze_all: bool, output: str):
    """Analyze signal-to-return correlations.

    Shows correlation and hit rate by agent across time horizons.

    Examples:
      cents backtest analyze abc123        # Single backtest
      cents backtest analyze --symbol NVDA # All NVDA backtests
      cents backtest analyze --all         # All backtests
    """
    repo = BacktestRepository()

    # Gather backtests to analyze
    if backtest_id:
        bt = repo.get(backtest_id)
        if bt is None:
            click.echo(f"Backtest {backtest_id} not found.", err=True)
            raise SystemExit(1)
        backtests = [bt]
    elif symbol:
        backtests = repo.list(symbol=symbol.upper())
        if not backtests:
            click.echo(f"No backtests found for {symbol.upper()}.", err=True)
            raise SystemExit(1)
    elif analyze_all:
        backtests = repo.list()
        if not backtests:
            click.echo("No backtests found.", err=True)
            raise SystemExit(1)
    else:
        click.echo("Specify a backtest ID, --symbol, or --all.", err=True)
        raise SystemExit(1)

    # Collect all signals
    all_signals = []
    for bt in backtests:
        all_signals.extend(repo.get_signals(bt.id))

    if not all_signals:
        click.echo("No signals found.", err=True)
        raise SystemExit(1)

    # Group by agent
    agent_data: dict[str, list[tuple[float, dict, dict]]] = {}  # delta, returns, dimensions
    for s in all_signals:
        if s.agent_name not in agent_data:
            agent_data[s.agent_name] = []
        agent_data[s.agent_name].append((s.conviction_delta, s.forward_returns, s.dimension_scores))

    # Calculate stats per agent
    horizons = ["5d", "20d", "60d"]
    results = []

    for agent_name in sorted(agent_data.keys()):
        data = agent_data[agent_name]
        deltas = [d[0] for d in data]
        n = len(deltas)
        avg_delta = sum(deltas) / n if n else 0

        agent_result = {
            "agent": agent_name,
            "signals": n,
            "avg_delta": avg_delta,
            "correlations": {},
            "hit_rates": {},
        }

        for horizon in horizons:
            # Filter to signals that have this horizon's returns
            pairs = [(d[0], d[1][horizon]) for d in data if horizon in d[1]]
            if pairs:
                h_deltas = [p[0] for p in pairs]
                h_returns = [p[1] for p in pairs]

                corr = _calculate_correlation(h_deltas, h_returns)
                hit = _calculate_hit_rate(h_deltas, h_returns)

                agent_result["correlations"][horizon] = corr
                agent_result["hit_rates"][horizon] = hit

        results.append(agent_result)

    # Also calculate dimension-level stats
    dimension_data: dict[str, list[tuple[float, dict]]] = {}
    for s in all_signals:
        for dim, score in s.dimension_scores.items():
            if dim not in dimension_data:
                dimension_data[dim] = []
            dimension_data[dim].append((score, s.forward_returns))

    dimension_results = []
    for dim_name in sorted(dimension_data.keys()):
        data = dimension_data[dim_name]
        scores = [d[0] for d in data]
        n = len(scores)

        dim_result = {
            "dimension": dim_name,
            "signals": n,
            "correlations": {},
            "hit_rates": {},
        }

        for horizon in horizons:
            pairs = [(d[0], d[1][horizon]) for d in data if horizon in d[1]]
            if pairs:
                h_scores = [p[0] for p in pairs]
                h_returns = [p[1] for p in pairs]

                corr = _calculate_correlation(h_scores, h_returns)
                hit = _calculate_hit_rate(h_scores, h_returns)

                dim_result["correlations"][horizon] = corr
                dim_result["hit_rates"][horizon] = hit

        dimension_results.append(dim_result)

    if output == "json":
        click.echo(json.dumps({
            "backtests": len(backtests),
            "total_signals": len(all_signals),
            "agents": results,
            "dimensions": dimension_results,
        }, indent=2))
    else:
        symbols = list(set(bt.symbol for bt in backtests))
        click.echo(f"Analysis: {len(backtests)} backtest(s), {len(all_signals)} signals")
        click.echo(f"Symbols: {', '.join(symbols)}")
        click.echo()

        # Agent correlations table
        click.echo("Agent Correlations (conviction_delta vs forward returns):")
        click.echo(f"  {'Agent':<14} {'N':<6} {'Corr 5d':<10} {'Corr 20d':<10} {'Corr 60d':<10}")
        click.echo("  " + "-" * 56)

        for r in results:
            corrs = r["correlations"]
            c5 = f"{corrs.get('5d', 0):+.2f}" if corrs.get('5d') is not None else "N/A"
            c20 = f"{corrs.get('20d', 0):+.2f}" if corrs.get('20d') is not None else "N/A"
            c60 = f"{corrs.get('60d', 0):+.2f}" if corrs.get('60d') is not None else "N/A"
            click.echo(f"  {r['agent']:<14} {r['signals']:<6} {c5:<10} {c20:<10} {c60:<10}")

        click.echo()

        # Agent hit rates table
        click.echo("Agent Hit Rates (% signals where delta sign matches return sign):")
        click.echo(f"  {'Agent':<14} {'Hit 5d':<10} {'Hit 20d':<10} {'Hit 60d':<10}")
        click.echo("  " + "-" * 46)

        for r in results:
            hits = r["hit_rates"]
            h5 = f"{hits.get('5d', 0)*100:.0f}%" if hits.get('5d') is not None else "N/A"
            h20 = f"{hits.get('20d', 0)*100:.0f}%" if hits.get('20d') is not None else "N/A"
            h60 = f"{hits.get('60d', 0)*100:.0f}%" if hits.get('60d') is not None else "N/A"
            click.echo(f"  {r['agent']:<14} {h5:<10} {h20:<10} {h60:<10}")

        if dimension_results:
            click.echo()
            click.echo("Dimension Correlations:")
            click.echo(f"  {'Dimension':<14} {'N':<6} {'Corr 5d':<10} {'Corr 20d':<10} {'Corr 60d':<10}")
            click.echo("  " + "-" * 56)

            for r in dimension_results:
                corrs = r["correlations"]
                c5 = f"{corrs.get('5d', 0):+.2f}" if corrs.get('5d') is not None else "N/A"
                c20 = f"{corrs.get('20d', 0):+.2f}" if corrs.get('20d') is not None else "N/A"
                c60 = f"{corrs.get('60d', 0):+.2f}" if corrs.get('60d') is not None else "N/A"
                click.echo(f"  {r['dimension']:<14} {r['signals']:<6} {c5:<10} {c20:<10} {c60:<10}")


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
