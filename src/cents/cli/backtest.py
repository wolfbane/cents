"""Backtest CLI commands."""

import logging
from datetime import date, timedelta

import click

from cents.agents import AGENTS
from cents.db import BacktestRepository
from cents.models import Backtest, BacktestSignal

from ._disclosures import disclosure_text, low_n_warning
from ._shared import (
    calculate_correlation,
    calculate_hit_rate,
    parse_agents,
    parse_date_range,
    parse_symbols,
    render_output,
    validate_symbol,
)

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


@click.group("backtest", invoke_without_command=True)
@click.pass_context
def backtest(ctx):
    """Run and analyze agent backtests."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_backtests)


def _run_single_backtest(
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str,
    selected_agents: dict,
    provider,
    verbose: bool = True,
) -> tuple[str, int, list[str]]:
    """Run backtest for a single symbol. Returns (backtest_id, signal_count, errors)."""
    repo = BacktestRepository()
    bt = Backtest(symbol=symbol, start_date=start_date, end_date=end_date)
    repo.create(bt)

    eval_dates = _generate_dates(start_date, end_date, interval)
    signal_count = 0
    errors = []

    for i, eval_date in enumerate(eval_dates):
        if verbose:
            click.echo(f"  [{i+1}/{len(eval_dates)}] {eval_date}...", nl=False)

        date_signals = 0
        for agent_name, agent_class in selected_agents.items():
            try:
                agent = agent_class()
                result = agent.research(symbol, thesis=None, as_of=eval_date)
                forward_returns = _calculate_forward_returns(symbol, eval_date, provider)

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

        if verbose:
            click.echo(f" {date_signals} signals")

    return bt.id, signal_count, errors


@backtest.command("run")
@click.argument("symbol", required=False)
@click.option(
    "--symbols", "symbols_str", default=None,
    help="Comma-separated list of symbols (alternative to positional arg)"
)
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
    symbol: str | None,
    symbols_str: str | None,
    start_str: str,
    end_str: str | None,
    interval: str,
    agent_names: str | None,
    output: str,
):
    """Run a backtest for one or more symbols over a date range.

    Examples:
      cents backtest run NVDA --start 2023-01-01 --end 2024-01-01
      cents backtest run --symbols AAPL,MSFT,NVDA --start 2023-01-01
    """
    symbols = parse_symbols(symbol, symbols_str)
    start_date, end_date = parse_date_range(start_str, end_str)
    selected_agents = parse_agents(agent_names, BACKTEST_AGENTS)

    # Initialize price provider
    try:
        from cents.data.alpaca import get_price_provider
        provider = get_price_provider()
    except Exception as e:
        click.echo(f"Could not initialize price provider: {e}", err=True)
        raise SystemExit(1)

    # Run backtests for each symbol
    results = []
    total_signals = 0
    total_errors = 0

    for sym in symbols:
        if output == "text":
            click.echo(f"=== {sym} ===")
            click.echo(f"Period: {start_date} to {end_date}, Interval: {interval}")
            click.echo(f"Agents: {', '.join(selected_agents.keys())}")
            click.echo()

        bt_id, signal_count, errors = _run_single_backtest(
            symbol=sym,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            selected_agents=selected_agents,
            provider=provider,
            verbose=(output == "text"),
        )

        results.append({
            "backtest_id": bt_id,
            "symbol": sym,
            "signal_count": signal_count,
            "error_count": len(errors),
        })
        total_signals += signal_count
        total_errors += len(errors)

        if output == "text":
            click.echo()
            click.echo(f"Completed: {signal_count} signals recorded")
            if errors:
                click.echo(f"Errors: {len(errors)}")
            click.echo()

    def _render_text():
        if len(symbols) > 1:
            click.echo("=" * 40)
            click.echo(f"Total: {len(symbols)} symbols, {total_signals} signals")
            if total_errors:
                click.echo(f"Total errors: {total_errors}")
            click.echo(f"\nAnalyze all: cents backtest analyze --all")
        else:
            click.echo(f"View results: cents backtest show {results[0]['backtest_id']}")

    render_output(
        output,
        _render_text,
        {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "interval": interval,
            "backtests": results,
            "total_signals": total_signals,
            "total_errors": total_errors,
        },
    )


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

    data = [
        {
            "id": bt.id,
            "symbol": bt.symbol,
            "start_date": bt.start_date.isoformat(),
            "end_date": bt.end_date.isoformat(),
            "created_at": bt.created_at.isoformat(),
        }
        for bt in backtests
    ]

    def _render_text():
        if not backtests:
            click.echo("No backtests found.")
            return

        click.echo(f"{'ID':<10} {'Symbol':<8} {'Start':<12} {'End':<12} {'Created'}")
        click.echo("-" * 60)
        for bt in backtests:
            click.echo(f"{bt.id:<10} {bt.symbol:<8} {bt.start_date} {bt.end_date} {bt.created_at.strftime('%Y-%m-%d')}")

    render_output(output, _render_text, data)


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

    data = {
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
    }

    def _render_text():
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
                deltas_with_ret = [d[0] for d in data if "20d" in d[1]]
                corr = calculate_correlation(deltas_with_ret, returns_20d)
                corr_str = f"{corr:+.2f}" if corr is not None else "N/A"

                click.echo(f"  {agent_name:<14} {len(data):<8} {avg_delta:+.2f}      {corr_str}")

    render_output(output, _render_text, data)


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

                corr = calculate_correlation(h_deltas, h_returns)
                hit = calculate_hit_rate(h_deltas, h_returns)

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

                corr = calculate_correlation(h_scores, h_returns)
                hit = calculate_hit_rate(h_scores, h_returns)

                dim_result["correlations"][horizon] = corr
                dim_result["hit_rates"][horizon] = hit

        dimension_results.append(dim_result)

    # Low-N is judged against total_signals: with fewer than the threshold
    # number of model evaluations, hit-rate and correlation are dominated by
    # variance, not skill.
    is_low_n = low_n_warning(len(all_signals)) is not None

    data = {
        "backtests": len(backtests),
        "total_signals": len(all_signals),
        "agents": results,
        "dimensions": dimension_results,
        "_disclosure": disclosure_text(),
        "_low_n": is_low_n,
    }

    def _render_text():
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

        warning = low_n_warning(len(all_signals))
        if warning:
            click.echo()
            click.echo(warning)
        click.echo()
        click.echo(disclosure_text())

    render_output(output, _render_text, data)


@backtest.command("report")
@click.argument("backtest_id")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def report_backtest(backtest_id: str, output: str):
    """Generate detailed backtest report with best/worst signals."""
    repo = BacktestRepository()
    bt = repo.get(backtest_id)

    if bt is None:
        click.echo(f"Backtest {backtest_id} not found.", err=True)
        raise SystemExit(1)

    signals = repo.get_signals(backtest_id)
    if not signals:
        click.echo("No signals found.", err=True)
        raise SystemExit(1)

    # Calculate signal scores (delta * 20d return for direction alignment)
    scored_signals = []
    for s in signals:
        ret_20d = s.forward_returns.get("20d")
        if ret_20d is not None:
            # Score = how well delta predicted return direction
            # Positive score = correct direction, negative = wrong direction
            score = s.conviction_delta * ret_20d * 100  # Scale for readability
            scored_signals.append({
                "date": s.date,
                "agent": s.agent_name,
                "delta": s.conviction_delta,
                "return_20d": ret_20d,
                "score": score,
            })

    if not scored_signals:
        click.echo("No signals with forward returns found.", err=True)
        raise SystemExit(1)

    # Sort by score
    sorted_signals = sorted(scored_signals, key=lambda x: x["score"], reverse=True)
    best_signals = sorted_signals[:5]
    worst_signals = sorted_signals[-5:][::-1]  # Reverse to show worst first

    # Calculate cumulative accuracy over time
    chronological = sorted(scored_signals, key=lambda x: x["date"])
    cumulative = []
    running_hits = 0
    for i, s in enumerate(chronological):
        is_hit = (s["delta"] > 0 and s["return_20d"] > 0) or (s["delta"] < 0 and s["return_20d"] < 0)
        if is_hit:
            running_hits += 1
        cumulative.append({
            "date": s["date"],
            "accuracy": running_hits / (i + 1),
            "hits": running_hits,
            "total": i + 1,
        })

    data = {
        "backtest_id": bt.id,
        "symbol": bt.symbol,
        "period": f"{bt.start_date} to {bt.end_date}",
        "total_signals": len(scored_signals),
        "best_signals": best_signals,
        "worst_signals": worst_signals,
        "cumulative_accuracy": [
            {"date": c["date"].isoformat(), "accuracy": c["accuracy"]}
            for c in cumulative
        ],
    }

    def _render_text():
        click.echo(f"Backtest Report: {bt.id}")
        click.echo(f"Symbol: {bt.symbol} | Period: {bt.start_date} to {bt.end_date}")
        click.echo(f"Total signals with returns: {len(scored_signals)}")
        click.echo()

        click.echo("Best Signals (delta aligned with returns):")
        click.echo(f"  {'Date':<12} {'Agent':<14} {'Delta':<8} {'20d Ret':<10} {'Score'}")
        click.echo("  " + "-" * 55)
        for s in best_signals:
            click.echo(f"  {s['date']} {s['agent']:<14} {s['delta']:+5.1f}   {s['return_20d']*100:+6.1f}%    {s['score']:+.1f}")

        click.echo()
        click.echo("Worst Signals (delta opposite to returns):")
        click.echo(f"  {'Date':<12} {'Agent':<14} {'Delta':<8} {'20d Ret':<10} {'Score'}")
        click.echo("  " + "-" * 55)
        for s in worst_signals:
            click.echo(f"  {s['date']} {s['agent']:<14} {s['delta']:+5.1f}   {s['return_20d']*100:+6.1f}%    {s['score']:+.1f}")

        click.echo()
        click.echo("Accuracy Over Time:")
        # Show a few checkpoints
        checkpoints = [0, len(cumulative)//4, len(cumulative)//2, 3*len(cumulative)//4, len(cumulative)-1]
        checkpoints = sorted(set(c for c in checkpoints if 0 <= c < len(cumulative)))
        for i in checkpoints:
            c = cumulative[i]
            bar = "█" * int(c["accuracy"] * 20) + "░" * (20 - int(c["accuracy"] * 20))
            click.echo(f"  {c['date']} [{bar}] {c['accuracy']*100:.0f}% ({c['hits']}/{c['total']})")

    render_output(output, _render_text, data)


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
