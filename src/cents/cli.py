"""CLI entry point for cents."""

from datetime import date
from typing import Optional

import click

from cents.db import (
    ThesisRepository,
    PositionRepository,
    OutcomeRepository,
    EvidenceRepository,
    WatchlistRepository,
    AlertRepository,
)
from cents.models import (
    Thesis,
    ThesisStatus,
    Position,
    PositionSide,
    PositionStatus,
    Outcome,
    ThesisAccuracy,
    EvidenceType,
    WatchlistItem,
    Alert,
    AlertType,
)
from cents.agents import AGENTS


@click.group()
@click.version_option()
def cli():
    """Cents: Agentic investing guidance."""
    pass


# --- Research command ---


@cli.command("research")
@click.argument("symbol")
@click.option("--thesis", "-t", "thesis_id", help="Thesis ID to evaluate against")
@click.option(
    "--agent",
    "-a",
    "agent_name",
    type=click.Choice(list(AGENTS.keys())),
    help="Run specific agent only",
)
@click.option("--save/--no-save", default=True, help="Save evidence to database")
def research(symbol: str, thesis_id: Optional[str], agent_name: Optional[str], save: bool):
    """Run research agents on a symbol."""
    # Get thesis if specified
    thesis = None
    if thesis_id:
        thesis_repo = ThesisRepository()
        thesis = thesis_repo.get(thesis_id)
        if thesis is None:
            click.echo(f"Thesis {thesis_id} not found.", err=True)
            raise SystemExit(1)
        click.echo(f"Evaluating against thesis: {thesis.title}\n")

    # Determine which agents to run
    if agent_name:
        agents_to_run = {agent_name: AGENTS[agent_name]}
    else:
        agents_to_run = AGENTS

    total_conviction_delta = 0.0
    all_evidence = []

    for name, agent_class in agents_to_run.items():
        click.echo(f"--- {name.upper()} ---")
        agent = agent_class()
        result = agent.research(symbol.upper(), thesis)

        click.echo(f"Summary: {result.summary}")
        click.echo(f"Conviction delta: {result.conviction_delta:+.1f}")

        if result.evidence:
            click.echo("Evidence:")
            for e in result.evidence:
                icon = {"supporting": "+", "contradicting": "-", "neutral": "~"}[e.type.value]
                click.echo(f"  [{icon}] {e.content}")

        total_conviction_delta += result.conviction_delta
        all_evidence.extend(result.evidence)
        click.echo()

    # Save evidence and update thesis if requested
    if save and all_evidence and thesis:
        evidence_repo = EvidenceRepository()
        for e in all_evidence:
            e.thesis_id = thesis.id
            evidence_repo.create(e)

        thesis_repo = ThesisRepository()
        thesis.update_conviction(total_conviction_delta)
        thesis_repo.update(thesis)

        click.echo(f"Saved {len(all_evidence)} evidence items")
        click.echo(f"Thesis conviction: {thesis.conviction:.1f}% ({total_conviction_delta:+.1f})")
    elif not thesis and all_evidence:
        click.echo(f"Generated {len(all_evidence)} evidence items (not saved - no thesis linked)")


# --- Thesis commands ---


@cli.group()
def thesis():
    """Manage investment theses."""
    pass


@thesis.command("create")
@click.argument("title")
@click.option("--hypothesis", "-h", default="", help="Detailed thesis statement")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
def thesis_create(title: str, hypothesis: str, tags: str):
    """Create a new investment thesis."""
    repo = ThesisRepository()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    t = Thesis(title=title, hypothesis=hypothesis, tags=tag_list)
    repo.create(t)
    click.echo(f"Created thesis {t.id}: {t.title}")


@thesis.command("list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["open", "closed", "invalidated"]),
    help="Filter by status",
)
def thesis_list(status: Optional[str]):
    """List theses."""
    repo = ThesisRepository()
    status_filter = ThesisStatus(status) if status else None
    theses = repo.list(status=status_filter)

    if not theses:
        click.echo("No theses found.")
        return

    for t in theses:
        status_icon = {"open": "+", "closed": "-", "invalidated": "x"}[t.status.value]
        click.echo(f"[{status_icon}] {t.id}: {t.title} (conviction: {t.conviction:.0f}%)")


@thesis.command("show")
@click.argument("thesis_id")
def thesis_show(thesis_id: str):
    """Show thesis details."""
    repo = ThesisRepository()
    t = repo.get(thesis_id)

    if t is None:
        click.echo(f"Thesis {thesis_id} not found.", err=True)
        raise SystemExit(1)

    click.echo(f"ID:         {t.id}")
    click.echo(f"Title:      {t.title}")
    click.echo(f"Status:     {t.status.value}")
    click.echo(f"Conviction: {t.conviction:.1f}%")
    if t.hypothesis:
        click.echo(f"Hypothesis: {t.hypothesis}")
    if t.tags:
        click.echo(f"Tags:       {', '.join(t.tags)}")
    click.echo(f"Created:    {t.created_at.strftime('%Y-%m-%d %H:%M')}")
    click.echo(f"Updated:    {t.updated_at.strftime('%Y-%m-%d %H:%M')}")


@thesis.command("update")
@click.argument("thesis_id")
@click.option("--conviction", "-c", type=float, help="Set conviction (0-100)")
@click.option("--status", "-s", type=click.Choice(["open", "closed", "invalidated"]))
def thesis_update(thesis_id: str, conviction: Optional[float], status: Optional[str]):
    """Update a thesis."""
    repo = ThesisRepository()
    t = repo.get(thesis_id)

    if t is None:
        click.echo(f"Thesis {thesis_id} not found.", err=True)
        raise SystemExit(1)

    if conviction is not None:
        t.conviction = max(0.0, min(100.0, conviction))
    if status:
        t.status = ThesisStatus(status)

    repo.update(t)
    click.echo(f"Updated thesis {t.id}")


# --- Position commands ---


@cli.group()
def position():
    """Manage positions."""
    pass


@position.command("open")
@click.argument("symbol")
@click.argument("size", type=float)
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
    """Open a new paper position."""
    repo = PositionRepository()
    side = PositionSide.SHORT if short else PositionSide.LONG
    p = Position(
        symbol=symbol.upper(),
        size=size,
        entry_price=price,
        side=side,
        thesis_id=thesis_id,
        notes=notes,
        paper=True,
    )
    repo.create(p)
    click.echo(f"Opened {side.value} position {p.id}: {size} {symbol.upper()} @ ${price:.2f}")


@position.command("close")
@click.argument("position_id")
@click.argument("exit_price", type=float)
def position_close(position_id: str, exit_price: float):
    """Close a position."""
    repo = PositionRepository()
    p = repo.get(position_id)

    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    if p.status == PositionStatus.CLOSED:
        click.echo(f"Position {position_id} is already closed.", err=True)
        raise SystemExit(1)

    p.close(exit_price)
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
def position_list(status: Optional[str]):
    """List positions."""
    repo = PositionRepository()
    status_filter = PositionStatus(status) if status else None
    positions = repo.list(status=status_filter)

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


# --- Outcome commands ---


@cli.group()
def outcome():
    """Track outcomes."""
    pass


@outcome.command("record")
@click.argument("position_id")
@click.option(
    "--accuracy",
    "-a",
    type=click.Choice(["correct", "incorrect", "partial", "unclear"]),
    default="unclear",
    help="Was the thesis correct?",
)
@click.option("--notes", "-n", default="", help="Retrospective notes")
def outcome_record(position_id: str, accuracy: str, notes: str):
    """Record outcome for a closed position."""
    pos_repo = PositionRepository()
    out_repo = OutcomeRepository()

    p = pos_repo.get(position_id)
    if p is None:
        click.echo(f"Position {position_id} not found.", err=True)
        raise SystemExit(1)

    if p.status != PositionStatus.CLOSED:
        click.echo(f"Position {position_id} is not closed yet.", err=True)
        raise SystemExit(1)

    existing = out_repo.get_for_position(position_id)
    if existing:
        click.echo(f"Outcome already recorded for position {position_id}.", err=True)
        raise SystemExit(1)

    o = Outcome(
        position_id=position_id,
        pnl=p.pnl,
        pnl_pct=p.pnl_pct,
        thesis_accuracy=ThesisAccuracy(accuracy),
        retrospective=notes,
    )
    out_repo.create(o)
    click.echo(f"Recorded outcome {o.id} for position {position_id}")


@outcome.command("list")
def outcome_list():
    """List all recorded outcomes."""
    repo = OutcomeRepository()
    outcomes = repo.list()

    if not outcomes:
        click.echo("No outcomes recorded.")
        return

    for o in outcomes:
        sign = "+" if o.pnl >= 0 else ""
        acc = o.thesis_accuracy.value[:1].upper()
        click.echo(
            f"[{acc}] {o.id}: position {o.position_id} {sign}${o.pnl:.2f} ({sign}{o.pnl_pct:.1f}%)"
        )


# --- Scan command ---


@cli.command("scan")
@click.option("--threshold", "-t", type=float, default=5.0, help="Conviction change threshold for alerts")
@click.option("--webhook", "-w", help="Webhook URL for notifications")
def scan(threshold: float, webhook: Optional[str]):
    """Scan watchlist and generate alerts for significant changes."""
    from cents.agents import OrchestratorAgent
    from cents.notify import notify

    watch_repo = WatchlistRepository()
    alert_repo = AlertRepository()
    thesis_repo = ThesisRepository()

    items = watch_repo.list()
    if not items:
        click.echo("Watchlist is empty. Add symbols with: cents watch add <SYMBOL>")
        return

    click.echo(f"Scanning {len(items)} symbols...\n")
    alerts_generated = 0

    for item in items:
        click.echo(f"--- {item.symbol} ---")

        # Get linked thesis if any
        thesis = None
        if item.thesis_id:
            thesis = thesis_repo.get(item.thesis_id)

        # Run orchestrator
        agent = OrchestratorAgent()
        result = agent.research(item.symbol, thesis)

        click.echo(f"  {result.summary}")

        # Check for significant conviction change
        if abs(result.conviction_delta) >= threshold:
            direction = "bullish" if result.conviction_delta > 0 else "bearish"
            alert = Alert(
                symbol=item.symbol,
                alert_type=AlertType.CONVICTION_CHANGE,
                message=f"Significant {direction} signal: {result.conviction_delta:+.1f} conviction",
                data={
                    "conviction_delta": result.conviction_delta,
                    "evidence_count": len(result.evidence),
                },
            )
            alert_repo.create(alert)
            alerts_generated += 1
            click.echo(f"  [!] Alert: {alert.message}")
            notify(alert, webhook)

        # Update last_scanned
        watch_repo.update_scanned(item.symbol)
        click.echo()

    click.echo(f"Scan complete. Generated {alerts_generated} alerts.")
    if alerts_generated > 0:
        click.echo("View alerts with: cents alert list")


# --- Watch commands ---


@cli.group()
def watch():
    """Manage watchlist."""
    pass


@watch.command("add")
@click.argument("symbol")
@click.option("--thesis", "-t", "thesis_id", help="Link to thesis ID")
@click.option("--notes", "-n", default="", help="Notes for this watch")
def watch_add(symbol: str, thesis_id: Optional[str], notes: str):
    """Add a symbol to watchlist."""
    repo = WatchlistRepository()
    existing = repo.get(symbol)
    if existing:
        click.echo(f"{symbol.upper()} is already on watchlist.")
        return

    item = WatchlistItem(symbol=symbol.upper(), thesis_id=thesis_id, notes=notes)
    repo.add(item)
    click.echo(f"Added {symbol.upper()} to watchlist")


@watch.command("remove")
@click.argument("symbol")
def watch_remove(symbol: str):
    """Remove a symbol from watchlist."""
    repo = WatchlistRepository()
    if repo.remove(symbol):
        click.echo(f"Removed {symbol.upper()} from watchlist")
    else:
        click.echo(f"{symbol.upper()} not found in watchlist.", err=True)


@watch.command("list")
def watch_list():
    """List watched symbols."""
    repo = WatchlistRepository()
    items = repo.list()

    if not items:
        click.echo("Watchlist is empty.")
        return

    for item in items:
        scanned = item.last_scanned.strftime("%Y-%m-%d %H:%M") if item.last_scanned else "never"
        thesis_str = f" (thesis: {item.thesis_id})" if item.thesis_id else ""
        click.echo(f"  {item.symbol}{thesis_str} - last scanned: {scanned}")


# --- Alert commands ---


@cli.group()
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
    else:
        click.echo("Specify alert ID or use --all", err=True)


# --- Broker commands ---


@cli.group()
def broker():
    """Alpaca broker integration."""
    pass


@broker.command("status")
def broker_status():
    """Check broker connection and account status."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]")
        return

    try:
        client = AlpacaClient(paper=True)
        account = client.get_account()
        click.echo("Connected to Alpaca (paper trading)")
        click.echo(f"  Buying Power: ${account['buying_power']:,.2f}")
        click.echo(f"  Cash: ${account['cash']:,.2f}")
        click.echo(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
    except Exception as e:
        click.echo(f"Connection failed: {e}", err=True)


@broker.command("positions")
def broker_positions():
    """List positions from broker."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]")
        return

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
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@broker.command("sync")
@click.option("--thesis", "-t", "thesis_id", help="Link synced positions to thesis")
def broker_sync(thesis_id: Optional[str]):
    """Sync positions from broker to cents."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]")
        return

    try:
        client = AlpacaClient(paper=True)
        positions = client.get_positions()

        if not positions:
            click.echo("No positions to sync.")
            return

        repo = PositionRepository()
        synced = 0

        for bp in positions:
            # Check if already tracked
            existing = [p for p in repo.list() if p.symbol == bp.symbol and p.status == PositionStatus.OPEN]
            if existing:
                click.echo(f"  {bp.symbol}: already tracked, skipping")
                continue

            pos = client.to_cents_position(bp, thesis_id)
            repo.create(pos)
            synced += 1
            click.echo(f"  {bp.symbol}: synced {bp.qty:.0f} shares @ ${bp.avg_entry_price:.2f}")

        click.echo(f"\nSynced {synced} positions")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@broker.command("buy")
@click.argument("symbol")
@click.argument("qty", type=float)
@click.option("--thesis", "-t", "thesis_id", help="Link to thesis ID")
@click.confirmation_option(prompt="Are you sure you want to execute this trade?")
def broker_buy(symbol: str, qty: float, thesis_id: Optional[str]):
    """Buy shares (paper trading)."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]")
        return

    try:
        client = AlpacaClient(paper=True)
        result = client.submit_order(symbol.upper(), qty, "buy")
        click.echo(f"Order submitted: {result.status}")
        click.echo(f"  Order ID: {result.order_id}")
        if result.filled_avg_price:
            click.echo(f"  Filled @ ${result.filled_avg_price:.2f}")
    except Exception as e:
        click.echo(f"Order failed: {e}", err=True)


@broker.command("sell")
@click.argument("symbol")
@click.argument("qty", type=float)
@click.confirmation_option(prompt="Are you sure you want to execute this trade?")
def broker_sell(symbol: str, qty: float):
    """Sell shares (paper trading)."""
    from cents.broker import ALPACA_AVAILABLE, AlpacaClient

    if not ALPACA_AVAILABLE:
        click.echo("Alpaca not installed. Install with: pip install cents[broker]")
        return

    try:
        client = AlpacaClient(paper=True)
        result = client.submit_order(symbol.upper(), qty, "sell")
        click.echo(f"Order submitted: {result.status}")
        click.echo(f"  Order ID: {result.order_id}")
        if result.filled_avg_price:
            click.echo(f"  Filled @ ${result.filled_avg_price:.2f}")
    except Exception as e:
        click.echo(f"Order failed: {e}", err=True)


if __name__ == "__main__":
    cli()
