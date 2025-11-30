"""CLI entry point for cents."""

import json
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
    Valuation,
    TimeHorizon,
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
from cents.config import get_settings


SETTINGS = get_settings()


def _evidence_to_dict(evidence):
    """Serialize Evidence objects for JSON output."""

    return {
        "type": evidence.type.value,
        "content": evidence.content,
        "source": evidence.source,
        "confidence": evidence.confidence,
        "metadata": evidence.metadata,
    }


@click.group()
@click.version_option(version="0.1.0", package_name="cents")
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
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default=SETTINGS.default_output,
    show_default=True,
    help="Output format for results",
)
@click.option("--quiet", is_flag=True, help="Suppress verbose logs for scripting")
def research(
    symbol: str,
    thesis_id: Optional[str],
    agent_name: Optional[str],
    save: bool,
    output: str,
    quiet: bool,
):
    """Run research agents on a symbol."""
    verbose = output == "text" and not quiet

    # Get thesis if specified
    thesis = None
    if thesis_id:
        thesis_repo = ThesisRepository()
        thesis = thesis_repo.get(thesis_id)
        if thesis is None:
            click.echo(f"Thesis {thesis_id} not found.", err=True)
            raise SystemExit(1)
        if verbose:
            click.echo(f"Evaluating against thesis: {thesis.title}\n")

    # Determine which agents to run
    agents_to_run = {agent_name: AGENTS[agent_name]} if agent_name else AGENTS

    total_conviction_delta = 0.0
    all_evidence = []
    agent_outputs = []

    for name, agent_class in agents_to_run.items():
        agent = agent_class()
        result = agent.research(symbol.upper(), thesis)

        if verbose:
            click.echo(f"--- {name.upper()} ---")
            click.echo(f"Summary: {result.summary}")
            click.echo(f"Conviction delta: {result.conviction_delta:+.1f}")

            if result.evidence:
                click.echo("Evidence:")
                for e in result.evidence:
                    icon = {"supporting": "+", "contradicting": "-", "neutral": "~"}[e.type.value]
                    click.echo(f"  [{icon}] {e.content}")
            click.echo()

        agent_outputs.append(
            {
                "agent": name,
                "summary": result.summary,
                "conviction_delta": result.conviction_delta,
                "evidence": [_evidence_to_dict(e) for e in result.evidence],
            }
        )

        total_conviction_delta += result.conviction_delta
        all_evidence.extend(result.evidence)

    evidence_saved = False
    if save and all_evidence and thesis:
        evidence_repo = EvidenceRepository()
        for e in all_evidence:
            e.thesis_id = thesis.id
            evidence_repo.create(e)

        thesis_repo = ThesisRepository()
        thesis.update_conviction(total_conviction_delta)
        thesis_repo.update(thesis)
        evidence_saved = True

        if verbose:
            click.echo(f"Saved {len(all_evidence)} evidence items")
            click.echo(f"Thesis conviction: {thesis.conviction:.1f}% ({total_conviction_delta:+.1f})")
    elif not thesis and all_evidence and verbose:
        click.echo(f"Generated {len(all_evidence)} evidence items (not saved - no thesis linked)")

    if output == "json":
        payload = {
            "symbol": symbol.upper(),
            "thesis_id": thesis.id if thesis else None,
            "total_conviction_delta": total_conviction_delta,
            "agents": agent_outputs,
            "evidence_saved": evidence_saved,
            "evidence_count": len(all_evidence),
        }
        click.echo(json.dumps(payload, indent=2))
    else:
        if quiet:
            click.echo(
                f"{symbol.upper()} conviction delta: {total_conviction_delta:+.1f}"
            )
        elif not verbose:
            # This happens when output was coerced to text but quiet disabled
            click.echo(f"Total conviction delta: {total_conviction_delta:+.1f}")


# --- Thesis commands ---


@cli.group()
def thesis():
    """Manage investment theses."""
    pass


@thesis.command("create")
@click.argument("title")
@click.option("--hypothesis", "-h", default="", help="Detailed thesis statement")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--symbol", "-S", help="Stock ticker (e.g., AAPL)")
@click.option("--business-quality", "-b", help="Quality assessment")
@click.option("--valuation", "-v", type=click.Choice(["undervalued", "fair", "overvalued"]), help="Valuation assessment")
@click.option("--moat", "-m", help="Competitive advantage")
@click.option("--time-horizon", "-T", type=click.Choice(["short", "medium", "long"]), help="Investment horizon")
@click.option("--horizon-end", help="Expiry date (YYYY-MM-DD)")
@click.option("--risks", "-r", default="", help="Comma-separated key risks")
def thesis_create(
    title: str,
    hypothesis: str,
    tags: str,
    symbol: Optional[str],
    business_quality: Optional[str],
    valuation: Optional[str],
    moat: Optional[str],
    time_horizon: Optional[str],
    horizon_end: Optional[str],
    risks: str,
):
    """Create a new investment thesis."""
    from datetime import datetime as dt
    repo = ThesisRepository()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    risk_list = [r.strip() for r in risks.split(",") if r.strip()] if risks else []

    horizon_dt = None
    if horizon_end:
        try:
            horizon_dt = dt.strptime(horizon_end, "%Y-%m-%d")
        except ValueError:
            click.echo("Invalid date format. Use YYYY-MM-DD.", err=True)
            raise SystemExit(1)

    t = Thesis(
        title=title,
        hypothesis=hypothesis,
        tags=tag_list,
        symbol=symbol.upper() if symbol else None,
        business_quality=business_quality,
        valuation=Valuation(valuation) if valuation else None,
        moat=moat,
        time_horizon=TimeHorizon(time_horizon) if time_horizon else None,
        horizon_end=horizon_dt,
        key_risks=risk_list,
    )
    repo.create(t)
    click.echo(f"Created thesis {t.id}: {t.title}")


@thesis.command("list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["open", "closed", "invalidated"]),
    help="Filter by status",
)
@click.option("--symbol", "-S", help="Filter by symbol")
def thesis_list(status: Optional[str], symbol: Optional[str]):
    """List theses."""
    repo = ThesisRepository()
    status_filter = ThesisStatus(status) if status else None
    theses = repo.list(status=status_filter)

    # Filter by symbol if specified
    if symbol:
        symbol_upper = symbol.upper()
        theses = [t for t in theses if t.symbol and t.symbol.upper() == symbol_upper]

    if not theses:
        click.echo("No theses found.")
        return

    for t in theses:
        status_icon = {"open": "+", "closed": "-", "invalidated": "x"}[t.status.value]
        symbol_str = f" [{t.symbol}]" if t.symbol else ""
        click.echo(f"[{status_icon}] {t.id}:{symbol_str} {t.title} (conviction: {t.conviction:.0f}%)")


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
    if t.symbol:
        click.echo(f"Symbol:     {t.symbol}")
    click.echo(f"Status:     {t.status.value}")
    click.echo(f"Conviction: {t.conviction:.1f}%")
    if t.hypothesis:
        click.echo(f"Hypothesis: {t.hypothesis}")
    if t.business_quality:
        click.echo(f"Quality:    {t.business_quality}")
    if t.valuation:
        click.echo(f"Valuation:  {t.valuation.value}")
    if t.moat:
        click.echo(f"Moat:       {t.moat}")
    if t.time_horizon:
        click.echo(f"Horizon:    {t.time_horizon.value}")
    if t.horizon_end:
        click.echo(f"Expires:    {t.horizon_end.strftime('%Y-%m-%d')}")
    if t.key_risks:
        click.echo(f"Risks:      {', '.join(t.key_risks)}")
    if t.tags:
        click.echo(f"Tags:       {', '.join(t.tags)}")
    click.echo(f"Created:    {t.created_at.strftime('%Y-%m-%d %H:%M')}")
    click.echo(f"Updated:    {t.updated_at.strftime('%Y-%m-%d %H:%M')}")


@thesis.command("update")
@click.argument("thesis_id")
@click.option("--conviction", "-c", type=float, help="Set conviction (0-100)")
@click.option("--status", "-s", type=click.Choice(["open", "closed", "invalidated"]))
@click.option("--symbol", "-S", help="Stock ticker")
@click.option("--business-quality", "-b", help="Quality assessment")
@click.option("--valuation", "-v", type=click.Choice(["undervalued", "fair", "overvalued"]))
@click.option("--moat", "-m", help="Competitive advantage")
@click.option("--time-horizon", "-T", type=click.Choice(["short", "medium", "long"]))
@click.option("--horizon-end", help="Expiry date (YYYY-MM-DD)")
@click.option("--risks", "-r", help="Comma-separated key risks")
def thesis_update(
    thesis_id: str,
    conviction: Optional[float],
    status: Optional[str],
    symbol: Optional[str],
    business_quality: Optional[str],
    valuation: Optional[str],
    moat: Optional[str],
    time_horizon: Optional[str],
    horizon_end: Optional[str],
    risks: Optional[str],
):
    """Update a thesis."""
    from datetime import datetime as dt
    repo = ThesisRepository()
    t = repo.get(thesis_id)

    if t is None:
        click.echo(f"Thesis {thesis_id} not found.", err=True)
        raise SystemExit(1)

    if conviction is not None:
        t.conviction = max(0.0, min(100.0, conviction))
    if status:
        t.status = ThesisStatus(status)
    if symbol is not None:
        t.symbol = symbol.upper() if symbol else None
    if business_quality is not None:
        t.business_quality = business_quality if business_quality else None
    if valuation is not None:
        t.valuation = Valuation(valuation) if valuation else None
    if moat is not None:
        t.moat = moat if moat else None
    if time_horizon is not None:
        t.time_horizon = TimeHorizon(time_horizon) if time_horizon else None
    if horizon_end is not None:
        if horizon_end:
            try:
                t.horizon_end = dt.strptime(horizon_end, "%Y-%m-%d")
            except ValueError:
                click.echo("Invalid date format. Use YYYY-MM-DD.", err=True)
                raise SystemExit(1)
        else:
            t.horizon_end = None
    if risks is not None:
        t.key_risks = [r.strip() for r in risks.split(",") if r.strip()] if risks else []

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
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=SETTINGS.default_scan_threshold,
    show_default=True,
    help="Default conviction change threshold for alerts",
)
@click.option("--webhook", "-w", help="Webhook URL for notifications")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default=SETTINGS.default_output,
    show_default=True,
    help="Output format for scan results",
)
@click.option("--quiet", is_flag=True, help="Suppress verbose logs for scripting")
def scan(threshold: float, webhook: Optional[str], output: str, quiet: bool):
    """Scan watchlist and generate alerts for significant changes."""
    from cents.agents import OrchestratorAgent
    from cents.notify import notify

    verbose = output == "text" and not quiet

    watch_repo = WatchlistRepository()
    alert_repo = AlertRepository()
    thesis_repo = ThesisRepository()

    items = watch_repo.list()
    if not items:
        if output == "json":
            click.echo(json.dumps([], indent=2))
        else:
            click.echo("Watchlist is empty. Add symbols with: cents watch add <SYMBOL>")
        return

    if verbose:
        click.echo(f"Scanning {len(items)} symbols...\n")

    alerts_generated = 0
    scan_results: list[dict] = []

    for item in items:
        if verbose:
            click.echo(f"--- {item.symbol} ---")

        # Get linked thesis if any
        thesis = thesis_repo.get(item.thesis_id) if item.thesis_id else None

        # Run orchestrator
        agent = OrchestratorAgent()
        result = agent.research(item.symbol, thesis)

        effective_threshold = item.threshold if item.threshold is not None else threshold
        destination = webhook or item.alert_destination or SETTINGS.default_webhook

        if verbose:
            click.echo(f"  {result.summary}")
            click.echo(f"  Threshold: {effective_threshold:+.1f}")

        triggered = False
        alert_message = None

        # Check for significant conviction change
        if abs(result.conviction_delta) >= effective_threshold:
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
            triggered = True
            alert_message = alert.message
            if verbose:
                click.echo(f"  [!] Alert: {alert.message}")
            notify(alert, destination, quiet=quiet)

        # Update last_scanned
        watch_repo.update_scanned(item.symbol)
        if verbose:
            click.echo()

        scan_results.append(
            {
                "symbol": item.symbol,
                "summary": result.summary,
                "conviction_delta": result.conviction_delta,
                "threshold": effective_threshold,
                "alerted": triggered,
                "alert_message": alert_message,
                "alert_destination": destination if triggered else None,
            }
        )

    if output == "json":
        click.echo(json.dumps(scan_results, indent=2))
        return

    click.echo(f"Scan complete. Generated {alerts_generated} alerts.")
    if alerts_generated > 0 and not quiet:
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
@click.option(
    "--threshold",
    type=float,
    help="Custom conviction delta threshold for this symbol",
)
@click.option("--webhook", help="Custom webhook/alert destination for this symbol")
def watch_add(
    symbol: str,
    thesis_id: Optional[str],
    notes: str,
    threshold: Optional[float],
    webhook: Optional[str],
):
    """Add a symbol to watchlist."""
    repo = WatchlistRepository()
    existing = repo.get(symbol)
    if existing:
        click.echo(f"{symbol.upper()} is already on watchlist.")
        return

    item = WatchlistItem(
        symbol=symbol.upper(),
        thesis_id=thesis_id,
        notes=notes,
        threshold=threshold,
        alert_destination=webhook,
    )
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
        extras = []
        if item.threshold is not None:
            extras.append(f"threshold: {item.threshold:.1f}")
        if item.alert_destination:
            extras.append("alert: custom")
        extras_str = f" | {'; '.join(extras)}" if extras else ""
        click.echo(f"  {item.symbol}{thesis_str} - last scanned: {scanned}{extras_str}")


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
