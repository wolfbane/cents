"""Thesis management CLI commands."""

import json
from datetime import datetime as dt

import click

from cents.agents import AGENTS
from cents.db import ThesisRepository
from cents.models import (
    Thesis,
    ThesisStatus,
    Valuation,
    TimeHorizon,
    ThesisOutcome,
)

from ._shared import validate_symbol, generate_thesis_suggestion, evidence_to_dict, get_settings_lazy


@click.group()
def thesis():
    """Manage investment theses."""
    pass


@thesis.command("create")
@click.option("--title", "-T", required=True, help="Thesis title")
@click.option("--hypothesis", "-h", default="", help="Detailed thesis statement")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--symbol", "-S", help="Stock ticker (e.g., AAPL)")
@click.option("--business-quality", "-b", help="Quality assessment")
@click.option("--valuation", "-v", type=click.Choice(["undervalued", "fair", "overvalued"]), help="Valuation assessment")
@click.option("--moat", "-m", help="Competitive advantage")
@click.option("--time-horizon", "-H", type=click.Choice(["short", "medium", "long"]), help="Investment horizon")
@click.option("--horizon-end", help="Expiry date (YYYY-MM-DD)")
@click.option("--risks", "-r", default="", help="Comma-separated key risks")
@click.option("--target-price", type=float, help="Target price to close thesis")
@click.option("--stop-price", type=float, help="Stop price to close thesis")
@click.option("--from-research", "research_symbol", help="Auto-populate from research on symbol")
def thesis_create(
    title: str,
    hypothesis: str,
    tags: str,
    symbol: str | None,
    business_quality: str | None,
    valuation: str | None,
    moat: str | None,
    time_horizon: str | None,
    horizon_end: str | None,
    risks: str,
    target_price: float | None,
    stop_price: float | None,
    research_symbol: str | None,
):
    """Create a new investment thesis.

    Example: cents thesis create --title "NVDA bull case" --symbol NVDA
    """
    repo = ThesisRepository()

    # If --from-research is specified, run research and get suggestions
    suggestion = None
    if research_symbol:
        click.echo(f"Running research on {research_symbol.upper()}...")
        agent_outputs = []
        agent_deltas: dict[str, float] = {}

        for name, agent_class in AGENTS.items():
            agent = agent_class()
            result = agent.research(research_symbol.upper(), None)
            agent_outputs.append({
                "agent": name,
                "summary": result.summary,
                "conviction_delta": result.conviction_delta,
                "evidence": [evidence_to_dict(e) for e in result.evidence],
            })
            agent_deltas[name] = result.conviction_delta

        # Use orchestrator's delta (weighted aggregate) if present
        if "orchestrator" in agent_deltas:
            total_conviction_delta = agent_deltas["orchestrator"]
        else:
            total_conviction_delta = sum(agent_deltas.values())

        suggestion = generate_thesis_suggestion(research_symbol, agent_outputs, total_conviction_delta)
        click.echo(f"Research complete. Conviction delta: {total_conviction_delta:+.1f}\n")

    # Use suggestion values as defaults if not explicitly provided
    if suggestion:
        symbol = symbol or suggestion.get("symbol")
        hypothesis = hypothesis or suggestion.get("hypothesis", "")
        business_quality = business_quality or suggestion.get("business_quality")
        valuation = valuation or suggestion.get("valuation")
        if not risks and suggestion.get("key_risks"):
            risks = ",".join(suggestion["key_risks"])

    # Validate symbol if provided
    if symbol:
        symbol = validate_symbol(symbol)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    risk_list = [r.strip() for r in risks.split(",") if r.strip()] if risks else []

    horizon_dt = None
    if horizon_end:
        try:
            horizon_dt = dt.strptime(horizon_end, "%Y-%m-%d")
        except ValueError:
            click.echo("Invalid date format. Use YYYY-MM-DD.", err=True)
            raise SystemExit(1)

    # Set initial conviction from research if available
    initial_conviction = 50.0
    if suggestion:
        initial_conviction = suggestion.get("conviction", 50.0)

    t = Thesis(
        title=title,
        hypothesis=hypothesis,
        conviction=initial_conviction,
        tags=tag_list,
        symbol=symbol.upper() if symbol else None,
        business_quality=business_quality,
        valuation=Valuation(valuation) if valuation else None,
        moat=moat,
        time_horizon=TimeHorizon(time_horizon) if time_horizon else None,
        horizon_end=horizon_dt,
        key_risks=risk_list,
        target_price=target_price,
        stop_price=stop_price,
    )
    repo.create(t)
    click.echo(f"Created thesis {t.id}: {t.title}")
    if suggestion:
        click.echo(f"  Symbol: {t.symbol}, Conviction: {t.conviction:.0f}%")
        if t.valuation:
            click.echo(f"  Valuation: {t.valuation.value}")
        if t.key_risks:
            click.echo(f"  Risks: {', '.join(t.key_risks[:3])}")


@thesis.command("list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["open", "closed", "invalidated"]),
    help="Filter by status",
)
@click.option("--symbol", "-S", help="Filter by symbol")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def thesis_list(status: str | None, symbol: str | None, output: str | None):
    """List theses."""
    if output is None:
        output = get_settings_lazy().default_output

    repo = ThesisRepository()
    status_filter = ThesisStatus(status) if status else None
    theses = repo.list(status=status_filter)

    # Filter by symbol if specified
    if symbol:
        symbol_upper = symbol.upper()
        theses = [t for t in theses if t.symbol and t.symbol.upper() == symbol_upper]

    if output == "json":
        result = [
            {
                "id": t.id,
                "title": t.title,
                "symbol": t.symbol,
                "status": t.status.value,
                "conviction": t.conviction,
                "valuation": t.valuation.value if t.valuation else None,
                "time_horizon": t.time_horizon.value if t.time_horizon else None,
                "outcome": t.outcome.value if t.outcome else None,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in theses
        ]
        click.echo(json.dumps(result, indent=2))
        return

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
    if t.outcome:
        click.echo(f"Outcome:    {t.outcome.value}")
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
    if t.target_price is not None:
        click.echo(f"Target:     ${t.target_price:.2f}")
    if t.stop_price is not None:
        click.echo(f"Stop:       ${t.stop_price:.2f}")
    if t.key_risks:
        click.echo(f"Risks:      {', '.join(t.key_risks)}")
    if t.tags:
        click.echo(f"Tags:       {', '.join(t.tags)}")
    click.echo(f"Created:    {t.created_at.strftime('%Y-%m-%d %H:%M')}")
    if t.closed_at:
        click.echo(f"Closed:     {t.closed_at.strftime('%Y-%m-%d %H:%M')}")
    click.echo(f"Updated:    {t.updated_at.strftime('%Y-%m-%d %H:%M')}")


@thesis.command("update")
@click.argument("thesis_id")
@click.option("--conviction", "-c", type=float, help="Set conviction (0-100)")
@click.option("--status", "-s", type=click.Choice(["open", "closed", "invalidated"]))
@click.option("--symbol", "-S", help="Stock ticker")
@click.option("--business-quality", "-b", help="Quality assessment")
@click.option("--valuation", "-v", type=click.Choice(["undervalued", "fair", "overvalued"]))
@click.option("--moat", "-m", help="Competitive advantage")
@click.option("--time-horizon", "-H", type=click.Choice(["short", "medium", "long"]))
@click.option("--horizon-end", help="Expiry date (YYYY-MM-DD)")
@click.option("--risks", "-r", help="Comma-separated key risks")
@click.option("--target-price", type=float, help="Target price")
@click.option("--stop-price", type=float, help="Stop price")
def thesis_update(
    thesis_id: str,
    conviction: float | None,
    status: str | None,
    symbol: str | None,
    business_quality: str | None,
    valuation: str | None,
    moat: str | None,
    time_horizon: str | None,
    horizon_end: str | None,
    risks: str | None,
    target_price: float | None,
    stop_price: float | None,
):
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
    if target_price is not None:
        t.target_price = target_price if target_price > 0 else None
    if stop_price is not None:
        t.stop_price = stop_price if stop_price > 0 else None

    repo.update(t)
    click.echo(f"Updated thesis {t.id}")


@thesis.command("close")
@click.argument("thesis_id")
@click.option(
    "--outcome",
    "-o",
    type=click.Choice(["correct", "incorrect", "partial", "unclear"]),
    help="Was the thesis correct?",
)
@click.option("--notes", "-n", help="Closing notes (updates hypothesis)")
def thesis_close(thesis_id: str, outcome: str | None, notes: str | None):
    """Close a thesis with outcome assessment."""
    repo = ThesisRepository()
    t = repo.get(thesis_id)

    if t is None:
        click.echo(f"Thesis {thesis_id} not found.", err=True)
        raise SystemExit(1)

    if t.status == ThesisStatus.CLOSED:
        click.echo(f"Thesis {thesis_id} is already closed.", err=True)
        raise SystemExit(1)

    outcome_enum = ThesisOutcome(outcome) if outcome else None
    t.close(outcome_enum)
    if notes:
        t.hypothesis = f"{t.hypothesis}\n\n[Closing notes]: {notes}" if t.hypothesis else f"[Closing notes]: {notes}"

    repo.update(t)
    outcome_str = f" ({outcome})" if outcome else ""
    click.echo(f"Closed thesis {t.id}{outcome_str}")
