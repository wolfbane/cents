"""Research CLI command."""
import logging
from datetime import date, datetime
from pathlib import Path

import click

from cents.agents import AGENTS
from cents.data.alpaca import get_price_provider
from cents.db import ThesisRepository, EvidenceRepository
from cents.exceptions import CostCapExceeded
from cents.llm_usage import cost_cap

from cents.serialization import serialize
from ._research_html import EVIDENCE_ICONS, render_research_html
from ._shared import (
    echo_json,
    exit_with_error,
    generate_thesis_suggestion,
    parse_date,
    resolve_output_format,
    validate_symbol,
)

logger = logging.getLogger(__name__)


def _capture_research_result(
    name: str,
    result,
    verbose: bool,
    agent_outputs: list,
    agent_deltas: dict,
    all_evidence: list,
) -> None:
    """Append one agent's result to the in-progress research aggregates."""
    if verbose:
        click.echo(f"--- {name.upper()} ---")
        click.echo(f"Summary: {result.summary}")
        click.echo(f"Conviction delta: {result.conviction_delta:+.1f}")

        if result.evidence:
            click.echo("Evidence:")
            for e in result.evidence:
                icon = EVIDENCE_ICONS.get(e.type.value, "~")
                click.echo(f"  [{icon}] {e.content}")
        click.echo()

    agent_outputs.append(
        {
            "agent": name,
            "summary": result.summary,
            "conviction_delta": result.conviction_delta,
            "dimension_scores": dict(result.dimension_scores),
            "evidence": [serialize(e) for e in result.evidence],
        }
    )

    agent_deltas[name] = result.conviction_delta
    all_evidence.extend(result.evidence)


@click.command("research")
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
    default=None,
    help="Output format for results (default: from config)",
)
@click.option("--quiet", is_flag=True, help="Suppress verbose logs for scripting")
@click.option("--suggest-thesis", is_flag=True, help="Generate thesis suggestion from research")
@click.option(
    "--as-of",
    "as_of_str",
    type=str,
    default=None,
    help="Historical date for backtesting (YYYY-MM-DD format)",
)
@click.option(
    "--export-html",
    "export_html_path",
    type=click.Path(dir_okay=False, resolve_path=False),
    default=None,
    metavar="PATH",
    help="Export a self-contained HTML report (e.g., --export-html /tmp/report.html)",
)
@click.option(
    "--max-cost-usd",
    "max_cost_usd",
    type=float,
    default=None,
    help=(
        "Abort this research call if cumulative LLM spend would exceed this many "
        "USD. Checked PRE-call against a token estimate."
    ),
)
def research(
    symbol: str,
    thesis_id: str | None,
    agent_name: str | None,
    save: bool,
    output: str | None,
    quiet: bool,
    suggest_thesis: bool,
    as_of_str: str | None,
    export_html_path: str | None,
    max_cost_usd: float | None,
):
    """Run research agents on a symbol."""
    symbol = validate_symbol(symbol)
    output = resolve_output_format(output)
    verbose = output == "text" and not quiet

    as_of: date | None = None
    if as_of_str:
        as_of = parse_date(as_of_str, "as-of")
        if verbose:
            click.echo(f"Historical analysis as of: {as_of}\n")

    thesis = None
    if thesis_id:
        thesis_repo = ThesisRepository()
        thesis = thesis_repo.get(thesis_id)
        if thesis is None:
            exit_with_error(f"Thesis {thesis_id} not found.")
        if verbose:
            click.echo(f"Evaluating against thesis: {thesis.title}\n")

    # Default to the orchestrator: it runs every agent internally and returns
    # the weighted aggregate. Running agents individually here would double-count.
    if agent_name:
        agents_to_run = {agent_name: AGENTS[agent_name]}
    else:
        agents_to_run = {"orchestrator": AGENTS["orchestrator"]}

    price: float | None = None
    try:
        provider = get_price_provider()
        price = provider.get_latest_price(symbol.upper(), as_of=as_of)
    except Exception as e:
        logger.debug("Could not fetch price: %s", e)

    if verbose and price:
        price_label = f"Price as of {as_of}" if as_of else "Current price"
        click.echo(f"{price_label}: ${price:.2f}\n")

    all_evidence = []
    agent_outputs = []
    agent_deltas: dict[str, float] = {}

    try:
        with cost_cap(max_cost_usd):
            for name, agent_class in agents_to_run.items():
                agent = agent_class()
                result = agent.research(symbol.upper(), thesis, as_of=as_of)
                _capture_research_result(
                    name, result, verbose, agent_outputs, agent_deltas, all_evidence
                )
    except CostCapExceeded as exc:
        exit_with_error(str(exc))
        return  # pragma: no cover

    # Orchestrator's delta is the weighted aggregate — prefer it over a sum.
    if "orchestrator" in agent_deltas:
        total_conviction_delta = agent_deltas["orchestrator"]
    else:
        total_conviction_delta = sum(agent_deltas.values())

    evidence_saved = False
    evidence_count = 0
    evidence_skipped = 0
    if save and all_evidence:
        evidence_repo = EvidenceRepository()
        for e in all_evidence:
            e.symbol = symbol.upper()
            if thesis:
                e.thesis_id = thesis.id
            if evidence_repo.create(e, dedupe=True):
                evidence_count += 1
            else:
                evidence_skipped += 1
        evidence_saved = evidence_count > 0

        if thesis and evidence_count > 0:
            # Scale the conviction delta by the share of new evidence so
            # repeated scans of the same data don't keep moving the needle.
            scale = evidence_count / (evidence_count + evidence_skipped)
            scaled_delta = total_conviction_delta * scale
            thesis_repo = ThesisRepository()
            thesis.update_conviction(scaled_delta)
            thesis_repo.update(thesis)

        if verbose:
            if evidence_skipped > 0:
                click.echo(f"Saved {evidence_count} evidence items ({evidence_skipped} duplicates skipped)")
            else:
                click.echo(f"Saved {evidence_count} evidence items")
            if thesis:
                click.echo(f"Thesis conviction: {thesis.conviction:.1f}% ({total_conviction_delta:+.1f})")
            elif evidence_count > 0:
                click.echo(f"Evidence saved for {symbol.upper()} (no thesis linked)")
                click.echo(f"Link later with: cents evidence link {symbol.upper()} --thesis <ID>")

    thesis_suggestion = None
    if suggest_thesis:
        thesis_suggestion = generate_thesis_suggestion(symbol, agent_outputs, total_conviction_delta)

    research_payload: dict = {
        "symbol": symbol.upper(),
        "price": price,
        "as_of": as_of.isoformat() if as_of else None,
        "thesis_id": thesis.id if thesis else None,
        "total_conviction_delta": total_conviction_delta,
        "agents": agent_outputs,
        "evidence_saved": evidence_saved,
        "evidence_count": len(all_evidence),
    }
    if thesis_suggestion:
        research_payload["thesis_suggestion"] = thesis_suggestion

    if export_html_path is not None:
        html_payload = {
            **research_payload,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            Path(export_html_path).write_text(
                render_research_html(html_payload), encoding="utf-8"
            )
        except OSError as e:
            exit_with_error(f"Failed to write HTML export to {export_html_path}: {e}")
        if verbose:
            click.echo(f"Exported HTML report to {export_html_path}")

    if output == "json":
        echo_json(research_payload)
    else:
        if quiet:
            click.echo(
                f"{symbol.upper()} conviction delta: {total_conviction_delta:+.1f}"
            )
        elif not verbose:
            click.echo(f"Total conviction delta: {total_conviction_delta:+.1f}")

        if thesis_suggestion and not quiet:
            click.echo("\n--- THESIS SUGGESTION ---")
            click.echo(f"Title:      {thesis_suggestion['title']}")
            click.echo(f"Symbol:     {thesis_suggestion['symbol']}")
            click.echo(f"Conviction: {thesis_suggestion['conviction']:.0f}%")
            if thesis_suggestion.get("valuation"):
                click.echo(f"Valuation:  {thesis_suggestion['valuation']}")
            if thesis_suggestion.get("business_quality"):
                click.echo(f"Quality:    {thesis_suggestion['business_quality']}")
            if thesis_suggestion.get("key_risks"):
                click.echo(f"Risks:      {', '.join(thesis_suggestion['key_risks'][:3])}")
            click.echo(f"\nCreate with: cents thesis create --title \"{thesis_suggestion['title']}\" --from-research {symbol.upper()}")
