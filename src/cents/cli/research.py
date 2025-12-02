"""Research CLI command."""

import json
from typing import Optional

import click

from cents.agents import AGENTS
from cents.db import ThesisRepository, EvidenceRepository

from ._shared import get_settings_lazy, validate_symbol, generate_thesis_suggestion, evidence_to_dict


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
def research(
    symbol: str,
    thesis_id: Optional[str],
    agent_name: Optional[str],
    save: bool,
    output: Optional[str],
    quiet: bool,
    suggest_thesis: bool,
):
    """Run research agents on a symbol."""
    symbol = validate_symbol(symbol)
    if output is None:
        output = get_settings_lazy().default_output
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

    all_evidence = []
    agent_outputs = []
    agent_deltas: dict[str, float] = {}

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
                "evidence": [evidence_to_dict(e) for e in result.evidence],
            }
        )

        agent_deltas[name] = result.conviction_delta
        all_evidence.extend(result.evidence)

    # Use orchestrator's delta if present (it's the weighted aggregate),
    # otherwise sum individual agent deltas
    if "orchestrator" in agent_deltas:
        total_conviction_delta = agent_deltas["orchestrator"]
    else:
        total_conviction_delta = sum(agent_deltas.values())

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

    # Generate thesis suggestion if requested
    thesis_suggestion = None
    if suggest_thesis:
        thesis_suggestion = generate_thesis_suggestion(symbol, agent_outputs, total_conviction_delta)

    if output == "json":
        payload = {
            "symbol": symbol.upper(),
            "thesis_id": thesis.id if thesis else None,
            "total_conviction_delta": total_conviction_delta,
            "agents": agent_outputs,
            "evidence_saved": evidence_saved,
            "evidence_count": len(all_evidence),
        }
        if thesis_suggestion:
            payload["thesis_suggestion"] = thesis_suggestion
        click.echo(json.dumps(payload, indent=2))
    else:
        if quiet:
            click.echo(
                f"{symbol.upper()} conviction delta: {total_conviction_delta:+.1f}"
            )
        elif not verbose:
            # This happens when output was coerced to text but quiet disabled
            click.echo(f"Total conviction delta: {total_conviction_delta:+.1f}")

        # Display thesis suggestion in text mode
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
