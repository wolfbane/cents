"""LLM usage reporting CLI commands."""

import json
from datetime import datetime, timedelta

import click

from cents.db import LLMUsageRepository
from cents.pricing import estimate_cost_usd

from ._shared import default_subcommand


@default_subcommand("summary")
def usage(ctx):
    """Report on LLM token usage and cost."""


def _format_cost(cost: float | None) -> str:
    if cost is None:
        return "-"
    return f"${cost:.4f}"


@usage.command("summary")
@click.option(
    "--since-days",
    type=int,
    default=30,
    show_default=True,
    help="Window in days.",
)
@click.option(
    "--by",
    type=click.Choice(["agent", "model", "day", "operation"]),
    default="agent",
    show_default=True,
    help="Dimension to aggregate by.",
)
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def usage_summary(since_days: int, by: str, output: str):
    """Aggregate LLM usage by agent, model, day, or operation."""
    repo = LLMUsageRepository()
    since = datetime.now() - timedelta(days=since_days)
    rows = repo.aggregate(by, since=since)

    # `aggregate` returns one row per (bucket, model) so cost can use the
    # right rate for each. Collapse to one row per bucket for display.
    collapsed: dict[str, dict] = {}
    for row in rows:
        bucket = row["bucket"]
        cost = estimate_cost_usd(
            row["model"],
            row["input_tokens"],
            row["output_tokens"],
            cache_read=row["cache_read"],
            cache_write=row["cache_write"],
        )
        agg = collapsed.setdefault(
            bucket,
            {
                "bucket": bucket,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read": 0,
                "cache_write": 0,
                "est_cost_usd": 0.0,
                "_cost_known": True,
            },
        )
        agg["calls"] += row["calls"]
        agg["input_tokens"] += row["input_tokens"]
        agg["output_tokens"] += row["output_tokens"]
        agg["cache_read"] += row["cache_read"]
        agg["cache_write"] += row["cache_write"]
        if cost is None:
            agg["_cost_known"] = False
        else:
            agg["est_cost_usd"] += cost

    results = sorted(collapsed.values(), key=lambda r: r["calls"], reverse=True)

    if output == "json":
        payload = [
            {
                by: r["bucket"],
                "calls": r["calls"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cache_read": r["cache_read"],
                "cache_write": r["cache_write"],
                "est_cost_usd": round(r["est_cost_usd"], 6) if r["_cost_known"] else None,
            }
            for r in results
        ]
        click.echo(json.dumps(payload, indent=2))
        return

    if not results:
        click.echo("No usage recorded in window.")
        return

    header = f"{by:<20} {'calls':>6} {'in':>10} {'out':>10} {'cache_r':>10} {'cache_w':>10} {'cost':>10}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in results:
        cost_str = _format_cost(r["est_cost_usd"] if r["_cost_known"] else None)
        click.echo(
            f"{str(r['bucket']):<20} "
            f"{r['calls']:>6} "
            f"{r['input_tokens']:>10} "
            f"{r['output_tokens']:>10} "
            f"{r['cache_read']:>10} "
            f"{r['cache_write']:>10} "
            f"{cost_str:>10}"
        )


@usage.command("list")
@click.option(
    "--since-days",
    type=int,
    default=7,
    show_default=True,
    help="Window in days.",
)
@click.option("--limit", type=int, default=50, show_default=True)
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def usage_list(since_days: int, limit: int, output: str):
    """Show individual recent LLM calls, newest first."""
    repo = LLMUsageRepository()
    since = datetime.now() - timedelta(days=since_days)
    rows = repo.list_recent(since=since, limit=limit)

    if output == "json":
        payload = [
            {
                "id": r.id,
                "called_at": r.called_at.isoformat(),
                "model": r.model,
                "agent": r.agent,
                "operation": r.operation,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_read": r.cache_read_input_tokens,
                "cache_write": r.cache_creation_input_tokens,
                "context": r.context,
                "est_cost_usd": estimate_cost_usd(
                    r.model,
                    r.input_tokens,
                    r.output_tokens,
                    cache_read=r.cache_read_input_tokens,
                    cache_write=r.cache_creation_input_tokens,
                ),
            }
            for r in rows
        ]
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    if not rows:
        click.echo("No usage recorded in window.")
        return

    for r in rows:
        when = r.called_at.strftime("%m-%d %H:%M")
        cost = estimate_cost_usd(
            r.model,
            r.input_tokens,
            r.output_tokens,
            cache_read=r.cache_read_input_tokens,
            cache_write=r.cache_creation_input_tokens,
        )
        cost_str = _format_cost(cost)
        ctx_str = f" ({r.context})" if r.context else ""
        click.echo(
            f"{r.id} {when} {r.agent}.{r.operation}{ctx_str} "
            f"in={r.input_tokens} out={r.output_tokens} {cost_str}"
        )
