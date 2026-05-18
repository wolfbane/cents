"""Eval CLI — run the LLM eval harness against golden sets.

The harness exercises the LIVE Anthropic API; if ANTHROPIC_API_KEY is not
configured, the runner returns a result with `skipped_reason` set and the
command exits non-zero so CI / cron can detect missing credentials.

TODO(cron): wire `cents eval run` into a nightly job once golden sets stabilize
and we have a baseline to detect drift against.
"""

from __future__ import annotations

import click

from cents.eval.runner import (
    load_premise_golden,
    load_sentiment_golden,
    run_eval,
)
from cents.eval.report import (
    print_eval,
    print_premise_golden,
    print_sentiment_golden,
)

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


_SET_CHOICES = ["premise", "sentiment", "all"]


@default_subcommand("run")
def eval_(ctx):
    """Run the LLM eval harness against golden sets."""


@eval_.command("run")
@click.option(
    "--set",
    "set_name",
    type=click.Choice(_SET_CHOICES),
    default="all",
    help="Which eval to run (default: all).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap fixtures per set (handy for smoke-testing).",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def eval_run(set_name: str, limit: int | None, output: str | None):
    """Run the eval against the live Anthropic API."""
    output = resolve_output_format(output)
    result = run_eval(sets=set_name, limit=limit)

    # If both subevals are skipped (the only realistic skip path: missing key),
    # surface as a non-zero exit so CI / cron treat it as a failure.
    parts = [r for r in (result.premise, result.sentiment) if r is not None]
    all_skipped = parts and all(r.skipped_reason for r in parts)

    respond_with_output(
        output,
        result.to_dict(),
        lambda: print_eval(result),
    )
    if all_skipped:
        raise SystemExit(1)


_SET_CHOICES_GOLDEN = ["premise", "sentiment"]


@eval_.group("golden")
def golden():
    """Inspect the golden fixture sets."""


@golden.command("show")
@click.option(
    "--set",
    "set_name",
    type=click.Choice(_SET_CHOICES_GOLDEN),
    required=True,
    help="Which golden set to inspect.",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def golden_show(set_name: str, output: str | None):
    """List fixtures in a golden set."""
    output = resolve_output_format(output)
    if set_name == "premise":
        fixtures = load_premise_golden()
        respond_with_output(
            output,
            {"set": "premise", "count": len(fixtures), "fixtures": fixtures},
            lambda: print_premise_golden(fixtures),
        )
    elif set_name == "sentiment":
        fixtures = load_sentiment_golden()
        respond_with_output(
            output,
            {"set": "sentiment", "count": len(fixtures), "fixtures": fixtures},
            lambda: print_sentiment_golden(fixtures),
        )
    else:
        exit_with_error(f"Unknown set: {set_name}")
