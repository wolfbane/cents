"""Shared utilities for CLI commands."""

import json
import re
from functools import wraps
from typing import Any, Callable, NoReturn, TypedDict

import click

from cents.config import get_settings


class ThesisSuggestion(TypedDict):
    """Type definition for thesis suggestion returned by generate_thesis_suggestion."""

    symbol: str
    title: str
    hypothesis: str
    business_quality: str | None
    valuation: str | None
    key_risks: list[str]
    conviction: float


def get_settings_lazy():
    """Lazy-load settings to avoid import-time configuration errors."""
    return get_settings()


def resolve_output_format(output: str | None) -> str:
    """Resolve output format with config fallback."""

    if output is None:
        return get_settings_lazy().default_output
    return output


def echo_json(payload: Any) -> None:
    """Pretty-print JSON payload consistently."""

    click.echo(json.dumps(payload, indent=2))


def exit_with_error(message: str) -> NoReturn:
    """Emit a standardized error message and exit."""

    click.echo(message, err=True)
    raise SystemExit(1)


def respond_with_output(
    output: str,
    json_payload: Any,
    text_printer: Callable[[], None],
    *,
    quiet: bool = False,
    quiet_message: str | None = None,
) -> None:
    """Respond in JSON or text form, honoring quiet flags."""

    if output == "json":
        echo_json(json_payload)
        return

    if quiet and quiet_message is not None:
        click.echo(quiet_message)
        return

    text_printer()


def default_subcommand(default_command: str):
    """Decorator to make click groups fall back to a default subcommand."""

    def decorator(func: Callable):
        @click.group(name=func.__name__, invoke_without_command=True)
        @click.pass_context
        @wraps(func)
        def wrapper(ctx: click.Context, *args, **kwargs):
            if ctx.invoked_subcommand is None:
                default = ctx.command.get_command(ctx, default_command)
                if default is not None:
                    return ctx.invoke(default)
            return func(ctx, *args, **kwargs)

        return wrapper

    return decorator


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a stock symbol.

    Args:
        symbol: Raw symbol input from user

    Returns:
        Normalized uppercase symbol

    Raises:
        click.BadParameter: If symbol is invalid
    """
    s = symbol.strip().upper()
    if not s:
        raise click.BadParameter("Symbol cannot be empty")
    if len(s) > 10:
        raise click.BadParameter(f"Symbol too long: {symbol}")
    if not re.match(r"^[A-Z0-9\.\-]+$", s):
        raise click.BadParameter(f"Invalid symbol characters: {symbol}")
    return s


def generate_thesis_suggestion(
    symbol: str, agent_outputs: list[dict[str, Any]], total_conviction_delta: float
) -> ThesisSuggestion:
    """Generate thesis field suggestions from research agent outputs.

    Analyzes agent research results to suggest initial values for thesis fields
    like valuation assessment, business quality notes, and key risks.

    Args:
        symbol: Stock ticker symbol (will be uppercased)
        agent_outputs: List of dicts with keys 'agent', 'summary', 'evidence'.
            Each evidence item has 'metadata' with metric-specific values.
        total_conviction_delta: Aggregated conviction change from all agents

    Returns:
        ThesisSuggestion with populated fields based on research findings
    """
    suggestion = {
        "symbol": symbol.upper(),
        "title": f"{symbol.upper()} investment thesis",
        "hypothesis": "",
        "business_quality": None,
        "valuation": None,
        "key_risks": [],
        "conviction": 50.0 + total_conviction_delta,
    }

    hypotheses = []
    quality_notes = []
    risks = []

    for output in agent_outputs:
        agent = output["agent"]
        summary = output["summary"]
        evidence_list = output.get("evidence", [])

        # Add summary to hypothesis
        if summary and "No data" not in summary and "Failed" not in summary:
            hypotheses.append(f"[{agent}] {summary}")

        # Extract valuation from fundamentals
        if agent == "fundamentals":
            for ev in evidence_list:
                metadata = ev.get("metadata", {})
                metric = metadata.get("metric")
                value = metadata.get("value")

                if metric == "pe_ratio" and value:
                    if value < 15:
                        suggestion["valuation"] = "undervalued"
                    elif value > 30:
                        suggestion["valuation"] = "overvalued"
                    else:
                        suggestion["valuation"] = "fair"

                if metric == "profit_margin" and value:
                    margin_pct = value * 100 if abs(value) < 1 else value
                    if margin_pct > 20:
                        quality_notes.append(f"High margins ({margin_pct:.0f}%)")
                    elif margin_pct < 5:
                        quality_notes.append(f"Low margins ({margin_pct:.0f}%)")

                if metric == "revenue_growth" and value:
                    growth_pct = value * 100 if abs(value) < 10 else value
                    if growth_pct > 20:
                        quality_notes.append(f"Strong growth ({growth_pct:.0f}%)")
                    elif growth_pct < 0:
                        risks.append(f"Declining revenue ({growth_pct:.0f}%)")

                if metric == "debt_to_equity" and value:
                    d_e = value * 100 if value < 10 else value
                    if d_e > 200:
                        risks.append(f"High debt (D/E: {d_e:.0f}%)")

        # Extract risks from contradicting evidence
        for ev in evidence_list:
            if ev.get("type") == "contradicting":
                content = ev.get("content", "")
                if content and content not in risks:
                    risks.append(content)

    # Build final suggestion
    suggestion["hypothesis"] = "\n".join(hypotheses) if hypotheses else ""
    suggestion["business_quality"] = "; ".join(quality_notes) if quality_notes else None
    suggestion["key_risks"] = risks[:5]  # Limit to 5 risks
    suggestion["conviction"] = max(0, min(100, suggestion["conviction"]))

    return suggestion
