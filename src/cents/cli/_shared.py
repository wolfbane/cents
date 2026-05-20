"""Shared utilities for CLI commands."""

import json
import re
from datetime import date, datetime, timedelta
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
                # The decorator stamps a click.Group via @click.group above,
                # but the type system sees ctx.command as click.Command. Cast
                # so mypy and any reader knows the get_command call is valid.
                group = ctx.command
                if isinstance(group, click.Group):
                    default = group.get_command(ctx, default_command)
                    if default is not None:
                        return ctx.invoke(default)
            return func(ctx, *args, **kwargs)

        return wrapper

    return decorator


def parse_date(value: str, label: str) -> date:
    """Parse a YYYY-MM-DD date string.

    Args:
        value: String to parse.
        label: Human label for error messages ("start"/"end").

    Returns:
        Parsed ``date`` object.

    Raises:
        SystemExit: If the date is invalid; prints a click error first.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid {label} date: {value}. Use YYYY-MM-DD.", err=True)
        raise SystemExit(1)


def parse_date_range(start_str: str, end_str: str | None, default_end_days: int = 60) -> tuple[date, date]:
    """Parse a start/end date range with validation.

    Args:
        start_str: Start date string.
        end_str: Optional end date string.
        default_end_days: Days before today to use when end_str is absent.

    Returns:
        Tuple of (start_date, end_date).
    """
    start_date = parse_date(start_str, "start")

    if end_str:
        end_date = parse_date(end_str, "end")
    else:
        end_date = date.today() - timedelta(days=default_end_days)

    if start_date >= end_date:
        click.echo("Start date must be before end date.", err=True)
        raise SystemExit(1)

    return start_date, end_date


def parse_symbols(symbol: str | None, symbols_str: str | None) -> list[str]:
    """Parse symbols from either positional arg or comma-separated option."""
    if symbols_str:
        return [validate_symbol(s.strip()) for s in symbols_str.split(",")]
    if symbol:
        return [validate_symbol(symbol)]

    click.echo("Specify a symbol or use --symbols.", err=True)
    raise SystemExit(1)


def parse_agents(agent_names: str | None, available_agents: dict[str, Any]) -> dict[str, Any]:
    """Parse a comma-separated list of agents into a mapping."""
    if not agent_names:
        return available_agents

    selected: dict[str, Any] = {}
    for name in agent_names.split(","):
        name = name.strip()
        if name not in available_agents:
            click.echo(
                f"Unknown agent: {name}. Available: {', '.join(available_agents.keys())}",
                err=True,
            )
            raise SystemExit(1)
        selected[name] = available_agents[name]
    return selected


def render_output(output: str, text_renderer: Callable[[], None], data: Any):
    """Render either JSON (indent=2) or text via callback."""
    if output == "json":
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        text_renderer()


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
                    # Sector-adjusted thresholds match the FundamentalsAgent's
                    # evidence-typing rule. Previously the CLI hardcoded
                    # <15/>30 regardless of sector, so an Energy name (sector
                    # median ~12) routinely surfaced as "fair" when the agent
                    # saw it as "overvalued" — silent drift between the user-
                    # facing suggestion and the engine's actual conviction.
                    from cents.agents.fundamentals import pe_thresholds_for_sector
                    pe_low, pe_high = pe_thresholds_for_sector(metadata.get("sector"))
                    if value < pe_low:
                        suggestion["valuation"] = "undervalued"
                    elif value > pe_high:
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


def calculate_correlation(x: list[float], y: list[float]) -> float | None:
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


def calculate_hit_rate(deltas: list[float], returns: list[float]) -> float | None:
    """Calculate hit rate: % of NON-NEUTRAL signals where delta sign matches return sign.

    Neutral signals (delta == 0) are "no prediction" and excluded from BOTH the
    numerator AND denominator. The previous implementation counted them in the
    denominator only, which forced agents that emit delta=0 most of the time
    (like ``insider``, which has no cluster activity in most monthly snapshots)
    to score absurdly low hit rates (~1-2%) instead of reflecting their actual
    accuracy when they DO have a view.

    Returns None when no non-neutral signals exist.
    """
    if not deltas or len(deltas) != len(returns):
        return None

    non_neutral = [(d, r) for d, r in zip(deltas, returns) if d != 0]
    if not non_neutral:
        return None

    hits = sum(1 for d, r in non_neutral if (d > 0 and r > 0) or (d < 0 and r < 0))
    return hits / len(non_neutral)
