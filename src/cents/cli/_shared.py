"""Shared utilities for CLI commands."""

import re
from typing import Optional

import click

from cents.config import get_settings


def get_settings_lazy():
    """Lazy-load settings to avoid import-time configuration errors."""
    return get_settings()


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


def generate_thesis_suggestion(symbol: str, agent_outputs: list, total_conviction_delta: float) -> dict:
    """Generate thesis field suggestions from research results."""
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


def evidence_to_dict(evidence):
    """Serialize Evidence objects for JSON output."""
    return {
        "type": evidence.type.value,
        "content": evidence.content,
        "source": evidence.source,
        "confidence": evidence.confidence,
        "metadata": evidence.metadata,
    }
