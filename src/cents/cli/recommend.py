"""Recommend command - rule-based decision engine."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import click

from cents.db import ThesisRepository, PositionRepository
from cents.models import ThesisStatus, PositionStatus

from ._shared import resolve_output_format, respond_with_output

logger = logging.getLogger(__name__)


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REVIEW = "review"


@dataclass
class Recommendation:
    """A recommended action for a thesis/position."""

    symbol: str
    action: Action
    reason: str
    thesis_id: str | None
    current_price: float | None
    conviction: float
    priority: int  # 1=urgent, 2=normal, 3=hold
    target_price: float | None = None
    stop_price: float | None = None
    position_id: str | None = None
    position_size: float | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "reason": self.reason,
            "thesis_id": self.thesis_id,
            "current_price": self.current_price,
            "conviction": self.conviction,
            "priority": self.priority,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "position_id": self.position_id,
            "position_size": self.position_size,
        }


def evaluate_thesis(
    thesis,
    position,
    current_price: float | None,
    buy_threshold: float,
    sell_threshold: float,
) -> Recommendation:
    """Apply rules to generate a recommendation for a thesis."""

    symbol = thesis.symbol or "UNKNOWN"
    base = {
        "symbol": symbol,
        "thesis_id": thesis.id,
        "current_price": current_price,
        "conviction": thesis.conviction,
        "target_price": thesis.target_price,
        "stop_price": thesis.stop_price,
        "position_id": position.id if position else None,
        "position_size": position.size if position else None,
    }

    # RULE 1: Stop loss triggered (highest priority)
    if position and thesis.stop_price and current_price:
        if current_price <= thesis.stop_price:
            return Recommendation(
                action=Action.CLOSE,
                reason=f"Stop loss triggered (${current_price:.2f} <= ${thesis.stop_price:.2f})",
                priority=1,
                **base,
            )

    # RULE 2: Target price reached - take profit
    if position and thesis.target_price and current_price:
        if current_price >= thesis.target_price:
            return Recommendation(
                action=Action.CLOSE,
                reason=f"Target reached (${current_price:.2f} >= ${thesis.target_price:.2f})",
                priority=1,
                **base,
            )

    # RULE 3: Thesis invalidated
    if thesis.status == ThesisStatus.INVALIDATED:
        if position:
            return Recommendation(
                action=Action.CLOSE,
                reason="Thesis invalidated",
                priority=1,
                **base,
            )
        return Recommendation(
            action=Action.REVIEW,
            reason="Thesis invalidated (no position)",
            priority=2,
            **base,
        )

    # RULE 4: Thesis expired
    if thesis.horizon_end and thesis.horizon_end < datetime.now():
        days_ago = (datetime.now() - thesis.horizon_end).days
        if position:
            return Recommendation(
                action=Action.REVIEW,
                reason=f"Thesis expired {days_ago}d ago, has open position",
                priority=1,
                **base,
            )
        return Recommendation(
            action=Action.REVIEW,
            reason=f"Thesis expired {days_ago}d ago",
            priority=3,
            **base,
        )

    # RULE 5: Low conviction with position - sell signal
    if position and thesis.conviction < sell_threshold:
        return Recommendation(
            action=Action.SELL,
            reason=f"Conviction {thesis.conviction:.0f}% below {sell_threshold:.0f}% threshold",
            priority=2,
            **base,
        )

    # RULE 6: High conviction without position - buy signal
    if not position and thesis.conviction >= buy_threshold:
        # Check if there's still upside
        if thesis.target_price and current_price:
            upside = (thesis.target_price - current_price) / current_price * 100
            if upside > 5:  # At least 5% upside
                return Recommendation(
                    action=Action.BUY,
                    reason=f"Conviction {thesis.conviction:.0f}%, {upside:.0f}% upside to target",
                    priority=2,
                    **base,
                )
            else:
                return Recommendation(
                    action=Action.REVIEW,
                    reason=f"High conviction but only {upside:.1f}% upside remaining",
                    priority=3,
                    **base,
                )
        else:
            return Recommendation(
                action=Action.BUY,
                reason=f"Conviction {thesis.conviction:.0f}% (no target set)",
                priority=2,
                **base,
            )

    # RULE 7: Default - hold
    return Recommendation(
        action=Action.HOLD,
        reason="Thesis intact, no action needed",
        priority=3,
        **base,
    )


def get_recommendations(
    buy_threshold: float = 70.0,
    sell_threshold: float = 30.0,
) -> list[Recommendation]:
    """Generate recommendations for all open theses."""

    thesis_repo = ThesisRepository()
    position_repo = PositionRepository()

    # Get all open theses
    theses = [t for t in thesis_repo.list() if t.status == ThesisStatus.OPEN]

    if not theses:
        return []

    # Get current prices
    symbols = [t.symbol for t in theses if t.symbol]
    prices: dict[str, float] = {}
    if symbols:
        try:
            from cents.data.alpaca import get_price_provider

            provider = get_price_provider()
            prices = provider.get_latest_prices(symbols)
        except Exception as e:
            logger.debug("Could not fetch prices: %s", e)

    # Get open positions indexed by thesis_id
    positions = position_repo.list(status=PositionStatus.OPEN)
    positions_by_thesis = {p.thesis_id: p for p in positions if p.thesis_id}

    recommendations = []
    for thesis in theses:
        position = positions_by_thesis.get(thesis.id)
        price = prices.get(thesis.symbol) if thesis.symbol else None

        rec = evaluate_thesis(
            thesis, position, price, buy_threshold, sell_threshold
        )
        recommendations.append(rec)

    # Sort by priority, then by action importance
    action_order = {Action.CLOSE: 0, Action.SELL: 1, Action.BUY: 2, Action.REVIEW: 3, Action.HOLD: 4}
    recommendations.sort(key=lambda r: (r.priority, action_order.get(r.action, 5)))

    return recommendations


@click.command("recommend")
@click.option(
    "--buy-threshold",
    type=float,
    default=70.0,
    help="Minimum conviction to recommend BUY (default: 70)",
)
@click.option(
    "--sell-threshold",
    type=float,
    default=30.0,
    help="Maximum conviction to recommend SELL (default: 30)",
)
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format (default: from config)",
)
@click.option(
    "--actionable",
    is_flag=True,
    help="Only show actionable recommendations (exclude HOLD)",
)
def recommend(
    buy_threshold: float,
    sell_threshold: float,
    output: str | None,
    actionable: bool,
):
    """Generate buy/sell/hold recommendations based on thesis rules.

    Evaluates all open theses against current prices and conviction levels
    to produce actionable recommendations.

    Rules (in priority order):
      1. Stop loss triggered → CLOSE
      2. Target price reached → CLOSE
      3. Thesis invalidated → CLOSE/REVIEW
      4. Thesis expired → REVIEW
      5. Low conviction with position → SELL
      6. High conviction without position → BUY
      7. Otherwise → HOLD
    """
    output = resolve_output_format(output)

    recommendations = get_recommendations(buy_threshold, sell_threshold)

    respond_with_output(
        output,
        [r.to_dict() for r in recommendations],
        lambda: _print_recommendations(recommendations, actionable),
    )


def _print_recommendations(recommendations: list[Recommendation], actionable: bool) -> None:
    """Render recommendations in text form."""
    if actionable:
        recommendations = [r for r in recommendations if r.action != Action.HOLD]

    if not recommendations:
        click.echo("No open theses to evaluate.")
        return

    # Group by priority
    urgent = [r for r in recommendations if r.priority == 1]
    normal = [r for r in recommendations if r.priority == 2]
    holds = [r for r in recommendations if r.priority == 3 and r.action == Action.HOLD]
    reviews = [r for r in recommendations if r.priority == 3 and r.action == Action.REVIEW]

    def format_rec(r: Recommendation) -> str:
        price_str = f"${r.current_price:.2f}" if r.current_price else "N/A"
        action_str = r.action.value.upper().ljust(6)
        return f"  {action_str} {r.symbol.ljust(6)} {price_str.rjust(10)}  {r.reason}"

    if urgent:
        click.echo(click.style("ACTION (review):", fg="red", bold=True))
        for r in urgent:
            click.echo(format_rec(r))
        click.echo()

    if normal:
        click.echo(click.style("RECOMMENDATIONS:", fg="yellow", bold=True))
        for r in normal:
            click.echo(format_rec(r))
        click.echo()

    if reviews and not actionable:
        click.echo(click.style("REVIEW:", fg="cyan"))
        for r in reviews:
            click.echo(format_rec(r))
        click.echo()

    if holds and not actionable:
        click.echo(click.style("HOLD (no action):", fg="green"))
        for r in holds:
            click.echo(format_rec(r))

    # Summary
    action_counts = {}
    for r in recommendations:
        action_counts[r.action.value] = action_counts.get(r.action.value, 0) + 1

    summary = ", ".join(f"{v} {k}" for k, v in action_counts.items())
    click.echo(f"\nSummary: {summary}")
