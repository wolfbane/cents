"""Portfolio-level drawdown tracking + kill switch.

The Risk reviewer's headline: "an autonomous loop with no global stop is the
textbook tail-risk profile." This module computes current portfolio
drawdown and daily realized loss, and exposes a single ``check_kill_switch``
hook for the factory engine to gate its open phase.

The drawdown is measured against the peak cost-basis of currently-open
positions (a simple definition that works for paper, where there is no
external cash account). Realized daily loss sums P&L of positions closed
today against their cost basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DrawdownState:
    """Snapshot used both for gating and for persisting to factory_runs."""

    open_cost_basis_usd: float
    open_market_value_usd: float
    unrealized_pnl_usd: float
    unrealized_drawdown_pct: float
    realized_pnl_today_usd: float
    realized_loss_today_pct: float
    gate_open: bool
    gate_reason: str | None = None


def compute_drawdown(
    *,
    open_positions: list,
    closed_today: list,
    price_provider,
) -> DrawdownState:
    """Compute portfolio drawdown state from raw positions + a price provider.

    Pure function — does not gate. Use ``check_kill_switch`` to apply config thresholds.
    """
    cost_basis = 0.0
    market_value = 0.0
    for pos in open_positions:
        cb = pos.entry_price * pos.size
        mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
        mv = mark * pos.size
        cost_basis += cb
        if getattr(pos.side, "value", pos.side) == "short":
            # Short P&L flips sign — market value contributes inversely.
            market_value += cb + (cb - mv)
        else:
            market_value += mv

    unrealized_pnl = market_value - cost_basis
    unrealized_dd_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0

    realized_today = 0.0
    realized_cost_basis = 0.0
    for pos in closed_today:
        pnl = pos.pnl if pos.pnl is not None else 0.0
        realized_today += pnl
        realized_cost_basis += pos.entry_price * pos.size
    realized_pct = (
        realized_today / realized_cost_basis * 100.0 if realized_cost_basis > 0 else 0.0
    )

    return DrawdownState(
        open_cost_basis_usd=cost_basis,
        open_market_value_usd=market_value,
        unrealized_pnl_usd=unrealized_pnl,
        unrealized_drawdown_pct=unrealized_dd_pct,
        realized_pnl_today_usd=realized_today,
        realized_loss_today_pct=realized_pct,
        gate_open=True,
        gate_reason=None,
    )


def check_kill_switch(
    state: DrawdownState,
    *,
    max_portfolio_drawdown_pct: float,
    max_daily_loss_pct: float,
) -> DrawdownState:
    """Apply config thresholds. Returns a copy of ``state`` with gate fields set.

    ``max_portfolio_drawdown_pct`` and ``max_daily_loss_pct`` are positive
    numbers — the function compares to negative actuals (i.e. a -10 unrealized
    DD against a 10.0 cap triggers the gate).
    """
    reasons: list[str] = []
    if state.unrealized_drawdown_pct <= -abs(max_portfolio_drawdown_pct):
        reasons.append(
            f"portfolio drawdown {state.unrealized_drawdown_pct:.2f}% breaches "
            f"max {max_portfolio_drawdown_pct:.2f}%"
        )
    if state.realized_loss_today_pct <= -abs(max_daily_loss_pct):
        reasons.append(
            f"realized loss today {state.realized_loss_today_pct:.2f}% breaches "
            f"max {max_daily_loss_pct:.2f}%"
        )
    if reasons:
        return DrawdownState(
            open_cost_basis_usd=state.open_cost_basis_usd,
            open_market_value_usd=state.open_market_value_usd,
            unrealized_pnl_usd=state.unrealized_pnl_usd,
            unrealized_drawdown_pct=state.unrealized_drawdown_pct,
            realized_pnl_today_usd=state.realized_pnl_today_usd,
            realized_loss_today_pct=state.realized_loss_today_pct,
            gate_open=False,
            gate_reason="; ".join(reasons),
        )
    return state
