"""Portfolio-level drawdown tracking + kill switch.

The Risk reviewer's headline: "an autonomous loop with no global stop is the
textbook tail-risk profile." This module computes current portfolio
drawdown and daily realized loss, and exposes a single ``check_kill_switch``
hook for the factory engine to gate its open phase.

The drawdown is measured against ``budget_usd`` — the configured total
capital. Measuring against currently-open cost basis instead makes the gate
*more* sensitive as the book shrinks, which is exactly backwards. Realized
daily loss sums P&L of positions closed today, also normalized to budget.

Paired-cohort theses own both a long and short leg under the same
``thesis_id``. Their cost-basis is netted (``abs(long_notional - short_notional)``)
so a $5k long + $5k short doesn't double-count as $10k of risk.
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


def _net_paired_cost_basis(open_positions: list) -> float:
    """Sum cost-basis with paired legs netted by ``thesis_id``.

    Positions without a ``thesis_id`` (or whose ``thesis_id`` is unique among
    open positions) contribute their full cost basis. Positions sharing a
    ``thesis_id`` are netted: ``abs(long_notional - short_notional)``.
    """
    by_thesis: dict[str, dict[str, float]] = {}
    unattached = 0.0
    for pos in open_positions:
        cb = pos.entry_price * pos.size
        tid = getattr(pos, "thesis_id", None)
        if not tid:
            unattached += cb
            continue
        bucket = by_thesis.setdefault(tid, {"long": 0.0, "short": 0.0})
        side_value = getattr(pos.side, "value", pos.side)
        if side_value == "short":
            bucket["short"] += cb
        else:
            bucket["long"] += cb
    netted = unattached
    for bucket in by_thesis.values():
        netted += abs(bucket["long"] - bucket["short"])
    return netted


def compute_drawdown(
    *,
    open_positions: list,
    closed_today: list,
    price_provider,
    budget_usd: float,
) -> DrawdownState:
    """Compute portfolio drawdown state from raw positions + a price provider.

    Pure function — does not gate. Use ``check_kill_switch`` to apply config thresholds.

    Drawdown percentages are normalized to ``budget_usd`` (the configured
    total capital), not to currently-open cost basis. This keeps the gate's
    sensitivity tied to the size of the book the operator has committed to,
    not to whatever's currently deployed.
    """
    gross_cost_basis = 0.0
    market_value = 0.0
    for pos in open_positions:
        cb = pos.entry_price * pos.size
        mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
        mv = mark * pos.size
        gross_cost_basis += cb
        if getattr(pos.side, "value", pos.side) == "short":
            # Short P&L flips sign — market value contributes inversely.
            market_value += cb + (cb - mv)
        else:
            market_value += mv

    # Netted cost basis is what we display; the budget is what we divide by.
    netted_cost_basis = _net_paired_cost_basis(open_positions)

    unrealized_pnl = market_value - gross_cost_basis
    unrealized_dd_pct = (unrealized_pnl / budget_usd * 100.0) if budget_usd > 0 else 0.0

    realized_today = 0.0
    for pos in closed_today:
        pnl = pos.pnl if pos.pnl is not None else 0.0
        realized_today += pnl
    realized_pct = (realized_today / budget_usd * 100.0) if budget_usd > 0 else 0.0

    return DrawdownState(
        open_cost_basis_usd=netted_cost_basis,
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
