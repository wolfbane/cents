"""Finance primitives — sizing, costs, hedging, liquidity, portfolio risk.

These modules turn the factory engine from an equal-dollar signal-follower
into something that at least *names* the controls a real trading system
would need. None of this is production-grade — see /scope/ — but it makes
the cohort/backtest numbers honest enough to study.
"""

from cents.finance.costs import (
    Cost,
    apply_open_cost,
    apply_close_cost,
)
from cents.finance.hedging import (
    beta_match_ratio,
    estimate_beta,
)
from cents.finance.liquidity import (
    average_daily_volume,
    passes_borrow_gate,
    passes_liquidity_gate,
)
from cents.finance.portfolio import (
    DrawdownState,
    compute_drawdown,
    check_kill_switch,
)
from cents.finance.sizing import (
    realized_vol_pct,
    vol_scaled_shares,
)
from cents.finance.triggers import (
    stop_hit,
    target_hit,
)

__all__ = [
    "Cost",
    "apply_open_cost",
    "apply_close_cost",
    "beta_match_ratio",
    "estimate_beta",
    "average_daily_volume",
    "passes_borrow_gate",
    "passes_liquidity_gate",
    "DrawdownState",
    "compute_drawdown",
    "check_kill_switch",
    "realized_vol_pct",
    "vol_scaled_shares",
    "stop_hit",
    "target_hit",
]
