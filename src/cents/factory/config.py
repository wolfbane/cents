"""Factory configuration — TOML file at ~/.cents/factory.toml or CENTS_FACTORY_CONFIG."""

from __future__ import annotations

import logging
import os
import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_TOML = """\
# cents factory configuration

universe = "default"           # name of universe to walk (or "default" to use the marked-default)
budget_usd = 100000.0          # max gross notional across open positions
target_positions = 30          # informs default per-position sizing (budget / target_positions)
entry_threshold = 7.0          # |conviction_delta| required to open a new thesis
preemption_margin = 5.0        # new thesis must beat lowest open conviction by this margin to preempt
cohort_mode = "paired"          # "paired" (long + sector ETF short twin) or "directional_only"
default_horizon_days = 30
default_stop_pct = -5.0        # close-position trigger as % off entry (negative)
default_target_pct = 10.0      # close-position trigger as % off entry (positive)
max_new_per_run = 5            # rate-limit on new theses opened per run
max_per_premise_tag = 2        # max open theses sharing any single premise tag (0 disables)

# Premise-invalidation behaviour (v0.11). When a policy event opposes an open
# thesis's premise the EventAgent records a PREMISE_INVALIDATION alert (a
# covariate). By default the thesis is NOT closed — it runs to target / stop /
# horizon so the forward-return outcome is actually observed, not censored at
# the event. Set true to also force-close on invalidation (pre-v0.11 behaviour;
# closes ~86% of theses at ~3 days on tag-overlapping events — research-invalid).
close_on_invalidation = false

# Vol-scaled sizing (v0.10). When enabled, replaces equal-dollar sizing with
# inverse-vol scaling toward a target per-position daily $-vol fraction.
sizing_mode = "equal_dollar"   # "equal_dollar" (research default) or "vol_scaled"
target_vol_pct_per_position = 0.5   # per-position annual $-vol as % of budget
max_position_pct = 5.0         # hard cap on single-position dollar weight as % of budget
vol_lookback_days = 20

# Transaction cost model (v0.10). Subtracted from P&L on close and persisted on
# the position so cohort analytics see net-of-cost results.
commission_per_share_usd = 0.0     # 0 for paper / Alpaca; non-zero for other venues
slippage_bps = 5.0                  # bps of notional per side
gap_slippage_bps = 25.0             # extra bps applied when a stop is breached / gapped through
borrow_rate_pa_pct = 3.0            # synthetic annual borrow rate applied to shorts

# Paired-hedge beta matching (v0.10). When enabled, hedge leg notional is
# scaled by a 60-day historical beta vs the hedge ETF rather than dollar-matched.
beta_match_hedge = false       # equal-dollar hedge by default (research mode)
default_beta = 1.0
beta_lookback_days = 60
beta_min = 0.10
beta_max = 5.0
beta_min_r_squared = 0.5   # reject beta estimate when fit R² falls below this

# Liquidity + borrow gates (v0.10).
min_adv_multiple = 50.0        # require median daily $-volume >= this x position size
liquidity_lookback_days = 20

# Portfolio kill switch (v0.10). Halts the open phase when breached.
max_portfolio_drawdown_pct = 10.0   # halt opens at <= -10% unrealized DD
max_daily_loss_pct = 3.0            # halt opens at <= -3% realized loss today

# Lookahead defence (Batch J). When set, NewsAPI sentiment fetches drop
# articles whose publishedAt >= today's HH:MM (US/Eastern). Set to your
# market-open time ("09:30") to keep same-day intraday news out of the
# signal. Empty string = no filter (research mode is leaky by default).
news_cutoff_time = ""
"""


@dataclass
class FactoryConfig:
    """Resolved factory configuration."""

    universe: str = "default"
    budget_usd: float = 100000.0
    target_positions: int = 30
    entry_threshold: float = 7.0
    preemption_margin: float = 5.0
    cohort_mode: str = "paired"
    default_horizon_days: int = 30
    default_stop_pct: float = -5.0
    default_target_pct: float = 10.0
    max_new_per_run: int = 5
    max_per_premise_tag: int = 2
    # Premise-invalidation (v0.11). False = record the alert but let the thesis
    # run to target/stop/horizon (research default); True = force-close on
    # invalidation (pre-v0.11 trading-style behaviour).
    close_on_invalidation: bool = False
    # Sizing (v0.10).
    sizing_mode: str = "equal_dollar"
    target_vol_pct_per_position: float = 0.5
    max_position_pct: float = 5.0
    vol_lookback_days: int = 20
    # Costs (v0.10).
    commission_per_share_usd: float = 0.0
    slippage_bps: float = 5.0
    gap_slippage_bps: float = 25.0
    borrow_rate_pa_pct: float = 3.0
    # Hedging (v0.10).
    beta_match_hedge: bool = False
    default_beta: float = 1.0
    beta_lookback_days: int = 60
    beta_min: float = 0.10
    beta_max: float = 5.0
    beta_min_r_squared: float = 0.5
    # Liquidity (v0.10).
    min_adv_multiple: float = 50.0
    liquidity_lookback_days: int = 20
    # Kill switch (v0.10).
    max_portfolio_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 3.0
    # Lookahead defence (Batch J). When set to "HH:MM" (e.g. "09:30"),
    # NewsAPI sentiment fetches drop articles whose publishedAt is on/after
    # this wall-clock time today (US/Eastern). Default "" means no filter
    # (back-compat) — operators running a research-grade study should set
    # this to their market-open time to keep same-day intraday news out of
    # the signal. The lookahead-audit error message points at this field.
    news_cutoff_time: str = ""

    def __post_init__(self) -> None:
        if self.cohort_mode not in {"paired", "directional_only"}:
            raise ValueError(
                f"cohort_mode must be 'paired' or 'directional_only', got {self.cohort_mode!r}"
            )
        if self.sizing_mode not in {"vol_scaled", "equal_dollar"}:
            raise ValueError(
                f"sizing_mode must be 'vol_scaled' or 'equal_dollar', got {self.sizing_mode!r}"
            )
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.target_positions <= 0:
            raise ValueError("target_positions must be positive")
        if self.entry_threshold < 0:
            raise ValueError("entry_threshold must be non-negative")
        if self.max_new_per_run < 0:
            raise ValueError("max_new_per_run must be non-negative")
        if self.max_per_premise_tag < 0:
            raise ValueError("max_per_premise_tag must be non-negative")
        if self.target_vol_pct_per_position <= 0:
            raise ValueError("target_vol_pct_per_position must be positive")
        if self.max_position_pct <= 0:
            raise ValueError("max_position_pct must be positive")
        if self.slippage_bps < 0 or self.gap_slippage_bps < 0:
            raise ValueError("slippage_bps fields must be non-negative")
        if self.borrow_rate_pa_pct < 0:
            raise ValueError("borrow_rate_pa_pct must be non-negative")
        if self.beta_min < 0 or self.beta_max <= self.beta_min:
            raise ValueError("beta_min must be >=0 and beta_max must exceed beta_min")
        if not 0.0 <= self.beta_min_r_squared <= 1.0:
            raise ValueError("beta_min_r_squared must be within [0, 1]")
        if self.min_adv_multiple < 0:
            raise ValueError("min_adv_multiple must be non-negative")
        if self.max_portfolio_drawdown_pct < 0 or self.max_daily_loss_pct < 0:
            raise ValueError("kill-switch thresholds must be non-negative")

    @property
    def position_size_usd(self) -> float:
        """Equal-dollar fallback sizing — used when vol unavailable or sizing_mode='equal_dollar'."""
        return self.budget_usd / self.target_positions


_CASTERS: dict[type, type] = {str: str, int: int, float: float, bool: bool}


def get_factory_config_path() -> Path:
    """Resolve the factory config path.

    Priority:
    1. CENTS_FACTORY_CONFIG env var
    2. ~/.cents/factory.toml
    """
    env_path = os.environ.get("CENTS_FACTORY_CONFIG")
    if env_path:
        return Path(env_path)
    return Path.home() / ".cents" / "factory.toml"


def load_factory_config(path: Path | None = None) -> FactoryConfig:
    """Load factory config from TOML, falling back to defaults if missing."""
    target = path or get_factory_config_path()
    if not target.exists():
        return FactoryConfig()

    try:
        data = tomllib.loads(target.read_text())
    except tomllib.TOMLDecodeError as exc:
        logger.warning("Failed to parse factory config %s: %s", target, exc)
        return FactoryConfig()
    except OSError as exc:
        logger.warning("Failed to read factory config %s: %s", target, exc)
        return FactoryConfig()

    # Drive fallbacks off the dataclass so DEFAULT_TOML, FactoryConfig defaults,
    # and this loader can't diverge.
    fields: dict[str, object] = {}
    for name, anno in typing.get_type_hints(FactoryConfig).items():
        if name not in data:
            continue
        cast = _CASTERS.get(anno)
        if cast is None:
            raise TypeError(
                f"FactoryConfig field {name!r} has annotation {anno!r} with no "
                "caster; extend _CASTERS or add an explicit coercion."
            )
        fields[name] = cast(data[name])
    return FactoryConfig(**fields)


def scaffold_factory_config(path: Path | None = None, force: bool = False) -> Path:
    """Write the default TOML at the resolved path. Returns the path written."""
    target = path or get_factory_config_path()
    if target.exists() and not force:
        raise FileExistsError(f"Factory config already exists at {target}. Use --force to overwrite.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_TOML)
    return target
