"""Adaptive signal mode detection based on historical accuracy.

Determines whether momentum or contrarian signals work better for each symbol
by analyzing backtest history. The system self-improves as more data is collected.
"""

from enum import Enum
import sqlite3

from cents.db.repository import BacktestRepository
from cents.models import BacktestSignal


class SignalMode(Enum):
    """Signal generation mode."""

    MOMENTUM = "momentum"  # Follow trends (buy strength, sell weakness)
    CONTRARIAN = "contrarian"  # Fade trends (buy weakness, sell strength)
    NEUTRAL = "neutral"  # Reduce signal strength due to low confidence


# Thresholds for mode detection
MIN_SIGNALS_FOR_ADAPTIVE = 15  # Need at least this many signals to adapt
CONTRARIAN_THRESHOLD = 0.35  # Below this hit rate, flip to contrarian
MOMENTUM_THRESHOLD = 0.55  # Above this hit rate, stay momentum
# Between thresholds = neutral (not enough edge either way)


def calculate_hit_rate(
    signals: list[BacktestSignal],
    horizon: str = "20d",
) -> float | None:
    """Calculate hit rate for a list of signals.

    Hit = signal direction matched return direction.

    Args:
        signals: List of BacktestSignal with forward_returns
        horizon: Which forward return to use ("5d", "20d", "60d")

    Returns:
        Hit rate as float (0.0 to 1.0), or None if no valid signals
    """
    hits = 0
    total = 0

    for signal in signals:
        forward_return = signal.forward_returns.get(horizon)
        if forward_return is None:
            continue

        delta = signal.conviction_delta
        if delta == 0:
            continue  # Skip neutral signals

        total += 1
        # Hit if signs match (positive delta + positive return, or negative + negative)
        if (delta > 0 and forward_return > 0) or (delta < 0 and forward_return < 0):
            hits += 1

    return hits / total if total > 0 else None


def get_signal_mode(
    symbol: str,
    agent_name: str = "technical",
    conn: sqlite3.Connection | None = None,
    horizon: str = "20d",
    lookback: int = 50,
) -> tuple[SignalMode, dict]:
    """Determine optimal signal mode for a symbol based on historical accuracy.

    Args:
        symbol: Stock symbol
        agent_name: Agent to check (default: technical)
        conn: Optional DB connection for testing
        horizon: Forward return horizon to evaluate
        lookback: Number of recent signals to analyze

    Returns:
        Tuple of (SignalMode, metadata dict with hit_rate, signal_count, etc.)
    """
    # cents-38i: Disable adaptive mode during an active pre-registered experiment.
    # Adaptive mode reads from BacktestRepository which is populated by the SAME
    # live-arm outcomes the agent is currently predicting — that's a feedback
    # loop / p-hack risk that confounds the LLM-vs-random hit-rate delta. During
    # experiments, force MOMENTUM (= no transformation) so the agent's signal is
    # what it computed, not what the running cohort taught it.
    try:
        from cents.experiments import get_active_experiment
        active = get_active_experiment()
    except Exception:  # pragma: no cover — defensive
        active = None
    if active is not None:
        return SignalMode.MOMENTUM, {
            "symbol": symbol,
            "agent": agent_name,
            "horizon": horizon,
            "signal_count": 0,
            "hit_rate": None,
            "reason": (
                f"adaptive mode disabled during active experiment "
                f"{active.name!r} (cents-38i)"
            ),
        }

    repo = BacktestRepository(conn=conn)
    signals = repo.get_signal_history(
        symbol=symbol,
        agent_name=agent_name,
        limit=lookback,
        horizon=horizon,
    )

    metadata = {
        "symbol": symbol,
        "agent": agent_name,
        "horizon": horizon,
        "signal_count": len(signals),
        "hit_rate": None,
        "reason": None,
    }

    # Not enough data - default to momentum
    if len(signals) < MIN_SIGNALS_FOR_ADAPTIVE:
        metadata["reason"] = f"insufficient data ({len(signals)} < {MIN_SIGNALS_FOR_ADAPTIVE})"
        return SignalMode.MOMENTUM, metadata

    hit_rate = calculate_hit_rate(signals, horizon)

    if hit_rate is None:
        metadata["reason"] = "no valid signals with returns"
        return SignalMode.MOMENTUM, metadata

    metadata["hit_rate"] = hit_rate

    if hit_rate < CONTRARIAN_THRESHOLD:
        # Consistently wrong - flip signals
        metadata["reason"] = f"hit_rate {hit_rate:.1%} < {CONTRARIAN_THRESHOLD:.0%} threshold"
        return SignalMode.CONTRARIAN, metadata
    elif hit_rate > MOMENTUM_THRESHOLD:
        # Consistently right - keep signals as-is
        metadata["reason"] = f"hit_rate {hit_rate:.1%} > {MOMENTUM_THRESHOLD:.0%} threshold"
        return SignalMode.MOMENTUM, metadata
    else:
        # In the middle - not enough edge, reduce confidence
        metadata["reason"] = f"hit_rate {hit_rate:.1%} in neutral zone ({CONTRARIAN_THRESHOLD:.0%}-{MOMENTUM_THRESHOLD:.0%})"
        return SignalMode.NEUTRAL, metadata


def apply_signal_mode(
    conviction_delta: float,
    mode: SignalMode,
    neutral_multiplier: float = 0.3,
) -> float:
    """Apply signal mode transformation to conviction delta.

    Args:
        conviction_delta: Raw conviction delta from agent
        mode: Signal mode to apply
        neutral_multiplier: How much to reduce signal in neutral mode

    Returns:
        Transformed conviction delta
    """
    if mode == SignalMode.CONTRARIAN:
        return -conviction_delta
    elif mode == SignalMode.NEUTRAL:
        return conviction_delta * neutral_multiplier
    else:  # MOMENTUM
        return conviction_delta
