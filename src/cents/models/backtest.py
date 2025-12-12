"""Backtest domain models."""

from dataclasses import dataclass, field
from datetime import date, datetime
from uuid import uuid4


@dataclass
class Backtest:
    """A backtest run for measuring agent accuracy over a historical period."""

    symbol: str
    start_date: date
    end_date: date
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BacktestSignal:
    """A signal recorded during a backtest for a specific date and agent."""

    backtest_id: str
    date: date
    agent_name: str
    conviction_delta: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    forward_returns: dict[str, float] = field(default_factory=dict)  # {"1d": 0.02, ...}
    id: str = field(default_factory=lambda: str(uuid4())[:8])
