"""Insider cluster screener — multiple distinct insiders net-buying.

Filters (FMP insider-trading search, 30d window):
  - >= 3 distinct insiders with P-Purchase transactions
  - net insider dollar flow > 0 (sum of buy notional > sum of sell notional)

We allow some offsetting sells because at any large company there's almost
always someone with a scheduled 10b5-1 sale plan running; requiring zero
sells would filter out essentially every large-cap. What we care about is
whether the cluster's collective conviction is *net positive* in dollars.

Signal strength = number of distinct buyers (cluster breadth).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from cents.screeners._base import (
    DEFAULT_LIMIT,
    _get_fundamentals_provider,
    run_per_symbol_screen,
)

WINDOW_DAYS = 30
MIN_BUYERS = 3


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _notional(trade: dict) -> float:
    """Best-effort dollar value of a single insider transaction."""
    try:
        shares = float(trade.get("securitiesTransacted") or 0)
        price = float(trade.get("price") or 0)
    except (TypeError, ValueError):
        return 0.0
    return shares * price


class InsiderClusterScreener:
    name = "insider_cluster"

    def __init__(
        self,
        window_days: int = WINDOW_DAYS,
        min_buyers: int = MIN_BUYERS,
        limit: int = DEFAULT_LIMIT,
        fundamentals_provider=None,
        now: datetime | None = None,
    ) -> None:
        self.window_days = window_days
        self.min_buyers = min_buyers
        self.limit = limit
        self._provider = fundamentals_provider
        self._now = now

    @property
    def provider(self):
        if self._provider is None:
            self._provider = _get_fundamentals_provider()
        return self._provider

    def describe(self) -> dict:
        return {
            "description": "Cluster buying with net positive insider dollar flow.",
            "rules": [
                f">= {self.min_buyers} distinct insiders with buy transactions in last {self.window_days} days",
                f"net insider dollar flow > 0 across the same {self.window_days}-day window",
            ],
        }

    def screen(self, candidate_symbols: list[str] | None = None) -> list[str]:
        return run_per_symbol_screen(self._score_symbol, candidate_symbols, self.limit)

    def _score_symbol(self, symbol: str) -> float | None:
        trades = self.provider.get_insider_trades(symbol, limit=100)
        if not trades:
            return None
        cutoff = (self._now or datetime.now()) - timedelta(days=self.window_days)
        buyers: set[str] = set()
        net_dollars = 0.0
        for trade in trades:
            when = _parse_date(trade.get("transactionDate"))
            if when is None or when < cutoff:
                continue
            tx_type = trade.get("transactionType", "")
            amount = _notional(trade)
            if tx_type == "P-Purchase":
                name = trade.get("reportingName")
                if name:
                    buyers.add(name)
                net_dollars += amount
            elif tx_type == "S-Sale":
                net_dollars -= amount
        if net_dollars <= 0:
            return None
        if len(buyers) < self.min_buyers:
            return None
        return float(len(buyers))
