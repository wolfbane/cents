"""Mean-reversion screener — oversold *quality*, not just oversold.

Filters:
  - RSI(14) < 30 (oversold)
  - return on equity (TTM) > 0 (quality gate — avoid catching falling knives)

Signal strength = -RSI (lower RSI = more oversold = higher rank).
"""

from __future__ import annotations

from cents.screeners._base import (
    DEFAULT_LIMIT,
    _get_fundamentals_provider,
    _get_price_provider,
    rank_and_limit,
    safe_per_symbol,
)

RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
HISTORY_DAYS = 60  # ~30 trading days; need at least RSI_PERIOD + 1


def _compute_rsi(closes: list[float], period: int = RSI_PERIOD) -> float | None:
    """Simple Wilder-style RSI on the most-recent ``period`` closes."""
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for prev, curr in zip(closes[-(period + 1):-1], closes[-period:]):
        change = curr - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionScreener:
    name = "mean_reversion"

    def __init__(
        self,
        rsi_oversold: float = RSI_OVERSOLD,
        limit: int = DEFAULT_LIMIT,
        price_provider=None,
        fundamentals_provider=None,
    ) -> None:
        self.rsi_oversold = rsi_oversold
        self.limit = limit
        self._price = price_provider
        self._fundamentals = fundamentals_provider

    @property
    def price_provider(self):
        if self._price is None:
            self._price = _get_price_provider()
        return self._price

    @property
    def fundamentals_provider(self):
        if self._fundamentals is None:
            self._fundamentals = _get_fundamentals_provider()
        return self._fundamentals

    def describe(self) -> dict:
        return {
            "description": "Oversold quality — low RSI with positive trailing ROE.",
            "rules": [
                f"RSI(14) < {self.rsi_oversold}",
                "trailing-twelve-months return on equity > 0",
            ],
        }

    def screen(self, candidate_symbols: list[str] | None = None) -> list[str]:
        if candidate_symbols is not None and not candidate_symbols:
            return []
        candidates = candidate_symbols or []

        scored: list[tuple[str, float]] = []
        for symbol in candidates:
            score = safe_per_symbol(self._score_symbol, symbol)
            if score is not None:
                scored.append((symbol, score))
        return rank_and_limit(scored, self.limit)

    def _score_symbol(self, symbol: str) -> float | None:
        history = self.price_provider.get_history(symbol, days=HISTORY_DAYS)
        rsi = _compute_rsi(history.closes)
        if rsi is None or rsi >= self.rsi_oversold:
            return None

        fundamentals = self.fundamentals_provider.get_fundamentals(symbol)
        roe = fundamentals.return_on_equity
        if roe is None or roe <= 0:
            return None
        return -rsi
