"""Growth screener — revenue CAGR with gross-margin durability.

Filters (using FMP annual income-statement, 4 most-recent periods):
  - 3-year revenue CAGR > rev_growth_min
  - latest gross margin > gm_min
  - gross margin nondecreasing over the 3-year window

Signal strength = revenue CAGR (higher = better rank).
"""

from __future__ import annotations

from cents.screeners._base import (
    DEFAULT_LIMIT,
    _get_fundamentals_provider,
    rank_and_limit,
    safe_per_symbol,
)


class GrowthScreener:
    name = "growth"

    def __init__(
        self,
        rev_growth_min: float = 0.15,
        gm_min: float = 0.40,
        limit: int = DEFAULT_LIMIT,
        fundamentals_provider=None,
    ) -> None:
        self.rev_growth_min = rev_growth_min
        self.gm_min = gm_min
        self.limit = limit
        self._provider = fundamentals_provider

    @property
    def provider(self):
        if self._provider is None:
            self._provider = _get_fundamentals_provider()
        return self._provider

    def describe(self) -> dict:
        return {
            "description": "Revenue compounders whose gross margin holds up.",
            "rules": [
                f"3y revenue CAGR > {self.rev_growth_min}",
                f"latest gross margin > {self.gm_min}",
                "gross margin nondecreasing over 3y",
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
        data = self.provider._fetch_json(
            "income-statement",
            symbol=symbol,
            period="annual",
            limit=4,
            use_cache=True,
        )
        if not data or len(data) < 4:
            return None

        # FMP returns newest-first. Indices: 0=latest, 3=base 3y ago.
        revenues = [row.get("revenue") for row in data]
        if any(r is None or r <= 0 for r in revenues):
            return None
        cagr = (revenues[0] / revenues[3]) ** (1 / 3) - 1
        if cagr <= self.rev_growth_min:
            return None

        margins = []
        for row in data:
            rev = row.get("revenue")
            gross = row.get("grossProfit")
            if rev is None or gross is None or rev <= 0:
                return None
            margins.append(gross / rev)
        if margins[0] <= self.gm_min:
            return None
        # Nondecreasing in chronological order (oldest → newest).
        chronological = list(reversed(margins))
        for a, b in zip(chronological, chronological[1:]):
            if b < a:
                return None
        return cagr
