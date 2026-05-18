"""Value screener — quality-tilted classic value filter.

Filters:
  - P/E (TTM) < pe_max
  - debt/equity < de_max
  - return on equity > roe_min
  - latest quarterly revenue growth > 0 (avoid melting ice cubes)

Signal strength = inverse P/E (cheaper = higher rank), with positive revenue
growth treated as a quality gate that just has to be met.
"""

from __future__ import annotations

from cents.screeners._base import (
    DEFAULT_LIMIT,
    _get_fundamentals_provider,
    run_per_symbol_screen,
)


class ValueScreener:
    name = "value"

    def __init__(
        self,
        pe_max: float = 15.0,
        de_max: float = 0.5,
        roe_min: float = 0.10,
        limit: int = DEFAULT_LIMIT,
        fundamentals_provider=None,
    ) -> None:
        self.pe_max = pe_max
        self.de_max = de_max
        self.roe_min = roe_min
        self.limit = limit
        self._provider = fundamentals_provider

    @property
    def provider(self):
        if self._provider is None:
            self._provider = _get_fundamentals_provider()
        return self._provider

    def describe(self) -> dict:
        return {
            "description": "Classic value with quality and growth guard.",
            "rules": [
                f"P/E (TTM) < {self.pe_max}",
                f"debt/equity < {self.de_max}",
                f"return on equity > {self.roe_min}",
                "latest quarterly revenue growth > 0",
            ],
        }

    def screen(self, candidate_symbols: list[str] | None = None) -> list[str]:
        return run_per_symbol_screen(
            self._score_symbol,
            candidate_symbols if candidate_symbols is not None else self._default_candidates(),
            self.limit,
        )

    def _score_symbol(self, symbol: str) -> float | None:
        fundamentals = self.provider.get_fundamentals(symbol)
        pe = fundamentals.pe_ratio
        de = fundamentals.debt_to_equity
        roe = fundamentals.return_on_equity

        if pe is None or pe <= 0 or pe >= self.pe_max:
            return None
        if de is None or de >= self.de_max:
            return None
        if roe is None or roe <= self.roe_min:
            return None
        if not self._latest_revenue_growth_positive(symbol):
            return None
        return 1.0 / pe

    def _latest_revenue_growth_positive(self, symbol: str) -> bool:
        """Compare the two most recent quarterly revenue prints from FMP."""
        data = self.provider.get_income_statement(symbol, period="quarter", limit=2)
        if not data or len(data) < 2:
            return False
        latest = data[0].get("revenue")
        prior = data[1].get("revenue")
        if latest is None or prior is None or prior <= 0:
            return False
        return latest > prior

    def _default_candidates(self) -> list[str]:
        """No built-in universe — full-universe runs must come from the resolver."""
        return []
