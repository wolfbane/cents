"""Momentum screener — price + volume confirmation.

Filters (using ~80 daily bars from Alpaca to give us 50d MA + 3m change):
  - latest close > 50-day simple moving average
  - 3-month price change > pct_min
  - last 5-day average volume > 1.5x trailing 50-day average

Signal strength = 3-month price change.
"""

from __future__ import annotations

from cents.screeners._base import (
    DEFAULT_LIMIT,
    _get_price_provider,
    run_per_symbol_screen,
)


class MomentumScreener:
    name = "momentum"

    MA_PERIOD = 50
    LOOKBACK_3M_BARS = 63  # ~3 months of trading days
    VOL_RECENT = 5
    VOL_AVG = 50
    VOL_RATIO_MIN = 1.5
    HISTORY_DAYS = 120

    def __init__(
        self,
        pct_min: float = 0.10,
        limit: int = DEFAULT_LIMIT,
        price_provider=None,
    ) -> None:
        self.pct_min = pct_min
        self.limit = limit
        self._provider = price_provider

    @property
    def provider(self):
        if self._provider is None:
            self._provider = _get_price_provider()
        return self._provider

    def describe(self) -> dict:
        return {
            "description": "Trend-following with volume confirmation.",
            "rules": [
                f"latest close > {self.MA_PERIOD}-day SMA",
                f"3-month price change > {self.pct_min}",
                f"5d avg volume > {self.VOL_RATIO_MIN}x trailing {self.VOL_AVG}d avg",
            ],
        }

    def screen(self, candidate_symbols: list[str] | None = None) -> list[str]:
        return run_per_symbol_screen(self._score_symbol, candidate_symbols, self.limit)

    def _score_symbol(self, symbol: str) -> float | None:
        history = self.provider.get_history(symbol, days=self.HISTORY_DAYS)
        closes = history.closes
        volumes = history.volumes
        if len(closes) < self.MA_PERIOD or len(closes) <= self.LOOKBACK_3M_BARS:
            return None
        if len(volumes) < self.VOL_AVG:
            return None

        latest_close = closes[-1]
        ma_50 = sum(closes[-self.MA_PERIOD:]) / self.MA_PERIOD
        if latest_close <= ma_50:
            return None

        ref_close = closes[-1 - self.LOOKBACK_3M_BARS]
        if ref_close <= 0:
            return None
        change_3m = (latest_close - ref_close) / ref_close
        if change_3m <= self.pct_min:
            return None

        recent_vol = sum(volumes[-self.VOL_RECENT:]) / self.VOL_RECENT
        avg_vol = sum(volumes[-self.VOL_AVG:]) / self.VOL_AVG
        if avg_vol <= 0 or recent_vol / avg_vol <= self.VOL_RATIO_MIN:
            return None
        return change_3m
