"""Technical agent - analyzes price action and momentum."""

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.data import PriceDataProvider, get_price_provider
from cents.models import EvidenceType, Thesis, ThesisDimension

# Moving average periods
MA_SHORT_PERIOD = 20   # Short-term moving average (20 days)
MA_LONG_PERIOD = 50    # Long-term moving average (50 days)

# Momentum thresholds (percentage change)
MOMENTUM_STRONG_PCT = 10    # ±10% monthly change = strong momentum

# Volume analysis
VOLUME_AVG_PERIOD = 20      # Days for average volume calculation
VOLUME_RECENT_PERIOD = 5    # Days for recent volume comparison
VOLUME_HIGH_RATIO = 1.5     # 1.5x average = high volume

# Volatility analysis
VOLATILITY_PERIOD = 14      # Days for ATR-style volatility
VOLATILITY_HIGH_PCT = 5     # 5% daily range = high volatility

# 52-week position thresholds (percentage of range)
RANGE_52W_HIGH_PCT = 80     # Above 80% = near highs
RANGE_52W_LOW_PCT = 20      # Below 20% = near lows


def _rolling_mean(values: list[float], window: int) -> float | None:
    """Calculate rolling mean of last N values."""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _safe_get(values: list, idx: int, default=None):
    """Safely get value at index from end (0 = last, 5 = 5 from end)."""
    actual_idx = len(values) - 1 - idx
    return values[actual_idx] if 0 <= actual_idx < len(values) else default


class TechnicalAgent(BaseAgent):
    """Agent that analyzes technical indicators and price action."""

    name = "technical"

    def __init__(self, price_provider: PriceDataProvider | None = None):
        """
        Initialize technical agent.

        Args:
            price_provider: Price data provider (defaults to Alpaca)
        """
        super().__init__()
        self._provider = price_provider

    @property
    def provider(self) -> PriceDataProvider:
        """Get price data provider, creating default if needed."""
        if self._provider is None:
            self._provider = get_price_provider()
        return self._provider

    def research(self, symbol: str, thesis: Thesis | None = None) -> AgentResult:
        """Research technical indicators for a symbol."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []
        thesis_id = thesis.id if thesis else None

        try:
            history = self._with_retries(lambda: self.provider.get_history(symbol, days=365))
            if not history.bars:
                return AgentResult(
                    evidence=[],
                    conviction_delta=0,
                    summary=f"No historical data for {symbol}",
                )
        except RECOVERABLE_EXCEPTIONS as e:
            return self._error_result(symbol, e)

        closes = history.closes
        volumes = history.volumes
        highs = history.highs
        lows = history.lows

        # Current price and recent performance
        current_price = closes[-1]
        price_1w = _safe_get(closes, 5, closes[0])
        price_1m = _safe_get(closes, 21, closes[0])
        price_3m = _safe_get(closes, 63, closes[0])

        # Use explicit > 0 check to avoid division by zero (0.0 is falsy but explicit is clearer)
        change_1w = (current_price - price_1w) / price_1w * 100 if price_1w and price_1w > 0 else 0
        change_1m = (current_price - price_1m) / price_1m * 100 if price_1m and price_1m > 0 else 0
        change_3m = (current_price - price_3m) / price_3m * 100 if price_3m and price_3m > 0 else 0

        # Momentum signal (TECHNICAL dimension)
        ev_type = EvidenceType.NEUTRAL
        tech_delta = 0.0
        if change_1m > MOMENTUM_STRONG_PCT:
            ev_type = EvidenceType.SUPPORTING
            tech_delta = 3
            summaries.append(f"Strong momentum (+{change_1m:.1f}% 1M)")
        elif change_1m < -MOMENTUM_STRONG_PCT:
            ev_type = EvidenceType.CONTRADICTING
            tech_delta = -3
            summaries.append(f"Weak momentum ({change_1m:.1f}% 1M)")

        conviction_delta += tech_delta
        dimension_scores["technical"] = dimension_scores.get("technical", 0) + tech_delta

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Price: ${current_price:.2f} | 1W: {change_1w:+.1f}% | 1M: {change_1m:+.1f}% | 3M: {change_3m:+.1f}%",
                source="alpaca",
                evidence_type=ev_type,
                confidence=0.7,
                dimension=ThesisDimension.TECHNICAL,
                metadata={
                    "metric": "price_momentum",
                    "current": current_price,
                    "change_1w": change_1w,
                    "change_1m": change_1m,
                    "change_3m": change_3m,
                },
            )
        )

        # Moving averages (TECHNICAL dimension)
        ma_20 = _rolling_mean(closes, MA_SHORT_PERIOD)
        ma_50 = _rolling_mean(closes, MA_LONG_PERIOD)

        if ma_20 and ma_50:
            ev_type = EvidenceType.NEUTRAL
            ma_delta = 0.0
            if current_price > ma_20 > ma_50:
                ev_type = EvidenceType.SUPPORTING
                ma_delta = 2
                summaries.append("Above MAs (bullish)")
            elif current_price < ma_20 < ma_50:
                ev_type = EvidenceType.CONTRADICTING
                ma_delta = -2
                summaries.append("Below MAs (bearish)")

            conviction_delta += ma_delta
            dimension_scores["technical"] = dimension_scores.get("technical", 0) + ma_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"MA20: ${ma_20:.2f} | MA50: ${ma_50:.2f} | Price vs MA20: {((current_price/ma_20)-1)*100:+.1f}%",
                    source="alpaca",
                    evidence_type=ev_type,
                    confidence=0.65,
                    dimension=ThesisDimension.TECHNICAL,
                    metadata={"metric": "moving_averages", "ma20": ma_20, "ma50": ma_50},
                )
            )

        # Volume analysis (TECHNICAL dimension)
        avg_volume = _rolling_mean([float(v) for v in volumes], VOLUME_AVG_PERIOD) or sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-VOLUME_RECENT_PERIOD:]) / min(VOLUME_RECENT_PERIOD, len(volumes))
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        ev_type = EvidenceType.NEUTRAL
        vol_delta = 0.0
        if volume_ratio > VOLUME_HIGH_RATIO:
            ev_type = EvidenceType.SUPPORTING if change_1w > 0 else EvidenceType.CONTRADICTING
            vol_delta = 2 if change_1w > 0 else -2
            summaries.append(f"High volume ({volume_ratio:.1f}x avg)")

        conviction_delta += vol_delta
        dimension_scores["technical"] = dimension_scores.get("technical", 0) + vol_delta

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Volume: {recent_volume/1e6:.1f}M avg (last 5d) | {volume_ratio:.1f}x 20d average",
                source="alpaca",
                evidence_type=ev_type,
                confidence=0.6,
                dimension=ThesisDimension.TECHNICAL,
                metadata={"metric": "volume", "ratio": volume_ratio},
            )
        )

        # Volatility - RISK dimension (affects risk assessment)
        high_low_ranges = [h - l for h, l in zip(highs, lows)]
        avg_range = _rolling_mean(high_low_ranges, VOLATILITY_PERIOD) or sum(high_low_ranges) / len(high_low_ranges)
        volatility_pct = (avg_range / current_price) * 100 if current_price else 0

        ev_type = EvidenceType.NEUTRAL
        risk_delta = 0.0
        if volatility_pct > VOLATILITY_HIGH_PCT:
            ev_type = EvidenceType.CONTRADICTING
            risk_delta = -1
            summaries.append(f"High volatility ({volatility_pct:.1f}%)")

        conviction_delta += risk_delta
        dimension_scores["risk"] = dimension_scores.get("risk", 0) + risk_delta

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Avg Daily Range: {volatility_pct:.2f}% of price",
                source="alpaca",
                evidence_type=ev_type,
                confidence=0.55,
                dimension=ThesisDimension.RISK,
                metadata={"metric": "volatility", "value": volatility_pct},
            )
        )

        # 52-week position (TECHNICAL dimension)
        high_52w = max(closes)
        low_52w = min(closes)
        position_52w = (current_price - low_52w) / (high_52w - low_52w) * 100 if high_52w != low_52w else 50

        ev_type = EvidenceType.NEUTRAL
        range_delta = 0.0
        if position_52w > RANGE_52W_HIGH_PCT:
            ev_type = EvidenceType.SUPPORTING
            range_delta = 1
            summaries.append("Near 52w high")
        elif position_52w < RANGE_52W_LOW_PCT:
            ev_type = EvidenceType.CONTRADICTING
            range_delta = -1
            summaries.append("Near 52w low")

        conviction_delta += range_delta
        dimension_scores["technical"] = dimension_scores.get("technical", 0) + range_delta

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"52W Range: ${low_52w:.2f} - ${high_52w:.2f} | Position: {position_52w:.0f}%",
                source="alpaca",
                evidence_type=ev_type,
                confidence=0.5,
                dimension=ThesisDimension.TECHNICAL,
                metadata={
                    "metric": "52w_range",
                    "high": high_52w,
                    "low": low_52w,
                    "position": position_52w,
                },
            )
        )

        # Build summary
        if summaries:
            summary = f"{symbol}: " + "; ".join(summaries)
        else:
            summary = f"{symbol}: No significant technical signals"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )
