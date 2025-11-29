"""Technical agent - analyzes price action and momentum."""

from typing import Optional

import yfinance as yf
import pandas as pd

from cents.agents.base import BaseAgent, AgentResult
from cents.models import Evidence, EvidenceType, Thesis


class TechnicalAgent(BaseAgent):
    """Agent that analyzes technical indicators and price action."""

    name = "technical"

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Research technical indicators for a symbol."""
        ticker = yf.Ticker(symbol)
        evidence = []
        conviction_delta = 0.0
        summaries = []

        thesis_id = thesis.id if thesis else "standalone"

        try:
            # Get historical data
            hist = ticker.history(period="6mo")
            if hist.empty:
                return AgentResult(
                    evidence=[],
                    conviction_delta=0,
                    summary=f"No historical data for {symbol}",
                )
        except Exception as e:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"Failed to fetch data for {symbol}: {e}",
            )

        close = hist["Close"]
        volume = hist["Volume"]

        # Current price and recent performance
        current_price = close.iloc[-1]
        price_1w = close.iloc[-5] if len(close) >= 5 else close.iloc[0]
        price_1m = close.iloc[-21] if len(close) >= 21 else close.iloc[0]
        price_3m = close.iloc[-63] if len(close) >= 63 else close.iloc[0]

        change_1w = (current_price - price_1w) / price_1w * 100
        change_1m = (current_price - price_1m) / price_1m * 100
        change_3m = (current_price - price_3m) / price_3m * 100

        # Momentum signal
        ev_type = EvidenceType.NEUTRAL
        if change_1m > 10:
            ev_type = EvidenceType.SUPPORTING
            conviction_delta += 3
            summaries.append(f"Strong momentum (+{change_1m:.1f}% 1M)")
        elif change_1m < -10:
            ev_type = EvidenceType.CONTRADICTING
            conviction_delta -= 3
            summaries.append(f"Weak momentum ({change_1m:.1f}% 1M)")

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Price: ${current_price:.2f} | 1W: {change_1w:+.1f}% | 1M: {change_1m:+.1f}% | 3M: {change_3m:+.1f}%",
                source="yfinance",
                evidence_type=ev_type,
                confidence=0.7,
                metadata={
                    "metric": "price_momentum",
                    "current": current_price,
                    "change_1w": change_1w,
                    "change_1m": change_1m,
                    "change_3m": change_3m,
                },
            )
        )

        # Moving averages
        ma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None

        if ma_20 and ma_50:
            ev_type = EvidenceType.NEUTRAL
            if current_price > ma_20 > ma_50:
                ev_type = EvidenceType.SUPPORTING
                conviction_delta += 2
                summaries.append("Above MAs (bullish)")
            elif current_price < ma_20 < ma_50:
                ev_type = EvidenceType.CONTRADICTING
                conviction_delta -= 2
                summaries.append("Below MAs (bearish)")

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"MA20: ${ma_20:.2f} | MA50: ${ma_50:.2f} | Price vs MA20: {((current_price/ma_20)-1)*100:+.1f}%",
                    source="yfinance",
                    evidence_type=ev_type,
                    confidence=0.65,
                    metadata={"metric": "moving_averages", "ma20": ma_20, "ma50": ma_50},
                )
            )

        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        recent_volume = volume.iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        ev_type = EvidenceType.NEUTRAL
        if volume_ratio > 1.5:
            ev_type = EvidenceType.SUPPORTING if change_1w > 0 else EvidenceType.CONTRADICTING
            conviction_delta += 2 if change_1w > 0 else -2
            summaries.append(f"High volume ({volume_ratio:.1f}x avg)")

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Volume: {recent_volume/1e6:.1f}M avg (last 5d) | {volume_ratio:.1f}x 20d average",
                source="yfinance",
                evidence_type=ev_type,
                confidence=0.6,
                metadata={"metric": "volume", "ratio": volume_ratio},
            )
        )

        # Volatility (simple ATR-like measure)
        high_low_range = (hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]
        volatility_pct = (high_low_range / current_price) * 100

        ev_type = EvidenceType.NEUTRAL
        if volatility_pct > 5:
            ev_type = EvidenceType.CONTRADICTING
            conviction_delta -= 1
            summaries.append(f"High volatility ({volatility_pct:.1f}%)")

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Avg Daily Range: {volatility_pct:.2f}% of price",
                source="yfinance",
                evidence_type=ev_type,
                confidence=0.55,
                metadata={"metric": "volatility", "value": volatility_pct},
            )
        )

        # 52-week position
        high_52w = close.max()
        low_52w = close.min()
        position_52w = (current_price - low_52w) / (high_52w - low_52w) * 100 if high_52w != low_52w else 50

        ev_type = EvidenceType.NEUTRAL
        if position_52w > 80:
            ev_type = EvidenceType.SUPPORTING
            conviction_delta += 1
            summaries.append("Near 52w high")
        elif position_52w < 20:
            ev_type = EvidenceType.CONTRADICTING
            conviction_delta -= 1
            summaries.append("Near 52w low")

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"52W Range: ${low_52w:.2f} - ${high_52w:.2f} | Position: {position_52w:.0f}%",
                source="yfinance",
                evidence_type=ev_type,
                confidence=0.5,
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
        )
