"""Fundamentals agent - analyzes company financials."""

from typing import Optional

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.data import FundamentalsDataProvider, FundamentalsData
from cents.models import EvidenceType, Thesis, ThesisDimension, Valuation

# Sector benchmarks for relative valuation (approximate medians as of 2024)
# Source: Various financial databases, S&P sector data
SECTOR_PE_MEDIANS: dict[str, float] = {
    "Technology": 28.0,
    "Healthcare": 22.0,
    "Financial Services": 14.0,
    "Consumer Cyclical": 18.0,
    "Consumer Defensive": 20.0,
    "Industrials": 20.0,
    "Basic Materials": 12.0,
    "Energy": 12.0,
    "Utilities": 16.0,
    "Real Estate": 35.0,
    "Communication Services": 18.0,
}

SECTOR_MARGIN_MEDIANS: dict[str, float] = {
    # Profit margin medians (as decimal)
    "Technology": 0.18,
    "Healthcare": 0.10,
    "Financial Services": 0.22,
    "Consumer Cyclical": 0.06,
    "Consumer Defensive": 0.08,
    "Industrials": 0.08,
    "Basic Materials": 0.08,
    "Energy": 0.08,
    "Utilities": 0.12,
    "Real Estate": 0.25,
    "Communication Services": 0.12,
}

SECTOR_DE_NORMS: dict[str, float] = {
    # Debt/Equity norms (as percentage) - higher is normal for capital-intensive
    "Technology": 50.0,
    "Healthcare": 60.0,
    "Financial Services": 200.0,  # Banks normally have high leverage
    "Consumer Cyclical": 100.0,
    "Consumer Defensive": 80.0,
    "Industrials": 100.0,
    "Basic Materials": 80.0,
    "Energy": 60.0,
    "Utilities": 150.0,  # Utilities are capital-intensive
    "Real Estate": 150.0,  # REITs use leverage
    "Communication Services": 100.0,
}

# Default thresholds when sector is unknown
DEFAULT_PE_LOW = 15.0
DEFAULT_PE_HIGH = 30.0
DEFAULT_MARGIN_LOW = 0.05
DEFAULT_MARGIN_HIGH = 0.20
DEFAULT_DE_LOW = 50.0
DEFAULT_DE_HIGH = 200.0

# Sector-relative threshold multipliers
# Values below median * SECTOR_LOW_MULT are considered favorable
# Values above median * SECTOR_HIGH_MULT are considered unfavorable
SECTOR_PE_LOW_MULT = 0.7   # 30% below sector median = undervalued
SECTOR_PE_HIGH_MULT = 1.3  # 30% above sector median = overvalued
SECTOR_MARGIN_LOW_MULT = 0.5   # 50% below sector median = weak
SECTOR_MARGIN_HIGH_MULT = 1.5  # 50% above sector median = strong
SECTOR_DE_LOW_MULT = 0.5   # 50% below sector norm = conservative
SECTOR_DE_HIGH_MULT = 1.5  # 50% above sector norm = leveraged

# Forward P/E comparison thresholds
FORWARD_PE_GROWTH_MULT = 0.8   # Forward < trailing * 0.8 = growth expected
FORWARD_PE_DECLINE_MULT = 1.2  # Forward > trailing * 1.2 = decline expected

# Growth rate thresholds (percentages)
HIGH_GROWTH_THRESHOLD = 0.20      # 20% revenue growth = high growth
STRONG_REVENUE_GROWTH_PCT = 20    # 20% for display comparison
STRONG_EARNINGS_GROWTH_PCT = 15   # 15% earnings growth = strong
EARNINGS_DECLINE_PCT = -10        # -10% earnings = decline


def _get_fundamentals_provider():
    """Lazy import to avoid circular dependencies."""
    from cents.data import get_fundamentals_provider
    return get_fundamentals_provider()


class FundamentalsAgent(BaseAgent):
    """Agent that analyzes fundamental company data."""

    name = "fundamentals"

    def __init__(self, fundamentals_provider: Optional[FundamentalsDataProvider] = None):
        """
        Initialize fundamentals agent.

        Args:
            fundamentals_provider: Fundamentals data provider (defaults to FMP)
        """
        super().__init__()
        self._provider = fundamentals_provider

    @property
    def provider(self) -> FundamentalsDataProvider:
        """Get fundamentals data provider, creating default if needed."""
        if self._provider is None:
            self._provider = _get_fundamentals_provider()
        return self._provider

    def _get_pe_thresholds(self, sector: Optional[str]) -> tuple[float, float]:
        """Get P/E thresholds adjusted for sector.

        Returns (low, high) where:
        - P/E < low = undervalued (bullish)
        - P/E > high = overvalued (bearish)
        """
        if sector and sector in SECTOR_PE_MEDIANS:
            median = SECTOR_PE_MEDIANS[sector]
            return (median * SECTOR_PE_LOW_MULT, median * SECTOR_PE_HIGH_MULT)
        return (DEFAULT_PE_LOW, DEFAULT_PE_HIGH)

    def _get_margin_thresholds(self, sector: Optional[str]) -> tuple[float, float]:
        """Get profit margin thresholds adjusted for sector."""
        if sector and sector in SECTOR_MARGIN_MEDIANS:
            median = SECTOR_MARGIN_MEDIANS[sector]
            return (median * SECTOR_MARGIN_LOW_MULT, median * SECTOR_MARGIN_HIGH_MULT)
        return (DEFAULT_MARGIN_LOW, DEFAULT_MARGIN_HIGH)

    def _get_de_thresholds(self, sector: Optional[str]) -> tuple[float, float]:
        """Get debt/equity thresholds adjusted for sector."""
        if sector and sector in SECTOR_DE_NORMS:
            norm = SECTOR_DE_NORMS[sector]
            return (norm * SECTOR_DE_LOW_MULT, norm * SECTOR_DE_HIGH_MULT)
        return (DEFAULT_DE_LOW, DEFAULT_DE_HIGH)

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Research fundamental data for a symbol."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        try:
            data = self._with_retries(lambda: self.provider.get_fundamentals(symbol))
        except RECOVERABLE_EXCEPTIONS as e:
            return self._error_result(symbol, e)

        # Check if we got any meaningful data
        has_data = any([
            data.pe_ratio, data.profit_margin, data.debt_to_equity,
            data.recommendation, data.revenue_growth, data.forward_pe,
            data.earnings_growth
        ])
        if not has_data:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"No data available for {symbol}",
            )

        thesis_id = thesis.id if thesis else "standalone"
        company_name = data.name or symbol

        # Valuation metrics (VALUATION dimension) - sector-relative
        if data.pe_ratio:
            ev_type = EvidenceType.NEUTRAL
            val_delta = 0.0
            pe_low, pe_high = self._get_pe_thresholds(data.sector)
            sector_note = f" [vs {data.sector} median]" if data.sector else ""

            # Handle negative P/E (unprofitable company) separately
            if data.pe_ratio < 0:
                # Check if high-growth company
                has_strong_growth = (
                    data.revenue_growth is not None and data.revenue_growth > HIGH_GROWTH_THRESHOLD
                )
                if has_strong_growth:
                    # High-growth unprofitable = neutral (acceptable for growth stocks)
                    ev_type = EvidenceType.NEUTRAL
                    val_delta = 0
                    summaries.append(f"Unprofitable but high growth{sector_note}")
                else:
                    # Unprofitable without strong growth = bearish
                    ev_type = EvidenceType.CONTRADICTING
                    val_delta = -3
                    summaries.append(f"Unprofitable (negative P/E){sector_note}")
            # If thesis has valuation expectation, evaluate against it
            elif thesis and thesis.valuation:
                if thesis.valuation == Valuation.UNDERVALUED:
                    # Expecting undervalued - low P/E supports, high contradicts
                    if data.pe_ratio < pe_low:
                        ev_type = EvidenceType.SUPPORTING
                        val_delta = 4
                        summaries.append(f"P/E {data.pe_ratio:.1f} supports undervalued thesis{sector_note}")
                    elif data.pe_ratio > pe_high:
                        ev_type = EvidenceType.CONTRADICTING
                        val_delta = -3
                        summaries.append(f"P/E {data.pe_ratio:.1f} challenges undervalued thesis{sector_note}")
                elif thesis.valuation == Valuation.OVERVALUED:
                    # Expecting overvalued - high P/E supports, low contradicts
                    if data.pe_ratio > pe_high:
                        ev_type = EvidenceType.SUPPORTING
                        val_delta = 3
                    elif data.pe_ratio < pe_low:
                        ev_type = EvidenceType.CONTRADICTING
                        val_delta = -3
            else:
                # No thesis valuation - use sector-relative scoring
                if data.pe_ratio < pe_low:
                    ev_type = EvidenceType.SUPPORTING
                    val_delta = 3
                    summaries.append(f"Low P/E ({data.pe_ratio:.1f}){sector_note}")
                elif data.pe_ratio > pe_high:
                    ev_type = EvidenceType.CONTRADICTING
                    val_delta = -2
                    summaries.append(f"High P/E ({data.pe_ratio:.1f}){sector_note}")

            conviction_delta += val_delta
            dimension_scores["valuation"] = dimension_scores.get("valuation", 0) + val_delta

            content = f"P/E Ratio: {data.pe_ratio:.2f}"
            if data.forward_pe:
                content += f" (Forward: {data.forward_pe:.2f})"

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=content,
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.8,
                    dimension=ThesisDimension.VALUATION,
                    metadata={"metric": "pe_ratio", "value": data.pe_ratio},
                )
            )

        # Forward P/E (VALUATION dimension) - indicates expected earnings growth
        if data.forward_pe is not None and data.pe_ratio is not None:
            ev_type = EvidenceType.NEUTRAL
            fwd_delta = 0.0

            # Forward P/E significantly lower than trailing = earnings growth expected
            if data.forward_pe < data.pe_ratio * FORWARD_PE_GROWTH_MULT:
                ev_type = EvidenceType.SUPPORTING
                fwd_delta = 2
                summaries.append(f"Forward P/E {data.forward_pe:.1f} < trailing (growth expected)")
            elif data.forward_pe > data.pe_ratio * FORWARD_PE_DECLINE_MULT:
                # Forward P/E higher = earnings decline expected
                ev_type = EvidenceType.CONTRADICTING
                fwd_delta = -2
                summaries.append(f"Forward P/E {data.forward_pe:.1f} > trailing (decline expected)")

            conviction_delta += fwd_delta
            dimension_scores["valuation"] = dimension_scores.get("valuation", 0) + fwd_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Forward P/E: {data.forward_pe:.2f} (Trailing: {data.pe_ratio:.2f})",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.7,  # Lower confidence for forward estimates
                    dimension=ThesisDimension.VALUATION,
                    metadata={"metric": "forward_pe", "value": data.forward_pe},
                )
            )

        # Growth metrics (QUALITY dimension)
        if data.revenue_growth is not None:
            # FMP returns as decimal, convert to percentage
            growth_pct = data.revenue_growth * 100 if abs(data.revenue_growth) < 10 else data.revenue_growth
            ev_type = EvidenceType.NEUTRAL
            quality_delta = 0.0

            if growth_pct > STRONG_REVENUE_GROWTH_PCT:
                ev_type = EvidenceType.SUPPORTING
                quality_delta = 5
                summaries.append(f"Strong revenue growth ({growth_pct:.0f}%)")
            elif growth_pct < 0:
                ev_type = EvidenceType.CONTRADICTING
                quality_delta = -5
                summaries.append(f"Negative revenue growth ({growth_pct:.0f}%)")

            conviction_delta += quality_delta
            dimension_scores["quality"] = dimension_scores.get("quality", 0) + quality_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Revenue Growth: {growth_pct:.1f}%",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.85,
                    dimension=ThesisDimension.QUALITY,
                    metadata={"metric": "revenue_growth", "value": data.revenue_growth},
                )
            )

        # Earnings growth (QUALITY dimension) - based on analyst estimates
        if data.earnings_growth is not None:
            growth_pct = data.earnings_growth * 100
            ev_type = EvidenceType.NEUTRAL
            eg_delta = 0.0

            if growth_pct > STRONG_EARNINGS_GROWTH_PCT:
                ev_type = EvidenceType.SUPPORTING
                eg_delta = 3
                summaries.append(f"Expected earnings growth ({growth_pct:.0f}%)")
            elif growth_pct < EARNINGS_DECLINE_PCT:
                ev_type = EvidenceType.CONTRADICTING
                eg_delta = -3
                summaries.append(f"Expected earnings decline ({growth_pct:.0f}%)")

            conviction_delta += eg_delta
            dimension_scores["quality"] = dimension_scores.get("quality", 0) + eg_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Expected Earnings Growth: {growth_pct:.1f}%",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.7,  # Lower confidence for estimates
                    dimension=ThesisDimension.QUALITY,
                    metadata={"metric": "earnings_growth", "value": data.earnings_growth},
                )
            )

        # Profitability (QUALITY dimension) - sector-relative
        if data.profit_margin is not None:
            # FMP returns as decimal, convert to percentage for comparison
            margin_pct = data.profit_margin * 100 if abs(data.profit_margin) < 1 else data.profit_margin
            margin_decimal = margin_pct / 100  # Convert back for threshold comparison
            margin_low, margin_high = self._get_margin_thresholds(data.sector)
            ev_type = EvidenceType.NEUTRAL
            quality_delta = 0.0
            sector_note = f" [vs {data.sector}]" if data.sector else ""

            if margin_decimal > margin_high:
                ev_type = EvidenceType.SUPPORTING
                quality_delta = 2
            elif margin_decimal < margin_low:
                ev_type = EvidenceType.CONTRADICTING
                quality_delta = -2

            conviction_delta += quality_delta
            dimension_scores["quality"] = dimension_scores.get("quality", 0) + quality_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Profit Margin: {margin_pct:.1f}%{sector_note}",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.8,
                    dimension=ThesisDimension.QUALITY,
                    metadata={"metric": "profit_margin", "value": data.profit_margin, "sector": data.sector},
                )
            )

        # Balance sheet strength (RISK dimension) - sector-relative
        if data.debt_to_equity is not None:
            # FMP may return as ratio or percentage depending on endpoint
            d_e = data.debt_to_equity * 100 if data.debt_to_equity < 10 else data.debt_to_equity
            de_low, de_high = self._get_de_thresholds(data.sector)
            ev_type = EvidenceType.NEUTRAL
            risk_delta = 0.0
            sector_note = f" [vs {data.sector}]" if data.sector else ""

            if d_e < de_low:
                ev_type = EvidenceType.SUPPORTING
                risk_delta = 2
            elif d_e > de_high:
                ev_type = EvidenceType.CONTRADICTING
                risk_delta = -3
                summaries.append(f"High debt ({d_e:.0f}% D/E){sector_note}")

            conviction_delta += risk_delta
            dimension_scores["risk"] = dimension_scores.get("risk", 0) + risk_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Debt/Equity: {d_e:.1f}%{sector_note}",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.75,
                    dimension=ThesisDimension.RISK,
                    metadata={"metric": "debt_to_equity", "value": data.debt_to_equity, "sector": data.sector},
                )
            )

        # Analyst recommendations (SENTIMENT dimension)
        if data.recommendation:
            rec_map = {
                "strong_buy": (EvidenceType.SUPPORTING, 3),
                "buy": (EvidenceType.SUPPORTING, 2),
                "hold": (EvidenceType.NEUTRAL, 0),
                "sell": (EvidenceType.CONTRADICTING, -2),
                "strong_sell": (EvidenceType.CONTRADICTING, -3),
            }
            ev_type, sent_delta = rec_map.get(data.recommendation, (EvidenceType.NEUTRAL, 0))
            conviction_delta += sent_delta
            dimension_scores["sentiment"] = dimension_scores.get("sentiment", 0) + sent_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Analyst Recommendation: {data.recommendation.replace('_', ' ').title()}",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.6,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata={"metric": "recommendation", "value": data.recommendation},
                )
            )

        # Build summary
        if summaries:
            summary = f"{company_name}: " + "; ".join(summaries)
        else:
            summary = f"{company_name}: No significant signals"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )
