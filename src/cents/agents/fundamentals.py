"""Fundamentals agent - analyzes company financials."""

from typing import Optional

from cents.agents.base import BaseAgent, AgentResult
from cents.data import FundamentalsDataProvider, FundamentalsData
from cents.models import EvidenceType, Thesis, ThesisDimension, Valuation


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

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Research fundamental data for a symbol."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        try:
            data = self._with_retries(lambda: self.provider.get_fundamentals(symbol))
        except Exception as e:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"Failed to fetch data for {symbol} after retries: {e}",
            )

        # Check if we got any meaningful data
        has_data = any([
            data.pe_ratio, data.profit_margin, data.debt_to_equity,
            data.recommendation, data.revenue_growth
        ])
        if not has_data:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"No data available for {symbol}",
            )

        thesis_id = thesis.id if thesis else "standalone"
        company_name = data.name or symbol

        # Valuation metrics (VALUATION dimension)
        if data.pe_ratio:
            ev_type = EvidenceType.NEUTRAL
            val_delta = 0.0

            # If thesis has valuation expectation, evaluate against it
            if thesis and thesis.valuation:
                if thesis.valuation == Valuation.UNDERVALUED:
                    # Expecting undervalued - low P/E supports, high contradicts
                    if data.pe_ratio < 15:
                        ev_type = EvidenceType.SUPPORTING
                        val_delta = 4
                        summaries.append(f"P/E {data.pe_ratio:.1f} supports undervalued thesis")
                    elif data.pe_ratio > 25:
                        ev_type = EvidenceType.CONTRADICTING
                        val_delta = -3
                        summaries.append(f"P/E {data.pe_ratio:.1f} challenges undervalued thesis")
                elif thesis.valuation == Valuation.OVERVALUED:
                    # Expecting overvalued - high P/E supports, low contradicts
                    if data.pe_ratio > 30:
                        ev_type = EvidenceType.SUPPORTING
                        val_delta = 3
                    elif data.pe_ratio < 15:
                        ev_type = EvidenceType.CONTRADICTING
                        val_delta = -3
            else:
                # No thesis valuation - use default scoring
                if data.pe_ratio < 15:
                    ev_type = EvidenceType.SUPPORTING
                    val_delta = 3
                    summaries.append(f"Low P/E ({data.pe_ratio:.1f})")
                elif data.pe_ratio > 30:
                    ev_type = EvidenceType.CONTRADICTING
                    val_delta = -2
                    summaries.append(f"High P/E ({data.pe_ratio:.1f})")

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

        # Growth metrics (QUALITY dimension)
        if data.revenue_growth is not None:
            # FMP returns as decimal, convert to percentage
            growth_pct = data.revenue_growth * 100 if abs(data.revenue_growth) < 10 else data.revenue_growth
            ev_type = EvidenceType.NEUTRAL
            quality_delta = 0.0

            if growth_pct > 20:
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

        # Profitability (QUALITY dimension)
        if data.profit_margin is not None:
            # FMP returns as decimal, convert to percentage
            margin_pct = data.profit_margin * 100 if abs(data.profit_margin) < 1 else data.profit_margin
            ev_type = EvidenceType.NEUTRAL
            quality_delta = 0.0

            if margin_pct > 20:
                ev_type = EvidenceType.SUPPORTING
                quality_delta = 2
            elif margin_pct < 5:
                ev_type = EvidenceType.CONTRADICTING
                quality_delta = -2

            conviction_delta += quality_delta
            dimension_scores["quality"] = dimension_scores.get("quality", 0) + quality_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Profit Margin: {margin_pct:.1f}%",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.8,
                    dimension=ThesisDimension.QUALITY,
                    metadata={"metric": "profit_margin", "value": data.profit_margin},
                )
            )

        # Balance sheet strength (RISK dimension)
        if data.debt_to_equity is not None:
            # FMP may return as ratio or percentage depending on endpoint
            d_e = data.debt_to_equity * 100 if data.debt_to_equity < 10 else data.debt_to_equity
            ev_type = EvidenceType.NEUTRAL
            risk_delta = 0.0

            if d_e < 50:
                ev_type = EvidenceType.SUPPORTING
                risk_delta = 2
            elif d_e > 200:
                ev_type = EvidenceType.CONTRADICTING
                risk_delta = -3
                summaries.append(f"High debt ({d_e:.0f}% D/E)")

            conviction_delta += risk_delta
            dimension_scores["risk"] = dimension_scores.get("risk", 0) + risk_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Debt/Equity: {d_e:.1f}%",
                    source="fmp",
                    evidence_type=ev_type,
                    confidence=0.75,
                    dimension=ThesisDimension.RISK,
                    metadata={"metric": "debt_to_equity", "value": data.debt_to_equity},
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
