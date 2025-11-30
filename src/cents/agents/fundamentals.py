"""Fundamentals agent - analyzes company financials using yfinance."""

from typing import Optional

import yfinance as yf

from cents.agents.base import BaseAgent, AgentResult
from cents.models import Evidence, EvidenceType, Thesis


class FundamentalsAgent(BaseAgent):
    """Agent that analyzes fundamental company data."""

    name = "fundamentals"

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Research fundamental data for a symbol."""
        ticker = yf.Ticker(symbol)
        evidence = []
        conviction_delta = 0.0
        summaries = []

        # Get basic info
        try:
            info = self._with_retries(lambda: ticker.info)
        except Exception as e:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"Failed to fetch data for {symbol} after retries: {e}",
            )

        if not info:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"No data available for {symbol}",
            )

        thesis_id = thesis.id if thesis else "standalone"

        # Valuation metrics
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        peg_ratio = info.get("pegRatio")

        if pe_ratio:
            ev_type = EvidenceType.NEUTRAL
            if pe_ratio < 15:
                ev_type = EvidenceType.SUPPORTING
                conviction_delta += 3
                summaries.append(f"Low P/E ({pe_ratio:.1f})")
            elif pe_ratio > 30:
                ev_type = EvidenceType.CONTRADICTING
                conviction_delta -= 2
                summaries.append(f"High P/E ({pe_ratio:.1f})")

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"P/E Ratio: {pe_ratio:.2f}" + (f" (Forward: {forward_pe:.2f})" if forward_pe else ""),
                    source="yfinance",
                    evidence_type=ev_type,
                    confidence=0.8,
                    metadata={"metric": "pe_ratio", "value": pe_ratio},
                )
            )

        # Growth metrics
        revenue_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")

        if revenue_growth is not None:
            growth_pct = revenue_growth * 100
            ev_type = EvidenceType.NEUTRAL
            if growth_pct > 20:
                ev_type = EvidenceType.SUPPORTING
                conviction_delta += 5
                summaries.append(f"Strong revenue growth ({growth_pct:.0f}%)")
            elif growth_pct < 0:
                ev_type = EvidenceType.CONTRADICTING
                conviction_delta -= 5
                summaries.append(f"Negative revenue growth ({growth_pct:.0f}%)")

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Revenue Growth: {growth_pct:.1f}%",
                    source="yfinance",
                    evidence_type=ev_type,
                    confidence=0.85,
                    metadata={"metric": "revenue_growth", "value": revenue_growth},
                )
            )

        # Profitability
        profit_margin = info.get("profitMargins")
        roe = info.get("returnOnEquity")

        if profit_margin is not None:
            margin_pct = profit_margin * 100
            ev_type = EvidenceType.NEUTRAL
            if margin_pct > 20:
                ev_type = EvidenceType.SUPPORTING
                conviction_delta += 2
            elif margin_pct < 5:
                ev_type = EvidenceType.CONTRADICTING
                conviction_delta -= 2

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Profit Margin: {margin_pct:.1f}%",
                    source="yfinance",
                    evidence_type=ev_type,
                    confidence=0.8,
                    metadata={"metric": "profit_margin", "value": profit_margin},
                )
            )

        # Balance sheet strength
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")

        if debt_to_equity is not None:
            ev_type = EvidenceType.NEUTRAL
            if debt_to_equity < 50:
                ev_type = EvidenceType.SUPPORTING
                conviction_delta += 2
            elif debt_to_equity > 200:
                ev_type = EvidenceType.CONTRADICTING
                conviction_delta -= 3
                summaries.append(f"High debt ({debt_to_equity:.0f}% D/E)")

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Debt/Equity: {debt_to_equity:.1f}%",
                    source="yfinance",
                    evidence_type=ev_type,
                    confidence=0.75,
                    metadata={"metric": "debt_to_equity", "value": debt_to_equity},
                )
            )

        # Analyst recommendations
        recommendation = info.get("recommendationKey")
        if recommendation:
            rec_map = {
                "strong_buy": (EvidenceType.SUPPORTING, 3),
                "buy": (EvidenceType.SUPPORTING, 2),
                "hold": (EvidenceType.NEUTRAL, 0),
                "sell": (EvidenceType.CONTRADICTING, -2),
                "strong_sell": (EvidenceType.CONTRADICTING, -3),
            }
            ev_type, delta = rec_map.get(recommendation, (EvidenceType.NEUTRAL, 0))
            conviction_delta += delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Analyst Recommendation: {recommendation.replace('_', ' ').title()}",
                    source="yfinance",
                    evidence_type=ev_type,
                    confidence=0.6,
                    metadata={"metric": "recommendation", "value": recommendation},
                )
            )

        # Build summary
        company_name = info.get("shortName", symbol)
        if summaries:
            summary = f"{company_name}: " + "; ".join(summaries)
        else:
            summary = f"{company_name}: No significant signals"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
        )
