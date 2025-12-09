"""Moat agent - analyzes competitive advantage durability."""

import statistics

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.models import EvidenceType, Thesis, ThesisDimension

# Sector benchmarks for gross margin (approximate medians)
# Used for pricing power analysis
SECTOR_GROSS_MARGIN_MEDIANS: dict[str, float] = {
    "Technology": 0.55,
    "Healthcare": 0.55,
    "Financial Services": 0.60,
    "Consumer Cyclical": 0.35,
    "Consumer Defensive": 0.35,
    "Industrials": 0.30,
    "Basic Materials": 0.25,
    "Energy": 0.35,
    "Utilities": 0.40,
    "Real Estate": 0.60,
    "Communication Services": 0.55,
}

DEFAULT_GROSS_MARGIN_MEDIAN = 0.40


def _get_fundamentals_provider():
    """Lazy import to avoid circular dependencies."""
    from cents.data.fmp import get_fundamentals_provider
    return get_fundamentals_provider()


class MoatAgent(BaseAgent):
    """Agent that analyzes competitive advantage durability (moat)."""

    name = "moat"

    def __init__(self, fundamentals_provider=None):
        """
        Initialize moat agent.

        Args:
            fundamentals_provider: FMP provider instance (defaults to singleton)
        """
        super().__init__()
        self._provider = fundamentals_provider

    @property
    def provider(self):
        """Get fundamentals data provider, creating default if needed."""
        if self._provider is None:
            self._provider = _get_fundamentals_provider()
        return self._provider

    def research(self, symbol: str, thesis: Thesis | None = None) -> AgentResult:
        """Research moat characteristics for a symbol."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        try:
            # Fetch 5 years of historical ratios
            ratios = self._with_retries(
                lambda: self.provider.get_historical_ratios(symbol, years=5)
            )
            # Also get current fundamentals for sector info
            fundamentals = self._with_retries(
                lambda: self.provider.get_fundamentals(symbol)
            )
        except RECOVERABLE_EXCEPTIONS as e:
            return self._error_result(symbol, e)

        if not ratios:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"{symbol}: No historical data for moat analysis",
            )

        thesis_id = thesis.id if thesis else None
        sector = fundamentals.sector if fundamentals else None
        company_name = fundamentals.name if fundamentals else symbol

        # Analyze return on capital
        roic_evidence, roic_delta, roic_dims = self._analyze_return_on_capital(
            ratios, thesis_id
        )
        evidence.extend(roic_evidence)
        conviction_delta += roic_delta
        for dim, score in roic_dims.items():
            dimension_scores[dim] = dimension_scores.get(dim, 0) + score
        if roic_delta > 2:
            summaries.append("Strong returns on capital")
        elif roic_delta < -2:
            summaries.append("Weak returns on capital")

        # Analyze margin stability
        margin_evidence, margin_delta, margin_dims = self._analyze_margin_stability(
            ratios, thesis_id
        )
        evidence.extend(margin_evidence)
        conviction_delta += margin_delta
        for dim, score in margin_dims.items():
            dimension_scores[dim] = dimension_scores.get(dim, 0) + score
        if margin_delta > 2:
            summaries.append("Stable margins")
        elif margin_delta < -2:
            summaries.append("Volatile margins")

        # Analyze pricing power
        pp_evidence, pp_delta, pp_dims = self._analyze_pricing_power(
            ratios, sector, thesis_id
        )
        evidence.extend(pp_evidence)
        conviction_delta += pp_delta
        for dim, score in pp_dims.items():
            dimension_scores[dim] = dimension_scores.get(dim, 0) + score
        if pp_delta > 1.5:
            summaries.append("Pricing power")
        elif pp_delta < -1.5:
            summaries.append("Weak pricing power")

        # Build summary
        if summaries:
            summary = f"{company_name}: " + "; ".join(summaries)
        else:
            summary = f"{company_name}: Average moat characteristics"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )

    def _analyze_return_on_capital(
        self, ratios: list[dict], thesis_id: str
    ) -> tuple[list, float, dict]:
        """Analyze ROIC/ROE consistency and level.

        Returns:
            Tuple of (evidence list, conviction delta, dimension scores)
        """
        evidence = []
        delta = 0.0
        dims: dict[str, float] = {}

        # Extract ROIC values (filter None)
        roic_values = [r["roic"] for r in ratios if r.get("roic") is not None]
        roe_values = [r["returnOnEquity"] for r in ratios if r.get("returnOnEquity") is not None]

        # Use ROIC if available, otherwise fall back to ROE
        capital_returns = roic_values if roic_values else roe_values
        metric_name = "ROIC" if roic_values else "ROE"

        if len(capital_returns) < 2:
            return evidence, 0, dims

        avg_return = statistics.mean(capital_returns)
        std_return = statistics.stdev(capital_returns) if len(capital_returns) > 1 else 0

        # Determine evidence type based on average return level
        if avg_return > 0.15:
            ev_type = EvidenceType.SUPPORTING
            delta = 4.0
        elif avg_return > 0.10:
            ev_type = EvidenceType.SUPPORTING
            delta = 2.0
        elif avg_return < 0.10:
            ev_type = EvidenceType.CONTRADICTING
            delta = -2.0
        else:
            ev_type = EvidenceType.NEUTRAL

        dims["moat"] = delta
        # High ROIC also indicates quality
        dims["quality"] = delta * 0.5

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"{len(capital_returns)}-year avg {metric_name}: {avg_return:.1%}",
                source="fmp",
                evidence_type=ev_type,
                confidence=0.85,
                dimension=ThesisDimension.MOAT,
                metadata={
                    "metric": metric_name.lower(),
                    "avg": avg_return,
                    "years": len(capital_returns),
                },
            )
        )

        # Penalize high variance (inconsistent returns = weaker moat)
        if std_return > 0.06:
            variance_delta = -1.5
            delta += variance_delta
            dims["moat"] = dims.get("moat", 0) + variance_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"High {metric_name} variance: {std_return:.1%} std dev",
                    source="fmp",
                    evidence_type=EvidenceType.CONTRADICTING,
                    confidence=0.70,
                    dimension=ThesisDimension.MOAT,
                    metadata={"metric": f"{metric_name.lower()}_variance", "std": std_return},
                )
            )
        elif std_return < 0.03:
            # Low variance = consistent returns = stronger moat signal
            consistency_delta = 1.0
            delta += consistency_delta
            dims["moat"] = dims.get("moat", 0) + consistency_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Consistent {metric_name}: {std_return:.1%} std dev",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.75,
                    dimension=ThesisDimension.MOAT,
                    metadata={"metric": f"{metric_name.lower()}_consistency", "std": std_return},
                )
            )

        return evidence, delta, dims

    def _analyze_margin_stability(
        self, ratios: list[dict], thesis_id: str
    ) -> tuple[list, float, dict]:
        """Analyze gross/operating margin variance over time.

        Returns:
            Tuple of (evidence list, conviction delta, dimension scores)
        """
        evidence = []
        delta = 0.0
        dims: dict[str, float] = {}

        # Extract gross margin values
        gross_margins = [
            r["grossProfitMargin"]
            for r in ratios
            if r.get("grossProfitMargin") is not None
        ]

        if len(gross_margins) < 2:
            return evidence, 0, dims

        avg_margin = statistics.mean(gross_margins)
        std_margin = statistics.stdev(gross_margins)

        # Check for margin trend (expanding or contracting)
        # ratios are most recent first, so reverse for chronological order
        chronological = list(reversed(gross_margins))
        if len(chronological) >= 3:
            recent_avg = statistics.mean(chronological[-2:])  # Last 2 years
            early_avg = statistics.mean(chronological[:2])    # First 2 years
            margin_trend = recent_avg - early_avg
        else:
            margin_trend = 0

        # Low variance = stable business = strong moat
        if std_margin < 0.02:
            ev_type = EvidenceType.SUPPORTING
            stability_delta = 3.0
        elif std_margin < 0.05:
            ev_type = EvidenceType.NEUTRAL
            stability_delta = 0.0
        else:
            ev_type = EvidenceType.CONTRADICTING
            stability_delta = -2.0

        delta += stability_delta
        dims["moat"] = stability_delta

        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Gross margin stability: {std_margin:.1%} variance over {len(gross_margins)} years",
                source="fmp",
                evidence_type=ev_type,
                confidence=0.80,
                dimension=ThesisDimension.MOAT,
                metadata={
                    "metric": "gross_margin_variance",
                    "avg": avg_margin,
                    "std": std_margin,
                    "years": len(gross_margins),
                },
            )
        )

        # Check margin trend
        if margin_trend > 0.03:
            trend_delta = 2.0
            delta += trend_delta
            dims["moat"] = dims.get("moat", 0) + trend_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Expanding margins: +{margin_trend:.1%} over period",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.75,
                    dimension=ThesisDimension.MOAT,
                    metadata={"metric": "margin_trend", "value": margin_trend},
                )
            )
        elif margin_trend < -0.03:
            trend_delta = -3.0
            delta += trend_delta
            dims["moat"] = dims.get("moat", 0) + trend_delta

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Contracting margins: {margin_trend:.1%} over period",
                    source="fmp",
                    evidence_type=EvidenceType.CONTRADICTING,
                    confidence=0.75,
                    dimension=ThesisDimension.MOAT,
                    metadata={"metric": "margin_trend", "value": margin_trend},
                )
            )

        return evidence, delta, dims

    def _analyze_pricing_power(
        self, ratios: list[dict], sector: str | None, thesis_id: str
    ) -> tuple[list, float, dict]:
        """Analyze pricing power signals.

        Returns:
            Tuple of (evidence list, conviction delta, dimension scores)
        """
        evidence = []
        delta = 0.0
        dims: dict[str, float] = {}

        # Get most recent gross margin
        recent_margins = [
            r["grossProfitMargin"]
            for r in ratios[:2]  # Most recent 2 years
            if r.get("grossProfitMargin") is not None
        ]

        if not recent_margins:
            return evidence, 0, dims

        current_margin = statistics.mean(recent_margins)

        # Compare to sector median
        sector_median = SECTOR_GROSS_MARGIN_MEDIANS.get(sector, DEFAULT_GROSS_MARGIN_MEDIAN)
        margin_vs_sector = current_margin - sector_median

        if margin_vs_sector > 0.10:
            # Premium margins = pricing power
            ev_type = EvidenceType.SUPPORTING
            pp_delta = 2.0
        elif margin_vs_sector > 0.05:
            ev_type = EvidenceType.SUPPORTING
            pp_delta = 1.0
        elif margin_vs_sector < -0.10:
            # Below sector = weak pricing
            ev_type = EvidenceType.CONTRADICTING
            pp_delta = -2.0
        else:
            ev_type = EvidenceType.NEUTRAL
            pp_delta = 0.0

        delta += pp_delta
        dims["moat"] = pp_delta

        sector_note = f" [vs {sector}]" if sector else ""
        evidence.append(
            self.create_evidence(
                thesis_id=thesis_id,
                content=f"Gross margin {current_margin:.1%}{sector_note}, {margin_vs_sector:+.1%} vs sector",
                source="fmp",
                evidence_type=ev_type,
                confidence=0.75,
                dimension=ThesisDimension.MOAT,
                metadata={
                    "metric": "pricing_power",
                    "current_margin": current_margin,
                    "sector_median": sector_median,
                    "vs_sector": margin_vs_sector,
                },
            )
        )

        return evidence, delta, dims
