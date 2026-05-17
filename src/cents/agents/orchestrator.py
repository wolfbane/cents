"""Orchestrator agent - runs and synthesizes all agents."""

from datetime import date, datetime

from cents.agents.base import BaseAgent, AgentResult
from cents.agents.fundamentals import FundamentalsAgent
from cents.agents.technical import TechnicalAgent
from cents.agents.macro import MacroAgent
from cents.agents.sentiment import SentimentAgent
from cents.agents.moat import MoatAgent
from cents.agents.insider import InsiderAgent
from cents.agents.event import EventAgent
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension

# Evidence TTL by dimension (days) - older evidence is weighted less
DIMENSION_TTL_DAYS: dict[str, int] = {
    "technical": 7,
    "sentiment": 7,
    "macro": 30,
    "valuation": 30,
    "quality": 90,
    "moat": 90,
    "risk": 30,
}
DEFAULT_TTL_DAYS = 30
AGE_WEIGHT_FLOOR = 0.1  # Minimum weight for very old evidence


def evidence_age_weight(evidence: Evidence) -> float:
    """Calculate age-based weight for evidence.

    Returns 1.0 for fresh evidence, decaying linearly to AGE_WEIGHT_FLOOR
    as evidence approaches its dimension-specific TTL.

    Evidence older than TTL still gets AGE_WEIGHT_FLOOR (not fully ignored)
    to preserve historical context.
    """
    dimension = evidence.dimension.value if evidence.dimension else None
    ttl = DIMENSION_TTL_DAYS.get(dimension, DEFAULT_TTL_DAYS)

    age_days = (datetime.now() - evidence.timestamp).days
    if age_days <= 0:
        return 1.0
    if age_days >= ttl:
        return AGE_WEIGHT_FLOOR

    # Linear decay from 1.0 to AGE_WEIGHT_FLOOR over TTL period
    decay_range = 1.0 - AGE_WEIGHT_FLOOR
    return 1.0 - (age_days / ttl) * decay_range


class OrchestratorAgent(BaseAgent):
    """Agent that orchestrates all research agents and synthesizes results."""

    name = "orchestrator"

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.agents = [
            FundamentalsAgent(),
            TechnicalAgent(),
            MacroAgent(),
            SentimentAgent(),
            EventAgent(),
        ]

        # Only include FMP-dependent agents when API key is configured to avoid startup failures
        if settings.fmp_api_key:
            self.agents.extend([MoatAgent(), InsiderAgent()])

    def _weighted_conviction(self, result: AgentResult) -> float:
        """Weight conviction delta by evidence confidence and age.

        High-confidence, fresh evidence influences conviction more than
        low-confidence or stale evidence. If no evidence, use raw delta.

        The weighting combines:
        - Confidence (0-1): How certain the agent is about the finding
        - Age weight (0.1-1): How fresh the evidence is (decays per dimension TTL)
        """
        if not result.evidence or result.conviction_delta == 0:
            return result.conviction_delta

        # Weight each evidence by both confidence and age
        weighted_sum = sum(
            e.confidence * evidence_age_weight(e) for e in result.evidence
        )
        avg_weighted = weighted_sum / len(result.evidence)
        return result.conviction_delta * avg_weighted

    def research(
        self, symbol: str, thesis: Thesis | None = None, as_of: date | None = None
    ) -> AgentResult:
        """Run all agents and synthesize results."""
        thesis_id = thesis.id if thesis else None

        all_evidence = []
        agent_results = {}
        total_conviction_delta = 0.0
        aggregated_dimensions: dict[str, float] = {}

        # Run each agent
        for agent in self.agents:
            result = agent.research(symbol, thesis, as_of=as_of)
            agent_results[agent.name] = result
            all_evidence.extend(result.evidence)

            # Weight conviction delta by evidence confidence
            weighted_delta = self._weighted_conviction(result)
            total_conviction_delta += weighted_delta

            # Aggregate dimension scores
            for dim, score in result.dimension_scores.items():
                aggregated_dimensions[dim] = aggregated_dimensions.get(dim, 0) + score

        # Synthesize results
        synthesis = self._synthesize(symbol, agent_results, thesis_id, aggregated_dimensions)
        all_evidence.append(synthesis)

        # Build summary with weighted deltas
        summaries = []
        for name, result in agent_results.items():
            weighted = self._weighted_conviction(result)
            if weighted != 0:
                sign = "+" if weighted > 0 else ""
                summaries.append(f"{name}: {sign}{weighted:.1f}")

        if summaries:
            summary = f"{symbol} synthesis: " + " | ".join(summaries) + f" = {total_conviction_delta:+.1f} total (weighted)"
        else:
            summary = f"{symbol}: No significant signals from any agent"

        return AgentResult(
            evidence=all_evidence,
            conviction_delta=total_conviction_delta,
            summary=summary,
            dimension_scores=aggregated_dimensions,
        )

    def _synthesize(
        self,
        symbol: str,
        results: dict[str, AgentResult],
        thesis_id: str,
        dimension_scores: dict[str, float],
    ) -> Evidence:
        """Create synthesis evidence from all agent results."""
        # Count supporting vs contradicting signals
        supporting = 0
        contradicting = 0
        neutral = 0

        for result in results.values():
            for e in result.evidence:
                if e.type == EvidenceType.SUPPORTING:
                    supporting += 1
                elif e.type == EvidenceType.CONTRADICTING:
                    contradicting += 1
                else:
                    neutral += 1

        # Determine overall signal
        total = supporting + contradicting + neutral
        if total == 0:
            synthesis_type = EvidenceType.NEUTRAL
            synthesis_text = "Insufficient data for synthesis"
        elif supporting > contradicting * 1.5:
            synthesis_type = EvidenceType.SUPPORTING
            synthesis_text = f"Overall bullish: {supporting} supporting vs {contradicting} contradicting signals"
        elif contradicting > supporting * 1.5:
            synthesis_type = EvidenceType.CONTRADICTING
            synthesis_text = f"Overall bearish: {contradicting} contradicting vs {supporting} supporting signals"
        else:
            synthesis_type = EvidenceType.NEUTRAL
            synthesis_text = f"Mixed signals: {supporting} supporting, {contradicting} contradicting, {neutral} neutral"

        # Agent agreement
        deltas = [r.conviction_delta for r in results.values()]
        bullish_agents = sum(1 for d in deltas if d > 0)
        bearish_agents = sum(1 for d in deltas if d < 0)

        if bullish_agents >= 3:
            synthesis_text += " | Strong agent agreement (bullish)"
        elif bearish_agents >= 3:
            synthesis_text += " | Strong agent agreement (bearish)"
        elif bullish_agents >= 2 and bearish_agents == 0:
            synthesis_text += " | Moderate bullish consensus"
        elif bearish_agents >= 2 and bullish_agents == 0:
            synthesis_text += " | Moderate bearish consensus"

        # Add dimension summary
        if dimension_scores:
            dim_parts = []
            for dim, score in sorted(dimension_scores.items()):
                if score != 0:
                    dim_parts.append(f"{dim}: {score:+.0f}")
            if dim_parts:
                synthesis_text += " | Dimensions: " + ", ".join(dim_parts)

        return self.create_evidence(
            thesis_id=thesis_id,
            content=synthesis_text,
            source="orchestrator",
            evidence_type=synthesis_type,
            confidence=0.75,
            metadata={
                "supporting": supporting,
                "contradicting": contradicting,
                "neutral": neutral,
                "bullish_agents": bullish_agents,
                "bearish_agents": bearish_agents,
                "dimension_scores": dimension_scores,
            },
        )
