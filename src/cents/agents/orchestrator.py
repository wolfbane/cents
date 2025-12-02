"""Orchestrator agent - runs and synthesizes all agents."""

from cents.agents.base import BaseAgent, AgentResult
from cents.agents.fundamentals import FundamentalsAgent
from cents.agents.technical import TechnicalAgent
from cents.agents.macro import MacroAgent
from cents.agents.sentiment import SentimentAgent
from cents.agents.moat import MoatAgent
from cents.agents.insider import InsiderAgent
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension


class OrchestratorAgent(BaseAgent):
    """Agent that orchestrates all research agents and synthesizes results."""

    name = "orchestrator"

    def __init__(self):
        super().__init__()
        self.agents = [
            FundamentalsAgent(),
            TechnicalAgent(),
            MacroAgent(),
            SentimentAgent(),
            MoatAgent(),
            InsiderAgent(),
        ]

    def _weighted_conviction(self, result: AgentResult) -> float:
        """Weight conviction delta by average evidence confidence.

        High-confidence evidence should influence conviction more than
        low-confidence evidence. If no evidence, use raw delta.
        """
        if not result.evidence or result.conviction_delta == 0:
            return result.conviction_delta

        avg_confidence = sum(e.confidence for e in result.evidence) / len(result.evidence)
        return result.conviction_delta * avg_confidence

    def research(self, symbol: str, thesis: Thesis | None = None) -> AgentResult:
        """Run all agents and synthesize results."""
        thesis_id = thesis.id if thesis else "standalone"

        all_evidence = []
        agent_results = {}
        total_conviction_delta = 0.0
        aggregated_dimensions: dict[str, float] = {}

        # Run each agent
        for agent in self.agents:
            result = agent.research(symbol, thesis)
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
