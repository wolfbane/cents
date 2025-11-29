"""Orchestrator agent - runs and synthesizes all agents."""

from typing import Optional

from cents.agents.base import BaseAgent, AgentResult
from cents.agents.fundamentals import FundamentalsAgent
from cents.agents.technical import TechnicalAgent
from cents.agents.macro import MacroAgent
from cents.agents.sentiment import SentimentAgent
from cents.models import Evidence, EvidenceType, Thesis


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
        ]

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Run all agents and synthesize results."""
        thesis_id = thesis.id if thesis else "standalone"

        all_evidence = []
        agent_results = {}
        total_conviction_delta = 0.0

        # Run each agent
        for agent in self.agents:
            result = agent.research(symbol, thesis)
            agent_results[agent.name] = result
            all_evidence.extend(result.evidence)
            total_conviction_delta += result.conviction_delta

        # Synthesize results
        synthesis = self._synthesize(symbol, agent_results, thesis_id)
        all_evidence.append(synthesis)

        # Build summary
        summaries = []
        for name, result in agent_results.items():
            if result.conviction_delta != 0:
                sign = "+" if result.conviction_delta > 0 else ""
                summaries.append(f"{name}: {sign}{result.conviction_delta:.0f}")

        if summaries:
            summary = f"{symbol} synthesis: " + " | ".join(summaries) + f" = {total_conviction_delta:+.0f} total"
        else:
            summary = f"{symbol}: No significant signals from any agent"

        return AgentResult(
            evidence=all_evidence,
            conviction_delta=total_conviction_delta,
            summary=summary,
        )

    def _synthesize(
        self, symbol: str, results: dict[str, AgentResult], thesis_id: str
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
            },
        )
