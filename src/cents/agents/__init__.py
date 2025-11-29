"""Research agents for cents."""

from cents.agents.base import BaseAgent, AgentResult
from cents.agents.fundamentals import FundamentalsAgent
from cents.agents.technical import TechnicalAgent
from cents.agents.macro import MacroAgent
from cents.agents.sentiment import SentimentAgent
from cents.agents.orchestrator import OrchestratorAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "FundamentalsAgent",
    "TechnicalAgent",
    "MacroAgent",
    "SentimentAgent",
    "OrchestratorAgent",
]

# Registry of available agents
AGENTS = {
    "fundamentals": FundamentalsAgent,
    "technical": TechnicalAgent,
    "macro": MacroAgent,
    "sentiment": SentimentAgent,
    "orchestrator": OrchestratorAgent,
}
