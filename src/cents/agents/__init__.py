"""Research agents for cents."""

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.agents.fundamentals import FundamentalsAgent
from cents.agents.technical import TechnicalAgent
from cents.agents.macro import MacroAgent
from cents.agents.sentiment import SentimentAgent
from cents.agents.moat import MoatAgent
from cents.agents.insider import InsiderAgent
from cents.agents.orchestrator import OrchestratorAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "RECOVERABLE_EXCEPTIONS",
    "FundamentalsAgent",
    "TechnicalAgent",
    "MacroAgent",
    "SentimentAgent",
    "MoatAgent",
    "InsiderAgent",
    "OrchestratorAgent",
]

# Registry of available agents
AGENTS = {
    "fundamentals": FundamentalsAgent,
    "technical": TechnicalAgent,
    "macro": MacroAgent,
    "sentiment": SentimentAgent,
    "moat": MoatAgent,
    "insider": InsiderAgent,
    "orchestrator": OrchestratorAgent,
}
