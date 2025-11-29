"""Research agents for cents."""

from cents.agents.base import BaseAgent, AgentResult
from cents.agents.fundamentals import FundamentalsAgent
from cents.agents.technical import TechnicalAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "FundamentalsAgent",
    "TechnicalAgent",
]

# Registry of available agents
AGENTS = {
    "fundamentals": FundamentalsAgent,
    "technical": TechnicalAgent,
}
