"""Random orchestrator — control arm for "did the LLM signal do anything?" experiments.

This agent mirrors OrchestratorAgent's ``research()`` surface but returns a
``conviction_delta`` drawn uniformly from ``[-30, +30]`` with no evidence and
no LLM calls. Theses opened by this orchestrator are tagged with
``orchestrator_label = "random"`` so cohort analytics can compare LLM-arm
outcomes against the matched random-arm outcomes — the falsifiability test
that makes the cents pipeline an experiment rather than a story.

Usage:

    cents factory run --orchestrator random

Or in code, inject explicitly:

    engine = FactoryEngine(orchestrator=RandomOrchestrator(seed=42))

Why this exists: the paired-neutral cohort is a control for *regime beta*,
not for *signal value*. Without a same-universe / same-cadence baseline,
no cohort spread the LLM arm produces can be attributed to the LLM signal
versus the act of opening theses in a momentum tape.
"""

from __future__ import annotations

import random as _random
from datetime import date

from cents.agents.base import (
    MAX_AGGREGATE_CONVICTION_DELTA,
    AgentResult,
    BaseAgent,
)
from cents.models import Thesis


class RandomOrchestrator(BaseAgent):
    """Drop-in replacement for OrchestratorAgent that emits random signals.

    The orchestrator label propagated to opened theses is ``"random"``. The
    seed parameter makes runs reproducible — without it, two control arms
    started at the same time would diverge on the first call.
    """

    name = "random_orchestrator"
    orchestrator_label = "random"

    def __init__(self, seed: int | None = None):
        super().__init__()
        # Use a dedicated Random instance so callers don't perturb the
        # global RNG and so we can re-seed deterministically per experiment.
        self._rng = _random.Random(seed)

    def research(
        self, symbol: str, thesis: Thesis | None = None,
        as_of: date | None = None,
    ) -> AgentResult:
        """Return a uniform-random AgentResult at the aggregate ±30 scale.

        ``as_of`` is accepted for signature parity with OrchestratorAgent so
        the two arms are plug-replaceable; the random control deliberately
        ignores it (the whole point is that nothing is conditioned on data).
        """
        delta = self._rng.uniform(
            -MAX_AGGREGATE_CONVICTION_DELTA, MAX_AGGREGATE_CONVICTION_DELTA,
        )
        return AgentResult(
            evidence=[],
            conviction_delta=delta,
            summary=f"random control: {symbol} → delta={delta:+.2f}",
            dimension_scores={},
            metadata={"orchestrator": "random"},
            aggregate=True,
        )
