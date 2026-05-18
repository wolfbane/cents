"""Pre-registered experiment scaffolding (cents-hvz).

An experiment binds a hypothesis, primary metric, sample-size target, and
the factory configuration in effect at registration time. Once active, the
engine stamps every opened thesis with the experiment's id; ``factory run``
warns when the live ``factory.toml`` drifts from the frozen SHA.

The YAML schema accepted by ``cents experiment register``:

    experiment: llm_vs_random_v1
    hypothesis: "LLM-arm paired-cohort win_rate exceeds random-arm by >= 10pp"
    primary_metric: paired_cohort_win_rate_delta
    minimum_n_per_arm: 300
    stopping_rule: "first of: 90 days OR N=300 per arm"

The ``frozen_config_sha`` field on the Experiment record is computed from
the current ``factory.toml`` at registration time — the engine treats any
drift from that SHA as a discipline violation.
"""

from cents.experiments.registry import (
    REQUIRED_FIELDS,
    ExperimentSpecError,
    compute_factory_config_sha,
    finalize_experiment,
    get_active_experiment,
    load_experiment_spec,
    register_experiment,
    status_snapshot,
)

__all__ = [
    "REQUIRED_FIELDS",
    "ExperimentSpecError",
    "compute_factory_config_sha",
    "finalize_experiment",
    "get_active_experiment",
    "load_experiment_spec",
    "register_experiment",
    "status_snapshot",
]
