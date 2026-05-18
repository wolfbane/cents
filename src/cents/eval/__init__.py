"""LLM eval harness for cents classifiers.

Four LLM call sites in cents do classification:
- `cents.factory.premise.classify_premise_tags` — selects controlled-vocab tags
- `cents.agents.event.EventAgent._tag_event` — tags Federal Register events
- `cents.agents.sentiment.SentimentAgent._filter_relevant_articles` — picks relevant news
- `cents.agents.sentiment.SentimentAgent._score_with_llm` — scores news -1..+1

All four currently mock the LLM in tests. This package provides a golden-set
eval harness that exercises the LIVE Anthropic API and reports precision /
recall (for tag-set classifiers) and a Brier score (for sentiment regression).

Usage:
    cents eval run --set all
    cents eval golden show --set premise

The runner skips with a clear message when ANTHROPIC_API_KEY is not set, so
it's safe to import in test contexts.

TODO(cron): wire this into a nightly job once the golden sets have stabilized.
For now, this is a manual `cents eval run` invoked by a human looking for
drift across model upgrades.
"""

from cents.eval.baseline import (
    detect_drift,
    evaluate_gate,
    load_baseline,
    load_history,
    load_thresholds,
    persist_baseline,
    persist_history_row,
    persist_thresholds,
)
from cents.eval.calibrate import CalibrationResult, calibrate_thresholds
from cents.eval.runner import (
    EvalResult,
    PremiseEvalResult,
    SentimentEvalResult,
    bootstrap_ci,
    load_premise_golden,
    load_sentiment_golden,
    run_premise_eval,
    run_sentiment_eval,
)

__all__ = [
    "CalibrationResult",
    "EvalResult",
    "PremiseEvalResult",
    "SentimentEvalResult",
    "bootstrap_ci",
    "calibrate_thresholds",
    "detect_drift",
    "evaluate_gate",
    "load_baseline",
    "load_history",
    "load_premise_golden",
    "load_sentiment_golden",
    "load_thresholds",
    "persist_baseline",
    "persist_history_row",
    "persist_thresholds",
    "run_premise_eval",
    "run_sentiment_eval",
]
