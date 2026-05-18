"""Eval runner: exercises the LIVE Anthropic-backed classifiers against
hand-authored golden sets.

Two evals:

- Premise tag classification (multi-label set membership). For each fixture,
  predicted_tags is compared to expected_tags. We report micro precision,
  recall, F1 across the whole set, plus per-fixture details and a per-tag
  confusion summary.

- Sentiment scoring (regression onto [-1, +1]). For each fixture, the LLM
  emits a score, we bucket it into bullish (>0.3) / neutral / bearish (<-0.3),
  and report a 3x3 confusion matrix + a Brier score against pseudo-targets
  (bullish=+0.6, neutral=0, bearish=-0.6) so a single number tracks both
  direction and magnitude drift.

The runner constructs the Anthropic client itself and uses the SAME LLM call
sites (`classify_premise_tags`, `SentimentAgent._score_with_llm`) as production
— it is a read-only invocation, never mutating the DB.

TODO(cron): wire to a scheduled job; today this is human-invoked via
`cents eval run`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable, Iterator

from cents.config import get_settings
from cents.models import EVENT_TAGS

logger = logging.getLogger(__name__)


# --- Golden-set loading ---


def _golden_path(filename: str):
    """Return a `Path`-like for a packaged golden-set file."""
    return resources.files("cents.eval").joinpath(filename)


def _iter_jsonl(filename: str) -> Iterator[dict]:
    path = _golden_path(filename)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_premise_golden(limit: int | None = None) -> list[dict]:
    """Load the premise golden set. Validates expected_tags against EVENT_TAGS."""
    fixtures: list[dict] = []
    for rec in _iter_jsonl("golden_premise.jsonl"):
        for tag in rec.get("expected_tags", []):
            if tag not in EVENT_TAGS:
                raise ValueError(
                    f"Premise fixture {rec.get('id')} has expected_tag '{tag}' "
                    f"not in EVENT_TAGS vocabulary."
                )
        fixtures.append(rec)
        if limit is not None and len(fixtures) >= limit:
            break
    return fixtures


_VALID_BANDS = {"bullish", "neutral", "bearish"}


def load_sentiment_golden(limit: int | None = None) -> list[dict]:
    """Load the sentiment golden set. Validates expected_score_band."""
    fixtures: list[dict] = []
    for rec in _iter_jsonl("golden_sentiment.jsonl"):
        band = rec.get("expected_score_band")
        if band not in _VALID_BANDS:
            raise ValueError(
                f"Sentiment fixture {rec.get('id')} has invalid "
                f"expected_score_band '{band}' (must be one of {_VALID_BANDS})."
            )
        fixtures.append(rec)
        if limit is not None and len(fixtures) >= limit:
            break
    return fixtures


# --- Scoring helpers ---


def _bucket_score(score: float) -> str:
    """Map a float score in [-1, 1] to one of {bullish, neutral, bearish}.

    Matches the thresholds documented in golden_sentiment.jsonl: bullish >0.3,
    bearish <-0.3, otherwise neutral.
    """
    if score > 0.3:
        return "bullish"
    if score < -0.3:
        return "bearish"
    return "neutral"


# Pseudo-targets for the Brier-style score. Sentiment is a regression, but our
# golden labels are categorical bands — we map the band to a representative
# score in the middle of the band so a single metric (mean squared error
# against pseudo-target) captures both direction and magnitude drift.
_BAND_TARGET = {"bullish": 0.6, "neutral": 0.0, "bearish": -0.6}


def _confusion_matrix(rows: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Build a {expected: {predicted: count}} confusion matrix.

    All three bands always appear in both dimensions so the JSON output shape
    is stable and easy to render.
    """
    matrix = {
        e: {p: 0 for p in _VALID_BANDS} for e in _VALID_BANDS
    }
    for expected, predicted in rows:
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1
    return matrix


# --- Result dataclasses ---


@dataclass
class PremiseEvalResult:
    """Per-fixture record + aggregate metrics for premise eval."""

    fixtures_run: int
    # Per-fixture details: [{id, symbol, expected, predicted, tp, fp, fn}]
    fixtures: list[dict] = field(default_factory=list)
    # Aggregate (micro) — sums TP/FP/FN across all fixtures.
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    skipped_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "set": "premise",
            "fixtures_run": self.fixtures_run,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fixtures": self.fixtures,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class SentimentEvalResult:
    fixtures_run: int
    fixtures: list[dict] = field(default_factory=list)
    correct_band: int = 0
    accuracy: float = 0.0
    brier_score: float = 0.0  # Mean squared error vs pseudo-target.
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "set": "sentiment",
            "fixtures_run": self.fixtures_run,
            "correct_band": self.correct_band,
            "accuracy": self.accuracy,
            "brier_score": self.brier_score,
            "confusion_matrix": self.confusion_matrix,
            "fixtures": self.fixtures,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class EvalResult:
    """Combined result. Either or both sub-results may be None if skipped."""

    premise: PremiseEvalResult | None = None
    sentiment: SentimentEvalResult | None = None
    model: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "premise": self.premise.to_dict() if self.premise else None,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
        }


# --- Anthropic client helper ---


def _build_anthropic_client():
    """Return an Anthropic client or None if the key / library isn't available.

    Returns None silently — the caller decides how to report it.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


_SKIP_MSG = (
    "ANTHROPIC_API_KEY not configured; eval skipped. "
    "Set ANTHROPIC_API_KEY env var or anthropic_api_key in ~/.cents/config.toml."
)


# --- Premise eval ---


def run_premise_eval(
    limit: int | None = None,
    *,
    anthropic_client=None,
) -> PremiseEvalResult:
    """Run the premise classifier against the golden set.

    Pass `anthropic_client` for tests; otherwise the Settings-derived client is
    used. Returns a result with `skipped_reason` set if no client is available.
    """
    fixtures = load_premise_golden(limit=limit)
    client = anthropic_client if anthropic_client is not None else _build_anthropic_client()
    if client is None:
        return PremiseEvalResult(
            fixtures_run=0,
            skipped_reason=_SKIP_MSG,
        )

    # Late import to avoid a circular: classify_premise_tags pulls in
    # `cents.factory` which has no eval-related side-effects but is a heavy
    # graph.
    from cents.factory.premise import classify_premise_tags

    result = PremiseEvalResult(fixtures_run=0)
    for fixture in fixtures:
        predicted_tags, _predicted_directions = classify_premise_tags(
            fixture["symbol"],
            fixture["thesis_summary"],
            fixture.get("evidence", []),
            anthropic_client=client,
        )
        expected_set = set(fixture["expected_tags"])
        predicted_set = set(predicted_tags)
        tp = len(expected_set & predicted_set)
        fp = len(predicted_set - expected_set)
        fn = len(expected_set - predicted_set)
        result.tp += tp
        result.fp += fp
        result.fn += fn
        result.fixtures_run += 1
        result.fixtures.append({
            "id": fixture["id"],
            "symbol": fixture["symbol"],
            "expected": sorted(expected_set),
            "predicted": sorted(predicted_set),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    denom_p = result.tp + result.fp
    denom_r = result.tp + result.fn
    result.precision = result.tp / denom_p if denom_p else 0.0
    result.recall = result.tp / denom_r if denom_r else 0.0
    denom_f = result.precision + result.recall
    result.f1 = (
        2 * result.precision * result.recall / denom_f if denom_f else 0.0
    )
    return result


# --- Sentiment eval ---


def run_sentiment_eval(
    limit: int | None = None,
    *,
    anthropic_client=None,
) -> SentimentEvalResult:
    """Run the sentiment scorer against the golden set.

    Reuses `SentimentAgent._score_with_llm` so we exercise the same prompt and
    parsing the production agent does. The article-cache is bypassed by
    constructing articles without a URL.
    """
    fixtures = load_sentiment_golden(limit=limit)
    client = anthropic_client if anthropic_client is not None else _build_anthropic_client()
    if client is None:
        return SentimentEvalResult(
            fixtures_run=0,
            confusion_matrix=_confusion_matrix([]),
            skipped_reason=_SKIP_MSG,
        )

    from cents.agents.sentiment import SentimentAgent
    from cents.models import Thesis

    agent = SentimentAgent(anthropic_client=client)
    result = SentimentEvalResult(
        fixtures_run=0,
        confusion_matrix=_confusion_matrix([]),
    )
    band_rows: list[tuple[str, str]] = []
    squared_errors: list[float] = []

    for fixture in fixtures:
        article = {
            "title": fixture["article_title"],
            "description": fixture.get("description", ""),
            # No URL so we never hit the production score cache.
            "url": "",
        }
        thesis = Thesis(title="eval", hypothesis=fixture.get("thesis", ""))
        _ev_type, score, _confidence, metadata, _provenance = agent._score_with_llm(
            article, fixture["symbol"], thesis
        )
        # The agent falls back to keyword scoring on LLM failure; keyword
        # scores are integer counts (-3..3 ish) rather than [-1, 1], so we
        # only count LLM results in the eval. Falling back means the LLM
        # failed — we surface that as a fixture-level error.
        method = metadata.get("scoring_method")
        if method == "llm":
            normalized_score = max(-1.0, min(1.0, float(score)))
            predicted_band = _bucket_score(normalized_score)
            squared_err = (normalized_score - _BAND_TARGET[fixture["expected_score_band"]]) ** 2
            squared_errors.append(squared_err)
        else:
            normalized_score = 0.0
            predicted_band = "neutral"
            # Penalize fallback with full squared error against the target.
            squared_errors.append(
                (0.0 - _BAND_TARGET[fixture["expected_score_band"]]) ** 2
            )

        expected_band = fixture["expected_score_band"]
        if predicted_band == expected_band:
            result.correct_band += 1
        band_rows.append((expected_band, predicted_band))
        result.fixtures_run += 1
        result.fixtures.append({
            "id": fixture["id"],
            "symbol": fixture["symbol"],
            "expected_band": expected_band,
            "predicted_band": predicted_band,
            "score": normalized_score,
            "scoring_method": method,
        })

    result.confusion_matrix = _confusion_matrix(band_rows)
    result.accuracy = (
        result.correct_band / result.fixtures_run if result.fixtures_run else 0.0
    )
    result.brier_score = (
        sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
    )
    return result


# --- Top-level run helper ---


def run_eval(
    sets: str = "all",
    limit: int | None = None,
    *,
    anthropic_client=None,
) -> EvalResult:
    """Run one or both evals. `sets` is 'premise' | 'sentiment' | 'all'."""
    if sets not in {"premise", "sentiment", "all"}:
        raise ValueError(f"Unknown set '{sets}'. Choose premise, sentiment, or all.")

    premise_res: PremiseEvalResult | None = None
    sentiment_res: SentimentEvalResult | None = None

    if sets in {"premise", "all"}:
        premise_res = run_premise_eval(limit=limit, anthropic_client=anthropic_client)
    if sets in {"sentiment", "all"}:
        sentiment_res = run_sentiment_eval(limit=limit, anthropic_client=anthropic_client)

    # Pull model name off whichever client we ended up using. We can't know
    # without an Anthropic SDK probe what model was actually used — record the
    # constant used by the two call sites.
    from cents.factory.premise import _LLM_MODEL as PREMISE_MODEL  # noqa: WPS437
    return EvalResult(
        premise=premise_res,
        sentiment=sentiment_res,
        model=PREMISE_MODEL,
    )
