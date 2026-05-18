"""Threshold calibration over the sentiment golden set.

Today the sentiment agent hardcodes (positive_threshold=0.3, negative_threshold=-0.2)
for the LLM-score → band mapping. With the golden set in hand, we can search
for the threshold pair that maximises band-accuracy and write the result to
`thresholds.json` so the agent picks it up at runtime.

The calibration uses *only* the score and the expected band — it does NOT
hit the live API. To use it you need a list of fixtures with both
``score`` (the LLM-returned score in [-1, 1]) and ``expected_score_band``,
either by:

- running ``cents eval run --set sentiment --output json`` and feeding the
  ``fixtures`` array in (this requires a prior live run); OR
- supplying synthetic fixtures in tests.

In the CLI we wire this to ``cents eval calibrate-thresholds`` which loads
the golden set, runs the live API once to populate scores, then picks
thresholds. **Tests should never hit the live API** — they exercise the
``calibrate_thresholds`` helper directly with synthetic fixture lists.
"""

from __future__ import annotations

from dataclasses import dataclass


# Search-grid bounds. The task specified 0.1..0.5 step 0.05 for positive and
# -0.5..-0.1 step 0.05 for negative, which gives a 9x9 = 81-cell grid.
_POSITIVE_GRID = [round(0.1 + 0.05 * i, 2) for i in range(9)]  # 0.10..0.50
_NEGATIVE_GRID = [round(-0.5 + 0.05 * i, 2) for i in range(9)]  # -0.5..-0.1


def _bucket(score: float, positive_t: float, negative_t: float) -> str:
    """Map a score to a band using the given thresholds."""
    if score > positive_t:
        return "bullish"
    if score < negative_t:
        return "bearish"
    return "neutral"


def _band_distribution(predictions: list[str]) -> dict[str, int]:
    return {
        "bullish": sum(1 for p in predictions if p == "bullish"),
        "neutral": sum(1 for p in predictions if p == "neutral"),
        "bearish": sum(1 for p in predictions if p == "bearish"),
    }


def _band_balance(predictions: list[str]) -> float:
    """Penalty for degenerate predictions (everything one band).

    Returns a number in [0, 1] where 1 means perfect balance across the three
    bands and 0 means all-same-band. Used as a tie-breaker so the search
    doesn't pick a threshold pair that achieves "accuracy" by always
    predicting the most common band.
    """
    counts = _band_distribution(predictions)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    # Negative entropy normalised by log(3) — H/log(3) in [0,1].
    import math
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(3)


@dataclass(frozen=True)
class CalibrationResult:
    positive_threshold: float
    negative_threshold: float
    accuracy: float
    balance: float
    n_fixtures: int
    grid_searched: int  # how many threshold pairs were considered

    def to_dict(self) -> dict:
        return {
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "accuracy": self.accuracy,
            "balance": self.balance,
            "n_fixtures": self.n_fixtures,
            "grid_searched": self.grid_searched,
        }


def calibrate_thresholds(
    scored_fixtures: list[dict],
    *,
    positive_grid: list[float] | None = None,
    negative_grid: list[float] | None = None,
) -> CalibrationResult:
    """Find the threshold pair that maximises band-accuracy on the inputs.

    ``scored_fixtures`` must contain ``score`` (float in [-1, 1]) and
    ``expected_score_band`` (one of {bullish, neutral, bearish}). Fixtures
    missing either are silently skipped.

    Tie-breaking: when multiple threshold pairs share the top accuracy, pick
    the one with the highest band-balance (entropy across predicted bands).
    Without this, the search may collapse on degenerate thresholds where
    everything predicts one band.
    """
    positive_grid = positive_grid or _POSITIVE_GRID
    negative_grid = negative_grid or _NEGATIVE_GRID

    valid = [
        (float(f["score"]), f["expected_score_band"])
        for f in scored_fixtures
        if "score" in f and f.get("expected_score_band") in {"bullish", "neutral", "bearish"}
    ]
    if not valid:
        return CalibrationResult(
            positive_threshold=0.3,
            negative_threshold=-0.2,
            accuracy=0.0,
            balance=0.0,
            n_fixtures=0,
            grid_searched=0,
        )

    best: tuple[float, float, float, float] | None = None  # (acc, balance, pos, neg)
    grid_searched = 0
    for pos_t in positive_grid:
        for neg_t in negative_grid:
            if neg_t >= pos_t:
                # Skip degenerate inverted thresholds (every score is in two bands at once).
                continue
            grid_searched += 1
            preds = [_bucket(s, pos_t, neg_t) for s, _ in valid]
            correct = sum(1 for (s, eb), p in zip(valid, preds) if p == eb)
            acc = correct / len(valid)
            bal = _band_balance(preds)
            if best is None or (acc, bal) > (best[0], best[1]):
                best = (acc, bal, pos_t, neg_t)

    assert best is not None
    return CalibrationResult(
        positive_threshold=best[2],
        negative_threshold=best[3],
        accuracy=best[0],
        balance=best[1],
        n_fixtures=len(valid),
        grid_searched=grid_searched,
    )
