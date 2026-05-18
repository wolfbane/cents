"""Baseline + history + drift utilities for the eval harness.

Three small pieces of state live on disk:

- ``src/cents/eval/baseline.json`` (packaged with the project): the *locked*
  reference metrics. When ``locked_at`` is None the CI gate is permissive —
  ``cents eval run --persist-baseline`` writes today's metrics and stamps a
  timestamp.

- ``src/cents/eval/thresholds.json`` (packaged, optional): the *calibrated*
  sentiment thresholds, written by ``cents eval calibrate-thresholds``. If
  present, ``cents.agents.sentiment`` reads from here before falling back to
  the hardcoded defaults.

- ``~/.cents/data/eval_history/YYYY-MM-DD.jsonl`` (user state): per-run
  metric history. ``cents eval drift-check`` walks this directory to detect
  drift against a trailing median.

Everything here is read-or-write JSON / JSONL — no DB schema involvement.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, date
from importlib import resources
from pathlib import Path
from typing import Any

from cents.db.schema import get_db_path


# --- baseline.json (packaged) ---


def _packaged_baseline_path() -> Path:
    """Return the on-disk path to the packaged baseline.json."""
    # `resources.files` returns a Traversable; for editable installs (`pip
    # install -e .`) this is a regular Path we can write to. For wheel
    # installs, persistence would still need a writable override — we don't
    # support that today (cents is editable-install in practice).
    return Path(str(resources.files("cents.eval").joinpath("baseline.json")))


def load_baseline() -> dict[str, Any]:
    """Load baseline.json. Returns the default (permissive) record on missing/bad file."""
    path = _packaged_baseline_path()
    if not path.exists():
        return {
            "premise_f1": 0.0,
            "sentiment_brier": 1.0,
            "locked_at": None,
            "model_snapshot": None,
        }
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "premise_f1": 0.0,
            "sentiment_brier": 1.0,
            "locked_at": None,
            "model_snapshot": None,
        }


def persist_baseline(
    premise_f1: float | None,
    sentiment_brier: float | None,
    model_snapshot: str | None,
) -> dict[str, Any]:
    """Write today's metrics into baseline.json with `locked_at = now`.

    Missing metrics (eval skipped one set) are preserved at their prior values
    so a partial run doesn't corrupt the other half of the baseline.
    """
    path = _packaged_baseline_path()
    current = load_baseline()
    new_record = {
        "premise_f1": premise_f1 if premise_f1 is not None else current.get("premise_f1", 0.0),
        "sentiment_brier": (
            sentiment_brier if sentiment_brier is not None else current.get("sentiment_brier", 1.0)
        ),
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "model_snapshot": model_snapshot,
        "note": (
            "Locked baseline. The CI gate now compares incoming metrics to "
            "these values. Re-run `cents eval run --persist-baseline` to "
            "re-lock after an intentional model bump."
        ),
    }
    path.write_text(json.dumps(new_record, indent=2) + "\n")
    return new_record


def evaluate_gate(
    premise_f1: float | None,
    sentiment_brier: float | None,
    *,
    tolerance_pp: float,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare incoming metrics to the baseline.

    Returns a dict with `passed`, `permissive` (True when baseline isn't
    locked), `messages` (per-metric details), and the diffs themselves.

    A FAIL is recorded when either:
      - premise_f1 drops by more than ``tolerance_pp`` percentage points, OR
      - sentiment_brier worsens (rises) by more than ``tolerance_pp / 100``.

    Brier is bounded in [0, 4]; using the same magnitude as F1's pp comparator
    keeps the operator's mental model uniform. If you find that too sensitive
    or too lax, set a tighter ``tolerance-pp`` on the CLI.
    """
    if baseline is None:
        baseline = load_baseline()
    locked = baseline.get("locked_at") is not None
    messages: list[str] = []
    passed = True

    if not locked:
        return {
            "passed": True,
            "permissive": True,
            "messages": [
                "Baseline is not locked — gate is permissive. "
                "Run `cents eval run --persist-baseline` to lock the current metrics."
            ],
            "baseline": baseline,
        }

    tolerance_frac = tolerance_pp / 100.0
    premise_diff: float | None = None
    brier_diff: float | None = None

    if premise_f1 is not None:
        base_f1 = float(baseline.get("premise_f1", 0.0))
        premise_diff = premise_f1 - base_f1
        if base_f1 - premise_f1 > tolerance_frac:
            passed = False
            messages.append(
                f"premise_f1 regression: {premise_f1:.3f} vs baseline {base_f1:.3f} "
                f"(delta {premise_diff*100:+.1f}pp, tolerance {tolerance_pp:.1f}pp)"
            )
        else:
            messages.append(
                f"premise_f1 OK: {premise_f1:.3f} vs baseline {base_f1:.3f} "
                f"(delta {premise_diff*100:+.1f}pp)"
            )

    if sentiment_brier is not None:
        base_brier = float(baseline.get("sentiment_brier", 1.0))
        brier_diff = sentiment_brier - base_brier
        # Higher Brier is worse.
        if sentiment_brier - base_brier > tolerance_frac:
            passed = False
            messages.append(
                f"sentiment_brier regression: {sentiment_brier:.3f} vs baseline {base_brier:.3f} "
                f"(delta {brier_diff*100:+.1f}pp, tolerance {tolerance_pp:.1f}pp)"
            )
        else:
            messages.append(
                f"sentiment_brier OK: {sentiment_brier:.3f} vs baseline {base_brier:.3f} "
                f"(delta {brier_diff*100:+.1f}pp)"
            )

    return {
        "passed": passed,
        "permissive": False,
        "messages": messages,
        "baseline": baseline,
        "premise_diff": premise_diff,
        "brier_diff": brier_diff,
    }


# --- eval history (user state) ---


def _history_dir() -> Path:
    """Return ~/.cents/data/eval_history (created if missing).

    Anchored on `get_db_path()` so test fixtures pointing at a tmp DB path
    (via ``CENTS_DB_PATH``) automatically redirect history into the same
    sandbox.
    """
    db_path = get_db_path()
    base = db_path.parent / "eval_history"
    base.mkdir(parents=True, exist_ok=True)
    return base


def persist_history_row(
    *,
    premise_f1: float | None,
    sentiment_brier: float | None,
    premise_f1_ci: tuple[float, float] | None,
    sentiment_brier_ci: tuple[float, float] | None,
    model_snapshot: str | None,
    n_fixtures_premise: int | None,
    n_fixtures_sentiment: int | None,
    when: datetime | None = None,
) -> Path:
    """Append one history row to ~/.cents/data/eval_history/YYYY-MM-DD.jsonl.

    Returns the path written to. The file is appended (one row per call), so
    multiple eval runs in one day stack rather than overwrite.
    """
    when = when or datetime.now()
    fname = when.date().isoformat() + ".jsonl"
    path = _history_dir() / fname
    row = {
        "date": when.date().isoformat(),
        "ts": when.isoformat(timespec="seconds"),
        "premise_f1": premise_f1,
        "sentiment_brier": sentiment_brier,
        "premise_f1_ci": list(premise_f1_ci) if premise_f1_ci else None,
        "sentiment_brier_ci": list(sentiment_brier_ci) if sentiment_brier_ci else None,
        "model_snapshot": model_snapshot,
        "n_fixtures_premise": n_fixtures_premise,
        "n_fixtures_sentiment": n_fixtures_sentiment,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return path


def load_history(limit: int | None = None) -> list[dict]:
    """Read all history rows, oldest first.

    Sorted by date ascending. If multiple rows share a date (multiple runs
    in a day), insertion order within the file is preserved.
    """
    base = _history_dir()
    rows: list[dict] = []
    for fpath in sorted(base.glob("*.jsonl")):
        try:
            with fpath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    if limit is not None:
        rows = rows[-limit:]
    return rows


# --- drift detection ---


def detect_drift(
    *,
    threshold_pp: float = 5.0,
    window: int = 7,
    history: list[dict] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Compare today's premise_f1 to the trailing-window median F1.

    Returns a dict with `drift_detected`, `today_f1`, `median_f1`,
    `delta_pp`, `window_size`, and `messages`. `drift_detected` is True only
    when today's F1 is more than ``threshold_pp`` below the window median AND
    the window has at least 3 rows of history.
    """
    rows = history if history is not None else load_history()
    today = today or date.today()
    # Split history into today's rows and prior-window rows.
    todays_rows = [
        r for r in rows
        if r.get("date") == today.isoformat() and r.get("premise_f1") is not None
    ]
    prior_rows = [
        r for r in rows
        if r.get("date") != today.isoformat() and r.get("premise_f1") is not None
    ][-window:]

    messages: list[str] = []
    if not todays_rows:
        return {
            "drift_detected": False,
            "today_f1": None,
            "median_f1": None,
            "delta_pp": None,
            "window_size": len(prior_rows),
            "messages": ["No premise_f1 row recorded for today; nothing to compare."],
        }
    today_f1 = todays_rows[-1]["premise_f1"]  # newest row if multiple today.

    if len(prior_rows) < 3:
        return {
            "drift_detected": False,
            "today_f1": today_f1,
            "median_f1": None,
            "delta_pp": None,
            "window_size": len(prior_rows),
            "messages": [
                f"Insufficient history (window={len(prior_rows)} < 3); "
                "drift check skipped."
            ],
        }
    prior_f1s = sorted(r["premise_f1"] for r in prior_rows)
    n = len(prior_f1s)
    median_f1 = (
        prior_f1s[n // 2]
        if n % 2 == 1
        else (prior_f1s[n // 2 - 1] + prior_f1s[n // 2]) / 2.0
    )
    delta_pp = (today_f1 - median_f1) * 100.0
    drift = (median_f1 - today_f1) > (threshold_pp / 100.0)
    if drift:
        messages.append(
            f"DRIFT: today_f1={today_f1:.3f} below trailing-{n} median "
            f"{median_f1:.3f} by {abs(delta_pp):.1f}pp (threshold {threshold_pp:.1f}pp)"
        )
    else:
        messages.append(
            f"OK: today_f1={today_f1:.3f} vs trailing-{n} median {median_f1:.3f} "
            f"(delta {delta_pp:+.1f}pp, threshold {threshold_pp:.1f}pp)"
        )
    return {
        "drift_detected": drift,
        "today_f1": today_f1,
        "median_f1": median_f1,
        "delta_pp": delta_pp,
        "window_size": n,
        "messages": messages,
    }


# --- thresholds.json (packaged, optional) ---


def _packaged_thresholds_path() -> Path:
    return Path(str(resources.files("cents.eval").joinpath("thresholds.json")))


def load_thresholds() -> dict[str, float] | None:
    """Return calibrated sentiment thresholds if present, else None.

    Shape:
        {"positive_threshold": 0.3, "negative_threshold": -0.2,
         "calibrated_at": "...", "model_snapshot": "..."}

    The sentiment agent merges these over its hardcoded defaults; missing
    or unreadable thresholds.json means "use the hardcoded defaults".
    """
    path = _packaged_thresholds_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def persist_thresholds(
    *,
    positive_threshold: float,
    negative_threshold: float,
    accuracy: float,
    model_snapshot: str | None,
) -> Path:
    """Write the calibrated thresholds to thresholds.json."""
    path = _packaged_thresholds_path()
    record = {
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "calibrated_accuracy": accuracy,
        "model_snapshot": model_snapshot,
        "note": (
            "Calibrated by `cents eval calibrate-thresholds` on the sentiment "
            "golden set. Delete this file to revert to the hardcoded defaults."
        ),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
