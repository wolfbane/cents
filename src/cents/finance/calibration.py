"""Conviction calibration — map raw `conviction_delta` onto `P(target | open)`.

The orchestrator's `conviction_delta` (±30) is a feature, not a probability.
This module fits a logistic regression on closed theses with realised outcomes
so that new theses can be opened with a calibrated `P(target hits before stop)`
that downstream sizing can use as a Kelly fraction.

Inputs (features) per thesis:
    aggregate_conviction_delta   the ±30 directional signal
    regime_net_polarity          regime_snapshot["net_polarity"]
    regime_event_count           regime_snapshot["recent_event_count"]
    one-hot discovery_*          (when pandas is available)
    one-hot cohort_*             (when pandas is available)

Labels: 1 when ``Thesis.outcome == ThesisOutcome.CORRECT``, 0 when
``ThesisOutcome.INCORRECT``. Unclear / preempted / invalidated theses are
excluded — they have no informative win/loss signal for calibration.

Persistence: ``~/.cents/data/calibration/YYYYMMDD.{joblib,pkl}``. We prefer
joblib when installed; otherwise we fall back to stdlib pickle (graceful —
no extra runtime deps required).
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from cents.models import Thesis, ThesisOutcome

# Optional sklearn / joblib / pandas — soft imports.
try:  # pragma: no cover — exercised only in environments that have sklearn
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import brier_score_loss, roc_auc_score  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore[assignment]
    _HAS_SKLEARN = False

try:  # pragma: no cover
    import joblib  # type: ignore

    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    joblib = None  # type: ignore[assignment]
    _HAS_JOBLIB = False

try:  # pragma: no cover
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


# Default location for serialised models. Tests override via the `dir`
# argument to `save_model` / `load_latest_model`.
DEFAULT_MODEL_DIR = Path.home() / ".cents" / "data" / "calibration"


@dataclass(frozen=True)
class CalibrationModel:
    """A fitted logistic-regression model over thesis features.

    `coef` keyed by feature name (so feature order isn't load-bearing once
    fitted). `feature_names` is preserved separately so `predict` can resolve
    missing features deterministically (treats absent feature as 0.0).
    """

    coef: dict[str, float]
    intercept: float
    brier_score: float  # in-sample (training set)
    auc: float          # in-sample (training set)
    n_observations: int  # total observations used (train + holdout)
    fit_at: datetime
    feature_names: list[str] = field(default_factory=list)
    # Bug F (r3): held-out metrics — None when no holdout was requested.
    # When set, these are the honest generalisation estimates; the
    # ``brier_score``/``auc`` fields above remain in-sample for back-compat.
    brier_holdout: float | None = None
    auc_holdout: float | None = None
    n_train: int | None = None
    n_holdout: int | None = None

    def predict(self, features: dict[str, float]) -> float:
        """Return P(label = 1) ∈ [0, 1] for a feature vector."""
        z = self.intercept
        for name in self.feature_names:
            z += self.coef.get(name, 0.0) * float(features.get(name, 0.0))
        # sigmoid with overflow protection
        if z >= 0:
            ez = math.exp(-z)
            p = 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            p = ez / (1.0 + ez)
        # Clamp into [0, 1] defensively.
        return max(0.0, min(1.0, p))


# ---- feature extraction -------------------------------------------------


def _extract_features(
    thesis: Thesis,
    delta: float | None = None,
    *,
    horizon_days: int | None = None,
) -> dict[str, float]:
    """Pull numeric features out of a thesis (or a candidate delta + thesis).

    For prediction-time use the caller can pass `delta` overriding the value
    from the (still-unsaved) thesis when conviction has not been persisted,
    and ``horizon_days`` overriding the persisted horizon (factory engine
    passes the configured default before the thesis row is written).
    """
    snapshot = thesis.regime_snapshot or {}
    aggregate_delta = delta if delta is not None else _thesis_implied_delta(thesis)
    horizon = horizon_days if horizon_days is not None else _thesis_horizon_days(thesis)
    return {
        "aggregate_conviction_delta": float(aggregate_delta or 0.0),
        "regime_net_polarity": float(snapshot.get("net_polarity") or 0.0),
        "regime_event_count": float(snapshot.get("recent_event_count") or 0.0),
        # Horizon: log1p so 30/60/90-day brackets don't trivially dominate
        # the linear coefficient. A short-horizon thesis closes faster and
        # is over-represented in the training set by count; this feature
        # lets the model account for that explicitly.
        "horizon_days_log1p": math.log1p(max(0, horizon)),
    }


def _thesis_horizon_days(thesis: Thesis) -> int:
    """Best-effort horizon in days from a Thesis. Falls back to 30 (the
    factory default) when nothing is persisted — better than treating an
    unset horizon as zero."""
    if getattr(thesis, "horizon_days", None):
        return int(thesis.horizon_days)
    horizon_end = getattr(thesis, "horizon_end", None)
    created_at = getattr(thesis, "created_at", None)
    if horizon_end and created_at:
        delta = horizon_end - created_at
        return max(1, int(delta.total_seconds() / 86400))
    return 30


def _thesis_implied_delta(thesis: Thesis) -> float:
    """Reconstruct the entry-time signed delta from a thesis.

    The factory engine derives target/stop from the delta sign and conviction
    from `50 + delta`. Inverting that lets calibration use any historical
    thesis even if the per-thesis delta wasn't stored anywhere else.
    """
    if thesis.conviction is None:
        return 0.0
    return float(thesis.conviction) - 50.0


def _one_hot(row: dict[str, str | None], prefix: str, value: str | None) -> None:
    if not value:
        return
    safe = str(value).replace(" ", "_")
    row[f"{prefix}_{safe}"] = 1.0


def _build_feature_rows(
    theses: Iterable[Thesis],
) -> tuple[list[dict[str, float]], list[int], list[datetime]]:
    """Turn closed theses into (X rows, y labels, closed_at timestamps).

    Random-arm theses are filtered out: their conviction_delta is uniform
    noise by construction, so including them in the training set shrinks
    the LLM-arm coefficient toward zero — the model becomes a blended
    "what happens to an opened thesis" model, not "what happens given the
    LLM's signal." We calibrate the LLM arm; random-arm theses get no
    predicted p_correct.

    Numeric features always present. Categorical features (discovery,
    cohort) emit as 1.0 indicators; caller unions feature names across rows.
    The third return value is the per-row close timestamp so callers can
    do a time-based holdout split (regime features otherwise leak under
    a random split).
    """
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    closed_at: list[datetime] = []
    for t in theses:
        if t.outcome not in (ThesisOutcome.CORRECT, ThesisOutcome.INCORRECT):
            continue
        if getattr(t, "orchestrator_label", "llm") != "llm":
            continue
        row = _extract_features(t)
        _one_hot(row, "discovery", t.discovery_source)
        _one_hot(row, "cohort", t.cohort.value if t.cohort is not None else None)
        rows.append(row)
        labels.append(1 if t.outcome == ThesisOutcome.CORRECT else 0)
        # Use closed_at if available, else updated_at, else created_at — the
        # holdout split needs SOMETHING monotonic to order rows by.
        ts = getattr(t, "closed_at", None) or getattr(t, "updated_at", None) or getattr(t, "created_at", None)
        closed_at.append(ts if isinstance(ts, datetime) else datetime.now())
    return rows, labels, closed_at


def _dense_matrix(
    rows: list[dict[str, float]],
) -> tuple[list[list[float]], list[str]]:
    """Pivot a list of sparse feature dicts into a dense matrix + feature names.

    Drops zero-variance columns before returning — the diagonal-Newton IRLS
    fitter overshoots when correlated constant features fight the intercept
    over the same direction. Removing constants is a clean fix and the
    information loss is zero (a feature that never varies in training carries
    no signal anyway; predict-time falls back to the column's absence).
    """
    feature_names: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                feature_names.append(k)
    matrix: list[list[float]] = []
    for r in rows:
        matrix.append([float(r.get(k, 0.0)) for k in feature_names])
    # Drop zero-variance columns.
    if matrix:
        keep_idx: list[int] = []
        for j in range(len(feature_names)):
            col = [row[j] for row in matrix]
            if max(col) != min(col):
                keep_idx.append(j)
        if len(keep_idx) != len(feature_names):
            feature_names = [feature_names[j] for j in keep_idx]
            matrix = [[row[j] for j in keep_idx] for row in matrix]
    return matrix, feature_names


# ---- metrics (pure-Python fallbacks) ------------------------------------


def _brier(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def _auc(probs: list[float], labels: list[int]) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney U) identity.

    Returns 0.5 (chance) when only one class is present so the metric is
    well-defined even on tiny / degenerate eval sets.
    """
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


# ---- fit (sklearn + IRLS fallback) --------------------------------------


def _fit_irls(
    matrix: list[list[float]],
    labels: list[int],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    l2: float = 1.0,
) -> tuple[list[float], float]:
    """Newton-Raphson IRLS for logistic regression with L2 regularisation.

    Pure-Python, no numpy. Returns (coefficients, intercept). The L2 term
    matches sklearn's default behaviour (regularises both weights and the
    intercept lightly) so we don't blow up on small / separable datasets.
    """
    n = len(matrix)
    if n == 0:
        return ([], 0.0)
    d = len(matrix[0]) if matrix else 0
    # Prepend a 1.0 bias column so we fit intercept alongside coefficients.
    X = [[1.0] + row for row in matrix]
    y = labels
    beta = [0.0] * (d + 1)

    def sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    for _ in range(max_iter):
        # Predict, score, hessian
        p = [sigmoid(sum(b * x for b, x in zip(beta, row))) for row in X]
        grad = [0.0] * (d + 1)
        for i, row in enumerate(X):
            err = p[i] - y[i]
            for j, xj in enumerate(row):
                grad[j] += err * xj
        # L2 on all (intercept gets the same penalty for simplicity)
        for j in range(d + 1):
            grad[j] += l2 * beta[j]

        # Approximate Hessian diagonal — pure-diagonal Newton step is more
        # stable than full Newton when the design matrix is rank-deficient
        # (tiny datasets) and keeps the implementation < 50 lines.
        hess_diag = [l2] * (d + 1)
        for i, row in enumerate(X):
            w = p[i] * (1 - p[i])
            for j, xj in enumerate(row):
                hess_diag[j] += w * xj * xj

        step = [g / h if h > 0 else 0.0 for g, h in zip(grad, hess_diag)]
        beta = [b - s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < tol:
            break

    intercept = beta[0]
    coefs = beta[1:]
    return coefs, intercept


def _score_logistic(matrix: list[list[float]], coef_vec, intercept) -> list[float]:
    """Compute sigmoid probabilities over a dense feature matrix."""
    probs = []
    for row in matrix:
        z = intercept + sum(c * x for c, x in zip(coef_vec, row))
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            p = ez / (1.0 + ez)
        probs.append(p)
    return probs


def fit_calibration(
    closed_theses: list[Thesis],
    *,
    min_observations: int = 30,
    holdout_pct: float = 0.0,
    holdout_seed: int = 17,
) -> CalibrationModel | None:
    """Fit a calibration model over closed (CORRECT / INCORRECT) theses.

    Returns ``None`` when fewer than ``min_observations`` rows have a
    decided outcome — the model would just memorise noise.

    When ``holdout_pct > 0``, the holdout is the most recent ``holdout_pct``
    fraction of rows by ``closed_at`` (time-based, not random). Regime
    features (regime_net_polarity, regime_event_count) are correlated within
    a time window — adjacent-in-time theses share near-identical regime
    snapshots — so a random split lets the model effectively "see the
    future." A trailing-edge holdout gives honest generalisation metrics.
    ``holdout_seed`` is no longer used but kept for back-compat.
    """
    rows, labels, closed_at = _build_feature_rows(closed_theses)
    if len(rows) < min_observations:
        return None

    matrix, feature_names = _dense_matrix(rows)

    # Time-based holdout split.
    holdout_matrix: list[list[float]] = []
    holdout_labels: list[int] = []
    if holdout_pct > 0.0:
        n = len(matrix)
        n_holdout = max(1, int(round(n * holdout_pct)))
        # Sort indices oldest→newest so the trailing-edge slice is holdout.
        order = sorted(range(n), key=lambda i: closed_at[i])
        train_idx = set(order[: n - n_holdout])
        train_matrix = [m for i, m in enumerate(matrix) if i in train_idx]
        train_labels = [l for i, l in enumerate(labels) if i in train_idx]
        holdout_matrix = [m for i, m in enumerate(matrix) if i not in train_idx]
        holdout_labels = [l for i, l in enumerate(labels) if i not in train_idx]
        # Refuse to fit if either split is too small to be informative.
        if len(train_matrix) < min_observations:
            return None
    else:
        train_matrix, train_labels = matrix, labels

    if _HAS_SKLEARN:  # pragma: no cover — covered by environments with sklearn
        import numpy as np  # local — numpy is already a transitive dep

        clf = LogisticRegression(max_iter=200, C=1.0)
        X = np.asarray(train_matrix)
        y = np.asarray(train_labels)
        clf.fit(X, y)
        coef_vec = clf.coef_[0].tolist()
        intercept = float(clf.intercept_[0])
        probs = clf.predict_proba(X)[:, 1].tolist()
        brier = float(brier_score_loss(y, probs))
        auc = float(roc_auc_score(y, probs)) if len(set(train_labels)) > 1 else 0.5
    else:
        coef_vec, intercept = _fit_irls(train_matrix, train_labels)
        probs = _score_logistic(train_matrix, coef_vec, intercept)
        brier = _brier(probs, train_labels)
        auc = _auc(probs, train_labels)

    brier_holdout = None
    auc_holdout = None
    if holdout_matrix:
        ho_probs = _score_logistic(holdout_matrix, coef_vec, intercept)
        brier_holdout = _brier(ho_probs, holdout_labels)
        auc_holdout = _auc(ho_probs, holdout_labels) if len(set(holdout_labels)) > 1 else 0.5

    coef_map = {name: float(c) for name, c in zip(feature_names, coef_vec)}
    return CalibrationModel(
        coef=coef_map,
        intercept=float(intercept),
        brier_score=float(brier),
        auc=float(auc),
        n_observations=len(rows),
        fit_at=datetime.now(),
        feature_names=feature_names,
        brier_holdout=brier_holdout,
        auc_holdout=auc_holdout,
        n_train=len(train_matrix),
        n_holdout=len(holdout_matrix) if holdout_matrix else None,
    )


# ---- persistence --------------------------------------------------------


def _model_filename(when: datetime) -> str:
    suffix = "joblib" if _HAS_JOBLIB else "pkl"
    return f"{when.strftime('%Y%m%d_%H%M%S')}.{suffix}"


def save_model(model: CalibrationModel, path: Path | None = None) -> Path:
    """Persist a model to disk. Returns the path written.

    Prefers joblib (smaller artefacts) when installed; falls back to
    pickle in the standard library so we have no hard runtime dep on it.
    """
    if path is None:
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = DEFAULT_MODEL_DIR / _model_filename(model.fit_at)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_JOBLIB:  # pragma: no cover
        joblib.dump(model, path)
    else:
        with path.open("wb") as fh:
            pickle.dump(model, fh)
    return path


def load_latest_model(dir: Path | None = None) -> CalibrationModel | None:
    """Load the most recently-written model from `dir`. None when missing/empty."""
    directory = Path(dir) if dir is not None else DEFAULT_MODEL_DIR
    if not directory.exists():
        return None
    candidates = sorted(
        (p for p in directory.iterdir() if p.suffix in {".joblib", ".pkl"}),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        if latest.suffix == ".joblib" and _HAS_JOBLIB:  # pragma: no cover
            return joblib.load(latest)
        with latest.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


# ---- feature extraction at predict time --------------------------------


def build_predict_features(
    *,
    delta: float,
    regime_snapshot: dict | None,
    discovery_source: str | None,
    cohort: str | None,
    horizon_days: int | None = None,
) -> dict[str, float]:
    """Build the predict-time feature dict, matching `_build_feature_rows`."""
    snapshot = regime_snapshot or {}
    features: dict[str, float] = {
        "aggregate_conviction_delta": float(delta),
        "regime_net_polarity": float(snapshot.get("net_polarity") or 0.0),
        "regime_event_count": float(snapshot.get("recent_event_count") or 0.0),
        "horizon_days_log1p": math.log1p(max(0, horizon_days if horizon_days is not None else 30)),
    }
    if discovery_source:
        safe = str(discovery_source).replace(" ", "_")
        features[f"discovery_{safe}"] = 1.0
    if cohort:
        safe = str(cohort).replace(" ", "_")
        features[f"cohort_{safe}"] = 1.0
    return features


# ---- reliability diagram (used by `cents calibration report`) ----------


def reliability_buckets(
    model: CalibrationModel,
    closed_theses: list[Thesis],
    *,
    n_buckets: int = 10,
) -> list[dict]:
    """Bucket predicted probability and report empirical hit-rate per bucket.

    Returns a list of dicts (one per non-empty bucket) ordered by bucket
    centre. Buckets are uniform over [0, 1].
    """
    rows, labels, _ = _build_feature_rows(closed_theses)
    if not rows:
        return []
    preds = [model.predict(r) for r in rows]
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_buckets)]
    for p, y in zip(preds, labels):
        idx = min(n_buckets - 1, int(p * n_buckets))
        buckets[idx].append((p, y))
    out: list[dict] = []
    for i, items in enumerate(buckets):
        if not items:
            continue
        avg_pred = sum(p for p, _ in items) / len(items)
        avg_actual = sum(y for _, y in items) / len(items)
        out.append({
            "bucket": i,
            "bucket_low": i / n_buckets,
            "bucket_high": (i + 1) / n_buckets,
            "n": len(items),
            "avg_predicted": avg_pred,
            "avg_actual": avg_actual,
        })
    return out
