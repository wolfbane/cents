"""Tests for the calibration module (pure unit, no DB)."""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path

import pytest

from cents.finance.calibration import (
    CalibrationModel,
    build_predict_features,
    fit_calibration,
    load_latest_model,
    reliability_buckets,
    save_model,
)
from cents.models import Thesis, ThesisOutcome, ThesisStatus


def _decided_thesis(*, label: int, delta: float, **kwargs) -> Thesis:
    """Build a closed thesis with a labeled CORRECT/INCORRECT outcome."""
    conviction = max(0.0, min(100.0, 50.0 + delta))
    t = Thesis(
        title=f"t-{delta}",
        symbol="X",
        conviction=conviction,
        regime_snapshot=kwargs.pop("regime_snapshot", {"net_polarity": 0, "recent_event_count": 0}),
        discovery_source=kwargs.pop("discovery_source", None),
        **kwargs,
    )
    t.status = ThesisStatus.CLOSED
    t.outcome = ThesisOutcome.CORRECT if label == 1 else ThesisOutcome.INCORRECT
    return t


class TestFitCalibration:
    def test_returns_none_below_min_observations(self):
        rng = random.Random(0)
        theses = [
            _decided_thesis(label=rng.randint(0, 1), delta=rng.uniform(-10, 10))
            for _ in range(5)
        ]
        assert fit_calibration(theses, min_observations=10) is None

    def test_ignores_undecided_outcomes(self):
        """Theses with non-correct/incorrect outcomes are excluded from the fit."""
        # 20 decided + 20 unclear/preempted — should not satisfy min_observations=30.
        decided = [
            _decided_thesis(label=i % 2, delta=float(i - 10)) for i in range(20)
        ]
        for outcome in (ThesisOutcome.UNCLEAR, ThesisOutcome.PREEMPTED, ThesisOutcome.INVALIDATED):
            for i in range(7):
                t = Thesis(title="x", symbol="X")
                t.status = ThesisStatus.CLOSED
                t.outcome = outcome
                decided.append(t)
        assert fit_calibration(decided, min_observations=30) is None

    def test_recovers_known_signed_coefficient(self):
        """A dataset where larger delta → higher win-rate should produce
        a positive coefficient on aggregate_conviction_delta."""
        rng = random.Random(42)
        theses = []
        for _ in range(150):
            delta = rng.uniform(-20, 20)
            # Logit truth: P(win) = sigmoid(0.2 * delta).
            p_true = 1 / (1 + math.exp(-0.2 * delta))
            label = 1 if rng.random() < p_true else 0
            theses.append(_decided_thesis(label=label, delta=delta))
        model = fit_calibration(theses, min_observations=30)
        assert model is not None
        coef = model.coef["aggregate_conviction_delta"]
        # The recovered slope should be positive — separating sign is the
        # load-bearing property. We don't assert proximity to 0.2 since the
        # IRLS fallback uses L2 regularisation that shrinks coefficients.
        assert coef > 0.0
        # Predictions should rank correctly.
        p_strong = model.predict({"aggregate_conviction_delta": 15.0})
        p_weak = model.predict({"aggregate_conviction_delta": -15.0})
        assert p_strong > p_weak

    def test_predict_clamps_to_unit_interval(self):
        """Even with extreme features, prediction lands in [0, 1]."""
        model = CalibrationModel(
            coef={"x": 1000.0},
            intercept=0.0,
            brier_score=0.0,
            auc=0.5,
            n_observations=1,
            fit_at=datetime.now(),
            feature_names=["x"],
        )
        assert 0.0 <= model.predict({"x": 1e9}) <= 1.0
        assert 0.0 <= model.predict({"x": -1e9}) <= 1.0
        # Missing feature treated as zero.
        assert math.isclose(model.predict({}), 0.5, abs_tol=1e-6)

    def test_brier_and_auc_populated(self):
        rng = random.Random(7)
        theses = [
            _decided_thesis(
                label=1 if rng.random() < 0.6 else 0,
                delta=rng.uniform(-15, 15),
            )
            for _ in range(60)
        ]
        model = fit_calibration(theses, min_observations=30)
        assert model is not None
        assert 0.0 <= model.brier_score <= 1.0
        assert 0.0 <= model.auc <= 1.0
        assert model.n_observations == 60


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        model = CalibrationModel(
            coef={"aggregate_conviction_delta": 0.1, "regime_net_polarity": 0.05},
            intercept=-0.2,
            brier_score=0.21,
            auc=0.68,
            n_observations=42,
            fit_at=datetime(2026, 1, 1, 12, 0, 0),
            feature_names=["aggregate_conviction_delta", "regime_net_polarity"],
        )
        path = save_model(model, path=tmp_path / "test.pkl")
        assert path.exists()

        loaded = load_latest_model(dir=tmp_path)
        assert loaded is not None
        assert loaded.coef == model.coef
        assert loaded.intercept == model.intercept
        assert loaded.n_observations == model.n_observations
        # Round-trip predict identity.
        features = {"aggregate_conviction_delta": 10.0, "regime_net_polarity": 1.0}
        assert math.isclose(loaded.predict(features), model.predict(features), abs_tol=1e-9)

    def test_load_latest_returns_none_when_empty(self, tmp_path: Path):
        assert load_latest_model(dir=tmp_path) is None

    def test_load_latest_returns_none_when_missing_dir(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        assert load_latest_model(dir=missing) is None

    def test_load_latest_picks_most_recent(self, tmp_path: Path):
        # Write two models, ensure load_latest_model picks the newest by mtime.
        old = CalibrationModel(
            coef={"x": 0.0}, intercept=-1.0, brier_score=0.0, auc=0.5,
            n_observations=10, fit_at=datetime(2026, 1, 1), feature_names=["x"],
        )
        new = CalibrationModel(
            coef={"x": 0.0}, intercept=+1.0, brier_score=0.0, auc=0.5,
            n_observations=20, fit_at=datetime(2026, 2, 1), feature_names=["x"],
        )
        save_model(old, path=tmp_path / "a.pkl")
        # Force a later mtime on the second file.
        import time
        time.sleep(0.01)
        save_model(new, path=tmp_path / "b.pkl")
        loaded = load_latest_model(dir=tmp_path)
        assert loaded is not None
        assert loaded.intercept == 1.0
        assert loaded.n_observations == 20


class TestPredictFeatures:
    def test_build_predict_features_includes_one_hot(self):
        features = build_predict_features(
            delta=12.0,
            regime_snapshot={"net_polarity": -2, "recent_event_count": 7},
            discovery_source="value",
            cohort="directional",
        )
        assert features["aggregate_conviction_delta"] == 12.0
        assert features["regime_net_polarity"] == -2.0
        assert features["regime_event_count"] == 7.0
        assert features["discovery_value"] == 1.0
        assert features["cohort_directional"] == 1.0

    def test_build_predict_features_handles_none(self):
        features = build_predict_features(
            delta=0.0,
            regime_snapshot=None,
            discovery_source=None,
            cohort=None,
        )
        assert features["regime_net_polarity"] == 0.0
        assert features["regime_event_count"] == 0.0
        # No spurious one-hot keys when categorical inputs are missing.
        assert not any(k.startswith("discovery_") for k in features)
        assert not any(k.startswith("cohort_") for k in features)


class TestReliabilityBuckets:
    def test_reliability_buckets_returns_non_empty_only(self):
        rng = random.Random(11)
        theses = [
            _decided_thesis(label=rng.randint(0, 1), delta=rng.uniform(-10, 10))
            for _ in range(40)
        ]
        model = fit_calibration(theses, min_observations=20)
        assert model is not None
        buckets = reliability_buckets(model, theses, n_buckets=5)
        assert all(b["n"] > 0 for b in buckets)
        for b in buckets:
            assert 0.0 <= b["avg_predicted"] <= 1.0
            assert 0.0 <= b["avg_actual"] <= 1.0
