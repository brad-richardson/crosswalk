"""Unit tests for probability calibration (isotonic)."""

import numpy as np
import pytest

from matcher.matching.calibration import (
    MIN_CALIBRATION_ROWS,
    IsotonicCalibrator,
    apply_knots,
    brier_score,
    expected_calibration_error,
    fit_isotonic_oof,
)


@pytest.fixture
def noisy_probs():
    """Miscalibrated scores: true prob is score**2, so raw scores are overconfident-low."""
    rng = np.random.RandomState(0)
    scores = rng.rand(2000)
    labels = (rng.rand(2000) < scores**2).astype(int)
    return scores, labels


def test_fit_returns_calibrator(noisy_probs):
    scores, labels = noisy_probs
    calib = fit_isotonic_oof(scores, labels)
    assert isinstance(calib, IsotonicCalibrator)
    assert calib.method == "isotonic"


def test_calibration_improves_ece(noisy_probs):
    scores, labels = noisy_probs
    calib = fit_isotonic_oof(scores, labels)
    ece_raw = expected_calibration_error(scores, labels)
    ece_cal = expected_calibration_error(calib.transform(scores), labels)
    assert ece_cal < ece_raw


def test_transform_is_monotone(noisy_probs):
    scores, labels = noisy_probs
    calib = fit_isotonic_oof(scores, labels)
    xs = np.linspace(0, 1, 101)
    ys = calib.transform(xs)
    assert np.all(np.diff(ys) >= -1e-9)


def test_transform_clips_to_unit_interval(noisy_probs):
    scores, labels = noisy_probs
    calib = fit_isotonic_oof(scores, labels)
    out = calib.transform(np.array([-0.5, 0.0, 1.0, 1.5]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_matches_sklearn_transform(noisy_probs):
    """Portable knot interpolation must reproduce sklearn.transform bit-for-bit."""
    from sklearn.isotonic import IsotonicRegression

    scores, labels = noisy_probs
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, labels)
    calib = IsotonicCalibrator.from_sklearn(iso)
    test = np.linspace(-0.1, 1.1, 200)
    np.testing.assert_allclose(calib.transform(test), iso.transform(test), atol=1e-12)


def test_knots_round_trip(noisy_probs):
    scores, labels = noisy_probs
    calib = fit_isotonic_oof(scores, labels)
    knots = calib.to_knots()
    # JSON-serialisable primitives
    assert knots["method"] == "isotonic"
    assert all(isinstance(x, float) for x in knots["x_thresholds"])
    restored = IsotonicCalibrator.from_knots(knots)
    test = np.linspace(0, 1, 50)
    np.testing.assert_allclose(restored.transform(test), calib.transform(test), atol=1e-12)


def test_apply_knots_matches_transform(noisy_probs):
    scores, labels = noisy_probs
    calib = fit_isotonic_oof(scores, labels)
    test = np.linspace(0, 1, 30)
    np.testing.assert_allclose(
        apply_knots(test, calib.x_thresholds, calib.y_thresholds),
        calib.transform(test),
    )


def test_insufficient_data_returns_none():
    rng = np.random.RandomState(1)
    n = MIN_CALIBRATION_ROWS - 1
    scores = rng.rand(n)
    labels = (rng.rand(n) < 0.5).astype(int)
    assert fit_isotonic_oof(scores, labels) is None


def test_single_class_returns_none():
    scores = np.linspace(0, 1, 500)
    labels = np.ones(500, dtype=int)
    assert fit_isotonic_oof(scores, labels) is None


def test_nan_oof_rows_dropped():
    """Rows without an OOF prediction (NaN) must be ignored, not crash the fit."""
    rng = np.random.RandomState(2)
    scores = rng.rand(500)
    labels = (rng.rand(500) < scores).astype(int)
    scores[::5] = np.nan  # 20% missing OOF
    calib = fit_isotonic_oof(scores, labels)
    assert isinstance(calib, IsotonicCalibrator)


def test_ece_and_brier_perfect_predictor():
    labels = np.array([0, 0, 1, 1])
    probs = labels.astype(float)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
    assert brier_score(probs, labels) == pytest.approx(0.0)


def test_ece_empty_is_nan():
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))
    assert np.isnan(brier_score(np.array([]), np.array([])))


def _tiny_matcher():
    """MLMatcher wrapping a trivial 2-feature XGBoost model and a calibrator."""
    xgb = pytest.importorskip("xgboost")
    from matcher.matching.ml import MLMatcher

    rng = np.random.RandomState(3)
    X = rng.rand(300, 2)
    y = (X[:, 0] + rng.rand(300) * 0.3 > 0.7).astype(int)
    m = MLMatcher()
    m.feature_names = ["f0", "f1"]
    m.model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
    m.model.fit(X, y)
    m.feature_version = 999
    return m


def test_matcher_save_load_round_trips_calibrator(tmp_path):
    m = _tiny_matcher()
    # A non-identity calibrator so we can detect it survived the round trip.
    m.calibrator = IsotonicCalibrator(
        x_thresholds=np.array([0.0, 1.0]), y_thresholds=np.array([0.2, 0.9])
    )
    path = tmp_path / "m.joblib"
    m.save_model(str(path))

    from matcher.matching.ml import MLMatcher

    loaded = MLMatcher(str(path))
    assert loaded.calibrator is not None
    test = np.linspace(0, 1, 20)
    np.testing.assert_allclose(
        loaded.calibrator.transform(test), m.calibrator.transform(test), atol=1e-12
    )


def test_matcher_save_load_without_calibrator(tmp_path):
    m = _tiny_matcher()
    m.calibrator = None
    path = tmp_path / "m.joblib"
    m.save_model(str(path))

    from matcher.matching.ml import MLMatcher

    loaded = MLMatcher(str(path))
    assert loaded.calibrator is None


def test_predict_calibrated_flag(monkeypatch):
    m = _tiny_matcher()
    m.calibrator = IsotonicCalibrator(
        x_thresholds=np.array([0.0, 1.0]), y_thresholds=np.array([0.0, 0.5])
    )
    feats = [{"f0": 0.9, "f1": 0.1}, {"f0": 0.2, "f1": 0.5}]
    raw = m.predict(feats, calibrated=False)
    cal = m.predict(feats, calibrated=True)
    # calibrator halves the score -> calibrated strictly below raw for raw>0
    assert np.all(cal <= raw + 1e-9)
    assert np.any(cal < raw)

    # settings.enable_calibration=False disables calibration even with a calibrator
    from matcher import config

    monkeypatch.setattr(config.settings, "enable_calibration", False)
    np.testing.assert_allclose(m.predict(feats, calibrated=True), raw)
