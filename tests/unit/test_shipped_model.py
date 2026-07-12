"""Lockstep guard for the pretrained model shipped inside the package.

The wheel bundles a pretrained model (``src/crosswalk/_model/matcher_model_combined.joblib``)
so a fresh clone / ``pip install`` can ``crosswalk stitch`` with zero training. The
danger: model load only *warns* by default on a ``feature_version`` mismatch (now a
hard error unless explicitly allowed), and the shipped path deliberately trusts the
artifact. If ``FEATURE_VERSION`` bumps without re-exporting the shipped model, the
bundled model would silently score against a stale feature contract.

This test fails whenever the shipped model's ``feature_version`` diverges from the
current ``FEATURE_VERSION`` — forcing a retrain + reship in the *same* PR that bumps
features. To reship: ``uv run crosswalk train -o src/crosswalk/_model/matcher_model_combined.joblib``.
"""

import joblib
import pytest

from crosswalk.config import FEATURE_COLUMNS, FEATURE_VERSION, bundled_model_path


@pytest.fixture(scope="module")
def shipped_model():
    path = bundled_model_path()
    assert path.exists(), (
        f"Shipped model missing at {path}. It must be committed so a fresh clone / "
        "pip install can stitch without training. Reship with: "
        "uv run crosswalk train -o src/crosswalk/_model/matcher_model_combined.joblib"
    )
    return joblib.load(path)


def test_shipped_model_feature_version_in_lockstep(shipped_model):
    """The shipped model's feature_version MUST equal the current FEATURE_VERSION.

    This is the CI lockstep gate: bumping FEATURE_VERSION without reshipping the
    model fails here, so features and the shipped artifact stay in sync.
    """
    shipped_version = shipped_model.get("feature_version")
    assert shipped_version == FEATURE_VERSION, (
        f"Shipped model feature_version={shipped_version!r} != current "
        f"FEATURE_VERSION={FEATURE_VERSION!r}. Retrain and reship the bundled model "
        "in this PR: uv run crosswalk train -o src/crosswalk/_model/matcher_model_combined.joblib"
    )


def test_shipped_model_has_active_calibration(shipped_model):
    """Calibration must be present — decision thresholds assume it is active (#287/#288)."""
    calibration = shipped_model.get("calibration")
    assert calibration is not None, (
        "Shipped model has no calibration knots. Thresholds assume an active "
        "isotonic calibrator; reship a calibrated model."
    )
    assert calibration.get("x_thresholds") and calibration.get("y_thresholds"), (
        "Shipped model calibration knots are empty."
    )


def test_shipped_model_feature_names_match_config(shipped_model):
    """Feature ordering in the shipped model must match config.FEATURE_COLUMNS."""
    assert shipped_model.get("feature_names") == FEATURE_COLUMNS


def test_shipped_model_has_reproducible_training_metadata(shipped_model):
    """The release artifact identifies its exact data, split, and training rows."""
    metadata = shipped_model.get("training_metadata")
    assert metadata is not None, (
        "Shipped model has no training provenance. Retrain and reship it with the "
        "current deterministic training pipeline."
    )
    assert metadata.get("schema_version") == 1
    fingerprints = metadata.get("fingerprints") or {}
    assert set(fingerprints) == {
        "labeled_data_sha256",
        "split_sha256",
        "training_data_sha256",
    }
    assert all(isinstance(value, str) and len(value) == 64 for value in fingerprints.values())


def test_shipped_model_loads_via_mlmatcher():
    """The shipped model loads cleanly through MLMatcher (no version mismatch)."""
    from crosswalk.matching.ml import MLMatcher

    matcher = MLMatcher(model_path=str(bundled_model_path()))
    assert matcher.model is not None
    assert matcher.feature_version == FEATURE_VERSION
    assert matcher.calibrator is not None


def test_pipeline_calibration_probe_sees_bundled_model(tmp_path, monkeypatch):
    """The calibration probe must inspect the bundled model when no local model exists.

    Regression guard: the optimizer's glue operating point depends on
    _calibration_active(). If the probe only looked at settings.model_path, the
    fresh-clone path (bundled fallback) would score with calibrated confidences
    but prune at the raw operating point.
    """
    from crosswalk.config import settings
    from crosswalk.pipeline.runner import _calibration_active, _default_model_path

    monkeypatch.setattr(settings, "model_path", tmp_path / "nonexistent.joblib")
    assert _default_model_path() == bundled_model_path()
    if settings.enable_calibration:
        assert _calibration_active() is True
