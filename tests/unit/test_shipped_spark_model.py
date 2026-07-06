"""Lockstep guard for the Spark-portable model shipped inside the package.

The wheel bundles a Spark-portable XGBoost booster + manifest
(``src/matcher/_model/spark_model.json`` / ``spark_manifest.json``) so Spark
consumers (the tf-data-platform sister project) can import them straight from
the package (``from matcher.spark import spark_model_json, spark_manifest``)
instead of hand-copying stale files.

The danger mirrors the combined-model case (``test_shipped_model.py``): if
``FEATURE_VERSION`` or ``SPARK_PORTABLE_FEATURES`` change without re-exporting,
the shipped Spark artifact scores against a stale feature contract with no
runtime guard (the Spark scorer just broadcasts ``manifest["features"]``).

This test fails whenever the shipped manifest diverges from the current config —
forcing a re-export + reship in the *same* PR. To reship:
``uv run matcher export-spark-model -o data/models/export`` then copy
``model.json``/``manifest.json`` to
``src/matcher/_model/spark_model.json``/``spark_manifest.json`` (see
docs/RELEASING.md).
"""

import json

import numpy as np
import pytest

from matcher.config import (
    FEATURE_VERSION,
    SPARK_PORTABLE_FEATURES,
    bundled_spark_manifest_path,
    bundled_spark_model_path,
)

_RESHIP = (
    "Re-export and reship the Spark model in this PR: "
    "uv run matcher export-spark-model -o data/models/export && "
    "cp data/models/export/model.json src/matcher/_model/spark_model.json && "
    "cp data/models/export/manifest.json src/matcher/_model/spark_manifest.json"
)


@pytest.fixture(scope="module")
def shipped_manifest():
    path = bundled_spark_manifest_path()
    assert path.exists(), (
        f"Shipped Spark manifest missing at {path}. It must be committed so a "
        f"pip install can consume it. {_RESHIP}"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_shipped_spark_model_file_exists():
    path = bundled_spark_model_path()
    assert path.exists(), f"Shipped Spark model missing at {path}. {_RESHIP}"


def test_shipped_spark_manifest_feature_version_in_lockstep(shipped_manifest):
    """The shipped manifest's feature_version MUST equal the current FEATURE_VERSION."""
    shipped_version = shipped_manifest.get("feature_version")
    assert shipped_version == FEATURE_VERSION, (
        f"Shipped Spark manifest feature_version={shipped_version!r} != current "
        f"FEATURE_VERSION={FEATURE_VERSION!r}. {_RESHIP}"
    )


def test_shipped_spark_manifest_features_match_config(shipped_manifest):
    """Feature list must be exactly SPARK_PORTABLE_FEATURES, order preserved.

    The Spark scorer broadcasts this list as the column order for the DMatrix,
    so any drift silently misaligns features to the booster.
    """
    assert shipped_manifest.get("features") == SPARK_PORTABLE_FEATURES, (
        "Shipped Spark manifest features differ from config.SPARK_PORTABLE_FEATURES "
        f"(order-sensitive). {_RESHIP}"
    )
    assert shipped_manifest.get("n_features") == len(SPARK_PORTABLE_FEATURES)


def test_shipped_spark_manifest_calibration_present_and_monotonic(shipped_manifest):
    """Calibration knots must be present and monotonic (isotonic remap).

    The knots ship so the Spark job can apply interp(score, xs, ys); a
    non-monotone or empty table would be a broken calibrator.
    """
    calibration = shipped_manifest.get("calibration")
    assert calibration is not None, f"Shipped Spark manifest has no calibration knots. {_RESHIP}"
    xs = calibration.get("x_thresholds")
    ys = calibration.get("y_thresholds")
    assert xs and ys, "Shipped Spark calibration knots are empty."
    assert len(xs) == len(ys), "Calibration knot arrays have mismatched lengths."
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    assert np.all(np.diff(xs_arr) >= 0), "Calibration x_thresholds are not non-decreasing."
    assert np.all(np.diff(ys_arr) >= 0), "Calibration y_thresholds are not non-decreasing."


def test_shipped_spark_model_loads_and_predicts(shipped_manifest):
    """The shipped booster loads in xgboost and predicts on a zeros row."""
    import xgboost as xgb

    from matcher.spark import spark_model_json

    booster = xgb.Booster()
    booster.load_model(bytearray(spark_model_json().encode()))
    features = shipped_manifest["features"]
    dmatrix = xgb.DMatrix(
        np.zeros((1, len(features)), dtype=np.float32),
        feature_names=features,
    )
    pred = booster.predict(dmatrix)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])
