"""Lockstep guard for the Spark-portable model shipped inside the package.

The wheel bundles a Spark-portable XGBoost booster + manifest
(``src/crosswalk/_model/spark_model.json`` / ``spark_manifest.json``) so Spark
consumers (the tf-data-platform sister project) can import them straight from
the package (``from crosswalk.spark import spark_model_json, spark_manifest``)
instead of hand-copying stale files.

The danger mirrors the combined-model case (``test_shipped_model.py``): if
``FEATURE_VERSION`` or ``SPARK_PORTABLE_FEATURES`` change without re-exporting,
the shipped Spark artifact scores against a stale feature contract with no
runtime guard (the Spark scorer just broadcasts ``manifest["features"]``).

This test fails whenever the shipped manifest diverges from the current config —
forcing a re-export + reship in the *same* PR. To reship:
``uv run crosswalk export-spark-model -o data/models/export`` then copy
``model.json``/``manifest.json`` to
``src/crosswalk/_model/spark_model.json``/``spark_manifest.json`` (see
docs/RELEASING.md).
"""

import json

import numpy as np
import pytest

from crosswalk.config import (
    FEATURE_VERSION,
    SPARK_PORTABLE_FEATURES,
    SPARK_PORTABLE_XGB_PARAMS,
    bundled_spark_manifest_path,
    bundled_spark_model_path,
)

_RESHIP = (
    "Re-export and reship the Spark model in this PR: "
    "uv run crosswalk export-spark-model -o data/models/export && "
    "cp data/models/export/model.json src/crosswalk/_model/spark_model.json && "
    "cp data/models/export/manifest.json src/crosswalk/_model/spark_manifest.json"
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


def test_shipped_spark_manifest_hyperparams_match_config(shipped_manifest):
    """Manifest hyperparams must match config.SPARK_PORTABLE_XGB_PARAMS.

    Without this, retuning the Spark hyperparams in config without re-exporting
    would ship a stale booster with no failing test (the booster itself carries
    no marker of the params it was trained with).
    """
    hyperparams = shipped_manifest.get("hyperparams") or {}
    mismatched = {
        k: (hyperparams.get(k), v)
        for k, v in SPARK_PORTABLE_XGB_PARAMS.items()
        if hyperparams.get(k) != v
    }
    assert not mismatched, (
        f"Shipped Spark manifest hyperparams diverge from config.SPARK_PORTABLE_XGB_PARAMS "
        f"(manifest, config): {mismatched}. {_RESHIP}"
    )


def test_shipped_spark_manifest_has_reproducible_training_metadata(shipped_manifest):
    """Spark consumers can identify the exact labels, split, and rows used."""
    metadata = shipped_manifest.get("training_metadata")
    assert metadata is not None, f"Shipped Spark manifest has no training provenance. {_RESHIP}"
    assert metadata.get("schema_version") == 1
    fingerprints = metadata.get("fingerprints") or {}
    assert set(fingerprints) == {
        "labeled_data_sha256",
        "split_sha256",
        "training_data_sha256",
    }
    assert all(isinstance(value, str) and len(value) == 64 for value in fingerprints.values())


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
    """The shipped booster loads in xgboost and predicts on a zeros row.

    NOTE: XGBoost's native JSON does not persist Python feature names, so this
    can only guard the feature *count* (num_feature), not name/order — the
    booster happily predicts on any 28 columns. Feature identity/order is
    guaranteed by construction: `crosswalk export-spark-model` writes
    ``matcher.feature_names`` (the training column order) into the manifest the
    booster ships with, and ``test_shipped_spark_manifest_features_match_config``
    pins that manifest to config. Consumers MUST feed columns in
    ``manifest["features"]`` order.
    """
    import xgboost as xgb

    from crosswalk.spark import spark_model_json

    booster = xgb.Booster()
    booster.load_model(bytearray(spark_model_json().encode()))
    features = shipped_manifest["features"]
    assert booster.num_features() == len(features), (
        f"Shipped booster expects {booster.num_features()} features but the manifest "
        f"declares {len(features)}. {_RESHIP}"
    )
    dmatrix = xgb.DMatrix(
        np.zeros((1, len(features)), dtype=np.float32),
        feature_names=features,
    )
    pred = booster.predict(dmatrix)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_shipped_booster_tree_count_matches_manifest(shipped_manifest):
    """Tie the booster to the manifest by TREE COUNT, not just feature count.

    ``test_shipped_spark_model_loads_and_predicts`` already checks
    ``num_features``, but that is the only model<->manifest link, and it is blind
    to the most likely reship mistake: retune the hyperparameters, update
    ``config`` and re-export ``manifest.json``, and forget to re-copy
    ``model.json``. When the feature count is unchanged -- which it is for any
    pure retune -- every other gate in this file still passes and the shipped
    manifest simply lies about the artifact beside it.

    ``num_boosted_rounds()`` closes that: it is read off the booster itself and
    ``n_estimators`` is written from the trained estimator's params, so a stale
    pairing diverges here.
    """
    import xgboost as xgb

    from crosswalk.spark import spark_model_json

    booster = xgb.Booster()
    booster.load_model(bytearray(spark_model_json().encode()))
    assert booster.num_boosted_rounds() == shipped_manifest["n_estimators"], (
        f"Shipped booster has {booster.num_boosted_rounds()} trees but the manifest "
        f"declares n_estimators={shipped_manifest['n_estimators']} — model.json and "
        f"manifest.json are from different exports. {_RESHIP}"
    )


def test_shipped_manifest_declares_a_feature_contract_version(shipped_manifest):
    """The manifest must carry a ``contract_version`` derived from the feature list.

    This is what a downstream consumer pins or diffs on. It has to be present and
    has to actually track the shipped feature list, or it is worse than nothing --
    a consumer would pin a constant and believe it was protected.

    Why it matters: XGBoost accepts a DMatrix with FEWER columns than the booster
    expects and predicts without raising, and the booster carries no feature
    names. The feature list also grows by insertion (it is kept in
    FEATURE_COLUMNS order), so a consumer on a stale list misaligns columns
    rather than merely dropping them. See crosswalk.spark.check_feature_columns.
    """
    from crosswalk.model_export import _feature_contract_version

    declared = shipped_manifest.get("contract_version")
    assert declared, f"Shipped Spark manifest has no contract_version. {_RESHIP}"
    expected = _feature_contract_version(list(shipped_manifest["features"]))
    assert declared == expected, (
        f"contract_version {declared!r} does not match the manifest's own feature "
        f"list (expected {expected!r}) — it is not tracking what it claims to. {_RESHIP}"
    )
