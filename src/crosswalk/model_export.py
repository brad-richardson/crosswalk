"""Shared construction helpers for portable model export metadata."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _feature_contract_version(features: list[str]) -> str:
    """Short stable hash of the ordered feature list.

    Order-sensitive on purpose: the scorer builds its DMatrix in this order, so a
    reordering is as breaking as an addition even though the feature *set* is
    unchanged.
    """
    digest = hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()
    return f"{len(features)}f-{digest[:12]}"


def build_spark_model_manifest(matcher: Any) -> dict[str, Any]:
    """Build the manifest written by ``crosswalk export-spark-model``.

    Keeping this construction outside the CLI makes the serialized contract
    directly testable. In particular, native and Spark artifacts must carry the
    same ``training_metadata`` record, including source-commit provenance.
    """
    xgb_params = matcher.model.get_params()
    hyperparams: dict[str, Any] = {}
    for key, value in xgb_params.items():
        if value is None or callable(value):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        try:
            json.dumps(value)
            hyperparams[key] = value
        except (TypeError, ValueError):
            hyperparams[key] = str(value)

    manifest: dict[str, Any] = {
        "features": matcher.feature_names,
        "n_features": len(matcher.feature_names),
        # Stable fingerprint of the FEATURE CONTRACT -- the ordered feature list a
        # consumer must emit. It changes whenever a feature is added, removed, or
        # reordered, and does NOT change on a retrain or hyperparameter retune.
        #
        # This exists because the failure mode it guards is silent. XGBoost accepts
        # a DMatrix with FEWER columns than the booster expects and predicts
        # without error, and the booster carries no feature names. The list also
        # grows by insertion (it is kept in FEATURE_COLUMNS order), so a consumer
        # pinned to an older list misaligns every column after the first insertion
        # point rather than merely missing the new ones -- measured at 10x score
        # inflation on an obvious non-match when the 28-column list met the
        # 34-feature booster. A consumer can pin or diff on this instead of
        # discovering the change in production. See crosswalk.spark for the
        # matching `check_feature_columns` guard.
        "contract_version": _feature_contract_version(matcher.feature_names),
        "n_estimators": xgb_params.get("n_estimators"),
        "threshold": 0.5,
        "is_binary": matcher.is_binary,
        "feature_version": matcher.feature_version,
        "label_encoder": matcher.label_encoder,
        "hyperparams": hyperparams,
        "training_metadata": matcher.training_metadata,
    }
    if matcher.calibrator is not None:
        calibration = matcher.calibrator.to_knots()
        calibration["applied"] = False
        manifest["calibration"] = calibration
    return manifest
