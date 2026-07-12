"""Shared construction helpers for portable model export metadata."""

from __future__ import annotations

import json
import math
from typing import Any


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
