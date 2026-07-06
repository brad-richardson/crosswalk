"""Zero-cost accessors for the Spark-portable model shipped in the package.

Spark consumers (the tf-data-platform sister project) can ``pip install
road-matcher`` and read the bundled XGBoost-native booster + manifest straight
from the wheel — no hand-copying stale files, no heavy imports. This module is
deliberately import-light: at *module import* time it touches **only the
standard library** (``importlib.resources`` / ``json``), so ``import
matcher.spark`` never drags in shapely/geopandas/xgboost/pandas. ``numpy`` is
imported lazily inside :func:`apply_calibration` (the one function that needs
it), so a job that only reads the model/manifest bytes stays numpy-free too.

The shipped artifacts live at ``matcher/_model/spark_model.json`` and
``matcher/_model/spark_manifest.json``; their ``feature_version`` is kept in
lockstep with ``config.FEATURE_VERSION`` by
``tests/unit/test_shipped_spark_model.py``.

Typical Spark use::

    from matcher.spark import spark_model_json, spark_manifest, apply_calibration

    manifest = spark_manifest()
    features = manifest["features"]          # broadcast; column order matters
    booster.load_model(bytearray(spark_model_json().encode()))
    # ... score to raw P(match), then optionally:
    calibrated = apply_calibration(raw_scores, manifest["calibration"])
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

# Package subdirectory + filenames for the shipped Spark artifacts. Kept in sync
# with config.bundled_spark_model_path / bundled_spark_manifest_path and the
# pyproject wheel `artifacts` include rules.
_MODEL_PACKAGE = "matcher._model"
_MODEL_RESOURCE = "spark_model.json"
_MANIFEST_RESOURCE = "spark_manifest.json"


def spark_model_json() -> str:
    """Return the Spark-portable XGBoost booster as a JSON string.

    Load it into an ``xgboost.Booster`` with
    ``booster.load_model(bytearray(spark_model_json().encode()))``.
    """
    return resources.files(_MODEL_PACKAGE).joinpath(_MODEL_RESOURCE).read_text(encoding="utf-8")


def spark_manifest() -> dict[str, Any]:
    """Return the Spark-portable model manifest as a dict.

    Keys include ``features`` (ordered feature list — broadcast to the scorer),
    ``feature_version``, ``hyperparams``, and ``calibration`` (isotonic knots
    under ``x_thresholds``/``y_thresholds`` with ``applied: false``).
    """
    text = resources.files(_MODEL_PACKAGE).joinpath(_MANIFEST_RESOURCE).read_text(encoding="utf-8")
    return json.loads(text)


def apply_calibration(scores: Any, knots: dict[str, Any]) -> np.ndarray:
    """Apply the isotonic calibration knots to raw ``P(match)`` scores.

    Piecewise-linear interpolation over ``knots["x_thresholds"]`` /
    ``knots["y_thresholds"]`` with endpoint clipping — ``np.interp`` clamps to
    the first/last knot value outside the knot range, matching
    ``IsotonicRegression(out_of_bounds="clip")`` and reproducing
    ``matcher.matching.calibration.IsotonicCalibrator.transform`` exactly.

    ``numpy`` is imported lazily here (not at module import) so that a job which
    only reads the model/manifest bytes never pays for numpy.

    Args:
        scores: Raw predicted probabilities (array-like).
        knots: The manifest's ``calibration`` dict (or any dict with
            ``x_thresholds`` / ``y_thresholds`` lists).

    Returns:
        Calibrated probabilities as a ``float64`` numpy array (a 0-d
        ``np.float64`` scalar for scalar input, matching ``np.interp`` /
        ``IsotonicCalibrator.transform`` semantics).
    """
    import numpy as np

    x_thresholds = np.asarray(knots["x_thresholds"], dtype=np.float64)
    y_thresholds = np.asarray(knots["y_thresholds"], dtype=np.float64)
    return np.interp(np.asarray(scores, dtype=np.float64), x_thresholds, y_thresholds)
