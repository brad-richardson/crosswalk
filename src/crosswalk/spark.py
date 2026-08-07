"""Zero-cost accessors for the Spark-portable model shipped in the package.

Spark consumers (the tf-data-platform sister project) can ``pip install
crosswalk-py`` and read the bundled XGBoost-native booster + manifest straight
from the wheel — no hand-copying stale files, no heavy imports. This module is
deliberately import-light: at *module import* time it touches **only the
standard library** (``importlib.resources`` / ``json``), so ``import
crosswalk.spark`` never drags in shapely/geopandas/xgboost/pandas. ``numpy`` is
imported lazily inside :func:`apply_calibration` (the one function that needs
it), so a job that only reads the model/manifest bytes stays numpy-free too.

The shipped artifacts live at ``crosswalk/_model/spark_model.json`` and
``crosswalk/_model/spark_manifest.json``; their ``feature_version`` is kept in
lockstep with ``config.FEATURE_VERSION`` by
``tests/unit/test_shipped_spark_model.py``.

Typical Spark use::

    from crosswalk.spark import (
        spark_model_json, spark_manifest, apply_calibration, check_feature_columns
    )

    manifest = spark_manifest()
    features = manifest["features"]          # broadcast; column order matters
    check_feature_columns(df.columns, manifest)   # fail fast, see below
    booster.load_model(bytearray(spark_model_json().encode()))
    # ... score to raw P(match), then optionally:
    calibrated = apply_calibration(raw_scores, manifest["calibration"])

.. warning::

   **Feeding too FEW columns does not raise — it silently mis-scores.** XGBoost
   only errors when a DMatrix has *more* columns than the booster expects; with
   fewer it pads and predicts happily. The booster carries no feature names
   (``booster.feature_names is None``), so it cannot self-check either.

   This matters because the feature list grows by **insertion**, not appending:
   it is kept in ``config.FEATURE_COLUMNS`` order, so the 6 name features added
   on 2026-08-07 landed at indices 9 and 11-15. A consumer pinned to the older
   28-column list therefore misaligns every column from index 9 onward rather
   than merely missing the new ones. Measured on the toy rows in
   ``tests/integration/test_spark_sedona_e2e.py``, an obvious non-match scored
   **0.033 correctly and 0.321 through a stale 28-column vector** — a 10x
   inflation, well across a 0.5 threshold, with no error raised anywhere.

   Always select columns *by name in* ``manifest["features"]`` order, and call
   :func:`check_feature_columns` at job start so a wheel upgrade fails loudly
   instead of quietly degrading. ``manifest["contract_version"]`` changes
   whenever the feature list changes, so a consumer can also pin on it.
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
_MODEL_PACKAGE = "crosswalk._model"
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
    ``crosswalk.matching.calibration.IsotonicCalibrator.transform`` exactly.

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


class FeatureContractError(ValueError):
    """Raised when a consumer's columns do not satisfy the manifest contract."""


def check_feature_columns(columns: Any, manifest: dict[str, Any] | None = None) -> list[str]:
    """Validate that ``columns`` can supply every feature the booster expects.

    Call this once at job start. It exists because the failure it catches is
    otherwise **silent**: an ``xgboost.DMatrix`` with fewer columns than the
    booster was trained on predicts without error, and the shipped booster
    carries no feature names to cross-check against. See the module docstring
    for the measured 10x mis-scoring this produces.

    Args:
        columns: Any iterable of column names (e.g. ``spark_df.columns``).
        manifest: Parsed manifest; loaded via :func:`spark_manifest` if omitted.

    Returns:
        The manifest's feature list, in the order the scorer must supply it.
        Use the return value to select columns -- ``df.select(*returned)`` --
        so ordering comes from the manifest rather than the caller.

    Raises:
        FeatureContractError: if any required feature is absent from ``columns``.
    """
    manifest = spark_manifest() if manifest is None else manifest
    features = list(manifest["features"])
    available = set(columns)
    missing = [f for f in features if f not in available]
    if missing:
        raise FeatureContractError(
            f"{len(missing)} of {len(features)} required features are missing from the "
            f"input columns: {missing}. This would NOT have raised at predict time -- "
            "XGBoost silently accepts an under-wide DMatrix and the resulting scores are "
            "wrong, not merely degraded. Emit every feature in "
            "manifest['features'] (contract_version "
            f"{manifest.get('contract_version', 'unknown')}), or pin an older crosswalk-py."
        )
    return features
