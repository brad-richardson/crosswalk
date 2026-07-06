"""Tests for the zero-cost Spark accessor module (``crosswalk.spark``).

Covers:
- calibration parity with the in-process ``IsotonicCalibrator``,
- import hygiene: ``import crosswalk.spark`` + ``spark_manifest()`` must not pull
  in shapely/geopandas/xgboost/pandas (numpy allowed only after
  ``apply_calibration`` runs).
"""

import os
import subprocess
import sys
import textwrap

import numpy as np

from crosswalk.matching.calibration import IsotonicCalibrator
from crosswalk.spark import apply_calibration, spark_manifest, spark_model_json


def test_accessors_return_expected_types():
    model_json = spark_model_json()
    assert isinstance(model_json, str) and model_json.strip().startswith("{")
    manifest = spark_manifest()
    assert isinstance(manifest, dict)
    assert manifest["features"] and isinstance(manifest["features"], list)


def test_apply_calibration_matches_isotonic_calibrator():
    """apply_calibration must reproduce IsotonicCalibrator.transform bit-for-bit."""
    manifest = spark_manifest()
    knots = manifest["calibration"]
    calibrator = IsotonicCalibrator.from_knots(knots)

    rng = np.random.default_rng(42)
    # Cover the interior plus out-of-range values to exercise endpoint clipping.
    scores = np.concatenate([rng.random(1000), np.array([-0.5, -0.001, 0.0, 1.0, 1.001, 1.5])])

    expected = calibrator.transform(scores)
    actual = apply_calibration(scores, knots)

    np.testing.assert_array_equal(actual, expected)


def test_apply_calibration_accepts_list_and_returns_float64_array():
    manifest = spark_manifest()
    result = apply_calibration([0.1, 0.9], manifest["calibration"])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (2,)


def test_import_matcher_spark_stays_dependency_light():
    """`import crosswalk.spark; spark_manifest()` must not import heavy deps.

    shapely/geopandas/xgboost/pandas must be absent from sys.modules; numpy is
    absent until apply_calibration is called. Runs in a subprocess so the parent
    test process's already-imported modules don't mask a regression.
    """
    script = textwrap.dedent(
        """
        import sys

        import crosswalk.spark as ms

        heavy = ("shapely", "geopandas", "xgboost", "pandas", "sklearn")

        manifest = ms.spark_manifest()
        assert isinstance(manifest, dict) and manifest["features"]
        model_json = ms.spark_model_json()
        assert isinstance(model_json, str)

        leaked = [m for m in heavy if m in sys.modules]
        assert not leaked, f"heavy deps leaked after manifest/model read: {leaked}"
        assert "numpy" not in sys.modules, "numpy imported before apply_calibration"

        # numpy is allowed to appear only after apply_calibration is invoked.
        ms.apply_calibration([0.1, 0.5, 0.9], manifest["calibration"])
        assert "numpy" in sys.modules

        still_leaked = [m for m in heavy if m in sys.modules]
        assert not still_leaked, f"heavy deps leaked after apply_calibration: {still_leaked}"

        print("OK")
        """
    )
    # Propagate the parent's import path so the child interpreter can find the
    # (editable-installed / src-layout) matcher package.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"import-hygiene subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip().endswith("OK")
