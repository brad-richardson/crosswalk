"""LOO-by-type CV regression test - gate on cross-dataset generalization.

The standard training gate (test_training.py) only measures within-distribution
(segment-split) generalization. This test runs the full leave-one-out-by-type
cross-validation (the only cross-dataset / spatial holdout eval) and asserts a
macro-F1 floor per type group, so real cross-dataset regressions fail CI.

The full LOO run is one fold per eligible dataset (33 as of 2026-08-07, since
#474 made this a true leave-one-out), so we gate on the full run rather than a
subset. It takes ~60s locally with ``-n 0``; note that the repo's default
``addopts = "-n auto"`` makes every xdist worker re-run the module-scoped
fixture, which is why this file is much slower under the default invocation.
"""

import os
from pathlib import Path

import pytest

from crosswalk.eval_utils import run_loo_by_type_cv

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true",
    reason="LOO CV regression test only runs on CI",
)

# Per-type-group macro-F1 floors.
#
# Baseline re-measured 2026-08-07 (seed=42, quality_threshold=0.5,
# DEFAULT_XGB_PARAMS) after two changes that both moved these numbers: the
# harness became a true leave-one-out (#474, 33 folds instead of 5 co-holdout
# rounds) and every stored label feature was recomputed against current raw
# data (the re-key + global backfill).
#
#   group      pre-backfill  post-backfill  delta
#   road_good      0.8878        0.8832     -0.0046
#   road_poor      0.9266        0.9244     -0.0022
#   sidewalk       0.8714        0.8698     -0.0016
#   other          0.8957        0.8891     -0.0066
#
# The uniform small drop is expected: some of the previously measured
# performance rested on features computed against raw data that had since been
# re-fetched. Corrected inputs score slightly lower and are the honest number.
#
# Floors were originally (baseline - 0.05) rounded DOWN to 2 decimals, and are
# deliberately left UNCHANGED here so the gate keeps measuring against the
# pre-backfill bar rather than being re-fitted to whatever we just produced.
#
# CAVEAT on `other` -- read this before "fixing" a failure in that group.
# Its margin is now only +0.0011 (0.8891 vs 0.88), and its across-seed spread
# widened from 0.0026 to 0.0168; at seed 7 it scores 0.8787, below the floor.
# This test is seeded (SEED=42) and therefore deterministic, so it does not
# flake -- but the next data change will likely trip `other` for reasons that
# have nothing to do with that change. Per-dataset attribution (2026-08-07):
#
#   dataset                    n    pre s42  post s42   seed swing (s7-s42)
#   ch_geneva_hiking_routes    50   0.6780    0.6667    0.0000 -> -0.0226
#   co_bogota_bike_network     29   0.9825    0.9643    0.0000 -> -0.0188
#   us_boston_bike_network     86   0.9487    0.9620    0.0000 ->  0.0000
#   us_frisco_trails          177   0.9735    0.9634   -0.0101 ->  0.0000
#
# `other` is a macro-average over 4 datasets, so ch_geneva_hiking_routes at
# ~0.67 (0.30 below the rest, and already that low BEFORE the backfill) is why
# the group sits near its floor at all; co_bogota_bike_network's 29 labels are
# why it is seed-sensitive. Neither is a code defect -- the fixes are more
# labels for bogota bike and a data-quality look at geneva hiking. Note the
# re-key is exonerated here: bogota bike had 100% of its labels geometrically
# re-keyed and still scores 0.96, which a mis-key would have destroyed.
MIN_GROUP_MACRO_F1 = {
    "road_good": 0.85,
    "road_poor": 0.87,
    "sidewalk": 0.82,
    "other": 0.88,
}

# Type groups whose evaluated labels total fewer than this are skipped
# rather than gated - the macro-F1 of a tiny group is too noisy to gate on.
MIN_GROUP_LABELS = 100

CV_FOLDS = 5
SEED = 42


@pytest.fixture(scope="module")
def loo_group_metrics():
    """Run the full LOO-by-type CV once per module and return group metrics."""
    labels_dir = Path(__file__).parent.parent.parent / "labels"
    result = run_loo_by_type_cv(
        labels=labels_dir,
        cv_folds=CV_FOLDS,
        seed=SEED,
    )
    assert result.rows, "LOO-by-type CV produced no results - check labels/ and type groups"
    return result.group_metrics()


class TestLooCvRegression:
    """Regression gate for cross-dataset (LOO-by-type) generalization."""

    @pytest.mark.parametrize("group", sorted(MIN_GROUP_MACRO_F1))
    def test_group_macro_f1_meets_floor(self, group, loo_group_metrics):
        """Each type group's macro-F1 must stay above its baseline floor."""
        if group not in loo_group_metrics:
            pytest.skip(
                f"Type group '{group}' produced no LOO evaluations "
                "(group composition may have changed)"
            )

        metrics = loo_group_metrics[group]
        if metrics["n_labels"] < MIN_GROUP_LABELS:
            pytest.skip(
                f"Type group '{group}' has only {metrics['n_labels']} evaluated labels "
                f"(< {MIN_GROUP_LABELS}) - macro-F1 too noisy to gate on"
            )

        floor = MIN_GROUP_MACRO_F1[group]
        assert metrics["f1_mean"] >= floor, (
            f"LOO macro-F1 for '{group}' is {metrics['f1_mean']:.4f}, below floor {floor} "
            f"({metrics['n_evals']} datasets, {metrics['n_labels']} labels: "
            f"{', '.join(metrics['datasets'])}). "
            "Cross-dataset generalization regressed - if this drop is expected "
            "(e.g. new hard datasets were labeled), re-measure the baseline and "
            "update MIN_GROUP_MACRO_F1."
        )
