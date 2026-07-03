"""LOO-by-type CV regression test - gate on cross-dataset generalization.

The standard training gate (test_training.py) only measures within-distribution
(segment-split) generalization. This test runs the full leave-one-out-by-type
cross-validation (the only cross-dataset / spatial holdout eval) and asserts a
macro-F1 floor per type group, so real cross-dataset regressions fail CI.

The full LOO run (5 folds, all labeled datasets) takes ~15s locally, so we gate
on the full run rather than a subset.
"""

import os
from pathlib import Path

import pytest

from matcher.eval_utils import run_loo_by_type_cv

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true",
    reason="LOO CV regression test only runs on CI",
)

# Per-type-group macro-F1 floors.
#
# Baseline measured 2026-07-02 on main (seed=42, cv_folds=5,
# quality_threshold=0.5, DEFAULT_XGB_PARAMS), repo labels/:
#   road_good: macro-F1 0.9093  (5 datasets evaluated, 777 labels)
#   road_poor: macro-F1 0.9256  (4 datasets evaluated, 565 labels)
#   sidewalk:  macro-F1 0.8714  (5 datasets evaluated, 661 labels)
#   other:     macro-F1 0.9319  (4 datasets evaluated, 342 labels)
#
# Floors are (observed baseline - 0.05), rounded DOWN to 2 decimals. The
# margin is deliberately generous: hyperparameter retuning and new labeled
# datasets shift these numbers slightly, and the gate should catch real
# cross-dataset regressions (multi-point drops), not parameter jitter.
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
