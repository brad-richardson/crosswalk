import runpy
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "ablation_study.py"))


def test_paired_cv_delta_uses_foldwise_differences():
    mean, std = SCRIPT["paired_cv_delta"]([0.8, 0.9, 1.0], [0.7, 0.9, 0.9])
    assert mean == pytest.approx(0.0666667)
    assert std == pytest.approx(0.04714045)


def test_paired_cv_delta_rejects_unpaired_scores():
    with pytest.raises(ValueError, match="identical shapes"):
        SCRIPT["paired_cv_delta"]([0.8], [0.8, 0.9])


def test_feature_coverage_distinguishes_missing_constant_and_usable():
    df = pd.DataFrame(
        {
            "dataset": ["a", "a", "b"],
            "label": ["match", "no_match", "match"],
            "usable": [0.0, 1.0, 1.0],
            "constant": [5.0, 5.0, 5.0],
            "empty": [None, None, None],
        }
    )
    rows = SCRIPT["build_feature_coverage"](
        df,
        {"family": ["usable", "constant", "empty", "absent"]},
    )
    dataset_a = next(row for row in rows if row["dataset"] == "a")
    dataset_b = next(row for row in rows if row["dataset"] == "b")
    assert dataset_a["usable_features"] == 1
    assert dataset_a["constant_features"] == "constant"
    assert dataset_a["all_missing_features"] == "empty,absent"
    assert dataset_b["usable_features"] == 0


def test_summary_classifies_and_ranks_by_grouped_cv_not_holdout():
    results = [
        {
            "experiment_type": "single_feature",
            "excluded_features": "cv_valuable",
            "f1_delta": 0.02,
            "cv_f1_delta": -0.02,
            "cv_f1_delta_std": 0.003,
            "classification": "important",
        },
        {
            "experiment_type": "single_feature",
            "excluded_features": "holdout_valuable",
            "f1_delta": -0.02,
            "cv_f1_delta": 0.01,
            "cv_f1_delta_std": 0.004,
            "classification": "noise",
        },
    ]
    baseline = {
        "accuracy": 0.9,
        "f1": 0.9,
        "cv_f1_mean": 0.9,
        "cv_f1_std": 0.01,
        "n_features_used": 2,
    }
    summary = SCRIPT["generate_summary"](results, baseline)
    assert summary["feature_ranking_by_importance"][0]["feature"] == "cv_valuable"
    assert summary["noise_candidates"] == ["holdout_valuable"]
    assert summary["recommendations"]["safe_to_remove"] == []
