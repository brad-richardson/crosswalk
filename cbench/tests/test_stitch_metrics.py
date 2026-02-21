"""Tests for stitch-level evaluation metrics."""

import json

import pandas as pd
import pytest

from cbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups


@pytest.fixture
def bridge_df():
    """Bridge with edges for two groups."""
    return pd.DataFrame(
        {
            "ref_id": ["r1", "r1", "r2", "r3", "r3", "r4"],
            "target_id": ["t1", "t2", "t1", "t3", "t4", "t5"],
            "confidence": [0.9, 0.8, 0.7, 0.95, 0.6, 0.85],
        }
    )


@pytest.fixture
def stitch_labels():
    """Two labeled groups."""
    return pd.DataFrame(
        {
            "group_id": ["aaa", "bbb"],
            "dataset_id": ["test", "test"],
            "selected_edges": [
                json.dumps(
                    [
                        {"ref_id": "r1", "target_id": "t1"},
                        {"ref_id": "r2", "target_id": "t1"},
                    ]
                ),
                json.dumps(
                    [
                        {"ref_id": "r3", "target_id": "t3"},
                    ]
                ),
            ],
            "match_type": ["N:1", "1:1"],
            "num_refs": [2, 1],
            "num_targets": [1, 1],
        }
    )


def test_evaluate_stitch_groups_basic(bridge_df, stitch_labels):
    """Curated edges found in bridge, extra edges detected."""
    result = evaluate_stitch_groups(bridge_df, stitch_labels)

    assert result.groups_evaluated == 2
    # Group aaa: curated {r1-t1, r2-t1}, bridge has {r1-t1, r1-t2, r2-t1}
    #   recall = 2/2 = 1.0, precision = 2/3 = 0.667
    # Group bbb: curated {r3-t3}, bridge has {r3-t3, r3-t4}
    #   recall = 1/1 = 1.0, precision = 1/2 = 0.5
    assert result.recall == pytest.approx(1.0)  # macro avg
    assert result.precision == pytest.approx((2 / 3 + 0.5) / 2, abs=0.01)
    assert result.total_curated_edges == 3
    assert result.total_extra_edges == 2  # r1-t2 and r3-t4


def test_evaluate_stitch_groups_missing_edges(bridge_df, stitch_labels):
    """When bridge is missing curated edges, recall drops."""
    # Remove r2-t1 from bridge
    bridge_missing = bridge_df[
        ~((bridge_df["ref_id"] == "r2") & (bridge_df["target_id"] == "t1"))
    ]
    result = evaluate_stitch_groups(bridge_missing, stitch_labels)

    # Group aaa: curated {r1-t1, r2-t1}, bridge has {r1-t1, r1-t2}
    #   recall = 1/2 = 0.5, precision = 1/2 = 0.5
    assert result.recall == pytest.approx(0.75)  # macro avg (0.5 + 1.0) / 2


def test_evaluate_stitch_groups_empty_labels():
    """No stitch labels -> zero metrics."""
    bridge = pd.DataFrame(
        {"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]}
    )
    labels = pd.DataFrame(
        columns=[
            "group_id",
            "dataset_id",
            "selected_edges",
            "match_type",
            "num_refs",
            "num_targets",
        ]
    )
    result = evaluate_stitch_groups(bridge, labels)
    assert result.groups_evaluated == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_stitch_eval_result_to_dict():
    r = StitchEvalResult(
        groups_evaluated=5,
        precision=0.8,
        recall=0.9,
        f1=0.847,
        total_curated_edges=10,
        total_extra_edges=2,
    )
    d = r.to_dict()
    assert d["groups_evaluated"] == 5
    assert d["stitch_f1"] == 0.847
