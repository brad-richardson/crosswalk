import json

import pandas as pd

from crosswalk.config import FEATURE_COLUMNS
from crosswalk.labeling.stitch_pair_bridge import (
    candidate_features_for_pairs,
    derive_stitch_pair_labels,
    filter_against_human_truth,
)


def test_dual_identity_and_weak_positive_semantics():
    labels = pd.DataFrame(
        [
            {
                "group_id": "identity",
                "labeler": "brad",
                "labeled_at": "2026-07-18T00:00:00Z",
                "adjudication_scope": "exact_identity",
                "edge_dispositions": json.dumps(
                    [
                        {
                            "ref_id": "r1",
                            "target_id": "t1",
                            "resolution": "keep",
                            "identity": "match",
                        },
                        {
                            "ref_id": "r2",
                            "target_id": "t1",
                            "resolution": "drop",
                            "identity": "match",
                        },
                        {
                            "ref_id": "r3",
                            "target_id": "t1",
                            "resolution": "drop",
                            "identity": "no_match",
                        },
                        {
                            "ref_id": "r4",
                            "target_id": "t1",
                            "resolution": "drop",
                            "identity": "unsure",
                        },
                    ]
                ),
                "selected_edges": "[]",
            },
            {
                "group_id": "legacy",
                "labeler": "panel",
                "labeled_at": "2026-07-17T00:00:00Z",
                "adjudication_scope": "",
                "edge_dispositions": "",
                "selected_edges": json.dumps([{"ref_id": "r5", "target_id": "t2"}]),
            },
        ]
    )

    derived, stats = derive_stitch_pair_labels(labels, "ds")

    by_pair = derived.set_index(["gers_id", "target_id"])
    assert by_pair.loc[("r1", "t1"), "label"] == "match"
    # A contextual stitch drop remains a positive identity label.
    assert by_pair.loc[("r2", "t1"), "label"] == "match"
    assert by_pair.loc[("r2", "t1"), "resolution"] == "drop"
    assert by_pair.loc[("r3", "t1"), "label"] == "no_match"
    assert ("r4", "t1") not in by_pair.index
    assert by_pair.loc[("r5", "t2"), "confidence"] == 0.7
    assert stats["unsure_skipped"] == 1


def test_explicit_identity_overrides_weak_positive_and_conflicts_quarantine():
    labels = pd.DataFrame(
        [
            {
                "group_id": "weak",
                "selected_edges": json.dumps([{"ref_id": "r", "target_id": "t"}]),
            },
            {
                "group_id": "strong",
                "adjudication_scope": "exact_identity",
                "edge_dispositions": json.dumps(
                    [
                        {
                            "ref_id": "r",
                            "target_id": "t",
                            "resolution": "drop",
                            "identity": "no_match",
                        }
                    ]
                ),
            },
        ]
    ).fillna("")
    derived, _ = derive_stitch_pair_labels(labels, "ds")
    assert len(derived) == 1
    assert derived.iloc[0]["label"] == "no_match"

    conflict = pd.concat([labels.iloc[[1]], labels.iloc[[1]].copy()], ignore_index=True)
    dispositions = json.loads(conflict.loc[1, "edge_dispositions"])
    dispositions[0]["identity"] = "match"
    conflict.loc[1, "edge_dispositions"] = json.dumps(dispositions)
    derived, stats = derive_stitch_pair_labels(conflict, "ds")
    assert derived.empty
    assert stats["conflicting_pairs"] == 1


def test_human_truth_filter_and_candidate_feature_materialization():
    derived = pd.DataFrame(
        [
            {"gers_id": "same", "target_id": "t", "label": "match"},
            {"gers_id": "conflict", "target_id": "t", "label": "match"},
            {"gers_id": "new", "target_id": "t", "label": "no_match"},
        ]
    )
    human = pd.DataFrame(
        [
            {"gers_id": "same", "target_id": "t", "label": "match"},
            {"gers_id": "conflict", "target_id": "t", "label": "no_match"},
        ]
    )
    filtered, audit = filter_against_human_truth(derived, human)
    assert list(filtered["gers_id"]) == ["new"]
    assert audit == {"human_redundant": 1, "human_conflicts": 1}

    row = {name: float(index) for index, name in enumerate(FEATURE_COLUMNS)}
    candidates = pd.DataFrame(
        [
            {"ref_id": "new", "target_id": "t", **row},
            {"ref_id": "other", "target_id": "x", **row},
        ]
    )
    features, missing = candidate_features_for_pairs(candidates, [("new", "t"), ("missing", "t")])
    assert list(features["gers_id"]) == ["new"]
    assert missing == [("missing", "t")]
    assert set(FEATURE_COLUMNS) <= set(features.columns)
