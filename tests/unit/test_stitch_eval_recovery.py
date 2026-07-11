import json

import pandas as pd

from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups


def _pair_label(group_id: str, edges: list[tuple[str, str]]) -> dict:
    return {
        "group_id": group_id,
        "label_semantics": "pair",
        "selected_edges": json.dumps(
            [{"ref_id": ref_id, "target_id": target_id} for ref_id, target_id in edges]
        ),
        "ref_ids": "",
        "target_ids": "",
    }


def test_pair_label_recovers_from_fresh_candidate_graph():
    groups = [
        {
            "group_id": "fresh",
            "edges": [],
            "candidate_edges": [
                {"ref_id": "r1", "target_id": "t1", "selected": False},
            ],
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
        }
    ]
    labels = pd.DataFrame([_pair_label("historical", [("r1", "t1")])])

    recovered = recover_labeled_groups(groups, labels)

    assert recovered["clean"] == [("historical", "fresh")]
    assert recovered["lost"] == []


def test_pair_label_recovers_from_rejected_edge():
    groups = [
        {
            "group_id": "fresh",
            "edges": [],
            "rejected_edges": [
                {"ref_id": "r2", "target_id": "t2", "selected": False},
                {"ref_id": "malformed"},
                None,
            ],
        }
    ]
    labels = pd.DataFrame([_pair_label("historical", [("r2", "t2")])])

    recovered = recover_labeled_groups(groups, labels)

    assert recovered["clean"] == [("historical", "fresh")]


def test_set_label_recovers_from_explicit_group_membership_without_edges():
    groups = [
        {
            "group_id": "fresh",
            "edges": [],
            "candidate_edges": [],
            "ref_ids": ["r1", "r2"],
            "target_ids": ["t1"],
        }
    ]
    labels = pd.DataFrame(
        [
            {
                "group_id": "historical-set",
                "label_semantics": "set",
                "selected_edges": "[]",
                "ref_ids": '["r1", "r2"]',
                "target_ids": '["t1"]',
            }
        ]
    )

    recovered = recover_labeled_groups(groups, labels)

    assert recovered["set"] == [("historical-set", "fresh")]
    assert recovered["set_lost"] == []
