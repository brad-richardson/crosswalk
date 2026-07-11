import pandas as pd

from crosswalk.resolver.heuristic_ablation import (
    PrunePolicy,
    apply_prune_policy,
    evaluate_policies,
    reconstruct_preprune_selection,
)


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "d",
                "group_id": "g1",
                "human_group_id": "h1",
                "ref_id": "r1",
                "target_id": "t1",
                "match_type": "M:N",
                "confidence": 0.99,
                "selected": True,
                "pruned": False,
                "keep": 1,
                "provenance": "clean",
                "is_bridge": False,
                "degree_ref": 2,
                "degree_tgt": 2,
            },
            {
                "dataset_id": "d",
                "group_id": "g1",
                "human_group_id": "h1",
                "ref_id": "r2",
                "target_id": "t2",
                "match_type": "M:N",
                "confidence": 0.95,
                "selected": False,
                "pruned": True,
                "keep": 1,
                "provenance": "clean",
                "is_bridge": True,
                "degree_ref": 1,
                "degree_tgt": 1,
            },
            {
                "dataset_id": "d",
                "group_id": "g1",
                "human_group_id": "h1",
                "ref_id": "r3",
                "target_id": "t3",
                "match_type": "M:N",
                "confidence": 0.90,
                "selected": True,
                "pruned": False,
                "keep": 0,
                "provenance": "clean",
                "is_bridge": False,
                "degree_ref": 1,
                "degree_tgt": 1,
            },
        ]
    )


def test_reconstruct_preprune_unions_selected_and_pruned():
    assert reconstruct_preprune_selection(_table()).tolist() == [True, True, True]


def test_threshold_margin_and_bridge_guard_are_independent():
    table = _table()
    assert apply_prune_policy(table, PrunePolicy("threshold", threshold=0.96)).tolist() == [
        True,
        False,
        False,
    ]
    assert apply_prune_policy(table, PrunePolicy("margin", margin=0.05)).tolist() == [
        True,
        True,
        False,
    ]
    guarded = PrunePolicy("guarded", threshold=0.96, preserve_bridge_backbone=True)
    assert apply_prune_policy(table, guarded).tolist() == [True, True, False]


def test_policy_never_empties_group_and_does_not_touch_one_to_one():
    table = _table()
    table.loc[:, "confidence"] = [0.2, 0.1, 0.05]
    assert apply_prune_policy(table, PrunePolicy("strict", threshold=1.0)).tolist() == [
        True,
        False,
        False,
    ]
    table.loc[:, "match_type"] = "1:1"
    assert apply_prune_policy(table, PrunePolicy("strict", threshold=1.0)).all()


def test_duplicate_recovered_rows_receive_same_edge_decision():
    table = _table()
    duplicate = table.iloc[[1]].copy()
    duplicate["human_group_id"] = "h2"
    combined = pd.concat([table, duplicate], ignore_index=True)
    prediction = apply_prune_policy(combined, PrunePolicy("strict", threshold=0.96))
    assert prediction.iloc[1] == prediction.iloc[3]


def test_evaluation_uses_dataset_and_human_group_for_exact_match():
    table = _table()
    other = table.copy()
    other["dataset_id"] = "other"
    other["keep"] = other["selected"].astype(int)
    combined = pd.concat([table, other], ignore_index=True)
    result = evaluate_policies(combined, [PrunePolicy("preprune")])
    current = result[(result["policy"] == "optimizer_current") & (result["slice"] == "all")]
    assert current.iloc[0]["groups"] == 2
