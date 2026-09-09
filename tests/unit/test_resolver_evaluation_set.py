"""Evaluation truth and weak-data exclusion must survive drift and overlap."""

import json

import pandas as pd
import pytest

from crosswalk.resolver.evaluation_set import (
    audit_existing_labels,
    exclude_frozen_scopes,
    freeze_partitions,
    load_frozen_evaluation,
    observation_tokens,
)


def group(gid="g"):
    edges = [
        {"ref_id": r, "target_id": "T", "confidence": 0.9, "selected": True} for r in ["A", "B"]
    ]
    return {
        "group_id": gid,
        "ref_ids": ["A", "B"],
        "target_ids": ["T"],
        "edges": edges,
        "candidate_edges": edges,
    }


def label(gid="g", selected=("A",), **updates):
    return {
        "group_id": gid,
        "labeler": "brad",
        "label_semantics": "pair",
        "selected_edges": json.dumps([{"ref_id": r, "target_id": "T"} for r in selected]),
        "session_id": "deanchored_v1",
        **updates,
    }


def test_exact_human_keep_and_reject_all_both_survive():
    for selected in [("A",), ()]:
        table, audit = audit_existing_labels(
            "ds", [group()], pd.DataFrame([label(selected=selected)])
        )
        assert len(table) == 2
        assert table["keep"].sum() == len(selected)
        assert audit[0]["status"] == "eligible"
        assert not table["anchored"].any()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"label_semantics": "set"}, "membership_not_exact_edges"),
        ({"group_id": "old"}, "changed_group_without_original_scope"),
        ({"selected": ("A", "LOST")}, "split_or_missing_selected_edges"),
        ({"notes": "partial_identity"}, "partial_identity_not_resolution_truth"),
    ],
)
def test_ineligible_human_records_have_reasons_and_blocking_scope(change, reason):
    table, audit = audit_existing_labels("ds", [group()], pd.DataFrame([label(**change)]))
    assert table.empty
    assert audit[0]["status"] == reason
    assert "ref:A" in audit[0]["scope_tokens"]


def test_complete_original_membership_allows_clean_drift_mapping():
    record = label("old", ref_ids='["A", "B"]', target_ids='["T"]')
    table, audit = audit_existing_labels("ds", [group()], pd.DataFrame([record]))
    assert len(table) == 2
    assert audit[0]["status"] == "eligible"


def test_conflicting_human_labels_quarantine_whole_group():
    records = [label(), label("old", selected=("B",), ref_ids='["A", "B"]', target_ids='["T"]')]
    table, audit = audit_existing_labels("ds", [group()], pd.DataFrame(records))
    assert table.empty
    assert {r["status"] for r in audit} == {"contradictory_human_labels"}


def test_disputed_human_label_is_not_replaced_with_agent_truth():
    table, audit = audit_existing_labels(
        "ds", [group()], pd.DataFrame([label()]), exclusions={("ds", "g"): "disputed"}
    )
    assert table.empty
    assert audit[0]["status"].startswith("disputed:")


def row(gid, ref, target="T", dataset="ds", **updates):
    return {"dataset_id": dataset, "group_id": gid, "ref_id": ref, "target_id": target, **updates}


def test_unknown_child_transitively_blocks_siblings_and_shared_global_refs():
    gold = pd.DataFrame([row("human", "A"), row("independent", "Z", "Q")])
    weak = pd.DataFrame(
        [
            row("parent__p0000000001", "A", "U", known=False),
            row("parent__p0000000002", "B", "V", known=True),
            row("other", "B", "different-local-id", dataset="ds2", known=True),
            row("safe", "C", "T", dataset="ds2", known=True),
        ]
    )
    audit = [{"scope_tokens": sorted(observation_tokens(r))} for r in gold.to_dict("records")]
    frozen, partition, index = freeze_partitions(gold, weak, audit, n_folds=2)
    assert partition["excluded_human_scope"].tolist() == [True, True, True, False]
    assert frozen["fold"].nunique() == 2
    # Shuffling rows cannot change frozen components or folds.
    again, _, _ = freeze_partitions(gold.iloc[::-1], weak.iloc[::-1], audit, n_folds=2)
    assert (
        frozen.set_index("group_id")["fold"].to_dict()
        == again.set_index("group_id")["fold"].to_dict()
    )
    # A new wave connects previously independent siblings to a frozen human
    # component; propagate the exclusion without needing user adjudication.
    new = pd.DataFrame([row("new", "B", "NEW"), row("new", "FRESH", "ELSE")])
    assert exclude_frozen_scopes(new, index, set(frozen.component_id)).all()


def test_shared_segment_human_groups_cannot_cross_folds():
    gold = pd.DataFrame([row("h1", "A"), row("h2", "B"), row("h3", "Z", "Q")])
    weak = pd.DataFrame([row("unused", "C", "U")])
    audit = [{"scope_tokens": sorted(observation_tokens(r))} for r in gold.to_dict("records")]
    frozen, _, _ = freeze_partitions(gold, weak, audit, n_folds=2)
    assert frozen.iloc[0]["fold"] == frozen.iloc[1]["fold"]
    assert frozen.iloc[0]["fold"] != frozen.iloc[2]["fold"]


def test_changed_frozen_output_cannot_be_used_for_training(tmp_path):
    from crosswalk.agent_labeling.stitch_provenance import sha256_file

    outputs = {}
    for name in ["gold.parquet", "weak_partition.parquet", "audit.json", "components.json"]:
        (tmp_path / name).write_text("original")
        outputs[name] = sha256_file(tmp_path / name)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "outputs": outputs}))
    (tmp_path / "gold.parquet").write_text("changed truth")
    with pytest.raises(ValueError, match="integrity failure: gold.parquet"):
        load_frozen_evaluation(tmp_path)
