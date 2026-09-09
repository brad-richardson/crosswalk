"""Regression cases for partial, versioned weak supervision."""

import json

import numpy as np
import pandas as pd
import pytest

from crosswalk.agent_labeling.stitch_provenance import sha256_json
from crosswalk.resolver.observations import build_vote_observations, reconcile_observations


def evidence(group="parent__p0000000001", edges=(("A", "T"), ("B", "T")), **updates):
    payload = {
        "group_id": group,
        "matching_rubric_version": "rubric1",
        "displayed_edges": [{"ref_id": r, "target_id": t} for r, t in edges],
        "source_artifacts": {"groups_sidecar": {"sha256": "same-source", "path": "groups.json"}},
        **updates,
    }
    payload["evidence_id"] = sha256_json(payload)
    return {
        "dataset_id": "ds",
        "group_id": group,
        "evidence_id": payload["evidence_id"],
        "evidence_pack_sha256": "pack1",
        "evidence": json.dumps(payload),
    }


def panel(ev, selected=(("A", "T"),), **updates):
    return [
        {
            "dataset_id": ev["dataset_id"],
            "group_id": ev["group_id"],
            "provider": provider,
            "model": f"{provider}-v1",
            "evidence_id": ev["evidence_id"],
            "evidence_pack_sha256": ev["evidence_pack_sha256"],
            "panel_invocation_sha256": "panel1",
            "source_batch": "batch1",
            "timestamp": "2026-09-09T00:00:00Z",
            "choice": "A",
            "edge_set": json.dumps(selected),
            "none_reason": "",
            **updates,
        }
        for provider in ["claude", "codex", "muse"]
    ]


def build(rows, evidences, **kwargs):
    return build_vote_observations(pd.DataFrame(rows), pd.DataFrame(evidences), **kwargs)


def test_partial_child_has_no_claim_on_parent_complement():
    ev = evidence()
    result = build(panel(ev), [ev])
    assert set(result["ref_id"]) == {"A", "B"}
    assert result.set_index("ref_id")["soft_keep"].to_dict() == {"A": 1, "B": 0}
    assert set(result["parent_group_id"]) == {"parent"}
    assert set(result["vote_tier"]) == {"unanimous"}
    assert set(result["supervision_kind"]) == {"resolution"}


def test_insufficient_evidence_masks_unknowns_and_cannot_supply_third_voter():
    ev = evidence()
    votes = panel(ev)
    votes[2].update(choice="NONE", none_reason="insufficient_evidence", edge_set="[]")
    result = build(votes, [ev])
    assert set(result["n_providers"]) == {2}
    assert set(result["vote_tier"]) == {"below_quorum"}
    for vote in votes:
        vote.update(choice="NONE", none_reason="insufficient_evidence", edge_set="[]")
    result = build(votes, [ev])
    assert not result["known"].any()
    assert result["soft_keep"].isna().all()


def test_none_menu_gap_and_insufficient_evidence_are_not_agreement():
    ev = evidence()
    votes = panel(ev, selected=(("A", "T"), ("B", "T")))
    votes[0].update(
        choice="NONE", none_reason="no_exact_option", edge_set="[]", desired_edges='[["R1", "T1"]]'
    )
    votes[1].update(choice="NONE", none_reason="insufficient_evidence", edge_set="[]")
    maps = {ev["evidence_id"]: {"reference": {"R1": "A"}, "target": {"T1": "T"}}}
    result = build(votes, [ev], label_maps=maps).set_index("ref_id")
    assert result.loc["A", "n_keep"] == 2
    assert result.loc["B", "n_keep"] == result.loc["B", "n_drop"] == 1
    assert not result.loc["B", "known"]
    assert not (result["vote_tier"] == "majority").any()


def test_mixed_evidence_and_invocations_never_form_quorum():
    old = evidence()
    new = evidence(matching_rubric_version="rubric2")
    result = build(panel(old)[:2] + panel(new)[2:], [old, new])
    assert result["observation_id"].nunique() == 2
    assert result["n_providers"].max() == 2
    assert set(result["vote_tier"]) == {"below_quorum"}
    votes = panel(old)[:2] + panel(old, panel_invocation_sha256="other-config")[2:]
    assert build(votes, [old])["n_providers"].max() == 2


def test_repeated_observations_do_not_increase_quorum_or_weight():
    ev = evidence()
    first = build(panel(ev)[:2], [ev])
    second = build(panel(ev, panel_invocation_sha256="repeat")[:2], [ev])
    result = reconcile_observations(pd.concat([first, first, second], ignore_index=True))
    assert len(result) == 2
    assert set(result["n_providers"]) == {2}
    assert set(result["vote_tier"]) == {"below_quorum"}
    assert set(result["n_observations"]) == {2}


def test_overlapping_children_quarantine_conflict_and_preserve_other_edges():
    left = evidence()
    right = evidence(group="parent__p0000000002", edges=(("A", "T"), ("C", "T")))
    observations = build(panel(left) + panel(right, selected=(("C", "T"),)), [left, right])
    result = reconcile_observations(observations).set_index("ref_id")
    assert result.loc["A", "overlap_conflict"]
    assert not result.loc["A", "known"]
    assert np.isnan(result.loc["A", "soft_keep"])
    assert result.loc["B", "soft_keep"] == 0
    assert result.loc["C", "soft_keep"] == 1
    assert len(json.loads(result.loc["A", "observation_ids"])) == 2


@pytest.mark.parametrize("corruption", ["outside", "hash", "pack", "duplicate_model"])
def test_corrupt_inputs_cannot_create_supervision(corruption):
    ev = evidence()
    votes = panel(ev)
    if corruption == "outside":
        for vote in votes:
            vote["edge_set"] = '[["UNSEEN", "T"]]'
    elif corruption == "hash":
        payload = json.loads(ev["evidence"])
        payload["matching_rubric_version"] = "tampered"
        ev["evidence"] = json.dumps(payload)
    elif corruption == "pack":
        ev["evidence_pack_sha256"] = "different-rendering"
    else:
        votes += [{**vote, "model": "different-model"} for vote in votes]
    result = build(votes, [ev])
    assert result.empty or not result["known"].any()


def test_load_votes_preserves_changed_evidence_and_model_configuration(tmp_path):
    from crosswalk.resolver.votes import load_votes

    path = tmp_path / "dataset=ds"
    path.mkdir()
    old, new = evidence(), evidence(matching_rubric_version="rubric2")
    rows = panel(old) + panel(new)
    rows.append({**rows[0], "timestamp": "2026-09-09T01:00:00Z"})
    pd.DataFrame(rows).to_csv(path / "votes.csv", index=False)
    loaded = load_votes([path / "votes.csv"])
    assert len(loaded) == 6
    assert loaded["evidence_id"].nunique() == 2


def test_training_masks_unknowns_after_computing_full_candidate_context():
    from crosswalk.resolver.features import FEATURE_COLUMNS
    from crosswalk.resolver.train import _prepare_soft_for_train

    group = {
        "group_id": "parent",
        "match_type": "M:N",
        "ref_ids": ["A", "B"],
        "target_ids": ["T"],
        "candidate_edges": [
            {"ref_id": "A", "target_id": "T", "confidence": 0.4, "selected": False},
            {"ref_id": "B", "target_id": "T", "confidence": 0.9, "selected": True},
        ],
    }
    soft = pd.DataFrame(
        [
            {
                "dataset_id": "ds",
                "group_id": "parent",
                "ref_id": "A",
                "target_id": "T",
                "soft_keep": 1.0,
                "known": True,
                "lineage_ids": '["lineage"]',
            },
            {
                "dataset_id": "ds",
                "group_id": "parent",
                "ref_id": "B",
                "target_id": "T",
                "soft_keep": np.nan,
                "known": False,
                "lineage_ids": '["lineage"]',
            },
        ]
    )
    result = _prepare_soft_for_train(
        soft, {"ds": [group]}, set(), FEATURE_COLUMNS, False, use_float_label=True
    )
    assert len(result) == 1
    assert result.iloc[0]["conf_rel_max"] == pytest.approx(-0.5)
    assert result.iloc[0]["n_share_tgt"] == 2
    assert result.iloc[0]["lineage_ids"] == '["lineage"]'
    assert result.iloc[0]["keep"] == 1
