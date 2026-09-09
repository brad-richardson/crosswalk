"""Direct ballots preserve partial uncertainty and separate identity from resolution."""

import json

import pandas as pd
import pytest

from crosswalk.agent_labeling.stitch_edge_ballots import (
    EDGE_BALLOT_VERSION,
    build_direct_evidence,
    build_direct_prompt,
    panel_configuration,
    parse_edge_ballot,
    validate_edge_decisions,
)
from crosswalk.agent_labeling.stitch_provenance import sha256_json
from crosswalk.resolver.observations import build_vote_observations


def decision(ref="A", identity="same", resolution="keep"):
    return {
        "ref_id": ref,
        "target_id": "T",
        "identity": identity,
        "resolution": resolution,
        "confidence": 0.8,
        "reason": "Aligned same physical way",
    }


def evidence():
    body = {
        "group_id": "g",
        "protocol_version": EDGE_BALLOT_VERSION,
        "displayed_edges": [{"ref_id": r, "target_id": "T"} for r in ["A", "B"]],
        "label_maps": {"reference": {"R1": "A", "R2": "B"}, "target": {"T1": "T"}},
    }
    return {**body, "evidence_id": sha256_json(body)}


def test_identity_compatible_assignment_drop_is_not_an_identity_negative():
    decisions = [decision(), decision("B", resolution="drop")]
    assert validate_edge_decisions(decisions, {("A", "T"), ("B", "T")}, "") == {
        ("A", "T"): 1,
        ("B", "T"): 0,
    }
    assert decisions[1]["identity"] == "same"


@pytest.mark.parametrize(
    "corruption", ["omit", "duplicate", "outside", "identity", "reason", "confidence"]
)
def test_invalid_ballot_cannot_infer_a_complement(corruption):
    decisions = [decision(), decision("B", identity="unknown", resolution="unknown")]
    reason = "insufficient_evidence"
    if corruption == "omit":
        decisions.pop()
    elif corruption == "duplicate":
        decisions.append(decisions[0])
    elif corruption == "outside":
        decisions[0]["ref_id"] = "UNSEEN"
    elif corruption == "identity":
        decisions[0]["identity"] = "different"
    elif corruption == "reason":
        reason = "no_exact_option"
    elif corruption == "confidence":
        decisions[0]["confidence"] = float("nan")
    with pytest.raises(ValueError):
        validate_edge_decisions(decisions, {("A", "T"), ("B", "T")}, reason)


def test_explicit_reject_all_differs_from_insufficient_evidence():
    decisions = [decision(r, identity="different", resolution="drop") for r in ["A", "B"]]
    assert (
        sum(
            validate_edge_decisions(
                decisions, {("A", "T"), ("B", "T")}, "all_edges_no_match"
            ).values()
        )
        == 0
    )
    with pytest.raises(ValueError, match="NONE reason"):
        validate_edge_decisions(decisions, {("A", "T"), ("B", "T")}, "insufficient_evidence")


def test_parser_binds_short_labels_to_exact_evidence():
    ev = evidence()
    ballot = {
        "protocol_version": EDGE_BALLOT_VERSION,
        "evidence_id": ev["evidence_id"],
        "none_reason": "",
        "reasoning": "Both pairs match",
        "edge_decisions": [{**decision(r), "target_id": "T1"} for r in ["R1", "R2"]],
    }
    parsed = parse_edge_ballot(json.dumps(ballot), ev)
    assert [e["ref_id"] for e in parsed["edge_decisions"]] == ["A", "B"]
    ballot["evidence_id"] = "stale"
    with pytest.raises(ValueError, match="different evidence"):
        parse_edge_ballot(json.dumps(ballot), ev)


def test_partial_direct_ballots_aggregate_per_edge_without_abstention_votes():
    ev = evidence()
    base = {
        "dataset_id": "ds",
        "group_id": "g",
        "evidence_id": ev["evidence_id"],
        "evidence_pack_sha256": "pack",
        "panel_invocation_sha256": "panel",
        "protocol_version": EDGE_BALLOT_VERSION,
        "none_reason": "",
        "timestamp": "now",
    }
    ballots = []
    for i, provider in enumerate(["fable", "astra", "muse"]):
        decisions = [decision(), decision("B", resolution="drop" if i < 2 else "unknown")]
        ballots.append(
            {
                **base,
                "provider": provider,
                "model": f"{provider}-pinned",
                "edge_decisions": json.dumps(decisions),
                "none_reason": "" if i < 2 else "insufficient_evidence",
            }
        )
    result = build_vote_observations(
        pd.DataFrame(ballots), pd.DataFrame([{**base, "evidence": json.dumps(ev)}])
    ).set_index("ref_id")
    assert result.loc["A", "vote_tier"] == "unanimous"
    assert result.loc["B", "vote_tier"] == "below_quorum"
    assert result.loc["B", "n_providers"] == 2
    # A legacy-looking ballot against direct evidence cannot bypass validation.
    for ballot in ballots:
        ballot["protocol_version"] = ""
    invalid = build_vote_observations(
        pd.DataFrame(ballots), pd.DataFrame([{**base, "evidence": json.dumps(ev)}])
    )
    assert not invalid["known"].any()


def test_direct_pack_has_no_menu_or_optimizer_selection_and_pins_new_models():
    ev = evidence()
    metadata = {
        "segments": {
            "reference": [{"label": "R1", "id": "A"}, {"label": "R2", "id": "B"}],
            "target": [{"label": "T1", "id": "T"}],
        },
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edges": [
                    {
                        "ref_id": "A",
                        "target_id": "T",
                        "selected": True,
                        "confidence": 0.9,
                        "decision": "match",
                        "overlap_m": 20,
                    }
                ],
            }
        ],
    }
    manifest = {
        "evidence_pack_sha256": "original",
        "evidence": {**ev, "source_candidate_edges": ev["displayed_edges"]},
    }
    direct = build_direct_evidence(manifest, metadata)
    prompt = build_direct_prompt(direct, ["overview.png"])
    assert '"is_optimizer"' not in prompt and '"selected"' not in prompt
    assert ["R2", "T1"] in json.loads(prompt.split("\n\n")[-1])["displayed_pairs"]
    assert direct["evidence_id"] != ev["evidence_id"]
    config = panel_configuration({})
    assert config["seat_weights"] == {"fable": 1.0, "astra": 1.0, "muse": 1.0}
    assert config["panel"][1]["model"] == "gpt-6-astra"
    assert sha256_json(config) != sha256_json(panel_configuration({"changed": True}))
