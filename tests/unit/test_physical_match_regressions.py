from __future__ import annotations

import json
from pathlib import Path

from crosswalk.utils.physical import clip_physical_attributes, summarize_physical

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "physical_match_regressions.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def test_v7_physical_regression_fixture_has_unique_provenanced_pairs() -> None:
    fixture = _fixture()
    pairs = fixture["pair_cases"]
    keys = [
        (case["dataset_id"], case["group_id"], case["ref_id"], case["target_id"]) for case in pairs
    ]

    assert fixture["source_wave"] == "breadth_v7"
    assert len(keys) == len(set(keys))
    assert sum(case["pair_truth"] == "negative" for case in pairs) == 6
    assert all("v6" not in case["source_batch"] for case in pairs)


def test_hong_kong_known_negatives_clip_to_tunnel_vs_ground() -> None:
    cases = [case for case in _fixture()["pair_cases"] if case["group_id"] == "4eed5e80"]

    assert len(cases) == 3
    for case in cases:
        ref = clip_physical_attributes(case["ref_physical"], *case["ref_alignment"])
        target = clip_physical_attributes(case["target_physical"], *case["target_alignment"])
        assert summarize_physical(ref) == "layer -1; tunnel"
        assert summarize_physical(target) == "layer 0"
        assert case["pair_truth"] == "negative"


def test_sydney_tunnel_negative_uses_a_real_target_analog() -> None:
    case = next(
        case
        for case in _fixture()["pair_cases"]
        if case["target_id"] == "au_sydney_831394_88be0e3435"
    )

    assert case["target_analog"] == {"field": "roadontype", "raw_value": 3}
    assert summarize_physical(case["target_physical"]) == "tunnel"
    assert case["pair_truth"] == "negative"


def test_geneva_case_is_coincidence_context_not_synthetic_layer_truth() -> None:
    case = _fixture()["group_cases"][0]

    assert case["case_type"] == "same-side-coincident-alternatives"
    assert case["ambiguous_ref_in_human_set"] is False
    assert all(member["in_human_set"] for member in case["coincident_members"])
    assert all(
        member["fraction_within_3m_of_ambiguous_ref"] == 1.0
        for member in case["coincident_members"]
    )
    assert case["derived_coincidence_context"]["alternative_count"] == 2
    assert case["derived_coincidence_context"]["covered_fraction"] == 0.331
    assert case["derived_coincidence_context"]["has_role_conflict"] is True
    assert "same-side coincidence" in case["expected_use"]
