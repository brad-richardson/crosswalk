"""Real Boston fixtures for corridor splitting and alignment rescue.

These are evidence fixtures, not curated truth labels. Each assertion encodes a
narrow optimizer invariant supported by the persisted geometry, alignment, and
name provenance recorded in the fixture. Label coverage is stated explicitly so
future audits can distinguish a logic regression from new adjudication.
"""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import shape

from crosswalk.matching.optimizer import (
    DECOMPOSED_SINGLETON_REVIEW_FLAG,
    apply_confidence_drop_prune,
    optimize_matches_with_grouping,
)
from crosswalk.matching.types import MatchDecision, MatchResult

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "optimizer_boston_corridor_rescue.json"
WILLOW_FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "optimizer_boston_willow_decomposition.json"
)


@pytest.fixture(scope="module")
def boston_corridor_cases() -> dict[str, dict]:
    payload = json.loads(FIXTURE_PATH.read_text())
    return {case["id"]: case for case in payload["cases"]}


def _frame(records: list[dict], id_column: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            id_column: [record["id"] for record in records],
            "names": [record["names"] for record in records],
            "geometry": [shape(record["geometry"]) for record in records],
        },
        crs="EPSG:4326",
    ).to_crs("EPSG:32619")


@pytest.mark.parametrize("case_id", ["bowman", "clifton", "metropolitan", "lanark"])
def test_complementary_named_spans_survive_corridor_split(
    boston_corridor_cases: dict[str, dict],
    case_id: str,
) -> None:
    case = boston_corridor_cases[case_id]
    reference = _frame(case["references"], "id")
    target = _frame(case["targets"], "local_id")
    candidates = [
        MatchResult(
            edge["ref_id"],
            edge["target_id"],
            MatchDecision.MATCH,
            edge["confidence"],
            {},
            {},
            gers_start_frac=edge["gers_start_frac"],
            gers_end_frac=edge["gers_end_frac"],
            local_start_frac=edge["local_start_frac"],
            local_end_frac=edge["local_end_frac"],
        )
        for edge in case["edges"]
    ]

    optimized_before_prune = optimize_matches_with_grouping(
        candidates,
        reference,
        target,
        min_confidence=0.1,
        contiguity_tolerance=5.0,
        corridor_aware=True,
    )
    optimized, pruned_pairs = apply_confidence_drop_prune(optimized_before_prune, 0.96)

    expected_selected = {
        (edge["ref_id"], edge["target_id"])
        for edge in case["edges"]
        if edge.get("expected_selected", True)
    }
    assert {(result.ref_id, result.target_id) for result in optimized} == expected_selected

    before_prune_pairs = {(result.ref_id, result.target_id) for result in optimized_before_prune}
    for edge in case["edges"]:
        pair = (edge["ref_id"], edge["target_id"])
        disposition = edge.get("expected_disposition")
        if disposition == "rejected":
            assert pair not in before_prune_pairs
            assert pair not in pruned_pairs
        elif disposition == "pruned":
            assert pair in before_prune_pairs
            assert pair in pruned_pairs

    assert {result.features["match_type"] for result in optimized} == {case["expected_match_type"]}
    assert len({result.features["group_id"] for result in optimized}) == 1
    assert all(result.decision == MatchDecision.MATCH for result in optimized)


def test_real_willow_singleton_keeps_contested_review_provenance() -> None:
    """Below-glue competitors cannot promote the human-rejected Willow edge."""
    case = json.loads(WILLOW_FIXTURE_PATH.read_text())
    reference = _frame(case["references"], "id")
    target = _frame(case["targets"], "local_id")
    candidates = [
        MatchResult(
            edge["ref_id"],
            edge["target_id"],
            MatchDecision.MATCH if edge["confidence"] >= 0.5 else MatchDecision.REVIEW,
            edge["confidence"],
            {},
            {},
            gers_start_frac=edge["gers_start_frac"],
            gers_end_frac=edge["gers_end_frac"],
            local_start_frac=edge["local_start_frac"],
            local_end_frac=edge["local_end_frac"],
        )
        for edge in case["edges"]
    ]

    optimized = optimize_matches_with_grouping(
        candidates,
        reference,
        target,
        min_confidence=0.1,
        glue_min_confidence=0.575,
        contiguity_tolerance=5.0,
        corridor_aware=True,
    )

    assert [(result.ref_id, result.target_id) for result in optimized] == [
        tuple(case["selected_pair"])
    ]
    assert optimized[0].decision == MatchDecision.REVIEW
    assert optimized[0].features[DECOMPOSED_SINGLETON_REVIEW_FLAG] == 1.0
