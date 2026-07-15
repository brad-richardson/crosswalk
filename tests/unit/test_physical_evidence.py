from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from shapely import LineString

from crosswalk.matching.types import MatchDecision, MatchResult
from crosswalk.pipeline.runner import _export_groups_sidecar
from crosswalk.utils.physical import (
    clip_physical_attributes,
    physical_attributes,
    physical_is_informative,
    summarize_physical,
)


def test_physical_rules_clip_to_the_aligned_interval() -> None:
    physical = physical_attributes(
        [
            {"between": [0.0, 0.5], "value": 0},
            {"between": [0.5, 1.0], "value": 1},
        ],
        [
            {"between": [0.0, 0.5], "value": []},
            {"between": [0.5, 1.0], "value": ["is_bridge"]},
        ],
    )

    clipped = clip_physical_attributes(physical, 0.9, 0.6)

    assert clipped["aligned_range"] == [0.6, 0.9]
    assert clipped["level_lr"] == [{"between": [0.6, 0.9], "value": 1}]
    assert clipped["road_flags_lr"] == [
        {"between": [0.6, 0.9], "value": ["is_bridge"]}
    ]
    assert summarize_physical(clipped) == "layer 1; bridge"
    assert physical_is_informative(clipped) is True


def test_missing_physical_metadata_stays_unknown() -> None:
    assert physical_attributes(None, None) == {}
    assert physical_attributes(
        [{"between": [0.0, 1.0], "value": None}],
        [{"between": [0.0, 1.0], "value": None}],
    ) == {}
    assert summarize_physical({}) == ""
    assert physical_is_informative(
        physical_attributes(
            [{"between": [0.0, 1.0], "value": 0}],
            [{"between": [0.0, 1.0], "value": []}],
        )
    ) is False


def test_physical_summary_uses_aligned_range_for_partial_flags() -> None:
    assert summarize_physical(
        {
            "aligned_range": [0.0, 1.0],
            "road_flags_lr": [
                {"between": [0.5, 1.0], "value": ["is_bridge"]}
            ],
        }
    ) == "bridge (partial)"
    assert summarize_physical(
        {"road_flags_lr": [{"between": [0.0, 1.0], "value": []}]}
    ) == ""


def _mr(ref: str, target: str, target_idx: int, ref_range: tuple[float, float]) -> MatchResult:
    return MatchResult(
        ref_id=ref,
        target_id=target,
        decision=MatchDecision.MATCH,
        confidence=0.95,
        score_breakdown={},
        features={"group_id": "physical-group"},
        ref_idx=0,
        target_idx=target_idx,
        gers_start_frac=ref_range[0],
        gers_end_frac=ref_range[1],
        local_start_frac=0.0,
        local_end_frac=1.0,
    )


def test_groups_sidecar_preserves_segment_rules_and_clips_edge_evidence(tmp_path) -> None:
    ref = gpd.GeoDataFrame(
        {"id": ["r1"]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    ref["level_lr"] = pd.Series(
        [[
            {"between": [0.0, 0.5], "value": 0},
            {"between": [0.5, 1.0], "value": 1},
        ]],
        dtype=object,
    )
    ref["road_flags_lr"] = pd.Series(
        [[
            {"between": [0.0, 0.5], "value": []},
            {"between": [0.5, 1.0], "value": ["is_bridge"]},
        ]],
        dtype=object,
    )
    target = gpd.GeoDataFrame(
        {"id": ["t1", "t2"]},
        geometry=[
            LineString([(60, 1), (90, 1)]),
            LineString([(10, 1), (40, 1)]),
        ],
        crs="EPSG:3857",
    )
    target["level_lr"] = pd.Series(
        [
            [{"between": [0.0, 1.0], "value": 0}],
            [{"between": [0.0, 1.0], "value": 0}],
        ],
        dtype=object,
    )
    target["road_flags_lr"] = pd.Series(
        [
            [{"between": [0.0, 1.0], "value": []}],
            [{"between": [0.0, 1.0], "value": []}],
        ],
        dtype=object,
    )

    results = [_mr("r1", "t1", 0, (0.6, 0.9)), _mr("r1", "t2", 1, (0.1, 0.4))]
    path = _export_groups_sidecar(
        results=results,
        optimized=results,
        output_path=tmp_path / "bridge.parquet",
        reference=ref,
        target=target,
        min_confidence=0.5,
        reference_proj=ref,
        target_proj=target,
    )

    group = json.loads(path.read_text())["groups"][0]
    assert group["ref_physical"]["r1"]["level_lr"] == [
        {"between": [0.0, 0.5], "value": 0},
        {"between": [0.5, 1.0], "value": 1},
    ]
    elevated = next(edge for edge in group["edges"] if edge["target_id"] == "t1")
    assert elevated["ref_physical"]["aligned_range"] == [0.6, 0.9]
    assert elevated["ref_physical"]["level_lr"] == [
        {"between": [0.6, 0.9], "value": 1}
    ]
    assert elevated["ref_physical"]["road_flags_lr"][0]["value"] == ["is_bridge"]
    assert elevated["target_physical"]["level_lr"][0]["value"] == 0
    assert "candidate_graph_bridge" in elevated
    assert "is_bridge" not in elevated
