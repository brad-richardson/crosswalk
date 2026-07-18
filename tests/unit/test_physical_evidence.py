from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from shapely import LineString

from crosswalk.agent_labeling.stitch_evidence import _edge_offset_str, build_prompt
from crosswalk.matching.types import MatchDecision, MatchResult
from crosswalk.pipeline.runner import _export_groups_sidecar
from crosswalk.utils.physical import (
    clip_physical_attributes,
    interval_union_length,
    normalize_lr_rules,
    physical_attributes,
    physical_is_informative,
    summarize_physical,
)

_PHYSICAL_GUIDANCE_MARKER = "'R physical' / 'T physical' reports bridge"


def _prompt_metadata(*, edge_physical: dict, segment_physical: dict) -> dict:
    """Minimal build_prompt metadata with configurable physical evidence."""
    return {
        "group_id": "g1",
        "match_type": "1:1",
        "n_ref_segments": 1,
        "n_target_segments": 1,
        "optimizer_letter": "A",
        "structure": {},
        "same_side_coincidence": {},
        "segments": {
            "reference": [
                {"label": "R1", "id": "r1", "name": "", "class": "", "physical": segment_physical}
            ],
            "target": [{"label": "T1", "id": "t1", "name": "", "class": "", "physical": {}}],
        },
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edge_count": 1,
                "total_confidence": 0.9,
                "mean_confidence": 0.9,
                "edges": [{"edge": "R1->T1", "confidence": 0.9, **edge_physical}],
            }
        ],
    }


def test_prompt_physical_guidance_fires_on_edge_physical_with_empty_segment_map(tmp_path) -> None:
    # Bug 1: edges carry ref_physical/target_physical (never a "structural" key),
    # so an edge-only physical pack must still print the interpretive guidance.
    metadata = _prompt_metadata(
        edge_physical={
            "ref_physical": {
                "aligned_range": [0.0, 1.0],
                "road_flags_lr": [{"between": [0.0, 1.0], "value": ["is_bridge"]}],
            },
            "target_physical": {},
        },
        segment_physical={},
    )
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert _PHYSICAL_GUIDANCE_MARKER in prompt
    # The legend still surfaces the R physical line, which the guidance explains.
    assert "R physical: bridge" in prompt


def test_prompt_physical_guidance_absent_without_any_physical(tmp_path) -> None:
    metadata = _prompt_metadata(edge_physical={}, segment_physical={})
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert _PHYSICAL_GUIDANCE_MARKER not in prompt


def test_interval_union_length_dedupes_overlaps() -> None:
    assert interval_union_length([[0.0, 0.6], [0.4, 0.9]]) == 0.9
    assert interval_union_length([[0.0, 0.5], [0.5, 1.0]]) == 1.0
    assert interval_union_length([[0.0, 0.4], [0.6, 1.0]]) == 0.8
    assert interval_union_length([]) == 0.0


def test_normalize_lr_rules_merges_overlapping_same_value() -> None:
    # Unsorted, overlapping, equal-valued rules collapse to one union range.
    merged = normalize_lr_rules(
        [
            {"between": [0.4, 0.9], "value": 1},
            {"between": [0.0, 0.6], "value": 1},
        ]
    )
    assert merged == [{"between": [0.0, 0.9], "value": 1}]


def test_normalize_lr_rules_keeps_overlapping_different_values() -> None:
    merged = normalize_lr_rules(
        [
            {"between": [0.0, 0.6], "value": 0},
            {"between": [0.4, 1.0], "value": 1},
        ]
    )
    assert merged == [
        {"between": [0.0, 0.6], "value": 0},
        {"between": [0.4, 1.0], "value": 1},
    ]


def test_summarize_physical_union_coverage_suppresses_false_partial() -> None:
    # Two overlapping same-flag rules cover the full segment as a union; a raw
    # sum would exceed 1.0, but union coverage == total, so NOT "(partial)".
    physical = {
        "road_flags_lr": [
            {"between": [0.0, 0.6], "value": ["is_bridge"]},
            {"between": [0.4, 1.0], "value": ["is_bridge"]},
        ]
    }
    assert summarize_physical(physical) == "bridge"


def test_summarize_physical_union_coverage_keeps_true_partial() -> None:
    physical = {
        "road_flags_lr": [
            {"between": [0.0, 0.3], "value": ["is_bridge"]},
            {"between": [0.2, 0.5], "value": ["is_bridge"]},
        ]
    }
    # Union = [0.0, 0.5] = 0.5 of the full segment -> still partial.
    assert summarize_physical(physical) == "bridge (partial)"


def test_summarize_physical_mid_segment_flag_is_partial() -> None:
    # Bug 3: a single [0.4, 0.6] flag with no aligned_range must read as partial
    # (denominator is the full segment 1.0, not the rule span).
    physical = {"road_flags_lr": [{"between": [0.4, 0.6], "value": ["is_tunnel"]}]}
    assert summarize_physical(physical) == "tunnel (partial)"


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
    assert clipped["road_flags_lr"] == [{"between": [0.6, 0.9], "value": ["is_bridge"]}]
    assert summarize_physical(clipped) == "layer 1; bridge"
    assert physical_is_informative(clipped) is True


def test_missing_physical_metadata_stays_unknown() -> None:
    assert physical_attributes(None, None) == {}
    assert (
        physical_attributes(
            [{"between": [0.0, 1.0], "value": None}],
            [{"between": [0.0, 1.0], "value": None}],
        )
        == {}
    )
    assert summarize_physical({}) == ""
    assert (
        physical_is_informative(
            physical_attributes(
                [{"between": [0.0, 1.0], "value": 0}],
                [{"between": [0.0, 1.0], "value": []}],
            )
        )
        is False
    )


def test_physical_summary_uses_aligned_range_for_partial_flags() -> None:
    assert (
        summarize_physical(
            {
                "aligned_range": [0.0, 1.0],
                "road_flags_lr": [{"between": [0.5, 1.0], "value": ["is_bridge"]}],
            }
        )
        == "bridge (partial)"
    )
    assert summarize_physical({"road_flags_lr": [{"between": [0.0, 1.0], "value": []}]}) == ""


def test_physical_summary_preserves_covered_and_indoor_flags() -> None:
    physical = {
        "road_flags_lr": [
            {"between": [0.0, 0.5], "value": ["is_covered"]},
            {"between": [0.5, 1.0], "value": ["is_indoor"]},
        ]
    }

    assert summarize_physical(physical) == "covered (partial); indoor (partial)"
    assert physical_is_informative(physical) is True


def _mr(
    ref: str,
    target: str,
    target_idx: int,
    ref_range: tuple[float, float],
    extra_features: dict | None = None,
) -> MatchResult:
    return MatchResult(
        ref_id=ref,
        target_id=target,
        decision=MatchDecision.MATCH,
        confidence=0.95,
        score_breakdown={},
        features={"group_id": "physical-group", **(extra_features or {})},
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
        [
            [
                {"between": [0.0, 0.5], "value": 0},
                {"between": [0.5, 1.0], "value": 1},
            ]
        ],
        dtype=object,
    )
    ref["road_flags_lr"] = pd.Series(
        [
            [
                {"between": [0.0, 0.5], "value": []},
                {"between": [0.5, 1.0], "value": ["is_bridge"]},
            ]
        ],
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
    assert elevated["ref_physical"]["level_lr"] == [{"between": [0.6, 0.9], "value": 1}]
    assert elevated["ref_physical"]["road_flags_lr"][0]["value"] == ["is_bridge"]
    assert elevated["target_physical"]["level_lr"][0]["value"] == 0
    assert "candidate_graph_bridge" in elevated
    assert "is_bridge" not in elevated


_OFFSET_LEGEND_MARKER = "'off≈Xm' is the measured lateral offset"


def _offset_prompt_metadata(edge_offset: dict) -> dict:
    """Minimal build_prompt metadata with configurable per-edge offset fields."""
    return {
        "group_id": "g1",
        "match_type": "1:1",
        "n_ref_segments": 1,
        "n_target_segments": 1,
        "optimizer_letter": "A",
        "structure": {},
        "same_side_coincidence": {},
        "segments": {
            "reference": [{"label": "R1", "id": "r1", "name": "", "class": "", "physical": {}}],
            "target": [{"label": "T1", "id": "t1", "name": "", "class": "", "physical": {}}],
        },
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edge_count": 1,
                "total_confidence": 0.9,
                "mean_confidence": 0.9,
                "edges": [{"edge": "R1->T1", "confidence": 0.9, **edge_offset}],
            }
        ],
    }


def test_edge_offset_str_full_trio() -> None:
    token = _edge_offset_str(
        {
            "lateral_offset_m": 1.704,
            "lateral_offset_p95_m": 2.851,
            "offset_over_expected_halfwidth": 0.440,
        }
    )
    assert token == "off≈1.7m (p95 2.9, 0.44×halfw)"


def test_edge_offset_str_primary_only() -> None:
    # Only the primary offset present: no parenthetical, still rendered.
    assert _edge_offset_str({"lateral_offset_m": 3.2}) == "off≈3.2m"


def test_edge_offset_str_renders_zero_values() -> None:
    # Regression guard: a measured 0.0 offset is present evidence and must render;
    # it must never be dropped as falsy. Gating is key-presence / ``is None``, not
    # truthiness, so a future refactor to ``if not off`` would fail here.
    assert _edge_offset_str({"lateral_offset_m": 0.0}) == "off≈0.0m"
    token = _edge_offset_str(
        {
            "lateral_offset_m": 0.0,
            "lateral_offset_p95_m": 0.0,
            "offset_over_expected_halfwidth": 0.0,
        }
    )
    assert token == "off≈0.0m (p95 0.0, 0.00×halfw)"


def test_edge_offset_str_absent_renders_nothing() -> None:
    # Absence reads as absence — the pack never fabricates an "unmeasured" token.
    assert _edge_offset_str({}) == ""
    assert _edge_offset_str({"lateral_offset_p95_m": 2.9}) == ""


def test_prompt_renders_offset_token_and_legend_when_present(tmp_path) -> None:
    metadata = _offset_prompt_metadata(
        {
            "lateral_offset_m": 1.704,
            "lateral_offset_p95_m": 2.851,
            "offset_over_expected_halfwidth": 0.440,
        }
    )
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert "off≈1.7m (p95 2.9, 0.44×halfw)" in prompt
    assert _OFFSET_LEGEND_MARKER in prompt


def test_prompt_omits_offset_token_and_legend_when_absent(tmp_path) -> None:
    metadata = _offset_prompt_metadata({})
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert "off≈" not in prompt
    assert _OFFSET_LEGEND_MARKER not in prompt


def test_groups_sidecar_carries_finite_lateral_offset_evidence(tmp_path) -> None:
    ref = gpd.GeoDataFrame(
        {"id": ["r1"]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    target = gpd.GeoDataFrame(
        {"id": ["t1", "t2"]},
        geometry=[LineString([(60, 1), (90, 1)]), LineString([(10, 1), (40, 1)])],
        crs="EPSG:3857",
    )
    # t1 has a full finite offset trio; t2 has a NaN primary offset (unmeasured).
    results = [
        _mr(
            "r1",
            "t1",
            0,
            (0.6, 0.9),
            extra_features={
                "lateral_offset_m": 1.7042,
                "lateral_offset_p95_m": 2.8514,
                "offset_over_expected_halfwidth": 0.4401,
            },
        ),
        _mr(
            "r1",
            "t2",
            1,
            (0.1, 0.4),
            extra_features={"lateral_offset_m": float("nan")},
        ),
    ]
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
    offset_edge = next(edge for edge in group["edges"] if edge["target_id"] == "t1")
    assert offset_edge["lateral_offset_m"] == 1.704
    assert offset_edge["lateral_offset_p95_m"] == 2.851
    assert offset_edge["offset_over_expected_halfwidth"] == 0.44
    # NaN primary offset is omitted rather than serialized.
    unmeasured_edge = next(edge for edge in group["edges"] if edge["target_id"] == "t2")
    assert "lateral_offset_m" not in unmeasured_edge
