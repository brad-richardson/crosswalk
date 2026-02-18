"""Integration tests for scoring to optimizer pipeline.

Tests that match results are correctly optimized with M:N grouping,
handling 1:1 conflicts, 1:N, N:1, and M:N groups.
"""

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.matching.optimizer import optimize_matches_with_grouping
from matcher.matching.types import MatchDecision, MatchResult


@pytest.fixture
def contiguous_target_gdf():
    """Target GeoDataFrame with contiguous segments."""
    return gpd.GeoDataFrame(
        {
            "local_id": ["t_1", "t_2", "t_3", "t_4", "t_5"],
            "geometry": [
                LineString([(0, 0), (50, 0)]),  # First half of ref_1
                LineString([(50, 0), (100, 0)]),  # Second half (contiguous with t_1)
                LineString([(100, 5), (100, 50)]),  # Near ref_2
                LineString([(200, 0), (250, 0)]),  # Not contiguous
                LineString([(300, 0), (350, 0)]),  # Far from t_4
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def simple_ref_gdf():
    """Reference GeoDataFrame with a few segments."""
    return gpd.GeoDataFrame(
        {
            "id": ["ref_1", "ref_2", "ref_3"],
            "geometry": [
                LineString([(0, -5), (100, -5)]),
                LineString([(100, 0), (100, 55)]),
                LineString([(200, -5), (350, -5)]),
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def contiguous_ref_gdf():
    """Reference GeoDataFrame with contiguous ref segments."""
    return gpd.GeoDataFrame(
        {
            "id": ["ref_1", "ref_2", "ref_3"],
            "geometry": [
                LineString([(0, 0), (50, 0)]),
                LineString([(50, 0), (100, 0)]),  # Contiguous with ref_1
                LineString([(500, 0), (550, 0)]),  # Far away
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def n_to_1_target_gdf():
    """Target GeoDataFrame for N:1 tests."""
    return gpd.GeoDataFrame(
        {
            "local_id": ["t_1", "t_2"],
            "geometry": [
                LineString([(0, 5), (100, 5)]),
                LineString([(500, 5), (550, 5)]),
            ],
        },
        crs="EPSG:32610",
    )


class TestOptimizeMatchesWithGrouping:
    """Tests for the full M:N grouping optimizer."""

    def test_1n_contiguous_targets(self, simple_ref_gdf, contiguous_target_gdf):
        """Contiguous targets matching same ref should form 1:N group."""
        results = [
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_1", "t_2", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_2", "t_3", MatchDecision.MATCH, 0.8, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            simple_ref_gdf,
            contiguous_target_gdf,
            min_confidence=0.5,
            contiguity_tolerance=5.0,
        )

        # Should have all 3 matches
        assert len(optimized) == 3
        target_ids = {r.target_id for r in optimized}
        assert target_ids == {"t_1", "t_2", "t_3"}

        # t_1 and t_2 should be 1:N
        group_matches = [r for r in optimized if r.features.get("match_type") == "1:N"]
        assert len(group_matches) == 2
        assert {r.target_id for r in group_matches} == {"t_1", "t_2"}

    def test_n_to_1_contiguous_refs(self, contiguous_ref_gdf, n_to_1_target_gdf):
        """Contiguous refs matching same target should form N:1 group."""
        results = [
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_2", "t_1", MatchDecision.MATCH, 0.85, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            contiguous_ref_gdf,
            n_to_1_target_gdf,
            min_confidence=0.5,
            contiguity_tolerance=5.0,
        )

        assert len(optimized) == 2
        for r in optimized:
            assert r.features["match_type"] == "N:1"
            assert r.features["group_ref_count"] == 2
            assert r.features["group_target_count"] == 1

    def test_m_to_n_both_sides(self):
        """M:N where both sides are fully contiguous."""
        ref_gdf = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2"],
                "geometry": [
                    LineString([(0, 0), (50, 0)]),
                    LineString([(50, 0), (100, 0)]),
                ],
            },
            crs="EPSG:32610",
        )
        target_gdf = gpd.GeoDataFrame(
            {
                "local_id": ["t1", "t2"],
                "geometry": [
                    LineString([(0, 5), (40, 5)]),
                    LineString([(40, 5), (100, 5)]),
                ],
            },
            crs="EPSG:32610",
        )

        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r1", "t2", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.8, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.75, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        assert len(optimized) == 4
        for r in optimized:
            assert r.features["match_type"] == "M:N"

    def test_mixed_groups_and_1to1(self, simple_ref_gdf, contiguous_target_gdf):
        """Mix of group types and 1:1 matches."""
        results = [
            # 1:N: ref_1 → t_1, t_2 (contiguous targets)
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_1", "t_2", MatchDecision.MATCH, 0.85, {}, {}),
            # 1:1: ref_2 → t_3
            MatchResult("ref_2", "t_3", MatchDecision.MATCH, 0.8, {}, {}),
            # 1:1: ref_3 → t_4
            MatchResult("ref_3", "t_4", MatchDecision.MATCH, 0.7, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            simple_ref_gdf,
            contiguous_target_gdf,
            min_confidence=0.5,
            contiguity_tolerance=5.0,
        )

        assert len(optimized) == 4
        target_ids = {r.target_id for r in optimized}
        assert target_ids == {"t_1", "t_2", "t_3", "t_4"}

    def test_no_double_claiming(self, contiguous_ref_gdf, n_to_1_target_gdf):
        """Target in N:1 group should be excluded from 1:1 pool."""
        results = [
            # N:1 group: ref_1, ref_2 → t_1 (contiguous refs)
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_2", "t_1", MatchDecision.MATCH, 0.85, {}, {}),
            # ref_3 also wants t_1 but is not contiguous with ref_1/ref_2
            MatchResult("ref_3", "t_1", MatchDecision.MATCH, 0.7, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            contiguous_ref_gdf,
            n_to_1_target_gdf,
            min_confidence=0.5,
            contiguity_tolerance=5.0,
        )

        # ref_1 and ref_2 form N:1 group, ref_3's claim on t_1 is blocked
        n1_matches = [r for r in optimized if r.features.get("match_type") == "N:1"]
        assert len(n1_matches) == 2
        assert {r.ref_id for r in n1_matches} == {"ref_1", "ref_2"}

        # ref_3 should NOT appear (t_1 is claimed, and ref_3 has no other targets)
        ref_3_matches = [r for r in optimized if r.ref_id == "ref_3"]
        assert len(ref_3_matches) == 0

    def test_non_contiguous_falls_back_to_1to1(self, contiguous_ref_gdf, n_to_1_target_gdf):
        """Non-contiguous refs matching same target should fall back to 1:1."""
        # ref_1 and ref_3 are NOT contiguous (ref_3 is far away)
        results = [
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_3", "t_1", MatchDecision.MATCH, 0.7, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            contiguous_ref_gdf,
            n_to_1_target_gdf,
            min_confidence=0.5,
            contiguity_tolerance=5.0,
        )

        # Should fall back to 1:1, greedy picks ref_1 (higher confidence)
        assert len(optimized) == 1
        assert optimized[0].ref_id == "ref_1"
        assert "match_type" not in optimized[0].features
