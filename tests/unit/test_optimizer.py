"""Unit tests for match optimization algorithms.

Tests the greedy optimizer, bipartite component detection,
contiguous ID grouping, and the M:N grouping entry point.
"""

import pytest
from shapely import LineString

from matcher.matching.optimizer import (
    _find_contiguous_id_groups,
    find_match_components,
    optimize_matches_greedy,
    optimize_matches_with_grouping,
)
from matcher.matching.types import MatchDecision, MatchResult


@pytest.fixture
def simple_results():
    """Simple match results with no conflicts."""
    return [
        MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
        MatchResult("ref_2", "t_2", MatchDecision.MATCH, 0.85, {}, {}),
        MatchResult("ref_3", "t_3", MatchDecision.REVIEW, 0.6, {}, {}),
    ]


@pytest.fixture
def conflicting_results():
    """Match results with conflicts - multiple refs for same target."""
    return [
        # ref_1 and ref_2 both want t_1
        MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
        MatchResult("ref_2", "t_1", MatchDecision.MATCH, 0.7, {}, {}),
        # ref_3 is uncontested
        MatchResult("ref_3", "t_2", MatchDecision.MATCH, 0.85, {}, {}),
        # ref_4 and ref_5 both want t_3
        MatchResult("ref_4", "t_3", MatchDecision.REVIEW, 0.6, {}, {}),
        MatchResult("ref_5", "t_3", MatchDecision.REVIEW, 0.55, {}, {}),
    ]


class TestOptimizeMatchesGreedy:
    """Tests for greedy optimization."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = optimize_matches_greedy([], min_confidence=0.5)
        assert result == []

    def test_no_conflicts_preserves_all(self, simple_results):
        """When no conflicts exist, all matches above threshold should be kept."""
        optimized = optimize_matches_greedy(simple_results, min_confidence=0.5)

        assert len(optimized) == 3
        ref_ids = {r.ref_id for r in optimized}
        assert ref_ids == {"ref_1", "ref_2", "ref_3"}

    def test_min_confidence_filters(self, simple_results):
        """Min confidence should filter low-confidence results."""
        optimized = optimize_matches_greedy(simple_results, min_confidence=0.8)

        assert len(optimized) == 2
        ref_ids = {r.ref_id for r in optimized}
        assert "ref_1" in ref_ids
        assert "ref_2" in ref_ids

    def test_conflicts_resolved_one_to_one(self, conflicting_results):
        """Conflicts should be resolved with 1:1 assignments."""
        optimized = optimize_matches_greedy(conflicting_results, min_confidence=0.5)

        # Each reference should appear at most once
        ref_ids = [r.ref_id for r in optimized]
        assert len(ref_ids) == len(set(ref_ids))

        # Each target should appear at most once
        target_ids = [r.target_id for r in optimized]
        assert len(target_ids) == len(set(target_ids))

    def test_greedy_takes_highest_first(self, conflicting_results):
        """Greedy should take highest confidence matches first."""
        optimized = optimize_matches_greedy(conflicting_results, min_confidence=0.5)

        # The highest confidence match (ref_1, t_1, 0.9) should be included
        ref_1_matches = [r for r in optimized if r.ref_id == "ref_1"]
        assert len(ref_1_matches) == 1
        assert ref_1_matches[0].target_id == "t_1"
        assert ref_1_matches[0].confidence == 0.9


class TestDuplicateCandidates:
    """Tests for handling duplicate candidates for the same ref-target pair."""

    def test_keeps_best_duplicate(self):
        """When multiple candidates exist for same pair, keep highest confidence."""
        results = [
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.8, {}, {}),  # Duplicate, lower
            MatchResult("ref_2", "t_2", MatchDecision.MATCH, 0.85, {}, {}),
        ]

        optimized = optimize_matches_greedy(results, min_confidence=0.5)

        # Should only have 2 unique pairs
        pairs = {(r.ref_id, r.target_id) for r in optimized}
        assert len(pairs) == 2

        # ref_1, t_1 should have the higher confidence
        ref_1_match = [r for r in optimized if r.ref_id == "ref_1"][0]
        assert ref_1_match.confidence == 0.9


class TestFindMatchComponents:
    """Tests for bipartite connected component detection."""

    def test_empty_input(self):
        """Empty input should return no components."""
        assert find_match_components([], min_confidence=0.5) == []

    def test_single_pair(self):
        """A single match pair should be one component."""
        results = [MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {})]
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 1
        assert len(components[0]) == 1

    def test_shared_ref_component(self):
        """Two matches sharing a ref should be one component."""
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r1", "t2", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 1
        assert len(components[0]) == 2

    def test_shared_target_component(self):
        """Two matches sharing a target should be one component."""
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 1
        assert len(components[0]) == 2

    def test_two_separate_components(self):
        """Disconnected pairs should be separate components."""
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 2

    def test_chain_forms_one_component(self):
        """A chain r1-t1-r2-t2 should be one component."""
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.8, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.7, {}, {}),
        ]
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 1
        assert len(components[0]) == 3

    def test_min_confidence_filters(self):
        """Results below min_confidence should be excluded."""
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.3, {}, {}),  # Below threshold
        ]
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 1
        assert components[0][0].ref_id == "r1"


class TestFindContiguousIdGroups:
    """Tests for generalized contiguous ID grouping."""

    def test_single_id(self):
        """Single ID should return one group."""
        geoms = {"a": LineString([(0, 0), (10, 0)])}
        groups = _find_contiguous_id_groups(["a"], geoms, tolerance=5.0)
        assert groups == [["a"]]

    def test_two_contiguous(self):
        """Two IDs with touching endpoints should form one group."""
        geoms = {
            "a": LineString([(0, 0), (50, 0)]),
            "b": LineString([(50, 0), (100, 0)]),
        }
        groups = _find_contiguous_id_groups(["a", "b"], geoms, tolerance=5.0)
        assert len(groups) == 1
        assert set(groups[0]) == {"a", "b"}

    def test_two_non_contiguous(self):
        """Two IDs with distant endpoints should form separate groups."""
        geoms = {
            "a": LineString([(0, 0), (50, 0)]),
            "b": LineString([(200, 0), (250, 0)]),
        }
        groups = _find_contiguous_id_groups(["a", "b"], geoms, tolerance=5.0)
        assert len(groups) == 2

    def test_partial_contiguity(self):
        """Three IDs where only two are contiguous."""
        geoms = {
            "a": LineString([(0, 0), (50, 0)]),
            "b": LineString([(50, 0), (100, 0)]),
            "c": LineString([(500, 0), (550, 0)]),
        }
        groups = _find_contiguous_id_groups(["a", "b", "c"], geoms, tolerance=5.0)
        assert len(groups) == 2
        group_sets = [set(g) for g in groups]
        assert {"a", "b"} in group_sets
        assert {"c"} in group_sets

    def test_missing_geometry(self):
        """IDs with missing geometry should form singleton groups."""
        geoms = {"a": LineString([(0, 0), (50, 0)])}
        groups = _find_contiguous_id_groups(["a", "b"], geoms, tolerance=5.0)
        assert len(groups) == 2

    def test_empty_input(self):
        """Empty ID list should return empty."""
        groups = _find_contiguous_id_groups([], {}, tolerance=5.0)
        assert groups == []


class TestOptimizeMatchesWithGrouping:
    """Tests for the full M:N grouping optimizer."""

    def _make_gdf(self, id_col, geom_dict):
        """Helper to create a GeoDataFrame from an id→geometry dict."""
        import geopandas as gpd

        return gpd.GeoDataFrame(
            {id_col: list(geom_dict.keys()), "geometry": list(geom_dict.values())},
            crs="EPSG:32610",
        )

    def test_n_to_1_contiguous_refs(self):
        """Multiple contiguous refs matching one target → N:1 group."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (100, 0)]),
        }
        target_geoms = {
            "t1": LineString([(0, 5), (100, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.85, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        assert len(optimized) == 2
        for r in optimized:
            assert r.features["match_type"] == "N:1"
            assert r.features["group_ref_count"] == 2
            assert r.features["group_target_count"] == 1

    def test_m_to_n_both_contiguous(self):
        """M:N where both sides are fully contiguous."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (100, 0)]),
        }
        target_geoms = {
            "t1": LineString([(0, 5), (40, 5)]),
            "t2": LineString([(40, 5), (100, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

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
            assert r.features["group_ref_count"] == 2
            assert r.features["group_target_count"] == 2

    def test_mixed_groups_and_1to1(self):
        """Mix of 1:N group and unrelated 1:1 match."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(200, 0), (250, 0)]),
        }
        target_geoms = {
            "t1": LineString([(0, 5), (30, 5)]),
            "t2": LineString([(30, 5), (60, 5)]),
            "t3": LineString([(200, 5), (250, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

        results = [
            # 1:N group: r1 → t1, t2 (contiguous targets)
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r1", "t2", MatchDecision.MATCH, 0.85, {}, {}),
            # 1:1: r2 → t3
            MatchResult("r2", "t3", MatchDecision.MATCH, 0.8, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        assert len(optimized) == 3

        # Check 1:N group
        group_matches = [r for r in optimized if r.features.get("match_type") == "1:N"]
        assert len(group_matches) == 2

        # Check 1:1
        solo_matches = [r for r in optimized if "match_type" not in r.features]
        assert len(solo_matches) == 1
        assert solo_matches[0].ref_id == "r2"

    def test_non_contiguous_falls_back_to_1to1(self):
        """Non-contiguous refs matching same target → fall back to 1:1."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(500, 0), (550, 0)]),  # Far away
        }
        target_geoms = {
            "t1": LineString([(0, 5), (550, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.7, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        # Non-contiguous refs → fall back to 1:1, greedy picks highest
        assert len(optimized) == 1
        assert optimized[0].ref_id == "r1"
        assert optimized[0].confidence == 0.9

    def test_post_expansion_adds_contiguous_targets(self):
        """Post-expansion adds contiguous targets to 1:1 greedy matches.

        Scenario: M:N component (refs not contiguous) falls to 1:1 greedy.
        After greedy assigns r1→t1, post-expansion detects t2 is contiguous
        with t1 and adds it → 1:N group for r1.
        """
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(200, 0), (250, 0)]),  # Far from r1
        }
        target_geoms = {
            "t1": LineString([(0, 5), (30, 5)]),
            "t2": LineString([(30, 5), (60, 5)]),  # Contiguous with t1
            "t3": LineString([(200, 5), (260, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

        # M:N component: r1→t1(0.9), r1→t2(0.85), r2→t2(0.7), r2→t3(0.75)
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r1", "t2", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.7, {}, {}),
            MatchResult("r2", "t3", MatchDecision.MATCH, 0.75, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        # Greedy assigns r1→t1 (highest conf), r2→t3 (next available)
        # Post-expansion: r1 has candidate t2, contiguous with t1 → add t2
        r1_matches = [r for r in optimized if r.ref_id == "r1"]
        assert len(r1_matches) == 2
        assert {r.target_id for r in r1_matches} == {"t1", "t2"}
        for r in r1_matches:
            assert r.features["match_type"] == "1:N"

        # r2 keeps its 1:1 assignment to t3
        r2_matches = [r for r in optimized if r.ref_id == "r2"]
        assert len(r2_matches) == 1
        assert r2_matches[0].target_id == "t3"

    def test_post_expansion_adds_contiguous_refs(self):
        """Post-expansion adds contiguous refs to 1:1 greedy matches (N:1)."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (100, 0)]),  # Contiguous with r1
            "r3": LineString([(200, 0), (250, 0)]),
        }
        target_geoms = {
            "t1": LineString([(0, 5), (100, 5)]),
            "t2": LineString([(200, 5), (250, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

        # M:N component via shared target t1
        # r1→t1(0.9), r2→t1(0.85), r3→t1(0.6), r3→t2(0.7)
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("r3", "t1", MatchDecision.MATCH, 0.6, {}, {}),
            MatchResult("r3", "t2", MatchDecision.MATCH, 0.7, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        # Greedy: r1→t1, r3→t2 (r2 blocked by r1 on t1)
        # Post-expansion N:1: t1 has candidate r2 (contiguous with r1) → add r2
        n1_matches = [r for r in optimized if r.features.get("match_type") == "N:1"]
        assert len(n1_matches) == 2
        assert {r.ref_id for r in n1_matches} == {"r1", "r2"}

        # r3 keeps 1:1 assignment to t2
        r3_matches = [r for r in optimized if r.ref_id == "r3"]
        assert len(r3_matches) == 1
        assert r3_matches[0].target_id == "t2"

    def test_preserves_alignment_fractions(self):
        """Group results should preserve original alignment fractions."""
        ref_geoms = {
            "r1": LineString([(0, 0), (100, 0)]),
        }
        target_geoms = {
            "t1": LineString([(0, 5), (50, 5)]),
            "t2": LineString([(50, 5), (100, 5)]),
        }
        ref_gdf = self._make_gdf("id", ref_geoms)
        target_gdf = self._make_gdf("local_id", target_geoms)

        results = [
            MatchResult(
                "r1",
                "t1",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=0.5,
                local_start_frac=0.0,
                local_end_frac=1.0,
            ),
            MatchResult(
                "r1",
                "t2",
                MatchDecision.MATCH,
                0.85,
                {},
                {},
                gers_start_frac=0.5,
                gers_end_frac=1.0,
                local_start_frac=0.0,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        # Should preserve alignment fractions
        t1_match = [r for r in optimized if r.target_id == "t1"][0]
        assert t1_match.gers_start_frac == 0.0
        assert t1_match.gers_end_frac == 0.5

        t2_match = [r for r in optimized if r.target_id == "t2"][0]
        assert t2_match.gers_start_frac == 0.5
        assert t2_match.gers_end_frac == 1.0
