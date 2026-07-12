"""Unit tests for match optimization algorithms.

Tests the greedy optimizer, bipartite component detection,
contiguous ID grouping, and the M:N grouping entry point.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from shapely import LineString

from crosswalk.matching.optimizer import (
    DECOMPOSED_SINGLETON_REVIEW_FLAG,
    LOW_CONFIDENCE_ADDITION_REVIEW_FLAG,
    PARALLEL_SIBLING_REVIEW_FLAG,
    PRUNED_SINGLETON_REVIEW_FLAG,
    _contested_small_span_review_pairs,
    _create_group_results,
    _endpoints_are_collinear,
    _expand_greedy_matches,
    _find_contiguous_id_groups,
    _normalized_names,
    _normalized_names_for_range,
    _validate_assignment_coverage,
    apply_confidence_drop_prune,
    build_contiguity_adjacency,
    find_match_components,
    group_is_structurally_simple,
    optimize_matches_greedy,
    optimize_matches_with_grouping,
)
from crosswalk.matching.types import MatchDecision, MatchResult, MatchType


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

    def test_exact_tie_prefers_aligned_meters_over_fraction(self):
        """Fractional coverage cannot make a short sliver beat a longer match."""
        results = [
            MatchResult(
                "ref_a",
                "target",
                MatchDecision.MATCH,
                0.9,
                {},
                {"aligned_length_m": 9.0},
                gers_start_frac=0.0,
                gers_end_frac=0.9,
                local_start_frac=0.0,
                local_end_frac=0.9,
            ),
            MatchResult(
                "ref_b",
                "target",
                MatchDecision.MATCH,
                0.9,
                {},
                {"aligned_length_m": 50.0},
                gers_start_frac=0.0,
                gers_end_frac=0.5,
                local_start_frac=0.0,
                local_end_frac=0.5,
            ),
        ]

        optimized = optimize_matches_greedy(results, min_confidence=0.5)

        assert [(result.ref_id, result.target_id) for result in optimized] == [("ref_b", "target")]


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

    def test_one_to_n_disconnected_chains_stay_separate(self):
        """One shared ref must not flatten two distant target chains into one group."""
        ref_geoms = {"r0": LineString([(0, 5), (300, 5)])}
        target_geoms = {
            "t1": LineString([(0, 0), (50, 0)]),
            "t2": LineString([(50, 0), (100, 0)]),
            "t3": LineString([(200, 0), (250, 0)]),
            "t4": LineString([(250, 0), (300, 0)]),
        }
        results = [
            MatchResult("r0", target_id, MatchDecision.MATCH, 0.9, {}, {})
            for target_id in target_geoms
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            contiguity_tolerance=5.0,
            corridor_aware=True,
        )

        groups: dict[str, set[str]] = {}
        for result in optimized:
            assert result.features["match_type"] == "1:N"
            assert result.features["group_size"] == 2
            groups.setdefault(result.features["group_id"], set()).add(result.target_id)
        assert set(map(frozenset, groups.values())) == {
            frozenset({"t1", "t2"}),
            frozenset({"t3", "t4"}),
        }

    def test_n_to_one_disconnected_chains_stay_separate(self):
        """One shared target must not flatten two distant ref chains into one group."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (100, 0)]),
            "r3": LineString([(200, 0), (250, 0)]),
            "r4": LineString([(250, 0), (300, 0)]),
        }
        target_geoms = {"t0": LineString([(0, 5), (300, 5)])}
        results = [
            MatchResult(ref_id, "t0", MatchDecision.MATCH, 0.9, {}, {}) for ref_id in ref_geoms
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            contiguity_tolerance=5.0,
            corridor_aware=True,
        )

        groups: dict[str, set[str]] = {}
        for result in optimized:
            assert result.features["match_type"] == "N:1"
            assert result.features["group_size"] == 2
            groups.setdefault(result.features["group_id"], set()).add(result.ref_id)
        assert set(map(frozenset, groups.values())) == {
            frozenset({"r1", "r2"}),
            frozenset({"r3", "r4"}),
        }

    def test_low_confidence_singleton_from_group_decomposition_stays_review(self):
        """Splitting a multi-node component cannot promote its weak winner to MATCH."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(200, 0), (250, 0)]),
        }
        target_geoms = {"t0": LineString([(0, 5), (250, 5)])}
        results = [
            MatchResult("r1", "t0", MatchDecision.MATCH, 0.68, {}, {}),
            MatchResult("r2", "t0", MatchDecision.MATCH, 0.65, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in optimized] == [("r1", "t0")]
        assert optimized[0].decision == MatchDecision.REVIEW
        assert optimized[0].features[DECOMPOSED_SINGLETON_REVIEW_FLAG] == 1.0

    def test_low_confidence_singleton_with_unattached_weak_siblings_stays_review(self):
        """Glue pruning cannot hide the multi-node provenance of a singleton."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(200, 0), (250, 0)]),
            "r3": LineString([(400, 0), (450, 0)]),
        }
        target_geoms = {"t0": LineString([(0, 5), (450, 5)])}
        results = [
            MatchResult("r1", "t0", MatchDecision.MATCH, 0.682759, {}, {}),
            MatchResult("r2", "t0", MatchDecision.REVIEW, 0.1875, {}, {}),
            MatchResult("r3", "t0", MatchDecision.REVIEW, 0.1875, {}, {}),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.1,
            glue_min_confidence=0.575,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in optimized] == [("r1", "t0")]
        assert optimized[0].decision == MatchDecision.REVIEW
        assert optimized[0].features[DECOMPOSED_SINGLETON_REVIEW_FLAG] == 1.0

    def test_native_low_confidence_one_to_one_keeps_scorer_decision(self):
        """The decomposition guard does not redefine ordinary 1:1 scoring."""
        result = MatchResult("r1", "t1", MatchDecision.MATCH, 0.68, {}, {})

        optimized = optimize_matches_with_grouping(
            [result],
            self._make_gdf("id", {"r1": LineString([(0, 0), (50, 0)])}),
            self._make_gdf("local_id", {"t1": LineString([(0, 5), (50, 5)])}),
            min_confidence=0.5,
            corridor_aware=True,
        )

        assert optimized == [result]
        assert DECOMPOSED_SINGLETON_REVIEW_FLAG not in optimized[0].features

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

    def test_cross_linked_post_expansion_emits_each_pair_once(self):
        """One edge shared by ref/target expansion recomposes into one M:N group."""
        greedy = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        cross_link = MatchResult("r1", "t2", MatchDecision.MATCH, 0.85, {}, {})

        expanded = _expand_greedy_matches(
            greedy,
            [*greedy, cross_link],
            {
                "r1": LineString([(0, 0), (50, 0)]),
                "r2": LineString([(50, 0), (100, 0)]),
            },
            {
                "t1": LineString([(0, 5), (50, 5)]),
                "t2": LineString([(50, 5), (100, 5)]),
            },
            tolerance=5.0,
            min_confidence=0.5,
            glue_min_confidence=0.575,
        )

        pairs = [(result.ref_id, result.target_id) for result in expanded]
        assert len(pairs) == len(set(pairs)) == 3
        assert {result.features["match_type"] for result in expanded} == {"M:N"}
        assert len({result.features["group_id"] for result in expanded}) == 1

    def test_low_confidence_post_expansion_addition_stays_review(self):
        """Group averaging cannot auto-accept a weak edge added after greedy."""
        anchor = MatchResult("r1", "t1", MatchDecision.MATCH, 0.95, {}, {})
        weak_addition = MatchResult("r1", "t2", MatchDecision.REVIEW, 0.6, {}, {})

        expanded = _expand_greedy_matches(
            [anchor],
            [anchor, weak_addition],
            {"r1": LineString([(0, 0), (100, 0)])},
            {
                "t1": LineString([(0, 5), (50, 5)]),
                "t2": LineString([(50, 5), (100, 5)]),
            },
            tolerance=5.0,
            min_confidence=0.5,
            glue_min_confidence=0.575,
        )
        by_target = {result.target_id: result for result in expanded}

        assert by_target["t1"].decision == MatchDecision.MATCH
        assert by_target["t2"].decision == MatchDecision.REVIEW
        assert by_target["t2"].features[LOW_CONFIDENCE_ADDITION_REVIEW_FLAG] == 1.0

    @pytest.mark.parametrize(
        "duplicate_order",
        [(0.6, 0.9), (0.9, 0.6), (float("nan"), 0.9)],
    )
    def test_post_expansion_duplicate_pair_uses_highest_confidence(self, duplicate_order):
        """Raw duplicate order cannot make a weaker row claim an expanded pair."""
        anchor = MatchResult("r1", "t1", MatchDecision.MATCH, 0.95, {}, {})
        duplicates = [
            MatchResult("r1", "t2", MatchDecision.MATCH, confidence, {}, {})
            for confidence in duplicate_order
        ]

        expanded = _expand_greedy_matches(
            [anchor],
            [anchor, *duplicates],
            {"r1": LineString([(0, 0), (100, 0)])},
            {
                "t1": LineString([(0, 5), (50, 5)]),
                "t2": LineString([(50, 5), (100, 5)]),
            },
            tolerance=5.0,
            min_confidence=0.5,
        )
        by_target = {result.target_id: result for result in expanded}

        assert by_target["t2"].confidence == 0.9
        assert by_target["t2"].decision == MatchDecision.MATCH
        assert LOW_CONFIDENCE_ADDITION_REVIEW_FLAG not in by_target["t2"].features

    def test_two_cross_links_recompose_to_four_unique_edges(self):
        """Bidirectional expansion emits a complete 2x2 once, as one M:N group."""
        anchors = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.95, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.94, {}, {}),
        ]
        cross_links = [
            MatchResult("r1", "t2", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.89, {}, {}),
        ]

        expanded = _expand_greedy_matches(
            anchors,
            [*anchors, *cross_links],
            {
                "r1": LineString([(0, 0), (50, 0)]),
                "r2": LineString([(50, 0), (100, 0)]),
            },
            {
                "t1": LineString([(0, 5), (50, 5)]),
                "t2": LineString([(50, 5), (100, 5)]),
            },
            tolerance=5.0,
            min_confidence=0.5,
        )
        pairs = [(result.ref_id, result.target_id) for result in expanded]

        assert len(pairs) == len(set(pairs)) == 4
        assert set(pairs) == {("r1", "t1"), ("r1", "t2"), ("r2", "t1"), ("r2", "t2")}
        assert {result.features["match_type"] for result in expanded} == {"M:N"}
        assert len({result.features["group_id"] for result in expanded}) == 1

    def test_below_glue_cross_links_cannot_weld_greedy_anchors(self):
        """The grouping-only floor also governs post-greedy expansion."""
        anchors = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.95, {}, {}),
            MatchResult("r2", "t2", MatchDecision.MATCH, 0.94, {}, {}),
        ]
        weak_cross_links = [
            MatchResult("r1", "t2", MatchDecision.REVIEW, 0.55, {}, {}),
            MatchResult("r2", "t1", MatchDecision.REVIEW, 0.54, {}, {}),
        ]

        expanded = _expand_greedy_matches(
            anchors,
            [*anchors, *weak_cross_links],
            {
                "r1": LineString([(0, 0), (50, 0)]),
                "r2": LineString([(50, 0), (100, 0)]),
            },
            {
                "t1": LineString([(0, 5), (50, 5)]),
                "t2": LineString([(50, 5), (100, 5)]),
            },
            tolerance=5.0,
            min_confidence=0.5,
            glue_min_confidence=0.575,
        )

        assert {(result.ref_id, result.target_id) for result in expanded} == {
            ("r1", "t1"),
            ("r2", "t2"),
        }
        assert all("group_id" not in result.features for result in expanded)

    def test_corridor_gate_blocks_perpendicular_one_to_n_expansion(self):
        """A junction touch cannot become 1:N without aligned-name evidence."""
        ref_geoms = {"r0": LineString([(0, 2), (50, 2)])}
        target_geoms = {
            "t1": LineString([(0, 0), (50, 0)]),
            "t2": LineString([(50, 0), (50, 50)]),
        }
        results = [
            MatchResult("r0", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r0", "t2", MatchDecision.MATCH, 0.8, {}, {}),
        ]

        gated = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            contiguity_tolerance=1.0,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in gated] == [("r0", "t1")]

    def test_corridor_gate_blocks_perpendicular_n_to_one_expansion(self):
        """The same sharp-turn rule applies symmetrically on the ref side."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (50, 50)]),
        }
        target_geoms = {"t0": LineString([(0, 2), (50, 2)])}
        results = [
            MatchResult("r1", "t0", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r2", "t0", MatchDecision.MATCH, 0.8, {}, {}),
        ]

        gated = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            contiguity_tolerance=1.0,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in gated] == [("r1", "t0")]

    def test_range_names_require_positive_rule_overlap(self):
        """Touching a name-rule boundary provides no evidence for the prior name."""
        names = {
            "primary": "Main Street",
            "rules": [
                {"between": [0.0, 0.5], "value": "Main Street"},
                {"between": [0.5, 1.0], "value": "Other Street"},
            ],
        }

        assert _normalized_names_for_range(names, (0.5, 0.8)) == {"other street"}
        assert _normalized_names_for_range(names, (0.49, 0.51)) == {
            "main street",
            "other street",
        }

    def test_common_names_parse_values_without_admitting_language_codes(self):
        """Target dicts and Overture arrays share aliases, never language tags."""
        target_names = {
            "primary": "Rue Principale",
            "common": {"en": "Main Street", "fr": "Rue Principale"},
        }
        overture_names = {
            "primary": "Rue Principale",
            "common": np.array(
                [["en", "Main Street"], ["fr", "Rue Principale"]],
                dtype=object,
            ),
        }

        expected = {"rue principale", "main street"}
        assert _normalized_names(target_names) == expected
        assert _normalized_names(overture_names) == expected
        assert "en" not in _normalized_names(overture_names)
        assert "fr" not in _normalized_names(overture_names)

    def test_range_name_rescue_uses_common_alias_only_on_primary_span(self):
        """A primary translation is valid on its span, not another LR name's span."""
        names = {
            "primary": "Rue Principale",
            "common": np.array([["en", "Main Street"]], dtype=object),
            "rules": np.array(
                [
                    {"between": [0.0, 0.5], "value": "Rue Principale"},
                    {"between": [0.5, 1.0], "value": "Rue Secondaire"},
                ],
                dtype=object,
            ),
        }

        assert _normalized_names_for_range(names, (0.0, 0.4)) == {
            "rue principale",
            "main street",
        }
        assert _normalized_names_for_range(names, (0.6, 1.0)) == {"rue secondaire"}

    def test_alignment_rescue_accepts_common_name_translation(self):
        """A scoped Overture common alias can join complementary target spans."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2", "connector"],
                "names": [
                    {
                        "primary": "Rue Principale",
                        "common": np.array([["en", "Main Street"]], dtype=object),
                        "rules": np.array(
                            [{"between": [0.0, 1.0], "value": "Rue Principale"}],
                            dtype=object,
                        ),
                    },
                    {"primary": "Main Street"},
                    {"primary": "Cross Street"},
                ],
                "geometry": [
                    LineString([(0, 0), (40, 0)]),
                    LineString([(50, 0), (100, 0)]),
                    LineString([(40, 0), (50, 0)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t"],
                "names": [{"primary": "Main Street"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        candidates = [
            MatchResult(
                "r1",
                "t",
                MatchDecision.MATCH,
                0.95,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.0,
                local_end_frac=0.4,
            ),
            MatchResult(
                "r2",
                "t",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.5,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            candidates,
            reference,
            target,
            min_confidence=0.5,
            contiguity_tolerance=1.0,
            corridor_aware=True,
        )

        assert {result.ref_id for result in optimized} == {"r1", "r2"}
        assert {result.features["match_type"] for result in optimized} == {"N:1"}

    def test_rescue_uses_multi_side_name_for_candidate_range(self):
        """A name that only touches the target span boundary cannot justify rescue."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r"],
                "names": [{"primary": "Main Street"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t1", "t2"],
                "names": [
                    {"primary": "Main Street"},
                    {
                        "primary": "Other Street",
                        "rules": [
                            {"between": [0.0, 0.5], "value": "Main Street"},
                            {"between": [0.5, 1.0], "value": "Other Street"},
                        ],
                    },
                ],
                "geometry": [
                    LineString([(0, 0), (50, 0)]),
                    LineString([(50, 0), (50, 50)]),
                ],
            },
            crs="EPSG:32619",
        )
        candidates = [
            MatchResult(
                "r",
                "t1",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=0.45,
                local_start_frac=0.0,
                local_end_frac=1.0,
            ),
            MatchResult(
                "r",
                "t2",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.55,
                gers_end_frac=1.0,
                local_start_frac=0.5,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            candidates,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in optimized] == [("r", "t1")]

    def test_equal_score_duplicate_alignment_is_order_invariant(self):
        """A conflicting duplicate row cannot change whether rescue succeeds."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r"],
                "names": [{"primary": "Main Street"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t1", "t2", "connector"],
                "names": [
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Cross Street"},
                ],
                "geometry": [
                    LineString([(0, 0), (40, 0)]),
                    LineString([(50, 0), (100, 0)]),
                    LineString([(40, 0), (50, 0)]),
                ],
            },
            crs="EPSG:32619",
        )
        t1 = MatchResult(
            "r",
            "t1",
            MatchDecision.MATCH,
            0.9,
            {},
            {},
            gers_start_frac=0.0,
            gers_end_frac=0.4,
            local_start_frac=0.0,
            local_end_frac=1.0,
        )
        good_t2 = replace(
            t1,
            target_id="t2",
            gers_start_frac=0.5,
            gers_end_frac=1.0,
        )
        conflicting_t2 = replace(t1, target_id="t2")

        outputs = []
        for duplicates in ((good_t2, conflicting_t2), (conflicting_t2, good_t2)):
            optimized = optimize_matches_with_grouping(
                [t1, *duplicates],
                reference,
                target,
                min_confidence=0.5,
                contiguity_tolerance=1.0,
                corridor_aware=True,
            )
            outputs.append(
                [
                    (
                        result.ref_id,
                        result.target_id,
                        result.gers_start_frac,
                        result.gers_end_frac,
                    )
                    for result in optimized
                ]
            )

        assert outputs[0] == outputs[1]
        assert {(ref_id, target_id) for ref_id, target_id, _, _ in outputs[0]} == {
            ("r", "t1"),
            ("r", "t2"),
        }
        assert next(row for row in outputs[0] if row[1] == "t2")[2:] == (0.5, 1.0)

    def test_alignment_rescue_is_gap_bounded_and_requires_match_confidence(self):
        """Only strong complementary same-name spans can bridge a short gap."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2", "connector"],
                "names": [
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Cross Street"},
                ],
                "geometry": [
                    LineString([(0, 0), (40, 0)]),
                    LineString([(60, 0), (100, 0)]),
                    LineString([(40, 0), (60, 0)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t0"],
                "names": [{"primary": "MAIN STREET"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        results = [
            MatchResult(
                "r1",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.0,
                local_end_frac=0.4,
            ),
            MatchResult(
                "r2",
                "t0",
                MatchDecision.MATCH,
                0.8,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.5,
                local_end_frac=1.0,
            ),
        ]

        rescued = optimize_matches_with_grouping(
            results,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
            alignment_rescue_max_gap_m=15.0,
        )
        by_ref = {result.ref_id: result for result in rescued}
        assert set(by_ref) == {"r1", "r2"}
        assert all(result.decision == MatchDecision.MATCH for result in rescued)

        not_rescued = optimize_matches_with_grouping(
            results,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
            alignment_rescue_max_gap_m=5.0,
        )
        assert [(result.ref_id, result.target_id) for result in not_rescued] == [("r1", "t0")]

        far_reference = reference.copy()
        far_reference.loc[far_reference["id"] == "r2", "geometry"] = [
            LineString([(100, 0), (140, 0)])
        ]
        far_not_rescued = optimize_matches_with_grouping(
            results,
            far_reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
            alignment_rescue_max_gap_m=15.0,
        )
        assert [(result.ref_id, result.target_id) for result in far_not_rescued] == [("r1", "t0")]

        weak_results = [results[0], replace(results[1], confidence=0.7)]
        weak_not_rescued = optimize_matches_with_grouping(
            weak_results,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
            alignment_rescue_max_gap_m=15.0,
        )
        assert [(result.ref_id, result.target_id) for result in weak_not_rescued] == [("r1", "t0")]

    def test_alignment_rescue_long_gap_requires_collinear_continuation(self):
        """A nearby same-name branch cannot use alignment alone across a long gap."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2", "connector"],
                "names": [
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Cross Street"},
                ],
                "geometry": [
                    LineString([(0, 0), (40, 0)]),
                    LineString([(60, 0), (60, 40)]),
                    LineString([(40, 0), (60, 0)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t0"],
                "names": [{"primary": "MAIN STREET"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        results = [
            MatchResult(
                "r1",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.0,
                local_end_frac=0.4,
            ),
            MatchResult(
                "r2",
                "t0",
                MatchDecision.MATCH,
                0.85,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.5,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in optimized] == [("r1", "t0")]

    def test_alignment_rescue_long_gap_requires_real_connector(self):
        """Parallel streets cannot bridge a lateral step without intervening topology."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2"],
                "names": [{"primary": "Main Street"}, {"primary": "Main Street"}],
                "geometry": [
                    LineString([(0, 0), (40, 0)]),
                    LineString([(80, 20), (40, 20)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t0"],
                "names": [{"primary": "MAIN STREET"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        results = [
            MatchResult(
                "r1",
                "t0",
                MatchDecision.MATCH,
                0.95,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.0,
                local_end_frac=0.4,
            ),
            MatchResult(
                "r2",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.5,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            reference,
            target,
            min_confidence=0.5,
            contiguity_tolerance=1.0,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in optimized] == [("r1", "t0")]

    def test_alignment_rescue_rejects_ambiguous_transitive_attachment(self):
        """Two duplicate tail spans must not attach to a prefix by ID order."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2", "r3", "connector_2", "connector_3"],
                "names": [
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Cross Street"},
                    {"primary": "Cross Street"},
                ],
                "geometry": [
                    LineString([(0, 0), (40, 0)]),
                    LineString([(60, 10), (100, 10)]),
                    LineString([(60, -10), (100, -10)]),
                    LineString([(40, 0), (60, 10)]),
                    LineString([(40, 0), (60, -10)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t0"],
                "names": [{"primary": "MAIN STREET"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        results = [
            MatchResult(
                ref_id,
                "t0",
                MatchDecision.MATCH,
                confidence,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=span[0],
                local_end_frac=span[1],
            )
            for ref_id, confidence, span in (
                ("r1", 0.95, (0.0, 0.4)),
                ("r2", 0.9, (0.5, 1.0)),
                ("r3", 0.85, (0.5, 1.0)),
            )
        ]

        optimized = optimize_matches_with_grouping(
            results,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
        )

        assert [(result.ref_id, result.target_id) for result in optimized] == [("r1", "t0")]

    def test_alignment_rescue_accepts_valid_transitive_corridor(self):
        """Whole-component validation still permits a complementary A-B-C chain."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2", "r3", "connector_12", "connector_23"],
                "names": [
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Main Street"},
                    {"primary": "Cross Street"},
                    {"primary": "Cross Street"},
                ],
                "geometry": [
                    LineString([(0, 0), (30, 0)]),
                    LineString([(40, 0), (70, 0)]),
                    LineString([(80, 0), (110, 0)]),
                    LineString([(30, 0), (40, 0)]),
                    LineString([(70, 0), (80, 0)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["t0"],
                "names": [{"primary": "MAIN STREET"}],
                "geometry": [LineString([(0, 2), (100, 2)])],
            },
            crs="EPSG:32619",
        )
        results = [
            MatchResult(
                ref_id,
                "t0",
                MatchDecision.MATCH,
                confidence,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=span[0],
                local_end_frac=span[1],
            )
            for ref_id, confidence, span in (
                ("r1", 0.95, (0.0, 0.3)),
                ("r2", 0.9, (0.4, 0.7)),
                ("r3", 0.85, (0.8, 1.0)),
            )
        ]

        optimized = optimize_matches_with_grouping(
            results,
            reference,
            target,
            min_confidence=0.5,
            corridor_aware=True,
        )

        assert {result.ref_id for result in optimized} == {"r1", "r2", "r3"}
        assert {result.features["match_type"] for result in optimized} == {"N:1"}
        assert len({result.features["group_id"] for result in optimized}) == 1

    def test_detached_alignment_singleton_is_not_added_without_rescue_evidence(self):
        """Non-overlap alone cannot attach a detached, unnamed singleton."""
        ref_geoms = {"r0": LineString([(0, 5), (100, 5)])}
        target_geoms = {
            "t1": LineString([(0, 0), (30, 0)]),
            "t2": LineString([(30, 0), (60, 0)]),
            "t3": LineString([(80, 0), (100, 0)]),
        }
        results = [
            MatchResult(
                "r0",
                target_id,
                MatchDecision.MATCH,
                confidence,
                {},
                {},
                gers_start_frac=span[0],
                gers_end_frac=span[1],
            )
            for target_id, confidence, span in (
                ("t1", 0.95, (0.0, 0.3)),
                ("t2", 0.9, (0.3, 0.6)),
                ("t3", 0.6, (0.8, 1.0)),
            )
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            corridor_aware=True,
        )
        by_target = {result.target_id: result for result in optimized}

        assert set(by_target) == {"t1", "t2"}

    def test_target_side_coverage_overlap_demotes_lower_confidence(self):
        """Two refs cannot both claim the same portion of one target."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (100, 0)]),
        }
        target_geoms = {"t0": LineString([(0, 5), (100, 5)])}
        results = [
            MatchResult(
                "r1",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                local_start_frac=0.0,
                local_end_frac=0.75,
            ),
            MatchResult(
                "r2",
                "t0",
                MatchDecision.MATCH,
                0.8,
                {},
                {},
                local_start_frac=0.25,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
        )
        by_ref = {result.ref_id: result for result in optimized}
        assert by_ref["r1"].decision == MatchDecision.MATCH
        assert by_ref["r2"].decision == MatchDecision.REVIEW
        assert by_ref["r2"].features["target_coverage_conflict"] == 1.0

    @pytest.mark.parametrize("side", ["ref", "target"])
    def test_coverage_review_does_not_cascade_into_compatible_match(self, side):
        """A demoted middle span cannot block a compatible lower-ranked tail."""
        spans = (("a", 0.9, 0.0, 0.4), ("b", 0.8, 0.3, 0.7), ("c", 0.7, 0.6, 1.0))
        if side == "target":
            results = [
                MatchResult(
                    name,
                    "shared",
                    MatchDecision.MATCH,
                    confidence,
                    {},
                    {},
                    local_start_frac=start,
                    local_end_frac=end,
                )
                for name, confidence, start, end in spans
            ]
            ref_geoms = {
                name: LineString([(0, index), (100, index)])
                for index, (name, _, _, _) in enumerate(spans)
            }
            target_geoms = {"shared": LineString([(0, 0), (100, 0)])}
            key_attr = "ref_id"
        else:
            results = [
                MatchResult(
                    "shared",
                    name,
                    MatchDecision.MATCH,
                    confidence,
                    {},
                    {},
                    gers_start_frac=start,
                    gers_end_frac=end,
                )
                for name, confidence, start, end in spans
            ]
            ref_geoms = {"shared": LineString([(0, 0), (100, 0)])}
            target_geoms = {
                name: LineString([(0, index), (100, index)])
                for index, (name, _, _, _) in enumerate(spans)
            }
            key_attr = "target_id"

        validated = _validate_assignment_coverage(results, ref_geoms, target_geoms)
        decisions = {getattr(result, key_attr): result.decision for result in validated}

        assert decisions == {
            "a": MatchDecision.MATCH,
            "b": MatchDecision.REVIEW,
            "c": MatchDecision.MATCH,
        }
        middle = next(result for result in validated if getattr(result, key_attr) == "b")
        assert middle.features[f"{side}_coverage_conflict"] == 1.0

    def test_equal_confidence_ref_coverage_tie_is_input_order_invariant(self):
        """Canonical target ID, not row order, wins an equal-score ref conflict."""
        ref_geoms = {"r0": LineString([(0, 0), (100, 0)])}
        target_geoms = {
            "t_a": LineString([(0, 5), (75, 5)]),
            "t_b": LineString([(25, 5), (100, 5)]),
        }
        candidates = [
            MatchResult(
                "r0",
                "t_a",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=0.75,
            ),
            MatchResult(
                "r0",
                "t_b",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.25,
                gers_end_frac=1.0,
            ),
        ]

        outcomes = []
        for ordered in (candidates, list(reversed(candidates))):
            validated = _validate_assignment_coverage(ordered, ref_geoms, target_geoms)
            outcomes.append({result.target_id: result.decision for result in validated})

        assert (
            outcomes[0]
            == outcomes[1]
            == {
                "t_a": MatchDecision.MATCH,
                "t_b": MatchDecision.REVIEW,
            }
        )

    def test_equal_confidence_target_coverage_tie_is_input_order_invariant(self):
        """Canonical ref ID, not row order, wins an equal-score target conflict."""
        ref_geoms = {
            "r_a": LineString([(0, 0), (75, 0)]),
            "r_b": LineString([(25, 0), (100, 0)]),
        }
        target_geoms = {"t0": LineString([(0, 5), (100, 5)])}
        candidates = [
            MatchResult(
                "r_a",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                local_start_frac=0.0,
                local_end_frac=0.75,
            ),
            MatchResult(
                "r_b",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                local_start_frac=0.25,
                local_end_frac=1.0,
            ),
        ]

        outcomes = []
        for ordered in (candidates, list(reversed(candidates))):
            validated = _validate_assignment_coverage(ordered, ref_geoms, target_geoms)
            outcomes.append({result.ref_id: result.decision for result in validated})

        assert (
            outcomes[0]
            == outcomes[1]
            == {
                "r_a": MatchDecision.MATCH,
                "r_b": MatchDecision.REVIEW,
            }
        )

    def test_target_side_touching_coverage_is_not_a_conflict(self):
        """Adjacent target intervals that only touch remain valid N:1 assignments."""
        ref_geoms = {
            "r1": LineString([(0, 0), (50, 0)]),
            "r2": LineString([(50, 0), (100, 0)]),
        }
        target_geoms = {"t0": LineString([(0, 5), (100, 5)])}
        results = [
            MatchResult(
                "r1",
                "t0",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                local_start_frac=0.0,
                local_end_frac=0.5,
            ),
            MatchResult(
                "r2",
                "t0",
                MatchDecision.MATCH,
                0.8,
                {},
                {},
                local_start_frac=0.5,
                local_end_frac=1.0,
            ),
        ]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
        )
        assert len(optimized) == 2
        assert all(result.decision == MatchDecision.MATCH for result in optimized)

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


class TestSliverFreeComponentGraph:
    """Junction slivers must not glue independent components together.

    Sliver classification uses the shared hybrid rule from ``crosswalk.config``
    (fraction < 0.10 AND absolute overlap < 5 m). Segments here are 100 m long,
    so a 0.01 alignment span = 1 m of overlap: comfortably a sliver.
    """

    def _make_gdf(self, id_col, geom_dict):
        import geopandas as gpd

        return gpd.GeoDataFrame(
            {id_col: list(geom_dict.keys()), "geometry": list(geom_dict.values())},
            crs="EPSG:32610",
        )

    def _full_edge(self, rid, tid, conf):
        return MatchResult(
            rid,
            tid,
            MatchDecision.MATCH,
            conf,
            {},
            {},
            gers_start_frac=0.0,
            gers_end_frac=1.0,
            local_start_frac=0.0,
            local_end_frac=1.0,
        )

    def _sliver_edge(self, rid, tid, conf=0.3):
        # ~1 m of a 100 m ref, ~0.5 m of a 100 m target: sliver under the
        # hybrid rule (max span 0.01 < 0.10 AND max abs overlap 1 m < 5 m).
        return MatchResult(
            rid,
            tid,
            MatchDecision.REVIEW,
            conf,
            {},
            {},
            gers_start_frac=0.99,
            gers_end_frac=1.0,
            local_start_frac=0.0,
            local_end_frac=0.005,
        )

    def _two_island_fixture(self):
        """Two independent 1:N islands joined ONLY by a sliver edge r1->t2a."""
        ref_geoms = {
            "r1": LineString([(0, 0), (100, 0)]),
            "r2": LineString([(200, 0), (300, 0)]),
        }
        target_geoms = {
            "t1a": LineString([(0, 5), (50, 5)]),
            "t1b": LineString([(50, 5), (100, 5)]),
            "t2a": LineString([(200, 5), (250, 5)]),
            "t2b": LineString([(250, 5), (300, 5)]),
        }
        results = [
            self._full_edge("r1", "t1a", 0.9),
            self._full_edge("r1", "t1b", 0.85),
            self._full_edge("r2", "t2a", 0.88),
            self._full_edge("r2", "t2b", 0.8),
            self._sliver_edge("r1", "t2a", 0.6),  # the junction kiss
        ]
        return (
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            results,
        )

    def test_sliver_does_not_glue_components(self):
        """Two islands joined only by a sliver -> two groups, not one."""
        ref_gdf, target_gdf, results = self._two_island_fixture()

        optimized = optimize_matches_with_grouping(
            results, ref_gdf, target_gdf, min_confidence=0.5, contiguity_tolerance=5.0
        )

        # The sliver edge is never selected into an assignment.
        pairs = {(r.ref_id, r.target_id) for r in optimized}
        assert ("r1", "t2a") not in pairs
        assert pairs == {("r1", "t1a"), ("r1", "t1b"), ("r2", "t2a"), ("r2", "t2b")}

        # Two distinct 1:N groups, not one welded M:N monster.
        group_ids = {r.features["group_id"] for r in optimized}
        assert len(group_ids) == 2
        for r in optimized:
            assert r.features["match_type"] == "1:N"

    def test_find_components_drops_cross_component_sliver(self):
        """A sliver whose endpoints land in different components is dropped."""
        _, _, results = self._two_island_fixture()
        sliver = {("r1", "t2a")}

        components = find_match_components(results, min_confidence=0.5, sliver_edges=sliver)

        assert len(components) == 2
        all_pairs = {(r.ref_id, r.target_id) for comp in components for r in comp}
        assert ("r1", "t2a") not in all_pairs

    def test_same_component_sliver_stays_in_group_edges(self):
        """A sliver whose endpoints share a component stays as a group edge."""
        results = [
            self._full_edge("r1", "t1", 0.9),
            self._full_edge("r1", "t2", 0.85),
            self._full_edge("r2", "t2", 0.8),
            self._sliver_edge("r2", "t1", 0.4),  # both endpoints in the component
        ]
        components = find_match_components(results, min_confidence=0.3, sliver_edges={("r2", "t1")})
        assert len(components) == 1
        pairs = {(r.ref_id, r.target_id) for r in components[0]}
        assert ("r2", "t1") in pairs  # attached as an ordinary group member

    def test_sliver_only_pair_never_matched(self):
        """A pair connected ONLY by a sliver edge produces no match at all."""
        ref_geoms = {"r1": LineString([(0, 0), (100, 0)])}
        target_geoms = {"t1": LineString([(99, 5), (199, 5)])}
        results = [self._sliver_edge("r1", "t1", 0.9)]

        optimized = optimize_matches_with_grouping(
            results,
            self._make_gdf("id", ref_geoms),
            self._make_gdf("local_id", target_geoms),
            min_confidence=0.5,
            contiguity_tolerance=5.0,
        )
        assert optimized == []

    def test_no_sliver_param_keeps_old_behavior(self):
        """find_match_components without sliver_edges is unchanged: the sliver
        edge glues both islands into one component."""
        _, _, results = self._two_island_fixture()
        components = find_match_components(results, min_confidence=0.5)
        assert len(components) == 1


class TestCollinearityGate:
    """Corridor-aware contiguity: only collinear continuations / same-name
    segments chain together at a shared endpoint."""

    def _turned(self, deg: float) -> LineString:
        """Segment starting at the shared point (10,0), turned `deg` from straight."""
        # A collinear continuation would leave (10,0) in the +x direction.
        rad = math.radians(deg)
        return LineString([(10.0, 0.0), (10.0 + 10 * math.cos(rad), 10 * math.sin(rad))])

    def test_straight_continuation_is_collinear(self):
        a = LineString([(0, 0), (10, 0)])
        adj = build_contiguity_adjacency(
            ["a", "b"], {"a": a, "b": self._turned(0)}, 1.0, require_collinear=True, max_turn_deg=40
        )
        assert adj["a"] == {"b"}

    def test_perpendicular_junction_not_collinear(self):
        a = LineString([(0, 0), (10, 0)])
        adj = build_contiguity_adjacency(
            ["a", "b"],
            {"a": a, "b": self._turned(90)},
            1.0,
            require_collinear=True,
            max_turn_deg=40,
        )
        assert adj["a"] == set()

    def test_angle_boundary_inclusive(self):
        a = LineString([(0, 0), (10, 0)])
        # turn == 40 is within the gate (<=), turn == 41 is outside.
        geoms = {"a": a, "b40": self._turned(40), "b41": self._turned(41)}
        adj = build_contiguity_adjacency(
            ["a", "b40", "b41"], geoms, 1.0, require_collinear=True, max_turn_deg=40
        )
        assert "b40" in adj["a"]
        assert "b41" not in adj["a"]

    def test_same_name_rescues_sharp_turn(self):
        a = LineString([(0, 0), (10, 0)])
        geoms = {"a": a, "b": self._turned(90)}
        names = {"a": "Beacon St", "b": "beacon st"}  # normalized-equal
        adj = build_contiguity_adjacency(
            ["a", "b"], geoms, 1.0, require_collinear=True, max_turn_deg=40, name_lookup=names
        )
        assert adj["a"] == {"b"}

    def test_unnamed_sharp_turn_not_rescued(self):
        a = LineString([(0, 0), (10, 0)])
        geoms = {"a": a, "b": self._turned(90)}
        names = {"a": "", "b": ""}  # empty names must not match
        adj = build_contiguity_adjacency(
            ["a", "b"], geoms, 1.0, require_collinear=True, max_turn_deg=40, name_lookup=names
        )
        assert adj["a"] == set()

    def test_endpoints_are_collinear_helper(self):
        # straight through the shared point (10,0)
        assert _endpoints_are_collinear((10, 0), (0, 0), (10, 0), (20, 0), 40)
        # right-angle turn
        assert not _endpoints_are_collinear((10, 0), (0, 0), (10, 0), (10, 10), 40)


class TestGroupingOnlyConfidencePrune:
    """Weak edges below glue_min_confidence do not weld components, but stay as
    scored candidates when their endpoints co-land via stronger edges."""

    def _mr(self, rid, tid, conf):
        return MatchResult(rid, tid, MatchDecision.MATCH, conf, {}, {})

    def test_weak_edge_does_not_glue_independent_components(self):
        # Two strong 1:1 pairs; only a weak edge links them.
        results = [
            self._mr("r1", "t1", 0.9),
            self._mr("r2", "t2", 0.9),
            self._mr("r1", "t2", 0.3),  # weak cross-link
        ]
        comps = find_match_components(results, min_confidence=0.1, glue_min_confidence=0.5)
        assert len(comps) == 2  # weak edge did NOT glue them
        # The weak edge, being cross-component, is dropped from grouping.
        all_pairs = {(r.ref_id, r.target_id) for c in comps for r in c}
        assert ("r1", "t2") not in all_pairs

    def test_weak_edge_retained_when_endpoints_coland(self):
        # Strong edges already put r1, r2, t1, t2 in one component; the weak
        # r1-t2 edge then stays as an in-component candidate.
        results = [
            self._mr("r1", "t1", 0.9),
            self._mr("r2", "t1", 0.9),
            self._mr("r2", "t2", 0.9),
            self._mr("r1", "t2", 0.3),  # weak, but endpoints co-land
        ]
        comps = find_match_components(results, min_confidence=0.1, glue_min_confidence=0.5)
        assert len(comps) == 1
        pairs = {(r.ref_id, r.target_id) for r in comps[0]}
        assert ("r1", "t2") in pairs  # retained as scored candidate

    def test_default_no_prune_matches_legacy(self):
        results = [
            self._mr("r1", "t1", 0.9),
            self._mr("r2", "t2", 0.9),
            self._mr("r1", "t2", 0.3),
        ]
        # Without glue_min_confidence, the weak edge glues into one component.
        comps = find_match_components(results, min_confidence=0.1)
        assert len(comps) == 1


class TestGlueMinConfidenceCalibratedOperatingPoint:
    """The grouping-only glue prune consumes ``MatchResult.confidence``, which is
    calibrated P(match) when ``enable_calibration`` is True. The prune constant
    is therefore the calibrated image of the raw-0.5 point the #267 corridor
    design validated (isotonic maps mid-range raw 0.5 -> ~0.575), so the effective
    prune population is preserved under calibration."""

    def _mr(self, rid, tid, conf):
        return MatchResult(rid, tid, MatchDecision.MATCH, conf, {}, {})

    def test_default_is_calibrated_equivalent_of_raw_half(self):
        from crosswalk.config import settings

        # Guard against a silent revert to the raw-scale 0.5: with calibration on
        # (the default), 0.5 would prune at an effective raw ~0.42 (weaker glue).
        assert settings.optimizer_glue_min_confidence == pytest.approx(0.575)
        assert settings.optimizer_glue_min_confidence > settings.scoring_match_threshold

    def test_configured_default_prunes_the_calibrated_band(self):
        from crosswalk.config import settings

        # An edge whose calibrated confidence sits in the [0.5, 0.575) band --
        # i.e. it maps below the raw-0.5 the design pruned -- must NOT glue at
        # the configured default (which optimize_matches_with_grouping passes in).
        assert settings.optimizer_glue_min_confidence > 0.55
        results = [
            self._mr("r1", "t1", 0.9),
            self._mr("r2", "t2", 0.9),
            self._mr("r1", "t2", 0.55),  # calibrated conf in the (0.5, 0.575) band
        ]
        comps = find_match_components(
            results,
            min_confidence=0.1,
            glue_min_confidence=settings.optimizer_glue_min_confidence,
        )
        assert len(comps) == 2  # 0.55 edge does not weld at the 0.575 default

    def test_band_edge_would_glue_at_old_raw_half(self):
        # The same 0.55 edge WOULD have glued under the old raw-scale 0.5 default,
        # confirming the operating-point shift is real (not a no-op relabel).
        results = [
            self._mr("r1", "t1", 0.9),
            self._mr("r2", "t2", 0.9),
            self._mr("r1", "t2", 0.55),
        ]
        comps = find_match_components(results, min_confidence=0.1, glue_min_confidence=0.5)
        assert len(comps) == 1

    def test_pipeline_selects_glue_by_calibration_state(self, monkeypatch):
        # The pipeline keys the glue prune off whether the loaded model actually
        # applies calibration: calibrated -> 0.575, raw -> 0.5. This keeps an
        # uncalibrated model at the raw-0.5 point #267 validated (no over-prune).
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        class _FakeMatcher:
            def __init__(self, *a, **k):
                pass

        _FakeMatcher.calibration_active = property(lambda self: self._active)

        def make(active):
            fm = _FakeMatcher()
            fm._active = active
            return fm

        # The helper imports MLMatcher lazily from crosswalk.matching.ml, so patch
        # it at the source module (patching runner.MLMatcher would be a no-op).
        import crosswalk.matching.ml as ml_mod

        monkeypatch.setattr(ml_mod, "MLMatcher", lambda *a, **k: make(True))
        assert runner._effective_glue_min_confidence() == pytest.approx(
            settings.optimizer_glue_min_confidence
        )

        monkeypatch.setattr(ml_mod, "MLMatcher", lambda *a, **k: make(False))
        assert runner._effective_glue_min_confidence() == pytest.approx(
            settings.optimizer_glue_min_confidence_raw
        )

    def test_pipeline_short_circuits_when_calibration_disabled(self, monkeypatch):
        # With calibration globally off, the helper returns the raw threshold
        # WITHOUT loading a model (calibration_active can never be True).
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", False)

        import crosswalk.matching.ml as ml_mod

        def _boom(*a, **k):
            raise AssertionError("MLMatcher must not be loaded when calibration is disabled")

        monkeypatch.setattr(ml_mod, "MLMatcher", _boom)
        assert runner._effective_glue_min_confidence() == pytest.approx(
            settings.optimizer_glue_min_confidence_raw
        )


class TestEffectivePruneThreshold:
    """Resolution of the resolver confidence-drop prune floor (#282/#284/#348).

    The allowlist is keyed on DATASET IDENTITY — the dataset name the runner is
    told (``crosswalk stitch`` dataset argument / factory pair name) — NEVER on
    anything derived from the output path (#348: the old filename-stem
    resolution silently skipped pruning for nonstandard output names). It was
    also validated ONLY on CALIBRATED confidence, so the prune is skipped when
    the active model applies no calibration (else it silently over-prunes).
    """

    def _patch_calibration(self, monkeypatch, active: bool):
        import crosswalk.matching.ml as ml_mod

        class _FakeMatcher:
            def __init__(self, *a, **k):
                pass

            @property
            def calibration_active(self):
                return active

        monkeypatch.setattr(ml_mod, "MLMatcher", lambda *a, **k: _FakeMatcher())

    def test_allowlisted_dataset_applies_its_threshold(self, monkeypatch):
        """A dataset present in the allowlist prunes at its validated floor."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(
            settings,
            "resolver_prune_overrides",
            {"us_boston_streets": 0.96, "us_seattle_sidewalks": 0.90},
        )
        self._patch_calibration(monkeypatch, active=True)

        assert runner._effective_prune_threshold("us_boston_streets") == pytest.approx(0.96)
        assert runner._effective_prune_threshold("us_seattle_sidewalks") == pytest.approx(0.90)

    def test_non_allowlisted_dataset_is_off(self, monkeypatch):
        """A dataset ABSENT from the allowlist is never pruned (opt-in only) —
        no global default floor is applied, so it returns 0.0."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(
            settings,
            "resolver_prune_overrides",
            {"us_boston_streets": 0.96, "us_seattle_sidewalks": 0.90},
        )
        self._patch_calibration(monkeypatch, active=True)

        # de_berlin_streets is not in the allowlist -> prune off.
        assert runner._effective_prune_threshold("de_berlin_streets") == 0.0

    def test_no_dataset_identity_is_off_and_logged(self, monkeypatch):
        """dataset_key=None (raw -r/-t path mode, no dataset name) -> prune off,
        WITHOUT loading a model, and a log line says why (never silent)."""
        import io

        from loguru import logger

        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})

        import crosswalk.matching.ml as ml_mod

        def _boom(*a, **k):
            raise AssertionError("MLMatcher must not be loaded without a dataset identity")

        monkeypatch.setattr(ml_mod, "MLMatcher", _boom)

        sink = io.StringIO()
        handler_id = logger.add(sink, format="{message}", level="INFO")
        try:
            assert runner._effective_prune_threshold(None) == 0.0
        finally:
            logger.remove(handler_id)
        assert "no dataset identity" in sink.getvalue()

    def test_master_switch_off_disables_all(self, monkeypatch):
        """resolver_prune_enabled=False turns the prune off for every dataset,
        even allowlisted ones — WITHOUT loading a model."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", False)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})

        import crosswalk.matching.ml as ml_mod

        def _boom(*a, **k):
            raise AssertionError("MLMatcher must not be loaded when the master switch is off")

        monkeypatch.setattr(ml_mod, "MLMatcher", _boom)
        assert runner._effective_prune_threshold("us_boston_streets") == 0.0

    def test_override_le_zero_disables_regardless(self, monkeypatch):
        """An allowlist value <= 0 keeps a listed dataset explicitly disabled."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"ds": 0.0})
        self._patch_calibration(monkeypatch, active=True)

        assert runner._effective_prune_threshold("ds") == 0.0

    def test_prune_skipped_when_calibration_globally_disabled(self, monkeypatch):
        """enable_calibration=False must skip the prune WITHOUT loading a model:
        the calibrated-tuned floor is invalid on raw scores. Uses an allowlisted
        dataset so the calibration guard (not the allowlist) is what disables it."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", False)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})

        import crosswalk.matching.ml as ml_mod

        def _boom(*a, **k):
            raise AssertionError("MLMatcher must not be loaded when calibration is disabled")

        monkeypatch.setattr(ml_mod, "MLMatcher", _boom)
        assert runner._effective_prune_threshold("us_boston_streets") == 0.0

    def test_prune_skipped_when_model_not_calibrated(self, monkeypatch):
        """Calibration enabled globally but the loaded model carries no
        calibrator (calibration_active False) -> prune must be skipped even for
        an allowlisted dataset."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(
            settings,
            "resolver_prune_overrides",
            {"us_boston_streets": 0.96, "us_seattle_sidewalks": 0.90},
        )
        self._patch_calibration(monkeypatch, active=False)

        # allowlisted datasets are still skipped when the model is uncalibrated
        assert runner._effective_prune_threshold("us_boston_streets") == 0.0
        assert runner._effective_prune_threshold("us_seattle_sidewalks") == 0.0

    def test_exact_key_only_no_substring_collision(self, monkeypatch):
        """Only exact keys count: a hypothetical ``us_boston_streets_2`` (no
        allowlist entry) must NOT resolve to the ``us_boston_streets`` override;
        with its own entry, its own value wins."""
        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})
        self._patch_calibration(monkeypatch, active=True)

        assert runner._effective_prune_threshold("us_boston_streets_2") == 0.0

        monkeypatch.setattr(
            settings,
            "resolver_prune_overrides",
            {"us_boston_streets": 0.96, "us_boston_streets_2": 0.88},
        )
        assert runner._effective_prune_threshold("us_boston_streets_2") == pytest.approx(0.88)
        assert runner._effective_prune_threshold("us_boston_streets") == pytest.approx(0.96)

    def test_enabled_run_logs_dataset_and_threshold(self, monkeypatch):
        """When the prune is ON, a loud log line names the dataset and its
        threshold — the run's prune state must never be silent (#348)."""
        import io

        from loguru import logger

        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})
        self._patch_calibration(monkeypatch, active=True)

        sink = io.StringIO()
        handler_id = logger.add(sink, format="{message}", level="INFO")
        try:
            assert runner._effective_prune_threshold("us_boston_streets") == pytest.approx(0.96)
        finally:
            logger.remove(handler_id)
        log_text = sink.getvalue()
        assert "prune ON" in log_text
        assert "us_boston_streets" in log_text
        assert "0.96" in log_text

    def test_non_allowlisted_dataset_logs_true_name(self, monkeypatch):
        """A non-allowlisted dataset stays OFF (correct) and the skip log names
        the true dataset — e.g. the factory's ``pair.name`` for its dataset-blind
        ``…/dataset=<name>/bridge.parquet`` outputs, which are otherwise
        indistinguishable across a multi-dataset sweep."""
        import io

        from loguru import logger

        from crosswalk.config import settings
        from crosswalk.pipeline import runner

        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})
        self._patch_calibration(monkeypatch, active=True)

        sink = io.StringIO()
        handler_id = logger.add(sink, format="{message}", level="INFO")
        try:
            assert runner._effective_prune_threshold("co_bogota_bike_network") == 0.0
        finally:
            logger.remove(handler_id)

        log_text = sink.getvalue()
        assert "co_bogota_bike_network" in log_text
        assert "bridge.parquet" not in log_text


class TestStructuralGate:
    def test_single_corridor_always_simple_within_backstop(self):
        assert group_is_structurally_simple(1, 1, 30, 2, 30, 40)
        assert group_is_structurally_simple(1, 5, 35, 2, 30, 40)  # single corridor, long

    def test_backstop_blocks_everything(self):
        assert not group_is_structurally_simple(1, 1, 41, 2, 30, 40)

    def test_multi_corridor_needs_few_components_and_soft_budget(self):
        assert group_is_structurally_simple(2, 2, 25, 2, 30, 40)
        assert not group_is_structurally_simple(3, 3, 10, 2, 30, 40)  # tangle
        assert not group_is_structurally_simple(2, 2, 31, 2, 30, 40)  # over soft budget


def _grp(rid, tid, conf, gid):
    """MatchResult helper: a group edge carries a group_id in features."""
    return MatchResult(rid, tid, MatchDecision.MATCH, conf, {}, {"group_id": gid} if gid else {})


class TestConfidenceDropPrune:
    """Tests for the M2 / resolver Phase-1 confidence-drop prune."""

    def test_disabled_is_identity_object(self):
        """threshold <= 0 returns the SAME list object and empty pruned set."""
        rs = [_grp("r1", "t1", 0.9, "g1"), _grp("r1", "t2", 0.2, "g1")]
        kept, pruned = apply_confidence_drop_prune(rs, 0.0)
        assert kept is rs
        assert pruned == set()

    def test_no_op_when_nothing_below_threshold(self):
        """When all group edges clear the floor, the input is returned unchanged."""
        rs = [_grp("r1", "t1", 0.99, "g1"), _grp("r2", "t2", 0.98, "g1")]
        kept, pruned = apply_confidence_drop_prune(rs, 0.5)
        assert kept is rs
        assert pruned == set()

    def test_drops_below_threshold(self):
        rs = [
            _grp("r1", "t1", 0.99, "g1"),
            _grp("r1", "t2", 0.40, "g1"),
            _grp("r2", "t3", 0.20, "g1"),
        ]
        kept, pruned = apply_confidence_drop_prune(rs, 0.5)
        assert pruned == {("r1", "t2"), ("r2", "t3")}
        assert {(r.ref_id, r.target_id) for r in kept} == {("r1", "t1")}

    def test_retains_group_top_even_if_below_threshold(self):
        """A group is never emptied: its highest-confidence edge always survives."""
        rs = [_grp("r1", "t1", 0.30, "g1"), _grp("r1", "t2", 0.10, "g1")]
        kept, pruned = apply_confidence_drop_prune(rs, 0.9)
        # top (0.30) kept despite being < 0.9; only the 0.10 edge dropped
        assert {(r.ref_id, r.target_id) for r in kept} == {("r1", "t1")}
        assert pruned == {("r1", "t2")}

    def test_low_confidence_top_left_as_singleton_is_review(self):
        """The never-empty backstop retains a weak survivor without auto-accepting it."""
        rs = [_grp("r1", "t1", 0.68, "g1"), _grp("r1", "t2", 0.51, "g1")]

        kept, pruned = apply_confidence_drop_prune(rs, 0.96)

        assert pruned == {("r1", "t2")}
        assert len(kept) == 1
        assert kept[0].decision == MatchDecision.REVIEW
        assert kept[0].features[PRUNED_SINGLETON_REVIEW_FLAG] == 1.0

    def test_exact_name_orphan_in_review_band_survives_as_review(self):
        """Pruning preserves the human-labeled Carver-shaped orphan, not its cross-link."""
        exact_name_features = {
            "group_id": "g1",
            "has_name_ref": 1.0,
            "has_name_target": 1.0,
            "name_is_generic": 0.0,
            "name_levenshtein": 1.0,
            "name_jaro_winkler": 1.0,
            "name_token_sort": 1.0,
            "name_soundex": 1.0,
            "name_metaphone": 1.0,
        }
        correct = MatchResult(
            "carver_ref",
            "carver_target",
            MatchDecision.MATCH,
            0.515528,
            {},
            exact_name_features,
            gers_start_frac=0.27,
            gers_end_frac=0.997,
            local_start_frac=0.0,
            local_end_frac=1.0,
        )
        wrong_cross_link = MatchResult(
            "carver_ref",
            "townsend_target",
            MatchDecision.MATCH,
            0.682759,
            {},
            {
                **exact_name_features,
                "name_levenshtein": 0.22,
                "name_jaro_winkler": 0.45,
                "name_token_sort": 0.37,
                "name_soundex": 0.0,
                "name_metaphone": 0.22,
            },
            gers_start_frac=0.0,
            gers_end_frac=0.30,
            local_start_frac=0.42,
            local_end_frac=1.0,
        )
        service_survivor = _grp("service_ref", "townsend_target", 0.730994, "g1")

        kept, pruned = apply_confidence_drop_prune(
            [correct, wrong_cross_link, service_survivor],
            0.96,
        )
        by_pair = {(result.ref_id, result.target_id): result for result in kept}

        assert set(by_pair) == {
            ("carver_ref", "carver_target"),
            ("service_ref", "townsend_target"),
        }
        assert pruned == {("carver_ref", "townsend_target")}
        assert all(result.decision == MatchDecision.REVIEW for result in kept)
        assert (
            by_pair[("carver_ref", "carver_target")].features[PRUNED_SINGLETON_REVIEW_FLAG] == 1.0
        )
        assert (
            by_pair[("service_ref", "townsend_target")].features[PRUNED_SINGLETON_REVIEW_FLAG]
            == 1.0
        )

    def test_orphan_rescue_does_not_override_match_band_prune_policy(self):
        """Exact-name orphan edges above the REVIEW band remain ordinary prunes."""
        orphan = MatchResult(
            "r_orphan",
            "t_orphan",
            MatchDecision.MATCH,
            0.85,
            {},
            {
                "group_id": "g1",
                "has_name_ref": 1.0,
                "has_name_target": 1.0,
                "name_is_generic": 0.0,
                "name_levenshtein": 1.0,
                "name_jaro_winkler": 1.0,
                "name_token_sort": 1.0,
                "name_soundex": 1.0,
                "name_metaphone": 1.0,
            },
            gers_start_frac=0.0,
            gers_end_frac=1.0,
            local_start_frac=0.0,
            local_end_frac=1.0,
        )
        survivor = _grp("r_kept", "t_kept", 0.9, "g1")

        kept, pruned = apply_confidence_drop_prune([orphan, survivor], 0.96)

        assert {(result.ref_id, result.target_id) for result in kept} == {("r_kept", "t_kept")}
        assert pruned == {("r_orphan", "t_orphan")}

    def test_never_touches_1to1_matches(self):
        """1:1 matches (no group_id) are outside the prune's scope."""
        rs = [
            _grp("rA", "tA", 0.10, None),  # 1:1, very low conf
            _grp("r1", "t1", 0.99, "g1"),
            _grp("r1", "t2", 0.20, "g1"),
        ]
        kept, pruned = apply_confidence_drop_prune(rs, 0.5)
        assert ("rA", "tA") not in pruned
        assert any(r.ref_id == "rA" for r in kept)
        assert pruned == {("r1", "t2")}

    def test_threshold_applied_per_group_independently(self):
        rs = [
            _grp("r1", "t1", 0.99, "g1"),
            _grp("r1", "t2", 0.40, "g1"),
            _grp("r3", "t4", 0.95, "g2"),
            _grp("r3", "t5", 0.30, "g2"),
        ]
        _, pruned = apply_confidence_drop_prune(rs, 0.5)
        assert pruned == {("r1", "t2"), ("r3", "t5")}


def _canonical_digest(optimized):
    """Order-sensitive fingerprint of an optimizer result list.

    Captures both WHICH edges were selected (and their decision/group) AND the
    order they are emitted in — the sidecar's group array order derives from
    this list order, so a determinism fix must stabilize both.
    """
    import hashlib

    h = hashlib.sha256()
    for r in optimized:
        gid = r.features.get("group_id", "")
        h.update(f"{r.ref_id}|{r.target_id}|{r.decision.value}|{gid}\n".encode())
    return h.hexdigest()


def _churn_scenario():
    """A grouping scenario whose selection is a hash-order tie-break trap.

    Each target has K non-contiguous candidate refs at IDENTICAL confidence.
    Non-contiguous refs decompose to singleton leftovers all competing for the
    one target, so the greedy assignment must break an exact confidence tie —
    historically resolved by Python set/dict iteration order (hash-seed and
    input-order dependent). Returns ``(results, ref_gdf, tgt_gdf)``.
    """
    import geopandas as gpd

    results = []
    ref_geoms = {}
    tgt_geoms = {}
    for c in range(12):
        tid = f"tgt_{c:03d}"
        tgt_geoms[tid] = LineString([(c * 1000, 0), (c * 1000 + 50, 0)])
        for k in range(4):
            rid = f"ref_{c:03d}_{k}"
            # far apart from each other -> NOT contiguous -> singleton leftovers
            ref_geoms[rid] = LineString(
                [(c * 1000 + k * 500, 5000 + k * 5000), (c * 1000 + k * 500 + 40, 5000 + k * 5000)]
            )
            results.append(MatchResult(rid, tid, MatchDecision.MATCH, 0.80, {}, {}))

    ref_gdf = gpd.GeoDataFrame(
        {"id": list(ref_geoms.keys())}, geometry=list(ref_geoms.values()), crs="EPSG:32619"
    )
    tgt_gdf = gpd.GeoDataFrame(
        {"local_id": list(tgt_geoms.keys())}, geometry=list(tgt_geoms.values()), crs="EPSG:32619"
    )
    return results, ref_gdf, tgt_gdf


class TestOptimizerDeterminism:
    """The optimizer output must be independent of Python hash-seed iteration
    order (set/dict) and of input list order. See PR: source-level determinism
    fix (sorted id dedup, canonical BFS neighbour order, explicit greedy
    tie-break) replacing the old PYTHONHASHSEED=0 workaround.
    """

    def test_greedy_tiebreak_invariant_to_input_order(self):
        """Equal-confidence edges competing for a shared target must resolve to
        the same canonical winner regardless of input order."""
        import random

        base = [
            MatchResult("ref_b", "t_x", MatchDecision.MATCH, 0.8, {}, {}),
            MatchResult("ref_a", "t_x", MatchDecision.MATCH, 0.8, {}, {}),
            MatchResult("ref_c", "t_x", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        winners = set()
        rng = random.Random(0)
        for _ in range(25):
            shuffled = base[:]
            rng.shuffle(shuffled)
            out = optimize_matches_greedy(shuffled, min_confidence=0.5)
            winners.add(tuple((r.ref_id, r.target_id) for r in out))
        # Exactly one canonical outcome across all permutations.
        assert len(winners) == 1
        # Canonical tie-break is smallest string id.
        assert winners == {(("ref_a", "t_x"),)}

    def test_grouping_invariant_to_input_order(self):
        """Full M:N grouping output must be byte-stable under input reordering."""
        import random

        results, ref_gdf, tgt_gdf = _churn_scenario()
        digests = set()
        rng = random.Random(1234)
        for _ in range(20):
            shuffled = results[:]
            rng.shuffle(shuffled)
            out = optimize_matches_with_grouping(
                shuffled,
                ref_gdf,
                tgt_gdf,
                min_confidence=0.5,
                ref_id_column="id",
                target_id_column="local_id",
            )
            digests.add(_canonical_digest(out))
        assert len(digests) == 1, f"input-order sensitive: {len(digests)} distinct outputs"

    def test_independent_group_component_order_is_canonical(self):
        """Independent group outputs cannot inherit first-candidate order."""
        import geopandas as gpd

        reference = gpd.GeoDataFrame(
            {
                "id": ["r_a", "r_b"],
                "geometry": [
                    LineString([(0, 5), (20, 5)]),
                    LineString([(100, 5), (120, 5)]),
                ],
            },
            crs="EPSG:32619",
        )
        target = gpd.GeoDataFrame(
            {
                "local_id": ["a1", "a2", "b1", "b2"],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(10, 0), (20, 0)]),
                    LineString([(100, 0), (110, 0)]),
                    LineString([(110, 0), (120, 0)]),
                ],
            },
            crs="EPSG:32619",
        )
        candidates = [
            MatchResult("r_a", "a1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r_a", "a2", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r_b", "b1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r_b", "b2", MatchDecision.MATCH, 0.9, {}, {}),
        ]

        outputs = []
        for ordered in (candidates, list(reversed(candidates))):
            optimized = optimize_matches_with_grouping(
                ordered,
                reference,
                target,
                min_confidence=0.5,
                contiguity_tolerance=1.0,
                corridor_aware=False,
            )
            outputs.append([(result.ref_id, result.target_id) for result in optimized])

        assert (
            outputs[0]
            == outputs[1]
            == [
                ("r_a", "a1"),
                ("r_a", "a2"),
                ("r_b", "b1"),
                ("r_b", "b2"),
            ]
        )

    @pytest.mark.slow
    def test_grouping_invariant_to_pythonhashseed(self):
        """The optimizer output must be identical across processes with
        different PYTHONHASHSEED — the honest cross-process determinism check.

        Runs the same tie-break-trap scenario in fresh subprocesses under
        several hash seeds and asserts a single canonical digest.
        """
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import hashlib
            import geopandas as gpd
            from shapely import LineString
            from crosswalk.matching.optimizer import optimize_matches_with_grouping
            from crosswalk.matching.types import MatchDecision, MatchResult

            results, ref_geoms, tgt_geoms = [], {}, {}
            for c in range(12):
                tid = f"tgt_{c:03d}"
                tgt_geoms[tid] = LineString([(c*1000, 0), (c*1000+50, 0)])
                for k in range(4):
                    rid = f"ref_{c:03d}_{k}"
                    ref_geoms[rid] = LineString(
                        [(c*1000+k*500, 5000+k*5000), (c*1000+k*500+40, 5000+k*5000)]
                    )
                    results.append(MatchResult(rid, tid, MatchDecision.MATCH, 0.80, {}, {}))
            ref_gdf = gpd.GeoDataFrame(
                {"id": list(ref_geoms)}, geometry=list(ref_geoms.values()), crs="EPSG:32619"
            )
            tgt_gdf = gpd.GeoDataFrame(
                {"local_id": list(tgt_geoms)}, geometry=list(tgt_geoms.values()), crs="EPSG:32619"
            )
            out = optimize_matches_with_grouping(
                results, ref_gdf, tgt_gdf, min_confidence=0.5,
                ref_id_column="id", target_id_column="local_id",
            )
            h = hashlib.sha256()
            for r in out:
                gid = r.features.get("group_id", "")
                h.update(f"{r.ref_id}|{r.target_id}|{r.decision.value}|{gid}\\n".encode())
            print(h.hexdigest())
            """
        )
        digests = set()
        for seed in ("0", "1", "2", "31337"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            digests.add(proc.stdout.strip())
        assert len(digests) == 1, f"hash-seed sensitive: {digests}"


def _edge(ref_id, target_id, conf, ref_span, tgt_span):
    """MatchResult with alignment fractions producing the given spans (from 0)."""
    return MatchResult(
        ref_id,
        target_id,
        MatchDecision.MATCH,
        conf,
        {},
        {},
        gers_start_frac=0.0,
        gers_end_frac=ref_span,
        local_start_frac=0.0,
        local_end_frac=tgt_span,
    )


def _len_geoms(edges, length=100.0):
    """`length`-meter LineString geoms for every ref/target id in ``edges``.

    The absolute-overlap gate reads ``geom.length``; 100 m segments make a
    ~10%-span stub ~10 m of overlap (below the 75 m gate), so stubs stay
    demotable while the fraction shape drives the test.
    """
    line = LineString([(0.0, 0.0), (0.0, length)])
    refs = {e.ref_id: line for e in edges}
    tgts = {e.target_id: line for e in edges}
    return refs, tgts


class TestContestedSmallSpanReviewDemotion:
    """#367 Mode A: contested small-span M:N stubs demote to REVIEW, not dropped."""

    def test_contested_stub_flagged_for_review(self):
        """A small-span edge contested on BOTH sides is flagged (193ac00f shape)."""
        # r_b covers ~88% of its ref to t_a (conf 1.0); r_a covers ~90% of t_b
        # (conf 0.998); the stub r_b->t_b covers only ~11%/8% of ref/target
        # (conf 0.97) and is contested on both its ref (r_b, 1.0) and target
        # (t_b, 0.998) — the parallel-sibling crossing stub.
        edges = [
            _edge("r_a", "t_b", 0.998, 0.986, 0.906),
            _edge("r_b", "t_a", 1.0, 0.881, 1.0),
            _edge("r_b", "t_b", 0.97, 0.108, 0.079),
        ]
        rg, tg = _len_geoms(edges)
        demote = _contested_small_span_review_pairs(edges, rg, tg)
        assert demote == {("r_b", "t_b")}

    def test_genuine_asymmetric_coverage_match_not_flagged(self):
        """A short ref fully consumed by a long target (tgt_span low, ref_span
        ~1.0) is NEVER flagged, even if contested — max(span) stays high."""
        edges = [
            # r_short fully covers itself into t_long (ref_span 1.0) but only a
            # small fraction of the long target — a legitimate asymmetric match.
            _edge("r_short", "t_long", 0.97, 1.0, 0.06),
            # a higher-confidence rival on the same target makes it "contested".
            _edge("r_other", "t_long", 1.0, 0.5, 0.9),
        ]
        rg, tg = _len_geoms(edges)
        assert _contested_small_span_review_pairs(edges, rg, tg) == set()

    def test_long_corridor_small_fraction_not_flagged(self):
        """A small-FRACTION edge whose ABSOLUTE overlap is large (long segment)
        is a genuine corridor edge, not a segmentation stub — exempt even when
        contested and both endpoints are anchored. The absolute-overlap gate is
        the only thing separating it from the demotable short-segment shape."""
        edges = [
            _edge("r_long", "t_anchor", 1.0, 0.9, 0.9),  # anchors r_long
            _edge("r_long", "t_long", 0.97, 0.2, 0.2),  # small fraction, under test
            _edge("r_other", "t_long", 1.0, 0.9, 0.9),  # higher-conf rival → contested
        ]
        # 2 km segments: 0.2 fraction = 400 m of real overlap, far above the 75 m
        # gate → exempt.
        rg_long, tg_long = _len_geoms(edges, length=2000.0)
        assert _contested_small_span_review_pairs(edges, rg_long, tg_long) == set()
        # Identical shape on 100 m segments: 0.2 * 100 = 20 m < 75 m gate, and both
        # endpoints are anchored, so it DOES demote — proving the gate is the cause.
        rg_short, tg_short = _len_geoms(edges, length=100.0)
        assert ("r_long", "t_long") in _contested_small_span_review_pairs(edges, rg_short, tg_short)

    def test_uncontested_small_span_not_flagged(self):
        """A small-span edge with no higher-confidence rival is left alone."""
        edges = [
            _edge("r_a", "t_a", 0.97, 0.1, 0.08),  # small but sole claimant
            _edge("r_b", "t_b", 0.99, 0.9, 0.9),
        ]
        rg, tg = _len_geoms(edges)
        assert _contested_small_span_review_pairs(edges, rg, tg) == set()

    def test_orphan_guard_keeps_sole_edge_as_match(self):
        """A small contested edge that is a node's ONLY edge is never demoted;
        an otherwise-identical stub whose endpoints are both anchored IS."""
        edges = [
            _edge("r_a", "t_a", 1.0, 0.9, 0.9),  # anchors r_a and t_a
            _edge("r_b", "t_b", 1.0, 0.9, 0.9),  # anchors r_b and t_b
            # Stub whose ref (r_b) and target (t_a) are BOTH anchored → demoted.
            _edge("r_b", "t_a", 0.97, 0.1, 0.1),
            # Small contested edge that is r_c's SOLE edge → orphan-guard keeps it.
            _edge("r_c", "t_a", 0.96, 0.1, 0.1),
        ]
        rg, tg = _len_geoms(edges)
        demote = _contested_small_span_review_pairs(edges, rg, tg)
        assert ("r_b", "t_a") in demote
        assert ("r_c", "t_a") not in demote  # rescued: r_c would be orphaned

    def test_create_group_results_demotes_review_pairs_without_dropping(self):
        """review_pairs edges become REVIEW+flagged; all edges are retained."""
        edges = [
            _edge("r_a", "t_b", 0.998, 0.986, 0.906),
            _edge("r_b", "t_a", 1.0, 0.881, 1.0),
            _edge("r_b", "t_b", 0.97, 0.108, 0.079),
        ]
        tagged = _create_group_results(edges, MatchType.M_TO_N, review_pairs={("r_b", "t_b")})
        # Nothing dropped.
        assert len(tagged) == len(edges)
        by_pair = {(r.ref_id, r.target_id): r for r in tagged}
        stub = by_pair[("r_b", "t_b")]
        assert stub.decision == MatchDecision.REVIEW
        assert stub.features.get(PARALLEL_SIBLING_REVIEW_FLAG) == 1.0
        # The dominant edges keep the (MATCH) group decision and are not flagged.
        for pair in (("r_a", "t_b"), ("r_b", "t_a")):
            r = by_pair[pair]
            assert r.decision == MatchDecision.MATCH
            assert PARALLEL_SIBLING_REVIEW_FLAG not in r.features
