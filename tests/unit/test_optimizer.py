"""Unit tests for match optimization algorithms.

Tests the sparse optimizer, greedy fallback, and auto-selection logic.
"""

import pytest

from matcher.matching.optimizer import (
    optimize_matches,
    optimize_matches_auto,
    optimize_matches_greedy,
    optimize_matches_sparse,
)
from matcher.matching.rules import MatchDecision, MatchResult


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


@pytest.fixture
def large_conflict_results():
    """Larger set of conflicting results for testing optimization quality."""
    results = []
    # Create a matrix of candidates where global optimization matters
    # ref_1 prefers t_1, ref_2 prefers t_1 (but should get t_2)
    # ref_3 prefers t_2 (but ref_2 should get t_2, so ref_3 gets t_3)
    results.append(MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.95, {}, {}))
    results.append(MatchResult("ref_1", "t_2", MatchDecision.MATCH, 0.7, {}, {}))
    results.append(MatchResult("ref_2", "t_1", MatchDecision.MATCH, 0.9, {}, {}))
    results.append(MatchResult("ref_2", "t_2", MatchDecision.MATCH, 0.88, {}, {}))
    results.append(MatchResult("ref_3", "t_2", MatchDecision.MATCH, 0.85, {}, {}))
    results.append(MatchResult("ref_3", "t_3", MatchDecision.MATCH, 0.8, {}, {}))
    return results


class TestOptimizeMatchesSparse:
    """Tests for sparse optimization using LAPJV (scipy linear_sum_assignment)."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = optimize_matches_sparse([], min_confidence=0.5)
        assert result == []

    def test_no_conflicts_preserves_all(self, simple_results):
        """When no conflicts exist, all matches above threshold should be kept."""
        optimized = optimize_matches_sparse(simple_results, min_confidence=0.5)

        assert len(optimized) == 3
        ref_ids = {r.ref_id for r in optimized}
        assert ref_ids == {"ref_1", "ref_2", "ref_3"}

    def test_min_confidence_filters(self, simple_results):
        """Min confidence should filter low-confidence results."""
        optimized = optimize_matches_sparse(simple_results, min_confidence=0.8)

        assert len(optimized) == 2
        ref_ids = {r.ref_id for r in optimized}
        assert "ref_1" in ref_ids
        assert "ref_2" in ref_ids

    def test_conflicts_resolved_one_to_one(self, conflicting_results):
        """Conflicts should be resolved with 1:1 assignments."""
        optimized = optimize_matches_sparse(conflicting_results, min_confidence=0.5)

        # Each reference should appear at most once
        ref_ids = [r.ref_id for r in optimized]
        assert len(ref_ids) == len(set(ref_ids))

        # Each target should appear at most once
        target_ids = [r.target_id for r in optimized]
        assert len(target_ids) == len(set(target_ids))

    def test_optimal_global_assignment(self, large_conflict_results):
        """Sparse should produce same result as Hungarian for global optimization."""
        sparse_result = optimize_matches_sparse(large_conflict_results, min_confidence=0.5)
        dense_result = optimize_matches(large_conflict_results, min_confidence=0.5)

        # Both should have same total confidence (global optimum)
        sparse_total = sum(r.confidence for r in sparse_result)
        dense_total = sum(r.confidence for r in dense_result)

        # Allow small floating point tolerance
        assert abs(sparse_total - dense_total) < 0.01, (
            f"Sparse ({sparse_total:.4f}) != Dense ({dense_total:.4f})"
        )


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


class TestOptimizeMatchesAuto:
    """Tests for auto-selection of optimization strategy."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = optimize_matches_auto([], min_confidence=0.5)
        assert result == []

    def test_small_problem_uses_dense(self, simple_results):
        """Small problems should use dense Hungarian algorithm."""
        optimized = optimize_matches_auto(simple_results, min_confidence=0.5)

        assert len(optimized) == 3

    def test_respects_memory_limit(self, simple_results):
        """Should respect memory limit parameter."""
        # Very small limit should still work (falls back gracefully)
        optimized = optimize_matches_auto(simple_results, min_confidence=0.5, memory_limit_gb=0.001)

        assert len(optimized) == 3

    def test_produces_valid_assignment(self, conflicting_results):
        """Auto should produce valid 1:1 assignment."""
        optimized = optimize_matches_auto(conflicting_results, min_confidence=0.5)

        # Each reference should appear at most once
        ref_ids = [r.ref_id for r in optimized]
        assert len(ref_ids) == len(set(ref_ids))

        # Each target should appear at most once
        target_ids = [r.target_id for r in optimized]
        assert len(target_ids) == len(set(target_ids))


class TestOptimizerEquivalence:
    """Tests that different optimization strategies produce equivalent results."""

    def test_sparse_equals_dense_simple(self, simple_results):
        """Sparse and dense should produce identical results for simple case."""
        sparse = optimize_matches_sparse(simple_results, min_confidence=0.5)
        dense = optimize_matches(simple_results, min_confidence=0.5)

        sparse_pairs = {(r.ref_id, r.target_id) for r in sparse}
        dense_pairs = {(r.ref_id, r.target_id) for r in dense}

        assert sparse_pairs == dense_pairs

    def test_sparse_equals_dense_conflicting(self, conflicting_results):
        """Sparse and dense should produce same global optimum for conflicts."""
        sparse = optimize_matches_sparse(conflicting_results, min_confidence=0.5)
        dense = optimize_matches(conflicting_results, min_confidence=0.5)

        # Total confidence should be the same (both are optimal)
        sparse_total = sum(r.confidence for r in sparse)
        dense_total = sum(r.confidence for r in dense)

        assert abs(sparse_total - dense_total) < 0.01

    def test_greedy_near_optimal(self, large_conflict_results):
        """Greedy should be at least 90% of optimal for typical cases."""
        greedy = optimize_matches_greedy(large_conflict_results, min_confidence=0.5)
        optimal = optimize_matches(large_conflict_results, min_confidence=0.5)

        greedy_total = sum(r.confidence for r in greedy)
        optimal_total = sum(r.confidence for r in optimal)

        # Greedy should be at least 90% of optimal
        assert greedy_total >= 0.9 * optimal_total, (
            f"Greedy ({greedy_total:.4f}) < 90% of optimal ({optimal_total:.4f})"
        )


class TestDuplicateCandidates:
    """Tests for handling duplicate candidates for the same ref-target pair."""

    def test_keeps_best_duplicate(self):
        """When multiple candidates exist for same pair, keep highest confidence."""
        results = [
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_1", "t_1", MatchDecision.MATCH, 0.8, {}, {}),  # Duplicate, lower
            MatchResult("ref_2", "t_2", MatchDecision.MATCH, 0.85, {}, {}),
        ]

        for optimize_fn in [optimize_matches, optimize_matches_sparse, optimize_matches_greedy]:
            optimized = optimize_fn(results, min_confidence=0.5)

            # Should only have 2 unique pairs
            pairs = {(r.ref_id, r.target_id) for r in optimized}
            assert len(pairs) == 2

            # ref_1, t_1 should have the higher confidence
            ref_1_match = [r for r in optimized if r.ref_id == "ref_1"][0]
            assert ref_1_match.confidence == 0.9
