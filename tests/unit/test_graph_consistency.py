"""Unit tests for post-optimizer graph consistency validation.

Tests the three consistency checks:
1. Junction contradiction
2. Neighbourhood coherence
3. Degree excess

And the public validate_graph_consistency() entry point.
"""

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.matching.graph_consistency import (
    _build_junction_nodes,
    _build_match_maps,
    _build_segment_adjacency_graph,
    _degree_excess_scores,
    _junction_contradiction_scores,
    _neighbourhood_coherence_scores,
    validate_graph_consistency,
)
from matcher.matching.types import MatchDecision, MatchResult

# ---------------------------------------------------------------------------
# Fixtures — simple grid network
# ---------------------------------------------------------------------------
#
#  Layout (metric CRS, units = metres):
#
#    ref_A: (0,0)─(100,0)   ref_B: (100,0)─(200,0)   ref_C: (100,0)─(100,100)
#
#    target_X: (0,0)─(100,0)   target_Y: (100,0)─(200,0)   target_Z: (100,0)─(100,100)
#    target_W: (500,500)─(600,500)   (isolated — not adjacent to anything)
#


def _make_ref_gdf():
    return gpd.GeoDataFrame(
        {
            "id": ["ref_A", "ref_B", "ref_C"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
                LineString([(100, 0), (100, 100)]),
            ],
        },
        crs="EPSG:32618",  # UTM 18N — already projected
    )


def _make_target_gdf():
    return gpd.GeoDataFrame(
        {
            "id": [
                "target_X",
                "target_Y",
                "target_Z",
                "target_W",
            ],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
                LineString([(100, 0), (100, 100)]),
                LineString([(500, 500), (600, 500)]),  # isolated
            ],
        },
        crs="EPSG:32618",
    )


@pytest.fixture
def ref_gdf():
    return _make_ref_gdf()


@pytest.fixture
def target_gdf():
    return _make_target_gdf()


# ---------------------------------------------------------------------------
# Tests for _build_segment_adjacency_graph
# ---------------------------------------------------------------------------


class TestBuildSegmentAdjacencyGraph:
    def test_adjacent_segments_connected(self, ref_gdf):
        g = _build_segment_adjacency_graph(ref_gdf, "id", tolerance=2.0)
        # ref_A, ref_B, ref_C all share endpoint (100, 0)
        assert g.has_edge("ref_A", "ref_B")
        assert g.has_edge("ref_A", "ref_C")
        assert g.has_edge("ref_B", "ref_C")

    def test_non_adjacent_segments_not_connected(self, target_gdf):
        g = _build_segment_adjacency_graph(target_gdf, "id", tolerance=2.0)
        # target_W is far away from everything else
        assert not g.has_edge("target_X", "target_W")
        assert not g.has_edge("target_Y", "target_W")
        assert not g.has_edge("target_Z", "target_W")

    def test_empty_gdf(self):
        gdf = gpd.GeoDataFrame({"id": [], "geometry": []}, crs="EPSG:32618")
        g = _build_segment_adjacency_graph(gdf, "id", tolerance=2.0)
        assert g.n_nodes == 0

    def test_single_segment(self):
        gdf = gpd.GeoDataFrame(
            {"id": ["solo"], "geometry": [LineString([(0, 0), (10, 0)])]},
            crs="EPSG:32618",
        )
        g = _build_segment_adjacency_graph(gdf, "id", tolerance=2.0)
        assert g.n_nodes == 1
        assert g.n_edges == 0


# ---------------------------------------------------------------------------
# Tests for match map helpers
# ---------------------------------------------------------------------------


class TestBuildMatchMaps:
    def test_basic_maps(self):
        results = [
            MatchResult("r1", "t1", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("r1", "t2", MatchDecision.MATCH, 0.8, {}, {}),
            MatchResult("r2", "t1", MatchDecision.MATCH, 0.7, {}, {}),
        ]
        r2t, t2r = _build_match_maps(results)
        assert r2t["r1"] == {"t1", "t2"}
        assert r2t["r2"] == {"t1"}
        assert t2r["t1"] == {"r1", "r2"}
        assert t2r["t2"] == {"r1"}


# ---------------------------------------------------------------------------
# Tests for junction contradiction (Check 1)
# ---------------------------------------------------------------------------


class TestJunctionContradiction:
    def test_consistent_matches_score_zero(self, ref_gdf, target_gdf):
        """Matching ref_A→target_X and ref_B→target_Y is consistent because
        ref_A-ref_B share a junction and target_X-target_Y also share a junction."""
        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _junction_contradiction_scores(accepted, ref_graph, target_graph, r2t)
        assert scores[("ref_A", "target_X")] == 0.0
        assert scores[("ref_B", "target_Y")] == 0.0

    def test_contradicted_match_scores_high(self, ref_gdf, target_gdf):
        """Matching ref_A→target_W (isolated) while ref_B→target_Y should
        produce a high contradiction score for ref_A→target_W because
        ref_A's neighbour ref_B is matched to target_Y, but target_W is
        not adjacent to target_Y."""
        accepted = [
            MatchResult("ref_A", "target_W", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _junction_contradiction_scores(accepted, ref_graph, target_graph, r2t)
        # ref_A has matched neighbours ref_B (and ref_C is unmatched).
        # ref_B's target (target_Y) is NOT adjacent to target_W.
        assert scores[("ref_A", "target_W")] > 0.0

    def test_no_matched_neighbours_scores_zero(self, ref_gdf, target_gdf):
        """A match whose ref neighbours are all unmatched gets score 0."""
        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _junction_contradiction_scores(accepted, ref_graph, target_graph, r2t)
        assert scores[("ref_A", "target_X")] == 0.0


# ---------------------------------------------------------------------------
# Tests for neighbourhood coherence (Check 2)
# ---------------------------------------------------------------------------


class TestNeighbourhoodCoherence:
    def test_coherent_neighbourhood(self, ref_gdf, target_gdf):
        """All three refs matched to adjacent targets → coherence should be high
        (incoherence score low)."""
        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _neighbourhood_coherence_scores(accepted, ref_graph, target_graph, r2t)
        # ref_A's neighbours are ref_B and ref_C (both matched).
        # target_X's neighbours are target_Y and target_Z.
        # ref_B→target_Y is adjacent to target_X → coherent.
        # ref_C→target_Z is adjacent to target_X → coherent.
        assert scores[("ref_A", "target_X")] == 0.0

    def test_incoherent_isolated_match(self, ref_gdf, target_gdf):
        """ref_A→target_W while neighbours match to targets far from target_W
        should have high incoherence."""
        accepted = [
            MatchResult("ref_A", "target_W", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _neighbourhood_coherence_scores(accepted, ref_graph, target_graph, r2t)
        # ref_A's matched neighbours ref_B→target_Y and ref_C→target_Z
        # are NOT adjacent to target_W → fully incoherent
        assert scores[("ref_A", "target_W")] == 1.0


# ---------------------------------------------------------------------------
# Tests for degree excess (Check 3)
# ---------------------------------------------------------------------------


class TestDegreeExcess:
    def test_balanced_junction(self, ref_gdf):
        """3 refs at junction, 3 targets matched → no excess."""
        ref_junctions = _build_junction_nodes(ref_gdf, "id", 2.0)

        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        r2t, _ = _build_match_maps(accepted)

        scores = _degree_excess_scores(accepted, ref_junctions, r2t)
        for key in scores:
            assert scores[key] == 0.0

    def test_excess_junction(self, ref_gdf):
        """3 refs at junction, but 5 targets matched → excess."""
        ref_junctions = _build_junction_nodes(ref_gdf, "id", 2.0)

        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_A", "target_W", MatchDecision.MATCH, 0.8, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_B", "target_V", MatchDecision.MATCH, 0.7, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        r2t, _ = _build_match_maps(accepted)

        scores = _degree_excess_scores(accepted, ref_junctions, r2t)
        # 3 refs, 5 distinct targets → excess = (5-3)/3 ≈ 0.67
        for key in scores:
            assert scores[key] > 0.0


# ---------------------------------------------------------------------------
# Tests for validate_graph_consistency (public API)
# ---------------------------------------------------------------------------


class TestValidateGraphConsistency:
    def test_empty_results(self, ref_gdf, target_gdf):
        result = validate_graph_consistency([], ref_gdf, target_gdf)
        assert result == []

    def test_no_match_decisions_unchanged(self, ref_gdf, target_gdf):
        """REVIEW and NO_MATCH results should pass through unchanged."""
        results = [
            MatchResult("ref_A", "target_X", MatchDecision.REVIEW, 0.4, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.NO_MATCH, 0.2, {}, {}),
        ]
        validated = validate_graph_consistency(
            results, ref_gdf, target_gdf, ref_id_column="id", target_id_column="id"
        )
        assert len(validated) == 2
        assert validated[0].decision == MatchDecision.REVIEW
        assert validated[1].decision == MatchDecision.NO_MATCH

    def test_consistent_matches_preserved(self, ref_gdf, target_gdf):
        """Topologically consistent matches should stay MATCH."""
        results = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        validated = validate_graph_consistency(
            results,
            ref_gdf,
            target_gdf,
            ref_id_column="id",
            target_id_column="id",
            snap_tolerance=2.0,
        )
        decisions = [r.decision for r in validated]
        assert all(d == MatchDecision.MATCH for d in decisions)

    def test_inconsistent_match_demoted(self, ref_gdf, target_gdf):
        """ref_A→target_W is topologically inconsistent: target_W is isolated
        while ref_A connects to ref_B/ref_C which match to adjacent targets."""
        results = [
            MatchResult("ref_A", "target_W", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        validated = validate_graph_consistency(
            results,
            ref_gdf,
            target_gdf,
            ref_id_column="id",
            target_id_column="id",
            snap_tolerance=2.0,
        )
        # ref_A→target_W should be demoted to REVIEW
        a_result = next(r for r in validated if r.ref_id == "ref_A")
        assert a_result.decision == MatchDecision.REVIEW
        assert a_result.features.get("graph_consistency_flag") == 1.0
        assert "graph_consistency_reasons" in a_result.features
        assert len(a_result.features["graph_consistency_reasons"]) > 0

        # ref_B and ref_C should remain MATCH
        b_result = next(r for r in validated if r.ref_id == "ref_B")
        c_result = next(r for r in validated if r.ref_id == "ref_C")
        assert b_result.decision == MatchDecision.MATCH
        assert c_result.decision == MatchDecision.MATCH

    def test_confidence_preserved_on_demotion(self, ref_gdf, target_gdf):
        """Demoted matches should preserve their original confidence."""
        results = [
            MatchResult("ref_A", "target_W", MatchDecision.MATCH, 0.95, {}, {"feat": 1.0}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        validated = validate_graph_consistency(
            results,
            ref_gdf,
            target_gdf,
            ref_id_column="id",
            target_id_column="id",
            snap_tolerance=2.0,
        )
        a_result = next(r for r in validated if r.ref_id == "ref_A")
        assert a_result.confidence == 0.95
        assert a_result.features.get("feat") == 1.0

    def test_result_count_preserved(self, ref_gdf, target_gdf):
        """validate_graph_consistency should never add or remove results."""
        results = [
            MatchResult("ref_A", "target_W", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.REVIEW, 0.4, {}, {}),
        ]
        validated = validate_graph_consistency(
            results,
            ref_gdf,
            target_gdf,
            ref_id_column="id",
            target_id_column="id",
            snap_tolerance=2.0,
        )
        assert len(validated) == len(results)

    def test_alignment_fractions_preserved(self, ref_gdf, target_gdf):
        """Alignment fractions should survive demotion."""
        results = [
            MatchResult(
                "ref_A",
                "target_W",
                MatchDecision.MATCH,
                0.9,
                {},
                {},
                gers_start_frac=0.0,
                gers_end_frac=1.0,
                local_start_frac=0.1,
                local_end_frac=0.9,
            ),
            MatchResult("ref_B", "target_Y", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        validated = validate_graph_consistency(
            results,
            ref_gdf,
            target_gdf,
            ref_id_column="id",
            target_id_column="id",
            snap_tolerance=2.0,
        )
        a_result = next(r for r in validated if r.ref_id == "ref_A")
        assert a_result.gers_start_frac == 0.0
        assert a_result.gers_end_frac == 1.0
        assert a_result.local_start_frac == 0.1
        assert a_result.local_end_frac == 0.9


# ---------------------------------------------------------------------------
# Tests for N:1 match handling
# ---------------------------------------------------------------------------


class TestN1MatchHandling:
    """N:1 matches (multiple ref segments → same target) are a core optimizer
    output mode and must not be falsely demoted."""

    def test_n1_junction_contradiction_score_zero(self, ref_gdf, target_gdf):
        """Adjacent ref segments matched to the same target should have
        contradiction score 0, not 1."""
        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_X", MatchDecision.MATCH, 0.85, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _junction_contradiction_scores(accepted, ref_graph, target_graph, r2t)
        assert scores[("ref_A", "target_X")] == 0.0
        assert scores[("ref_B", "target_X")] == 0.0

    def test_n1_coherence_score_zero(self, ref_gdf, target_gdf):
        """Adjacent ref segments matched to the same target should have
        coherence score 0 (fully coherent), not 1."""
        accepted = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_X", MatchDecision.MATCH, 0.85, {}, {}),
        ]
        ref_graph = _build_segment_adjacency_graph(ref_gdf, "id", 2.0)
        target_graph = _build_segment_adjacency_graph(target_gdf, "id", 2.0)
        r2t, _ = _build_match_maps(accepted)

        scores = _neighbourhood_coherence_scores(accepted, ref_graph, target_graph, r2t)
        assert scores[("ref_A", "target_X")] == 0.0
        assert scores[("ref_B", "target_X")] == 0.0

    def test_n1_matches_not_demoted(self, ref_gdf, target_gdf):
        """End-to-end: N:1 grouped matches should not be demoted by
        validate_graph_consistency."""
        results = [
            MatchResult("ref_A", "target_X", MatchDecision.MATCH, 0.9, {}, {}),
            MatchResult("ref_B", "target_X", MatchDecision.MATCH, 0.85, {}, {}),
            MatchResult("ref_C", "target_Z", MatchDecision.MATCH, 0.8, {}, {}),
        ]
        validated = validate_graph_consistency(
            results,
            ref_gdf,
            target_gdf,
            ref_id_column="id",
            target_id_column="id",
            snap_tolerance=2.0,
        )
        decisions = [r.decision for r in validated]
        assert all(d == MatchDecision.MATCH for d in decisions)
