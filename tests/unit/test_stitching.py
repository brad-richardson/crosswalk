"""Unit tests for stitching review modules.

Tests alternatives generation, batch selection, stitching label store,
compute_group_id, and stitching label data integrity.
"""

import json

import pandas as pd
import pytest

from crosswalk.labeling.stitching_store import DEFAULT_STITCHING_DIR, STITCHING_LABEL_COLUMNS
from crosswalk.matching.alternatives import generate_top_k_alternatives
from crosswalk.matching.batch_selection import (
    BORDERLINE_MIN_SCORE,
    compute_borderline_score,
    select_stitching_batch,
)
from crosswalk.matching.optimizer import compute_group_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _edge(ref, tgt, conf):
    return {"ref_id": ref, "target_id": tgt, "confidence": conf}


@pytest.fixture
def edges_1_to_n():
    """1:N component — one ref matched to three targets."""
    return [
        _edge("r1", "t1", 0.9),
        _edge("r1", "t2", 0.8),
        _edge("r1", "t3", 0.3),
    ]


@pytest.fixture
def edges_m_to_n():
    """M:N component — two refs matched to two targets."""
    return [
        _edge("r1", "t1", 0.9),
        _edge("r1", "t2", 0.4),
        _edge("r2", "t1", 0.3),
        _edge("r2", "t2", 0.85),
    ]


def _make_group(group_id, edges, alternatives=None, match_type="N:1"):
    """Build a minimal group dict for batch selection tests."""
    return {
        "group_id": group_id,
        "match_type": match_type,
        "edges": edges,
        "alternatives": alternatives or [],
    }


# ---------------------------------------------------------------------------
# compute_group_id
# ---------------------------------------------------------------------------


class TestComputeGroupId:
    def test_deterministic(self):
        a = compute_group_id({"r1", "r2"}, {"t1"})
        b = compute_group_id({"r2", "r1"}, {"t1"})
        assert a == b

    def test_length_is_8(self):
        assert len(compute_group_id({"r1"}, {"t1", "t2"})) == 8

    def test_different_ids_differ(self):
        a = compute_group_id({"r1"}, {"t1"})
        b = compute_group_id({"r1"}, {"t2"})
        assert a != b

    def test_ref_target_order_matters(self):
        """Swapping ref and target sets should produce a different ID."""
        a = compute_group_id({"r1"}, {"t1"})
        b = compute_group_id({"t1"}, {"r1"})
        assert a != b


# ---------------------------------------------------------------------------
# generate_top_k_alternatives
# ---------------------------------------------------------------------------


class TestGenerateTopKAlternatives:
    def test_empty_input(self):
        assert generate_top_k_alternatives([]) == []

    def test_1_to_n_returns_sorted_by_confidence(self, edges_1_to_n):
        alts = generate_top_k_alternatives(edges_1_to_n, k=5)
        confidences = [a["total_confidence"] for a in alts]
        assert confidences == sorted(confidences, reverse=True)

    def test_1_to_n_top_alternative_is_greedy(self, edges_1_to_n):
        alts = generate_top_k_alternatives(edges_1_to_n, k=5)
        top = alts[0]
        # Greedy picks all three targets assigned to r1
        edge_pairs = {(e["ref_id"], e["target_id"]) for e in top["edges"]}
        assert ("r1", "t1") in edge_pairs
        assert ("r1", "t2") in edge_pairs
        assert ("r1", "t3") in edge_pairs

    def test_m_to_n_has_multiple_alternatives(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=5)
        assert len(alts) > 1

    def test_option_index_sequential(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=5)
        indices = [a["option_index"] for a in alts]
        assert indices == list(range(len(alts)))

    def test_k_limits_results(self, edges_m_to_n):
        # Organic enumeration is bounded by k (seed options are counted separately;
        # see TestSeedOptions for the +2 bound).
        alts = generate_top_k_alternatives(edges_m_to_n, k=2, include_seed_options=False)
        assert len(alts) <= 2

    def test_no_duplicate_edge_sets(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=10)
        edge_sets = []
        for a in alts:
            es = frozenset((e["ref_id"], e["target_id"]) for e in a["edges"])
            edge_sets.append(es)
        assert len(edge_sets) == len(set(edge_sets))

    def test_summary_present(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=1)
        assert "summary" in alts[0]
        assert "->" in alts[0]["summary"]

    @pytest.mark.parametrize(
        "edges",
        [
            [_edge("r1", "t1", 0.5)],
            [_edge("r1", "t1", 0.5), _edge("r1", "t2", 0.3)],
        ],
        ids=["single_edge", "two_edges"],
    )
    def test_always_returns_at_least_one(self, edges):
        alts = generate_top_k_alternatives(edges, k=5)
        assert len(alts) >= 1

    def test_n_to_1_enumerates_ref_subsets(self):
        """N:1 group should enumerate which refs map to the single target."""
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r2", "t1", 0.8),
            _edge("r3", "t1", 0.3),
        ]
        alts = generate_top_k_alternatives(edges, k=10)
        # 3 refs -> 2^3 - 1 = 7 non-empty subsets
        assert len(alts) == 7
        # Top alternative should include all three refs (highest total)
        top = alts[0]
        assert len(top["edges"]) == 3
        assert top["total_confidence"] == pytest.approx(2.0, abs=0.01)

    def test_duplicate_edges_keeps_highest_confidence(self):
        edges = [
            _edge("r1", "t1", 0.5),
            _edge("r1", "t1", 0.9),
        ]
        alts = generate_top_k_alternatives(edges, k=5)
        # Only one unique edge set possible
        assert len(alts) == 1
        assert alts[0]["edges"][0]["confidence"] == 0.9

    def test_alignment_fracs_preserved(self):
        """Alignment fractions from input edges should flow through to output."""
        edges = [
            {
                "ref_id": "r1",
                "target_id": "t1",
                "confidence": 0.9,
                "gers_start_frac": 0.0,
                "gers_end_frac": 0.5,
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
            },
            {
                "ref_id": "r1",
                "target_id": "t2",
                "confidence": 0.8,
                "gers_start_frac": 0.5,
                "gers_end_frac": 1.0,
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
            },
        ]
        alts = generate_top_k_alternatives(edges, k=3)
        top = alts[0]
        # Top alternative has both edges; check fracs are preserved
        for edge in top["edges"]:
            assert "gers_start_frac" in edge
            assert "gers_end_frac" in edge
            assert "local_start_frac" in edge
            assert "local_end_frac" in edge
        # Verify specific values
        t1_edge = next(e for e in top["edges"] if e["target_id"] == "t1")
        assert t1_edge["gers_start_frac"] == 0.0
        assert t1_edge["gers_end_frac"] == 0.5

    def test_alignment_fracs_absent_when_not_in_input(self):
        """Edges without alignment fracs should not have them in output."""
        edges = [_edge("r1", "t1", 0.9)]
        alts = generate_top_k_alternatives(edges, k=1)
        assert "gers_start_frac" not in alts[0]["edges"][0]


# ---------------------------------------------------------------------------
# Contiguous multi-ref chains (the "T4 spans R3+R5" gap, panel eval 2026-07-03)
# ---------------------------------------------------------------------------


def _line(coords):
    """GeoJSON-style LineString mapping (as stored in the groups sidecar)."""
    return {"type": "LineString", "coordinates": coords}


def _targets_to_refs(alt):
    """Map each target -> frozenset of refs it is assigned to in an alternative."""
    by_t: dict[str, set[str]] = {}
    for e in alt["edges"]:
        by_t.setdefault(e["target_id"], set()).add(e["ref_id"])
    return {t: frozenset(rs) for t, rs in by_t.items()}


class TestMultiRefContiguousChains:
    """A target that spans multiple CONTIGUOUS refs must be an expressible option."""

    def _spanning_group(self, contiguous: bool):
        """t1 has edges to two refs (ra, rb); rc serves t2. ra/rb contiguous iff asked."""
        edges = [
            _edge("ra", "t1", 0.9),
            _edge("rb", "t1", 0.8),
            _edge("rc", "t2", 0.85),
        ]
        # Shared exact endpoint [0.001, 0] makes ra/rb contiguous (distance 0,
        # robust to the lon/lat->meter reprojection inside the generator).
        rb_coords = [[0.001, 0], [0.002, 0]] if contiguous else [[5, 5], [5.001, 5]]
        ref_geoms = {
            "ra": _line([[0, 0], [0.001, 0]]),
            "rb": _line(rb_coords),
            "rc": _line([[9, 9], [9.001, 9]]),
        }
        return edges, ref_geoms

    def test_multiref_chain_option_generated_and_ranks(self):
        """The {ra->t1, rb->t1, ...} span is generated AND ranks into the top-K."""
        edges, ref_geoms = self._spanning_group(contiguous=True)
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=5)
        # Some option assigns t1 to BOTH ra and rb (the previously inexpressible shape).
        assert any(_targets_to_refs(a).get("t1") == frozenset({"ra", "rb"}) for a in alts)
        # It should rank at/near the top: the max-confidence option keeps both
        # ra->t1 and rb->t1 (0.9 + 0.8) alongside rc->t2.
        top = alts[0]
        assert _targets_to_refs(top).get("t1") == frozenset({"ra", "rb"})

    def test_chain_requires_contiguity(self):
        """Two refs that both touch t1 but are NOT contiguous form no chain.

        Enumeration invariant only: seeds are disabled because the
        full-candidate-set seed deliberately offers the "accept every edge"
        union regardless of contiguity (see TestSeedOptions).
        """
        edges, ref_geoms = self._spanning_group(contiguous=False)
        alts = generate_top_k_alternatives(
            edges, ref_geoms=ref_geoms, k=10, include_seed_options=False
        )
        assert not any(_targets_to_refs(a).get("t1") == frozenset({"ra", "rb"}) for a in alts)

    def test_chain_is_subset_of_existing_edges(self):
        """A chain is only offered when EVERY constituent edge exists in the group."""
        # rb is contiguous to ra but only has an edge to t2, not t1.
        edges = [
            _edge("ra", "t1", 0.9),
            _edge("rb", "t2", 0.8),
        ]
        ref_geoms = {
            "ra": _line([[0, 0], [0.001, 0]]),
            "rb": _line([[0.001, 0], [0.002, 0]]),
        }
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=10)
        # t1 never pairs with rb: the (rb, t1) edge does not exist.
        assert not any("rb" in _targets_to_refs(a).get("t1", frozenset()) for a in alts)

    def test_no_geoms_means_no_multiref(self):
        """Without geometry, enumeration is unchanged: only single-ref options.

        Enumeration invariant only (seeds disabled): the full-set seed is a
        whole-group union and is exempt from the per-target single-ref rule.
        """
        edges, _ = self._spanning_group(contiguous=True)
        alts = generate_top_k_alternatives(edges, k=10, include_seed_options=False)
        for a in alts:
            for refs in _targets_to_refs(a).values():
                assert len(refs) == 1

    def test_chain_length_bounded(self):
        """Chains never exceed MAX_REF_CHAIN_LEN refs for a single target."""
        from crosswalk.matching.alternatives import MAX_REF_CHAIN_LEN

        # Five refs in a contiguous line, all matched to t1.
        edges = [_edge(f"r{i}", "t1", 0.9 - i * 0.05) for i in range(5)]
        # Add a second target so this stays on the M:N (per-target) path.
        edges.append(_edge("r0", "t2", 0.5))
        ref_geoms = {f"r{i}": _line([[i * 0.001, 0], [(i + 1) * 0.001, 0]]) for i in range(5)}
        # Enumeration invariant only (seeds disabled): the full-set seed
        # intentionally offers all refs->t1 as the "accept everything" union.
        alts = generate_top_k_alternatives(
            edges, ref_geoms=ref_geoms, k=50, include_seed_options=False
        )
        for a in alts:
            for refs in _targets_to_refs(a).values():
                assert len(refs) <= MAX_REF_CHAIN_LEN

    def test_dedup_with_chains(self):
        """Chain-derived options are still deduplicated by edge set."""
        edges, ref_geoms = self._spanning_group(contiguous=True)
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=50)
        keys = [frozenset((e["ref_id"], e["target_id"]) for e in a["edges"]) for a in alts]
        assert len(keys) == len(set(keys))

    def test_chain_enumeration_beam_bounded_on_dense_graph(self):
        """A dense adjacency graph (all refs mutually contiguous) cannot blow up
        intermediate chain enumeration: the beam caps each frontier."""
        from crosswalk.matching.alternatives import (
            MAX_CHAIN_FRONTIER,
            _enumerate_contiguous_chains,
        )

        n = 40  # complete graph: ~C(40,3)=9880 size-3 connected subsets exist
        refs = [f"r{i}" for i in range(n)]
        adjacency = {r: set(refs) - {r} for r in refs}
        conf = {r: 1.0 - i * 0.01 for i, r in enumerate(refs)}

        chains = _enumerate_contiguous_chains(refs, adjacency, 3, conf=conf)

        # Bounded: at most max_frontier per size step (sizes 2 and 3)
        assert len(chains) <= 2 * MAX_CHAIN_FRONTIER
        # Beam keeps the best: the top-confidence pair survives
        assert frozenset({"r0", "r1"}) in chains

    def test_large_group_bounded_and_proposes_multiref(self):
        """Big groups fall back to greedy but can still propose multi-ref edges."""
        # 8 targets (> MAX_EXHAUSTIVE_TARGETS) each fed by 3 contiguous refs ->
        # greedy path. Must stay bounded (<= k) and still surface a multi-ref span.
        edges = []
        ref_geoms = {}
        for t in range(8):
            for j in range(3):
                rid = f"r{t}_{j}"
                edges.append(_edge(rid, f"t{t}", 0.9 - j * 0.1))
                base = t * 10 + j
                ref_geoms[rid] = _line([[base * 0.001, 0], [(base + 1) * 0.001, 0]])
        # Organic bound only (seeds disabled): greedy enumeration stays <= k.
        alts = generate_top_k_alternatives(
            edges, ref_geoms=ref_geoms, k=5, include_seed_options=False
        )
        assert 0 < len(alts) <= 5
        assert any(any(len(refs) >= 2 for refs in _targets_to_refs(a).values()) for a in alts)


class TestSeedOptions:
    """Whole-group seed options (full-candidate-set + optimizer-selected-set).

    Seeds close the measured expressibility gap where the per-target enumeration
    cannot express a settled answer (e.g. a target legitimately spanning more
    refs than MAX_REF_CHAIN_LEN allows, so the correct answer is "accept every
    candidate edge"), or where the optimizer's selection ranks below the
    confidence-sorted top-K.
    """

    def _full(self, alts):
        return [frozenset((e["ref_id"], e["target_id"]) for e in a["edges"]) for a in alts]

    def test_full_candidate_set_always_offered(self):
        """The union of all group edges is always an option, even when the
        per-target model cannot enumerate it (a target spanning many refs)."""
        # One target fed by 4 refs -> exceeds MAX_REF_CHAIN_LEN, so no enumerated
        # chain covers all 4; only the full-set seed can express "accept all".
        edges = [_edge(f"r{i}", "t1", 0.9 - i * 0.1) for i in range(4)]
        edges.append(_edge("r0", "t2", 0.5))
        full = frozenset((e["ref_id"], e["target_id"]) for e in edges)
        with_seed = self._full(generate_top_k_alternatives(edges, k=5))
        without = self._full(generate_top_k_alternatives(edges, k=5, include_seed_options=False))
        assert full in with_seed
        assert full not in without  # organic enumeration cannot express it

    def test_optimizer_selected_seed_offered(self):
        """Edges flagged ``selected`` are offered as an option even when they are
        not the confidence-greedy pick and optimizer_assignment is unavailable."""
        # 7 targets -> greedy path; the optimizer selected a low-confidence-ranked
        # subset (drop one target) that greedy would not surface at small k.
        edges = []
        for t in range(7):
            e = _edge(f"r{t}", f"t{t}", 0.9)
            # Mark all but the last target as the optimizer's selection.
            e["selected"] = t < 6
            edges.append(e)
        selected = frozenset((e["ref_id"], e["target_id"]) for e in edges if e.get("selected"))
        offered = self._full(generate_top_k_alternatives(edges, k=3))
        assert selected in offered

    def test_seeds_bounded_to_plus_two(self):
        """Seeds add at most two options beyond the organic top-K."""
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r1", "t2", 0.4),
            _edge("r2", "t1", 0.3),
            _edge("r2", "t2", 0.85),
        ]
        for e in edges:
            e["selected"] = True
        organic = generate_top_k_alternatives(edges, k=3, include_seed_options=False)
        seeded = generate_top_k_alternatives(edges, k=3, include_seed_options=True)
        assert len(seeded) <= len(organic) + 2

    def test_seeds_deduped_against_organic(self):
        """A seed equal to an organic option is not added twice."""
        # N:1: the full set is the all-refs subset, already enumerated organically.
        edges = [_edge("r1", "t1", 0.9), _edge("r2", "t1", 0.8), _edge("r3", "t1", 0.3)]
        keys = self._full(generate_top_k_alternatives(edges, k=10))
        assert len(keys) == len(set(keys))

    def test_seeds_are_subset_of_group_edges(self):
        """Seed options obey the subset invariant (only real group edges)."""
        edges = [_edge("r1", "t1", 0.9), _edge("r2", "t2", 0.8)]
        for e in edges:
            e["selected"] = True
        valid = {(e["ref_id"], e["target_id"]) for e in edges}
        for a in generate_top_k_alternatives(edges, k=5):
            for e in a["edges"]:
                assert (e["ref_id"], e["target_id"]) in valid

    def test_seeds_tagged_is_seed_organic_not(self):
        """Seed options carry is_seed=True; organic alternatives do not."""
        # t1 spans 4 refs -> full-set seed cannot be organic; mark a selection too.
        edges = [_edge(f"r{i}", "t1", 0.9 - i * 0.1) for i in range(4)]
        edges.append(_edge("r0", "t2", 0.5))
        edges[0]["selected"] = True
        alts = generate_top_k_alternatives(edges, k=5)
        seeds = [a for a in alts if a.get("is_seed")]
        organic = [a for a in alts if not a.get("is_seed")]
        assert 1 <= len(seeds) <= 2
        assert organic  # organic alternatives never carry the tag
        full = frozenset((e["ref_id"], e["target_id"]) for e in edges)
        assert any(
            frozenset((e["ref_id"], e["target_id"]) for e in s["edges"]) == full for s in seeds
        )

    def test_selected_flag_survives_duplicate_dedup(self):
        """The selected-seed uses flags from ALL input edges: a duplicate pair
        whose flagged copy has LOWER confidence still contributes its pair."""
        edges = []
        for t in range(7):  # 7 targets -> greedy path (selection not organic)
            edges.append(_edge(f"r{t}", f"t{t}", 0.9))
        # Optimizer selected 6 of 7 pairs...
        for e in edges[:6]:
            e["selected"] = True
        # ...but pair (r0, t0)'s flag lives on a lower-confidence duplicate.
        edges[0]["selected"] = False
        dup = _edge("r0", "t0", 0.2)
        dup["selected"] = True
        edges.append(dup)
        selected = frozenset((f"r{t}", f"t{t}") for t in range(6))
        offered = self._full(generate_top_k_alternatives(edges, k=3))
        assert selected in offered

    def test_batch_selection_scores_ignore_seeds(self):
        """Seed options must not skew select_stitching_batch's tier scoring:
        the full-set seed is a superset of every proper assignment and would
        otherwise always win max(total_confidence)."""
        # 3-target M:N with one high- and one low-confidence ref per target.
        edges = []
        for t in range(3):
            edges.append(_edge(f"rh{t}", f"t{t}", 0.9))
            edges.append(_edge(f"rl{t}", f"t{t}", 0.1))

        def _group_with(seeded: bool):
            return {
                "group_id": "g",
                "match_type": "M:N",
                "edges": edges,
                "alternatives": generate_top_k_alternatives(
                    edges, k=5, include_seed_options=seeded
                ),
            }

        sel_seeded = select_stitching_batch([_group_with(True)], set(), k=1)
        sel_organic = select_stitching_batch([_group_with(False)], set(), k=1)
        assert sel_seeded[0]["review_score"] == sel_organic[0]["review_score"]
        assert sel_seeded[0]["review_tier"] == sel_organic[0]["review_tier"]

    def test_n_to_1_ignores_geoms_full_powerset(self):
        """N:1 mirror: refs already enumerate the full power set; geoms are a no-op."""
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r2", "t1", 0.8),
            _edge("r3", "t1", 0.3),
        ]
        ref_geoms = {
            "r1": _line([[0, 0], [0.001, 0]]),
            "r2": _line([[0.001, 0], [0.002, 0]]),
            "r3": _line([[0.002, 0], [0.003, 0]]),
        }
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=10)
        assert len(alts) == 7  # 2^3 - 1 non-empty subsets, unchanged by geoms


class TestAlternativeOutputInvariant:
    """Every emitted alternative must be a deduplicated subset of the group edges.

    Regression for the option-generation defect where a large M:N group's
    alternatives (computed before ``_fill_spatial_context`` clipped the group to
    a 500m envelope) leaked edges outside the surviving edge set: group
    ``701d491e`` (9 ref x 7 target, 16 edges) shipped alternatives with 230
    edges and total_confidence ~227.
    """

    def _big_mn_group(self):
        """A 701d491e-shaped M:N group: many targets, each with a few refs."""
        edges = []
        ref_geoms = {}
        target_geoms = {}
        for t in range(7):
            target_geoms[f"t{t}"] = _line([[t * 0.001, 0.0001], [(t + 1) * 0.001, 0.0001]])
            for j in range(3):
                rid = f"r{t}_{j}"
                edges.append(_edge(rid, f"t{t}", 0.95 - j * 0.1))
                base = t * 10 + j
                ref_geoms[rid] = _line([[base * 0.001, 0], [(base + 1) * 0.001, 0]])
        return edges, ref_geoms, target_geoms

    def test_all_alternatives_are_valid_dedup_subsets(self):
        edges, ref_geoms, target_geoms = self._big_mn_group()
        group_keys = {(e["ref_id"], e["target_id"]) for e in edges}
        max_conf = sum(e["confidence"] for e in edges)
        alts = generate_top_k_alternatives(
            edges, ref_geoms=ref_geoms, target_geoms=target_geoms, k=5
        )
        assert alts
        for a in alts:
            keys = [(e["ref_id"], e["target_id"]) for e in a["edges"]]
            # subset of the group's candidate edges
            assert set(keys) <= group_keys
            # no duplicate edges within an option
            assert len(keys) == len(set(keys))
            # never more edges than the group has
            assert len(a["edges"]) <= len(group_keys)
            # total confidence cannot exceed the group's own edge-confidence sum
            assert a["total_confidence"] <= max_conf + 1e-6

    def test_sanitize_drops_invalid_and_duplicate_edges(self):
        """A malformed alternative is repaired to a valid dedup subset."""
        from crosswalk.matching.alternatives import _sanitize_alternative

        valid_keys = {("r1", "t1"), ("r2", "t2")}
        alt = {
            "edges": [
                _edge("r1", "t1", 0.9),
                _edge("r1", "t1", 0.9),  # duplicate
                _edge("r2", "t2", 0.8),
                _edge("phantom", "ghost", 5.0),  # not in group
            ],
        }
        cleaned = _sanitize_alternative(alt, valid_keys, ["r1", "r2"])
        keys = [(e["ref_id"], e["target_id"]) for e in cleaned["edges"]]
        assert set(keys) == valid_keys
        assert len(keys) == 2  # dedup + phantom dropped
        assert cleaned["total_confidence"] == pytest.approx(1.7)


class TestPruneGroupOptionsToEdges:
    """After a group's edges are clipped, options must be re-synced to them."""

    def test_prune_resyncs_alternatives_and_optimizer(self):
        from crosswalk.matching.alternatives import prune_group_options_to_edges

        # Group originally had (r1,t1),(r2,t2),(r3,t3); clipped to just (r1,t1).
        group = {
            "edges": [_edge("r1", "t1", 0.9)],
            "optimizer_assignment": [
                {"ref_id": "r1", "target_id": "t1"},
                {"ref_id": "r2", "target_id": "t2"},  # no longer in group
            ],
            "alternatives": [
                {
                    "edges": [
                        {"ref_id": "r1", "target_id": "t1"},
                        {"ref_id": "r2", "target_id": "t2"},
                        {"ref_id": "r3", "target_id": "t3"},
                    ],
                    "total_confidence": 99.0,
                }
            ],
            "ref_geometries": {},
            "target_geometries": {},
        }
        prune_group_options_to_edges(group)
        # optimizer pruned to surviving edges only
        assert group["optimizer_assignment"] == [{"ref_id": "r1", "target_id": "t1"}]
        # alternatives regenerated as valid subsets of the single surviving edge
        for a in group["alternatives"]:
            for e in a["edges"]:
                assert (e["ref_id"], e["target_id"]) == ("r1", "t1")
            assert a["total_confidence"] <= 0.9 + 1e-6


# ---------------------------------------------------------------------------
# _export_groups_sidecar — optimizer_assignment membership join
# ---------------------------------------------------------------------------


class TestExportGroupsSidecarOptimizerAssignment:
    """The sidecar must mirror the optimizer's DECOMPOSED grouping.

    A raw connected component that is not one coherent corridor is decomposed
    into per-corridor sub-groups (plus greedy 1:1 leftovers). The sidecar emits
    one group per genuine multi-edge sub-group, each carrying its own
    ``optimizer_assignment``; a residue that resolves to pure 1:1 matches does
    NOT create a stitching group (there is nothing to review).
    """

    def _build_decomposed_mn(self):
        import geopandas as gpd
        from shapely.geometry import LineString

        from crosswalk.matching.types import MatchDecision, MatchResult

        # A genuine 1:N corridor (r1 -> t1,t1b, two collinear contiguous
        # targets) plus a far-away independent pair (r2-t2). A weak cross-edge
        # (r1-t2) links everything into ONE raw connected component, which the
        # corridor-aware optimizer decomposes: the 1:N corridor survives as a
        # group, the far pair falls out as a 1:1 (no group).
        ref = gpd.GeoDataFrame(
            {
                "id": ["r1", "r2"],
                "geometry": [
                    LineString([(0, 0), (20, 0)]),
                    LineString([(1000, 0), (1010, 0)]),
                ],
            },
            crs="EPSG:3857",
        )
        target = gpd.GeoDataFrame(
            {
                "id": ["t1", "t1b", "t2"],
                "geometry": [
                    LineString([(0, 1), (10, 1)]),
                    LineString([(10, 1), (20, 1)]),  # collinear continuation of t1
                    LineString([(1000, 1), (1010, 1)]),
                ],
            },
            crs="EPSG:3857",
        )

        def _mr(rid, tid, conf):
            return MatchResult(
                ref_id=rid,
                target_id=tid,
                decision=MatchDecision.MATCH,
                confidence=conf,
                score_breakdown={},
                features={},
            )

        results = [
            _mr("r1", "t1", 0.9),
            _mr("r1", "t1b", 0.85),  # real 1:N corridor with t1
            _mr("r2", "t2", 0.9),
            _mr("r1", "t2", 0.6),  # weak cross-edge connecting the component
        ]
        return results, ref, target

    def test_decomposed_component_gets_optimizer_assignment(self, tmp_path):
        from crosswalk.filenames import groups_sidecar_path
        from crosswalk.matching.optimizer import optimize_matches_with_grouping
        from crosswalk.pipeline.runner import _export_groups_sidecar

        results, ref, target = self._build_decomposed_mn()

        optimized = optimize_matches_with_grouping(
            results, ref, target, min_confidence=0.5, target_id_column="id"
        )

        out = tmp_path / "toy_bridge.parquet"
        sidecar_path = _export_groups_sidecar(
            results,
            optimized,
            out,
            ref,
            target,
            min_confidence=0.5,
            ref_id_column="id",
            target_id_column="id",
        )
        assert sidecar_path == groups_sidecar_path(out)

        data = json.loads(sidecar_path.read_text())
        groups = data["groups"]
        # Exactly the genuine 1:N corridor survives as a group; the far r2-t2
        # pair resolves to a 1:1 match and produces no stitching group.
        assert len(groups) == 1
        group = groups[0]
        assert group["match_type"] == "1:N"

        assignment = group["optimizer_assignment"]
        assert assignment, "optimizer_assignment must not be empty for decomposed group"

        group_refs = set(group["ref_ids"])
        group_targets = set(group["target_ids"])
        for e in assignment:
            assert e["ref_id"] in group_refs
            assert e["target_id"] in group_targets

        pairs = {(e["ref_id"], e["target_id"]) for e in assignment}
        assert pairs == {("r1", "t1"), ("r1", "t1b")}

        # Structure fields are persisted for the resolver.
        assert "n_edges" in group
        assert "n_corridors" in group
        assert "oversized_group" in group
        for e in group["edges"]:
            assert "is_bridge" in e
            assert "corridor_ref" in e
            assert "selected" in e


# ---------------------------------------------------------------------------
# select_stitching_batch
# ---------------------------------------------------------------------------


class TestSelectStitchingBatch:
    def _alt(self, conf):
        return {"total_confidence": conf, "edges": [], "summary": ""}

    def _alt_e(self, triples):
        """Alternative from ``(ref, tgt, conf)`` triples (real per-edge confidence)."""
        return {
            "total_confidence": round(sum(c for _, _, c in triples), 4),
            "edges": [{"ref_id": r, "target_id": t, "confidence": c} for r, t, c in triples],
            "summary": "",
        }

    def test_empty_groups(self):
        result = select_stitching_batch([], set(), k=10)
        assert result == []

    def test_skips_already_reviewed(self):
        groups = [
            _make_group("g1", [_edge("r1", "t1", 0.9)], [self._alt(0.9)]),
            _make_group("g2", [_edge("r2", "t2", 0.8)], [self._alt(0.8)]),
        ]
        result = select_stitching_batch(groups, {"g1"}, k=10)
        ids = {g["group_id"] for g in result}
        assert "g1" not in ids
        assert "g2" in ids

    def test_all_reviewed_returns_empty(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, {"g1"}, k=10)
        assert result == []

    def test_tier_balancing_with_enough_groups(self):
        """A pool spanning all four signal shapes, k=20, should fill all four tiers.

        Uses alternatives with real per-edge confidence so the contestedness-based
        borderline metric has something to bite on (empty-edge alternatives score
        0 and are no longer borderline).
        """
        groups = []
        # 10 LARGE groups (12 near-certain edges): fill the large tier.
        for i in range(10):
            triples = [(f"L{i}_{j}", f"t{i}_{j}", 0.95) for j in range(12)]
            edges = [_edge(r, t, c) for r, t, c in triples]
            groups.append(
                _make_group(f"big{i}", edges, [self._alt_e(triples), self._alt_e(triples[:11])])
            )
        # 5 BORDERLINE groups: top-2 differ by a genuine 0.5-confidence coin-flip edge.
        for i in range(5):
            triples = [(f"B{i}a", f"B{i}t", 0.9), (f"B{i}b", f"B{i}t", 0.5)]
            edges = [_edge(r, t, c) for r, t, c in triples]
            groups.append(
                _make_group(f"bord{i}", edges, [self._alt_e(triples), self._alt_e(triples[:1])])
            )
        # 5 LOW-CONFIDENCE groups: a single weak alternative.
        for i in range(5):
            groups.append(
                _make_group(
                    f"low{i}",
                    [_edge(f"W{i}", f"Wt{i}", 0.2)],
                    [self._alt_e([(f"W{i}", f"Wt{i}", 0.2)])],
                )
            )
        # 10 CLEAR-WINNER groups: top-2 differ only by a near-certain 0.98 edge.
        for i in range(10):
            triples = [(f"C{i}a", f"C{i}t", 0.98), (f"C{i}b", f"C{i}t", 0.98)]
            edges = [_edge(r, t, c) for r, t, c in triples]
            groups.append(
                _make_group(f"clear{i}", edges, [self._alt_e(triples), self._alt_e(triples[:1])])
            )
        result = select_stitching_batch(groups, set(), k=20)
        assert len(result) == 20

        tiers = {g["review_tier"] for g in result}
        assert tiers == {"large", "borderline", "low_confidence", "clear_winner"}

    @pytest.mark.parametrize("k", [5, 10, 20])
    def test_respects_k(self, k):
        groups = [
            _make_group(f"g{i}", [_edge("r1", "t1", 0.5)], [self._alt(0.5)]) for i in range(50)
        ]
        result = select_stitching_batch(groups, set(), k=k)
        assert len(result) == k

    def test_largest_group_always_selected(self):
        """The single largest group should always be selected as 'large' tier."""
        big_edges = [_edge(f"r_big_{j}", f"t_big_{j}", 0.9) for j in range(15)]
        small_edges = [_edge("r_small", "t_small", 0.9)]
        groups = [
            _make_group("small", small_edges, [self._alt(0.9)]),
            _make_group("big", big_edges, [self._alt(0.9)]),
        ]
        result = select_stitching_batch(groups, set(), k=2)
        big_group = next(g for g in result if g["group_id"] == "big")
        assert big_group["review_tier"] == "large"

    def test_review_tier_and_score_present(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, set(), k=5)
        assert "review_tier" in result[0]
        assert "review_score" in result[0]

    def test_no_internal_keys_leaked(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, set(), k=5)
        for g in result:
            assert "_n_edges" not in g
            assert "_borderline_score" not in g
            assert "_low_conf_score" not in g
            assert "_review_value" not in g

    def test_low_confidence_tier(self):
        """Groups with low best-alternative confidence should get low_confidence tier."""
        # Need enough groups so all tier slots get allocated (k=10 gives 3+3+3+1)
        groups = []
        for i in range(20):
            conf = 0.9 + i * 0.005
            groups.append(
                _make_group(
                    f"filler{i}",
                    [_edge(f"r{i}", f"t{i}", conf)],
                    [
                        {"total_confidence": conf, "edges": [{"confidence": conf}], "summary": ""},
                        {
                            "total_confidence": conf - 0.01,
                            "edges": [{"confidence": conf - 0.01}],
                            "summary": "",
                        },
                    ],
                )
            )
        # Add a distinctly low-confidence group
        groups.append(
            _make_group(
                "low",
                [_edge("rlow", "tlow", 0.2)],
                [{"total_confidence": 0.2, "edges": [{"confidence": 0.2}], "summary": ""}],
            )
        )
        result = select_stitching_batch(groups, set(), k=10)
        low_group = next((g for g in result if g["group_id"] == "low"), None)
        assert low_group is not None, "Low confidence group should be selected"
        assert low_group["review_tier"] == "low_confidence"


# ---------------------------------------------------------------------------
# compute_borderline_score (length-bias fix)
# ---------------------------------------------------------------------------


def _alt(triples):
    """Alternative dict from ``(ref, tgt, conf)`` triples."""
    return {
        "total_confidence": round(sum(c for _, _, c in triples), 4),
        "edges": [{"ref_id": r, "target_id": t, "confidence": c} for r, t, c in triples],
    }


class TestComputeBorderlineScore:
    def test_fewer_than_two_alternatives_is_zero(self):
        assert compute_borderline_score([]) == 0.0
        assert compute_borderline_score([_alt([("r", "t", 0.9)])]) == 0.0

    def test_identical_edge_sets_is_zero(self):
        a = _alt([("r", "t", 0.9)])
        b = _alt([("r", "t", 0.9)])
        assert compute_borderline_score([a, b]) == 0.0

    def test_clean_long_chain_scores_near_zero(self):
        """552e0bd5-style: an 8-edge ~0.99 chain vs the same chain minus one edge.

        The old summed-ratio metric scored this ~0.88 (1 - 1/8); the symmetric
        difference is a single ~0.99-confidence edge, which is NOT contested.
        """
        chain = [("r0", f"t{j}", 0.99) for j in range(8)]
        top = _alt(chain)
        runner_up = _alt(chain[:7])  # same chain minus one 0.99 edge
        score = compute_borderline_score([top, runner_up])
        assert score < 0.1

    def test_coin_flip_edge_scores_high(self):
        """Top-2 differ by a genuine 0.5-confidence edge -> maximally contested."""
        top = _alt([("r1", "t1", 0.9), ("r2", "t1", 0.5)])
        runner_up = _alt([("r1", "t1", 0.9)])
        score = compute_borderline_score([top, runner_up])
        assert score > 0.9

    def test_clean_chain_does_not_outrank_coin_flip(self):
        """The core regression: a clean long chain must NOT look more borderline
        than a genuinely contested small group."""
        chain = [("r0", f"t{j}", 0.99) for j in range(8)]
        clean_chain = compute_borderline_score([_alt(chain), _alt(chain[:7])])

        coin_flip = compute_borderline_score(
            [_alt([("ra", "tx", 0.8), ("rb", "tx", 0.42)]), _alt([("ra", "tx", 0.8)])]
        )
        assert coin_flip > clean_chain
        # And only the coin-flip clears the borderline-tier bar.
        assert clean_chain < BORDERLINE_MIN_SCORE <= coin_flip

    def test_max_not_length_biased(self):
        """Differing by many certain edges must not beat one 0.5 edge (no length bias)."""
        many_certain = compute_borderline_score(
            [
                _alt([("r", f"t{j}", 0.99) for j in range(6)]),
                _alt([("r", f"t{j}", 0.99) for j in range(1)]),
            ]
        )
        one_coin_flip = compute_borderline_score(
            [_alt([("r", "t0", 0.99), ("r", "t1", 0.5)]), _alt([("r", "t0", 0.99)])]
        )
        assert one_coin_flip > many_certain


# ---------------------------------------------------------------------------
# select_stitching_batch — human-queue gating (candidate_group_ids)
# ---------------------------------------------------------------------------


class TestSelectStitchingBatchGating:
    def _grp(self, gid, conf=0.5):
        return _make_group(
            gid, [_edge(f"r{gid}", f"t{gid}", conf)], [_alt([(f"r{gid}", f"t{gid}", conf)])]
        )

    def test_gating_restricts_to_candidates(self):
        groups = [self._grp(f"g{i}") for i in range(10)]
        result = select_stitching_batch(groups, set(), k=10, candidate_group_ids={"g2", "g5"})
        ids = {g["group_id"] for g in result}
        assert ids == {"g2", "g5"}

    def test_gating_still_excludes_reviewed(self):
        groups = [self._grp(f"g{i}") for i in range(5)]
        result = select_stitching_batch(groups, {"g1"}, k=10, candidate_group_ids={"g1", "g2"})
        ids = {g["group_id"] for g in result}
        assert ids == {"g2"}  # g1 is a candidate but already reviewed

    def test_empty_candidate_set_yields_empty_queue(self):
        groups = [self._grp(f"g{i}") for i in range(5)]
        assert select_stitching_batch(groups, set(), k=10, candidate_group_ids=set()) == []

    def test_none_candidates_considers_all(self):
        groups = [self._grp(f"g{i}") for i in range(5)]
        result = select_stitching_batch(groups, set(), k=10, candidate_group_ids=None)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# panel_routing — human-queue panel-failure gate
# ---------------------------------------------------------------------------


class TestPanelRouting:
    def _write_consensus(self, batch_dir, rows, mtime=None):
        import csv as _csv

        batch_dir.mkdir(parents=True, exist_ok=True)
        cpath = batch_dir / "consensus.csv"
        with open(cpath, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["group_id", "routing"])
            w.writeheader()
            for gid, routing in rows:
                w.writerow({"group_id": gid, "routing": routing})
        if mtime is not None:
            import os

            os.utime(cpath, (mtime, mtime))
        return cpath

    def test_failed_ids_are_human_review(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import panel_failed_group_ids

        root = tmp_path / "batches"
        self._write_consensus(
            root / "ds_a", [("g1", "human_review"), ("g2", "auto_accept")], mtime=1000
        )
        assert panel_failed_group_ids("ds_a", root) == {"g1"}

    def test_prefix_matches_dataset_dirs_only(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import panel_failed_group_ids

        root = tmp_path / "batches"
        self._write_consensus(root / "ds_a", [("g1", "human_review")], mtime=1000)
        self._write_consensus(root / "ds_a_phase2", [("g2", "human_review")], mtime=1001)
        # A different dataset that is NOT a prefix match must be ignored.
        self._write_consensus(root / "other", [("g9", "human_review")], mtime=1002)
        assert panel_failed_group_ids("ds_a", root) == {"g1", "g2"}

    def test_most_recent_vote_wins(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import (
            latest_panel_routing,
            panel_failed_group_ids,
        )

        root = tmp_path / "batches"
        # Older wave: g1 failed. Newer wave: g1 now auto_accepts.
        self._write_consensus(root / "ds_a", [("g1", "human_review")], mtime=1000)
        self._write_consensus(root / "ds_a_phase2", [("g1", "auto_accept")], mtime=2000)
        assert latest_panel_routing("ds_a", root)["g1"] == "auto_accept"
        assert panel_failed_group_ids("ds_a", root) == set()

    def test_missing_dataset_returns_empty(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import panel_failed_group_ids

        assert panel_failed_group_ids("nope", tmp_path / "batches") == set()


# ---------------------------------------------------------------------------
# panel_routing — route-reason derivation for historical consensus rows
# ---------------------------------------------------------------------------


def _crow(**kw):
    """A consensus.csv-shaped row (all-string values, like csv.DictReader)."""
    row = {
        "group_id": "g1",
        "consensus": "majority",
        "choice": "A",
        "edge_set": "[]",
        "routing": "human_review",
        "n_votes": "3",
        "n_valid": "3",
        "minority": "",
        "mean_confidence": "0.9",
        "route_reason": "",
    }
    row.update({k: str(v) for k, v in kw.items()})
    return row


class TestDeriveRouteReason:
    def _derive(self, **kw):
        from crosswalk.agent_labeling.panel_routing import derive_route_reason

        return derive_route_reason(_crow(**kw))

    def test_auto_accept_is_unanimous(self):
        assert self._derive(consensus="unanimous", routing="auto_accept") == "unanimous"

    def test_unanimous_none(self):
        assert (
            self._derive(consensus="unanimous", choice="NONE", routing="human_review")
            == "unanimous_none"
        )

    def test_majority_with_dissent(self):
        assert self._derive(minority="agy=B") == "dissent:agy=B"

    def test_majority_with_multiple_dissenters_normalized(self):
        # Real minority strings join with "; " (e.g. Seattle 3fcab92c).
        assert self._derive(minority="codex=F; agy=A") == "dissent:codex=F,agy=A"

    def test_majority_no_dissent_below_quorum(self):
        # 2 agree + 1 abstain (unanimity needs >=3 valid).
        assert self._derive(n_valid="2", minority="") == "below_quorum:2"

    def test_majority_no_dissent_with_quorum_is_abstention(self):
        # 4-voter panel: 3 agree + 1 abstain — quorum met, abstention blocked it.
        assert self._derive(n_votes="4", n_valid="3", minority="") == "abstention"

    def test_none_is_no_majority(self):
        assert self._derive(consensus="none", choice="D", minority="agy=A") == "no_majority"

    def test_all_abstained(self):
        assert (
            self._derive(
                consensus="none",
                choice="",
                n_valid="0",
                minority="all providers abstained",
            )
            == "all_abstained"
        )

    def test_existing_reason_wins(self):
        # An informative stamped reason (the class gate's, or phase-2's
        # size_gated) is authoritative — never re-derived.
        assert (
            self._derive(consensus="unanimous", route_reason="class-mismatch") == "class-mismatch"
        )
        assert self._derive(route_reason="size_gated", minority="codex=D") == "size_gated"

    def test_legacy_tier_echo_stamps_are_rederived(self):
        # Phase-2 stamped bare tier echoes ("majority"/"none") — strictly less
        # informative than the row's own columns, so they are re-derived.
        assert self._derive(route_reason="majority", minority="codex=D") == "dissent:codex=D"
        assert self._derive(route_reason="none", consensus="none", minority="agy=A") == (
            "no_majority"
        )
        # Its unanimous_NONE spelling is normalized to the canonical code.
        assert (
            self._derive(route_reason="unanimous_NONE", consensus="unanimous", choice="NONE")
            == "unanimous_none"
        )

    def test_unanimous_non_none_human_review_is_class_mismatch(self):
        # Pre-stamp history: only the class gate demotes a unanimous non-NONE
        # verdict to human_review.
        assert (
            self._derive(consensus="unanimous", choice="A", routing="human_review")
            == "class-mismatch"
        )

    def test_nan_and_missing_columns_are_safe(self):
        from crosswalk.agent_labeling.panel_routing import derive_route_reason

        # pandas-shaped row: NaN route_reason/minority, numeric counts.
        row = {
            "consensus": "majority",
            "choice": "A",
            "routing": "human_review",
            "minority": float("nan"),
            "n_votes": 3,
            "n_valid": 2,
            "route_reason": float("nan"),
        }
        assert derive_route_reason(row) == "below_quorum:2"
        assert derive_route_reason({}) == ""

    def test_stamp_matches_derivation(self):
        # The runner's stamped reason and the historical derivation must agree:
        # blank out the stamp and re-derive from the row's own columns.
        from crosswalk.agent_labeling.panel_routing import derive_route_reason
        from crosswalk.agent_labeling.stitch_runner import Vote, compute_consensus

        def vote(provider, choice):
            return Vote(
                group_id="g",
                provider=provider,
                model="m",
                choice=choice,
                confidence=0.9,
                reasoning="",
                edge_set=frozenset({("r1", "t1")}),
            )

        panels = [
            [vote("claude", "A"), vote("codex", "A"), vote("agy", "A")],
            [vote("claude", "NONE"), vote("codex", "NONE"), vote("agy", "NONE")],
            [vote("claude", "A"), vote("codex", "A"), vote("agy", "B")],
            [vote("claude", "A"), vote("codex", "B"), vote("agy", "NONE")],
            [vote("claude", "A"), vote("codex", "A"), vote("agy", "ABSTAIN")],
            [vote("claude", "ABSTAIN"), vote("codex", "ABSTAIN"), vote("agy", "ABSTAIN")],
        ]
        for votes in panels:
            c = compute_consensus(votes)
            assert c.route_reason
            rederived = derive_route_reason(
                {
                    "consensus": c.consensus,
                    "choice": c.choice,
                    "routing": c.routing,
                    "minority": c.minority,
                    "n_votes": c.n_votes,
                    "n_valid": c.n_valid,
                    "route_reason": "",
                }
            )
            assert rederived == c.route_reason


class TestHumanizeRouteReason:
    def test_known_codes(self):
        from crosswalk.agent_labeling.panel_routing import humanize_route_reason as h

        assert h("unanimous_none") == "panel unanimous: none of the options fit"
        assert h("dissent:codex=B") == "codex dissented — voted B"
        assert h("dissent:codex=F,agy=A") == "codex dissented — voted F; agy dissented — voted A"
        assert h("below_quorum:2") == "only 2 valid votes — below quorum"
        assert "cross-mode" in h("class-mismatch")
        assert h("no_majority") == "panel split — no majority choice"
        assert h("all_abstained") == "all panelists abstained"
        assert h("abstention") == "an abstention blocked unanimity"
        assert h("unanimous") == "panel unanimous — auto-accepted"

    def test_unknown_and_blank(self):
        from crosswalk.agent_labeling.panel_routing import humanize_route_reason as h

        assert h("") == ""
        assert h("size_gated") == "over the size gate — too large to auto-accept"
        assert h("some_new_code") == "some new code"


class TestAttachPanelRouteReasons:
    """The stitch-batch queue writer attaches panel_route_reason per group."""

    def _write_consensus_rows(self, batch_dir, rows, mtime=None):
        import csv as _csv

        batch_dir.mkdir(parents=True, exist_ok=True)
        cpath = batch_dir / "consensus.csv"
        with open(cpath, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if mtime is not None:
            import os

            os.utime(cpath, (mtime, mtime))

    def test_attaches_reason_and_human_variant(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import attach_panel_route_reasons

        root = tmp_path / "batches"
        self._write_consensus_rows(
            root / "ds_a",
            [
                _crow(group_id="g1", minority="codex=B"),
                _crow(group_id="g2", consensus="unanimous", choice="NONE"),
            ],
            mtime=1000,
        )
        groups = [{"group_id": "g1"}, {"group_id": "g2"}, {"group_id": "never_voted"}]
        n = attach_panel_route_reasons(groups, "ds_a", root)
        assert n == 2
        assert groups[0]["panel_route_reason"] == "dissent:codex=B"
        assert groups[0]["panel_route_reason_human"] == "codex dissented — voted B"
        assert groups[1]["panel_route_reason"] == "unanimous_none"
        assert groups[1]["panel_route_reason_human"] == ("panel unanimous: none of the options fit")
        # Never-voted groups are left untouched.
        assert "panel_route_reason" not in groups[2]

    def test_latest_wave_reason_wins(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import attach_panel_route_reasons

        root = tmp_path / "batches"
        self._write_consensus_rows(
            root / "ds_a", [_crow(group_id="g1", minority="codex=B")], mtime=1000
        )
        self._write_consensus_rows(
            root / "ds_a_phase2",
            [_crow(group_id="g1", consensus="none", choice="D", minority="agy=A")],
            mtime=2000,
        )
        groups = [{"group_id": "g1"}]
        attach_panel_route_reasons(groups, "ds_a", root)
        assert groups[0]["panel_route_reason"] == "no_majority"

    def test_annotation_only_never_reshapes_groups(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import attach_panel_route_reasons

        root = tmp_path / "batches"
        self._write_consensus_rows(root / "ds_a", [_crow(group_id="g1")], mtime=1000)
        groups = [{"group_id": "g1", "edges": [{"ref_id": "r", "target_id": "t"}]}]
        before_ids = [g["group_id"] for g in groups]
        attach_panel_route_reasons(groups, "ds_a", root)
        assert [g["group_id"] for g in groups] == before_ids
        assert groups[0]["edges"] == [{"ref_id": "r", "target_id": "t"}]

    def test_no_batches_is_noop(self, tmp_path):
        from crosswalk.agent_labeling.panel_routing import attach_panel_route_reasons

        groups = [{"group_id": "g1"}]
        assert attach_panel_route_reasons(groups, "ds_a", tmp_path / "none") == 0
        assert "panel_route_reason" not in groups[0]


# ---------------------------------------------------------------------------
# StitchingLabelStore
# ---------------------------------------------------------------------------


class TestStitchingLabelStore:
    @pytest.fixture
    def store(self, tmp_path):
        from crosswalk.labeling.stitching_store import StitchingLabelStore

        return StitchingLabelStore("test_dataset", labels_dir=tmp_path / "stitching")

    def test_empty_store(self, store):
        assert len(store.df) == 0
        assert store.get_reviewed_group_ids("test_dataset") == set()

    def test_add_and_load(self, store):
        store.add(
            group_id="abc123",
            selected_edges=[{"ref_id": "r1", "target_id": "t1"}],
            match_type="1:N",
            num_refs=1,
            num_targets=2,
            labeler="tester",
            session_id="sess1",
        )
        assert len(store.df) == 1
        assert store.df.iloc[0]["group_id"] == "abc123"
        assert store.df.iloc[0]["match_type"] == "1:N"
        assert store.df.iloc[0]["dataset_id"] == "test_dataset"

    def test_pair_label_defaults_semantics(self, store):
        """A normal add() defaults to pair semantics with empty membership cols."""
        store.add(
            group_id="abc123",
            selected_edges=[{"ref_id": "r1", "target_id": "t1"}],
            match_type="1:N",
            num_refs=1,
            num_targets=1,
            labeler="tester",
            session_id="sess1",
        )
        row = store.df.iloc[0]
        assert row["label_semantics"] == "pair"
        assert row["ref_ids"] == ""
        assert row["target_ids"] == ""

    def test_set_label_round_trip(self, store):
        """A set label persists membership (sorted JSON) with empty selected_edges."""
        store.add(
            group_id="gset",
            selected_edges=[],
            match_type="M:N",
            num_refs=2,
            num_targets=3,
            labeler="brad",
            session_id="sess2",
            label_semantics="set",
            ref_ids=["rB", "rA"],
            target_ids=["t3", "t1", "t2"],
        )
        # Reload from a fresh instance to exercise CSV persistence + _ensure_schema.
        from crosswalk.labeling.stitching_store import StitchingLabelStore

        reloaded = StitchingLabelStore("test_dataset", labels_dir=store.labels_dir)
        row = reloaded.df.iloc[0]
        assert row["label_semantics"] == "set"
        assert row["selected_edges"] == "[]"
        assert json.loads(row["ref_ids"]) == ["rA", "rB"]  # sorted
        assert json.loads(row["target_ids"]) == ["t1", "t2", "t3"]

    def test_legacy_csv_missing_columns_defaults_to_pair(self, store):
        """A CSV written before the set-semantics columns loads as a pair label."""
        store.partition_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "group_id": "gold",
                    "dataset_id": "test_dataset",
                    "selected_edges": "[]",
                    "match_type": "1:N",
                    "num_refs": 1,
                    "num_targets": 1,
                    "labeler": "brad",
                    "labeled_at": "2026-01-01T00:00:00+00:00",
                    "session_id": "s",
                }
            ]
        ).to_csv(store.csv_path, index=False)
        row = store.df.iloc[0]
        assert row["label_semantics"] == "pair"
        assert row["ref_ids"] == ""

    def test_dedup_replaces_on_same_group_id(self, store):
        for i in range(3):
            store.add(
                group_id="abc123",
                selected_edges=[{"ref_id": f"r{i}", "target_id": "t1"}],
                match_type="N:1",
                num_refs=2,
                num_targets=1,
                labeler="tester",
                session_id=f"sess{i}",
            )
        assert len(store.df) == 1
        assert store.df.iloc[0]["session_id"] == "sess2"

    def test_get_reviewed_group_ids(self, store):
        store.add("g1", [], "1:N", 1, 2, "tester", "s1")
        store.add("g2", [], "N:1", 2, 1, "tester", "s2")
        reviewed = store.get_reviewed_group_ids("test_dataset")
        assert reviewed == {"g1", "g2"}

    def test_get_reviewed_filters_by_dataset(self, store):
        from crosswalk.labeling.stitching_store import StitchingLabelStore

        store.add("g1", [], "1:N", 1, 2, "tester", "s1")
        # Second store for a different dataset
        store2 = StitchingLabelStore("other_dataset", labels_dir=store.labels_dir)
        store2.add("g2", [], "1:N", 1, 2, "tester", "s2")
        assert store.get_reviewed_group_ids("test_dataset") == {"g1"}
        assert store2.get_reviewed_group_ids("other_dataset") == {"g2"}

    def test_selected_edges_stored_as_json(self, store):
        edges = [{"ref_id": "r1", "target_id": "t1"}, {"ref_id": "r1", "target_id": "t2"}]
        store.add("g1", edges, "1:N", 1, 2, "tester", "s1")
        raw = store.df.iloc[0]["selected_edges"]
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["ref_id"] == "r1"

    def test_persistence_across_instances(self, store):
        from crosswalk.labeling.stitching_store import StitchingLabelStore

        store.add("g1", [], "1:N", 1, 2, "tester", "s1")

        # New instance reads from disk
        store2 = StitchingLabelStore("test_dataset", labels_dir=store.labels_dir)
        assert len(store2.df) == 1
        assert store2.df.iloc[0]["group_id"] == "g1"

    def test_atomic_backup_exists_after_save(self, store):
        store.add("g1", [], "1:N", 1, 2, "tester", "s1")
        # First save creates no backup (no prior file)
        assert store.csv_path.exists()
        # Second save creates backup
        store.add("g2", [], "N:1", 2, 1, "tester", "s2")
        backup = store.csv_path.with_suffix(".csv.bak")
        assert backup.exists()


# ---------------------------------------------------------------------------
# Stitching label data integrity
# ---------------------------------------------------------------------------

# Discover all stitching label datasets on disk
_STITCHING_DATASETS = sorted(
    p.parent.name.removeprefix("dataset=") for p in DEFAULT_STITCHING_DIR.glob("dataset=*/data.csv")
)


@pytest.mark.skipif(not _STITCHING_DATASETS, reason="no stitching labels on disk")
class TestStitchingLabelIntegrity:
    """Ensure committed stitching labels are well-formed and internally consistent."""

    @pytest.fixture(params=_STITCHING_DATASETS)
    def label_df(self, request):
        # Load through the store so the effective (schema-ensured) frame is
        # tested: committed CSVs predating the set-semantics columns are filled
        # with defaults on load, and the set columns migrate lazily on next save.
        from crosswalk.labeling.stitching_store import StitchingLabelStore

        dataset = request.param
        return StitchingLabelStore(dataset).load(dataset)

    def test_has_required_columns(self, label_df):
        missing = set(STITCHING_LABEL_COLUMNS) - set(label_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_empty_rows(self, label_df):
        assert len(label_df) > 0, "Label file exists but has no rows"

    def test_group_ids_unique(self, label_df):
        dupes = label_df["group_id"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate group_ids"

    def test_match_types_valid(self, label_df):
        valid = {"1:1", "1:N", "N:1", "M:N"}
        actual = set(label_df["match_type"].unique())
        invalid = actual - valid
        assert not invalid, f"Invalid match_type values: {invalid}"

    def test_selected_edges_parseable(self, label_df):
        for idx, row in label_df.iterrows():
            edges = json.loads(row["selected_edges"])
            assert isinstance(edges, list), f"Row {idx}: edges is not a list"
            for edge in edges:
                assert "ref_id" in edge, f"Row {idx}: edge missing ref_id"
                assert "target_id" in edge, f"Row {idx}: edge missing target_id"

    def test_num_refs_and_targets_non_negative(self, label_df):
        """Pair labels with edges must have positive counts; pair rejections
        (empty edges) have 0. SET labels store no edges but carry the membership
        counts, which must match ref_ids/target_ids."""
        is_set = label_df.get("label_semantics", "pair").astype(str) == "set"
        pair_df = label_df[~is_set]
        has_edges = pair_df["selected_edges"].apply(lambda x: len(json.loads(x)) > 0)
        with_edges = pair_df[has_edges]
        if len(with_edges) > 0:
            assert (with_edges["num_refs"] >= 1).all(), "num_refs must be >= 1 when edges selected"
            assert (with_edges["num_targets"] >= 1).all(), (
                "num_targets must be >= 1 when edges selected"
            )
        # Pair rejections should have 0 refs and 0 targets.
        rejections = pair_df[~has_edges]
        if len(rejections) > 0:
            assert (rejections["num_refs"] == 0).all(), "num_refs must be 0 for rejections"
            assert (rejections["num_targets"] == 0).all(), "num_targets must be 0 for rejections"
        # Set labels: counts equal the stored membership sizes.
        for _, row in label_df[is_set].iterrows():
            assert row["num_refs"] == len(json.loads(row["ref_ids"] or "[]"))
            assert row["num_targets"] == len(json.loads(row["target_ids"] or "[]"))


# ---------------------------------------------------------------------------
# stitch-batch retains alternatives + optimizer_assignment
# ---------------------------------------------------------------------------


class TestStitchBatchRetainsOptionData:
    """The batch file must carry the data the option-picking UI depends on.

    Mirrors what the `stitch-batch` CLI does (compute alternatives, then select
    a batch) and asserts both `alternatives` and `optimizer_assignment` survive
    onto the selected groups — they must NOT be stripped anymore.
    """

    def _synthetic_sidecar_group(self, gid):
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r1", "t2", 0.4),
            _edge("r2", "t1", 0.3),
            _edge("r2", "t2", 0.85),
        ]
        return {
            "group_id": gid,
            "match_type": "M:N",
            "ref_ids": ["r1", "r2"],
            "target_ids": ["t1", "t2"],
            "edges": edges,
            # The optimizer's own proposed assignment (from the runner sidecar)
            "optimizer_assignment": [
                {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                {"ref_id": "r2", "target_id": "t2", "confidence": 0.85},
            ],
        }

    def test_alternatives_and_optimizer_assignment_retained(self):
        groups = [self._synthetic_sidecar_group(f"g{i}") for i in range(3)]
        # Replicate the CLI's compute step
        for g in groups:
            g["alternatives"] = generate_top_k_alternatives(
                component_edges=g["edges"],
                ref_geoms={},
                target_geoms={},
                k=5,
            )
        selected = select_stitching_batch(groups, set(), k=5)
        assert selected
        for g in selected:
            assert g.get("alternatives"), "alternatives must be retained on batch groups"
            assert g.get("optimizer_assignment"), (
                "optimizer_assignment must be retained on batch groups"
            )

    def test_alternatives_carry_no_geometries(self):
        """Serialized alternatives are just ID pairs + confidence (small)."""
        alts = generate_top_k_alternatives(
            component_edges=self._synthetic_sidecar_group("g")["edges"], k=5
        )
        blob = json.dumps(alts)
        assert "coordinates" not in blob
        assert "geometry" not in blob


# ---------------------------------------------------------------------------
# _build_stitch_options — option picker + optimizer pre-seed
# ---------------------------------------------------------------------------


class TestBuildStitchOptions:
    def _mn_group(self, **overrides):
        group = {
            "group_id": "g1",
            "match_type": "M:N",
            "ref_ids": ["r1", "r2"],
            "target_ids": ["t1", "t2"],
            "edges": [
                _edge("r1", "t1", 0.9),
                _edge("r1", "t2", 0.4),
                _edge("r2", "t1", 0.3),
                _edge("r2", "t2", 0.85),
            ],
            "optimizer_assignment": [
                {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                {"ref_id": "r2", "target_id": "t2", "confidence": 0.85},
            ],
            "alternatives": [],
        }
        group.update(overrides)
        return group

    def test_preseed_from_optimizer_assignment(self):
        from crosswalk.web.routes.stitching import _build_stitch_options

        ctx = _build_stitch_options(self._mn_group())
        assert ctx["has_preseed"] is True
        assert ctx["preseed_active_refs"] == ["r1", "r2"]
        assert ctx["preseed_active_targets"] == ["t1", "t2"]
        # First option is the optimizer's, carrying its exact edge set
        assert ctx["options"][0]["is_optimizer"] is True
        opt_edges = {(e["ref_id"], e["target_id"]) for e in ctx["options"][0]["edges"]}
        assert opt_edges == {("r1", "t1"), ("r2", "t2")}
        # preseed_edges mirrors the optimizer option so an unmodified pick
        # submits exactly that edge set
        assert ctx["preseed_edges"] == ctx["options"][0]["edges"]

    def test_missing_optimizer_assignment_falls_back(self):
        """Old-format group without optimizer_assignment -> no pre-seed."""
        from crosswalk.web.routes.stitching import _build_stitch_options

        group = self._mn_group()
        del group["optimizer_assignment"]
        ctx = _build_stitch_options(group)
        assert ctx["has_preseed"] is False
        assert ctx["preseed_active_refs"] is None
        assert ctx["preseed_active_targets"] is None
        assert ctx["preseed_inactive_ids"] == []
        # No optimizer option present
        assert all(not o["is_optimizer"] for o in ctx["options"])

    def test_empty_optimizer_assignment_falls_back(self):
        """An empty optimizer_assignment (optimizer dropped the group) -> no pre-seed."""
        from crosswalk.web.routes.stitching import _build_stitch_options

        ctx = _build_stitch_options(self._mn_group(optimizer_assignment=[]))
        assert ctx["has_preseed"] is False
        assert ctx["preseed_active_refs"] is None

    def test_inactive_segment_ids_derived(self):
        """Group segments the optimizer left out are reported inactive."""
        from crosswalk.web.routes.stitching import _build_stitch_options

        group = self._mn_group(
            ref_ids=["r1", "r2", "r3"],
            target_ids=["t1"],
            edges=[
                _edge("r1", "t1", 0.9),
                _edge("r2", "t1", 0.8),
                _edge("r3", "t1", 0.2),
            ],
            optimizer_assignment=[
                {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                {"ref_id": "r2", "target_id": "t1", "confidence": 0.8},
            ],
        )
        ctx = _build_stitch_options(group)
        assert ctx["preseed_active_refs"] == ["r1", "r2"]
        assert ctx["preseed_inactive_ids"] == ["r3"]

    def test_options_deduplicated_against_optimizer(self):
        """An alternative identical to the optimizer's answer is not duplicated."""
        from crosswalk.web.routes.stitching import _build_stitch_options

        group = self._mn_group(
            alternatives=[
                {
                    "edges": [
                        {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                        {"ref_id": "r2", "target_id": "t2", "confidence": 0.85},
                    ],
                    "total_confidence": 1.75,
                },
                {
                    "edges": [
                        {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                    ],
                    "total_confidence": 0.9,
                },
            ]
        )
        ctx = _build_stitch_options(group)
        edge_sets = [
            frozenset((e["ref_id"], e["target_id"]) for e in o["edges"]) for o in ctx["options"]
        ]
        # The optimizer's edge set appears exactly once despite the matching alt
        assert edge_sets.count(frozenset({("r1", "t1"), ("r2", "t2")})) == 1


# ---------------------------------------------------------------------------
# Junction sliver classification (hybrid fraction + absolute-meters rule)
# ---------------------------------------------------------------------------


def _frac_edge(ref, tgt, gs, ge, ls, le, conf=0.5):
    return {
        "ref_id": ref,
        "target_id": tgt,
        "confidence": conf,
        "gers_start_frac": gs,
        "gers_end_frac": ge,
        "local_start_frac": ls,
        "local_end_frac": le,
    }


def _line_of_length_m(length_m, lat=42.36, lon=-71.06):
    """Build a roughly east-west WGS84 LineString of ~``length_m`` meters."""
    import math

    deg = length_m / (111000.0 * math.cos(math.radians(lat)))
    return {"type": "LineString", "coordinates": [[lon, lat], [lon + deg, lat]]}


class TestSliverConfigRule:
    """The centralized numeric hybrid rule in ``crosswalk.config.is_sliver_edge``.

    Sliver iff BOTH: max(ref_span, tgt_span) < 0.10 (fraction gate) AND
    max(ref_span*ref_len, tgt_span*tgt_len) < 5.0 m (absolute-overlap gate).
    """

    def test_threshold_values(self):
        from crosswalk.config import SLIVER_ABS_OVERLAP_M, SLIVER_SPAN_THRESHOLD

        assert SLIVER_SPAN_THRESHOLD == 0.10
        assert SLIVER_ABS_OVERLAP_M == 5.0

    def test_tiny_overlap_short_segments_is_sliver(self):
        from crosswalk.config import is_sliver_edge

        # 0.2 m of a 162 m ref (0.0012*162=0.19 m) and 0.67 m of a 10 m target:
        # both fraction and absolute gates pass -> sliver.
        assert is_sliver_edge(0.0012, 0.067, 162.0, 10.0)

    def test_long_ref_nine_percent_not_sliver(self):
        from crosswalk.config import is_sliver_edge

        # 9% of a 2 km ref = 180 m of real road. Passes the fraction gate
        # (0.09 < 0.10) but the 180 m absolute overlap is far above 5 m, so the
        # hybrid rule keeps it as a SUBSTANTIVE edge (the key fix over frac-only).
        assert not is_sliver_edge(0.09, 0.05, 2000.0, 50.0)

    def test_short_stub_large_fraction_not_sliver_residual_limit(self):
        from crosswalk.config import is_sliver_edge

        # 0.6 m of a 4 m stub = 15% span. Only 0.6 m physically overlaps, yet the
        # AND rule requires the fraction gate and 0.15 is NOT < 0.10, so this is
        # (documented) NOT a sliver -- the residual limitation of the AND rule.
        assert not is_sliver_edge(0.15, 0.15, 4.0, 4.0)

    def test_boundary_010_not_sliver(self):
        from crosswalk.config import is_sliver_edge

        # Fraction exactly at threshold: strict < excludes it regardless of length.
        assert not is_sliver_edge(0.10, 0.10, 1.0, 1.0)

    def test_uses_max_not_min_fraction(self):
        from crosswalk.config import is_sliver_edge

        # One large span keeps it substantive even if the other is tiny.
        assert not is_sliver_edge(0.5, 0.001, 100.0, 100.0)

    def test_missing_fracs_default_substantive(self):
        from crosswalk.config import is_sliver_edge

        assert not is_sliver_edge(None, None, 10.0, 10.0)

    def test_nan_fracs_default_substantive(self):
        import math

        from crosswalk.config import is_sliver_edge

        assert not is_sliver_edge(math.nan, math.nan, 10.0, 10.0)

    def test_missing_lengths_never_sliver(self):
        from crosswalk.config import is_sliver_edge

        # Tiny fractions but unknown lengths -> absolute overlap unknown (+inf)
        # -> never drop what we cannot measure.
        assert not is_sliver_edge(0.001, 0.001)


class TestSliverGroupHelpers:
    """Group/edge-dict helpers in ``crosswalk.matching.sliver``."""

    def test_explicit_null_fracs_never_sliver(self):
        from crosswalk.matching.sliver import edge_is_sliver, edge_span_fracs

        edge = {"gers_end_frac": None, "local_start_frac": None}
        assert edge_span_fracs(edge) == (1.0, 1.0)
        assert not edge_is_sliver(edge)

    def test_edge_is_sliver_with_group_lengths(self):
        from crosswalk.matching.sliver import edge_is_sliver, group_segment_lengths_m

        group = {
            "ref_geometries": {"r": _line_of_length_m(162.0)},
            "target_geometries": {"t": _line_of_length_m(10.0)},
        }
        ref_lens, tgt_lens = group_segment_lengths_m(group)
        edge = _frac_edge("r", "t", 0.0, 0.0012, 0.0, 0.067)
        assert edge_is_sliver(edge, ref_lens, tgt_lens)

    def test_long_ref_overlap_not_sliver(self):
        from crosswalk.matching.sliver import edge_is_sliver, group_segment_lengths_m

        group = {
            "ref_geometries": {"r": _line_of_length_m(2000.0)},
            "target_geometries": {"t": _line_of_length_m(50.0)},
        }
        ref_lens, tgt_lens = group_segment_lengths_m(group)
        # 9% of a 2 km ref = 180 m real road -> substantive despite small fraction.
        edge = _frac_edge("r", "t", 0.0, 0.09, 0.0, 0.05)
        assert not edge_is_sliver(edge, ref_lens, tgt_lens)

    def test_annotate_adds_flag_and_count(self):
        from crosswalk.matching.sliver import annotate_group_sliver_flags

        group = {
            "edges": [
                _frac_edge("r", "t1", 0.0, 1.0, 0.0, 1.0),  # substantive
                _frac_edge("r", "t2", 0.0, 0.02, 0.0, 0.05),  # sliver (short segs)
            ],
            "ref_geometries": {"r": _line_of_length_m(50.0)},
            "target_geometries": {
                "t1": _line_of_length_m(50.0),
                "t2": _line_of_length_m(10.0),
            },
        }
        annotated, count = annotate_group_sliver_flags(group)
        assert annotated[0]["is_sliver"] is False
        assert annotated[1]["is_sliver"] is True
        assert count == 1
        # Original edges are not mutated.
        assert "is_sliver" not in group["edges"][0]


class TestSliverOverlapAndBorderline:
    """DISPLAY-ONLY helpers: absolute overlap meters + the BORDERLINE band.

    BORDERLINE (pack display only, never consumed by the optimizer or label
    gates): an edge that is NOT a strict sliver yet whose larger coverage
    fraction is below ``SLIVER_BORDERLINE_SPAN_THRESHOLD`` (1.5x the sliver span
    threshold). Captures the junction-kiss edges the strict rule leaves untagged
    on long urban segments (fail the sliver test only on the 5 m absolute floor)
    plus those sitting just above the span threshold.
    """

    def test_overlap_m_single_definition(self):
        # The config-level overlap must match the sliver rule's absolute gate.
        from crosswalk.config import SLIVER_ABS_OVERLAP_M, is_sliver_edge, sliver_overlap_m

        # 9% of a 2 km ref = 180 m absolute overlap.
        ov = sliver_overlap_m(0.09, 0.05, 2000.0, 50.0)
        assert ov == pytest.approx(180.0)
        # 180 m overlap is above the 5 m floor, and the rule agrees (not a sliver).
        assert ov >= SLIVER_ABS_OVERLAP_M
        assert not is_sliver_edge(0.09, 0.05, 2000.0, 50.0)
        # Tiny short-segment overlap is below the floor -> classified sliver.
        assert sliver_overlap_m(0.0012, 0.067, 162.0, 10.0) < SLIVER_ABS_OVERLAP_M
        assert is_sliver_edge(0.0012, 0.067, 162.0, 10.0)

    def test_overlap_m_missing_length_is_inf(self):
        import math

        from crosswalk.config import sliver_overlap_m

        assert math.isinf(sliver_overlap_m(0.01, 0.01))

    def test_edge_overlap_m_from_group_lengths(self):
        from crosswalk.matching.sliver import edge_overlap_m, group_segment_lengths_m

        group = {
            "ref_geometries": {"r": _line_of_length_m(200.0)},
            "target_geometries": {"t": _line_of_length_m(50.0)},
        }
        ref_lens, tgt_lens = group_segment_lengths_m(group)
        # 2.9% of a 200 m ref = ~5.8 m absolute overlap (the Berlin R8->T3 regime).
        edge = _frac_edge("r", "t", 0.0, 0.029, 0.0, 0.025)
        assert edge_overlap_m(edge, ref_lens, tgt_lens) == pytest.approx(5.8, abs=0.1)

    def test_borderline_when_sliver_blocked_only_by_abs_floor(self):
        # The named case: 2.9% span on a long ref maps to >5 m, so the strict
        # rule does NOT tag it a sliver, but it IS the contested junction-kiss.
        from crosswalk.matching.sliver import (
            edge_is_borderline,
            edge_is_sliver,
            edge_sliver_tag,
            group_segment_lengths_m,
        )

        group = {
            "ref_geometries": {"r": _line_of_length_m(200.0)},
            "target_geometries": {"t": _line_of_length_m(50.0)},
        }
        ref_lens, tgt_lens = group_segment_lengths_m(group)
        edge = _frac_edge("r", "t", 0.0, 0.029, 0.0, 0.025)
        assert not edge_is_sliver(edge, ref_lens, tgt_lens)
        assert edge_is_borderline(edge, ref_lens, tgt_lens)
        assert edge_sliver_tag(edge, ref_lens, tgt_lens) == "BORDERLINE"

    def test_borderline_just_above_span_threshold(self):
        # Span 0.12 (> 0.10 sliver gate, < 0.15 band) with short segments -> BORDERLINE.
        from crosswalk.matching.sliver import edge_is_borderline, edge_is_sliver

        group = {
            "ref_geometries": {"r": _line_of_length_m(30.0)},
            "target_geometries": {"t": _line_of_length_m(30.0)},
        }
        from crosswalk.matching.sliver import group_segment_lengths_m

        ref_lens, tgt_lens = group_segment_lengths_m(group)
        edge = _frac_edge("r", "t", 0.0, 0.12, 0.0, 0.12)
        assert not edge_is_sliver(edge, ref_lens, tgt_lens)
        assert edge_is_borderline(edge, ref_lens, tgt_lens)

    def test_sliver_is_never_also_borderline(self):
        from crosswalk.matching.sliver import (
            edge_is_borderline,
            edge_is_sliver,
            edge_sliver_tag,
            group_segment_lengths_m,
        )

        group = {
            "ref_geometries": {"r": _line_of_length_m(50.0)},
            "target_geometries": {"t": _line_of_length_m(10.0)},
        }
        ref_lens, tgt_lens = group_segment_lengths_m(group)
        edge = _frac_edge("r", "t", 0.0, 0.02, 0.0, 0.05)  # true sliver
        assert edge_is_sliver(edge, ref_lens, tgt_lens)
        assert not edge_is_borderline(edge, ref_lens, tgt_lens)
        assert edge_sliver_tag(edge, ref_lens, tgt_lens) == "SLIVER"

    def test_substantive_asymmetric_match_not_borderline(self):
        # A large coverage on one side (45%) is a legitimate asymmetric match,
        # not a junction-kiss, even if the absolute overlap is small.
        from crosswalk.matching.sliver import edge_is_borderline, edge_sliver_tag

        group = {
            "ref_geometries": {"r": _line_of_length_m(11.0)},
            "target_geometries": {"t": _line_of_length_m(220.0)},
        }
        from crosswalk.matching.sliver import group_segment_lengths_m

        ref_lens, tgt_lens = group_segment_lengths_m(group)
        edge = _frac_edge("r", "t", 0.0, 0.457, 0.0, 0.023)
        assert not edge_is_borderline(edge, ref_lens, tgt_lens)
        assert edge_sliver_tag(edge, ref_lens, tgt_lens) is None


# ---------------------------------------------------------------------------
# _parse_explicit_edges — edge-set fidelity validation
# ---------------------------------------------------------------------------


class TestParseExplicitEdges:
    def _group(self):
        return {
            "edges": [
                _edge("r1", "t1", 0.9),
                _edge("r2", "t2", 0.85),
            ]
        }

    def test_empty_payload_returns_none(self):
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        assert _parse_explicit_edges("", self._group()) is None

    def test_valid_payload_stripped_to_id_pairs(self):
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        raw = json.dumps([{"ref_id": "r1", "target_id": "t1", "confidence": 0.9}])
        parsed = _parse_explicit_edges(raw, self._group())
        assert parsed == [{"ref_id": "r1", "target_id": "t1"}]

    def test_non_group_edge_rejected(self):
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        raw = json.dumps([{"ref_id": "r1", "target_id": "t9"}])
        with pytest.raises(ValueError):
            _parse_explicit_edges(raw, self._group())

    def test_malformed_json_rejected(self):
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        with pytest.raises(ValueError):
            _parse_explicit_edges("not json", self._group())

    def test_non_list_rejected(self):
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        with pytest.raises(ValueError):
            _parse_explicit_edges(json.dumps({"ref_id": "r1"}), self._group())

    def _group_with_rejected(self):
        # A rejected candidate over the group's own segments (same structural
        # layer as selected edges). The submit-time pair-confirmation panel can
        # present such a pair for the reviewer to tick, so an explicit payload
        # referencing it must be accepted (validated against the candidate union).
        return {
            "edges": [_edge("r1", "t1", 0.9)],
            "rejected_edges": [_edge("r2", "t2", 0.42)],
        }

    def test_rejected_candidate_pair_accepted(self):
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        raw = json.dumps([{"ref_id": "r2", "target_id": "t2"}])
        parsed = _parse_explicit_edges(raw, self._group_with_rejected())
        assert parsed == [{"ref_id": "r2", "target_id": "t2"}]

    def test_true_non_candidate_still_rejected(self):
        # Not a selected edge nor a rejected candidate → still refused (#270:
        # context/forged pairs must never be recordable).
        from crosswalk.web.routes.stitching import _parse_explicit_edges

        raw = json.dumps([{"ref_id": "r1", "target_id": "t9"}])
        with pytest.raises(ValueError):
            _parse_explicit_edges(raw, self._group_with_rejected())


# ---------------------------------------------------------------------------
# /stitching-review/select — explicit edge set vs. cross-product
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="fastapi not installed")


class TestStitchingSelectRoute:
    """Edge-set fidelity of the submit endpoint (M:N group)."""

    DATASET = "test_ds"

    def _batch(self):
        # Full 2x2 cross-product so an option is a strict subset of the
        # cross-product of its endpoints.
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gmn",
                    "match_type": "M:N",
                    "ref_ids": ["r1", "r2"],
                    "target_ids": ["t1", "t2"],
                    "edges": [
                        _edge("r1", "t1", 0.9),
                        _edge("r1", "t2", 0.4),
                        _edge("r2", "t1", 0.3),
                        _edge("r2", "t2", 0.85),
                    ],
                }
            ],
        }

    def _client_and_recorder(self):
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        recorder = MagicMock()
        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch("crosswalk.web.routes.stitching.get_unreviewed_stitch_groups", return_value=[]),
            patch("crosswalk.web.routes.stitching.record_stitching_label", recorder),
        ]
        for p in patches:
            p.start()
        client = TestClient(create_app())
        return client, recorder, patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_explicit_edges_stored_verbatim(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "t1,t2",
                    "selected_edges": json.dumps(
                        [
                            {"ref_id": "r1", "target_id": "t1"},
                            {"ref_id": "r2", "target_id": "t2"},
                        ]
                    ),
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            stored = {(e["ref_id"], e["target_id"]) for e in kwargs["selected_edges"]}
            # Exactly the option's 2 edges — NOT the 4-edge cross product
            assert stored == {("r1", "t1"), ("r2", "t2")}
            assert kwargs["num_refs"] == 2
            assert kwargs["num_targets"] == 2
        finally:
            self._stop(patches)

    def test_manual_records_set_membership_not_cross_product(self):
        """Manual mode (no explicit edges) records a SET label: membership only,
        empty selected_edges — NOT the ref×target cross-product it used to."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "t1,t2",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []  # no expanded pairs
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
            assert kwargs["num_refs"] == 2
            assert kwargs["num_targets"] == 2
        finally:
            self._stop(patches)

    def test_rejects_non_group_edge(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1",
                    "included_targets": "t1",
                    "selected_edges": json.dumps([{"ref_id": "r1", "target_id": "t9"}]),
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_manual_toggle_records_active_pill_membership(self):
        """A manual pill edit clears selected_edges; the server records the SET
        membership of the (non-empty) active pill fields — not an empty set and
        not the expanded cross-product.

        included_refs/included_targets carry the still-active pills; selected_edges
        is blank. The membership is exactly those pills.
        """
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    # User kept both refs but only target t2 active.
                    "included_refs": "r1,r2",
                    "included_targets": "t2",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t2"}
            assert kwargs["num_refs"] == 2
            assert kwargs["num_targets"] == 1
        finally:
            self._stop(patches)

    def test_deliberate_full_deselect_stores_empty(self):
        """Deselecting EVERYTHING (empty pill fields, blank selected_edges) is a
        legitimate reject-all and must store []."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "",
                    "included_targets": "",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["num_refs"] == 0
            assert kwargs["num_targets"] == 0
            # Reject-all has no membership to overstate -> stays PAIR semantics.
            assert kwargs["label_semantics"] == "pair"
        finally:
            self._stop(patches)

    def test_inconsistent_submission_rejected(self):
        """Active-pill fields claim segments but none resolve to a group edge —
        an inconsistent submission that must 400, not silently store []."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    # r1 is a real ref but the only claimed target is unknown, so
                    # the cross-product against group edges is empty.
                    "included_refs": "r1",
                    "included_targets": "t_ghost",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_refs_active_but_all_targets_deselected_rejected(self):
        """Refs still active while every target is deselected cannot form an
        edge — treated as inconsistent (not a deliberate reject-all)."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_explicit_empty_list_treated_as_no_payload(self):
        """selected_edges='[]' with active pills must not bypass the guard.

        A real option always has >= 1 edge; an explicit empty list falls back
        to the manual-mode path, where non-empty pill fields resolving to zero
        edges are rejected as inconsistent.
        """
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "t1,t2",
                    "selected_edges": "[]",
                },
            )
            # '[]' is treated as no payload -> manual set path over the active
            # pills (membership, empty edges), not 4 expanded pairs.
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)


class TestStitchingSliverExclusion:
    """Set-semantics submits record MEMBERSHIP, so ``exclude_slivers`` no longer
    changes what is stored (there are no per-pair edges to drop). It still rides
    along for the client's confidence display. The explicit OPTION path is a
    curated exact edge set and is unaffected either way."""

    DATASET = "test_ds"

    def _batch(self):
        # 2x2 group where the diagonal is substantive and the off-diagonal is a
        # pair of junction slivers (max span 0.04-0.05 << 0.10, and with ~50 m
        # segments the absolute overlap is <5 m so the hybrid rule flags them).
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gsl",
                    "match_type": "M:N",
                    "ref_ids": ["r1", "r2"],
                    "target_ids": ["t1", "t2"],
                    "edges": [
                        _frac_edge("r1", "t1", 0.0, 1.0, 0.0, 1.0, conf=0.9),  # substantive
                        _frac_edge("r1", "t2", 0.0, 0.02, 0.0, 0.05, conf=0.3),  # sliver
                        _frac_edge("r2", "t1", 0.0, 0.03, 0.0, 0.04, conf=0.3),  # sliver
                        _frac_edge("r2", "t2", 0.0, 1.0, 0.0, 1.0, conf=0.85),  # substantive
                    ],
                    "ref_geometries": {
                        "r1": _line_of_length_m(50.0),
                        "r2": _line_of_length_m(50.0),
                    },
                    "target_geometries": {
                        "t1": _line_of_length_m(50.0),
                        "t2": _line_of_length_m(50.0),
                    },
                }
            ],
        }

    def _client_and_recorder(self):
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        recorder = MagicMock()
        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch("crosswalk.web.routes.stitching.get_unreviewed_stitch_groups", return_value=[]),
            patch("crosswalk.web.routes.stitching.record_stitching_label", recorder),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), recorder, patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def _post(self, client, **extra):
        data = {
            "dataset": self.DATASET,
            "group_id": "gsl",
            "group_index": 0,
            "included_refs": "r1,r2",
            "included_targets": "t1,t2",
            "selected_edges": "",
        }
        data.update(extra)
        return client.post("/stitching-review/select", data=data)

    def test_exclude_true_is_storage_noop_for_set(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(client, exclude_slivers="true")
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            # Set label: membership is all active pills; no edges stored, and the
            # sliver flag does not prune membership.
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)

    def test_field_absent_records_full_membership(self):
        # Old clients / tests that never send the field must be unaffected.
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(client)  # no exclude_slivers field at all
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)

    def test_exclude_false_records_full_membership(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(client, exclude_slivers="false")
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)

    def test_explicit_option_path_unaffected_by_exclusion(self):
        # An option is a curated exact edge set: even a sliver edge is stored
        # verbatim regardless of exclude_slivers.
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(
                client,
                exclude_slivers="true",
                selected_edges=json.dumps([{"ref_id": "r1", "target_id": "t2"}]),
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            stored = {(e["ref_id"], e["target_id"]) for e in kwargs["selected_edges"]}
            assert stored == {("r1", "t2")}  # the sliver survives (verbatim)
        finally:
            self._stop(patches)

    def test_sliver_only_selection_records_membership_not_400(self):
        # Selecting a pairing whose only edge is a sliver is a valid membership
        # (r1 ↔ t2 share candidate edge (r1,t2)); it is recorded as a set label,
        # not rejected and not emptied by sliver exclusion.
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(
                client, exclude_slivers="true", included_refs="r1", included_targets="t2"
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1"}
            assert set(kwargs["target_ids"]) == {"t2"}
        finally:
            self._stop(patches)


class TestStitchingDeepLink:
    """The main page route deep-links a specific group as a FULL page."""

    DATASET = "test_ds"

    def _batch(self):
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gdeep",
                    "match_type": "1:N",
                    "ref_ids": ["r1"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_edge("r1", "t1", 0.9), _edge("r1", "t2", 0.8)],
                }
            ],
        }

    def _client(self, unreviewed):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch(
                "crosswalk.web.routes.stitching.get_unreviewed_stitch_groups",
                return_value=unreviewed,
            ),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_deep_link_renders_full_page_even_when_reviewed(self):
        # Group already reviewed (unreviewed list empty) — deep link still works
        client, patches = self._client(unreviewed=[])
        try:
            resp = client.get(f"/stitching-review?dataset={self.DATASET}&group_id=gdeep")
            assert resp.status_code == 200
            # Full page (styles + map container), not the bare fragment
            assert "app.css" in resp.text
            assert 'id="map"' in resp.text
            assert "gdeep" in resp.text
            # Progress reflects the group's actual batch position, not the
            # reviewed count ("Group 1 of 1", never "Group 2 of 1")
            assert "Group 1 of 1" in resp.text
        finally:
            self._stop(patches)

    def test_deep_link_unknown_group_404s_without_reflecting_id(self):
        client, patches = self._client(unreviewed=[])
        try:
            resp = client.get(
                f"/stitching-review?dataset={self.DATASET}&group_id=<script>x</script>"
            )
            assert resp.status_code == 404
            assert "<script>x</script>" not in resp.text
        finally:
            self._stop(patches)


class TestStitchingReviewNavigation:
    """Position-based navigation walks ONLY the unreviewed queue.

    A reload of ``/stitching-review?dataset=...`` must land on the first
    unreviewed group, and next/skip/save-advance must never re-serve a group the
    reviewer already completed. The "N of M" counter is queue-relative (position
    within the unreviewed list / total unreviewed). Deep links by ``group_id``
    are the deliberate exception: they address the full batch (incl. reviewed
    groups) and report batch position / batch total.
    """

    DATASET = "test_ds"

    def _batch(self):
        # Four groups in batch order g0, g1, g2, g3.
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": f"g{i}",
                    "match_type": "1:N",
                    "ref_ids": [f"r{i}"],
                    "target_ids": [f"t{i}"],
                    "edges": [_edge(f"r{i}", f"t{i}", 0.9)],
                }
                for i in range(4)
            ],
        }

    def _group(self, gid):
        return next(g for g in self._batch()["groups"] if g["group_id"] == gid)

    def _client(self, unreviewed, recorder=None):
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        recorder = recorder or MagicMock()
        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch(
                "crosswalk.web.routes.stitching.get_unreviewed_stitch_groups",
                return_value=unreviewed,
            ),
            patch("crosswalk.web.routes.stitching.record_stitching_label", recorder),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), recorder, patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_reload_lands_on_first_unreviewed_with_queue_counter(self):
        # g0 and g2 already reviewed -> unreviewed queue is [g1, g3].
        unreviewed = [self._group("g1"), self._group("g3")]
        client, _, patches = self._client(unreviewed)
        try:
            resp = client.get(f"/stitching-review?dataset={self.DATASET}")
            assert resp.status_code == 200
            # Serves the first unreviewed group, not a completed one.
            assert 'data-group-id="g1"' in resp.text
            assert 'data-group-id="g0"' not in resp.text
            # Counter is queue-relative: first of the two remaining.
            assert "Group 1 of 2" in resp.text
        finally:
            self._stop(patches)

    def test_all_reviewed_full_page(self):
        client, _, patches = self._client(unreviewed=[])
        try:
            resp = client.get(f"/stitching-review?dataset={self.DATASET}")
            assert resp.status_code == 200
            assert "All groups reviewed!" in resp.text
            # No stale "Group 1 of N" progress line for a fully-reviewed batch.
            assert "Group 1 of" not in resp.text
        finally:
            self._stop(patches)

    def test_fragment_index_walks_unreviewed_not_batch(self):
        # g0 reviewed -> unreviewed queue is [g1, g2, g3]. Index 1 into the queue
        # is g2, NOT all_groups[1] (== g1) as the pre-fix code returned.
        unreviewed = [self._group("g1"), self._group("g2"), self._group("g3")]
        client, _, patches = self._client(unreviewed)
        try:
            resp = client.get(f"/stitching-review/group?dataset={self.DATASET}&group_index=1")
            assert resp.status_code == 200
            assert 'data-group-id="g2"' in resp.text
            assert 'data-group-id="g1"' not in resp.text
            assert "Group 2 of 3" in resp.text
        finally:
            self._stop(patches)

    def test_fragment_deep_link_by_id_uses_batch_position(self):
        # Deep link to an already-reviewed group (not in the unreviewed queue):
        # still resolves, and reports its BATCH position (g2 -> "3 of 4").
        unreviewed = [self._group("g1"), self._group("g3")]
        client, _, patches = self._client(unreviewed)
        try:
            resp = client.get(f"/stitching-review/group?dataset={self.DATASET}&group_id=g2")
            assert resp.status_code == 200
            assert 'data-group-id="g2"' in resp.text
            assert "Group 3 of 4" in resp.text
        finally:
            self._stop(patches)

    def test_skip_advances_within_unreviewed_queue(self):
        unreviewed = [self._group("g1"), self._group("g3")]
        client, _, patches = self._client(unreviewed)
        try:
            resp = client.post(
                "/stitching-review/skip",
                data={"dataset": self.DATASET, "group_id": "g1"},
            )
            assert resp.status_code == 200
            # Next unreviewed after g1 is g3 (a completed group is never served).
            assert 'data-group-id="g3"' in resp.text
            assert "Group 2 of 2" in resp.text
        finally:
            self._stop(patches)

    def test_skip_last_unreviewed_wraps_to_first(self):
        unreviewed = [self._group("g1"), self._group("g3")]
        client, _, patches = self._client(unreviewed)
        try:
            resp = client.post(
                "/stitching-review/skip",
                data={"dataset": self.DATASET, "group_id": "g3"},
            )
            assert resp.status_code == 200
            assert 'data-group-id="g1"' in resp.text
            assert "Group 1 of 2" in resp.text
        finally:
            self._stop(patches)

    def test_select_advances_to_first_remaining_never_repeats(self):
        # Simulate the queue AFTER g0 was recorded: it has dropped out, so the
        # save-advance serves g1 (never re-serves the just-labeled g0).
        after = [self._group("g1"), self._group("g2"), self._group("g3")]
        client, recorder, patches = self._client(after)
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "g0",
                    "group_index": 0,
                    "included_refs": "r0",
                    "included_targets": "t0",
                },
            )
            assert resp.status_code == 200
            assert recorder.called
            assert 'data-group-id="g1"' in resp.text
            assert 'data-group-id="g0"' not in resp.text
            assert "Group 1 of 3" in resp.text
        finally:
            self._stop(patches)

    def test_select_last_group_shows_all_reviewed(self):
        client, recorder, patches = self._client(unreviewed=[])
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "g3",
                    "group_index": 0,
                    "included_refs": "r3",
                    "included_targets": "t3",
                },
            )
            assert resp.status_code == 200
            assert recorder.called
            # Fragment renders the all-reviewed empty state (no group card).
            assert "All Done!" in resp.text
            assert "data-group-id=" not in resp.text
        finally:
            self._stop(patches)


class TestStitchingUiHooks:
    """The group card exposes the DOM hooks the client-side UX JS relies on:

    - per-pill class/name data attributes (client-side summary recompute)
    - stable summary element ids the JS targets
    - the collapse/expand toggle button
    """

    DATASET = "test_ds"

    def _batch(self):
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gui",
                    "match_type": "1:N",
                    "ref_ids": ["r1"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_edge("r1", "t1", 0.9), _edge("r1", "t2", 0.8)],
                    "ref_classes": {"r1": "residential"},
                    "ref_names": {"r1": "Main St"},
                    "target_classes": {"t1": "residential", "t2": "footway"},
                    "target_names": {"t1": "Main St", "t2": "Path"},
                }
            ],
        }

    def _client(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_group_fragment_exposes_ui_hooks(self):
        client, patches = self._client()
        try:
            resp = client.get(
                f"/stitching-review/group?dataset={self.DATASET}&group_id=gui&group_index=0"
            )
            assert resp.status_code == 200
            html = resp.text
            # Collapse toggle (issue 3)
            assert 'id="panel-collapse-btn"' in html
            assert "togglePanelCollapse()" in html
            # Summary element ids the JS recompute targets (issue 1)
            assert 'id="summary-class-value"' in html
            assert 'id="summary-name-value"' in html
            # Per-pill class/name data attributes drive the recompute (issue 1)
            assert 'data-cls="residential"' in html
            assert 'data-name="Main St"' in html
            assert 'data-cls="footway"' in html
            # Sliver-edge exclusion control (feature 3): default-unchecked
            # checkbox + hidden field defaulting to exclude ("true") + indicator.
            assert 'id="include-slivers-toggle"' in html
            assert "onSliverToggle()" in html
            assert 'id="exclude-slivers-field"' in html
            assert 'name="exclude_slivers"' in html
            assert 'value="true"' in html
            assert 'id="sliver-indicator"' in html
            # Per-edge sliver flag is exposed to the client for the live count
            # and coverage-gap overlay (features 2 & 3).
            assert "is_sliver" in html
        finally:
            self._stop(patches)


class TestContextDisplayCap:
    """Display-only cap on presented spatial-context segments.

    PR #262 expanded the context clip envelope to the group's full bounds, which
    ballooned the context layer for large groups (thousands of pills/geometries).
    ``_cap_context_ids`` bounds the PRESENTATION to the N nearest per side without
    touching group data, the cached JSON, or recorded labels.
    """

    @staticmethod
    def _pt_line(x0, y0, x1, y1):
        return {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]}

    def _big_group(self, n_ctx_ref=400, n_ctx_target=400):
        """Group at the origin with many context segments placed at increasing x
        distance so 'nearest' ordering is deterministic and checkable."""
        # Group's own geometry sits near x=0.
        ref_geoms = {"r1": self._pt_line(0.0, 0.0, 0.001, 0.0)}
        target_geoms = {"t1": self._pt_line(0.0, 0.0, 0.0, 0.001)}
        # Context refs/targets march away along +x: id ctxr{i} at x=i (far = large i)
        ctx_ref_ids = [f"ctxr{i}" for i in range(n_ctx_ref)]
        ctx_target_ids = [f"ctxt{i}" for i in range(n_ctx_target)]
        ctx_ref_geoms = {
            f"ctxr{i}": self._pt_line(0.01 * (i + 1), 0.0, 0.01 * (i + 1) + 0.001, 0.0)
            for i in range(n_ctx_ref)
        }
        ctx_target_geoms = {
            f"ctxt{i}": self._pt_line(0.01 * (i + 1), 0.0, 0.01 * (i + 1) + 0.001, 0.0)
            for i in range(n_ctx_target)
        }
        return {
            "group_id": "big",
            "match_type": "N:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1", 0.9)],
            "ref_geometries": ref_geoms,
            "target_geometries": target_geoms,
            "context_ref_ids": ctx_ref_ids,
            "context_target_ids": ctx_target_ids,
            "context_ref_geometries": ctx_ref_geoms,
            "context_target_geometries": ctx_target_geoms,
        }

    def test_large_group_capped_to_nearest(self):
        from crosswalk.web.routes.stitching import CONTEXT_DISPLAY_CAP, _cap_context_ids

        group = self._big_group(n_ctx_ref=400, n_ctx_target=300)
        capped = _cap_context_ids(group)
        # Totals report the true (uncapped) counts.
        assert capped["ref_total"] == 400
        assert capped["target_total"] == 300
        # Each side capped to CONTEXT_DISPLAY_CAP.
        assert len(capped["ref_ids"]) == CONTEXT_DISPLAY_CAP
        assert len(capped["target_ids"]) == CONTEXT_DISPLAY_CAP
        # Nearest kept: ctxr0..ctxr{cap-1} are closest to the group at x~0.
        kept = set(capped["ref_ids"])
        assert "ctxr0" in kept
        assert f"ctxr{CONTEXT_DISPLAY_CAP - 1}" in kept
        assert f"ctxr{CONTEXT_DISPLAY_CAP}" not in kept  # first dropped one
        assert "ctxr399" not in kept  # farthest dropped

    def test_small_group_unchanged(self):
        from crosswalk.web.routes.stitching import _cap_context_ids

        group = self._big_group(n_ctx_ref=8, n_ctx_target=7)
        capped = _cap_context_ids(group)
        # Under the cap: ids returned verbatim, in original order.
        assert capped["ref_ids"] == [f"ctxr{i}" for i in range(8)]
        assert capped["target_ids"] == [f"ctxt{i}" for i in range(7)]
        assert capped["ref_total"] == 8
        assert capped["target_total"] == 7

    def test_no_context_is_noop(self):
        from crosswalk.web.routes.stitching import _cap_context_ids

        capped = _cap_context_ids({"ref_ids": ["r1"], "target_ids": ["t1"]})
        assert capped["ref_ids"] == []
        assert capped["target_ids"] == []
        assert capped["ref_total"] == 0
        assert capped["target_total"] == 0

    def test_missing_anchor_geometry_keeps_prefix(self):
        """When group geometries don't parse, fall back to the first N by order."""
        from crosswalk.web.routes.stitching import CONTEXT_DISPLAY_CAP, _cap_context_ids

        group = self._big_group(n_ctx_ref=200, n_ctx_target=0)
        group["ref_geometries"] = {}  # no anchor
        group["target_geometries"] = {}
        capped = _cap_context_ids(group)
        assert capped["ref_ids"] == [f"ctxr{i}" for i in range(CONTEXT_DISPLAY_CAP)]

    def test_context_builder_surfaces_truncation(self):
        from crosswalk.web.routes.stitching import _build_group_context

        group = self._big_group(n_ctx_ref=400, n_ctx_target=50)
        # names/classes lookups default to "" — present so the builder runs.
        ctx = _build_group_context(group)
        assert ctx["context_capped"] is True
        assert ctx["context_ref_total"] == 400
        assert ctx["context_target_total"] == 50
        assert len(ctx["context_ref_details"]) == 150
        # target side is under the cap -> untouched
        assert len(ctx["context_target_details"]) == 50
        # Combined capped id list for the map JSON blob is bounded.
        assert len(ctx["context_ids"]) == 150 + 50

    def test_geojson_capped(self):
        from crosswalk.web.routes.stitching import CONTEXT_DISPLAY_CAP, _build_group_geojson

        group = self._big_group(n_ctx_ref=400, n_ctx_target=400)
        fc = _build_group_geojson(group)
        ctx_features = [
            f for f in fc["features"] if f["properties"].get("_id", "").startswith("ctx")
        ]
        # 1 group ref + 1 group target are full geoms too; count only context ids.
        assert len(ctx_features) == 2 * CONTEXT_DISPLAY_CAP


class TestContextMembershipHint:
    """The context-segment membership hint (fix/context-pill-selection-trap).

    Spatial-context pills can belong to a NEIGHBORING group (post corridor
    decomposition). ``_load_group_membership`` builds an id -> owning-group_id
    map from the groups sidecar, and ``_build_group_context`` annotates each
    context detail with the neighboring group so reviewers know where a
    corridor continuation will actually be reviewed.
    """

    DATASET = "membership_ds"

    def _write_sidecar(self, tmp_path, groups):
        """Write a groups sidecar where _load_group_membership will find it:
        <PROJECT_ROOT>/data/output/<dataset>_groups.json."""
        out = tmp_path / "data" / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.DATASET}_groups.json").write_text(json.dumps({"groups": groups}))
        return out

    def _patch_root(self, tmp_path):
        from unittest.mock import patch

        import crosswalk.web.routes.stitching as st

        # Clear the module cache so each test sees its own sidecar.
        st._MEMBERSHIP_CACHE.clear()
        return patch.object(st, "PROJECT_ROOT", tmp_path)

    def test_load_membership_maps_ids_to_owning_group(self, tmp_path):
        from crosswalk.web.routes.stitching import _load_group_membership

        self._write_sidecar(
            tmp_path,
            [
                {"group_id": "gA", "ref_ids": ["rA"], "target_ids": ["tA1", "tA2"]},
                {"group_id": "gB", "ref_ids": ["rB"], "target_ids": ["tB1"]},
            ],
        )
        with self._patch_root(tmp_path):
            mem = _load_group_membership(self.DATASET)
        assert mem["tA1"] == "gA"
        assert mem["tA2"] == "gA"
        assert mem["rB"] == "gB"
        assert mem["tB1"] == "gB"
        assert "missing" not in mem

    def test_missing_sidecar_returns_empty(self, tmp_path):
        from crosswalk.web.routes.stitching import _load_group_membership

        with self._patch_root(tmp_path):  # no sidecar written
            assert _load_group_membership(self.DATASET) == {}

    def test_empty_dataset_returns_empty(self):
        from crosswalk.web.routes.stitching import _load_group_membership

        assert _load_group_membership("") == {}

    def test_cache_invalidates_on_mtime(self, tmp_path):
        import os

        from crosswalk.web.routes.stitching import _load_group_membership

        out = self._write_sidecar(
            tmp_path, [{"group_id": "gA", "ref_ids": [], "target_ids": ["t1"]}]
        )
        sidecar = out / f"{self.DATASET}_groups.json"
        with self._patch_root(tmp_path):
            assert _load_group_membership(self.DATASET)["t1"] == "gA"
            # Rewrite with a newer mtime and different membership.
            sidecar.write_text(json.dumps({"groups": [{"group_id": "gZ", "target_ids": ["t1"]}]}))
            os.utime(sidecar, (os.stat(sidecar).st_atime, os.stat(sidecar).st_mtime + 10))
            assert _load_group_membership(self.DATASET)["t1"] == "gZ"

    def test_build_context_annotates_member_group(self, tmp_path):
        from crosswalk.web.routes.stitching import _build_group_context

        # Current group has one real edge and two context targets, one of which
        # belongs to a neighboring group; the other is unknown to the sidecar.
        group = {
            "group_id": "gcur",
            "match_type": "1:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1", 0.9)],
            "context_target_ids": ["ctxNeighbor", "ctxUnknown"],
            "context_target_names": {},
            "context_target_classes": {},
        }
        self._write_sidecar(
            tmp_path,
            [
                {"group_id": "gcur", "ref_ids": ["r1"], "target_ids": ["t1"]},
                {"group_id": "gNbr", "ref_ids": [], "target_ids": ["ctxNeighbor"]},
            ],
        )
        with self._patch_root(tmp_path):
            ctx = _build_group_context(group, dataset=self.DATASET)
        by_id = {d["id"]: d for d in ctx["context_target_details"]}
        assert by_id["ctxNeighbor"]["member_group"] == "gNbr"
        # Unknown to the sidecar -> no hint (None), never an error.
        assert by_id["ctxUnknown"]["member_group"] is None

    def test_context_id_owned_by_current_group_yields_no_hint(self, tmp_path):
        from crosswalk.web.routes.stitching import _build_group_context

        group = {
            "group_id": "gcur",
            "match_type": "1:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1", 0.9)],
            "context_target_ids": ["ctxSelf"],
            "context_target_names": {},
            "context_target_classes": {},
        }
        # Sidecar (pathologically) attributes the context id to the current group.
        self._write_sidecar(
            tmp_path, [{"group_id": "gcur", "ref_ids": ["r1"], "target_ids": ["t1", "ctxSelf"]}]
        )
        with self._patch_root(tmp_path):
            ctx = _build_group_context(group, dataset=self.DATASET)
        assert ctx["context_target_details"][0]["member_group"] is None

    def test_no_dataset_leaves_member_group_none(self):
        from crosswalk.web.routes.stitching import _build_group_context

        group = {
            "group_id": "gcur",
            "match_type": "1:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1", 0.9)],
            "context_ref_ids": ["ctxR"],
            "context_ref_names": {},
            "context_ref_classes": {},
        }
        ctx = _build_group_context(group)  # dataset omitted
        assert ctx["context_ref_details"][0]["member_group"] is None


class TestContextPillDistinction:
    """Context pills must be UNMISTAKABLE from group-member pills so a reviewer
    cannot confuse toggling map context with adding a group edge (silent-drop
    trap). Asserts the rendered markers the CSS/JS depend on.
    """

    DATASET = "ctxpill_ds"

    def _batch(self):
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gpill",
                    "match_type": "1:N",
                    "ref_ids": ["r1"],
                    "target_ids": ["t1"],
                    "edges": [_edge("r1", "t1", 0.9)],
                    "ref_classes": {"r1": "residential"},
                    "ref_names": {"r1": "Main St"},
                    "target_classes": {"t1": "residential"},
                    "target_names": {"t1": "Main St"},
                    "context_ref_ids": ["ctxR1"],
                    "context_target_ids": ["ctxT1"],
                    "context_ref_names": {"ctxR1": "Elm St"},
                    "context_ref_classes": {"ctxR1": "residential"},
                    "context_target_names": {"ctxT1": "Elm St"},
                    "context_target_classes": {"ctxT1": "residential"},
                }
            ],
        }

    def _client(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_context_pills_and_notice_markers_present(self):
        client, patches = self._client()
        try:
            resp = client.get(
                f"/stitching-review/group?dataset={self.DATASET}&group_id=gpill&group_index=0"
            )
            assert resp.status_code == 200
            html = resp.text
            # Context pills carry the distinct class + affix + data flag + tooltip.
            assert "segment-pill-context" in html
            assert 'data-context="true"' in html
            assert "pill-ctx-affix" in html
            assert "ctx</span>" in html
            assert "Nearby context" in html
            assert "won&#39;t be saved" in html or "won't be saved" in html
            # The non-blocking submit notice element exists.
            assert 'id="context-notice"' in html
            # Group pills carry data-seg-id (used by map click) and are NOT
            # tagged as context.
            assert 'data-seg-id="r1"' in html
            assert 'data-seg-id="t1"' in html
        finally:
            self._stop(patches)

    def test_context_pill_is_not_a_group_selection_pill(self):
        """A context pill must not carry the group ref/target selection classes,
        so the submit path (which reads .segment-pill-ref/-target.active) can
        never pick it up and silently drop it."""
        client, patches = self._client()
        try:
            resp = client.get(
                f"/stitching-review/group?dataset={self.DATASET}&group_id=gpill&group_index=0"
            )
            html = resp.text
            # Isolate the context ref pill markup and assert it lacks the plain
            # group selection class (it uses segment-pill-context-ref instead).
            assert "segment-pill-context-ref" in html
            assert "segment-pill-context-target" in html
            # The context pill line must not contain the bare group class token.
            for line in html.splitlines():
                if "segment-pill-context" in line:
                    assert 'segment-pill-ref"' not in line
                    assert 'segment-pill-target"' not in line
        finally:
            self._stop(patches)


class TestRejectedPairSubmit:
    """The manual-path inconsistency guard draws from selected edges UNION the
    rejected candidates, so a reviewer who pairs two segments whose only shared
    edge was rejected by the optimizer has that segment recorded in the SET
    membership (not silently dropped), while a true non-candidate pair is still
    rejected. Manual submits now record membership, not expanded pairs."""

    DATASET = "test_ds"

    def _batch(self):
        # r1,t1 is the optimizer-selected edge; r2,t2 is a REJECTED candidate
        # over the group's own segments. No (r2,t1) edge exists at all.
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "grej",
                    "match_type": "M:N",
                    "ref_ids": ["r1", "r2"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_edge("r1", "t1", 0.9)],
                    "rejected_edges": [_edge("r2", "t2", 0.42)],
                }
            ],
        }

    def _client_and_recorder(self):
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        recorder = MagicMock()
        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch("crosswalk.web.routes.stitching.get_unreviewed_stitch_groups", return_value=[]),
            patch("crosswalk.web.routes.stitching.record_stitching_label", recorder),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), recorder, patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def _post(self, client, refs, targets):
        return client.post(
            "/stitching-review/select",
            data={
                "dataset": self.DATASET,
                "group_id": "grej",
                "group_index": 0,
                "included_refs": refs,
                "included_targets": targets,
                "selected_edges": "",
            },
        )

    def test_rejected_only_pair_is_recorded_not_dropped(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            # Reviewer pairs r2+t2 whose only shared edge is REJECTED. The union
            # guard passes, so the membership is recorded (not dropped).
            resp = self._post(client, "r2", "t2")
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r2"}
            assert set(kwargs["target_ids"]) == {"t2"}
            assert kwargs["num_refs"] == 1
            assert kwargs["num_targets"] == 1
        finally:
            self._stop(patches)

    def test_union_guard_still_rejects_a_true_non_candidate_pair(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            # (r2,t1) is neither a selected nor a rejected candidate -> inconsistent.
            resp = self._post(client, "r2", "t1")
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_union_records_full_membership(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            # Activate all pills: membership spans both segments of the selected
            # (r1,t1) and rejected (r2,t2) candidate edges.
            resp = self._post(client, "r1,r2", "t1,t2")
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)

    def test_explicit_confirmed_rejected_pair_stored_verbatim(self):
        # The submit-time pair-confirmation panel resubmits the ticked pairs as
        # an explicit selected_edges list. A ticked pair whose only edge was
        # REJECTED by the optimizer must survive the explicit path (candidate
        # union validation), stored verbatim without cross-product inflation.
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "grej",
                    "group_index": 0,
                    "included_refs": "",
                    "included_targets": "",
                    "selected_edges": json.dumps([{"ref_id": "r2", "target_id": "t2"}]),
                },
            )
            assert resp.status_code == 200
            stored = {
                (e["ref_id"], e["target_id"]) for e in recorder.call_args.kwargs["selected_edges"]
            }
            assert stored == {("r2", "t2")}
        finally:
            self._stop(patches)


class TestStaleProposalUI:
    """A queue entry flagged ``stale_grouping`` (its group no longer exists in
    the current sidecar, so its proposal could not be refreshed) renders a
    visible stale-proposal notice and marks the card so the submit JS forces the
    pair-confirmation panel. A fresh entry does not."""

    DATASET = "test_ds"

    def _batch(self, stale):
        group = {
            "group_id": "gstale",
            "match_type": "1:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1", 0.9)],
        }
        if stale:
            group["stale_grouping"] = True
        return {"dataset_id": self.DATASET, "groups": [group]}

    def _client(self, stale):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=self._batch(stale),
            ),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _fragment(self, client):
        return client.get(
            f"/stitching-review/group?dataset={self.DATASET}&group_id=gstale&group_index=0"
        ).text

    def test_stale_group_shows_notice_and_marks_card(self):
        client, patches = self._client(stale=True)
        try:
            html = self._fragment(client)
            assert 'data-stale="1"' in html
            assert "stale-proposal-notice" in html
            assert "Stale proposal" in html
        finally:
            for p in patches:
                p.stop()

    def test_fresh_group_not_marked_stale(self):
        client, patches = self._client(stale=False)
        try:
            html = self._fragment(client)
            assert 'data-stale="0"' in html
            assert "stale-proposal-notice" not in html
        finally:
            for p in patches:
                p.stop()

    def test_pair_confirm_panel_present_in_fragment(self):
        client, patches = self._client(stale=False)
        try:
            html = self._fragment(client)
            # The inline confirmation panel + its confirm/cancel hooks always
            # render (shown by JS for manual/de-anchored and stale-option submits).
            assert 'id="pair-confirm-panel"' in html
            assert "confirmPairSubmit()" in html
            assert "cancelPairConfirm()" in html
        finally:
            for p in patches:
                p.stop()


class TestPanelRouteReasonChip:
    """A queued group carrying panel_route_reason renders a single chip telling
    the reviewer WHY the agent panel routed it to human review; groups without
    the field (never voted / legacy queue) render no chip."""

    DATASET = "test_ds"

    def _batch(self, with_reason):
        group = {
            "group_id": "greason",
            "match_type": "1:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1", 0.9)],
        }
        if with_reason:
            group["panel_route_reason"] = "dissent:codex=B"
            group["panel_route_reason_human"] = "codex dissented — voted B"
        return {"dataset_id": self.DATASET, "groups": [group]}

    def _fragment(self, with_reason):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=self._batch(with_reason),
            ),
        ]
        for p in patches:
            p.start()
        try:
            client = TestClient(create_app())
            return client.get(
                f"/stitching-review/group?dataset={self.DATASET}&group_id=greason&group_index=0"
            ).text
        finally:
            for p in patches:
                p.stop()

    def test_chip_renders_human_reason_and_code(self):
        html = self._fragment(with_reason=True)
        assert "panel-route-chip" in html
        assert "codex dissented — voted B" in html
        # Machine-readable code carried in the title for hover/debug.
        assert "dissent:codex=B" in html

    def test_no_chip_without_reason(self):
        html = self._fragment(with_reason=False)
        assert "panel-route-chip" not in html


class TestDeAnchoredMode:
    """De-anchored review mode: blank-slate rendering (no active pills, proposals
    collapsed, no leaked selection styling / confidence), provenance stamping,
    and mode persistence across the HTMX swap chain."""

    DATASET = "test_ds"

    def _batch(self):
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gda1",
                    "match_type": "M:N",
                    "ref_ids": ["r1", "r2"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_edge("r1", "t1", 0.9)],
                    "rejected_edges": [_edge("r2", "t2", 0.42)],
                    "optimizer_assignment": [_edge("r1", "t1", 0.9)],
                },
                {
                    "group_id": "gda2",
                    "match_type": "1:N",
                    "ref_ids": ["r3"],
                    "target_ids": ["t3"],
                    "edges": [_edge("r3", "t3", 0.8)],
                    "optimizer_assignment": [_edge("r3", "t3", 0.8)],
                },
            ],
        }

    def _client(self, recorder=None, unreviewed=None):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch(
                "crosswalk.web.routes.stitching.get_unreviewed_stitch_groups",
                return_value=unreviewed if unreviewed is not None else [],
            ),
        ]
        if recorder is not None:
            patches.append(patch("crosswalk.web.routes.stitching.record_stitching_label", recorder))
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def _fragment(self, client, deanchored):
        q = f"/stitching-review/group?dataset={self.DATASET}&group_id=gda1&group_index=0"
        if deanchored:
            q += "&deanchored=1"
        return client.get(q).text

    def test_blank_slate_no_active_pills(self):
        client, patches = self._client()
        try:
            html = self._fragment(client, deanchored=True)
            # No group selection pill starts active (blank slate).
            assert "segment-pill-ref active" not in html
            assert "segment-pill-target active" not in html
        finally:
            self._stop(patches)

    def test_normal_mode_keeps_optimizer_preseed(self):
        client, patches = self._client()
        try:
            html = self._fragment(client, deanchored=False)
            # Zero behavior change for normal mode: the optimizer pill is active
            # and its option is pre-selected.
            assert "segment-pill-ref active" in html
            assert "is-optimizer selected" in html
            assert "deanchored-proposals" not in html
        finally:
            self._stop(patches)

    def test_proposals_collapsed_and_no_option_preselected(self):
        client, patches = self._client()
        try:
            html = self._fragment(client, deanchored=True)
            # Proposals are collapsed behind an explicit reveal; nothing pre-picked.
            assert "deanchored-proposals" in html
            assert "Reveal optimizer proposals" in html
            assert "is-optimizer selected" not in html
        finally:
            self._stop(patches)

    def test_confidence_readout_blanked(self):
        client, patches = self._client()
        try:
            da = self._fragment(client, deanchored=True)
            normal = self._fragment(client, deanchored=False)
            # The confidence readout must not leak the proposal's score.
            assert "&mdash;" in da
            assert "&mdash;" not in normal
        finally:
            self._stop(patches)

    def test_candidate_union_in_client_edges(self):
        client, patches = self._client()
        try:
            html = self._fragment(client, deanchored=True)
            # The rejected candidate is present in the client edge payload so the
            # live confidence can reason about a reviewer-built rejected pair.
            assert '"ref_id": "r2"' in html and '"target_id": "t2"' in html
        finally:
            self._stop(patches)

    def test_mode_toggle_present_in_both_modes(self):
        client, patches = self._client()
        try:
            normal = self._fragment(client, deanchored=False)
            da = self._fragment(client, deanchored=True)
            assert "deanchor-toggle" in normal and "&deanchored=1" in normal
            # The toggle carries the `active` class + aria-checked in de-anchored
            # mode. It renders as `scratch-row deanchor-toggle active` (the
            # tabbed IA styles it as a full-width switch on the Review tab), so
            # match the class token and state rather than an exact attribute.
            assert "deanchor-toggle active" in da and 'aria-checked="true"' in da
        finally:
            self._stop(patches)

    def test_forms_carry_mode_flag(self):
        client, patches = self._client()
        try:
            html = self._fragment(client, deanchored=True)
            assert '<input type="hidden" name="deanchored" value="1">' in html
        finally:
            self._stop(patches)

    def test_provenance_stamped_deanchored(self):
        from unittest.mock import MagicMock

        recorder = MagicMock()
        client, patches = self._client(recorder=recorder)
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gda1",
                    "group_index": 0,
                    "included_refs": "r1",
                    "included_targets": "t1",
                    "selected_edges": "",
                    "deanchored": "1",
                },
            )
            assert resp.status_code == 200
            assert recorder.call_args.kwargs["session_id"] == "deanchored_v1"
        finally:
            self._stop(patches)

    def test_provenance_none_in_normal_mode(self):
        from unittest.mock import MagicMock

        recorder = MagicMock()
        client, patches = self._client(recorder=recorder)
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gda1",
                    "group_index": 0,
                    "included_refs": "r1",
                    "included_targets": "t1",
                    "selected_edges": "",
                    "deanchored": "0",
                },
            )
            assert resp.status_code == 200
            assert recorder.call_args.kwargs["session_id"] is None
        finally:
            self._stop(patches)

    def test_mode_persists_across_swap_chain(self):
        from unittest.mock import MagicMock

        recorder = MagicMock()
        # After submitting gda1, the next unreviewed group (gda2) is returned.
        client, patches = self._client(recorder=recorder, unreviewed=[self._batch()["groups"][1]])
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gda1",
                    "group_index": 0,
                    "included_refs": "r1",
                    "included_targets": "t1",
                    "selected_edges": "",
                    "deanchored": "1",
                },
            )
            assert resp.status_code == 200
            # The next group fragment stays de-anchored (blank slate + collapsed
            # proposals + mode flag carried in its own forms).
            assert "deanchored-proposals" in resp.text
            assert '<input type="hidden" name="deanchored" value="1">' in resp.text
            assert "segment-pill-ref active" not in resp.text
        finally:
            self._stop(patches)

    def test_skip_preserves_mode(self):
        client, patches = self._client(unreviewed=self._batch()["groups"])
        try:
            resp = client.post(
                "/stitching-review/skip",
                data={"dataset": self.DATASET, "group_id": "gda1", "deanchored": "1"},
            )
            assert resp.status_code == 200
            assert "deanchored-proposals" in resp.text
        finally:
            self._stop(patches)

    def test_empty_deanchored_submit_requires_confirmation(self):
        """A blank-slate misclick on Select must NOT record a reject-all label:
        in de-anchored mode 'no active pills' is the untouched default, not a
        deliberate deselection, so an unconfirmed empty submit is refused."""
        from unittest.mock import MagicMock

        recorder = MagicMock()
        client, patches = self._client(recorder=recorder)
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gda1",
                    "group_index": 0,
                    "included_refs": "",
                    "included_targets": "",
                    "selected_edges": "",
                    "deanchored": "1",
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_confirmed_empty_deanchored_submit_stores_reject_all(self):
        """With the explicit confirm flag (set by the client's confirm dialog) a
        deliberate de-anchored reject-all is stored as [] with provenance."""
        from unittest.mock import MagicMock

        recorder = MagicMock()
        client, patches = self._client(recorder=recorder)
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gda1",
                    "group_index": 0,
                    "included_refs": "",
                    "included_targets": "",
                    "selected_edges": "",
                    "deanchored": "1",
                    "confirm_reject_all": "1",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["session_id"] == "deanchored_v1"
        finally:
            self._stop(patches)

    def test_normal_mode_empty_submit_needs_no_confirmation(self):
        """Normal mode keeps its original semantics: both pill fields empty is a
        deliberate deselection of the pre-seed and stores [] without any flag."""
        from unittest.mock import MagicMock

        recorder = MagicMock()
        client, patches = self._client(recorder=recorder)
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gda1",
                    "group_index": 0,
                    "included_refs": "",
                    "included_targets": "",
                    "selected_edges": "",
                    "deanchored": "0",
                },
            )
            assert resp.status_code == 200
            assert recorder.call_args.kwargs["selected_edges"] == []
        finally:
            self._stop(patches)

    def test_confirm_field_present_in_select_form(self):
        client, patches = self._client()
        try:
            html = self._fragment(client, deanchored=True)
            assert 'id="confirm-reject-all"' in html
        finally:
            self._stop(patches)


class TestRejectedSliverExclusion:
    """Under set semantics, sliver exclusion is a storage no-op: a manual submit
    records the full active-pill MEMBERSHIP whether exclude_slivers is on or off
    (there are no per-pair edges to drop). The flag still drives the client's
    confidence display only."""

    DATASET = "test_ds"

    def _batch(self):
        # (r1,t1) selected substantive edge; (r2,t2) REJECTED sliver candidate
        # (2-5% spans on ~50 m segments -> hybrid rule flags it).
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "grsl",
                    "match_type": "M:N",
                    "ref_ids": ["r1", "r2"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_frac_edge("r1", "t1", 0.0, 1.0, 0.0, 1.0, conf=0.9)],
                    "rejected_edges": [_frac_edge("r2", "t2", 0.0, 0.02, 0.0, 0.05, conf=0.3)],
                    "ref_geometries": {
                        "r1": _line_of_length_m(50.0),
                        "r2": _line_of_length_m(50.0),
                    },
                    "target_geometries": {
                        "t1": _line_of_length_m(50.0),
                        "t2": _line_of_length_m(50.0),
                    },
                }
            ],
        }

    def _client_and_recorder(self):
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        recorder = MagicMock()
        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch("crosswalk.web.routes.stitching.get_unreviewed_stitch_groups", return_value=[]),
            patch("crosswalk.web.routes.stitching.record_stitching_label", recorder),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), recorder, patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def _post(self, client, exclude):
        return client.post(
            "/stitching-review/select",
            data={
                "dataset": self.DATASET,
                "group_id": "grsl",
                "group_index": 0,
                "included_refs": "r1,r2",
                "included_targets": "t1,t2",
                "selected_edges": "",
                "exclude_slivers": exclude,
            },
        )

    def test_membership_unaffected_when_excluding(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(client, "true")
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["label_semantics"] == "set"
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)

    def test_membership_unaffected_when_including(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = self._post(client, "false")
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert set(kwargs["ref_ids"]) == {"r1", "r2"}
            assert set(kwargs["target_ids"]) == {"t1", "t2"}
        finally:
            self._stop(patches)
