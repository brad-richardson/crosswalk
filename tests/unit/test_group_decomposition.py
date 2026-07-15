"""Tests for panel-sized decomposition of over-backstop stitch groups.

Synthetic bipartite candidate graphs exercise the splitting algorithm (chain of
blocks joined by bridges, star around an articulation point, irreducible
biconnected blob, under-backstop no-op), the deterministic content-hash id
scheme, edge-ownership (exact partition), sub-group derivation, and the
conservative recomposition rule (any failed/unvoted sub-problem blocks the
whole-group label).
"""

from __future__ import annotations

import random

from crosswalk.matching.group_decomposition import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_UNVOTED,
    SUBPROBLEM_SEPARATOR,
    build_subproblem_group,
    decompose_candidate_edges,
    decompose_group,
    recompose_subproblem_verdicts,
    subproblem_id,
)


def _k_bipartite(refs: list[str], tgts: list[str]) -> list[tuple[str, str]]:
    """Complete bipartite edge set (a single biconnected block for >=2 x >=2)."""
    return [(r, t) for r in refs for t in tgts]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_under_backstop_is_noop():
    edges = _k_bipartite(["r1", "r2"], ["t1", "t2"])  # 4 edges
    d = decompose_candidate_edges("g1", edges, max_edges=10)
    assert d.n_edges == 4
    assert d.is_decomposed is False
    assert d.subproblems == ()


def test_chain_of_blocks_joined_by_bridge_splits():
    # Two K2,2 blobs (4 edges each, biconnected) joined by one bridge edge.
    blob1 = _k_bipartite(["a1", "a2"], ["b1", "b2"])
    blob2 = _k_bipartite(["c1", "c2"], ["d1", "d2"])
    bridge = [("a2", "d1")]  # connects the blobs; removal disconnects them
    edges = blob1 + bridge + blob2  # 9 edges

    d = decompose_candidate_edges("g1", edges, max_edges=5)
    assert d.is_decomposed is True
    sizes = sorted(s.n_edges for s in d.subproblems)
    assert sizes == [4, 5]  # one blob absorbs the bridge, the other cannot
    assert all(not s.oversized for s in d.subproblems)
    # Exact edge partition: no edge voted twice, none lost.
    all_edges = [e for s in d.subproblems for e in s.edges]
    assert sorted(all_edges) == sorted(set(edges))
    assert len(all_edges) == len(set(all_edges))


def test_star_articulation_point_shared_across_subproblems():
    # A hub target with 6 pendant refs: 6 bridge edges, all singleton blocks
    # sharing the hub articulation vertex.
    edges = [(f"r{i}", "hub") for i in range(1, 7)]
    d = decompose_candidate_edges("g1", edges, max_edges=3)
    assert d.is_decomposed is True
    assert sorted(s.n_edges for s in d.subproblems) == [3, 3]
    # The articulation node appears in BOTH sub-problems (as an endpoint), but
    # every EDGE belongs to exactly one — the union is well-defined.
    assert all("hub" in s.target_ids for s in d.subproblems)
    all_edges = [e for s in d.subproblems for e in s.edges]
    assert len(all_edges) == len(set(all_edges)) == 6


def test_irreducible_biconnected_blob_is_not_decomposed():
    # K3,3 is one biconnected block of 9 edges: splitting achieves nothing, so
    # the group is reported as NOT decomposed and stays in the existing flow.
    edges = _k_bipartite(["r1", "r2", "r3"], ["t1", "t2", "t3"])
    d = decompose_candidate_edges("g1", edges, max_edges=5)
    assert d.n_edges == 9
    assert d.is_decomposed is False
    assert d.subproblems == ()


def test_oversized_block_flagged_but_pendants_split_off():
    # K3,3 blob (9 edges, irreducible at max 5) plus two pendant bridges off one
    # of its refs: the blob stays as an oversized human-routed sub-problem, the
    # pendants become a votable one.
    blob = _k_bipartite(["r1", "r2", "r3"], ["t1", "t2", "t3"])
    pendants = [("r1", "p1"), ("r1", "p2")]
    d = decompose_candidate_edges("g1", blob + pendants, max_edges=5)
    assert d.is_decomposed is True
    assert len(d.subproblems) == 2
    assert len(d.oversized_subproblems) == 1
    assert d.oversized_subproblems[0].n_edges == 9
    assert len(d.votable_subproblems) == 1
    assert d.votable_subproblems[0].n_edges == 2


def test_disconnected_components_never_merge():
    # Two disconnected 2-edge chains: no shared vertex, so they must stay
    # separate sub-problems even though 2 + 2 <= max_edges.
    edges = [("a1", "b1"), ("a2", "b1"), ("c1", "d1"), ("c2", "d1")]
    d = decompose_candidate_edges("g1", edges, max_edges=3)
    assert d.is_decomposed is True
    assert sorted(s.n_edges for s in d.subproblems) == [2, 2]
    for s in d.subproblems:
        refs = {r for r, _ in s.edges}
        assert refs in ({"a1", "a2"}, {"c1", "c2"})


def test_accepts_edge_dicts_and_dedupes():
    dict_edges = [
        {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
        {"ref_id": "r1", "target_id": "t1", "confidence": 0.8},  # dup pair
        {"ref_id": "r2", "target_id": "t1"},
    ]
    d1 = decompose_candidate_edges("g1", dict_edges, max_edges=10)
    d2 = decompose_candidate_edges("g1", [("r1", "t1"), ("r2", "t1")], max_edges=10)
    assert d1.n_edges == d2.n_edges == 2


def test_decompose_group_reads_group_dict():
    group = {
        "group_id": "gX",
        "edges": [{"ref_id": f"r{i}", "target_id": "hub"} for i in range(6)],
    }
    d = decompose_group(group, max_edges=3)
    assert d.parent_group_id == "gX"
    assert d.is_decomposed is True


# ---------------------------------------------------------------------------
# Determinism + id stability
# ---------------------------------------------------------------------------


def _big_synthetic_edges() -> list[tuple[str, str]]:
    """A multi-blob chain with bridges and pendants (deterministic content)."""
    edges: list[tuple[str, str]] = []
    for b in range(6):
        edges += _k_bipartite([f"r{b}a", f"r{b}b"], [f"t{b}a", f"t{b}b"])
        if b:
            edges.append((f"r{b - 1}b", f"t{b}a"))  # bridge to previous blob
    edges += [("r0a", f"pend{i}") for i in range(4)]  # pendants
    return edges


def test_decomposition_deterministic_under_input_shuffle():
    edges = _big_synthetic_edges()
    d1 = decompose_candidate_edges("g1", edges, max_edges=7)
    for seed in (1, 2, 3):
        shuffled = list(edges)
        random.Random(seed).shuffle(shuffled)
        d2 = decompose_candidate_edges("g1", shuffled, max_edges=7)
        assert d2 == d1  # frozen dataclasses: full structural equality


def test_subproblem_ids_are_content_hashes():
    edges = [("r1", "t1"), ("r2", "t1")]
    sid = subproblem_id("g1", edges)
    assert sid.startswith("g1" + SUBPROBLEM_SEPARATOR)
    # Order-independent, dedup-independent, content-sensitive.
    assert subproblem_id("g1", list(reversed(edges))) == sid
    assert subproblem_id("g1", edges + [("r1", "t1")]) == sid
    assert subproblem_id("g1", [("r1", "t1"), ("r3", "t1")]) != sid
    assert subproblem_id("g2", edges) != sid  # parent-scoped


def test_ids_stable_across_runs():
    # Regression pin: the hash must never drift (labels must be reproducible).
    assert subproblem_id("g1", [("r1", "t1"), ("r2", "t1")]) == "g1__pc308981522"


# ---------------------------------------------------------------------------
# Sub-group derivation
# ---------------------------------------------------------------------------


def test_build_subproblem_group_filters_parent():
    edges = [(f"r{i}", "hub") for i in range(1, 7)]
    parent = {
        "group_id": "g1",
        "match_type": "M:N",
        "ref_ids": [f"r{i}" for i in range(1, 7)],
        "target_ids": ["hub"],
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.9, "is_bridge": True, "selected": True}
            for r, t in edges
        ],
        "optimizer_assignment": [
            {"ref_id": "r1", "target_id": "hub", "confidence": 0.9},
            {"ref_id": "r4", "target_id": "hub", "confidence": 0.9},
        ],
        "ref_geometries": {f"r{i}": {"type": "LineString"} for i in range(1, 7)},
        "target_geometries": {"hub": {"type": "LineString"}},
        "ref_names": {f"r{i}": f"Ref {i}" for i in range(1, 7)},
        "target_names": {"hub": "Hub St"},
        "ref_classes": {f"r{i}": "residential" for i in range(1, 7)},
        "target_classes": {"hub": "residential"},
        "ref_physical": {
            f"r{i}": {"level_lr": [{"between": [0.0, 1.0], "value": i % 2}]}
            for i in range(1, 7)
        },
        "target_physical": {
            "hub": {"road_flags_lr": [{"between": [0.0, 1.0], "value": ["is_bridge"]}]}
        },
        "alternatives": ["should-not-carry-over"],
    }
    d = decompose_group(parent, max_edges=3)
    subs = [build_subproblem_group(parent, s, len(d.subproblems)) for s in d.subproblems]
    assert len(subs) == 2
    for sub, sp in zip(subs, d.subproblems):
        assert sub["group_id"] == sp.subproblem_id
        assert sub["parent_group_id"] == "g1"
        assert sub["n_subproblems"] == 2
        assert sub["match_type"] == "N:1"  # several refs, one target
        assert sub["n_edges"] == 3
        assert "alternatives" not in sub
        # Edge dicts carry over verbatim (confidence + structural fields).
        assert all(e["is_bridge"] for e in sub["edges"])
        # Geometries/names/classes filtered to the sub-problem's endpoints.
        assert set(sub["ref_geometries"]) == set(sub["ref_ids"])
        assert set(sub["target_geometries"]) == {"hub"}
        assert set(sub["ref_classes"]) == set(sub["ref_ids"])
        assert set(sub["ref_physical"]) == set(sub["ref_ids"])
        assert set(sub["target_physical"]) == {"hub"}
        # Optimizer assignment restricted to the sub-problem's edges.
        for e in sub["optimizer_assignment"]:
            assert (e["ref_id"], e["target_id"]) in set(sp.edges)
    # Every parent assignment edge lands in exactly one sub-group.
    assigned = [(e["ref_id"], e["target_id"]) for s in subs for e in s["optimizer_assignment"]]
    assert sorted(assigned) == [("r1", "hub"), ("r4", "hub")]


def test_build_subproblem_group_marks_oversized():
    blob = _k_bipartite(["r1", "r2", "r3"], ["t1", "t2", "t3"])
    parent = {
        "group_id": "g1",
        "edges": [{"ref_id": r, "target_id": t} for r, t in blob + [("r1", "p1"), ("r1", "p2")]],
    }
    d = decompose_group(parent, max_edges=5)
    over = d.oversized_subproblems[0]
    sub = build_subproblem_group(parent, over, len(d.subproblems))
    assert sub["subproblem_oversized"] is True
    votable = d.votable_subproblems[0]
    assert "subproblem_oversized" not in build_subproblem_group(parent, votable, 2)


# ---------------------------------------------------------------------------
# Recomposition
# ---------------------------------------------------------------------------


def test_recompose_all_accepts_completes_with_union():
    rec = recompose_subproblem_verdicts(
        "g1",
        ["s1", "s2"],
        {
            "s1": ("auto_accept", [("r1", "t1"), ("r2", "t1")]),
            "s2": ("auto_accept", [("r3", "t2")]),
        },
    )
    assert rec.status == STATUS_COMPLETE
    assert rec.union_edges == [("r1", "t1"), ("r2", "t1"), ("r3", "t2")]
    assert rec.n_subproblems == 2
    assert rec.n_resolved == 2
    assert rec.failed_subproblems == []
    assert rec.unvoted_subproblems == []


def test_recompose_one_failed_subproblem_blocks():
    rec = recompose_subproblem_verdicts(
        "g1",
        ["s1", "s2"],
        {
            "s1": ("auto_accept", [("r1", "t1")]),
            "s2": ("human_review", []),  # dissent / NONE / class demotion / ...
        },
    )
    assert rec.status == STATUS_FAILED
    assert rec.failed_subproblems == ["s2"]
    assert rec.union_edges == [("r1", "t1")]  # partial, informational only


def test_recompose_unvoted_subproblem_blocks():
    rec = recompose_subproblem_verdicts(
        "g1",
        ["s1", "s2", "s3"],
        {"s1": ("auto_accept", [("r1", "t1")])},
    )
    assert rec.status == STATUS_UNVOTED
    assert rec.unvoted_subproblems == ["s2", "s3"]
    assert rec.n_resolved == 1


def test_recompose_failure_trumps_unvoted():
    rec = recompose_subproblem_verdicts(
        "g1",
        ["s1", "s2", "s3"],
        {"s1": ("auto_accept", [("r1", "t1")]), "s2": ("human_review", [])},
    )
    assert rec.status == STATUS_FAILED
    assert rec.failed_subproblems == ["s2"]
    assert rec.unvoted_subproblems == ["s3"]


def test_recompose_ignores_verdicts_outside_roster():
    rec = recompose_subproblem_verdicts(
        "g1",
        ["s1"],
        {
            "s1": ("auto_accept", [("r1", "t1")]),
            "sX": ("auto_accept", [("r9", "t9")]),  # not in roster: ignored
        },
    )
    assert rec.status == STATUS_COMPLETE
    assert rec.union_edges == [("r1", "t1")]
