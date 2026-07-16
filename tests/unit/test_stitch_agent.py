"""Unit tests for the agent stitching-label pipeline (evidence, runner, eval).

No live CLI calls: subprocess invocation is mocked for runner tests. Synthetic
group fixtures exercise option letter<->edge-set mapping, vote parsing,
consensus rules, evidence metadata, and eval matching.
"""

from __future__ import annotations

import copy
import errno
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from crosswalk.agent_labeling import stitch_runner as sr
from crosswalk.agent_labeling.stitch_eval import (
    edge_prf,
    evaluate_batch,
    recover_empty_reject_all,
    recover_labeled_groups,
    summarize,
)
from crosswalk.agent_labeling.stitch_evidence import (
    build_metadata,
    build_prompt,
    generate_group_evidence,
    prune_options_for_panel,
    render_option,
)
from crosswalk.agent_labeling.stitch_provenance import (
    EvidenceProvenanceError,
    load_evidence_manifest,
)
from crosswalk.matching.stitch_options import build_stitch_options

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

R1, R2 = "ref-1", "ref-2"
T1, T2 = "tgt_1_88a", "tgt_2_88b"


def _line(coords):
    return {"type": "LineString", "coordinates": coords}


def make_group() -> dict:
    """A 2x2 M:N group with an optimizer assignment and two alternatives."""
    edges = [
        {
            "ref_id": R1,
            "target_id": T1,
            "confidence": 0.9,
            "gers_start_frac": 0.0,
            "gers_end_frac": 1.0,
            "local_start_frac": 0.0,
            "local_end_frac": 1.0,
        },
        {"ref_id": R1, "target_id": T2, "confidence": 0.4},
        {"ref_id": R2, "target_id": T2, "confidence": 0.8},
    ]
    return {
        "group_id": "grp001",
        "match_type": "M:N",
        "ref_ids": [R1, R2],
        "target_ids": [T1, T2],
        "edges": edges,
        "optimizer_assignment": [
            {"ref_id": R1, "target_id": T1},
            {"ref_id": R2, "target_id": T2},
        ],
        "alternatives": [
            {
                "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}],
                "total_confidence": 1.7,
            },
            {
                "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R1, "target_id": T2}],
                "total_confidence": 1.3,
            },
        ],
        "ref_geometries": {R1: _line([[0, 0], [1, 1]]), R2: _line([[1, 1], [2, 2]])},
        "target_geometries": {T1: _line([[0, 0.1], [1, 1.1]]), T2: _line([[1, 1.1], [2, 2.1]])},
        "ref_names": {R1: "Main St", R2: "Main St"},
        "target_names": {T1: "MAIN STREET", T2: "MAIN STREET"},
        "ref_classes": {R1: "residential", R2: "residential"},
        "target_classes": {T1: "local", T2: "local"},
    }


# ---------------------------------------------------------------------------
# Option image pair (edge) rendering
# ---------------------------------------------------------------------------


def _pair_group() -> dict:
    """A 2x2 group whose two options share members but differ only in pairing.

    R1/R2 and T1/T2 are near-vertical parallel segments (R1,T1 on the left;
    R2,T2 on the right). Option X ties R1-T1 + R2-T2 (short local ties); option Y
    ties R1-T2 + R2-T1 (long crossing ties). Both options' member sets are exactly
    {R1,R2}x{T1,T2}, so pre-fix member-only highlighting made them identical.
    """
    return {
        "group_id": "gx",
        "match_type": "M:N",
        "ref_ids": ["r1", "r2"],
        "target_ids": ["t1", "t2"],
        "ref_geometries": {
            "r1": _line([[0.0, 0.0], [0.0, 1.0]]),
            "r2": _line([[1.0, 0.0], [1.0, 1.0]]),
        },
        "target_geometries": {
            "t1": _line([[0.05, 0.0], [0.05, 1.0]]),
            "t2": _line([[1.05, 0.0], [1.05, 1.0]]),
        },
    }


def _pair_option(letter: str, pairs: list[tuple[str, str]], **edge_extra) -> dict:
    edges = [{"ref_id": r, "target_id": t, **edge_extra} for r, t in pairs]
    return {
        "letter": letter,
        "is_optimizer": False,
        "edges": edges,
        "active_refs": sorted({r for r, _ in pairs}),
        "active_targets": sorted({t for _, t in pairs}),
    }


def test_render_option_distinguishes_identical_members_different_edges():
    """Acceptance: same member set, different pairings -> different image bytes."""
    group = _pair_group()
    opt_x = _pair_option("X", [("r1", "t1"), ("r2", "t2")])
    opt_y = _pair_option("Y", [("r1", "t2"), ("r2", "t1")])
    # Sanity: the two options really do share an identical member set.
    assert opt_x["active_refs"] == opt_y["active_refs"]
    assert opt_x["active_targets"] == opt_y["active_targets"]

    img_x = render_option(group, opt_x)
    img_y = render_option(group, opt_y)
    assert img_x.size == img_y.size
    assert img_x.tobytes() != img_y.tobytes()


def test_render_option_deterministic():
    """Same group + option renders to byte-identical images across calls."""
    group = _pair_group()
    opt = _pair_option("X", [("r1", "t1"), ("r2", "t2")])
    assert render_option(group, opt).tobytes() == render_option(group, opt).tobytes()


def test_render_option_legacy_edges_without_fracs_do_not_crash():
    """Edges lacking alignment fracs (or absent entirely) fall back gracefully."""
    group = _pair_group()
    # Legacy edges: no gers_/local_ fracs -> geometric-midpoint fallback tie.
    legacy = _pair_option("X", [("r1", "t1"), ("r2", "t2")])
    img_legacy = render_option(group, legacy)
    assert img_legacy.size[0] > 0

    # No edges at all (pre-edges option dict): renders members only, no crash.
    no_edges = {
        "letter": "Z",
        "is_optimizer": False,
        "active_refs": ["r1"],
        "active_targets": ["t1"],
    }
    img_no_edges = render_option(group, no_edges)
    assert img_no_edges.size == img_legacy.size


def test_render_option_uses_aligned_span_midpoints():
    """Two edges sharing a ref but aligning to different spans tie at different
    points, so per-edge aligned-span midpoints (not one shared segment midpoint)
    drive the connectors."""
    group = _pair_group()
    # Same pairing, but edge fracs place the R1 tie endpoint at opposite ends.
    lower = _pair_option(
        "A",
        [("r1", "t1")],
        gers_start_frac=0.0,
        gers_end_frac=0.2,
        local_start_frac=0.0,
        local_end_frac=0.2,
    )
    upper = _pair_option(
        "B",
        [("r1", "t1")],
        gers_start_frac=0.8,
        gers_end_frac=1.0,
        local_start_frac=0.8,
        local_end_frac=1.0,
    )
    assert render_option(group, lower).tobytes() != render_option(group, upper).tobytes()


# ---------------------------------------------------------------------------
# Option letter <-> edge-set mapping
# ---------------------------------------------------------------------------


def test_build_options_letters_and_optimizer():
    ctx = build_stitch_options(make_group())
    letters = [o["letter"] for o in ctx["options"]]
    # A = optimizer, then deduped alternatives.
    assert letters[0] == "A"
    assert ctx["optimizer_letter"] == "A"
    assert ctx["options"][0]["is_optimizer"] is True
    # Optimizer edge set equals first alternative -> deduped, so B is the 2nd alt.
    edge_sets = {
        o["letter"]: {(e["ref_id"], e["target_id"]) for e in o["edges"]} for o in ctx["options"]
    }
    assert edge_sets["A"] == {(R1, T1), (R2, T2)}
    assert edge_sets["B"] == {(R1, T1), (R1, T2)}


def test_build_options_edges_enriched_with_confidence():
    ctx = build_stitch_options(make_group())
    opt_a = ctx["options"][0]
    e = next(x for x in opt_a["edges"] if x["target_id"] == T1)
    assert e["confidence"] == 0.9
    assert e["gers_start_frac"] == 0.0


def test_build_options_drops_non_group_edges():
    g = make_group()
    g["alternatives"].append({"edges": [{"ref_id": "phantom", "target_id": "ghost"}]})
    ctx = build_stitch_options(g)
    for o in ctx["options"]:
        for e in o["edges"]:
            assert (e["ref_id"], e["target_id"]) in {(R1, T1), (R1, T2), (R2, T2)}


def test_build_options_multiref_alternative_gets_stable_letter():
    """A multi-ref span alternative (T2 spans R1+R2) is a distinct, lettered option.

    Guards the UI / evidence-pack contract: letters stay sequential (A, B, C...)
    and a target-spans-two-refs alternative surfaces as its own option.
    """
    g = make_group()
    # T2 spans both refs R1 and R2 (both edges already exist in the group).
    g["alternatives"].append(
        {
            "edges": [
                {"ref_id": R1, "target_id": T1},
                {"ref_id": R1, "target_id": T2},
                {"ref_id": R2, "target_id": T2},
            ],
            "total_confidence": 2.1,
        }
    )
    ctx = build_stitch_options(g)
    letters = [o["letter"] for o in ctx["options"]]
    assert letters == list("ABC")[: len(letters)]  # sequential, no gaps
    edge_sets = {
        o["letter"]: {(e["ref_id"], e["target_id"]) for e in o["edges"]} for o in ctx["options"]
    }
    # The multi-ref span is present as exactly one option.
    multiref = {(R1, T1), (R1, T2), (R2, T2)}
    assert sum(1 for s in edge_sets.values() if s == multiref) == 1


def test_build_options_duplicated_raw_edges_do_not_inflate_confidence():
    """Duplicated / out-of-group raw edges must not inflate the displayed confidence.

    Regression for the option-generation defect: an alternative whose edge list
    contained many duplicated / cross-group edges (and an inflated stored
    ``total_confidence``) rendered as a nonsensical mean confidence (e.g.
    "3242%"). The displayed total/mean must be computed over the validated,
    deduplicated edge set actually shown.
    """
    g = make_group()
    g["alternatives"] = [
        {
            # Same in-group edge repeated 30x plus a phantom edge, with a wildly
            # inflated stored total_confidence.
            "edges": [{"ref_id": R1, "target_id": T1}] * 30
            + [{"ref_id": "phantom", "target_id": "ghost"}],
            "total_confidence": 227.0,
        }
    ]
    # Drop the optimizer so the alternative is option A.
    g["optimizer_assignment"] = []
    ctx = build_stitch_options(g)
    opt = ctx["options"][0]
    assert opt["edge_count"] == 1  # only the single distinct in-group edge shows
    assert opt["total_confidence"] == pytest.approx(0.9)  # R1->T1 confidence, not 227
    assert opt["mean_confidence"] == pytest.approx(0.9)


def test_choice_to_edge_set():
    obl = {"A": [(R1, T1), (R2, T2)], "B": [(R1, T1)]}
    assert sr.choice_to_edge_set("A", obl) == frozenset({(R1, T1), (R2, T2)})
    assert sr.choice_to_edge_set("NONE", obl) == frozenset()
    assert sr.choice_to_edge_set("Z", obl) == frozenset()


# ---------------------------------------------------------------------------
# Vote parsing / validation
# ---------------------------------------------------------------------------


def test_parse_vote_plain_json():
    raw = '{"choice": "A", "confidence": 0.9, "reasoning": "clear"}'
    choice, conf, reason = sr.parse_vote(raw, {"A", "B"})
    assert (choice, conf, reason) == ("A", 0.9, "clear")


def test_parse_vote_fenced_json():
    raw = 'Here you go:\n```json\n{"choice":"B","confidence":0.7,"reasoning":"x"}\n```\n'
    assert sr.parse_vote(raw, {"A", "B"})[0] == "B"


def test_parse_vote_embedded_in_prose():
    raw = 'I think {"choice": "NONE", "confidence": 0.2, "reasoning": "none fit"} is right.'
    assert sr.parse_vote(raw, {"A", "B"})[0] == "NONE"


def test_parse_vote_option_prefix_and_period():
    raw = '{"choice": "Option A.", "confidence": 1, "reasoning": "y"}'
    assert sr.parse_vote(raw, {"A"})[0] == "A"


def test_parse_vote_confidence_clamped():
    raw = '{"choice": "A", "confidence": 5, "reasoning": "y"}'
    assert sr.parse_vote(raw, {"A"})[1] == 1.0


def test_parse_vote_rejects_invalid_letter():
    with pytest.raises(ValueError):
        sr.parse_vote('{"choice": "Q", "confidence": 1, "reasoning": "y"}', {"A", "B"})


def test_parse_vote_rejects_garbage():
    with pytest.raises(ValueError):
        sr.parse_vote("total garbage no json here", {"A"})


def test_parse_vote_missing_choice():
    with pytest.raises(ValueError):
        sr.parse_vote('{"confidence": 1, "reasoning": "y"}', {"A"})


def test_parse_vote_reasoning_containing_braces():
    # A greedy {.*} regex would over-capture; non-greedy {.*?} would truncate
    # at the } inside the reasoning string. raw_decode handles both.
    raw = '{"choice": "A", "confidence": 0.8, "reasoning": "edge set {R1->T1} wins"}'
    choice, conf, reason = sr.parse_vote(raw, {"A"})
    assert choice == "A"
    assert reason == "edge set {R1->T1} wins"


def test_parse_vote_two_concatenated_objects_takes_first():
    raw = (
        '{"choice": "B", "confidence": 0.6, "reasoning": "first"}'
        '{"choice": "A", "confidence": 0.9, "reasoning": "second"}'
    )
    choice, _conf, reason = sr.parse_vote(raw, {"A", "B"})
    assert choice == "B"
    assert reason == "first"


def test_parse_vote_fenced_object_in_prose_with_braces():
    raw = (
        "Considering the options {A, B}:\n"
        "```json\n"
        '{"choice": "B", "confidence": 0.7, "reasoning": "map {x} shows overlap"}\n'
        "```\n"
        "Done."
    )
    choice, _conf, reason = sr.parse_vote(raw, {"A", "B"})
    assert choice == "B"
    assert reason == "map {x} shows overlap"


def test_extract_json_skips_unparseable_brace_runs():
    raw = '{not json} but later {"choice": "A", "confidence": 1, "reasoning": "ok"}'
    assert sr.parse_vote(raw, {"A"})[0] == "A"


# ---------------------------------------------------------------------------
# Consensus rules
# ---------------------------------------------------------------------------


def _vote(provider, choice, es=frozenset()):
    return sr.Vote(
        group_id="g",
        provider=provider,
        model="m",
        choice=choice,
        confidence=0.9,
        reasoning="",
        edge_set=es,
    )


def test_consensus_unanimous_auto_accept():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes)
    assert c.consensus == "unanimous"
    assert c.routing == "auto_accept"
    assert c.choice == "A"
    assert c.edge_set == es
    assert c.route_reason == "unanimous"


def test_consensus_unanimous_none_routes_to_human():
    votes = [_vote("claude", "NONE"), _vote("codex", "NONE"), _vote("agy", "NONE")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "unanimous"
    assert c.routing == "human_review"  # NONE never auto-accepts
    assert c.route_reason == "unanimous_none"


def test_consensus_majority():
    votes = [_vote("claude", "A"), _vote("codex", "A"), _vote("agy", "B")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.routing == "human_review"
    assert c.choice == "A"
    assert "agy=B" in c.minority
    assert c.route_reason == "dissent:agy=B"


def test_consensus_none_all_differ():
    votes = [_vote("claude", "A"), _vote("codex", "B"), _vote("agy", "NONE")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.routing == "human_review"
    assert c.route_reason == "no_majority"


def test_consensus_abstention_breaks_unanimity():
    # 2 agree + 1 abstain -> not unanimous (needs all 3 valid), so majority.
    votes = [_vote("claude", "A"), _vote("codex", "A"), _vote("agy", "ABSTAIN")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.n_valid == 2
    assert c.route_reason == "below_quorum:2"


def test_consensus_all_abstain():
    votes = [_vote("claude", "ABSTAIN"), _vote("codex", "ABSTAIN"), _vote("agy", "ABSTAIN")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.n_valid == 0
    assert c.route_reason == "all_abstained"


# ---------------------------------------------------------------------------
# Consensus: 4-voter (v5 quad) behavior under the QUORUM rule — auto-accept
# when all VALID votes agree and >=3 are valid. Full 4/4 unanimity and a
# 3-of-4 quorum accept (one abstention) are DISTINCT outcomes end-to-end
# (tier "unanimous" vs "quorum", reason "unanimous" vs "quorum"), and quorum
# forgives abstention ONLY — a dissenting valid vote still routes to a human.
# The pre-v5 "abstention blocks unanimity" behavior (REASON_ABSTENTION) is
# retired from live routing; historical rows keep deriving it (see
# test_stitching.py::TestDeriveRouteReason).
# ---------------------------------------------------------------------------


def test_consensus_quad_4of4_unanimous_auto_accept():
    """(a) 4/4 valid unanimity auto-accepts, exactly like 3/3 (n-agnostic)."""
    es = frozenset({(R1, T1)})
    votes = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("kimi", "A", es),
        _vote("muse", "A", es),
    ]
    c = sr.compute_consensus(votes)
    assert c.consensus == "unanimous"
    assert c.routing == "auto_accept"
    assert c.n_valid == 4
    assert c.route_reason == "unanimous"


def test_consensus_quad_3valid_1abstain_quorum_accepts():
    """(b) 3 valid votes agree + 1 ABSTAIN -> QUORUM auto-accept (the v5 rule).

    All valid votes agree and n_valid >= 3, so the group auto-accepts — the
    2026-07-10 quad calibration replay showed the old abstention-block sent
    clean 3-of-4 agreements to humans for no information gain. The verdict is
    the DISTINCT "quorum" tier / route reason (never "unanimous"): a 3-of-4
    accept over an abstention must stay distinguishable from a 4/4 accept all
    the way into the export labelers (panel_quorum_v5 vs panel_unanimous_v5).
    """
    es = frozenset({(R1, T1)})
    votes = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("kimi", "A", es),
        _vote("muse", "ABSTAIN"),
    ]
    c = sr.compute_consensus(votes)
    assert c.consensus == "quorum"
    assert c.routing == "auto_accept"
    assert c.choice == "A"
    assert c.edge_set == es
    assert c.n_valid == 3
    assert c.n_votes == 4
    assert c.route_reason == "quorum"


def test_consensus_quad_2_2_split_human_review():
    """A 2+2 split has no all-valid agreement -> human review (dissent stamped)."""
    es = frozenset({(R1, T1)})
    votes = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("kimi", "B"),
        _vote("muse", "B"),
    ]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.routing == "human_review"
    assert c.route_reason.startswith("dissent:")


def test_consensus_quad_4way_split_no_majority():
    """Four different choices -> no majority, human review."""
    votes = [
        _vote("claude", "A"),
        _vote("codex", "B"),
        _vote("kimi", "C"),
        _vote("muse", "NONE"),
    ]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.routing == "human_review"
    assert c.route_reason == "no_majority"


def test_consensus_quad_all_abstain():
    votes = [_vote(p, "ABSTAIN") for p in ("claude", "codex", "kimi", "muse")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.n_valid == 0
    assert c.routing == "human_review"
    assert c.route_reason == "all_abstained"


def test_consensus_quad_quorum_none_routes_to_human():
    """3 valid NONE + 1 abstain -> the QUORUM analog of unanimous_none.

    A NONE verdict never auto-accepts on either tier (same as unanimous-NONE);
    the distinct ``quorum_none`` reason keeps the abstention visible so the
    human-review queue can distinguish it without treating NONE as reject-all
    truth.
    """
    votes = [
        _vote("claude", "NONE"),
        _vote("codex", "NONE"),
        _vote("kimi", "NONE"),
        _vote("muse", "ABSTAIN"),
    ]
    c = sr.compute_consensus(votes)
    assert c.consensus == "quorum"
    assert c.choice == "NONE"
    assert c.routing == "human_review"
    assert c.n_valid == 3
    assert c.route_reason == "quorum_none"


def test_consensus_quad_quorum_accept_still_low_conf_gated():
    """The low-confidence gate applies to QUORUM accepts over the VALID votes:
    a 0.6-confidence valid vote in an otherwise quorum-clean group demotes."""
    es = frozenset({(R1, T1)})
    votes = [
        _vote_c("claude", "A", 0.9, es),
        _vote_c("codex", "A", 0.6, es),
        _vote_c("kimi", "A", 0.9, es),
        _vote_c("muse", "ABSTAIN", 0.0),
    ]
    c = sr.compute_consensus(votes, min_voter_confidence=0.75)
    assert c.consensus == "quorum"  # the valid votes still agreed
    assert c.routing == "human_review"
    assert c.route_reason == "low_confidence"
    # The abstain's synthetic 0.0 confidence is NOT counted (valid votes only):
    # with all three valid votes at/above the floor the quorum accept stands.
    votes_ok = [_vote_c(p, "A", 0.9, es) for p in ("claude", "codex", "kimi")]
    votes_ok.append(_vote_c("muse", "ABSTAIN", 0.0))
    c2 = sr.compute_consensus(votes_ok, min_voter_confidence=0.75)
    assert c2.routing == "auto_accept"
    assert c2.route_reason == "quorum"


def test_consensus_quad_quorum_accept_still_size_and_class_gated():
    """The size and class gates demote QUORUM accepts exactly like unanimous ones."""
    es = frozenset({(R1, T1)})
    votes = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("kimi", "A", es),
        _vote("muse", "ABSTAIN"),
    ]
    c = sr.compute_consensus(votes, n_candidate_edges=_backstop() + 1)
    assert c.consensus == "quorum"
    assert c.routing == "human_review"
    assert c.route_reason == "size_gated"
    c2 = sr.compute_consensus(votes, edge_classes=[("footway", "residential")])
    assert c2.routing == "human_review"
    assert c2.route_reason == "class-mismatch"


def test_consensus_quad_3_1_live_split_majority_human_review():
    """(c) A live 3-1 split (no abstains) -> majority to human review, dissent
    stamped. Quorum forgives ABSTENTION only, never disagreement: 3 agreeing
    valid votes with a 4th DISSENTING valid vote must never quorum-accept."""
    es = frozenset({(R1, T1)})
    votes = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("kimi", "A", es),
        _vote("muse", "B"),
    ]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.routing == "human_review"
    assert c.choice == "A"
    assert "muse=B" in c.minority
    assert c.route_reason == "dissent:muse=B"


def test_consensus_quad_2valid_2abstain_below_quorum():
    """(d) 2 valid agree + 2 ABSTAIN -> below quorum (n_valid<3), human review."""
    es = frozenset({(R1, T1)})
    votes = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("kimi", "ABSTAIN"),
        _vote("muse", "ABSTAIN"),
    ]
    c = sr.compute_consensus(votes)
    assert c.routing == "human_review"
    assert c.n_valid == 2
    assert c.route_reason == "below_quorum:2"


def test_quorum_rule_is_noop_for_3voter_panels():
    """REGRESSION PROOF: for ANY 3-voter panel (v2/v3/v4 composition re-runs),
    the v5 quorum rule routes byte-identically to the pre-v5 rule.

    Sweeps every 3-vote combination over {A, B, NONE, ABSTAIN} (64 panels) and
    compares the FULL Consensus row (tier, routing, choice, edge_set, counts,
    minority, mean_confidence, route_reason) from compute_consensus against a
    verbatim port of the pre-v5 core (auto-accept iff ``agree == len(votes) >=
    3``, with the pre-v5 reason derivation inlined so the oracle is fully
    independent of current code). With 3 voters the two rules coincide:
    all-valid agreement at quorum (>=3 valid) IS full unanimity, and any
    abstention drops n_valid below 3 (``below_quorum``) — so a 3-voter wave
    re-run under v5 reproduces its historical routing exactly. Gates are
    orthogonal (they run on the routed result over valid votes, unchanged) and
    are covered by their own tests.
    """
    import itertools

    es_by_choice = {"A": frozenset({(R1, T1)}), "B": frozenset({(R2, T2)})}
    conf_by_provider = {"claude": 0.8, "codex": 0.85, "agy": 0.95}

    def pre_v5_consensus(votes):
        """Verbatim port of the pre-v5 routing core (no gates), reasons inlined."""
        group_id = votes[0].group_id if votes else ""
        valid = [v for v in votes if v.choice != "ABSTAIN"]
        n_valid = len(valid)
        tally: dict[str, list] = {}
        for v in valid:
            tally.setdefault(v.choice, []).append(v)
        if not tally:
            return sr.Consensus(
                group_id,
                "none",
                "",
                frozenset(),
                "human_review",
                len(votes),
                0,
                "all providers abstained",
                0.0,
                route_reason="all_abstained",
            )
        top_choice = max(tally, key=lambda c: len(tally[c]))
        top_votes = tally[top_choice]
        agree = len(top_votes)
        minority_votes = [v for v in valid if v.choice != top_choice]
        minority = "; ".join(f"{v.provider}={v.choice}" for v in minority_votes)
        mean_conf = round(sum(v.confidence for v in top_votes) / len(top_votes), 3)
        edge_set = top_votes[0].edge_set
        if agree == len(votes) and agree >= 3 and not minority_votes:
            consensus = "unanimous"
            routing = "auto_accept" if top_choice != "NONE" else "human_review"
        elif agree >= 2:
            consensus = "majority"
            routing = "human_review"
        else:
            consensus = "none"
            routing = "human_review"
        # Pre-v5 reason derivation for every row shape reachable gate-free.
        if routing == "auto_accept":
            route_reason = "unanimous"
        elif consensus == "unanimous":  # only a NONE verdict reaches here gate-free
            route_reason = "unanimous_none"
        elif consensus == "majority":
            if minority:
                route_reason = "dissent:" + minority.replace("; ", ",").replace(" ", "")
            elif n_valid >= 3:
                route_reason = "abstention"  # unreachable with 3 voters
            else:
                route_reason = f"below_quorum:{n_valid}"
        else:
            route_reason = "all_abstained" if n_valid == 0 else "no_majority"
        return sr.Consensus(
            group_id=group_id,
            consensus=consensus,
            choice=top_choice,
            edge_set=edge_set,
            routing=routing,
            n_votes=len(votes),
            n_valid=n_valid,
            minority=minority,
            mean_confidence=mean_conf,
            route_reason=route_reason,
        )

    n_auto = 0
    for combo in itertools.product(("A", "B", "NONE", "ABSTAIN"), repeat=3):
        votes = [
            _vote_c(p, ch, conf_by_provider[p], es_by_choice.get(ch, frozenset()))
            for p, ch in zip(("claude", "codex", "agy"), combo, strict=True)
        ]
        new = sr.compute_consensus(votes)
        old = pre_v5_consensus(votes)
        assert new == old, f"3-voter routing diverged for {combo}: {new} != {old}"
        n_auto += new.routing == "auto_accept"
    # Sanity: exactly the two all-same-letter panels (AAA, BBB) auto-accept.
    assert n_auto == 2


def test_consensus_quad_low_conf_gate_min_over_4_valid():
    """(e) The low-confidence gate takes the MIN over all 4 valid votes."""
    es = frozenset({(R1, T1)})
    # 4/4 unanimous but the 4th voter is below a 0.5 floor -> demoted (min of 4).
    votes = [
        _vote_c("claude", "A", 0.9, es),
        _vote_c("codex", "A", 0.9, es),
        _vote_c("kimi", "A", 0.9, es),
        _vote_c("muse", "A", 0.3, es),
    ]
    c = sr.compute_consensus(votes, min_voter_confidence=0.5)
    assert c.consensus == "unanimous"  # the panel still agreed
    assert c.routing == "human_review"
    assert c.route_reason == "low_confidence"
    # All 4 at/above the floor -> auto-accepts (the gate is a strict <).
    votes_ok = [_vote_c(p, "A", 0.9, es) for p in ("claude", "codex", "kimi")]
    votes_ok.append(_vote_c("muse", "A", 0.5, es))
    c2 = sr.compute_consensus(votes_ok, min_voter_confidence=0.5)
    assert c2.routing == "auto_accept"


# ---------------------------------------------------------------------------
# Class-consistency gate
# ---------------------------------------------------------------------------


def test_mode_sets_documented_and_disjoint():
    # The gate's correctness depends on the mode sets being pairwise disjoint and
    # non-empty; a class in two sets would make is_cross_mode_edge ill-defined.
    assert sr.PEDESTRIAN_CLASSES and sr.VEHICULAR_CLASSES and sr.CYCLEWAY_CLASSES
    assert sr.PEDESTRIAN_CLASSES.isdisjoint(sr.VEHICULAR_CLASSES)
    assert sr.PEDESTRIAN_CLASSES.isdisjoint(sr.CYCLEWAY_CLASSES)
    assert sr.VEHICULAR_CLASSES.isdisjoint(sr.CYCLEWAY_CLASSES)
    # Spot-check the canonical members of each mode.
    assert {"footway", "sidewalk", "path"} <= sr.PEDESTRIAN_CLASSES
    assert {"residential", "primary", "service"} <= sr.VEHICULAR_CLASSES
    assert {"cycleway"} <= sr.CYCLEWAY_CLASSES


def test_road_class_mode_classification():
    assert sr.road_class_mode("footway") == "pedestrian"
    assert sr.road_class_mode("PRIMARY") == "vehicular"  # case-insensitive
    assert sr.road_class_mode("residential") == "vehicular"
    # cycleway is its own bike mode (no longer neutral).
    assert sr.road_class_mode("cycleway") == "bike"
    assert sr.road_class_mode("CYCLEWAY") == "bike"  # case-insensitive
    # Ambiguous / unknown / missing -> neutral (never gates).
    assert sr.road_class_mode("track") == "neutral"
    assert sr.road_class_mode("unknown") == "neutral"
    assert sr.road_class_mode("") == "neutral"
    assert sr.road_class_mode(None) == "neutral"


def test_is_cross_mode_edge():
    # Any two DIFFERENT non-neutral modes, either orientation -> cross-mode.
    assert sr.is_cross_mode_edge("footway", "residential")  # pedestrian<->vehicular
    assert sr.is_cross_mode_edge("primary", "sidewalk")
    # road<->cycleway is cross-mode (Brad's 2026-07-05 decision).
    assert sr.is_cross_mode_edge("cycleway", "residential")
    assert sr.is_cross_mode_edge("primary", "cycleway")
    # pedestrian<->cycleway is cross-mode (conservative default; see PR body).
    assert sr.is_cross_mode_edge("footway", "cycleway")
    assert sr.is_cross_mode_edge("cycleway", "sidewalk")
    # Same-mode pairs are not cross-mode.
    assert not sr.is_cross_mode_edge("residential", "primary")
    assert not sr.is_cross_mode_edge("footway", "path")
    assert not sr.is_cross_mode_edge("cycleway", "cycleway")  # bike<->bike stays same-mode
    # Any neutral/missing side passes (do not over-gate on absent data).
    assert not sr.is_cross_mode_edge("track", "residential")
    assert not sr.is_cross_mode_edge("footway", "")
    assert not sr.is_cross_mode_edge("primary", None)


def test_class_gate_demotes_cross_mode_auto_accept():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    # footway ref matched to a residential target -> cross-mode -> demoted.
    c = sr.compute_consensus(votes, edge_classes=[("footway", "residential")])
    assert c.consensus == "unanimous"  # the panel still agreed
    assert c.routing == "human_review"
    assert c.route_reason == "class-mismatch"


def test_class_gate_passes_same_mode_auto_accept():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes, edge_classes=[("residential", "primary")])
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_class_gate_passes_missing_or_neutral_class():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    # Missing target class -> pass.
    c = sr.compute_consensus(votes, edge_classes=[("footway", "")])
    assert c.routing == "auto_accept"
    # Neutral (track) vs vehicular -> pass.
    c2 = sr.compute_consensus(votes, edge_classes=[("track", "residential")])
    assert c2.routing == "auto_accept"


def test_class_gate_demotes_road_cycleway_auto_accept():
    # road<->cycleway (co_bogota_bike_network shape: primary ref, cycleway target)
    # is cross-mode and must route to human review, not auto-accept.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes, edge_classes=[("primary", "cycleway")])
    assert c.consensus == "unanimous"
    assert c.routing == "human_review"
    assert c.route_reason == "class-mismatch"


def test_class_gate_passes_cycleway_cycleway_auto_accept():
    # cycleway<->cycleway is same-mode and stays auto-acceptable.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes, edge_classes=[("cycleway", "cycleway")])
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_class_gate_only_affects_auto_accept():
    # A majority (non-auto-accept) verdict with a cross-mode chosen edge is NOT
    # relabeled "class-mismatch" — it was already routed to human review, and
    # its reason reflects the vote outcome (the dissent), not the class gate.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "B")]
    c = sr.compute_consensus(votes, edge_classes=[("footway", "residential")])
    assert c.routing == "human_review"
    assert c.route_reason == "dissent:agy=B"


def test_class_gate_disabled_without_edge_classes():
    # Backward-compat: no edge_classes -> gate is a no-op.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes)
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


# ---------------------------------------------------------------------------
# Consensus: size gate (export-backstop routing void, #322/#330 follow-up)
# ---------------------------------------------------------------------------


def _unanimous_votes():
    es = frozenset({(R1, T1)})
    return [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]


def _backstop():
    from crosswalk.config import settings

    return settings.stitch_export_backstop_max_edges


def test_size_gate_demotes_over_backstop_unanimous():
    # A unanimous verdict on a group over the export backstop can never mint a
    # label; it must route to a human instead of vanishing in the void.
    c = sr.compute_consensus(_unanimous_votes(), n_candidate_edges=_backstop() + 1)
    assert c.consensus == "unanimous"
    assert c.routing == "human_review"
    assert c.route_reason == "size_gated"


def test_size_gate_passes_at_and_under_backstop():
    # The gate is strictly ">" — a group AT the backstop is still exportable.
    for n in (_backstop(), 1):
        c = sr.compute_consensus(_unanimous_votes(), n_candidate_edges=n)
        assert c.routing == "auto_accept"
        assert c.route_reason == "unanimous"


def test_size_gate_disabled_without_count():
    # Backward-compat: no n_candidate_edges -> gate is a no-op.
    c = sr.compute_consensus(_unanimous_votes(), n_candidate_edges=None)
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_size_gate_only_affects_auto_accept():
    # A dissent verdict on an over-backstop group already routes to a human;
    # its reason keeps the (more specific) vote outcome.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "B")]
    c = sr.compute_consensus(votes, n_candidate_edges=_backstop() + 1)
    assert c.routing == "human_review"
    assert c.route_reason == "dissent:agy=B"


def test_size_gate_wins_over_class_gate():
    # When both gates would demote, the structural export block is the
    # decisive fact — size_gated wins.
    c = sr.compute_consensus(
        _unanimous_votes(),
        edge_classes=[("footway", "residential")],
        n_candidate_edges=_backstop() + 1,
    )
    assert c.routing == "human_review"
    assert c.route_reason == "size_gated"


# ---------------------------------------------------------------------------
# Consensus: low-confidence gate (panel calibration review follow-up)
# ---------------------------------------------------------------------------


def _vote_c(provider, choice, conf, es=frozenset()):
    """A vote with an explicit confidence (the ``_vote`` helper hardcodes 0.9)."""
    return sr.Vote(
        group_id="g",
        provider=provider,
        model="m",
        choice=choice,
        confidence=conf,
        reasoning="",
        edge_set=es,
    )


def _conf_floor():
    from crosswalk.config import settings

    return settings.stitch_min_voter_confidence


def _unanimous_confs(c1, c2, c3):
    """Unanimous 'A' votes with the three given confidences."""
    es = frozenset({(R1, T1)})
    return [
        _vote_c("claude", "A", c1, es),
        _vote_c("codex", "A", c2, es),
        _vote_c("agy", "A", c3, es),
    ]


def test_low_conf_gate_demotes_below_floor():
    # A unanimous verdict whose calibrated voter is below the floor is demoted.
    floor = _conf_floor()
    c = sr.compute_consensus(
        _unanimous_confs(floor - 0.1, 0.95, 0.98),
        min_voter_confidence=floor,
    )
    assert c.consensus == "unanimous"  # the panel still agreed
    assert c.routing == "human_review"
    assert c.route_reason == "low_confidence"


def test_low_conf_gate_uses_minimum_not_mean():
    # One low vote among two inflated ones still demotes: the MINIMUM governs
    # (the Gemini voters' high self-reports must not drown out the calibrated
    # voter's low one).
    floor = _conf_floor()
    votes = _unanimous_confs(floor - 0.2, 1.0, 1.0)
    assert sum(v.confidence for v in votes) / 3 > floor  # mean would pass
    c = sr.compute_consensus(votes, min_voter_confidence=floor)
    assert c.routing == "human_review"
    assert c.route_reason == "low_confidence"


def test_low_conf_gate_passes_at_floor():
    # Boundary: the gate is strictly "<", so a minimum EXACTLY at the floor
    # still auto-accepts.
    floor = _conf_floor()
    c = sr.compute_consensus(
        _unanimous_confs(floor, floor, floor),
        min_voter_confidence=floor,
    )
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_low_conf_gate_passes_above_floor():
    floor = _conf_floor()
    c = sr.compute_consensus(
        _unanimous_confs(floor + 0.01, 0.95, 0.98),
        min_voter_confidence=floor,
    )
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_low_conf_gate_nan_confidence_demotes():
    # A blank/NaN self-report on a valid vote must not silently pass a NaN
    # comparison — it counts as below the floor.
    c = sr.compute_consensus(
        _unanimous_confs(float("nan"), 0.98, 0.99),
        min_voter_confidence=_conf_floor(),
    )
    assert c.routing == "human_review"
    assert c.route_reason == "low_confidence"


def test_low_conf_gate_disabled_by_none():
    # Backward-compat: no floor -> gate is a no-op even for a low minimum.
    c = sr.compute_consensus(_unanimous_confs(0.1, 0.2, 0.3), min_voter_confidence=None)
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_low_conf_gate_disabled_by_zero_floor():
    # A non-positive floor disables the gate (and does not demote on NaN either).
    c = sr.compute_consensus(_unanimous_confs(float("nan"), 0.2, 0.3), min_voter_confidence=0.0)
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_low_conf_gate_only_affects_auto_accept():
    # A dissent verdict already routes to a human; a low confidence must not
    # overwrite its more specific reason.
    es = frozenset({(R1, T1)})
    votes = [
        _vote_c("claude", "A", 0.2, es),
        _vote_c("codex", "A", 0.2, es),
        _vote_c("agy", "B", 0.2),
    ]
    c = sr.compute_consensus(votes, min_voter_confidence=_conf_floor())
    assert c.routing == "human_review"
    assert c.route_reason == "dissent:agy=B"


def test_size_gate_wins_over_low_conf_gate():
    # Ordering: when both the size gate and the low-confidence gate would demote,
    # the structural export block wins — size_gated, not low_confidence.
    floor = _conf_floor()
    c = sr.compute_consensus(
        _unanimous_confs(floor - 0.2, 0.95, 0.98),
        n_candidate_edges=_backstop() + 1,
        min_voter_confidence=floor,
    )
    assert c.routing == "human_review"
    assert c.route_reason == "size_gated"


def test_class_gate_wins_over_low_conf_gate():
    # Ordering: the class gate runs before the low-confidence gate, so a
    # cross-mode low-confidence auto-accept stamps class-mismatch.
    floor = _conf_floor()
    c = sr.compute_consensus(
        _unanimous_confs(floor - 0.2, 0.95, 0.98),
        edge_classes=[("footway", "residential")],
        min_voter_confidence=floor,
    )
    assert c.routing == "human_review"
    assert c.route_reason == "class-mismatch"


# ---------------------------------------------------------------------------
# Runner: retry-once + abstention (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_provider_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return "garbage no json"
        return '{"choice": "A", "confidence": 0.8, "reasoning": "ok"}'

    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    vote = sr.run_provider_on_group(
        sr.ProviderSpec("claude", "m"),
        "g",
        None,
        "prompt",
        ["A", "B"],
        {"A": [(R1, T1)]},
    )
    assert calls["n"] == 2
    assert vote.choice == "A"
    assert vote.edge_set == frozenset({(R1, T1)})
    assert vote.evidence_delivery == ""  # direct helper: no verified manifest supplied


def test_run_provider_abstains_after_retries(monkeypatch):
    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return "still garbage"

    monkeypatch.setitem(sr._INVOKERS, "codex", fake_invoker)
    vote = sr.run_provider_on_group(
        sr.ProviderSpec("codex", "m"),
        "g",
        None,
        "prompt",
        ["A"],
        {"A": [(R1, T1)]},
    )
    assert vote.choice == "ABSTAIN"
    assert vote.error


def test_run_provider_cli_failure_hard_fails(monkeypatch):
    """A non-zero CLI exit is a provider-down signal -> hard-fail, not abstain.

    (Previously this abstained; the panel now halts on invocation/quota errors
    rather than silently dropping the voter. budget=0 hard-fails immediately.)
    """

    def failing_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        raise RuntimeError("codex exited with code 2: auth expired")

    monkeypatch.setitem(sr._INVOKERS, "codex", failing_invoker)
    with pytest.raises(sr.ProviderInvocationError, match="auth expired"):
        sr.run_provider_on_group(
            sr.ProviderSpec("codex", "m"),
            "g",
            None,
            "prompt",
            ["A"],
            {"A": [(R1, T1)]},
            invocation_budget_s=0.0,
        )


def test_check_exit_raises_with_truncated_stderr():
    import subprocess as sp

    result = sp.CompletedProcess(args=["x"], returncode=3, stdout="", stderr="boom " * 200)
    with pytest.raises(RuntimeError) as exc:
        sr._check_exit("agy", result)
    msg = str(exc.value)
    assert "agy exited with code 3" in msg
    assert len(msg) < 600  # stderr truncated


def test_check_exit_passes_on_zero():
    import subprocess as sp

    result = sp.CompletedProcess(args=["x"], returncode=0, stdout="ok", stderr="")
    sr._check_exit("claude", result)  # no raise


def test_image_paths_include_junction_zoom_crops(tmp_path):
    """codex only sees images attached via -i, so zoom_*.png must be listed.

    Regression for the enriched_ab1 wave: #302 packs referenced junction zoom
    crops in the prompt, but _image_paths omitted them, leaving codex blind to
    the crops that claude/agy read by path.
    """
    (tmp_path / "overview.png").write_bytes(b"png")
    (tmp_path / "option_A.png").write_bytes(b"png")
    (tmp_path / "zoom_R3_T8.png").write_bytes(b"png")
    (tmp_path / "zoom_R1_T2.png").write_bytes(b"png")

    imgs = sr._image_paths(tmp_path, ["A"])

    assert imgs[0].endswith("overview.png")
    assert imgs[1].endswith("option_A.png")
    assert [p for p in imgs if "zoom_" in p] == [
        str(tmp_path / "zoom_R1_T2.png"),
        str(tmp_path / "zoom_R3_T8.png"),
    ]


# ---------------------------------------------------------------------------
# Evidence metadata correctness
# ---------------------------------------------------------------------------


def test_build_metadata_option_table():
    g = make_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    assert meta["group_id"] == "grp001"
    assert meta["match_type"] == "M:N"
    assert meta["optimizer_letter"] == "A"
    assert meta["n_ref_segments"] == 2
    # Option A is optimizer with 2 edges.
    opt_a = next(o for o in meta["options"] if o["letter"] == "A")
    assert opt_a["is_optimizer"] is True
    assert opt_a["edge_count"] == 2
    # Edge label uses R#/T# and carries confidence.
    e = next(x for x in opt_a["edges"] if x["target"] == "T1")
    assert e["edge"] == "R1->T1"
    assert e["confidence"] == 0.9
    assert e["ref_aligned_frac"] == 1.0
    # Segment names/classes present.
    assert meta["segments"]["reference"][0]["name"] == "Main St"


def test_generate_group_evidence_writes_files(tmp_path):
    g = make_group()
    meta = generate_group_evidence(g, tmp_path / g["group_id"])
    d = tmp_path / g["group_id"]
    assert (d / "overview.png").exists()
    assert (d / "option_A.png").exists()
    assert (d / "option_B.png").exists()
    assert (d / "metadata.yaml").exists()
    assert (d / "prompt.txt").exists()
    assert (d / "evidence.json").exists()
    prompt = (d / "prompt.txt").read_text()
    assert '"choice"' in prompt
    assert "NONE" in prompt
    assert meta is not None


def test_evidence_manifest_records_exact_menu_and_decision_provenance(tmp_path):
    g = make_group()
    g["edges"][0].update(
        selected=False,
        decision="review",
        review_reason="coverage_conflict",
    )
    d = tmp_path / g["group_id"]

    generate_group_evidence(
        g,
        d,
        source_artifacts={"groups_sidecar": {"available": True, "sha256": "a" * 64}},
    )
    manifest = load_evidence_manifest(d, allow_legacy=False)
    evidence = manifest["evidence"]

    assert evidence["selectable_choices"] == ["A", "B", "NONE"]
    assert evidence["source_candidate_edges"] == [
        {"ref_id": R1, "target_id": T1},
        {"ref_id": R1, "target_id": T2},
        {"ref_id": R2, "target_id": T2},
    ]
    assert evidence["displayed_candidate_count"] == 3
    assert {(edge["ref_id"], edge["target_id"]) for edge in evidence["displayed_edges"]} == {
        (R1, T1),
        (R1, T2),
        (R2, T2),
    }
    reviewed = next(
        edge
        for edge in evidence["displayed_edges"]
        if (edge["ref_id"], edge["target_id"]) == (R1, T1)
    )
    assert reviewed["decision"] == "review"
    assert reviewed["review_reason"] == "coverage_conflict"
    assert evidence["source_artifacts"]["groups_sidecar"]["sha256"] == "a" * 64
    assert len(evidence["option_menu"]) == 2
    assert all(len(option["option_id"]) == 64 for option in evidence["option_menu"])
    assert len(manifest["evidence_pack_sha256"]) == 64


def test_evidence_empty_candidate_graph_falls_back_to_legacy_edge_universe(tmp_path):
    g = make_group()
    g["candidate_edges"] = []
    d = tmp_path / g["group_id"]

    generate_group_evidence(g, d)
    evidence = load_evidence_manifest(d, allow_legacy=False)["evidence"]

    assert evidence["source_universe_kind"] == "edges+rejected_edges"
    assert evidence["source_candidate_count"] == 3


def test_evidence_regeneration_removes_stale_managed_assets(tmp_path):
    g = make_group()
    d = tmp_path / g["group_id"]
    generate_group_evidence(g, d)
    (d / "option_Z.png").write_bytes(b"stale")
    (d / "zoom_stale.png").write_bytes(b"stale")

    generate_group_evidence(g, d)

    assert not (d / "option_Z.png").exists()
    assert not (d / "zoom_stale.png").exists()
    load_evidence_manifest(d, allow_legacy=False)  # exact managed-file set verifies


def test_evidence_manifest_rejects_pack_mutation(tmp_path):
    g = make_group()
    d = tmp_path / g["group_id"]
    generate_group_evidence(g, d)
    (d / "prompt.txt").write_text("tampered prompt")

    with pytest.raises(EvidenceProvenanceError, match="files changed"):
        load_evidence_manifest(d, allow_legacy=False)


def test_evidence_manifest_cannot_silently_downgrade_new_pack(tmp_path):
    g = make_group()
    d = tmp_path / g["group_id"]
    generate_group_evidence(g, d)
    (d / "evidence.json").unlink()

    with pytest.raises(EvidenceProvenanceError, match="provenance-aware"):
        load_evidence_manifest(d)


def test_evidence_manifest_binds_group_identity_to_directory(tmp_path):
    g = make_group()
    original = tmp_path / g["group_id"]
    renamed = tmp_path / "different-group"
    generate_group_evidence(g, original)
    original.rename(renamed)

    with pytest.raises(EvidenceProvenanceError, match="group identity mismatch"):
        load_evidence_manifest(renamed, allow_legacy=False)


@pytest.mark.parametrize("bad_id", ["", True, {"structured": "id"}, "../escape"])
def test_evidence_rejects_unsafe_ids_before_writing(tmp_path, bad_id):
    g = make_group()
    g["group_id"] = bad_id
    d = tmp_path / "would-be-pack"

    with pytest.raises(EvidenceProvenanceError):
        generate_group_evidence(g, d)
    assert not d.exists()


def test_prompt_dedupes_edge_descriptors_across_options(tmp_path):
    """Each distinct edge is described ONCE in an EDGES legend; options reference
    short ids. A shared edge is no longer re-printed in longhand per option.

    make_group() has options A={R1->T1, R2->T2} and B={R1->T1, R1->T2}, so R1->T1
    is shared -> its descriptor (`conf=`) line must appear exactly once.
    """
    g = make_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    prompt = build_prompt(tmp_path, meta, ctx)

    # Legend header present; options no longer print `conf=` lines themselves.
    assert "EDGES (" in prompt
    lines = prompt.splitlines()
    edges_i = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES ("))
    options_i = next(i for i, ln in enumerate(lines) if ln.startswith("OPTIONS:"))
    legend = lines[edges_i:options_i]
    options_block = lines[options_i:]

    # Distinct edge universe = {R1->T1, R2->T2, R1->T2} -> exactly 3 legend rows,
    # each carrying full detail once (the shared R1->T1 appears exactly once).
    legend_rows = [ln for ln in legend if "conf=" in ln]
    assert len(legend_rows) == 3
    assert sum("R1->T1" in ln for ln in legend_rows) == 1
    # No descriptor `conf=` lines leak into the OPTIONS section.
    assert not any("conf=" in ln for ln in options_block if ln.startswith("      "))

    # Build the short-id -> R#->T# map from the legend, then confirm each option's
    # `edges:` line reconstructs exactly the option's true edge set (unambiguous).
    id_to_edge = {}
    for ln in legend_rows:
        eid, rest = ln.strip().split(":", 1)
        id_to_edge[eid] = rest.strip().split()[0]
    # Options are emitted in order, each with one `edges:` line listing its short ids.
    edge_lines = [
        ln.strip()[len("edges: ") :] for ln in options_block if ln.strip().startswith("edges:")
    ]
    assert len(edge_lines) == len(meta["options"])
    for opt, refs in zip(meta["options"], edge_lines):
        got = {id_to_edge[r.strip()] for r in refs.split(",")}
        want = {e["edge"] for e in opt["edges"]}
        assert got == want


def test_generate_group_evidence_no_options_returns_none(tmp_path):
    g = make_group()
    d = tmp_path / "x"
    generate_group_evidence(g, tmp_path / g["group_id"])
    # Simulate reuse of this path for the refreshed no-option representation.
    (tmp_path / g["group_id"]).rename(d)
    g["optimizer_assignment"] = []
    g["alternatives"] = []
    assert generate_group_evidence(g, d) is None
    assert not (d / "prompt.txt").exists()
    assert not (d / "evidence.json").exists()


# ---------------------------------------------------------------------------
# #267 structural pass-through + junction-sliver display (packs enrichment)
# ---------------------------------------------------------------------------


def _line_len(length_m, lat=42.36, lon=-71.06):
    deg = length_m / (111000.0 * math.cos(math.radians(lat)))
    return {"type": "LineString", "coordinates": [[lon, lat], [lon + deg, lat]]}


def make_struct_group() -> dict:
    """A 1x2 group carrying #267 structural fields and a BORDERLINE junction edge.

    R1->T1 is a full-coverage continuation; R1->T2 shares only 2.9% of a 200 m
    ref (~5.8 m absolute overlap) -> fails the sliver test only on the 5 m floor
    -> BORDERLINE.
    """
    edges = [
        {
            "ref_id": R1,
            "target_id": T1,
            "confidence": 0.99,
            "gers_start_frac": 0.0,
            "gers_end_frac": 1.0,
            "local_start_frac": 0.0,
            "local_end_frac": 1.0,
            "degree_ref": 2,
            "degree_tgt": 1,
            "is_bridge": True,
            "biconnected_block": 3,
            "corridor_ref": 0,
            "corridor_tgt": 0,
        },
        {
            "ref_id": R1,
            "target_id": T2,
            "confidence": 0.88,
            "gers_start_frac": 0.0,
            "gers_end_frac": 0.029,
            "local_start_frac": 0.0,
            "local_end_frac": 0.025,
            "degree_ref": 4,
            "degree_tgt": 2,
            "is_bridge": False,
            "biconnected_block": 3,
            "corridor_ref": 0,
            "corridor_tgt": 1,
        },
    ]
    return {
        "group_id": "grp_struct",
        "match_type": "M:N",
        "ref_ids": [R1],
        "target_ids": [T1, T2],
        "edges": edges,
        "optimizer_assignment": [{"ref_id": R1, "target_id": T1}, {"ref_id": R1, "target_id": T2}],
        "alternatives": [{"edges": [{"ref_id": R1, "target_id": T1}]}],
        "ref_geometries": {R1: _line_len(200.0)},
        "target_geometries": {T1: _line_len(200.0), T2: _line_len(50.0)},
        "ref_names": {R1: "Main St"},
        "target_names": {T1: "Main", T2: "Side"},
        "ref_classes": {R1: "residential"},
        "target_classes": {T1: "residential", T2: "residential"},
        "n_corridors": 2,
        "n_assignment_components": 1,
        "largest_biconnected_block": 3,
        "oversized_group": False,
    }


def test_valid_edges_passes_through_structural_fields():
    """build_stitch_options must keep the six #267 per-edge fields."""
    ctx = build_stitch_options(make_struct_group())
    opt = ctx["options"][0]
    e = next(x for x in opt["edges"] if x["target_id"] == T2)
    for k in (
        "degree_ref",
        "degree_tgt",
        "candidate_graph_bridge",
        "biconnected_block",
        "corridor_ref",
        "corridor_tgt",
    ):
        assert k in e, f"{k} stripped from enriched edge"
    assert e["degree_ref"] == 4
    assert e["corridor_tgt"] == 1


def test_valid_edges_omits_missing_structural_fields():
    """Older sidecars without structural fields degrade gracefully (omitted)."""
    g = make_group()  # fixture with no #267 fields
    ctx = build_stitch_options(g)
    for opt in ctx["options"]:
        for e in opt["edges"]:
            assert "degree_ref" not in e
            assert "candidate_graph_bridge" not in e


def test_build_metadata_surfaces_overlap_tag_and_structure():
    g = make_struct_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    # Group-level structure summary present.
    assert meta["structure"]["n_corridors"] == 2
    assert meta["structure"]["oversized_group"] is False
    opt_a = next(o for o in meta["options"] if o["is_optimizer"])
    assert opt_a["borderline_edge_count"] == 1
    # The junction-kiss edge carries overlap + BORDERLINE tag + struct fields.
    e = next(x for x in opt_a["edges"] if x["target"] == "T2")
    assert e["tag"] == "BORDERLINE"
    assert e["overlap_m"] == pytest.approx(5.8, abs=0.2)
    assert e["degree_ref"] == 4
    assert e["is_sliver"] is False
    # The full-coverage continuation is untagged.
    e1 = next(x for x in opt_a["edges"] if x["target"] == "T1")
    assert "tag" not in e1


def test_metadata_and_prompt_surface_aligned_physical_evidence(tmp_path):
    g = make_struct_group()
    physical = {
        "aligned_range": [0.5, 1.0],
        "level_lr": [{"between": [0.5, 1.0], "value": 1}],
        "road_flags_lr": [{"between": [0.5, 1.0], "value": ["is_bridge"]}],
    }
    g["ref_physical"] = {R1: physical}
    g["edges"][0]["ref_physical"] = physical

    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    assert meta["segments"]["reference"][0]["physical"] == physical
    edge = next(e for e in meta["options"][0]["edges"] if e["target"] == "T1")
    assert edge["ref_physical"]["aligned_range"] == [0.5, 1.0]

    group_dir = tmp_path / "physical"
    generate_group_evidence(g, group_dir)
    prompt = (group_dir / "prompt.txt").read_text()
    assert "R physical: layer 1; bridge" in prompt
    assert "physical='layer 1; bridge'" in prompt
    assert "candidate-graph cut edge" in prompt
    assert "NOT a claim that either road" in prompt


def test_metadata_and_prompt_surface_same_side_coincidence(tmp_path):
    g = make_struct_group()
    g["ref_ids"] = [R1, R2]
    g["ref_geometries"] = {
        R1: _line([[6.0, 46.0], [6.001, 46.0]]),
        R2: _line([[6.0002, 46.00001], [6.0008, 46.00001]]),
    }
    g["ref_classes"] = {R1: "trunk", R2: "cycleway"}

    group_dir = tmp_path / "coincidence"
    meta = generate_group_evidence(g, group_dir, include_same_side_coincidence=True)
    rows = meta["same_side_coincidence"]["reference"]
    assert {row["label"] for row in rows} == {"R1", "R2"}
    assert all(row["role_conflict"] for row in rows)

    prompt = (group_dir / "prompt.txt").read_text()
    assert "Same-side coincidence (experimental geometry context):" in prompt
    assert "R1 overlaps R2" in prompt
    assert "NOT assert a bridge, tunnel, or layer" in prompt


def test_same_side_coincidence_is_opt_in_for_general_evidence(tmp_path):
    g = make_struct_group()
    g["ref_ids"] = [R1, R2]
    g["ref_geometries"] = {
        R1: _line([[6.0, 46.0], [6.001, 46.0]]),
        R2: _line([[6.0002, 46.00001], [6.0008, 46.00001]]),
    }

    group_dir = tmp_path / "no-coincidence"
    meta = generate_group_evidence(g, group_dir)

    assert meta["same_side_coincidence"] == {}
    assert "Same-side coincidence" not in (group_dir / "prompt.txt").read_text()


def test_build_metadata_graceful_without_structure():
    """No #267 fields anywhere -> empty structure dict, no per-edge struct keys."""
    g = make_group()
    meta = build_metadata(g, build_stitch_options(g))
    assert meta["structure"] == {}
    for opt in meta["options"]:
        for e in opt["edges"]:
            assert "degree_ref" not in e


def test_generate_evidence_renders_junction_zoom_and_prompt(tmp_path):
    g = make_struct_group()
    d = tmp_path / g["group_id"]
    meta = generate_group_evidence(g, d)
    # A zoom crop was rendered for the BORDERLINE edge and referenced in metadata.
    assert meta["zoom_crops"] == ["zoom_R1_T2.png"]
    assert (d / "zoom_R1_T2.png").exists()
    prompt = (d / "prompt.txt").read_text()
    assert "BORDERLINE" in prompt
    assert "overlap~" in prompt
    assert "deg R4/T2" in prompt
    assert "Group structure:" in prompt
    assert "junction zoom:" in prompt
    assert str(d.resolve() / "zoom_R1_T2.png") in prompt


def test_no_zoom_crop_for_edge_absent_from_all_options(tmp_path):
    """A flagged group edge that no option displays gets no crop (no orphan files)."""
    g = make_struct_group()
    # A second junction-kiss edge, present in the group's candidate edges but in
    # NO option (not in the optimizer assignment nor any alternative).
    t3 = "tgt_3_88c"
    g["target_ids"].append(t3)
    g["target_geometries"][t3] = _line_len(50.0)
    g["target_names"][t3] = "Orphan"
    g["target_classes"][t3] = "residential"
    g["edges"].append(
        {
            "ref_id": R1,
            "target_id": t3,
            "confidence": 0.5,
            "gers_start_frac": 0.0,
            "gers_end_frac": 0.029,
            "local_start_frac": 0.0,
            "local_end_frac": 0.025,
        }
    )
    d = tmp_path / g["group_id"]
    meta = generate_group_evidence(g, d)
    # Only the in-option BORDERLINE edge (R1->T2) gets a crop; the orphan does not.
    assert meta["zoom_crops"] == ["zoom_R1_T2.png"]
    assert not (d / "zoom_R1_T3.png").exists()
    # Every crop file on disk is referenced by the prompt.
    prompt = (d / "prompt.txt").read_text()
    for z in d.glob("zoom_*.png"):
        assert str(z.resolve()) in prompt


def test_pack_size_and_zoom_crop_cap_bounded(tmp_path):
    """Pack stays in the measured order of magnitude and zoom crops are capped."""
    from crosswalk.agent_labeling import stitch_evidence as se

    g = make_struct_group()
    d = tmp_path / g["group_id"]
    generate_group_evidence(g, d)
    pngs = list(d.glob("*.png"))
    zooms = list(d.glob("zoom_*.png"))
    assert len(zooms) <= se.MAX_ZOOM_CROPS
    total = sum(p.stat().st_size for p in d.glob("*") if p.is_file())
    # Hardest measured packs were ~620 KB; keep the same order of magnitude with
    # a generous ceiling so a runaway crop count would trip this.
    assert total < 1_500_000, f"pack too large: {total} bytes across {len(pngs)} PNGs"


# ---------------------------------------------------------------------------
# Panel option pruning (diverse subset for monster groups)
# ---------------------------------------------------------------------------


def _pairs(spec: list[int]) -> list[tuple[str, str]]:
    return [(f"r{i}", f"t{i}") for i in spec]


def _mk_option(key: str, pairs: list[tuple[str, str]], conf: float, is_optimizer=False) -> dict:
    """Hand-build an option dict shaped like build_stitch_options output."""
    edges = [{"ref_id": r, "target_id": t, "confidence": conf} for r, t in pairs]
    total = round(conf * len(edges), 4)
    return {
        "key": key,
        "label": key,
        "is_optimizer": is_optimizer,
        "edges": edges,
        "edge_count": len(edges),
        "total_confidence": total,
        "mean_confidence": conf,
        "active_refs": sorted({r for r, _ in pairs}),
        "active_targets": sorted({t for _, t in pairs}),
    }


def _mk_ctx(options: list[dict]) -> dict:
    for i, o in enumerate(options):
        o["letter"] = chr(ord("A") + i)
    opt_letter = next((o["letter"] for o in options if o["is_optimizer"]), None)
    return {"options": options, "optimizer_letter": opt_letter}


def _edge_sets(ctx: dict) -> list[frozenset]:
    return [frozenset((e["ref_id"], e["target_id"]) for e in o["edges"]) for o in ctx["options"]]


def test_prune_noop_below_option_count():
    """<= max_options is a no-op: ctx untouched, no provenance (byte-identical packs)."""
    g = make_group()
    ctx = build_stitch_options(g)
    snapshot = copy.deepcopy(ctx)
    assert prune_options_for_panel(ctx, g) is None  # settings defaults
    assert ctx == snapshot


def test_prune_noop_below_distinct_edge_trigger():
    """Many options over FEW distinct edges is a no-op: small groups keep everything."""
    g = make_group()  # 3 distinct candidate edges
    ctx = build_stitch_options(g)
    snapshot = copy.deepcopy(ctx)
    assert prune_options_for_panel(ctx, g, max_options=1, min_distinct_edges_trigger=5) is None
    assert ctx == snapshot


def test_prune_max_min_diversity_beats_confidence():
    """Greedy fill must pick the most DISTINCT option, not the most confident.

    The near-duplicate of the optimizer (symmetric difference 1) has much higher
    total confidence than the fully disjoint option (symmetric difference 10);
    greedy-by-confidence would keep the near-duplicate, diversity keeps the
    disjoint one.
    """
    opt = _mk_option("optimizer", _pairs(list(range(5))), 0.9, is_optimizer=True)
    near_dup = _mk_option("alt1", _pairs(list(range(4))), 0.9)  # optimizer minus one edge
    disjoint = _mk_option("alt2", _pairs(list(range(5, 10))), 0.4)
    ctx = _mk_ctx([opt, near_dup, disjoint])
    group = {
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.5} for r, t in _pairs(list(range(10)))
        ]
    }

    info = prune_options_for_panel(ctx, group, max_options=2, min_distinct_edges_trigger=5)

    assert info == {"n_before": 3, "n_after": 2, "dropped_keys": ["alt1"]}
    assert _edge_sets(ctx) == [
        frozenset(_pairs(list(range(5)))),
        frozenset(_pairs(list(range(5, 10)))),
    ]
    # Survivors re-lettered A, B; optimizer letter tracks the survivor.
    assert [o["letter"] for o in ctx["options"]] == ["A", "B"]
    assert ctx["optimizer_letter"] == "A"


def test_prune_keeps_optimizer_and_seed_options():
    """Optimizer + whole-group seeds (full set / optimizer-selected set) always survive.

    Seed identity does not survive build_stitch_options (no is_seed on options),
    so the pruner re-identifies them by edge set from the group's edges.
    """
    all_pairs = _pairs(list(range(6)))
    selected = all_pairs[:2]
    g = {
        "group_id": "grp_big",
        "match_type": "M:N",
        "ref_ids": [r for r, _ in all_pairs],
        "target_ids": [t for _, t in all_pairs],
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.9, "selected": (r, t) in selected}
            for r, t in all_pairs
        ],
        "optimizer_assignment": [{"ref_id": all_pairs[0][0], "target_id": all_pairs[0][1]}],
        "alternatives": [
            # Seeds as generate_top_k_alternatives appends them (identity lost).
            {"edges": [{"ref_id": r, "target_id": t} for r, t in all_pairs]},
            {"edges": [{"ref_id": r, "target_id": t} for r, t in selected]},
            # High-confidence near-duplicate fillers that diversity should drop.
            {"edges": [{"ref_id": r, "target_id": t} for r, t in [all_pairs[0], all_pairs[2]]]},
            {"edges": [{"ref_id": r, "target_id": t} for r, t in [all_pairs[0], all_pairs[3]]]},
            {"edges": [{"ref_id": r, "target_id": t} for r, t in [all_pairs[0], all_pairs[4]]]},
        ],
    }
    ctx = build_stitch_options(g)
    assert len(ctx["options"]) == 6

    info = prune_options_for_panel(ctx, g, max_options=3, min_distinct_edges_trigger=5)

    assert info is not None and info["n_after"] == 3
    kept = _edge_sets(ctx)
    assert frozenset(all_pairs[:1]) in kept  # optimizer proposal
    assert frozenset(all_pairs) in kept  # full-candidate-set seed
    assert frozenset(selected) in kept  # optimizer-selected-set seed
    assert ctx["optimizer_letter"] == "A"


def test_prune_without_protected_starts_from_highest_confidence():
    """No optimizer / seed option present: greedy seeds from max total_confidence."""
    a = _mk_option("alt1", _pairs(list(range(3))), 0.5)
    b = _mk_option("alt2", _pairs(list(range(3, 6))), 0.9)
    c = _mk_option("alt3", _pairs(list(range(6, 9))), 0.4)
    ctx = _mk_ctx([a, b, c])
    group = {
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.5} for r, t in _pairs(list(range(9)))
        ]
    }

    info = prune_options_for_panel(ctx, group, max_options=1, min_distinct_edges_trigger=5)

    assert info == {"n_before": 3, "n_after": 1, "dropped_keys": ["alt1", "alt3"]}
    assert _edge_sets(ctx) == [frozenset(_pairs(list(range(3, 6))))]
    assert [o["letter"] for o in ctx["options"]] == ["A"]
    assert ctx["optimizer_letter"] is None


def test_generate_evidence_prunes_and_reletters_consistently(tmp_path, monkeypatch):
    """End-to-end: pruning at the metadata level keeps letters, images, prompt,
    and provenance consistent (vote parsing is metadata-letter driven)."""
    from crosswalk.config import settings

    monkeypatch.setattr(settings, "stitch_panel_max_options", 2)
    monkeypatch.setattr(settings, "stitch_panel_prune_min_distinct_edges", 2)

    g = make_group()
    # alt1 = near-duplicate of the optimizer (dropped); alt2 = distinct (kept,
    # re-lettered from C to B).
    g["alternatives"] = [
        {"edges": [{"ref_id": R1, "target_id": T1}]},
        {"edges": [{"ref_id": R1, "target_id": T2}]},
    ]
    d = tmp_path / g["group_id"]
    meta = generate_group_evidence(g, d)

    assert meta["options_pruned"] == {"n_before": 3, "n_after": 2, "dropped_keys": ["alt1"]}
    assert [o["letter"] for o in meta["options"]] == ["A", "B"]
    assert meta["optimizer_letter"] == "A"
    # The survivor re-lettered to B is the distinct alternative, not the near-dup.
    opt_b = next(o for o in meta["options"] if o["letter"] == "B")
    assert {(e["ref"], e["target"]) for e in opt_b["edges"]} == {("R1", "T2")}
    # Images exist exactly for surviving letters.
    assert (d / "option_A.png").exists()
    assert (d / "option_B.png").exists()
    assert not (d / "option_C.png").exists()
    # Prompt agrees with metadata: pruned letters only, pruned choice string.
    prompt = (d / "prompt.txt").read_text()
    assert "Option A (optimizer):" in prompt
    assert "Option B:" in prompt
    assert "Option C" not in prompt
    assert '"<A|B|NONE>"' in prompt
    # Runner-side letter -> edge-set mapping loads the pruned metadata cleanly.
    letters, options_by_letter, _ = sr._load_group_context(d)
    assert letters == ["A", "B"]
    assert set(options_by_letter["B"]) == {(R1, T2)}


def test_generate_evidence_below_threshold_records_no_pruning(tmp_path, monkeypatch):
    """Below-threshold groups: no options_pruned key, full option set intact."""
    from crosswalk.config import settings

    monkeypatch.setattr(settings, "stitch_panel_max_options", 8)
    monkeypatch.setattr(settings, "stitch_panel_prune_min_distinct_edges", 200)

    g = make_group()
    meta = generate_group_evidence(g, tmp_path / g["group_id"])
    assert "options_pruned" not in meta
    assert len(meta["options"]) == 2


# ---------------------------------------------------------------------------
# Eval matching logic (exact + F1)
# ---------------------------------------------------------------------------


def test_edge_prf_exact():
    a = frozenset({(R1, T1), (R2, T2)})
    assert edge_prf(a, a) == (1.0, 1.0, 1.0)


def test_edge_prf_partial():
    pred = frozenset({(R1, T1), (R1, T2)})
    truth = frozenset({(R1, T1), (R2, T2)})
    prec, rec, f1 = edge_prf(pred, truth)
    assert prec == 0.5
    assert rec == 0.5
    assert f1 == 0.5


def test_edge_prf_both_empty_is_perfect():
    assert edge_prf(frozenset(), frozenset()) == (1.0, 1.0, 1.0)


def test_edge_prf_empty_pred_nonempty_truth():
    prec, rec, f1 = edge_prf(frozenset(), frozenset({(R1, T1)}))
    assert f1 == 0.0


def test_recover_labeled_groups_classifies():
    groups = [
        {
            "group_id": "gA",
            "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}],
        },
        {"group_id": "gB", "edges": [{"ref_id": "x", "target_id": "y"}]},
    ]
    human_df = pd.DataFrame(
        [
            # clean: both edges in gA
            {
                "group_id": "h_clean",
                "selected_edges": json.dumps(
                    [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}]
                ),
            },
            # empty selection -> NONE
            {"group_id": "h_empty", "selected_edges": "[]"},
            # lost: edge not in any group
            {
                "group_id": "h_lost",
                "selected_edges": json.dumps([{"ref_id": "gone", "target_id": "nowhere"}]),
            },
        ]
    )
    rec = recover_labeled_groups(groups, human_df)
    assert ("h_clean", "gA") in rec["clean"]
    assert "h_empty" in rec["empty"]
    assert "h_lost" in rec["lost"]
    assert rec["target_group_ids"] == ["gA"]


def test_recover_labeled_groups_tie_breaks_to_smallest_group_id():
    """#354/#367: when a label's edge overlaps two current groups equally, the
    winner must be the lexicographically smallest group_id — deterministically,
    not by hash-seed-dependent set iteration order. The single shared edge
    yields count 1 in each group (a genuine 2-way tie); 'g_aaa' must win over
    'g_zzz' regardless of the group list order below."""
    shared_edge = {"ref_id": R1, "target_id": T1}
    # groups deliberately listed largest-id-first so a non-sorted max() would
    # be free to return 'g_zzz' under some hash seeds.
    groups = [
        {"group_id": "g_zzz", "edges": [shared_edge]},
        {"group_id": "g_aaa", "edges": [shared_edge]},
    ]
    human_df = pd.DataFrame([{"group_id": "h_tie", "selected_edges": json.dumps([shared_edge])}])
    rec = recover_labeled_groups(groups, human_df)
    # single edge fully contained in the chosen group -> clean recovery
    assert ("h_tie", "g_aaa") in rec["clean"]
    assert all(gid != "g_zzz" for _, gid in rec["clean"])


def test_recover_labeled_groups_set_label_not_misread_as_empty():
    """A SET label (empty selected_edges but membership in ref_ids/target_ids)
    is a MATCH assertion: it must recover by membership overlap into the 'set'
    bucket and contribute its group to target_group_ids — never land in
    'empty' (reject-all/NONE)."""
    groups = [
        {
            "group_id": "gA",
            "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}],
        },
    ]
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_set",
                "selected_edges": "[]",
                "label_semantics": "set",
                "ref_ids": json.dumps([R1, R2]),
                "target_ids": json.dumps([T1]),
            },
            {
                "group_id": "h_set_lost",
                "selected_edges": "[]",
                "label_semantics": "set",
                "ref_ids": json.dumps(["vanished_ref"]),
                "target_ids": json.dumps(["vanished_tgt"]),
            },
        ]
    )
    rec = recover_labeled_groups(groups, human_df)
    assert rec["empty"] == []  # neither set row is a reject-all
    assert ("h_set", "gA") in rec["set"]
    assert rec["set_lost"] == ["h_set_lost"]
    assert rec["target_group_ids"] == ["gA"]


def test_recover_empty_reject_all_skips_set_labels():
    """A SET label with a verbatim-surviving group_id must NOT be recovered as
    a reject-all — it asserts a match, not 'no edges'."""
    groups = [{"group_id": "gA", "edges": [{"ref_id": R1, "target_id": T1}]}]
    human_df = pd.DataFrame(
        [
            {
                "group_id": "gA",
                "selected_edges": "[]",
                "label_semantics": "set",
                "ref_ids": json.dumps([R1]),
                "target_ids": json.dumps([T1]),
            },
        ]
    )
    rec = recover_empty_reject_all(groups, human_df)
    assert rec["recovered"] == []
    assert rec["unrecoverable"] == []


def test_recover_empty_reject_all():
    # gA survives verbatim; the reject-all label keyed on gA is recoverable,
    # the one keyed on a vanished group_id is not. Non-empty labels are ignored.
    groups = [{"group_id": "gA", "edges": [{"ref_id": R1, "target_id": T1}]}]
    human_df = pd.DataFrame(
        [
            {"group_id": "gA", "selected_edges": "[]"},  # reject-all, survives
            {"group_id": "gone", "selected_edges": "[]"},  # reject-all, lost
            {
                "group_id": "gA",  # non-empty label -> not a reject-all
                "selected_edges": json.dumps([{"ref_id": R1, "target_id": T1}]),
            },
        ]
    )
    rec = recover_empty_reject_all(groups, human_df)
    assert rec["recovered"] == ["gA"]
    assert rec["unrecoverable"] == ["gone"]


def test_evaluate_batch_end_to_end(tmp_path):
    """Generate a pack, fake a consensus/votes, run the eval."""
    g = make_group()
    batch_dir = tmp_path / "batch"
    generate_group_evidence(g, batch_dir / g["group_id"])

    # Human label picks the optimizer's edge set (A).
    human_es = [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}]
    human_df = pd.DataFrame([{"group_id": "hg", "selected_edges": json.dumps(human_es)}])

    # Panel unanimously chose A (matches human).
    edge_set_str = json.dumps(sorted([[R1, T1], [R2, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": edge_set_str,
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": p,
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": edge_set_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
            for p in ("claude", "codex", "agy")
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    results = evaluate_batch(batch_dir, human_df)
    assert len(results) == 1
    r = results[0]
    assert r.exact_match is True
    assert r.f1 == 1.0
    assert r.option_covered is True

    summary = summarize(results)
    assert summary["panel_exact_rate"] == 1.0
    assert summary["by_provider"]["claude"]["exact_rate"] == 1.0
    assert summary["by_consensus"]["unanimous"]["n"] == 1


def test_evaluate_batch_disagreement(tmp_path):
    g = make_group()
    batch_dir = tmp_path / "batch"
    generate_group_evidence(g, batch_dir / g["group_id"])

    # Human picked a DIFFERENT edge set than the panel.
    human_es = [{"ref_id": R1, "target_id": T2}]
    human_df = pd.DataFrame([{"group_id": "hg", "selected_edges": json.dumps(human_es)}])
    panel_str = json.dumps(sorted([[R1, T1], [R2, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": panel_str,
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": "claude",
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": panel_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    results = evaluate_batch(batch_dir, human_df)
    assert len(results) == 1
    assert results[0].exact_match is False


def test_evaluate_batch_sliver_filtered(tmp_path):
    """Panel includes a junction sliver the human omitted.

    Raw comparison disagrees; the sliver-filtered comparison (slivers removed
    from BOTH sides using batch.json geometries) agrees.
    """
    import math

    def _line_len(length_m, lat=42.36, lon=-71.06):
        deg = length_m / (111000.0 * math.cos(math.radians(lat)))
        return {"type": "LineString", "coordinates": [[lon, lat], [lon + deg, lat]]}

    group = {
        "group_id": "grp001",
        "match_type": "M:N",
        "ref_ids": [R1],
        "target_ids": [T1, T2],
        "edges": [
            {
                "ref_id": R1,
                "target_id": T1,
                "confidence": 0.9,
                "gers_start_frac": 0.0,
                "gers_end_frac": 1.0,
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
            },
            {  # junction sliver: 1 m of a 50 m ref, 0.5 m of a 10 m target
                "ref_id": R1,
                "target_id": T2,
                "confidence": 0.3,
                "gers_start_frac": 0.0,
                "gers_end_frac": 0.02,
                "local_start_frac": 0.0,
                "local_end_frac": 0.05,
            },
        ],
        "optimizer_assignment": [
            {"ref_id": R1, "target_id": T1},
            {"ref_id": R1, "target_id": T2},
        ],
        "alternatives": [],
        "ref_geometries": {R1: _line_len(50.0)},
        "target_geometries": {T1: _line_len(50.0), T2: _line_len(10.0)},
        "ref_names": {R1: "Main St"},
        "target_names": {T1: "Main", T2: "Side"},
        "ref_classes": {R1: "residential"},
        "target_classes": {T1: "residential", T2: "residential"},
    }

    batch_dir = tmp_path / "batch"
    generate_group_evidence(group, batch_dir / group["group_id"])
    (batch_dir / "batch.json").write_text(json.dumps({"groups": [group]}))

    # Human omitted the sliver edge; panel included it.
    human_es = [{"ref_id": R1, "target_id": T1}]
    human_df = pd.DataFrame([{"group_id": "hg", "selected_edges": json.dumps(human_es)}])
    panel_str = json.dumps(sorted([[R1, T1], [R1, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": panel_str,
                "routing": "auto_accept",
                "n_votes": 1,
                "n_valid": 1,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": "claude",
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": panel_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    results = evaluate_batch(batch_dir, human_df)
    assert len(results) == 1
    r = results[0]
    assert r.exact_match is False  # raw: panel has the sliver, human doesn't
    assert r.exact_match_filtered is True  # filtered: sliver removed from both
    assert r.f1_filtered == 1.0

    summary = summarize(results)
    assert summary["panel_exact_rate"] == 0.0
    assert summary["panel_exact_rate_filtered"] == 1.0
    assert summary["n_groups_sliver_affected"] == 1


# ---------------------------------------------------------------------------
# Human-label -> group mapping (edge-level overlap preferred)
# ---------------------------------------------------------------------------


def _mapping_fixtures():
    from crosswalk.agent_labeling.stitch_evidence import build_metadata

    g = make_group()
    ctx = build_stitch_options(g)
    metas = {g["group_id"]: build_metadata(g, ctx)}
    cand = {g["group_id"]: frozenset({(R1, T1), (R1, T2), (R2, T2)})}
    return metas, cand


def test_mapping_requires_edge_overlap_when_candidates_known():
    """A label whose segments are in the group but whose edges never existed
    as candidate edges must NOT map (would skew coverage/agreement metrics)."""
    from crosswalk.agent_labeling.stitch_eval import map_human_labels_to_groups

    metas, cand = _mapping_fixtures()
    # (R2, T1) uses group segments but is not a candidate edge.
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_phantom",
                "selected_edges": json.dumps([{"ref_id": R2, "target_id": T1}]),
            }
        ]
    )
    assert map_human_labels_to_groups(human_df, metas, cand) == {}


def test_mapping_prefers_max_edge_overlap():
    from crosswalk.agent_labeling.stitch_eval import map_human_labels_to_groups

    metas, cand = _mapping_fixtures()
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_one_edge",
                "selected_edges": json.dumps([{"ref_id": R1, "target_id": T1}]),
            },
            {
                "group_id": "h_two_edges",
                "selected_edges": json.dumps(
                    [
                        {"ref_id": R1, "target_id": T1},
                        {"ref_id": R2, "target_id": T2},
                    ]
                ),
            },
        ]
    )
    mapping = map_human_labels_to_groups(human_df, metas, cand)
    assert mapping == {"grp001": "h_two_edges"}


def test_mapping_falls_back_to_segment_membership_without_candidates():
    from crosswalk.agent_labeling.stitch_eval import map_human_labels_to_groups

    metas, _cand = _mapping_fixtures()
    # Same phantom edge; without candidate edges the segment fallback applies.
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_phantom",
                "selected_edges": json.dumps([{"ref_id": R2, "target_id": T1}]),
            }
        ]
    )
    mapping = map_human_labels_to_groups(human_df, metas, None)
    assert mapping == {"grp001": "h_phantom"}


def test_evaluate_batch_uses_batch_json_candidate_edges(tmp_path):
    """With batch.json present, a phantom-edge label must not be mapped."""
    g = make_group()
    batch_dir = tmp_path / "batch"
    generate_group_evidence(g, batch_dir / g["group_id"])
    (batch_dir / "batch.json").write_text(json.dumps({"groups": [g]}))

    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_phantom",
                "selected_edges": json.dumps([{"ref_id": R2, "target_id": T1}]),
            }
        ]
    )
    panel_str = json.dumps(sorted([[R1, T1], [R2, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": panel_str,
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": "claude",
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": panel_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    assert evaluate_batch(batch_dir, human_df) == []


# ---------------------------------------------------------------------------
# Diagnostic pack-feedback instrumentation (wave-local flag; default OFF)
# ---------------------------------------------------------------------------


def test_pack_feedback_column_present_in_votes_schema():
    # The votes.csv schema must carry the diagnostic self-report column.
    assert "pack_feedback" in sr.VOTES_COLUMNS


def test_abstain_reason_is_persisted_in_votes_schema():
    vote = sr.Vote(
        group_id="group",
        provider="codex",
        model="model",
        choice="ABSTAIN",
        confidence=0.0,
        reasoning="",
        abstain_reason=sr.AbstainReason.TIMEOUT,
    )

    assert "abstain_reason" in sr.VOTES_COLUMNS
    assert sr._vote_row(vote)["abstain_reason"] == "timeout"


def test_augment_prompt_only_adds_feedback_request():
    base = 'respond with {"choice": "A", ...}'
    aug = sr.augment_prompt_with_feedback(base)
    assert aug.startswith(base)
    assert "pack_feedback" in aug
    assert "missing_info" in aug and "ambiguities" in aug and "confidence_basis" in aug
    # Default prompt is untouched by the augmentation helper.
    assert "pack_feedback" not in base


def test_extract_pack_feedback_pulls_nested_object():
    raw = json.dumps(
        {
            "choice": "A",
            "confidence": 0.8,
            "reasoning": "clear",
            "pack_feedback": {
                "missing_info": ["no name on T1"],
                "ambiguities": [],
                "confidence_basis": "parallel geometry",
            },
        }
    )
    fb = sr._extract_pack_feedback(raw)
    parsed = json.loads(fb)
    assert parsed["missing_info"] == ["no name on T1"]
    assert parsed["confidence_basis"] == "parallel geometry"


def test_extract_pack_feedback_absent_returns_empty():
    raw = '{"choice": "A", "confidence": 0.8, "reasoning": "clear"}'
    assert sr._extract_pack_feedback(raw) == ""
    assert sr._extract_pack_feedback("total garbage no json") == ""


def test_extract_pack_feedback_empty_object_returns_empty():
    raw = '{"choice": "A", "confidence": 0.8, "reasoning": "x", "pack_feedback": {}}'
    assert sr._extract_pack_feedback(raw) == ""


def test_parse_vote_ignores_pack_feedback_key():
    # A vote carrying the diagnostic key still parses to a valid choice.
    raw = json.dumps(
        {
            "choice": "B",
            "confidence": 0.7,
            "reasoning": "ok",
            "pack_feedback": {"ambiguities": ["z"]},
        }
    )
    choice, conf, _ = sr.parse_vote(raw, {"A", "B"})
    assert choice == "B" and conf == 0.7


# ---------------------------------------------------------------------------
# Fourth voter: opencode invoker + named panel config (default OFF)
# ---------------------------------------------------------------------------


def test_default_panel_is_v5_quad():
    """The blessed v5 QUAD is the default: claude + codex/gpt-5.6-terra +
    kimi/Kimi K2.6 + muse/Muse Spark 1.1 (the 2026-07-10 bless, paired with
    the quorum consensus rule)."""
    assert [p.name for p in sr.DEFAULT_PANEL] == ["claude", "codex", "kimi", "muse"]
    claude, codex, kimi, muse = sr.DEFAULT_PANEL
    assert claude.model == "claude-opus-4-8" and claude.effort == "medium"
    assert codex.model == "gpt-5.6-terra" and codex.effort == "medium"
    assert kimi.model == "openrouter/moonshotai/kimi-k2.6"
    # Kimi's thinking runs long on large packs: the spec carries its own timeout.
    assert kimi.timeout == 480
    # Kimi runs tool-less under the same ``vote`` agent as Muse (7/30 -> 0/30
    # timeout evidence): a voter with pre-attached packs needs no tools.
    assert kimi.opencode_agent == "vote"
    # The muse seat: distinct provider name, Meta-API model ref, reasoning-model
    # timeout, tool-less agent.
    assert muse is sr.MUSE
    assert muse.name == "muse" and muse.model == "meta/muse-spark-1.1"
    assert muse.timeout == 480 and muse.opencode_agent == "vote"
    # Distinct provider names: no two voters collide on the keying field.
    assert len({p.name for p in sr.DEFAULT_PANEL}) == 4
    assert sr.get_panel("default") is sr.DEFAULT_PANEL
    assert sr.get_panel("v5") is sr.DEFAULT_PANEL
    # The calibration name that became v5 stays addressable as an alias.
    assert sr.get_panel("quad-candidate") is sr.DEFAULT_PANEL
    # Empty/None means "no choice made" -> the default panel.
    assert sr.get_panel(None) is sr.DEFAULT_PANEL
    assert sr.get_panel("") is sr.DEFAULT_PANEL


def test_v4_panel_remains_reproducible():
    """The former 3-seat default stays addressable as 'v4' (v4-era waves must
    be reproducible), and the v5 default is exactly v4 + muse."""
    v4 = sr.get_panel("v4")
    assert v4 is sr.PANEL_V4
    assert [p.name for p in v4] == ["claude", "codex", "kimi"]
    assert [p.model for p in v4] == [
        "claude-opus-4-8",
        "gpt-5.6-terra",
        "openrouter/moonshotai/kimi-k2.6",
    ]
    assert v4[0].effort == "medium" and v4[1].effort == "medium"
    assert v4 != sr.DEFAULT_PANEL
    # v5 == v4 + muse, seat for seat (composition lineage, not coincidence).
    assert [*v4, sr.MUSE] == sr.DEFAULT_PANEL


def test_v6_candidate_is_lean_claude_codex_muse_trio():
    """v6 is opt-in and removes Kimi without forcing a replacement seat."""
    v6 = sr.get_panel("v6-candidate")
    assert v6 is sr.PANEL_V6_CANDIDATE
    assert [p.name for p in v6] == ["claude", "codex", "muse"]
    assert [p.model for p in v6] == [
        "claude-opus-4-8",
        "gpt-5.6-terra",
        "meta/muse-spark-1.1",
    ]
    assert v6[2] is sr.MUSE
    assert {"kimi", "gemini"}.isdisjoint(p.name for p in v6)
    assert len({p.name for p in v6}) == 3
    # Candidate work must not silently change today's blessed default.
    assert sr.get_panel("default") is sr.DEFAULT_PANEL
    assert sr.get_panel("v5") is sr.DEFAULT_PANEL
    # Gemini remains available only on explicit four-seat calibration panels.
    assert [p.name for p in sr.get_panel("v6-agy-calibration")] == [
        "claude",
        "codex",
        "gemini",
        "muse",
    ]
    assert sr.get_panel("v6-agy-calibration")[2].routes == (sr.GEMINI_ROUTE_AGY,)
    assert sr.get_panel("v6-flex-calibration")[2].routes == (sr.GEMINI_ROUTE_OPENROUTER_FLEX,)


def test_v7_candidate_is_high_effort_sol_replay_trio():
    """V7 is a new Sol/high generation; v5/v6 reproduction stays unchanged."""
    v7 = sr.get_panel("v7-candidate")
    assert v7 is sr.PANEL_V7_CANDIDATE
    assert [p.name for p in v7] == ["claude", "codex", "muse"]
    assert [p.model for p in v7] == [
        "claude-opus-4-8",
        "gpt-5.6-sol",
        "meta/muse-spark-1.1",
    ]
    assert [p.effort for p in v7] == ["high", "high", "high"]
    assert v7[2] is sr.MUSE_HIGH_EFFORT
    assert v7[2].timeout == 480 and v7[2].opencode_agent == "vote"
    assert {"kimi", "gemini"}.isdisjoint(p.name for p in v7)

    # Historical candidates and the production default retain their exact
    # model/effort identities.
    v6 = sr.get_panel("v6-candidate")
    assert [p.model for p in v6] == [
        "claude-opus-4-8",
        "gpt-5.6-terra",
        "meta/muse-spark-1.1",
    ]
    assert [p.effort for p in v6] == ["medium", "medium", ""]
    assert sr.get_panel("default") is sr.DEFAULT_PANEL


def test_gemini_primary_route_records_agy(monkeypatch, tmp_path):
    calls = {"agy": 0, "openrouter": 0}

    def fake_agy(*args, **kwargs):
        calls["agy"] += 1
        return '{"choice":"A","confidence":0.9,"reasoning":"agy"}'

    def fake_openrouter(*args, **kwargs):
        calls["openrouter"] += 1
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(sr, "invoke_agy", fake_agy)
    monkeypatch.setattr(sr, "invoke_opencode", fake_openrouter)
    result = sr.invoke_gemini("p", tmp_path, ["A"], sr.GEMINI_MODEL)
    assert result.route == sr.GEMINI_ROUTE_AGY
    assert '"choice":"A"' in result.raw
    assert calls == {"agy": 1, "openrouter": 0}


def test_gemini_quota_fallback_opens_sticky_wave_circuit(monkeypatch, tmp_path):
    calls = {"agy": 0, "openrouter": 0}

    def quota_capped(*args, **kwargs):
        calls["agy"] += 1
        return ""  # observed agy quota-cap behavior: exit 0 + empty stdout

    def flex(*args, **kwargs):
        calls["openrouter"] += 1
        return '{"choice":"A","confidence":0.8,"reasoning":"flex"}'

    monkeypatch.setattr(sr, "invoke_agy", quota_capped)
    monkeypatch.setattr(sr, "invoke_opencode", flex)
    state = sr.ProviderRouteState()
    first = sr.invoke_gemini("p", tmp_path, ["A"], sr.GEMINI_MODEL, route_state=state)
    second = sr.invoke_gemini("p", tmp_path, ["A"], sr.GEMINI_MODEL, route_state=state)
    assert first.route == second.route == sr.GEMINI_ROUTE_OPENROUTER_FLEX
    assert calls == {"agy": 1, "openrouter": 2}
    assert sr.GEMINI_ROUTE_AGY in state.unavailable


def test_gemini_fallback_rejects_images_mutated_by_primary_route(monkeypatch, tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group_dir = _write_min_pack(batch_dir, "g1")
    manifest = load_evidence_manifest(group_dir)
    prompt = (group_dir / "prompt.txt").read_text()
    scratch, rewritten, scratch_ctx = sr._scratch_pack(group_dir, prompt, manifest)
    flex_called = False

    def mutating_agy(*args, **kwargs):
        (scratch / "overview.png").write_bytes(b"mutated by primary route")
        return ""  # provider failure would ordinarily fall back to flex

    def flex(*args, **kwargs):
        nonlocal flex_called
        flex_called = True
        return '{"choice":"A","confidence":0.8,"reasoning":"flex"}'

    monkeypatch.setattr(sr, "invoke_agy", mutating_agy)
    monkeypatch.setattr(sr, "invoke_opencode", flex)
    try:
        with pytest.raises(EvidenceProvenanceError, match="native attachment.*manifest"):
            sr.invoke_gemini(
                rewritten,
                scratch,
                ["A"],
                sr.GEMINI_MODEL,
                evidence_manifest=manifest,
            )
    finally:
        scratch_ctx.cleanup()

    assert not flex_called


def test_gemini_flex_only_calibration_never_touches_agy(monkeypatch, tmp_path):
    def unexpected_agy(*args, **kwargs):
        raise AssertionError("flex-only calibration must not consume agy quota")

    monkeypatch.setattr(sr, "invoke_agy", unexpected_agy)
    monkeypatch.setattr(
        sr,
        "invoke_opencode",
        lambda *a, **k: '{"choice":"A","confidence":0.8,"reasoning":"flex"}',
    )
    result = sr.invoke_gemini(
        "p",
        tmp_path,
        ["A"],
        sr.GEMINI_MODEL,
        routes=(sr.GEMINI_ROUTE_OPENROUTER_FLEX,),
    )
    assert result.route == sr.GEMINI_ROUTE_OPENROUTER_FLEX


def test_gemini_context_overflow_falls_back_without_poisoning_later_groups(monkeypatch, tmp_path):
    calls = {"agy": 0, "openrouter": 0}

    def overflow(*args, **kwargs):
        calls["agy"] += 1
        raise sr.GroupScopedProviderError("context window")

    def flex(*args, **kwargs):
        calls["openrouter"] += 1
        return '{"choice":"A","confidence":0.8,"reasoning":"flex"}'

    monkeypatch.setattr(sr, "invoke_agy", overflow)
    monkeypatch.setattr(sr, "invoke_opencode", flex)
    state = sr.ProviderRouteState()
    sr.invoke_gemini("p1", tmp_path, ["A"], sr.GEMINI_MODEL, route_state=state)
    sr.invoke_gemini("p2", tmp_path, ["A"], sr.GEMINI_MODEL, route_state=state)
    assert calls == {"agy": 2, "openrouter": 2}
    assert sr.GEMINI_ROUTE_AGY not in state.unavailable


def test_gemini_dual_route_failure_hard_fails_panel(monkeypatch):
    monkeypatch.setattr(sr, "invoke_agy", lambda *a, **k: "")

    def flex_down(*args, **kwargs):
        raise RuntimeError("flex unavailable")

    monkeypatch.setattr(sr, "invoke_opencode", flex_down)
    with pytest.raises(sr.ProviderInvocationError, match="Halting the run"):
        sr.run_provider_on_group(
            sr.GEMINI,
            "g",
            None,
            "p",
            ["A"],
            {"A": [(R1, T1)]},
            invocation_budget_s=0,
            route_state=sr.ProviderRouteState(),
        )


def test_gemini_vote_records_actual_route(monkeypatch):
    def fake_gemini(*args, **kwargs):
        return sr.InvocationResult(
            '{"choice":"A","confidence":0.9,"reasoning":"ok"}',
            sr.GEMINI_ROUTE_OPENROUTER_FLEX,
        )

    monkeypatch.setitem(sr._INVOKERS, "gemini", fake_gemini)
    vote = sr.run_provider_on_group(sr.GEMINI, "g", None, "p", ["A"], {"A": [(R1, T1)]})
    assert vote.choice == "A"
    assert vote.invocation_route == sr.GEMINI_ROUTE_OPENROUTER_FLEX
    assert sr._vote_row(vote)["invocation_route"] == sr.GEMINI_ROUTE_OPENROUTER_FLEX


def test_gemini_flex_endpoint_is_pinned_without_openrouter_rerouting():
    config = json.loads((Path(__file__).parents[2] / "opencode.json").read_text())
    policy = config["provider"]["openrouter"]["models"][sr.GEMINI_MODEL]["options"]["provider"]
    assert policy == {
        "only": ["google-ai-studio/flex"],
        "allow_fallbacks": False,
    }
    isolated = sr._gemini_flex_opencode_config(sr.GEMINI_MODEL)
    assert (
        isolated["provider"]["openrouter"]["models"][sr.GEMINI_MODEL]["options"]["provider"]
        == policy
    )
    assert isolated["agent"]["vote"]["tools"]
    assert not any(isolated["agent"]["vote"]["tools"].values())


def test_gemini_flex_injects_isolated_route_config(monkeypatch, tmp_path):
    captured = {}

    def fake_openrouter(*args, **kwargs):
        captured.update(kwargs)
        return '{"choice":"A","confidence":0.8,"reasoning":"flex"}'

    monkeypatch.setattr(sr, "invoke_opencode", fake_openrouter)
    result = sr.invoke_gemini(
        "p",
        tmp_path,
        ["A"],
        sr.GEMINI_MODEL,
        routes=(sr.GEMINI_ROUTE_OPENROUTER_FLEX,),
    )
    assert result.route == sr.GEMINI_ROUTE_OPENROUTER_FLEX
    config = captured["config_content"]
    assert config["model"] == f"openrouter/{sr.GEMINI_MODEL}"
    assert (
        config["provider"]["openrouter"]["models"][sr.GEMINI_MODEL]["options"]["provider"]
        == sr._GEMINI_FLEX_PROVIDER_POLICY
    )


def test_gemini_route_order_changes_panel_invocation_signature():
    from crosswalk.agent_labeling.stitch_provenance import invocation_signature

    reversed_routes = sr.ProviderSpec(
        name="gemini",
        model=sr.GEMINI_MODEL,
        opencode_agent="vote",
        routes=tuple(reversed(sr.GEMINI.routes)),
    )
    kwargs = {
        "timeout": None,
        "collect_feedback": False,
        "invocation_budget_s": 300.0,
        "effective_timeouts": [240],
        "runtime_contract_sha256": "a" * 64,
    }
    assert invocation_signature([sr.GEMINI], **kwargs) != invocation_signature(
        [reversed_routes], **kwargs
    )


def test_get_panel_unknown_name_is_a_hard_error():
    """Panel choice is era-load-bearing (it decides the export labeler
    generation), so a typo must error listing the valid names — never silently
    run the default panel (#398 review, finding 5)."""
    with pytest.raises(ValueError, match="unknown panel 'v3-candiate'.*valid panels"):
        sr.get_panel("v3-candiate")
    # The error enumerates every valid name.
    with pytest.raises(ValueError) as exc:
        sr.get_panel("bogus")
    for name in sr.PANELS:
        assert name in str(exc.value)


def test_v3_panels_remain_reproducible():
    """The former default stays addressable as 'v3'/'v2' (old batches must be
    reproducible), and the v3-era candidate/fallback panels build on it."""
    assert "opencode" in sr._INVOKERS
    v3 = sr.get_panel("v3")
    assert [p.name for p in v3] == ["claude", "codex", "agy"]
    assert [p.model for p in v3] == [
        "claude-opus-4-8",
        "gpt-5.5",
        "Gemini 3.5 Flash (Medium)",
    ]
    assert v3[1].effort == "low"  # codex ran low effort in the v3 era
    assert sr.get_panel("v2") is v3 or sr.get_panel("v2") == v3
    # v3-candidate adds opencode/Qwen as a distinct 4th voter on top of v3.
    v3c = sr.get_panel("v3-candidate")
    assert [p.name for p in v3c] == ["claude", "codex", "agy", "opencode"]
    assert v3c[3].model == "openrouter/qwen/qwen3-vl-235b-a22b-instruct"
    # no-agy swaps agy for opencode/Qwen (v3-era quota-outage fallback).
    noagy = sr.get_panel("no-agy")
    assert [p.name for p in noagy] == ["claude", "codex", "opencode"]
    assert noagy[1].model == "gpt-5.5"
    assert noagy[2].model == "openrouter/qwen/qwen3-vl-235b-a22b-instruct"


def test_v4_candidate_panel_is_the_397_validation_composition():
    """v4-candidate stays the EXACT #397 validation composition (codex still
    gpt-5.5/low, agy swapped for Kimi) — it is NOT an alias of the blessed v4
    default, whose codex model was also bumped. It therefore remains
    nonstandard to the stitch-export (provider, model) gate; kept only so the
    validation waves can be reproduced."""
    v4c = sr.get_panel("v4-candidate")
    assert [p.name for p in v4c] == ["claude", "codex", "kimi"]
    assert v4c[1].model == "gpt-5.5" and v4c[1].effort == "low"
    assert v4c[2].model == "openrouter/moonshotai/kimi-k2.6"
    assert v4c != sr.DEFAULT_PANEL


def test_meta_candidate_panel_composition():
    """meta-candidate remains the historical Muse-REPLACEMENT prototype: the
    v4 trio with the kimi/Kimi seat swapped for 'muse' (Muse Spark 1.1 on
    Meta's API). SUPERSEDED by v5, which seats muse as a FOURTH voter
    alongside kimi rather than replacing it — so meta-candidate is NOT the
    default and stays NONSTANDARD to the stitch-export (provider, model) gate
    (3 voters; no blessed 3-seat set contains the muse pair); kept only to
    reproduce the Muse validation waves. Muse's provider NAME is the distinct
    "muse" (not the transport name "opencode", nor the Kimi seat's "kimi") so
    it stays individually addressable at every provider-keyed site.
    """
    meta = sr.get_panel("meta-candidate")
    assert [p.name for p in meta] == ["claude", "codex", "muse"]
    claude, codex, muse = meta
    assert claude.model == "claude-opus-4-8" and claude.effort == "medium"
    assert codex.model == "gpt-5.6-terra" and codex.effort == "medium"
    # The Muse voter: distinct name, Meta-API model ref, reasoning-model timeout,
    # tool-less agent.
    assert muse is sr.MUSE
    assert muse.name == "muse"
    assert muse.model == "meta/muse-spark-1.1"
    assert muse.timeout == 480  # reasoning model runs long on large packs
    assert muse.opencode_agent == "vote"  # tool-less agent forces a pure-text vote
    # NOT the blessed default (v5 seats BOTH kimi and muse; meta-candidate
    # replaced kimi) -> nonstandard to the export gate.
    assert meta != sr.DEFAULT_PANEL
    assert "kimi" not in {p.name for p in meta}
    # Muse dispatches through the SAME transport as Kimi despite the distinct name.
    assert sr._INVOKERS["muse"] is sr.invoke_opencode
    # Both opencode-transport voters (kimi + muse) resolve to invoke_opencode.
    assert sr._INVOKERS["kimi"] is sr.invoke_opencode


def test_invoke_opencode_agent_flag(monkeypatch, tmp_path):
    """The opencode invoker adds ``--agent`` only when an agent is passed.

    Called with no agent (the residual Qwen seat) the command stays
    byte-identical to before this knob existed; passing ``agent="vote"`` runs
    under the tool-less agent (both Kimi and Muse do this — without it a voter
    burns its turn on auto-rejected ls/cat/read tool calls instead of answering).
    """
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return sp.CompletedProcess(
            cmd, 0, stdout='{"choice":"A","confidence":1,"reasoning":"x"}', stderr=""
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)

    # No agent (residual Qwen seat): no --agent token at all.
    sr.invoke_opencode("P", gdir, [], "openrouter/qwen/qwen3-vl-235b-a22b-instruct", timeout=99)
    assert "--agent" not in captured["cmd"]

    # agent="vote" (Kimi / Muse path): --agent vote is present.
    sr.invoke_opencode("P", gdir, [], "meta/muse-spark-1.1", timeout=99, agent="vote")
    cmd = captured["cmd"]
    assert "--agent" in cmd and cmd[cmd.index("--agent") + 1] == "vote"


def test_invoke_opencode_isolates_db_per_invocation(monkeypatch, tmp_path):
    """Each opencode invocation runs against its OWN sqlite DB (OPENCODE_DB) so
    concurrent kimi+muse votes don't contend on the shared ~/.opencode/opencode.db
    ("database is locked" retries). The override is ADDED to a copy of the ambient
    environment (the subprocess still needs META_API_KEY etc.), and the per-invocation
    temp DB dir is cleaned up afterward.
    """
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")

    # A sentinel ambient var must survive into the child env untouched.
    monkeypatch.setenv("META_API_KEY", "sentinel-key")

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return sp.CompletedProcess(
            cmd, 0, stdout='{"choice":"A","confidence":1,"reasoning":"x"}', stderr=""
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)

    sr.invoke_opencode("P", gdir, [], "openrouter/moonshotai/kimi-k2.6", timeout=99)

    env = captured["env"]
    assert env is not None, "invoke_opencode must pass an explicit env"
    # A per-invocation OPENCODE_DB override is present and points at a real path.
    assert "OPENCODE_DB" in env
    db_path = Path(env["OPENCODE_DB"])
    assert db_path.name == "opencode.db"
    # The ambient environment is preserved (copied, not replaced).
    assert env["META_API_KEY"] == "sentinel-key"
    # The per-invocation temp DB dir is cleaned up once the call returns.
    assert not db_path.parent.exists()


def test_invoke_opencode_injects_explicit_config_content(monkeypatch, tmp_path):
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return sp.CompletedProcess(
            cmd,
            0,
            stdout='{"choice":"A","confidence":0.9,"reasoning":"ok"}',
            stderr="",
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    explicit = {"model": "openrouter/google/gemini-3.5-flash"}
    sr.invoke_opencode("P", gdir, [], "openrouter/google/gemini-3.5-flash", config_content=explicit)
    assert json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"]) == explicit


def test_invoke_opencode_cleans_db_dir_on_timeout(monkeypatch, tmp_path):
    """The per-invocation OPENCODE_DB tmpdir is removed even when the subprocess
    raises TimeoutExpired (the CLI-timeout path _attempt_provider turns into an
    abstain) — the rmtree in the finally must not leak a temp dir per timeout.
    """
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        # Record the tmpdir the invoker created, then raise as a real timeout would.
        captured["db_path"] = Path(kwargs["env"]["OPENCODE_DB"])
        raise sp.TimeoutExpired(cmd, timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(sr.subprocess, "run", fake_run)

    with pytest.raises(sp.TimeoutExpired):
        sr.invoke_opencode("P", gdir, [], "openrouter/moonshotai/kimi-k2.6", timeout=1)

    db_path = captured["db_path"]
    # The exception propagated (the caller classifies it), but the finally still
    # ran: no leaked temp DB dir.
    assert not db_path.parent.exists()


def test_run_provider_on_group_threads_opencode_agent(monkeypatch, tmp_path):
    """run_provider_on_group forwards a spec's ``opencode_agent`` to the invoker
    as ``--agent`` (Kimi and Muse both run under the tool-less ``vote`` agent),
    and forwards NOTHING when it is unset (the residual Qwen seat keeps the plain
    call).

    Kimi and Muse dispatch under their distinct "kimi"/"muse" provider names yet
    the ``--agent`` threading — keyed on the RESOLVED invoker (``invoker is
    invoke_opencode``), not the name — still forwards their tool-less ``vote``
    agent. Driven at the SUBPROCESS boundary (not by swapping ``_INVOKERS``) so the
    real invoke_opencode runs and the guard's invoker-identity check is genuinely
    exercised.
    """
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")

    cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        cmds.append(cmd)
        return sp.CompletedProcess(
            cmd, 0, stdout='{"choice":"A","confidence":0.9,"reasoning":"ok"}', stderr=""
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)

    # MUSE (name="muse") dispatches to invoke_opencode and gets --agent vote.
    v = sr.run_provider_on_group(sr.MUSE, "g", gdir, "p", ["A"], {"A": [(R1, T1)]}, timeout=None)
    assert v.choice == "A"
    assert "--agent" in cmds[-1] and cmds[-1][cmds[-1].index("--agent") + 1] == "vote"

    # OPENCODE_KIMI (name="kimi") also carries opencode_agent="vote" -> --agent vote.
    v = sr.run_provider_on_group(
        sr.OPENCODE_KIMI, "g", gdir, "p", ["A"], {"A": [(R1, T1)]}, timeout=None
    )
    assert v.choice == "A"
    assert "--agent" in cmds[-1] and cmds[-1][cmds[-1].index("--agent") + 1] == "vote"

    # The residual Qwen seat (OPENCODE_QWEN, opencode_agent=None) -> no --agent token.
    v = sr.run_provider_on_group(
        sr.OPENCODE_QWEN, "g", gdir, "p", ["A"], {"A": [(R1, T1)]}, timeout=None
    )
    assert v.choice == "A"
    assert "--agent" not in cmds[-1]


def test_meta_candidate_cli_preserves_agent_and_muse_model_override(monkeypatch, tmp_path):
    """The CLI's per-provider override rebuild must carry ``opencode_agent``.

    ``--panel meta-candidate`` resolves to the tool-less Muse voter (name "muse")
    through the CLI, and an explicit ``--muse-model`` override changes the model
    WITHOUT dropping the vote agent (a naive ProviderSpec rebuild would silently
    revert Muse to the agentic ``build`` default). Post-rename, ``--muse-model``
    (not ``--opencode-model``) is the override that targets this seat.
    """
    from typer.testing import CliRunner

    from crosswalk.cli import app

    batch = tmp_path / "b"
    batch.mkdir()
    captured: dict = {}

    def fake_run_batch(batch_dir, panel, **kwargs):
        captured["panel"] = panel
        cons = pd.DataFrame({"consensus": ["unanimous"], "routing": ["auto"]})
        return pd.DataFrame({"x": [1]}), cons

    monkeypatch.setattr(sr, "run_batch", fake_run_batch)
    runner = CliRunner()

    # No override: the rebuild preserves Muse's tool-less agent + long timeout.
    r = runner.invoke(app, ["agent", "stitch-run", "-b", str(batch), "--panel", "meta-candidate"])
    assert r.exit_code == 0, r.output
    p = captured["panel"]
    assert [x.name for x in p] == ["claude", "codex", "muse"]
    assert p[2].model == "meta/muse-spark-1.1"
    assert p[2].opencode_agent == "vote" and p[2].timeout == 480

    # --muse-model overrides the model; the vote agent survives the rebuild.
    r = runner.invoke(
        app,
        [
            "agent",
            "stitch-run",
            "-b",
            str(batch),
            "--panel",
            "meta-candidate",
            "--muse-model",
            "meta/muse-spark-1.2",
        ],
    )
    assert r.exit_code == 0, r.output
    p = captured["panel"]
    assert p[2].model == "meta/muse-spark-1.2"
    assert p[2].opencode_agent == "vote"

    # --opencode-model targets only the residual "opencode"/Qwen seat (v3-era),
    # which meta-candidate does not have -> a NO-OP here (Muse untouched).
    r = runner.invoke(
        app,
        [
            "agent",
            "stitch-run",
            "-b",
            str(batch),
            "--panel",
            "meta-candidate",
            "--opencode-model",
            "openrouter/should/not-apply",
        ],
    )
    assert r.exit_code == 0, r.output
    p = captured["panel"]
    assert p[2].model == "meta/muse-spark-1.1"  # muse untouched by --opencode-model


def test_quad_candidate_cli_model_overrides_target_distinct_seats(monkeypatch, tmp_path):
    """On the 4-seat quad panel, --kimi-model hits ONLY the Kimi seat and
    --muse-model hits ONLY the Muse seat — the whole reason Kimi and Muse carry
    distinct provider names ("kimi"/"muse") on the shared opencode transport. A
    single override must not ambiguously rewrite both.
    """
    from typer.testing import CliRunner

    from crosswalk.cli import app

    batch = tmp_path / "b"
    batch.mkdir()
    captured: dict = {}

    def fake_run_batch(batch_dir, panel, **kwargs):
        captured["panel"] = panel
        cons = pd.DataFrame({"consensus": ["unanimous"], "routing": ["auto"]})
        return pd.DataFrame({"x": [1]}), cons

    monkeypatch.setattr(sr, "run_batch", fake_run_batch)
    runner = CliRunner()

    r = runner.invoke(
        app,
        [
            "agent",
            "stitch-run",
            "-b",
            str(batch),
            "--panel",
            "quad-candidate",
            "--kimi-model",
            "openrouter/moonshotai/kimi-k2.7",
            "--muse-model",
            "meta/muse-spark-1.2",
        ],
    )
    assert r.exit_code == 0, r.output
    p = captured["panel"]
    assert [x.name for x in p] == ["claude", "codex", "kimi", "muse"]
    # --kimi-model rewrote ONLY the Kimi seat; Muse kept its own model. The Kimi
    # seat's tool-less vote agent survives the override rebuild.
    assert p[2].name == "kimi" and p[2].model == "openrouter/moonshotai/kimi-k2.7"
    assert p[2].opencode_agent == "vote"
    # --muse-model rewrote ONLY the Muse seat; its tool-less vote agent survives.
    assert p[3].name == "muse" and p[3].model == "meta/muse-spark-1.2"
    assert p[3].opencode_agent == "vote"


def test_resolve_timeout_precedence():
    """Explicit caller/CLI timeout > per-spec timeout > global default."""
    kimi = sr.ProviderSpec(name="kimi", model="m", timeout=480)
    plain = sr.ProviderSpec(name="claude", model="m")
    # Per-spec beats the global default when nothing explicit is passed.
    assert sr.resolve_timeout(kimi, None) == 480
    # No spec timeout -> global default.
    assert sr.resolve_timeout(plain, None) == sr.DEFAULT_VOTE_TIMEOUT_S == 240
    # An explicit value beats BOTH (even when smaller than the spec's).
    assert sr.resolve_timeout(kimi, 300) == 300
    assert sr.resolve_timeout(plain, 600) == 600


def test_run_provider_on_group_resolves_per_spec_timeout(monkeypatch):
    """run_provider_on_group hands the RESOLVED timeout to the invoker: the
    spec's own timeout when the caller passes None, the caller's value when
    explicit."""
    seen: list[int] = []

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        seen.append(timeout)
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    monkeypatch.setitem(sr._INVOKERS, "opencode", fake_invoker)
    spec = sr.ProviderSpec(name="opencode", model="m", timeout=480)

    v = sr.run_provider_on_group(spec, "g1", None, "p", ["A"], {"A": []}, timeout=None)
    assert v.choice == "A" and seen[-1] == 480

    v = sr.run_provider_on_group(spec, "g1", None, "p", ["A"], {"A": []}, timeout=120)
    assert v.choice == "A" and seen[-1] == 120

    plain = sr.ProviderSpec(name="opencode", model="m")
    v = sr.run_provider_on_group(plain, "g1", None, "p", ["A"], {"A": []}, timeout=None)
    assert v.choice == "A" and seen[-1] == sr.DEFAULT_VOTE_TIMEOUT_S


def test_invoke_opencode_arg_construction(monkeypatch, tmp_path):
    """Prompt goes via stdin (not argv); -m model and one -f per attached image.

    The prompt used to be a positional arg, but a large group's prompt exceeds
    the OS single-arg limit (E2BIG), so it is now piped via stdin.
    """
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")
    (gdir / "option_A.png").write_bytes(b"\x89PNG")
    (gdir / "option_B.png").write_bytes(b"\x89PNG")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return sp.CompletedProcess(
            cmd, 0, stdout='{"choice":"A","confidence":1,"reasoning":"x"}', stderr=""
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    out = sr.invoke_opencode("PROMPT TEXT", gdir, ["A", "B"], "openrouter/qwen/x", timeout=99)

    cmd = captured["cmd"]
    assert cmd[:2] == ["opencode", "run"]
    # The prompt is piped via stdin, never placed on argv.
    assert captured["kwargs"]["input"] == "PROMPT TEXT"
    assert "PROMPT TEXT" not in cmd
    # -m model appears.
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "openrouter/qwen/x"
    # One -f per image: overview + option_A + option_B = 3.
    assert cmd.count("-f") == 3
    f_args = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-f"]
    assert f_args[0].endswith("overview.png")
    assert any(a.endswith("option_A.png") for a in f_args)
    assert any(a.endswith("option_B.png") for a in f_args)
    assert captured["kwargs"]["timeout"] == 99
    assert '"choice":"A"' in out


def test_invoke_opencode_raises_on_nonzero_exit(monkeypatch, tmp_path):
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")

    def fake_run(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, stdout="", stderr="quota exceeded")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        sr.invoke_opencode("P", gdir, [], "m")
    assert "opencode exited with code 1" in str(exc.value)
    assert "quota exceeded" in str(exc.value)


def test_opencode_vote_parses_through_runner(monkeypatch):
    """A valid opencode-style JSON answer parses to a real vote."""

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "B", "confidence": 0.66, "reasoning": "qwen picks B"}'

    monkeypatch.setitem(sr._INVOKERS, "opencode", fake_invoker)
    vote = sr.run_provider_on_group(
        sr.OPENCODE_QWEN, "g", None, "prompt", ["A", "B"], {"B": [(R1, T1)]}
    )
    assert vote.provider == "opencode"
    assert vote.choice == "B"
    assert vote.edge_set == frozenset({(R1, T1)})


def test_opencode_hard_fails_on_invocation_error(monkeypatch):
    """opencode quota exhaustion halts the run (was: abstain). budget=0 = immediate."""

    def failing(prompt, group_dir, letters, model, timeout, effort=""):
        raise RuntimeError("opencode exited with code 1: quota exceeded")

    monkeypatch.setitem(sr._INVOKERS, "opencode", failing)
    with pytest.raises(sr.ProviderInvocationError, match="quota exceeded"):
        sr.run_provider_on_group(
            sr.OPENCODE_QWEN,
            "g",
            None,
            "prompt",
            ["A"],
            {"A": [(R1, T1)]},
            invocation_budget_s=0.0,
        )


def test_four_voter_consensus_tiers():
    """4-voter tiers: 4/4 is unanimous; 3/4 with a live dissent is majority
    (human review); 3 agree + 1 abstain is a QUORUM accept (v5 rule)."""
    es = frozenset({(R1, T1)})
    four_agree = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("agy", "A", es),
        _vote("opencode", "A", es),
    ]
    c = sr.compute_consensus(four_agree)
    assert c.consensus == "unanimous"
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"
    # 3/4 with a lone dissenter is a majority -> human review (never auto-accept:
    # quorum forgives abstention only, never disagreement).
    three_one = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("agy", "A", es),
        _vote("opencode", "B"),
    ]
    c2 = sr.compute_consensus(three_one)
    assert c2.consensus == "majority"
    assert c2.routing == "human_review"
    assert "opencode=B" in c2.minority
    assert c2.route_reason == "dissent:opencode=B"
    # 3 agree + 1 abstain: all valid votes agree at quorum (>=3 valid) — the
    # v5 quorum rule auto-accepts under the DISTINCT "quorum" tier/reason
    # (pre-v5 this was blocked as "abstention").
    three_abstain = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("agy", "A", es),
        _vote("opencode", "ABSTAIN"),
    ]
    c3 = sr.compute_consensus(three_abstain)
    assert c3.consensus == "quorum"
    assert c3.routing == "auto_accept"
    assert c3.route_reason == "quorum"


# ---------------------------------------------------------------------------
# Resumable per-group batch driver
# ---------------------------------------------------------------------------


def _write_min_pack(batch_dir, gid):
    """Write a minimal evidence pack (metadata + prompt) for run_batch."""
    import yaml

    g = make_group()
    g["group_id"] = gid
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    d = batch_dir / gid
    d.mkdir(parents=True)
    (d / "metadata.yaml").write_text(yaml.safe_dump(meta))
    (d / "overview.png").write_bytes(b"minimal-overview")
    (d / "prompt.txt").write_text(
        f'- overview: {d.resolve() / "overview.png"}\nrespond {{"choice": "A"}}'
    )
    return d


def test_scratch_pack_ignores_unmanaged_png(tmp_path):
    group_dir = tmp_path / "g1"
    group_dir.mkdir()
    (group_dir / "overview.png").write_bytes(b"overview")
    (group_dir / "option_A.png").write_bytes(b"option")
    (group_dir / "other.png").write_bytes(b"unmanaged")
    external = tmp_path / "external images" / "context.png"
    external.parent.mkdir()
    external.write_bytes(b"external")

    scratch, rewritten, temp_dir = sr._scratch_pack(group_dir, f"    image: {external}")
    try:
        assert (scratch / "overview.png").exists()
        assert (scratch / "option_A.png").exists()
        assert not (scratch / "other.png").exists()
        assert not (scratch / "context.png").exists()
        assert str(external) in rewritten
    finally:
        temp_dir.cleanup()


def test_scratch_pack_relocates_stale_absolute_image_paths_after_batch_rename(tmp_path):
    group_dir = tmp_path / "renamed_batch" / "g1"
    group_dir.mkdir(parents=True)
    for name in ("overview.png", "option_A.png", "zoom_R1_T1.png"):
        (group_dir / name).write_bytes(name.encode())

    stale = tmp_path / "old_batch" / "g1"
    prompt = (
        f"- overview: {stale / 'overview.png'}\n"
        f"      junction zoom: {stale / 'zoom_R1_T1.png'}\n"
        f"    image: {stale / 'option_A.png'}\n"
        "Look at overview.png first."
    )

    scratch, rewritten, temp_dir = sr._scratch_pack(group_dir, prompt)
    try:
        assert str(stale) not in rewritten
        for name in ("overview.png", "option_A.png", "zoom_R1_T1.png"):
            relocated = scratch / name
            assert relocated.is_file()
            assert str(relocated) in rewritten
    finally:
        temp_dir.cleanup()


def test_missing_prompt_png_fails_before_provider_invocation(monkeypatch, tmp_path):
    group_dir = tmp_path / "renamed_batch" / "g1"
    group_dir.mkdir(parents=True)
    (group_dir / "overview.png").write_bytes(b"overview")
    stale = tmp_path / "old_batch" / "g1"
    prompt = f"- overview: {stale / 'overview.png'}\n    image: {stale / 'option_A.png'}"
    invoked = False

    def fake_invoker(*args, **kwargs):
        nonlocal invoked
        invoked = True
        return '{"choice":"A","confidence":1,"reasoning":"ok"}'

    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    with pytest.raises(ValueError, match=r"unresolved local PNG.*option_A\.png"):
        sr.run_provider_on_group(
            sr.ProviderSpec("claude", "m"),
            "g1",
            group_dir,
            prompt,
            ["A"],
            {"A": []},
        )
    assert not invoked


def test_manifest_hash_mismatch_fails_before_provider_invocation(monkeypatch, tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group_dir = _write_min_pack(batch_dir, "g1")
    manifest = load_evidence_manifest(group_dir)
    prompt = (group_dir / "prompt.txt").read_text()
    (group_dir / "overview.png").write_bytes(b"changed-after-manifest-verification")
    invoked = False

    def fake_invoker(*args, **kwargs):
        nonlocal invoked
        invoked = True
        return '{"choice":"A","confidence":1,"reasoning":"ok"}'

    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    with pytest.raises(EvidenceProvenanceError, match="scratch image bytes/count/set"):
        sr.run_provider_on_group(
            sr.ProviderSpec("claude", "m"),
            "g1",
            group_dir,
            prompt,
            ["A"],
            {"A": []},
            evidence_manifest=manifest,
        )
    assert not invoked


def test_attempt_preflight_failure_cleans_invocation_scratch(monkeypatch, tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group_dir = _write_min_pack(batch_dir, "g1")
    manifest = load_evidence_manifest(group_dir)
    scratch_path = None
    invoked = False

    def failing_preflight(scratch, letters, evidence_manifest):
        nonlocal scratch_path
        scratch_path = Path(scratch)
        raise EvidenceProvenanceError("forced native attachment mismatch")

    def fake_invoker(*args, **kwargs):
        nonlocal invoked
        invoked = True
        return '{"choice":"A","confidence":1,"reasoning":"ok"}'

    monkeypatch.setattr(sr, "_preflight_native_attachment_assets", failing_preflight)
    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    with pytest.raises(EvidenceProvenanceError, match="forced native attachment mismatch"):
        sr.run_provider_on_group(
            sr.ProviderSpec("claude", "m"),
            "g1",
            group_dir,
            (group_dir / "prompt.txt").read_text(),
            ["A"],
            {"A": []},
            evidence_manifest=manifest,
        )

    assert scratch_path is not None
    assert not scratch_path.exists()
    assert not invoked


def test_parse_retry_rejects_images_mutated_by_prior_attempt(monkeypatch, tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group_dir = _write_min_pack(batch_dir, "g1")
    manifest = load_evidence_manifest(group_dir)
    calls = 0

    def mutating_invalid_vote(prompt, scratch, letters, model, timeout, effort=""):
        nonlocal calls
        calls += 1
        (Path(scratch) / "overview.png").write_bytes(b"mutated during parse attempt")
        return '{"choice":"NOT_AN_OPTION","confidence":1,"reasoning":"bad"}'

    monkeypatch.setitem(sr._INVOKERS, "agy", mutating_invalid_vote)
    with pytest.raises(EvidenceProvenanceError, match="native attachment.*manifest"):
        sr.run_provider_on_group(
            sr.ProviderSpec("agy", "m"),
            "g1",
            group_dir,
            (group_dir / "prompt.txt").read_text(),
            ["A"],
            {"A": []},
            retries=1,
            evidence_manifest=manifest,
        )

    assert calls == 1


def test_abstain_records_delivery_for_actual_gemini_route(monkeypatch, tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group_dir = _write_min_pack(batch_dir, "g1")
    manifest = load_evidence_manifest(group_dir)

    def routed_timeout(*args, **kwargs):
        raise sr.GroupScopedProviderError(
            "flex timeout",
            kind=sr.AbstainReason.TIMEOUT,
            invocation_route=sr.GEMINI_ROUTE_OPENROUTER_FLEX,
        )

    monkeypatch.setitem(sr._INVOKERS, "gemini", routed_timeout)
    vote = sr.run_provider_on_group(
        sr.GEMINI,
        "g1",
        group_dir,
        (group_dir / "prompt.txt").read_text(),
        ["A"],
        {"A": []},
        evidence_manifest=manifest,
    )

    assert vote.choice == "ABSTAIN"
    assert vote.invocation_route == sr.GEMINI_ROUTE_OPENROUTER_FLEX
    record = sr.validate_evidence_delivery_record(
        vote.evidence_delivery,
        manifest,
        expected_delivery_mode=sr.DELIVERY_MODE_NATIVE_ATTACHMENT,
        expected_transport="opencode:-f",
    )
    assert record["preflight_status"] == "passed"


def test_run_batch_rejects_pack_outside_schema_v2_roster(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "stale")
    (batch_dir / "batch.json").write_text(
        json.dumps({"schema_version": 2, "groups": [{"group_id": "current"}]})
    )

    with pytest.raises(ValueError, match="outside the current schema-v2 batch roster"):
        sr.run_batch(batch_dir, panel=[])


def test_run_batch_binds_schema_v2_pack_to_exact_group_payload(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group = make_group()
    group["group_id"] = "g1"
    source_artifacts = {"groups_sidecar": {"available": True, "sha256": "a" * 64}}
    generation_source = {"source_commit": {"commit": "abc", "dirty": False}}
    generate_group_evidence(
        group,
        batch_dir / "g1",
        source_artifacts=source_artifacts,
        batch_generation_source=generation_source,
    )
    changed_group = copy.deepcopy(group)
    changed_group["edges"][0]["confidence"] = 0.01
    (batch_dir / "batch.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_artifacts": source_artifacts,
                "batch_generation_source": generation_source,
                "groups": [changed_group],
            }
        )
    )

    with pytest.raises(EvidenceProvenanceError, match="source group does not match"):
        sr.run_batch(batch_dir, panel=[])


def test_run_batch_resume_skips_completed_groups(tmp_path, monkeypatch):
    """resume=True reloads partials and does NOT re-invoke completed groups."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    _write_min_pack(batch_dir, "g2")

    # Each group runs in an isolated scratch copy, so the invoker sees a scratch
    # dir, not the group name — count invocations (3 providers per group) instead.
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)

    panel = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]

    # First run does g1 only, writing partials (3 invocations: one per provider).
    sr.run_batch(batch_dir, panel=panel, group_ids=["g1"])
    assert (batch_dir / "votes.partial.csv").exists()
    assert calls["n"] == 3

    calls["n"] = 0
    # Resume over both groups: g1 already recorded -> only g2 runs (3 more calls).
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel, resume=True)
    assert calls["n"] == 3  # g1 skipped; only g2 invoked
    # Final output carries BOTH groups.
    assert set(cons_df["group_id"]) == {"g1", "g2"}
    assert set(votes_df["group_id"]) == {"g1", "g2"}
    assert votes_df["evidence_delivery"].astype(str).str.len().gt(0).all()
    expected = {
        "claude": (sr.DELIVERY_MODE_PROMPT_PATH, "claude:Read"),
        "codex": (sr.DELIVERY_MODE_NATIVE_ATTACHMENT, "codex:-i"),
        "agy": (sr.DELIVERY_MODE_PROMPT_PATH, "agy:agent-read"),
    }
    for row in votes_df.to_dict("records"):
        mode, transport = expected[row["provider"]]
        manifest = load_evidence_manifest(batch_dir / row["group_id"])
        sr.validate_evidence_delivery_record(
            row["evidence_delivery"],
            manifest,
            expected_delivery_mode=mode,
            expected_transport=transport,
        )


@pytest.mark.parametrize("mutation", ["missing", "tampered", "mode_mismatch"])
def test_run_batch_resume_rejects_invalid_delivery_provenance(
    tmp_path,
    monkeypatch,
    mutation,
):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    panel = [sr.ProviderSpec(name, "m") for name in ("claude", "codex", "agy")]
    sr.run_batch(batch_dir, panel=panel)
    assert calls["n"] == 3

    partial = pd.read_csv(batch_dir / "votes.partial.csv")
    if mutation == "missing":
        partial = partial.drop(columns="evidence_delivery")
    else:
        row_index = partial.index[partial["provider"] == "claude"][0]
        record = json.loads(partial.loc[row_index, "evidence_delivery"])
        if mutation == "tampered":
            record["asset_count"] += 1
        else:
            record["delivery_mode"] = sr.DELIVERY_MODE_NATIVE_ATTACHMENT
            record["transport"] = "codex:-i"
        partial.loc[row_index, "evidence_delivery"] = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
    partial.to_csv(batch_dir / "votes.partial.csv", index=False)

    calls["n"] = 0
    repaired, _ = sr.run_batch(batch_dir, panel=panel, resume=True)

    assert calls["n"] == 3
    assert repaired["evidence_delivery"].astype(str).str.len().gt(0).all()


def test_run_batch_resume_requires_valid_gemini_route_provenance(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_gemini(*args, **kwargs):
        calls["n"] += 1
        return sr.InvocationResult(
            '{"choice":"A","confidence":0.9,"reasoning":"ok"}',
            sr.GEMINI_ROUTE_OPENROUTER_FLEX,
        )

    monkeypatch.setitem(sr._INVOKERS, "gemini", fake_gemini)
    panel = [sr.GEMINI]
    sr.run_batch(batch_dir, panel=panel)
    assert calls["n"] == 1

    calls["n"] = 0
    resumed, _ = sr.run_batch(batch_dir, panel=panel, resume=True)
    assert calls["n"] == 0
    assert set(resumed["invocation_route"]) == {sr.GEMINI_ROUTE_OPENROUTER_FLEX}
    delivery = json.loads(resumed.iloc[0]["evidence_delivery"])
    assert delivery["delivery_mode"] == sr.DELIVERY_MODE_NATIVE_ATTACHMENT
    assert delivery["transport"] == "opencode:-f"

    partial = pd.read_csv(batch_dir / "votes.partial.csv")
    partial["invocation_route"] = "unknown-route"
    partial.to_csv(batch_dir / "votes.partial.csv", index=False)
    calls["n"] = 0
    repaired, _ = sr.run_batch(batch_dir, panel=panel, resume=True)
    assert calls["n"] == 1
    assert set(repaired["invocation_route"]) == {sr.GEMINI_ROUTE_OPENROUTER_FLEX}


def test_run_batch_resume_requires_route_allowed_by_current_gemini_policy(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_gemini(*args, **kwargs):
        calls["n"] += 1
        return sr.InvocationResult(
            '{"choice":"A","confidence":0.9,"reasoning":"ok"}',
            sr.GEMINI_ROUTE_AGY,
        )

    monkeypatch.setitem(sr._INVOKERS, "gemini", fake_gemini)
    panel = [sr.GEMINI_AGY_ONLY]
    sr.run_batch(batch_dir, panel=panel)
    assert calls["n"] == 1
    recorded = pd.read_csv(batch_dir / "votes.csv").iloc[0]
    delivery = json.loads(recorded["evidence_delivery"])
    assert delivery["delivery_mode"] == sr.DELIVERY_MODE_PROMPT_PATH
    assert delivery["transport"] == "agy:agent-read"

    partial = pd.read_csv(batch_dir / "votes.partial.csv")
    partial["invocation_route"] = sr.GEMINI_ROUTE_OPENROUTER_FLEX
    partial.to_csv(batch_dir / "votes.partial.csv", index=False)
    calls["n"] = 0

    repaired, _ = sr.run_batch(batch_dir, panel=panel, resume=True)
    assert calls["n"] == 1
    assert set(repaired["invocation_route"]) == {sr.GEMINI_ROUTE_AGY}


def test_run_batch_size_gates_over_backstop_group(tmp_path, monkeypatch):
    """A unanimous vote on an over-backstop pack routes human_review/size_gated.

    This closes the routing void: before the gate, such a verdict was
    auto_accept (so never in the human queue) yet blocked at export by the
    backstop — it vanished, reviewed by no one.
    """
    import yaml

    from crosswalk.config import settings

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    d = _write_min_pack(batch_dir, "g1")
    # Inflate the pack's candidate-edge count over the export backstop.
    meta = yaml.safe_load((d / "metadata.yaml").read_text())
    meta["n_edges_full"] = settings.stitch_export_backstop_max_edges + 1
    (d / "metadata.yaml").write_text(yaml.safe_dump(meta))

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    panel = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]

    _votes_df, cons_df = sr.run_batch(batch_dir, panel=panel)
    row = cons_df.iloc[0]
    assert row["consensus"] == "unanimous"  # the vote outcome is preserved
    assert row["routing"] == "human_review"  # ...but it must reach a human
    assert row["route_reason"] == "size_gated"


def _breaker_panel_and_invokers(monkeypatch, failing_invoker):
    """3-provider panel where claude/codex vote A and agy uses failing_invoker."""

    def ok_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    monkeypatch.setitem(sr._INVOKERS, "claude", ok_invoker)
    monkeypatch.setitem(sr._INVOKERS, "codex", ok_invoker)
    monkeypatch.setitem(sr._INVOKERS, "agy", failing_invoker)
    return [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]


def test_run_batch_timeout_breaker_halts_after_consecutive(tmp_path, monkeypatch):
    """N consecutive timeout-abstains from ONE provider promote to provider-down halt.

    Completed groups must be flushed to the partials before the raise so the run
    stays resumable.
    """
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4"):
        _write_min_pack(batch_dir, gid)

    def always_timeout(prompt, group_dir, letters, model, timeout, effort=""):
        raise subprocess.TimeoutExpired(cmd="agy", timeout=timeout)

    panel = _breaker_panel_and_invokers(monkeypatch, always_timeout)
    with pytest.raises(sr.ProviderInvocationError, match="consecutive"):
        sr.run_batch(batch_dir, panel=panel)
    # Groups before the breaker group were flushed and are resumable.
    votes = pd.read_csv(batch_dir / "votes.partial.csv", dtype={"group_id": str})
    assert set(votes["group_id"]) == {"g1", "g2"}


def test_run_batch_cli_internal_timeout_abstains_trip_breaker(tmp_path, monkeypatch):
    """BUG 1 regression: agy's CLI-INTERNAL response timeout must trip the breaker.

    A network-blackholed agy fails every group with its OWN clean timeout — the
    CLI exits nonzero and _check_exit raises GroupScopedProviderError(kind=TIMEOUT)
    — NOT a subprocess SIGKILL. The breaker previously string-matched only
    "timeout after" (the subprocess flavor) and reset on this flavor, so it never
    tripped for the provider it was built for (the #334 silent-degradation mode).
    Now it counts AbstainReason.TIMEOUT and halts after N consecutive.
    """
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4"):
        _write_min_pack(batch_dir, gid)

    def always_cli_timeout(prompt, group_dir, letters, model, timeout, effort=""):
        # Same exception _check_exit raises for agy's "timeout waiting for response".
        raise sr.GroupScopedProviderError(
            "agy CLI-internal response timeout", kind=sr.AbstainReason.TIMEOUT
        )

    panel = _breaker_panel_and_invokers(monkeypatch, always_cli_timeout)
    with pytest.raises(sr.ProviderInvocationError, match="consecutive"):
        sr.run_batch(batch_dir, panel=panel)
    # Completed groups flushed before the halt -> resumable (breaker trips on g3).
    votes = pd.read_csv(batch_dir / "votes.partial.csv", dtype={"group_id": str})
    assert set(votes["group_id"]) == {"g1", "g2"}


def test_run_batch_cli_internal_timeout_breaker_resets_on_success(tmp_path, monkeypatch):
    """A success between CLI-internal timeouts resets the count — no false halt."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4", "g5"):
        _write_min_pack(batch_dir, gid)

    calls = {"n": 0}

    def intermittent(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        if calls["n"] == 3:  # succeed on the 3rd group only, breaking the streak
            return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'
        raise sr.GroupScopedProviderError(
            "agy CLI-internal response timeout", kind=sr.AbstainReason.TIMEOUT
        )

    panel = _breaker_panel_and_invokers(monkeypatch, intermittent)
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel)  # must NOT raise
    assert set(cons_df["group_id"]) == {"g1", "g2", "g3", "g4", "g5"}
    agy = votes_df[votes_df["provider"] == "agy"]
    assert (agy["choice"] == "ABSTAIN").sum() == 4


def test_run_batch_timeout_breaker_resets_on_success(tmp_path, monkeypatch):
    """A successful vote resets the consecutive-timeout count — no false halt."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4", "g5"):
        _write_min_pack(batch_dir, gid)

    calls = {"n": 0}

    def intermittent(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        if calls["n"] == 3:  # succeed on the 3rd group only
            return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'
        raise subprocess.TimeoutExpired(cmd="agy", timeout=timeout)

    panel = _breaker_panel_and_invokers(monkeypatch, intermittent)
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel)  # must NOT raise
    assert set(cons_df["group_id"]) == {"g1", "g2", "g3", "g4", "g5"}
    agy = votes_df[votes_df["provider"] == "agy"]
    assert (agy["choice"] == "ABSTAIN").sum() == 4


def test_empty_output_hard_fails_after_budget(monkeypatch):
    """Empty output from a zero-exit CLI is provider failure -> backoff, then halt.

    Observed live: agy at its daily quota cap returns exit 0 with empty
    stdout/stderr on EVERY call. Routing that through the parse path would
    silently degrade the panel on all remaining groups (the #334 failure mode,
    invisible to both the nonzero-exit halt and the timeout breaker).
    """
    _install_fake_clock(monkeypatch)

    def always_empty(*_a, **_k):
        return "   \n"

    with pytest.raises(sr.ProviderInvocationError, match="empty output"):
        _attempt(always_empty, budget=30.0)


def test_empty_output_recovers_on_retry(monkeypatch):
    """A transient empty response that clears on retry yields a normal vote."""
    _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def flaky_empty(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    vote = _attempt(flaky_empty, budget=60.0)
    assert vote.choice == "A"
    assert calls["n"] == 2


def test_no_agy_panel_swaps_in_opencode():
    """The quota-outage fallback panel is claude + codex + opencode (no agy)."""
    panel = sr.get_panel("no-agy")
    assert [p.name for p in panel] == ["claude", "codex", "opencode"]
    assert panel[2].model == sr.OPENCODE_QWEN.model


def test_run_batch_overflow_abstains_do_not_trip_breaker(tmp_path, monkeypatch):
    """Context overflow is a group property: monsters in a row never halt the run."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4"):
        _write_min_pack(batch_dir, gid)

    def always_overflow(prompt, group_dir, letters, model, timeout, effort=""):
        raise sr.GroupScopedProviderError(
            "agy context overflow: prompt exceeds the model's context window"
        )

    panel = _breaker_panel_and_invokers(monkeypatch, always_overflow)
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel)  # must NOT raise
    assert set(cons_df["group_id"]) == {"g1", "g2", "g3", "g4"}
    agy = votes_df[votes_df["provider"] == "agy"]
    assert (agy["choice"] == "ABSTAIN").all()


def test_quota_and_rate_limit_messages_not_misclassified_as_group_scoped():
    """Marker-set breadth guard: provider-down bodies must stay provider-down.

    Broadening _CONTEXT_OVERFLOW_MARKERS to something colliding with a realistic
    quota/rate-limit message would silently convert #334 halts into abstains.
    """
    provider_down_bodies = [
        "429 rate limit reached for gpt-5.5: limit 30000 tokens per min",
        "insufficient_quota: you exceeded your current quota",
        "401 unauthorized: invalid api key",
        "upstream connect error or disconnect/reset before headers",
        "billing hard limit has been reached",
    ]
    for body in provider_down_bodies:
        result = subprocess.CompletedProcess(args=["codex"], returncode=1, stdout="", stderr=body)
        with pytest.raises(RuntimeError) as exc_info:
            sr._check_exit("codex", result)
        assert not isinstance(exc_info.value, sr.GroupScopedProviderError), body


def test_run_batch_resume_rejects_mismatched_panel(tmp_path, monkeypatch):
    """Partials written by a DIFFERENT panel are ignored: every group re-runs.

    Guards the silent-cache trap: 3-voter partials satisfying a --panel
    v3-candidate --resume run would return cached 3-voter votes with the 4th
    voter never invoked.
    """
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")

    calls = {"n": 0, "providers": set()}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy", "opencode"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)

    panel3 = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]
    panel4 = [*panel3, sr.ProviderSpec("opencode", "m")]

    # Complete a 3-voter run: partials now record every group with 3 providers.
    sr.run_batch(batch_dir, panel=panel3)
    assert calls["n"] == 3

    calls["n"] = 0
    # Resume with the 4-voter panel: provider-set mismatch -> partials ignored,
    # g1 re-runs with all FOUR voters (not returned from the 3-voter cache).
    votes_df, _cons_df = sr.run_batch(batch_dir, panel=panel4, resume=True)
    assert calls["n"] == 4
    assert set(votes_df["provider"]) == {"claude", "codex", "agy", "opencode"}


def test_run_batch_resume_rejects_changed_model_with_same_provider_names(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    first = [sr.ProviderSpec(name, "model-v1") for name in ("claude", "codex", "agy")]
    changed = [sr.ProviderSpec(name, "model-v2") for name in ("claude", "codex", "agy")]
    sr.run_batch(batch_dir, panel=first)

    calls["n"] = 0
    votes, _ = sr.run_batch(batch_dir, panel=changed, resume=True)

    assert calls["n"] == 3
    assert set(votes["model"]) == {"model-v2"}


def test_run_batch_resume_rejects_regenerated_menu_for_same_group_id(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    group = make_group()
    group["group_id"] = "g1"
    group_dir = batch_dir / "g1"
    generate_group_evidence(group, group_dir)
    batch_payload = {
        "schema_version": 2,
        "dataset_id": "test",
        "source_artifacts": {"status": "unavailable"},
        "batch_generation_source": {"status": "unavailable"},
        "groups": [group],
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch_payload))
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    panel = [sr.ProviderSpec(name, "m") for name in ("claude", "codex", "agy")]
    first_votes, _ = sr.run_batch(batch_dir, panel=panel)
    old_evidence = first_votes.iloc[0]["evidence_id"]

    # Same group id, but a different selectable menu and newly hashed pack.
    group["alternatives"] = []
    generate_group_evidence(group, group_dir)
    batch_payload["groups"] = [group]
    (batch_dir / "batch.json").write_text(json.dumps(batch_payload))
    calls["n"] = 0
    votes, _ = sr.run_batch(batch_dir, panel=panel, resume=True)

    assert calls["n"] == 3
    assert votes.iloc[0]["evidence_id"] != old_evidence


def test_run_batch_resume_rejects_choice_edge_set_mismatch(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    panel = [sr.ProviderSpec(name, "m") for name in ("claude", "codex", "agy")]
    sr.run_batch(batch_dir, panel=panel)

    consensus = pd.read_csv(batch_dir / "consensus.partial.csv")
    consensus.loc[0, "edge_set"] = json.dumps([[R1, T2]])
    consensus.to_csv(batch_dir / "consensus.partial.csv", index=False)
    calls["n"] = 0

    sr.run_batch(batch_dir, panel=panel, resume=True)

    assert calls["n"] == 3


def test_run_batch_resume_rejects_incomplete_per_group_ballots(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    panel = [sr.ProviderSpec(name, "m") for name in ("claude", "codex", "agy")]
    sr.run_batch(batch_dir, panel=panel)
    partial = pd.read_csv(batch_dir / "votes.partial.csv", dtype={"group_id": str})
    partial[partial["provider"] != "agy"].to_csv(batch_dir / "votes.partial.csv", index=False)

    calls["n"] = 0
    votes, _ = sr.run_batch(batch_dir, panel=panel, resume=True)

    assert calls["n"] == 3
    assert set(votes["provider"]) == {"claude", "codex", "agy"}


def test_run_batch_resume_rejects_changed_consensus_policy(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)
    panel = [sr.ProviderSpec(name, "m") for name in ("claude", "codex", "agy")]
    sr.run_batch(batch_dir, panel=panel)
    monkeypatch.setattr(
        sr.settings,
        "stitch_min_voter_confidence",
        sr.settings.stitch_min_voter_confidence + 0.01,
    )

    calls["n"] = 0
    sr.run_batch(batch_dir, panel=panel, resume=True)

    assert calls["n"] == 3


def test_run_batch_resume_respects_group_selection(tmp_path, monkeypatch):
    """A filtered resume must not leak previously-done, unrequested groups."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    _write_min_pack(batch_dir, "g2")

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)

    panel = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]

    # Run everything once (partials record g1 + g2).
    sr.run_batch(batch_dir, panel=panel)
    # Filtered resume asking ONLY for g2: g1 must not appear in the output.
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel, group_ids=["g2"], resume=True)
    assert set(votes_df["group_id"]) == {"g2"}
    assert set(cons_df["group_id"]) == {"g2"}


def test_run_provider_collect_feedback_toggle(monkeypatch):
    """collect_feedback=True captures the self-report; False leaves it empty."""
    raw = json.dumps(
        {
            "choice": "A",
            "confidence": 0.9,
            "reasoning": "ok",
            "pack_feedback": {"missing_info": ["x"], "ambiguities": [], "confidence_basis": "y"},
        }
    )

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return raw

    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    args = (sr.ProviderSpec("claude", "m"), "g", None, "prompt", ["A"], {"A": [(R1, T1)]})

    vote_off = sr.run_provider_on_group(*args)
    assert vote_off.choice == "A"
    assert vote_off.pack_feedback == ""

    vote_on = sr.run_provider_on_group(*args, collect_feedback=True)
    assert vote_on.choice == "A"
    assert json.loads(vote_on.pack_feedback)["missing_info"] == ["x"]


# --- invocation hard-fail vs parse abstain (quota handling) -------------------


def _attempt(invoker, retries=1, budget=300.0):
    """Drive _attempt_provider with a fake invoker, group_dir=None (unit mode)."""
    return sr._attempt_provider(
        sr.ProviderSpec(name="claude", model="test-model"),
        "g1",
        None,  # group_dir -> skip scratch pack
        "prompt",
        ["A"],
        {"A": [("r1", "t1")]},
        {"A"},
        invoker,
        5,  # timeout
        retries,
        invocation_budget_s=budget,
    )


class _FakeClock:
    """Deterministic monotonic clock: only sleep() advances time.

    Lets the backoff/deadline logic run through many iterations with zero real
    wall-clock, so budget exhaustion and exponential backoff are actually
    exercised — a no-op sleep would freeze ``remaining`` and never exhaust the
    budget, leaving the core mechanism untested.
    """

    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _install_fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(sr.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sr.time, "sleep", clock.sleep)
    return clock


def test_persistent_invocation_error_hard_fails(monkeypatch):
    """A provider that keeps failing (e.g. quota) raises, not abstains."""
    _install_fake_clock(monkeypatch)

    def always_fail(*_a, **_k):
        raise RuntimeError("opencode exited with code 1: 429 insufficient_quota")

    with pytest.raises(sr.ProviderInvocationError, match="insufficient_quota"):
        _attempt(always_fail, budget=0.0)


def test_budget_exhaustion_backs_off_exponentially(monkeypatch):
    """Persistent failure retries with doubling backoff until the budget is spent."""
    clock = _install_fake_clock(monkeypatch)

    def always_fail(*_a, **_k):
        raise RuntimeError("exited with code 1: network unreachable")

    with pytest.raises(sr.ProviderInvocationError):
        _attempt(always_fail, budget=300.0)
    # Exponential backoff (5,10,20,40,...), capped at 60, final sleep clamped to
    # the remaining budget; the clock only advances via sleep, so the sleeps sum
    # to exactly the budget.
    assert clock.sleeps[:4] == [5.0, 10.0, 20.0, 40.0]
    assert max(clock.sleeps) == 60.0
    assert abs(sum(clock.sleeps) - 300.0) < 1e-6


def test_timeout_abstains_immediately_without_retry(monkeypatch):
    """A timeout is group-scoped: abstain on ONE attempt (no backoff), run continues.

    Retrying a timeout just burns another full timeout window, and halting lets
    one oversized group kill a whole wave; a hung *provider* still halts via the
    generic nonzero-exit path.
    """
    clock = _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def always_timeout(*_a, **_k):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd="claude", timeout=5)

    vote = _attempt(always_timeout, budget=300.0)
    assert vote.choice == "ABSTAIN"
    assert "timeout after" in vote.error
    assert vote.abstain_reason == sr.AbstainReason.TIMEOUT  # counts toward the breaker
    assert calls["n"] == 1  # single attempt, no retry
    assert clock.sleeps == []  # and no backoff sleeps


def test_cli_internal_timeout_abstains_and_is_breaker_countable(monkeypatch):
    """BUG 1: agy's CLI-internal response timeout abstains immediately AND is marked
    AbstainReason.TIMEOUT so the run_batch breaker counts it (not a reset).

    This is the path a network-blackholed agy takes: its own --print-timeout fires
    first, the CLI exits nonzero, and _check_exit raises GroupScopedProviderError
    with kind=TIMEOUT — distinct from a subprocess SIGKILL but the same fate.
    """
    clock = _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def cli_timeout(*_a, **_k):
        calls["n"] += 1
        raise sr.GroupScopedProviderError(
            "agy CLI-internal response timeout", kind=sr.AbstainReason.TIMEOUT
        )

    vote = _attempt(cli_timeout, budget=300.0)
    assert vote.choice == "ABSTAIN"
    assert vote.abstain_reason == sr.AbstainReason.TIMEOUT
    assert "response timeout" in vote.error
    assert calls["n"] == 1  # deterministic per group -> single attempt, no retry
    assert clock.sleeps == []  # no backoff


def test_context_overflow_abstains_and_continues(monkeypatch):
    """A context-window overflow is deterministic per group -> abstain, not halt."""
    clock = _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def overflowing(*_a, **_k):
        calls["n"] += 1
        raise sr.GroupScopedProviderError(
            "codex context overflow: prompt exceeds the model's context window"
        )

    vote = _attempt(overflowing, budget=300.0)
    assert vote.choice == "ABSTAIN"
    assert "context overflow" in vote.error
    # Overflow resets (does NOT count toward) the timeout breaker.
    assert vote.abstain_reason == sr.AbstainReason.CONTEXT_OVERFLOW
    assert calls["n"] == 1
    assert clock.sleeps == []


def test_check_exit_classifies_context_overflow_beyond_snippet():
    """Overflow markers are scanned in the FULL output, not the 500-char snippet.

    codex prints a long banner + echoed prompt to stderr before its overflow
    line; truncating first would misclassify the overflow as provider-down.
    """
    banner = "OpenAI Codex v0.142.5\n" + ("x" * 600) + "\n"
    result = subprocess.CompletedProcess(
        args=["codex"],
        returncode=1,
        stdout="",
        stderr=banner + "ERROR: Codex ran out of room in the model's context window.",
    )
    with pytest.raises(sr.GroupScopedProviderError, match="context overflow") as exc_info:
        sr._check_exit("codex", result)
    # kind rides onto the abstain so the breaker can distinguish it from a timeout.
    assert exc_info.value.kind == sr.AbstainReason.CONTEXT_OVERFLOW


def test_check_exit_classifies_cli_internal_timeout():
    """agy's own 'timeout waiting for response' is group-scoped, not provider-down.

    It carries kind=TIMEOUT so run_batch's consecutive-timeout breaker COUNTS it
    (BUG 1): a network-blackholed agy fails every group this way, and resetting on
    it would leave the breaker unreachable for the provider it was built for.
    """
    result = subprocess.CompletedProcess(
        args=["agy"], returncode=1, stdout="", stderr="Error: timeout waiting for response"
    )
    with pytest.raises(sr.GroupScopedProviderError, match="response timeout") as exc_info:
        sr._check_exit("agy", result)
    assert exc_info.value.kind == sr.AbstainReason.TIMEOUT


def test_check_exit_generic_failure_still_runtime_error():
    """Non-group-scoped nonzero exits keep the provider-down classification."""
    result = subprocess.CompletedProcess(
        args=["codex"], returncode=1, stdout="", stderr="401 unauthorized"
    )
    with pytest.raises(RuntimeError) as exc_info:
        sr._check_exit("codex", result)
    assert not isinstance(exc_info.value, sr.GroupScopedProviderError)


def test_invoke_agy_print_timeout_derived_from_timeout(monkeypatch, tmp_path):
    """agy's internal response timer follows the caller's timeout (was: 2m hardcoded)."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    sr.invoke_agy("PROMPT", tmp_path, ["A"], "Gemini 3.5 Flash (Medium)", timeout=240)
    assert "--print-timeout=225s" in captured["cmd"]
    # Small caller timeout: derived timer stays STRICTLY BELOW the subprocess
    # deadline (was max(30, timeout-15), which inverted to 30s >= a 20s deadline
    # so the SIGKILL fired first and agy's clean-error path never engaged).
    sr.invoke_agy("PROMPT", tmp_path, ["A"], "Gemini 3.5 Flash (Medium)", timeout=20)
    assert "--print-timeout=10s" in captured["cmd"]


@pytest.mark.parametrize("timeout", [30, 45, 60, 240])
def test_agy_print_timeout_strictly_below_subprocess_timeout(timeout):
    """BUG 2 regression: the derived agy timer must fire before the subprocess kill.

    If the internal timer >= the subprocess timeout, SIGKILL wins and agy never
    emits its clean 'timeout waiting for response' — the whole point of deriving
    the timer (#343). The old max(30, timeout-15) tied at 30s for timeout=30.
    """
    derived = sr._agy_print_timeout(timeout)
    assert derived < timeout, f"{derived} not < {timeout}"
    assert derived >= 1
    # Production 240s is unchanged from the original 15s-margin intent.
    if timeout == 240:
        assert derived == 225


def test_agy_print_timeout_tiny_caller_timeout_still_strictly_below():
    """Even absurdly small caller timeouts keep a strictly-less internal timer."""
    for t in (1, 5, 11, 15, 25):
        assert sr._agy_print_timeout(t) < t


def test_invocation_error_recovers_within_budget(monkeypatch):
    """A transient failure that clears on retry yields a normal vote (no raise)."""
    _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary 429 rate limit")
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    vote = _attempt(flaky, budget=60.0)
    assert vote.choice == "A" and vote.provider == "claude"
    assert calls["n"] == 2  # failed once, succeeded on retry


def test_unexpected_exception_propagates_not_retried(monkeypatch):
    """A programming bug in an invoker must fail fast, not masquerade as quota."""
    _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def buggy(*_a, **_k):
        calls["n"] += 1
        raise AttributeError("'NoneType' object has no attribute 'foo'")

    with pytest.raises(AttributeError):
        _attempt(buggy, budget=300.0)
    assert calls["n"] == 1  # not caught, not retried across the budget


def test_parse_error_still_abstains_not_hard_fail():
    """Malformed output abstains (soft) — it must NOT halt the run."""

    def garbage(*_a, **_k):
        return "no json here, just prose"

    vote = _attempt(garbage, retries=1)
    assert vote.choice == "ABSTAIN"
    assert "parse/validation" in vote.error
    # A parse failure is a property of the response, not provider health -> resets
    # (does NOT count toward) the timeout breaker.
    assert vote.abstain_reason == sr.AbstainReason.PARSE


def test_hard_fail_propagates_through_run_provider_on_group(monkeypatch):
    """run_provider_on_group must not swallow ProviderInvocationError."""
    _install_fake_clock(monkeypatch)

    def always_fail(*_a, **_k):
        raise RuntimeError("exited with code 1: quota exceeded")

    monkeypatch.setitem(sr._INVOKERS, "claude", always_fail)
    with pytest.raises(sr.ProviderInvocationError):
        sr.run_provider_on_group(
            sr.ProviderSpec(name="claude", model="m"),
            "g1",
            None,  # group_dir None -> skip scratch pack
            "prompt",
            ["A"],
            {"A": [("r1", "t1")]},
            invocation_budget_s=0.0,
        )


def test_deterministic_oserror_fast_fails_without_backoff(monkeypatch):
    """An E2BIG-class OSError hard-fails immediately (no wasted backoff budget)."""
    clock = _install_fake_clock(monkeypatch)

    def arg_too_long(*_a, **_k):
        raise OSError(errno.E2BIG, "Argument list too long")

    with pytest.raises(sr.ProviderInvocationError, match="not retryable"):
        _attempt(arg_too_long, budget=300.0)
    assert clock.sleeps == []  # failed on the first attempt, never slept


def test_missing_binary_oserror_fast_fails(monkeypatch):
    """A missing-CLI (ENOENT) failure is deterministic -> immediate hard-fail."""
    _install_fake_clock(monkeypatch)

    def no_binary(*_a, **_k):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory")

    with pytest.raises(sr.ProviderInvocationError, match="not retryable"):
        _attempt(no_binary, budget=300.0)


def test_transient_oserror_still_backs_off(monkeypatch):
    """A non-fatal OSError (e.g. ECONNREFUSED) keeps the backoff-then-hardfail path."""
    clock = _install_fake_clock(monkeypatch)

    def conn_refused(*_a, **_k):
        raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

    with pytest.raises(sr.ProviderInvocationError):
        _attempt(conn_refused, budget=300.0)
    assert clock.sleeps  # backed off (not a fatal errno), then gave up on budget


# --- invoker transport contract: prompt must NOT go on argv (E2BIG regression) ---


def _capture_subprocess_run(monkeypatch):
    """Patch subprocess.run to capture cmd/input/cwd and return a clean result."""
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = list(cmd)
        cap["input"] = input
        cap["cwd"] = kw.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"choice": "A"}', stderr="")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    return cap


# A prompt larger than the Linux single-arg limit (MAX_ARG_STRLEN ~128 KB): if any
# invoker still put this on argv, a real exec would raise E2BIG.
_BIG_PROMPT = "X" * 200_000


def test_invoke_codex_prompt_via_stdin_not_argv(tmp_path, monkeypatch):
    cap = _capture_subprocess_run(monkeypatch)
    sr.invoke_codex(_BIG_PROMPT, tmp_path, ["A"], "gpt-5.5")
    assert cap["input"] == _BIG_PROMPT  # prompt piped on stdin
    assert _BIG_PROMPT not in cap["cmd"]  # never an argv element
    assert "-" in cap["cmd"]  # stdin sentinel present


def test_invoke_opencode_prompt_via_stdin_not_argv(tmp_path, monkeypatch):
    cap = _capture_subprocess_run(monkeypatch)
    sr.invoke_opencode(_BIG_PROMPT, tmp_path, ["A"], "openrouter/qwen/x")
    assert cap["input"] == _BIG_PROMPT
    assert _BIG_PROMPT not in cap["cmd"]
    assert cap["cmd"][:2] == ["opencode", "run"]


def test_invoke_agy_prompt_via_file_not_argv(tmp_path, monkeypatch):
    cap = _capture_subprocess_run(monkeypatch)
    sr.invoke_agy(_BIG_PROMPT, tmp_path, ["A"], "Gemini 3.5 Flash (Medium)")
    assert _BIG_PROMPT not in cap["cmd"]  # not on argv
    assert cap["cwd"] == str(tmp_path)  # agy runs in the group dir
    pf = tmp_path / "panel_prompt.txt"
    assert pf.exists() and pf.read_text() == _BIG_PROMPT  # prompt written to file
    assert "--dangerously-skip-permissions" in cap["cmd"]
