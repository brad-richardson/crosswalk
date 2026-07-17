"""Tests for the consensus-desired-edges option-menu seed (agent_labeling).

Covers the detector (``consensus_desired_edge_sets``) and its glue
(``consensus_seed_edge_sets_for_group``): a group where >= 2 seats share a
mapped desired set yields a new seeded option; no agreement yields none; a
desired set that equals an already-offered option is not re-seeded; and
degenerate / unmappable desired sets are skipped.
"""

from __future__ import annotations

import pandas as pd

from crosswalk.agent_labeling.consensus_desired import (
    consensus_desired_edge_sets,
    consensus_seed_edge_sets_for_group,
    label_map_from_group,
    map_desired_to_ids,
    parse_desired_edges,
)
from crosswalk.matching.alternatives import generate_top_k_alternatives
from crosswalk.matching.stitch_options import build_stitch_options

# R# -> ref id, T# -> target id (mirrors stitch_diagnostic._label_maps).
LABEL_MAP = {
    "reference": {"R1": "ref-a", "R2": "ref-b", "R3": "ref-c"},
    "target": {"T1": "tgt-1", "T2": "tgt-2"},
}


def _ballot(provider, none_reason="", desired=None, choice="NONE"):
    """A minimal votes.csv-shaped ballot row.

    ``desired`` is a list of ``[R#, T#]`` display-label pairs (canonical JSON
    string, as the runner writes it), or None.
    """
    import json

    return {
        "provider": provider,
        "model": f"{provider}-model",
        "choice": choice,
        "none_reason": none_reason,
        "desired_edges": json.dumps(desired) if desired is not None else "",
    }


# --------------------------------------------------------------------------- #
# parse / map primitives
# --------------------------------------------------------------------------- #


def test_parse_desired_edges_from_json_pairs():
    assert parse_desired_edges('[["R1", "T4"], ["R2", "T1"]]') == [("R1", "T4"), ("R2", "T1")]


def test_parse_desired_edges_from_dicts():
    cell = [{"ref_id": "R1", "target_id": "T1"}, {"ref_id": "R2", "target_id": "T2"}]
    assert parse_desired_edges(cell) == [("R1", "T1"), ("R2", "T2")]


def test_parse_desired_edges_malformed_and_empty_yield_nothing():
    assert parse_desired_edges("") == []
    assert parse_desired_edges("not json") == []
    assert parse_desired_edges(None) == []
    assert parse_desired_edges(float("nan")) == []
    assert parse_desired_edges('[["R1"]]') == []  # wrong arity dropped


def test_map_desired_to_ids_maps_all():
    got = map_desired_to_ids([("R1", "T1"), ("R2", "T2")], LABEL_MAP)
    assert got == frozenset({("ref-a", "tgt-1"), ("ref-b", "tgt-2")})


def test_map_desired_to_ids_none_when_unmappable():
    # R9 has no entry -> whole set is degenerate/unmappable.
    assert map_desired_to_ids([("R1", "T1"), ("R9", "T1")], LABEL_MAP) is None


def test_map_desired_to_ids_none_when_empty():
    assert map_desired_to_ids([], LABEL_MAP) is None


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #


def test_two_seats_agree_yields_consensus_set():
    desired = [["R1", "T1"], ["R2", "T2"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    got = consensus_desired_edge_sets(ballots, LABEL_MAP, offered_option_sets=[])
    assert got == [frozenset({("ref-a", "tgt-1"), ("ref-b", "tgt-2")})]


def test_single_seat_no_consensus():
    ballots = [_ballot("claude", "no_exact_option", [["R1", "T1"], ["R2", "T2"]])]
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, offered_option_sets=[]) == []


def test_same_seat_multiple_attempts_not_two_seats():
    """One provider casting two attempts of the same set is still one seat."""
    desired = [["R1", "T1"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("claude", "no_exact_option", desired),
    ]
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, offered_option_sets=[]) == []


def test_two_seats_but_different_sets_no_consensus():
    ballots = [
        _ballot("claude", "no_exact_option", [["R1", "T1"]]),
        _ballot("codex", "no_exact_option", [["R2", "T2"]]),
    ]
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, offered_option_sets=[]) == []


def test_set_equal_to_offered_option_not_reseeded():
    desired = [["R1", "T1"], ["R2", "T2"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    offered = [[("ref-a", "tgt-1"), ("ref-b", "tgt-2")]]  # already on the menu
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, offered) == []


def test_unmappable_desired_set_skipped():
    desired = [["R1", "T1"], ["R9", "T1"]]  # R9 not in the label map
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, offered_option_sets=[]) == []


def test_non_no_exact_option_ballots_ignored():
    desired = [["R1", "T1"], ["R2", "T2"]]
    ballots = [
        _ballot("claude", "all_edges_no_match", desired),
        _ballot("codex", "insufficient_evidence", desired),
    ]
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, offered_option_sets=[]) == []


def test_set_outside_candidate_universe_skipped():
    """A mapped set referencing an edge not in the candidate universe cannot be
    an option, so it is dropped even with seat agreement."""
    desired = [["R1", "T1"], ["R2", "T2"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    candidate = [("ref-a", "tgt-1")]  # (ref-b, tgt-2) is NOT a candidate edge
    got = consensus_desired_edge_sets(
        ballots, LABEL_MAP, offered_option_sets=[], candidate_edges=candidate
    )
    assert got == []


def test_detector_accepts_dataframe_ballots():
    desired = [["R1", "T1"], ["R2", "T2"]]
    df = pd.DataFrame(
        [
            _ballot("claude", "no_exact_option", desired),
            _ballot("codex", "no_exact_option", desired),
        ]
    )
    got = consensus_desired_edge_sets(df, LABEL_MAP, offered_option_sets=[])
    assert got == [frozenset({("ref-a", "tgt-1"), ("ref-b", "tgt-2")})]


def test_three_seat_agreement_min_seats_configurable():
    desired = [["R1", "T1"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    # default min_seats=2 -> consensus
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, []) == [frozenset({("ref-a", "tgt-1")})]
    # require 3 -> none
    assert consensus_desired_edge_sets(ballots, LABEL_MAP, [], min_seats=3) == []


# --------------------------------------------------------------------------- #
# label map + group glue
# --------------------------------------------------------------------------- #


def test_label_map_from_group_orders_by_stored_ids():
    group = {"ref_ids": ["ref-a", "ref-b"], "target_ids": ["tgt-1", "tgt-2"]}
    lm = label_map_from_group(group)
    assert lm == {
        "reference": {"R1": "ref-a", "R2": "ref-b"},
        "target": {"T1": "tgt-1", "T2": "tgt-2"},
    }


def _wide_group():
    """A group where one target (t1) is fed by 4 refs (>MAX_REF_CHAIN_LEN).

    With no geometry the per-target enumeration can assign t1 at most ONE ref,
    so the 4-refs-to-t1 set is organically inexpressible and is not the full
    candidate set (which also carries ref-a->t2). Uniformly high confidence and
    ``selected`` flags mean no minus-flagged seed emits it either, so it is a
    genuine menu gap.
    """
    edges = [
        {"ref_id": "ref-a", "target_id": "tgt-1", "confidence": 0.99, "selected": True},
        {"ref_id": "ref-b", "target_id": "tgt-1", "confidence": 0.99, "selected": True},
        {"ref_id": "ref-c", "target_id": "tgt-1", "confidence": 0.99, "selected": True},
        {"ref_id": "ref-d", "target_id": "tgt-1", "confidence": 0.99, "selected": True},
        {"ref_id": "ref-a", "target_id": "tgt-2", "confidence": 0.99, "selected": True},
    ]
    return {
        "group_id": "g-wide",
        "ref_ids": ["ref-a", "ref-b", "ref-c", "ref-d"],
        "target_ids": ["tgt-1", "tgt-2"],
        "edges": edges,
    }


def test_glue_returns_unreachable_consensus_set():
    group = _wide_group()
    lm = label_map_from_group(group)
    # Both seats want all four refs on t1 (labels R1..R4 -> ref-a..ref-d).
    desired = [["R1", "T1"], ["R2", "T1"], ["R3", "T1"], ["R4", "T1"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    got = consensus_seed_edge_sets_for_group(group, ballots, lm, k=8)
    want = frozenset(
        {("ref-a", "tgt-1"), ("ref-b", "tgt-1"), ("ref-c", "tgt-1"), ("ref-d", "tgt-1")}
    )
    assert want in got


def test_glue_drops_set_already_on_menu():
    group = _wide_group()
    lm = label_map_from_group(group)
    # The full candidate set is always a seed option; wanting exactly it is not
    # a gap, so the glue returns nothing for it.
    desired = [
        ["R1", "T1"],
        ["R2", "T1"],
        ["R3", "T1"],
        ["R4", "T1"],
        ["R1", "T2"],
    ]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    assert consensus_seed_edge_sets_for_group(group, ballots, lm, k=8) == []


# --------------------------------------------------------------------------- #
# end-to-end: seed becomes a lettered option
# --------------------------------------------------------------------------- #


def test_consensus_seed_becomes_lettered_option():
    """The detector output, fed to the generator's seed_edge_sets, surfaces as a
    real one-click option via build_stitch_options."""
    group = _wide_group()
    lm = label_map_from_group(group)
    desired = [["R1", "T1"], ["R2", "T1"], ["R3", "T1"], ["R4", "T1"]]
    ballots = [
        _ballot("claude", "no_exact_option", desired),
        _ballot("codex", "no_exact_option", desired),
    ]
    seeds = consensus_seed_edge_sets_for_group(group, ballots, lm, k=8)
    assert seeds

    g = dict(group)
    g["alternatives"] = generate_top_k_alternatives(group["edges"], k=8, seed_edge_sets=seeds)
    ctx = build_stitch_options(g)
    option_sets = [
        frozenset((e["ref_id"], e["target_id"]) for e in opt["edges"]) for opt in ctx["options"]
    ]
    want = frozenset(
        {("ref-a", "tgt-1"), ("ref-b", "tgt-1"), ("ref-c", "tgt-1"), ("ref-d", "tgt-1")}
    )
    assert want in option_sets
