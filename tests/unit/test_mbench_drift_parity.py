"""Cross-package guard for the label -> current-group DRIFT MAPPING.

Stitch ``group_id`` is a content hash of a group's ref/target id set, so any
re-grouping re-mints ids and a stored label must be recovered onto whatever
current group/fragment now covers its geometry. Two packages recover labels, on
purpose with DIFFERENT semantics:

* crosswalk (``agent_labeling.stitch_eval``) recovers PANEL / review labels onto
  ONE best sidecar group (``recover_labeled_groups``), and shares its
  overlap-scoring core (``best_overlap_group``) plus its lenient
  ``selected_edges`` parser (``parse_selected_edge_set``) across the review
  queue / rekey / resolver / expressibility call sites.
* mbench (``eval.stitch_metrics``) evaluates the OPTIMIZER's fragment-level
  output, so a pair label maps to the UNION of every current fragment its edges
  touch (``map_labels_to_fragments``), and a set label prefers the still-
  overlapping VERBATIM group id (``map_set_labels_to_groups``). mbench cannot
  import crosswalk at runtime; only this test (which imports both) can compare.

This file pins BOTH the agreements and the intended divergences as explicit
assertions, so a future silent semantics change on either side fails here and
forces a conscious decision instead of quietly skewing metrics. It mirrors
``test_mbench_set_metric_parity.py`` / ``test_mbench_sliver_parity.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# mbench lives in its own package tree; add it to the path so the main test suite
# (which can import both) can compare the two implementations.
_MBENCH_SRC = Path(__file__).resolve().parents[2] / "mbench" / "src"
if str(_MBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_MBENCH_SRC))

from mbench.eval.stitch_metrics import (  # noqa: E402
    _curated_edge_set as mbench_curated_edge_set,
)
from mbench.eval.stitch_metrics import (  # noqa: E402
    _edge_fragment_index,
    map_labels_to_fragments,
    map_set_labels_to_groups,
)

from crosswalk.agent_labeling.stitch_eval import (  # noqa: E402
    _map_set_labels_to_groups as crosswalk_map_set_labels_to_groups,
)
from crosswalk.agent_labeling.stitch_eval import (  # noqa: E402
    parse_selected_edge_set,
    recover_labeled_groups,
)

# --------------------------------------------------------------------------- #
# Part 1: the selected_edges PARSER — lenient (crosswalk) vs strict (mbench).
# --------------------------------------------------------------------------- #

# Valid inputs both sides must decode identically.
_VALID_PARSER_CASES = [
    "[]",  # reject-all
    '[{"ref_id": "a", "target_id": "x"}]',
    '[{"ref_id": 1, "target_id": 2}]',  # numeric ids -> both str-cast
    '[{"ref_id": "a", "target_id": "x"}, {"ref_id": "b", "target_id": "y"}]',
]


@pytest.mark.parametrize("raw", _VALID_PARSER_CASES)
def test_parser_agrees_on_valid_input(raw):
    """On well-formed selected_edges JSON the two parsers must agree exactly."""
    assert parse_selected_edge_set(raw) == mbench_curated_edge_set(raw)


# UNDECODABLE inputs (fail at ``json.loads``): here the two DELIBERATELY
# diverge — crosswalk is lenient (empty frozenset, so one bad row never aborts a
# drift-mapping pass), mbench is strict (raises, so corrupt curation fails a
# benchmark loudly).
_UNDECODABLE_PARSER_CASES = [
    None,  # null cell
    float("nan"),  # blank CSV cell reads back as NaN
    "",  # empty string
    "   ",  # blank string
    "not json",  # unparseable
]


@pytest.mark.parametrize("raw", _UNDECODABLE_PARSER_CASES)
def test_parser_diverges_on_undecodable_input(raw):
    """Intended divergence: crosswalk returns empty; mbench raises ValueError.

    Do NOT "fix" this by making them match. The lenient parser keeps a single
    undecodable label row from aborting a whole review-queue / rekey /
    expressibility pass; the strict parser makes a corrupt curation label fail a
    benchmark run rather than be silently read as reject-all.
    """
    assert parse_selected_edge_set(raw) == frozenset()  # lenient
    with pytest.raises(ValueError):
        mbench_curated_edge_set(raw)  # strict


def test_parser_structurally_invalid_json_not_swallowed():
    """crosswalk's leniency stops at ``json.loads``; mbench validates structure.

    Decodable-but-wrong JSON (a non-list, an item missing a key, or a blank id)
    is NOT swallowed to empty by the crosswalk parser — it propagates the raw
    ``TypeError`` / ``KeyError``, exactly as every historical private copy did.
    mbench's strict parser instead raises a typed ``ValueError`` naming the
    problem. Pinned so neither the preserved crosswalk behavior nor mbench's
    validation drifts silently.
    """
    # Non-list JSON: crosswalk iterates the dict's keys and indexes a str.
    with pytest.raises(TypeError):
        parse_selected_edge_set('{"ref_id": "a"}')
    with pytest.raises(ValueError):
        mbench_curated_edge_set('{"ref_id": "a"}')

    # List item missing target_id: crosswalk raises KeyError; mbench ValueError.
    with pytest.raises(KeyError):
        parse_selected_edge_set('[{"ref_id": "a"}]')
    with pytest.raises(ValueError):
        mbench_curated_edge_set('[{"ref_id": "a"}]')

    # Blank id: crosswalk passes it through; mbench rejects it.
    assert parse_selected_edge_set('[{"ref_id": "a", "target_id": ""}]') == frozenset({("a", "")})
    with pytest.raises(ValueError):
        mbench_curated_edge_set('[{"ref_id": "a", "target_id": ""}]')


# --------------------------------------------------------------------------- #
# Part 2: the label -> group DRIFT MAPPING itself.
# --------------------------------------------------------------------------- #


def _pair_edges(pairs: list[tuple[str, str]]) -> str:
    """A selected_edges JSON cell for the given (ref, target) pairs."""
    import json

    return json.dumps([{"ref_id": r, "target_id": t} for r, t in pairs])


def _group(gid: str, edges: list[tuple[str, str]]) -> dict:
    """A minimal current sidecar group whose ref/target ids equal its endpoints."""
    return {
        "group_id": gid,
        "ref_ids": sorted({r for r, _ in edges}),
        "target_ids": sorted({t for _, t in edges}),
        "edges": [{"ref_id": r, "target_id": t} for r, t in edges],
    }


def _mbench_fragment_indexes(groups: list[dict]) -> tuple:
    """mbench's fragment-index tuple built from the same groups' selected edges."""
    primary = {
        str(g["group_id"]): frozenset(
            (str(e["ref_id"]), str(e["target_id"])) for e in g.get("edges", [])
        )
        for g in groups
    }
    return (_edge_fragment_index(primary),)


def _mbench_group_members(groups: list[dict]) -> dict[str, frozenset[str]]:
    """mbench's group_members map (segment ids per current group)."""
    return {
        str(g["group_id"]): frozenset(str(r) for r in g.get("ref_ids", []))
        | frozenset(str(t) for t in g.get("target_ids", []))
        for g in groups
    }


def test_pair_label_single_group_agrees():
    """A pair label fully inside ONE current group: both map to that one group.

    crosswalk returns it as a ``clean`` (old -> single best group) recovery;
    mbench returns the union of touched fragments, which here is that same
    single group. Agreement.
    """
    groups = [_group("g1", [("a", "x"), ("b", "y")]), _group("g2", [("c", "z")])]
    labels = pd.DataFrame(
        [{"group_id": "old1", "selected_edges": _pair_edges([("a", "x"), ("b", "y")])}]
    )

    rec = recover_labeled_groups(groups, labels)
    assert rec["clean"] == [("old1", "g1")]
    assert rec["split"] == []

    frag = map_labels_to_fragments(
        labels, _mbench_fragment_indexes(groups), frozenset({"g1", "g2"})
    )
    assert frag == {0: frozenset({"g1"})}

    # Same conclusion: the label belongs to exactly g1.
    crosswalk_group = rec["clean"][0][1]
    assert frag[0] == frozenset({crosswalk_group})


def test_pair_label_spanning_groups_diverges_best_vs_union():
    """INTENDED divergence: crosswalk picks ONE best group; mbench unions both.

    A pair label whose edges span two current groups is a ``split`` for
    crosswalk (recover-onto-one-best-group, for panel/queue adjudication) but a
    two-fragment union for mbench (which scores optimizer output across every
    fragment the curated edges touch). Neither is wrong; they answer different
    questions. Locking both sides here means a change to either must be
    deliberate.
    """
    groups = [_group("g1", [("a", "x")]), _group("g2", [("b", "y"), ("c", "z")])]
    labels = pd.DataFrame(
        [
            {
                "group_id": "old1",
                "selected_edges": _pair_edges([("a", "x"), ("b", "y"), ("c", "z")]),
            }
        ]
    )

    # crosswalk: single best group (g2 holds 2 of 3 edges) -> split, one target.
    rec = recover_labeled_groups(groups, labels)
    assert rec["clean"] == []
    assert rec["split"] == [("old1", "g2", 2, 3)]
    crosswalk_targets = {s for _, s, _, _ in rec["split"]}
    assert crosswalk_targets == {"g2"}

    # mbench: UNION of every fragment the edges touch -> both groups.
    frag = map_labels_to_fragments(
        labels, _mbench_fragment_indexes(groups), frozenset({"g1", "g2"})
    )
    assert frag == {0: frozenset({"g1", "g2"})}

    # The divergence: crosswalk keeps one group, mbench keeps both.
    assert frag[0] == {"g1", "g2"}
    assert frag[0] != crosswalk_targets


def _set_label(gid: str, ref_ids: list[str], target_ids: list[str]) -> dict:
    import json

    return {
        "group_id": gid,
        "selected_edges": "[]",
        "label_semantics": "set",
        "ref_ids": json.dumps(ref_ids),
        "target_ids": json.dumps(target_ids),
    }


def test_set_label_single_group_agrees():
    """A set label whose members live in exactly one current group: all agree."""
    groups = [_group("g1", [("a", "x"), ("b", "y")])]
    row = _set_label("g1", ["a", "b"], ["x", "y"])
    labels = pd.DataFrame([row])

    rec = recover_labeled_groups(groups, labels)
    assert rec["set"] == [("g1", "g1")]

    members = _mbench_group_members(groups)
    assert map_set_labels_to_groups(labels, members) == {0: "g1"}
    assert crosswalk_map_set_labels_to_groups(labels, members) == {"g1": "g1"}


def test_set_label_verbatim_preference_diverges():
    """INTENDED divergence: verbatim-id preference vs pure max-overlap.

    The label's own group id ``gA`` still overlaps (shares segment ``a``) but a
    different current group ``gB`` overlaps MORE (a, b, c). mbench — and
    crosswalk's panel-eval set mapper ``_map_set_labels_to_groups`` — prefer the
    still-overlapping verbatim id ``gA``; crosswalk's ``recover_labeled_groups``
    set branch is pure max-overlap and takes ``gB``. This intra-ecosystem split
    is deliberate: the verbatim-preferring mappers keep a reviewer's exact
    labeled group when it survives at all, while the review-queue/rekey/resolver
    recovery follows the geometry to wherever most of it now lives.
    """
    groups = [
        _group("gA", [("a", "p")]),  # verbatim survivor: overlaps members on 'a'
        _group("gB", [("a", "x"), ("b", "y"), ("c", "z")]),  # higher overlap
    ]
    # Label id is gA; members a, b, c (b/c only exist in gB).
    row = _set_label("gA", ["a", "b"], ["c"])
    labels = pd.DataFrame([row])

    # crosswalk recover_labeled_groups: pure max-overlap -> gB.
    rec = recover_labeled_groups(groups, labels)
    assert rec["set"] == [("gA", "gB")]

    members = _mbench_group_members(groups)
    # mbench: verbatim gA still overlaps -> preferred.
    assert map_set_labels_to_groups(labels, members) == {0: "gA"}
    # crosswalk's OWN panel-eval set mapper matches mbench (shared verbatim rule).
    assert crosswalk_map_set_labels_to_groups(labels, members) == {"gA": "gA"}

    # The divergence, pinned: recovery follows geometry (gB); the verbatim-
    # preferring mappers keep the surviving labeled id (gA).
    assert rec["set"][0][1] == "gB"
    assert map_set_labels_to_groups(labels, members)[0] == "gA"


def test_set_label_verbatim_gone_all_agree():
    """When the verbatim id is gone, all three mappers fall back to max-overlap."""
    groups = [_group("gB", [("a", "x"), ("b", "y"), ("c", "z")])]
    row = _set_label("old_missing", ["a", "b"], ["c"])  # id not a current group
    labels = pd.DataFrame([row])

    rec = recover_labeled_groups(groups, labels)
    assert rec["set"] == [("old_missing", "gB")]

    members = _mbench_group_members(groups)
    assert map_set_labels_to_groups(labels, members) == {0: "gB"}
    assert crosswalk_map_set_labels_to_groups(labels, members) == {"old_missing": "gB"}
