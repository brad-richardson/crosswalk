"""Unit tests for the experimental learned-group-resolver extraction harness.

Uses tiny synthetic sidecar groups + labels (no dependency on data/), so these
run in CI without the untracked runtime data. They pin the data contract:
label = edge in human selected set, provenance clean vs split, and that
featurize produces the declared feature columns.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from crosswalk.resolver.extract import (
    KEY_COLUMNS,
    build_edge_table,
    load_sidecar_groups,
)
from crosswalk.resolver.features import FEATURE_COLUMNS, featurize


def _edge(ref, tgt, conf, selected=True, **kw):
    d = {
        "ref_id": ref,
        "target_id": tgt,
        "confidence": conf,
        "selected": selected,
        "degree_ref": 1,
        "degree_tgt": 1,
        "is_bridge": True,
        "is_sliver": False,
        "biconnected_block": 0,
        "corridor_ref": 0,
        "corridor_tgt": 0,
        "gers_start_frac": 0.0,
        "gers_end_frac": 1.0,
        "local_start_frac": 0.0,
        "local_end_frac": 1.0,
    }
    d.update(kw)
    return d


def _group(gid, edges, match_type="N:1"):
    refs = sorted({e["ref_id"] for e in edges})
    tgts = sorted({e["target_id"] for e in edges})
    return {
        "group_id": gid,
        "match_type": match_type,
        "edges": edges,
        "ref_ids": refs,
        "target_ids": tgts,
        "n_edges": len(edges),
        "n_corridors": 1,
        "n_assignment_components": 1,
        "largest_biconnected_block": 1,
        "oversized_group": False,
    }


def _labels(rows):
    return pd.DataFrame(rows)


def _label_row(gid, edges, labeler="brad"):
    return {
        "group_id": gid,
        "dataset_id": "ds",
        "selected_edges": json.dumps([{"ref_id": r, "target_id": t} for r, t in edges]),
        "match_type": "N:1",
        "num_refs": 1,
        "num_targets": 1,
        "labeler": labeler,
        "labeled_at": "2026-01-01",
        "session_id": "x",
    }


def test_clean_label_keep_drop():
    # group g1 has edges A->T (kept) and B->T (should be dropped by human)
    groups = [_group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4)])]
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table(groups, human, "ds")
    assert len(df) == 2
    assert set(df["provenance"]) == {"clean"}
    keep_by_edge = dict(zip(df["ref_id"], df["keep"]))
    assert keep_by_edge["A"] == 1
    assert keep_by_edge["B"] == 0
    # optimizer baseline (selected) kept both -> would be 1 FP against human
    assert df["selected"].all()


def test_split_provenance_and_include_flag():
    # human selected spans two groups -> maps to best group, provenance=split
    groups = [
        _group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.99)]),
        _group("g2", [_edge("C", "U", 0.99)]),
    ]
    human = _labels([_label_row("hg1", [("A", "T"), ("B", "T"), ("C", "U")])])
    df_all = build_edge_table(groups, human, "ds", include_split=True)
    assert set(df_all["provenance"]) == {"split"}
    df_clean = build_edge_table(groups, human, "ds", include_split=False)
    assert df_clean.empty


def test_featurize_produces_declared_columns():
    groups = [_group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4)])]
    human = _labels([_label_row("hg1", [("A", "T")])])
    feat = featurize(build_edge_table(groups, human, "ds"))
    for col in FEATURE_COLUMNS:
        assert col in feat.columns, f"missing feature {col}"
    # relative-confidence: the dropped low-conf edge is the group min
    row_b = feat[feat["ref_id"] == "B"].iloc[0]
    assert row_b["conf_is_group_min"] == 1
    assert row_b["conf_rel_max"] < 0


def test_resolver_not_imported_by_production_code():
    """The experimental resolver must not be referenced by pipeline modules.

    Guards the 'zero production behavior change' invariant: nothing under the
    matching / features / cli / optimizer paths may import crosswalk.resolver.

    Catches BOTH absolute (``import crosswalk.resolver`` / ``from crosswalk.resolver
    import``) and relative (``from ..resolver import`` / ``from .resolver import``
    / ``from .. import resolver``) import forms — a plain ``crosswalk.resolver``
    substring check misses relative imports entirely.
    """
    import pathlib
    import re

    # Import statements that pull in the experimental resolver package, in any of:
    #   import crosswalk.resolver[...]
    #   from crosswalk.resolver[...] import ...
    #   from .resolver / ..resolver / ...resolver [...] import ...   (relative)
    #   from . / .. / ... import resolver                            (relative)
    patterns = [
        re.compile(r"^\s*import\s+crosswalk\.resolver\b", re.MULTILINE),
        re.compile(r"^\s*from\s+crosswalk\.resolver\b", re.MULTILINE),
        re.compile(r"^\s*from\s+\.+resolver\b", re.MULTILINE),
        re.compile(r"^\s*from\s+\.+\s+import\s+(?:[^\n]*[\s,(])?resolver\b", re.MULTILINE),
    ]

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "crosswalk"
    assert src.is_dir(), f"guard test points at a missing tree: {src}"
    scanned = 0
    offenders = []
    for py in src.rglob("*.py"):
        if "resolver" in py.parts:
            continue
        scanned += 1
        text = py.read_text()
        if any(p.search(text) for p in patterns):
            offenders.append(str(py.relative_to(src)))
    # If the package tree ever moves/renames again, fail loudly instead of
    # passing vacuously over zero files (the post-rename `src/matcher` bug).
    assert scanned > 50, f"guard scanned only {scanned} files under {src} — wrong tree?"
    assert not offenders, f"production code imports experimental resolver: {offenders}"


def test_load_sidecar_groups_roundtrip(tmp_path):
    groups = [_group("g1", [_edge("A", "T", 0.99)])]
    p = tmp_path / "x_groups.json"
    p.write_text(json.dumps({"n_groups": 1, "groups": groups}))
    loaded = load_sidecar_groups(p)
    assert len(loaded) == 1 and loaded[0]["group_id"] == "g1"


# --- M2: rejected candidate edges (under-selection) --------------------------


def test_rejected_edges_become_selected_false_rows():
    """The M2 rejected_edges list is folded into the per-edge table as extra
    rows with selected=False. This is what makes under-selection learnable."""
    grp = _group("g1", [_edge("A", "T", 0.99)])
    # C->T is a candidate the optimizer rejected (not selected anywhere)
    grp["rejected_edges"] = [_edge("C", "T", 0.35, selected=False)]
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert len(df) == 2
    by_ref = dict(zip(df["ref_id"], df["selected"]))
    assert by_ref["A"] is True or by_ref["A"] == True  # noqa: E712
    assert by_ref["C"] == False  # noqa: E712
    # C was not human-selected -> a true negative the optimizer got right
    assert dict(zip(df["ref_id"], df["keep"]))["C"] == 0


def test_rejected_edge_that_human_selected_is_under_selection_positive():
    """A rejected candidate the human DID select is keep=1 with selected=False —
    an under-selection error impossible to observe from the selected-only sidecar."""
    grp = _group("g1", [_edge("A", "T", 0.99)])
    grp["rejected_edges"] = [_edge("B", "T", 0.45, selected=False)]
    # human kept BOTH A->T and B->T
    human = _labels([_label_row("hg1", [("A", "T"), ("B", "T")])])
    df = build_edge_table([grp], human, "ds")
    row_b = df[df["ref_id"] == "B"].iloc[0]
    assert row_b["keep"] == 1
    assert bool(row_b["selected"]) is False


def test_include_rejected_flag_off_excludes_them():
    grp = _group("g1", [_edge("A", "T", 0.99)])
    grp["rejected_edges"] = [_edge("C", "T", 0.35, selected=False)]
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds", include_rejected=False)
    assert set(df["ref_id"]) == {"A"}


def test_rejected_edges_deduped_against_edges():
    """A pair present in both edges and rejected_edges is not double-counted."""
    grp = _group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4)])
    grp["rejected_edges"] = [_edge("B", "T", 0.4, selected=False)]  # dup of an edge
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert len(df) == 2  # A, B — not 3


def test_review_reason_key_is_tolerated_and_round_trips_group_unchanged():
    """The additive ``review_reason`` sidecar key (optimizer demotion reasons,
    see runner.py::_export_groups_sidecar) is not part of this module's
    declared edge schema. It must be silently ignored -- present on the raw
    edge dict but absent from ``_edge_row``'s output -- and every existing
    extracted field (``keep``, ``selected``, row count) must be byte-identical
    to the same group without the key."""
    grp_plain = _group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4)])
    grp_demoted = _group(
        "g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4, review_reason="parallel_sibling")]
    )
    human = _labels([_label_row("hg1", [("A", "T")])])

    df_plain = build_edge_table([grp_plain], human, "ds")
    df_demoted = build_edge_table([grp_demoted], human, "ds")

    pd.testing.assert_frame_equal(df_plain, df_demoted)
    assert "review_reason" not in df_demoted.columns


# --- #344 candidate_edges (design §3 stage-1 consumer) ----------------------


def _cand(ref, tgt, conf, selected, selected_elsewhere=False, pruned=False):
    """Minimal stage-1 candidate_edges record (topology + confidence + flags)."""
    d = {"ref_id": ref, "target_id": tgt, "confidence": conf, "selected": selected}
    if selected_elsewhere:
        d["selected_elsewhere"] = True
    if pruned:
        d["pruned"] = True
    return d


def _group_cg(gid, candidate_edges, ref_ids, target_ids, edges=None, match_type="M:N"):
    """A post-#344 sidecar group with an explicit candidate_edges universe.

    ``ref_ids`` / ``target_ids`` are the optimizer's SELECTED members (they drive
    the rule-5 endpoint-membership filter), independent of the candidate list.
    ``edges`` (the selected assignment, always present on real sidecars and used
    by ``recover_labeled_groups`` for label->group mapping) defaults to the
    selected candidate edges.
    """
    if edges is None:
        edges = [
            _edge(c["ref_id"], c["target_id"], c["confidence"])
            for c in candidate_edges
            if c.get("selected")
        ]
    return {
        "group_id": gid,
        "match_type": match_type,
        "edges": edges or [],
        "candidate_edges": candidate_edges,
        "n_candidate_edges": len(candidate_edges),
        "ref_ids": ref_ids,
        "target_ids": target_ids,
        "n_edges": len(candidate_edges),
        "n_corridors": 1,
        "n_assignment_components": 1,
        "largest_biconnected_block": 1,
        "oversized_group": False,
    }


def test_candidate_graph_flags_map_to_labels():
    """Selected -> positive-candidate, non-selected -> the under-selection
    NEGATIVE; the keep label is human ground truth, not the optimizer flag."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),  # optimizer kept + human kept
            _cand("B", "T", 0.40, selected=False),  # optimizer dropped, human dropped
            _cand("C", "T", 0.55, selected=False),  # optimizer dropped, human KEPT
        ],
        ref_ids=["A", "B", "C"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T"), ("C", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert len(df) == 3
    keep = dict(zip(df["ref_id"], df["keep"]))
    sel = dict(zip(df["ref_id"], df["selected"]))
    assert keep["A"] == 1 and sel["A"]  # selected + kept
    assert keep["B"] == 0 and not sel["B"]  # true negative the optimizer got right
    # under-selection positive: keep=1 but optimizer did NOT select it — only
    # observable because candidate_edges carries the non-selected candidate.
    assert keep["C"] == 1 and not sel["C"]


def test_candidate_graph_selected_elsewhere_excluded():
    """A selected_elsewhere edge IS an optimizer selection (in another group),
    so it must not be emitted as a drop/negative for this group."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.80, selected=False, selected_elsewhere=True),
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A"}  # B excluded (selected elsewhere)


def test_candidate_graph_rule5_filtered_by_endpoint_membership():
    """A candidate edge with NEITHER endpoint in the group's selected members
    (rule-5 attribution noise, #344 review) is dropped by default; opting out
    via filter_rule5=False keeps it."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("X", "Y", 0.60, selected=False),  # neither X nor Y is a member
        ],
        ref_ids=["A"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A"}  # X->Y filtered
    df_keep = build_edge_table([grp], human, "ds", filter_rule5=False)
    assert set(df_keep["ref_id"]) == {"A", "X"}


def test_candidate_graph_rule5_keeps_single_endpoint_member():
    """An edge sharing ONE endpoint with the group (a legitimate under-selection
    candidate — new target for an existing ref) survives the rule-5 filter."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("A", "U", 0.50, selected=False),  # ref A is a member; U is new
        ],
        ref_ids=["A"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T"), ("A", "U")])])
    df = build_edge_table([grp], human, "ds")
    assert set(zip(df["ref_id"], df["target_id"])) == {("A", "T"), ("A", "U")}
    row_u = df[df["target_id"] == "U"].iloc[0]
    assert row_u["keep"] == 1 and not row_u["selected"]


def test_candidate_graph_rule5_keeps_owned_pruned_pendant():
    """Pre-prune ownership is stronger than endpoint membership: a pruned pair
    with neither endpoint left in the group remains an observable positive."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("X", "Y", 0.60, selected=False, pruned=True),
        ],
        ref_ids=["A"],
        target_ids=["T"],
        edges=[_edge("A", "T", 0.99), _edge("X", "Y", 0.60, selected=False, pruned=True)],
    )
    human = _labels([_label_row("hg1", [("A", "T"), ("X", "Y")])])
    df = build_edge_table([grp], human, "ds")

    assert set(zip(df["ref_id"], df["target_id"])) == {("A", "T"), ("X", "Y")}
    pendant = df[(df["ref_id"] == "X") & (df["target_id"] == "Y")].iloc[0]
    assert pendant["keep"] == 1
    assert not bool(pendant["selected"])
    assert df.attrs["build_stats"]["rule5_filtered"] == 0


def test_candidate_graph_rule5_does_not_truthify_malformed_pruned_string():
    """Only the producer's canonical JSON boolean can claim the ownership
    exemption; a truthy string from malformed input remains rule-5 noise."""
    malformed = _cand("X", "Y", 0.60, selected=False)
    malformed["pruned"] = "false"
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, selected=True), malformed],
        ref_ids=["A"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T")])])

    df = build_edge_table([grp], human, "ds")
    assert set(zip(df["ref_id"], df["target_id"])) == {("A", "T")}
    assert df.attrs["build_stats"]["rule5_filtered"] == 1


def test_candidate_graph_enriches_structural_layer_from_edges():
    """candidate_edges is stage-1 (topology only); structural columns are
    enriched from the group's edges/rejected_edges by (ref_id, target_id).
    Genuinely-new candidates get NaN/default structure."""
    edges = [_edge("A", "T", 0.99, corridor_ref=7, degree_ref=3)]
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.40, selected=False),  # not in edges/rejected -> default
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
        edges=edges,
    )
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    row_a = df[df["ref_id"] == "A"].iloc[0]
    row_b = df[df["ref_id"] == "B"].iloc[0]
    assert row_a["corridor_ref"] == 7 and row_a["degree_ref"] == 3  # enriched
    assert row_b["corridor_ref"] == -1 and row_b["degree_ref"] == 0  # default


def test_candidate_graph_preferred_over_legacy_when_present():
    """With both candidate_edges and edges/rejected present, the uncapped
    candidate universe is used (non-selected candidate visible)."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.40, selected=False),
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
        edges=[_edge("A", "T", 0.99)],  # legacy would only see A
    )
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A", "B"}  # candidate graph exposes B
    # forcing legacy path only sees the selected assignment
    df_legacy = build_edge_table([grp], human, "ds", prefer_candidate_graph=False)
    assert set(df_legacy["ref_id"]) == {"A"}


def test_empty_candidate_edges_falls_back_to_legacy():
    """stitch_persist_candidate_graph=False emits candidate_edges=[]; that is
    treated as 'disabled' and routed to the legacy edges+rejected path."""
    grp = _group_cg(
        "g1",
        [],  # empty candidate_edges (feature disabled)
        ref_ids=["A"],
        target_ids=["T"],
        edges=[_edge("A", "T", 0.99)],
    )
    grp["rejected_edges"] = [_edge("C", "T", 0.3, selected=False)]
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A", "C"}  # legacy path used


def test_legacy_sidecar_without_candidate_edges_key():
    """A pre-#344 sidecar (no candidate_edges key at all) still builds via the
    legacy path — backward compatible."""
    grp = _group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4)])
    assert "candidate_edges" not in grp
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A", "B"}
    assert dict(zip(df["ref_id"], df["keep"])) == {"A": 1, "B": 0}


def test_key_columns_stable_across_paths():
    """(dataset_id, group_id, ref_id, target_id) — the stage-2 feature-join key —
    is present on rows from BOTH the candidate-graph and the legacy paths."""
    cg = _group_cg("g1", [_cand("A", "T", 0.99, selected=True)], ref_ids=["A"], target_ids=["T"])
    legacy = _group("g2", [_edge("A", "T", 0.99)])
    human = _labels([_label_row("hg1", [("A", "T")])])
    df_cg = build_edge_table([cg], human, "ds")
    df_lg = build_edge_table([legacy], human, "ds")
    for df in (df_cg, df_lg):
        for col in KEY_COLUMNS:
            assert col in df.columns, f"missing key column {col}"
    assert list(KEY_COLUMNS) == ["dataset_id", "group_id", "ref_id", "target_id"]


def test_identical_duplicate_label_truth_is_deduplicated_before_parquet_enrichment():
    """Historical labels recovering to one current group may repeat its entire
    candidate universe. Equal truth is safe to dedupe, and collision attrs must
    survive the stage-2 parquet merges."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.40, selected=False),
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    human = _labels(
        [
            _label_row("historical-1", [("A", "T")]),
            _label_row("historical-2", [("A", "T")]),
        ]
    )
    candidates = pd.DataFrame(
        [
            {"group_id": "g1", "ref_id": "A", "target_id": "T", "typed_value": 1.0},
            {"group_id": "g1", "ref_id": "B", "target_id": "T", "typed_value": 2.0},
        ]
    )

    df = build_edge_table([grp], human, "ds", candidates_df=candidates)

    assert len(df) == 2
    assert not df.duplicated(subset=list(KEY_COLUMNS)).any()
    assert list(df.sort_values("ref_id")["typed_value"]) == [1.0, 2.0]
    assert set(df["historical_human_group_ids"]) == {'["historical-1", "historical-2"]'}
    stats = df.attrs["build_stats"]
    assert stats["raw_rows"] == 4
    assert stats["duplicate_rows"] == 2
    assert stats["duplicate_keys"] == 2
    assert stats["conflicting_keys"] == 0
    assert stats["deduplicated_rows"] == 2
    assert stats["quarantined_groups"] == 0
    assert stats["quarantined_rows"] == 0
    assert stats["candidate_parquet_enriched"] == 2


def test_duplicate_candidate_within_one_historical_label_fails_closed():
    """A sidecar duplicate is not cross-label consensus and must not disappear."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("A", "T", 0.99, selected=True),
        ],
        ref_ids=["A"],
        target_ids=["T"],
    )
    human = _labels([_label_row("historical-1", [("A", "T")])])

    with pytest.raises(ValueError, match="repeat within one historical human_group_id"):
        build_edge_table([grp], human, "ds")


def test_conflicting_duplicate_labels_quarantine_the_entire_current_group():
    """Contradictory truth on any repeated candidate key makes every row from
    that current group ineligible; an unrelated group remains evaluable."""
    conflicting_group = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.40, selected=False),
            _cand("X", "T", 0.30, selected=False),
        ],
        ref_ids=["A", "B", "X"],
        target_ids=["T"],
    )
    unaffected_group = _group_cg(
        "g2",
        [_cand("C", "U", 0.95, selected=True)],
        ref_ids=["C"],
        target_ids=["U"],
    )
    conflicting_group["rejected_edges"] = [_edge("D", "T", 0.2, selected=False)]
    human = _labels(
        [
            _label_row("historical-a", [("A", "T"), ("D", "T")]),
            _label_row("historical-b", [("B", "T"), ("D", "T")]),
            _label_row("historical-c", [("C", "U")]),
        ]
    )

    df = build_edge_table([conflicting_group, unaffected_group], human, "ds")

    assert set(df["group_id"]) == {"g2"}
    assert list(df["keep"]) == [1]
    stats = df.attrs["build_stats"]
    assert stats["raw_rows"] == 7
    assert stats["duplicate_rows"] == 3
    assert stats["duplicate_keys"] == 3
    assert stats["conflicting_keys"] == 2
    assert stats["quarantined_groups"] == 1
    assert stats["quarantined_rows"] == 6
    assert stats["deduplicated_rows"] == 0
    assert stats["rows"] == 1
    assert stats["legacy_known_omission_occurrences"] == 2
    assert stats["legacy_known_omission_unique_raw_keys"] == 1
    assert stats["legacy_known_omission_unique_retained_keys"] == 0

    audit = df.attrs["build_audit"]
    assert audit["schema_version"] == 1
    assert audit["quarantined_groups"] == [
        {
            "case_id": "label_collision:ds:historical-a+historical-b",
            "dataset_id": "ds",
            "current_group_id": "g1",
            "historical_human_group_ids": ["historical-a", "historical-b"],
            "raw_row_occurrences": 6,
            "conflicting_key_count": 2,
            "conflicting_edges": [
                {
                    "ref_id": "A",
                    "target_id": "T",
                    "claims": [
                        {"human_group_id": "historical-a", "provenance": "clean", "keep": 1},
                        {"human_group_id": "historical-b", "provenance": "clean", "keep": 0},
                    ],
                },
                {
                    "ref_id": "B",
                    "target_id": "T",
                    "claims": [
                        {"human_group_id": "historical-a", "provenance": "clean", "keep": 0},
                        {"human_group_id": "historical-b", "provenance": "clean", "keep": 1},
                    ],
                },
            ],
            "historical_labels": [
                {
                    "human_group_id": "historical-a",
                    "provenance": "clean",
                    "labeler": "brad",
                    "labeled_at": "2026-01-01",
                    "label_semantics": "pair",
                    "selected_edges": [
                        {"ref_id": "A", "target_id": "T"},
                        {"ref_id": "D", "target_id": "T"},
                    ],
                },
                {
                    "human_group_id": "historical-b",
                    "provenance": "clean",
                    "labeler": "brad",
                    "labeled_at": "2026-01-01",
                    "label_semantics": "pair",
                    "selected_edges": [
                        {"ref_id": "B", "target_id": "T"},
                        {"ref_id": "D", "target_id": "T"},
                    ],
                },
            ],
        }
    ]
    assert audit["legacy_known_omissions"] == [
        {
            "case_id": "legacy_known_omission:ds:D:T:historical-a+historical-b",
            "dataset_id": "ds",
            "current_group_id": "g1",
            "ref_id": "D",
            "target_id": "T",
            "ref_name": "",
            "target_name": "",
            "retained_after_quarantine": False,
            "occurrence_count": 2,
            "occurrences": [
                {"human_group_id": "historical-a", "provenance": "clean"},
                {"human_group_id": "historical-b", "provenance": "clean"},
            ],
        }
    ]


def test_build_audit_is_deterministic_across_input_order():
    candidates = [
        _cand("A", "T", 0.99, selected=True),
        _cand("B", "T", 0.40, selected=False),
        _cand("X", "T", 0.30, selected=False),
    ]
    unaffected = _group_cg(
        "g2",
        [_cand("C", "U", 0.95, selected=True)],
        ref_ids=["C"],
        target_ids=["U"],
    )
    labels = [
        _label_row("historical-b", [("B", "T"), ("D", "T")]),
        _label_row("historical-a", [("A", "T"), ("D", "T")]),
        _label_row("historical-c", [("C", "U")]),
    ]

    first_group = _group_cg("g1", candidates, ref_ids=["A", "B", "X"], target_ids=["T"])
    first_group["rejected_edges"] = [_edge("D", "T", 0.2, selected=False)]
    first = build_edge_table([first_group, unaffected], _labels(labels), "ds")

    reversed_group = _group_cg(
        "g1", list(reversed(candidates)), ref_ids=["X", "B", "A"], target_ids=["T"]
    )
    reversed_group["rejected_edges"] = [_edge("D", "T", 0.2, selected=False)]
    second = build_edge_table([unaffected, reversed_group], _labels(list(reversed(labels))), "ds")

    assert json.dumps(first.attrs["build_audit"], sort_keys=True) == json.dumps(
        second.attrs["build_audit"], sort_keys=True
    )


def test_candidate_graph_featurize_over_full_universe():
    """featurize runs over the fuller candidate universe and produces every
    declared feature column, with the non-selected competitor as the group min."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.30, selected=False),
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T")])])
    feat = featurize(build_edge_table([grp], human, "ds"))
    for col in FEATURE_COLUMNS:
        assert col in feat.columns, f"missing feature {col}"
    row_b = feat[feat["ref_id"] == "B"].iloc[0]
    assert row_b["conf_is_group_min"] == 1
    assert row_b["n_share_tgt"] == 2  # A and B both compete for T


# --- empty-set (reject-all) labels (design §2.4a) ----------------------------


def test_empty_set_label_emits_all_candidates_keep0():
    """A reject-all label (selected_edges=[]) mapped by verbatim group_id emits
    the group's full candidate universe with keep=0 and provenance=empty — the
    'select nothing' shape the cross-mode defect requires (design §2.4a)."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.95, selected=True),
            _cand("B", "T", 0.40, selected=False),
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    human = _labels([_label_row("g1", [])])  # group_id matches verbatim
    df = build_edge_table([grp], human, "ds")
    assert len(df) == 2
    assert set(df["provenance"]) == {"empty"}
    assert (df["keep"] == 0).all()
    # the optimizer DID select A->T: that keep=0/selected=True row is the
    # over-selection signal an empty label uniquely provides
    row_a = df[df["ref_id"] == "A"].iloc[0]
    assert bool(row_a["selected"]) is True and row_a["keep"] == 0
    assert df.attrs["build_stats"]["empty_rows"] == 2


def test_empty_set_label_respects_rule5_and_selected_elsewhere():
    """Empty-label rows go through the same rule-5 / selected_elsewhere gates as
    labeled-group rows."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.95, selected=True),
            _cand("B", "T", 0.80, selected=False, selected_elsewhere=True),
            _cand("X", "Y", 0.60, selected=False),  # rule-5 noise
        ],
        ref_ids=["A"],
        target_ids=["T"],
    )
    human = _labels([_label_row("g1", [])])
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A"}
    stats = df.attrs["build_stats"]
    assert stats["rule5_filtered"] == 1
    assert stats["selected_elsewhere_excluded"] == 1


def test_empty_set_label_on_legacy_group_emits_zero_rows():
    """A reject-all label mapped to a group WITHOUT candidate_edges emits no
    rows (the capped legacy view cannot express the full candidate universe) and
    is counted in empty_legacy_skipped."""
    grp = _group("g1", [_edge("A", "T", 0.99)])  # pre-#344, no candidate_edges
    human = _labels([_label_row("g1", [])])
    df = build_edge_table([grp], human, "ds")
    assert df.empty
    assert df.attrs["build_stats"]["empty_legacy_skipped"] == 1


def test_empty_set_label_unrecovered_group_counted():
    """An empty label whose group_id no longer exists is counted, not emitted."""
    grp = _group_cg("g1", [_cand("A", "T", 0.95, selected=True)], ref_ids=["A"], target_ids=["T"])
    human = _labels([_label_row("gone", [])])
    df = build_edge_table([grp], human, "ds")
    assert df.empty
    assert df.attrs["build_stats"]["empty_unrecovered"] == 1


def test_include_empty_flag_off_excludes_reject_all_rows():
    grp = _group_cg("g1", [_cand("A", "T", 0.95, selected=True)], ref_ids=["A"], target_ids=["T"])
    human = _labels([_label_row("g1", [])])
    df = build_edge_table([grp], human, "ds", include_empty=False)
    assert df.empty


# --- self-reporting build stats ----------------------------------------------


def test_build_stats_attached_and_consistent():
    """The per-build counters are attached as df.attrs['build_stats'] and agree
    with the frame."""
    grp = _group_cg(
        "g1",
        [
            _cand("A", "T", 0.99, selected=True),
            _cand("B", "T", 0.40, selected=False),
        ],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    human = _labels([_label_row("hg1", [("A", "T")])])
    df = build_edge_table([grp], human, "ds")
    stats = df.attrs["build_stats"]
    assert stats["rows"] == len(df) == 2
    assert stats["positives"] == int(df["keep"].sum()) == 1
    assert stats["negatives"] == 1
    assert stats["candidate_groups"] == 1 and stats["legacy_groups"] == 0


def test_human_selected_outside_candidate_graph_counted():
    """A human-selected edge present in the legacy view (edges/rejected_edges)
    but missing from candidate_edges (below floor / glue-pruned / attributed
    elsewhere) is counted as a lost positive — visible, not silent."""
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, selected=True)],  # candidate universe misses B->T
        ref_ids=["A"],
        target_ids=["T"],
        edges=[_edge("A", "T", 0.99)],
    )
    grp["rejected_edges"] = [_edge("B", "T", 0.2, selected=False)]  # legacy knows B->T
    human = _labels([_label_row("hg1", [("A", "T"), ("B", "T")])])  # human kept BOTH
    df = build_edge_table([grp], human, "ds")
    assert set(df["ref_id"]) == {"A"}  # B->T not in the candidate universe
    stats = df.attrs["build_stats"]
    assert stats["human_selected_outside_candidate_graph"] == 1
    assert stats["human_selected_outside_candidate_graph_clean"] == 1
    assert stats["human_selected_outside_candidate_graph_split"] == 0
    assert stats["legacy_known_omission_occurrences"] == 1
    assert stats["legacy_known_omission_unique_raw_keys"] == 1
    assert stats["legacy_known_omission_unique_retained_keys"] == 1
    assert stats["legacy_known_omission_occurrences_clean"] == 1


def test_human_selected_outside_candidate_graph_split_counted_separately():
    """The total remains backward compatible while split-label misses are
    separable from clean candidate-recall defects."""
    g1 = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, selected=True)],
        ref_ids=["A"],
        target_ids=["T"],
        edges=[_edge("A", "T", 0.99)],
    )
    g1["rejected_edges"] = [_edge("B", "T", 0.2, selected=False)]
    g2 = _group_cg(
        "g2",
        [_cand("C", "U", 0.99, selected=True)],
        ref_ids=["C"],
        target_ids=["U"],
    )
    human = _labels([_label_row("hg1", [("A", "T"), ("B", "T"), ("C", "U")])])

    df = build_edge_table([g1, g2], human, "ds")
    stats = df.attrs["build_stats"]
    assert set(df["provenance"]) == {"split"}
    assert stats["human_selected_outside_candidate_graph"] == 1
    assert stats["human_selected_outside_candidate_graph_clean"] == 0
    assert stats["human_selected_outside_candidate_graph_split"] == 1
    assert stats["legacy_known_omission_occurrences_split"] == 1
    assert stats["legacy_known_omission_unique_raw_keys_split"] == 1
    assert stats["legacy_known_omission_unique_retained_keys_split"] == 1


def test_legacy_known_omissions_separate_occurrences_from_unique_keys():
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, selected=True)],
        ref_ids=["A"],
        target_ids=["T"],
        edges=[_edge("A", "T", 0.99)],
    )
    grp["rejected_edges"] = [_edge("B", "T", 0.2, selected=False)]
    human = _labels(
        [
            _label_row("historical-1", [("A", "T"), ("B", "T")]),
            _label_row("historical-2", [("A", "T"), ("B", "T")]),
        ]
    )

    df = build_edge_table([grp], human, "ds")

    stats = df.attrs["build_stats"]
    assert stats["legacy_known_omission_occurrences"] == 2
    assert stats["legacy_known_omission_unique_raw_keys"] == 1
    assert stats["legacy_known_omission_unique_retained_keys"] == 1
    assert df.attrs["build_audit"]["legacy_known_omissions"] == [
        {
            "case_id": "legacy_known_omission:ds:B:T:historical-1+historical-2",
            "dataset_id": "ds",
            "current_group_id": "g1",
            "ref_id": "B",
            "target_id": "T",
            "ref_name": "",
            "target_name": "",
            "retained_after_quarantine": True,
            "occurrence_count": 2,
            "occurrences": [
                {"human_group_id": "historical-1", "provenance": "clean"},
                {"human_group_id": "historical-2", "provenance": "clean"},
            ],
        }
    ]
