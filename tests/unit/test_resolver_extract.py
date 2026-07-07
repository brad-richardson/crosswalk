"""Unit tests for the experimental learned-group-resolver extraction harness.

Uses tiny synthetic sidecar groups + labels (no dependency on data/), so these
run in CI without the untracked runtime data. They pin the data contract:
label = edge in human selected set, provenance clean vs split, and that
featurize produces the declared feature columns.
"""

from __future__ import annotations

import json

import pandas as pd

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
        re.compile(r"^\s*import\s+matcher\.resolver\b", re.MULTILINE),
        re.compile(r"^\s*from\s+matcher\.resolver\b", re.MULTILINE),
        re.compile(r"^\s*from\s+\.+resolver\b", re.MULTILINE),
        re.compile(r"^\s*from\s+\.+\s+import\s+(?:[^\n]*[\s,(])?resolver\b", re.MULTILINE),
    ]

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "matcher"
    offenders = []
    for py in src.rglob("*.py"):
        if "resolver" in py.parts:
            continue
        text = py.read_text()
        if any(p.search(text) for p in patterns):
            offenders.append(str(py.relative_to(src)))
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


# --- #344 candidate_edges (design §3 stage-1 consumer) ----------------------


def _cand(ref, tgt, conf, selected, selected_elsewhere=False):
    """Minimal stage-1 candidate_edges record (topology + confidence + flags)."""
    d = {"ref_id": ref, "target_id": tgt, "confidence": conf, "selected": selected}
    if selected_elsewhere:
        d["selected_elsewhere"] = True
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
