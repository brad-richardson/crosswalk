"""Unit tests for the experimental learned-group-resolver extraction harness.

Uses tiny synthetic sidecar groups + labels (no dependency on data/), so these
run in CI without the untracked runtime data. They pin the data contract:
label = edge in human selected set, provenance clean vs split, and that
featurize produces the declared feature columns.
"""

from __future__ import annotations

import json

import pandas as pd

from matcher.resolver.extract import build_edge_table, load_sidecar_groups
from matcher.resolver.features import FEATURE_COLUMNS, featurize


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
    matching / features / cli / optimizer paths may import matcher.resolver.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "matcher"
    offenders = []
    for py in src.rglob("*.py"):
        if "resolver" in py.parts:
            continue
        if "matcher.resolver" in py.read_text():
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
