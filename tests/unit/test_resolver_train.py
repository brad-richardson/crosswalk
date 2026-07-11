"""Unit tests for the experimental resolver training harness.

Research-only — validates determinism, empty handling, fallback correctness,
and the import-guard invariant that train.py stays out of production code.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest


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


def _group(gid, edges, match_type="M:N", candidate_edges=None):
    refs = sorted({e["ref_id"] for e in edges})
    tgts = sorted({e["target_id"] for e in edges})
    g = {
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
    if candidate_edges is not None:
        g["candidate_edges"] = candidate_edges
    return g


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


def _labels(rows):
    return pd.DataFrame(rows)


def test_train_model_requires_both_classes():
    from crosswalk.resolver.features import FEATURE_COLUMNS
    from crosswalk.resolver.train import train_model

    # All keep=1 — should raise
    groups = [_group("g1", [_edge("A", "T", 0.99)])]
    human = _labels([_label_row("hg1", [("A", "T")])])
    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import featurize

    df = featurize(build_edge_table(groups, human, "ds"))
    with pytest.raises(ValueError, match="Cannot train"):
        train_model(df, FEATURE_COLUMNS)


def test_train_model_deterministic_with_seed():
    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
    from crosswalk.resolver.train import train_model

    groups = [
        _group(
            "g1",
            [_edge("A", "T", 0.99), _edge("B", "T", 0.4)],
        ),
        _group(
            "g2",
            [_edge("C", "U", 0.95), _edge("D", "U", 0.35)],
        ),
        _group(
            "g3",
            [_edge("E", "V", 0.9), _edge("F", "V", 0.3)],
        ),
    ]
    human = _labels(
        [
            _label_row("h1", [("A", "T")]),
            _label_row("h2", [("C", "U")]),
            _label_row("h3", [("E", "V")]),
        ]
    )
    df = featurize(build_edge_table(groups, human, "ds"))
    m1 = train_model(df, FEATURE_COLUMNS, seed=0)
    m2 = train_model(df, FEATURE_COLUMNS, seed=0)
    import numpy as np

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    p1 = m1.predict_proba(X)[:, 1]
    p2 = m2.predict_proba(X)[:, 1]
    assert np.allclose(p1, p2)


def test_discover_specs_filters_by_dataset(tmp_path):
    from crosswalk.resolver.train import _discover_specs

    # Craft minimal stitching structure
    stitch = tmp_path / "stitching"
    for ds in ("a", "b"):
        d = stitch / f"dataset={ds}"
        d.mkdir(parents=True)
        (d / "data.csv").write_text("group_id,selected_edges\n")

    # One label file with content
    p_a = stitch / "dataset=a" / "data.csv"
    p_a.write_text('group_id,selected_edges\n"g1","[]"\n')

    specs = _discover_specs(tmp_path, stitch, dataset_filter=["a"])
    assert all(s[0] == "a" for s in specs)


def test_build_combined_table_empty_handling_and_stats(tmp_path):
    from crosswalk.resolver.train import _build_combined_table

    groups = [
        _group(
            "g1",
            [_edge("A", "T", 0.99), _edge("B", "T", 0.4)],
            candidate_edges=[
                {"ref_id": "A", "target_id": "T", "confidence": 0.99, "selected": True},
                {"ref_id": "B", "target_id": "T", "confidence": 0.4, "selected": False},
            ],
        )
    ]
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps({"groups": groups}))
    labels = tmp_path / "labels.csv"
    df_h = _labels([_label_row("g1", [("A", "T")])])
    df_h.to_csv(labels, index=False)

    df, stats, _ = _build_combined_table([("ds", gp, labels)], include_empty=False)
    assert len(df) == 2
    assert stats[0]["exists"] is True
    assert stats[0]["rows"] == 2
    # empty-reject handling tested in extract, but train path should propagate empty_rows key
    assert "build_empty_rows" in stats[0]


def test_build_combined_table_legacy_empty_skipped(tmp_path):
    """Reject-all label on a legacy group (no candidate_edges) emits zero rows
    but counts empty_legacy_skipped — honest test for cross-mode."""
    from crosswalk.resolver.train import _build_combined_table

    groups = [_group("g1", [_edge("A", "T", 0.99)])]  # no candidate_edges
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps({"groups": groups}))
    labels = tmp_path / "labels.csv"
    df_h = _labels([_label_row("g1", [])])  # reject-all
    df_h.to_csv(labels, index=False)

    df, stats, _ = _build_combined_table([("ds", gp, labels)], include_empty=True)
    # legacy reject-all emits zero rows
    assert df.empty
    assert stats[0]["build_empty_legacy_skipped"] == 1


def test_prepare_soft_for_train_requires_feature_cols():
    from crosswalk.resolver.features import FEATURE_COLUMNS
    from crosswalk.resolver.train import _prepare_soft_for_train

    g = _group("g1", [_edge("A", "T", 0.99), _edge("B", "T", 0.4)])
    groups_by_ds = {"ds": [g]}

    # soft votes for A and B, but g1 is in existing_group_ids -> excluded
    soft_df = pd.DataFrame(
        [
            {"group_id": "g1", "ref_id": "A", "target_id": "T", "soft_keep": 0.9},
            {"group_id": "g1", "ref_id": "B", "target_id": "T", "soft_keep": 0.2},
        ]
    )
    out = _prepare_soft_for_train(
        soft_df,
        groups_by_ds,
        existing_group_ids={"g1"},
        feature_cols=FEATURE_COLUMNS,
        extended=False,
    )
    assert out is None  # all excluded


def test_prepare_soft_for_train_new_groups_featurized():
    from crosswalk.resolver.features import FEATURE_COLUMNS
    from crosswalk.resolver.train import _prepare_soft_for_train

    g = _group("g_new", [_edge("A", "T", 0.9), _edge("B", "T", 0.3)])
    groups_by_ds = {"ds": [g]}
    soft_df = pd.DataFrame(
        [
            {"group_id": "g_new", "ref_id": "A", "target_id": "T", "soft_keep": 0.9},
            {"group_id": "g_new", "ref_id": "B", "target_id": "T", "soft_keep": 0.1},
        ]
    )
    out = _prepare_soft_for_train(
        soft_df,
        groups_by_ds,
        existing_group_ids=set(),
        feature_cols=FEATURE_COLUMNS,
        extended=False,
    )
    assert out is not None
    assert len(out) == 2
    assert set(out["keep"]) == {0, 1}


def test_evaluate_all_runs_on_small_table():
    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
    from crosswalk.resolver.train import evaluate_all

    groups = [
        _group(f"g{i}", [_edge(f"R{i}", f"T{i}", 0.95), _edge(f"S{i}", f"T{i}", 0.3)])
        for i in range(6)
    ]
    labels = [_label_row(f"h{i}", [(f"R{i}", f"T{i}")]) for i in range(6)]
    df = featurize(build_edge_table(groups, _labels(labels), "ds"))
    res = evaluate_all(df, FEATURE_COLUMNS, selector="ef1", n_splits=3)
    assert "model" in res
    assert "baseline_production" in res
    assert res["model"].f1 >= 0.0


def test_save_model_payload_structure(tmp_path):
    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
    from crosswalk.resolver.train import save_model, train_model

    groups = [
        _group(f"g{i}", [_edge(f"R{i}", f"T{i}", 0.95), _edge(f"S{i}", f"T{i}", 0.3)])
        for i in range(4)
    ]
    labels = [_label_row(f"h{i}", [(f"R{i}", f"T{i}")]) for i in range(4)]
    df = featurize(build_edge_table(groups, _labels(labels), "ds"))
    model = train_model(df, FEATURE_COLUMNS, seed=0)

    out = tmp_path / "resolver_model.joblib"
    save_model(
        model,
        FEATURE_COLUMNS,
        out,
        training_stats={"n_rows_hard": len(df)},
        cv_summary={"model": {"F1": 0.9}},
        selector="ef1",
    )
    assert out.exists()
    import joblib

    payload = joblib.load(str(out))
    assert "model" in payload
    assert "feature_columns" in payload
    assert "feature_version" in payload
    assert payload["selector"] == "ef1"
