"""Unit tests for the experimental resolver training harness.

Research-only — validates determinism, empty handling, fallback correctness,
and the import-guard invariant that train.py stays out of production code.
"""

from __future__ import annotations

import json

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


def test_smoothed_training_target_keeps_hard_truth_and_predicts_probability():
    import numpy as np

    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
    from crosswalk.resolver.round2 import TRAIN_LABEL_COLUMN
    from crosswalk.resolver.train import predict_keep_probability, train_model

    groups = [
        _group(f"g{i}", [_edge(f"R{i}", f"T{i}", 0.95), _edge(f"S{i}", f"T{i}", 0.3)])
        for i in range(4)
    ]
    labels = [_label_row(f"h{i}", [(f"R{i}", f"T{i}")]) for i in range(4)]
    df = featurize(build_edge_table(groups, _labels(labels), "ds"))
    hard_truth = df["keep"].copy()
    df[TRAIN_LABEL_COLUMN] = np.where(df["keep"] == 1, 0.95, 0.05)

    model = train_model(df, FEATURE_COLUMNS, seed=0)
    probability = predict_keep_probability(model, df[FEATURE_COLUMNS].to_numpy(dtype=float))

    assert df["keep"].equals(hard_truth)
    assert model.get_xgb_params()["objective"] == "reg:logistic"
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


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
    pd.DataFrame(
        [
            {
                "group_id": "g1",
                "ref_id": "A",
                "target_id": "T",
                "lateral_offset_signed_m": 1.25,
            },
            {
                "group_id": "g1",
                "ref_id": "B",
                "target_id": "T",
                "lateral_offset_signed_m": -2.5,
            },
        ]
    ).to_parquet(tmp_path / "g_candidates.parquet")
    labels = tmp_path / "labels.csv"
    df_h = _labels([_label_row("g1", [("A", "T")])])
    df_h.to_csv(labels, index=False)

    df, stats, _ = _build_combined_table([("ds", gp, labels)], include_empty=False)
    assert len(df) == 2
    assert stats[0]["exists"] is True
    assert stats[0]["rows"] == 2
    assert stats[0]["build_candidate_parquet_enriched"] == 2
    assert stats[0]["build_candidate_parquet_missing_keys"] == 0
    assert list(df.sort_values("ref_id")["lateral_offset_signed_m"]) == [1.25, -2.5]
    # empty-reject handling tested in extract, but train path should propagate empty_rows key
    assert "build_empty_rows" in stats[0]


def test_build_combined_table_surfaces_duplicate_label_audit_stats(tmp_path):
    from crosswalk.resolver.train import _build_combined_table

    edges = [_edge("A", "T", 0.99), _edge("B", "T", 0.4, selected=False)]
    groups = [_group("g1", edges, candidate_edges=edges)]
    groups_path = tmp_path / "g.json"
    groups_path.write_text(json.dumps({"groups": groups}))
    labels_path = tmp_path / "labels.csv"
    _labels(
        [
            _label_row("historical-1", [("A", "T")]),
            _label_row("historical-2", [("A", "T")]),
        ]
    ).to_csv(labels_path, index=False)

    df, per_dataset, _ = _build_combined_table([("ds", groups_path, labels_path)])

    assert len(df) == 2
    assert per_dataset[0]["build_raw_rows"] == 4
    assert per_dataset[0]["build_duplicate_rows"] == 2
    assert per_dataset[0]["build_duplicate_keys"] == 2
    assert per_dataset[0]["build_conflicting_keys"] == 0
    assert per_dataset[0]["build_quarantined_groups"] == 0
    assert per_dataset[0]["build_quarantined_rows"] == 0


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
            {
                "dataset_id": "ds",
                "group_id": "g1",
                "ref_id": "A",
                "target_id": "T",
                "soft_keep": 0.9,
            },
            {
                "dataset_id": "ds",
                "group_id": "g1",
                "ref_id": "B",
                "target_id": "T",
                "soft_keep": 0.2,
            },
        ]
    )
    out = _prepare_soft_for_train(
        soft_df,
        groups_by_ds,
        existing_group_ids={("ds", "g1")},
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
            {
                "dataset_id": "ds",
                "group_id": "g_new",
                "ref_id": "A",
                "target_id": "T",
                "soft_keep": 0.9,
            },
            {
                "dataset_id": "ds",
                "group_id": "g_new",
                "ref_id": "B",
                "target_id": "T",
                "soft_keep": 0.1,
            },
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
    assert set(out["dataset_id"]) == {"ds"}


def test_edge_soft_labels_are_dataset_scoped_and_observed_only():
    from crosswalk.resolver.votes import edge_soft_labels

    groups = [
        _group(
            "g1",
            [_edge("A", "T", 0.9)],
            candidate_edges=[
                {"ref_id": "A", "target_id": "T"},
                {"ref_id": "B", "target_id": "T"},
                {"ref_id": "C", "target_id": "T"},
            ],
        )
    ]
    votes = pd.DataFrame(
        [
            {
                "dataset_id": "ds",
                "group_id": "old",
                "provider": "claude",
                "edge_set": json.dumps([["A", "T"]]),
            },
            {
                "dataset_id": "ds",
                "group_id": "old",
                "provider": "codex",
                "edge_set": json.dumps([["A", "T"], ["B", "T"]]),
            },
            {
                "dataset_id": "other",
                "group_id": "old",
                "provider": "claude",
                "edge_set": json.dumps([["C", "T"]]),
            },
        ]
    )

    soft = edge_soft_labels(groups, votes, dataset_id="ds")

    assert set(zip(soft["ref_id"], soft["target_id"])) == {("A", "T"), ("B", "T")}
    assert "C" not in set(soft["ref_id"])
    by_ref = soft.set_index("ref_id")
    assert by_ref.loc["A", "soft_keep"] == pytest.approx(1.0)
    assert by_ref.loc["B", "soft_keep"] == pytest.approx(1 / 3)
    assert by_ref.loc["A", "unanimous"] == 1
    assert by_ref.loc["B", "unanimous"] == 0


def test_load_votes_keeps_dataset_provenance_in_dedup_key(tmp_path):
    from crosswalk.resolver.votes import load_votes

    paths = []
    for dataset_id in ("ds1", "ds2"):
        batch = tmp_path / dataset_id
        batch.mkdir()
        (batch / "batch.json").write_text(json.dumps({"dataset_id": dataset_id}))
        pd.DataFrame(
            [
                {
                    "group_id": "same-hash",
                    "provider": "claude",
                    "edge_set": "[]",
                    "timestamp": "2026-07-12T00:00:00Z",
                    "error": None,
                }
            ]
        ).to_csv(batch / "votes.csv", index=False)
        paths.append(batch / "votes.csv")

    votes = load_votes(paths)

    assert len(votes) == 2
    assert set(votes["dataset_id"]) == {"ds1", "ds2"}


def test_edge_soft_labels_do_not_invent_none_vote_negatives():
    from crosswalk.resolver.votes import edge_soft_labels

    groups = [
        _group(
            "g1",
            [_edge("A", "T", 0.9)],
            candidate_edges=[{"ref_id": "A", "target_id": "T"}],
        )
    ]
    votes = pd.DataFrame(
        [
            {
                "dataset_id": "ds",
                "group_id": "g1",
                "provider": "claude",
                "edge_set": "[]",
            }
        ]
    )

    assert edge_soft_labels(groups, votes, dataset_id="ds").empty


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


def test_evaluate_all_scores_against_hard_truth_when_training_is_smoothed():
    import numpy as np

    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
    from crosswalk.resolver.round2 import TRAIN_LABEL_COLUMN
    from crosswalk.resolver.train import evaluate_all

    groups = [
        _group(f"g{i}", [_edge(f"R{i}", f"T{i}", 0.95), _edge(f"S{i}", f"T{i}", 0.3)])
        for i in range(6)
    ]
    labels = [_label_row(f"h{i}", [(f"R{i}", f"T{i}")]) for i in range(6)]
    df = featurize(build_edge_table(groups, _labels(labels), "ds"))
    smoothed = df.copy()
    smoothed[TRAIN_LABEL_COLUMN] = np.where(smoothed["keep"] == 1, 0.95, 0.05)

    hard = evaluate_all(df, FEATURE_COLUMNS, selector="ef1", n_splits=3, seed=0)
    smooth = evaluate_all(smoothed, FEATURE_COLUMNS, selector="ef1", n_splits=3, seed=0)

    assert smooth["baseline_production"].row() == hard["baseline_production"].row()
    assert smooth["baseline_conf_oracle"].row() == hard["baseline_conf_oracle"].row()

    corrupted_truth = df.copy()
    corrupted_truth["keep"] = np.where(corrupted_truth["keep"] == 1, 0.95, 0.05)
    with pytest.raises(ValueError, match="binary evaluation truth"):
        evaluate_all(corrupted_truth, FEATURE_COLUMNS, selector="ef1", n_splits=3, seed=0)


def test_paired_group_bootstrap_is_group_scoped_and_reproducible():
    import numpy as np

    from crosswalk.resolver.evaluate import paired_group_bootstrap

    df = pd.DataFrame(
        {
            "dataset_id": ["a", "a", "b", "b"],
            # Deliberate cross-dataset ID collision: these are two groups.
            "group_id": ["same", "same", "same", "same"],
            "keep": [1, 0, 1, 0],
            "is_sliver": [False] * 4,
        }
    )
    candidate = np.array([1, 0, 1, 0])
    baseline = np.array([1, 1, 1, 1])

    first = paired_group_bootstrap(
        df, candidate, baseline, n_resamples=100, seed=42, confidence=0.90
    )
    second = paired_group_bootstrap(
        df, candidate, baseline, n_resamples=100, seed=42, confidence=0.90
    )

    assert first == second
    assert first["n_groups"] == 2
    assert first["f1"]["delta"] == pytest.approx(1.0 / 3.0)
    assert first["f1"]["ci_low"] == pytest.approx(1.0 / 3.0)
    assert first["f1"]["ci_high"] == pytest.approx(1.0 / 3.0)
    assert first["group_exact"]["delta"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="predictions must be binary"):
        paired_group_bootstrap(df, np.array([1.0, 0.5, 1.0, 0.0]), baseline)
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_group_bootstrap(df, candidate.reshape(-1, 1), baseline)

    nonbinary_truth = df.copy()
    nonbinary_truth["keep"] = nonbinary_truth["keep"].astype(float)
    nonbinary_truth.loc[0, "keep"] = 0.5
    with pytest.raises(ValueError, match="binary evaluation truth"):
        paired_group_bootstrap(nonbinary_truth, candidate, baseline)

    missing_group = df.copy()
    missing_group.loc[0, "group_id"] = None
    with pytest.raises(ValueError, match="group keys"):
        paired_group_bootstrap(missing_group, candidate, baseline)


def _multi_dataset_resolver_table(n_datasets: int = 3, groups_per_dataset: int = 4):
    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import featurize

    frames = []
    for ds_idx in range(n_datasets):
        dataset_id = f"ds{ds_idx}"
        groups = [
            _group(
                f"g{i}",
                [
                    _edge(f"R{ds_idx}-{i}", f"T{ds_idx}-{i}", 0.95),
                    _edge(f"S{ds_idx}-{i}", f"T{ds_idx}-{i}", 0.30),
                ],
            )
            for i in range(groups_per_dataset)
        ]
        labels = [
            _label_row(f"h{i}", [(f"R{ds_idx}-{i}", f"T{ds_idx}-{i}")])
            for i in range(groups_per_dataset)
        ]
        frames.append(build_edge_table(groups, _labels(labels), dataset_id))
    return featurize(pd.concat(frames, ignore_index=True))


def test_repeated_grouped_cv_returns_paired_uncertainty():
    from crosswalk.resolver.features import FEATURE_COLUMNS
    from crosswalk.resolver.train import evaluate_repeated_grouped_cv

    df = _multi_dataset_resolver_table()
    result = evaluate_repeated_grouped_cv(
        df,
        FEATURE_COLUMNS,
        selector="ef1",
        n_splits=3,
        seeds=(3, 7),
        bootstrap_resamples=50,
        bootstrap_seed=11,
    )

    assert [run["seed"] for run in result["runs"]] == [3, 7]
    assert len(result["oof_prediction"]) == len(df)
    assert result["paired_bootstrap"]["n_groups"] == 12
    assert set(result["run_spread"]) == {
        "f1",
        "group_exact",
        "f1_delta",
        "group_exact_delta",
    }

    invalid_selected = df.copy()
    invalid_selected["selected"] = invalid_selected["selected"].astype(float)
    invalid_selected.loc[0, "selected"] = 0.5
    with pytest.raises(ValueError, match="predictions must be binary"):
        evaluate_repeated_grouped_cv(
            invalid_selected,
            FEATURE_COLUMNS,
            seeds=(3,),
            n_splits=3,
            bootstrap_resamples=10,
        )


def test_leave_one_dataset_out_predicts_every_row_without_dataset_leakage():
    from crosswalk.resolver.features import FEATURE_COLUMNS
    from crosswalk.resolver.train import evaluate_leave_one_dataset_out

    df = _multi_dataset_resolver_table()
    result = evaluate_leave_one_dataset_out(
        df,
        FEATURE_COLUMNS,
        selector="ef1",
        bootstrap_resamples=50,
    )

    assert [row["dataset_id"] for row in result["per_dataset"]] == ["ds0", "ds1", "ds2"]
    assert len(result["oof_prediction"]) == len(df)
    assert result["model"].n_groups == 12
    assert result["paired_bootstrap"]["n_groups"] == 12
    assert result["paired_bootstrap"]["n_resample_units"] == 3
    assert result["paired_bootstrap"]["resample_columns"] == ["dataset_id"]

    with pytest.raises(ValueError, match="at least two datasets"):
        evaluate_leave_one_dataset_out(
            df[df["dataset_id"] == "ds0"],
            FEATURE_COLUMNS,
            bootstrap_resamples=10,
        )


def test_feature_aggregation_scopes_same_group_id_by_dataset():
    from crosswalk.resolver.features import featurize

    base = {
        "group_id": "same-hash",
        "ref_id": "R",
        "target_id": "T",
        "gers_start_frac": 0.0,
        "gers_end_frac": 1.0,
        "local_start_frac": 0.0,
        "local_end_frac": 1.0,
        "degree_ref": 1,
        "degree_tgt": 1,
        "is_bridge": False,
        "is_sliver": False,
        "oversized_group": False,
        "match_type": "1:1",
    }
    df = pd.DataFrame(
        [
            {**base, "dataset_id": "a", "confidence": 0.9},
            {**base, "dataset_id": "b", "confidence": 0.4},
        ]
    )

    out = featurize(df)

    assert list(out["conf_rel_max"]) == [0.0, 0.0]
    assert list(out["n_share_ref"]) == [1, 1]


def test_group_exact_metric_scopes_same_group_id_by_dataset():
    import numpy as np

    from crosswalk.resolver.evaluate import _eval_from_predictions

    df = pd.DataFrame(
        {
            "dataset_id": ["a", "b"],
            "group_id": ["same-hash", "same-hash"],
            "keep": [1, 1],
            "is_sliver": [False, False],
        }
    )

    result = _eval_from_predictions("test", df, np.array([1, 0]))

    assert result.n_groups == 2
    assert result.group_exact_rate == pytest.approx(0.5)


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


def _train_tiny_model():
    from crosswalk.resolver.extract import build_edge_table
    from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
    from crosswalk.resolver.train import train_model

    groups = [
        _group(f"g{i}", [_edge(f"R{i}", f"T{i}", 0.95), _edge(f"S{i}", f"T{i}", 0.3)])
        for i in range(4)
    ]
    labels = [_label_row(f"h{i}", [(f"R{i}", f"T{i}")]) for i in range(4)]
    df = featurize(build_edge_table(groups, _labels(labels), "ds"))
    return train_model(df, FEATURE_COLUMNS, seed=0), FEATURE_COLUMNS, df


def test_save_model_stamps_resolver_feature_version_not_pairwise(tmp_path):
    from crosswalk.config import FEATURE_VERSION
    from crosswalk.resolver.features import RESOLVER_FEATURE_VERSION
    from crosswalk.resolver.train import save_model

    model, feat_cols, df = _train_tiny_model()
    out = tmp_path / "resolver_model.joblib"
    save_model(model, feat_cols, out, training_stats={"n": len(df)})

    import joblib

    payload = joblib.load(str(out))
    # The stamp must be the RESOLVER contract, never the pairwise one (they must
    # differ today, otherwise this assertion is vacuous).
    assert RESOLVER_FEATURE_VERSION != FEATURE_VERSION
    assert payload["feature_version"] == RESOLVER_FEATURE_VERSION


def test_load_model_round_trips_current_stamp(tmp_path):
    from crosswalk.resolver.features import RESOLVER_FEATURE_VERSION
    from crosswalk.resolver.train import load_model, save_model

    model, feat_cols, df = _train_tiny_model()
    out = tmp_path / "resolver_model.joblib"
    save_model(model, feat_cols, out, training_stats={"n": len(df)}, selector="ef1")

    payload = load_model(out)
    assert payload["feature_version"] == RESOLVER_FEATURE_VERSION
    assert payload["selector"] == "ef1"
    assert "model" in payload


def test_load_model_rejects_mismatched_stamp(tmp_path):
    """Old models stamped the pairwise FEATURE_VERSION under feature_version;
    those (and any other stale stamp) must fail loudly, not load silently."""
    import joblib

    from crosswalk.resolver.train import load_model

    out = tmp_path / "stale_resolver_model.joblib"
    joblib.dump(
        {"model": object(), "feature_columns": [], "feature_version": "2026-07-07.2"}, str(out)
    )

    with pytest.raises(ValueError, match="RESOLVER_FEATURE_VERSION"):
        load_model(out)

    # Escape hatch downgrades to a warning and still returns the payload.
    payload = load_model(out, allow_version_mismatch=True)
    assert payload["feature_version"] == "2026-07-07.2"


def test_load_model_rejects_missing_stamp(tmp_path):
    import joblib

    from crosswalk.resolver.train import load_model

    out = tmp_path / "unversioned_resolver_model.joblib"
    joblib.dump({"model": object(), "feature_columns": []}, str(out))

    with pytest.raises(ValueError, match="pre-versioning"):
        load_model(out)


def test_load_model_missing_file_raises(tmp_path):
    from crosswalk.resolver.train import load_model

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")
