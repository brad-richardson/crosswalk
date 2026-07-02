"""Evaluation-integrity tests for MLMatcher.train().

These tests guard against three leakage bugs:
1. In-training cross-validation folds must contain only training rows,
   never holdout test rows (otherwise cv_f1_mean is not independent of
   test_accuracy).
2. Agent labels (agent_weight > 0) must never appear in the holdout test
   set or CV folds — they are training-only signal, not ground truth.
3. Labels with stale feature_version must raise unless explicitly allowed
   (see tests/unit/test_model_version.py for those tests).
"""

import numpy as np
import pandas as pd
import pytest

import matcher.matching.ml as ml_module
from matcher.config import FEATURE_COLUMNS, FEATURE_VERSION
from matcher.matching.ml import MLMatcher

pytest.importorskip("xgboost")

# First feature column is used as a unique per-row identifier so rows can be
# traced through feature matrices. FEATURE_COLUMNS[0] is hausdorff_distance_m;
# ids must stay below the max_hausdorff_m=1000 validation threshold.
ID_FEATURE = FEATURE_COLUMNS[0]
AGENT_ID_OFFSET = 500

# Small/fast XGBoost params for tests
FAST_XGB = {"n_estimators": 5, "max_depth": 2}


def _write_features(features_dir, dataset, rows):
    """Write a features parquet for the given (gers_id, target_id, row_id) rows."""
    rng = np.random.default_rng(7)
    part_dir = features_dir / f"dataset={dataset}"
    part_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "gers_id": [r[0] for r in rows],
        "target_id": [r[1] for r in rows],
        "feature_version": [FEATURE_VERSION] * len(rows),
    }
    for col in FEATURE_COLUMNS:
        data[col] = rng.random(len(rows)).tolist()
    # Overwrite the id feature with the unique row id
    data[ID_FEATURE] = [float(r[2]) for r in rows]
    pd.DataFrame(data).to_parquet(part_dir / "data.parquet", index=False)


def _make_labels_dir(tmp_path, n_human=40, n_agent=0, agent_shares_human_gers=False):
    """Create a normalized labels directory with human (and optional agent) labels.

    Human pair i: gers G{i} <-> target HT{i}, row id = i.
    Agent pair i: target AT{i}, row id = AGENT_ID_OFFSET + i. If
    agent_shares_human_gers, agent pair i reuses gers G{i} (same segment
    group as human pair i); otherwise it gets its own gers AG{i}.
    """
    labels_dir = tmp_path / "labels"
    dataset = "test_ds"

    human_rows = [(f"G{i}", f"HT{i}", i) for i in range(n_human)]
    human_dir = labels_dir / "human" / f"dataset={dataset}"
    human_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "gers_id": [r[0] for r in human_rows],
            "target_id": [r[1] for r in human_rows],
            "label": ["match" if i % 2 == 0 else "no_match" for i in range(n_human)],
            "labeler": ["human"] * n_human,
            "labeled_at": ["2026-01-01T00:00:00"] * n_human,
            "session_id": ["s"] * n_human,
        }
    ).to_csv(human_dir / "data.csv", index=False)

    agent_rows = []
    if n_agent > 0:
        gers_prefix = "G" if agent_shares_human_gers else "AG"
        agent_rows = [(f"{gers_prefix}{i}", f"AT{i}", AGENT_ID_OFFSET + i) for i in range(n_agent)]
        agent_dir = labels_dir / "agent" / f"dataset={dataset}"
        agent_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "gers_id": [r[0] for r in agent_rows],
                "target_id": [r[1] for r in agent_rows],
                "label": ["match" if i % 2 == 0 else "no_match" for i in range(n_agent)],
                "confidence": [0.9] * n_agent,
                "reasoning": ["r"] * n_agent,
                "labeler": ["agent"] * n_agent,
                "labeled_at": ["2026-01-01T00:00:00"] * n_agent,
            }
        ).to_csv(agent_dir / "data.csv", index=False)

    _write_features(labels_dir / "features", dataset, human_rows + agent_rows)
    return labels_dir


@pytest.fixture
def spy_split(monkeypatch):
    """Capture the DataFrame and result of the segment_aware_split call in train()."""
    captured = {}
    real_split = ml_module.segment_aware_split

    def _spy(df, *args, **kwargs):
        result = real_split(df, *args, **kwargs)
        captured["df"] = df.copy()
        captured["train_idx"], captured["test_idx"] = result[0], result[1]
        return result

    monkeypatch.setattr(ml_module, "segment_aware_split", _spy)
    return captured


@pytest.fixture
def spy_groupkfold(monkeypatch):
    """Capture the (X, y, groups) arrays passed to GroupKFold.split in train()."""
    calls = []
    real_cls = ml_module.GroupKFold

    class RecordingGroupKFold(real_cls):
        def split(self, X, y=None, groups=None):
            calls.append(
                {
                    "X": np.asarray(X).copy(),
                    "y": np.asarray(y).copy(),
                    "groups": np.asarray(groups).copy(),
                }
            )
            return super().split(X, y, groups)

    monkeypatch.setattr(ml_module, "GroupKFold", RecordingGroupKFold)
    return calls


def _cv_row_ids(matcher, cv_call):
    """Extract the unique row ids (ID_FEATURE values) seen by the CV split."""
    id_col = matcher.feature_names.index(ID_FEATURE)
    return set(cv_call["X"][:, id_col].astype(int).tolist())


class TestCVExcludesTestRows:
    """Bug 1: in-training CV must run over training rows only."""

    def test_cv_rows_are_exactly_train_rows(self, tmp_path, spy_split, spy_groupkfold):
        labels_dir = _make_labels_dir(tmp_path, n_human=40)

        matcher = MLMatcher()
        results = matcher.train(labels_dir=str(labels_dir), test_size=0.3, **FAST_XGB)

        assert results["n_test"] > 0, "Fixture must produce a non-empty test set"
        assert len(spy_groupkfold) == 1, "CV should run exactly once"

        df = spy_split["df"]
        train_ids = set(df.iloc[spy_split["train_idx"]][ID_FEATURE].astype(int))
        test_ids = set(df.iloc[spy_split["test_idx"]][ID_FEATURE].astype(int))
        cv_ids = _cv_row_ids(matcher, spy_groupkfold[0])

        assert cv_ids == train_ids, "CV must see exactly the training rows"
        assert cv_ids.isdisjoint(test_ids), f"CV folds contain test rows: {cv_ids & test_ids}"

    def test_cv_groups_subset_matches_train_rows(self, tmp_path, spy_split, spy_groupkfold):
        """The group array passed to GroupKFold must align with the train rows."""
        labels_dir = _make_labels_dir(tmp_path, n_human=40)

        matcher = MLMatcher()
        matcher.train(labels_dir=str(labels_dir), test_size=0.3, **FAST_XGB)

        call = spy_groupkfold[0]
        assert len(call["groups"]) == len(spy_split["train_idx"])
        assert len(call["X"]) == len(spy_split["train_idx"])
        assert len(call["y"]) == len(spy_split["train_idx"])

    def test_cv_metrics_keys_unchanged(self, tmp_path):
        """Downstream consumers rely on these result keys."""
        labels_dir = _make_labels_dir(tmp_path, n_human=40)

        matcher = MLMatcher()
        results = matcher.train(labels_dir=str(labels_dir), test_size=0.3, **FAST_XGB)

        for key in (
            "n_train",
            "n_test",
            "test_accuracy",
            "cv_f1_mean",
            "cv_f1_std",
            "classification_report",
            "confusion_matrix",
            "feature_importance",
        ):
            assert key in results


class TestAgentLabelsNeverInTestSet:
    """Bug 2: agent labels must be training-only, never holdout ground truth."""

    def test_split_computed_on_human_labels_only(self, tmp_path, spy_split):
        labels_dir = _make_labels_dir(tmp_path, n_human=30, n_agent=10)

        matcher = MLMatcher()
        matcher.train(labels_dir=str(labels_dir), test_size=0.3, agent_weight=0.5, **FAST_XGB)

        df = spy_split["df"]
        assert len(df) == 30, "Split must be computed on human labels only"
        assert not df["target_id"].str.startswith("AT").any(), (
            "Agent-labeled pairs leaked into the train/test split input"
        )

    def test_no_agent_pair_in_test_set(self, tmp_path, spy_split):
        labels_dir = _make_labels_dir(tmp_path, n_human=30, n_agent=10)

        matcher = MLMatcher()
        results = matcher.train(
            labels_dir=str(labels_dir), test_size=0.3, agent_weight=0.5, **FAST_XGB
        )

        df = spy_split["df"]
        test_rows = df.iloc[spy_split["test_idx"]]
        assert len(test_rows) > 0
        assert not test_rows["target_id"].str.startswith("AT").any(), (
            "Agent-labeled pairs appeared in the holdout test set"
        )
        # Agent labels should have been added to the training portion
        assert results["n_train"] > len(spy_split["train_idx"])
        assert results["n_train"] <= len(spy_split["train_idx"]) + 10

    def test_agent_pairs_sharing_test_segments_are_dropped(self, tmp_path, spy_split):
        """Agent pairs that share a segment with a test pair must not be trained on."""
        labels_dir = _make_labels_dir(
            tmp_path, n_human=30, n_agent=30, agent_shares_human_gers=True
        )

        matcher = MLMatcher()
        results = matcher.train(
            labels_dir=str(labels_dir), test_size=0.3, agent_weight=0.5, **FAST_XGB
        )

        df = spy_split["df"]
        train_idx, test_idx = spy_split["train_idx"], spy_split["test_idx"]
        test_gers = set(df.iloc[test_idx]["gers_id"])
        # Agent pair i shares gers G{i} with human pair i, so exactly the agent
        # pairs whose gers is in the training set should survive
        n_agent_kept = sum(1 for i in range(30) if f"G{i}" not in test_gers)
        assert results["n_train"] == len(train_idx) + n_agent_kept
        assert n_agent_kept < 30, "Fixture must exercise the overlap-drop path"

    def test_cv_excludes_agent_rows(self, tmp_path, spy_groupkfold):
        labels_dir = _make_labels_dir(tmp_path, n_human=30, n_agent=10)

        matcher = MLMatcher()
        matcher.train(labels_dir=str(labels_dir), test_size=0.3, agent_weight=0.5, **FAST_XGB)

        cv_ids = _cv_row_ids(matcher, spy_groupkfold[0])
        agent_ids = {AGENT_ID_OFFSET + i for i in range(10)}
        assert cv_ids.isdisjoint(agent_ids), "Agent rows leaked into CV folds"

    def test_agent_weight_zero_unchanged(self, tmp_path, spy_split):
        """Default agent_weight=0.0 must behave exactly as before (human-only)."""
        labels_dir = _make_labels_dir(tmp_path, n_human=30, n_agent=10)

        matcher = MLMatcher()
        results = matcher.train(labels_dir=str(labels_dir), test_size=0.3, **FAST_XGB)

        df = spy_split["df"]
        assert len(df) == 30
        assert results["n_train"] == len(spy_split["train_idx"])
        assert results["n_train"] + results["n_test"] == 30
