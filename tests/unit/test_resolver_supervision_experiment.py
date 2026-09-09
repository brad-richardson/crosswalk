"""Regression tests for weighted fractional targets and source-pinned context."""

import json

import numpy as np
import pandas as pd
import pytest

from crosswalk.agent_labeling.stitch_provenance import sha256_file
from crosswalk.resolver.evaluation_set import connect_frozen_scopes, observation_tokens
from crosswalk.resolver.round2 import TRAIN_LABEL_COLUMN
from crosswalk.resolver.supervision_experiment import (
    prepare_verified_weak_context,
    weight_weak_rows,
)
from crosswalk.resolver.train import predict_keep_probability, train_model

SETTINGS = {"unanimous_weight": 0.35, "majority_weight": 0.15, "weak_component_weight_cap": 4.0}


def weak_row(ref="A", **updates):
    return {
        "dataset_id": "ds",
        "group_id": "parent__p0000000001",
        "parent_group_id": "parent",
        "ref_id": ref,
        "target_id": "T",
        "component_id": "component",
        "source_snapshot_id": "snapshot",
        "vote_tier": "majority",
        "known": True,
        "excluded_human_scope": False,
        "soft_keep": 2 / 3,
        TRAIN_LABEL_COLUMN: 2 / 3,
        **updates,
    }


def test_repeated_versions_share_one_pair_budget_and_children_have_component_cap():
    one = pd.DataFrame([weak_row()])
    repeated = pd.concat([one] * 30, ignore_index=True)
    assert weight_weak_rows(
        repeated, "human_majority", SETTINGS
    ).sample_weight.sum() == pytest.approx(0.15)
    children = pd.DataFrame(
        [weak_row(str(i), group_id=f"parent__p{i:010x}", vote_tier="unanimous") for i in range(100)]
    )
    weighted = weight_weak_rows(children, "human_unanimous", SETTINGS)
    assert weighted.sample_weight.sum() == pytest.approx(4.0)
    assert weight_weak_rows(one, "human_only", SETTINGS).empty
    assert weight_weak_rows(one, "human_unanimous", SETTINGS).empty
    assert weight_weak_rows(one, "human_majority", SETTINGS)[TRAIN_LABEL_COLUMN].iloc[0] == 2 / 3


def test_mixed_tier_versions_keep_the_strongest_pair_budget():
    """A unanimous version plus a majority version of one physical pair must not
    weigh less under human_majority than the unanimous version alone did under
    human_unanimous; the lower tier splits the budget, it does not shrink it."""
    versions = pd.DataFrame(
        [
            weak_row(vote_tier="unanimous", soft_keep=1.0, **{TRAIN_LABEL_COLUMN: 1.0}),
            weak_row(group_id="parent__p0000000002"),
        ]
    )
    unanimous_only = weight_weak_rows(versions, "human_unanimous", SETTINGS)
    both = weight_weak_rows(versions, "human_majority", SETTINGS)
    assert unanimous_only.sample_weight.sum() == pytest.approx(0.35)
    assert both.sample_weight.sum() == pytest.approx(0.35)
    assert both.sample_weight.tolist() == pytest.approx([0.175, 0.175])


def test_unknown_and_human_related_rows_never_receive_loss_weight():
    frame = pd.DataFrame(
        [
            weak_row(known=False),
            weak_row(excluded_human_scope=True),
            weak_row(vote_tier="below_quorum"),
        ]
    )
    assert weight_weak_rows(frame, "human_majority", SETTINGS).empty


def test_fractional_targets_and_weights_reach_logistic_loss_without_hardening(monkeypatch):
    import xgboost

    captured = {}

    class Regressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, x, y, sample_weight):
            captured.update(targets=y, weights=sample_weight)

    monkeypatch.setattr(xgboost, "XGBRegressor", Regressor)
    human = pd.DataFrame({"x": [0, 1], "keep": [0, 1]})
    extra = pd.DataFrame({"x": [0.5], "keep": [2 / 3], "sample_weight": [0.15]})
    train_model(human, ["x"], extra)
    np.testing.assert_array_equal(captured["targets"], [0, 1, 2 / 3])
    np.testing.assert_array_equal(captured["weights"], [1, 1, 0.15])
    assert captured["objective"] == "reg:logistic"

    def unavailable(**kwargs):
        raise RuntimeError("regressor failed")

    monkeypatch.setattr(xgboost, "XGBRegressor", unavailable)
    with pytest.raises(RuntimeError, match="regressor failed"):
        train_model(human, ["x"], extra)


def test_weighted_loss_changes_conflicting_evidence_in_expected_direction():
    # Identical features isolate the supervision contract from tree split details.
    frame = pd.DataFrame(
        {
            "x": np.zeros(40),
            "keep": [0.2] * 20 + [0.8] * 20,
            "sample_weight": [1.0] * 20 + [0.1] * 20,
        }
    )
    low = predict_keep_probability(train_model(frame, ["x"]), np.zeros((1, 1)))[0]
    frame["sample_weight"] = [0.1] * 20 + [1.0] * 20
    high = predict_keep_probability(train_model(frame, ["x"]), np.zeros((1, 1)))[0]
    assert 0.2 < low < 0.4 < 0.6 < high < 0.8


@pytest.mark.parametrize(
    "field,value",
    [("keep", float("nan")), ("keep", 1.1), ("sample_weight", -1), ("sample_weight", float("nan"))],
)
def test_invalid_targets_and_weights_fail_before_training(field, value):
    frame = pd.DataFrame({"x": [0, 1], "keep": [0, 1], "sample_weight": [1, 1]})
    frame[field] = frame[field].astype(float)
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        train_model(frame, ["x"])


def test_full_unvoted_context_propagates_human_exclusion_and_shared_budget():
    human = {"dataset_id": "ds", "group_id": "gold", "ref_id": "HUMAN", "target_id": "GOLD"}
    index = {token: "heldout" for token in observation_tokens(human)}
    frame = pd.DataFrame(
        [
            weak_row(context_scope_tokens='["ref:HUMAN", "ref:UNVOTED"]'),
            weak_row(
                "B",
                group_id="other",
                parent_group_id="other",
                context_scope_tokens='["ref:UNVOTED"]',
            ),
        ]
    )
    connected = connect_frozen_scopes(frame, index, {"heldout"})
    assert connected.excluded_human_scope.all()
    assert connected.component_id.nunique() == 1


def test_verified_context_rejects_changed_snapshot_and_featurizes_unvoted_edges(tmp_path):
    edges = [
        {"ref_id": r, "target_id": "T", "confidence": conf, "selected": True}
        for r, conf in [("A", 0.9), ("B", 0.7)]
    ]
    group = {
        "group_id": "parent",
        "ref_ids": ["A", "B"],
        "target_ids": ["T"],
        "candidate_edges": edges,
        "edges": edges,
    }
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"groups": [group]}))
    candidates = tmp_path / "candidates.parquet"
    pd.DataFrame([{**edge, "group_id": "parent"} for edge in edges]).to_parquet(candidates)
    artifacts = json.dumps(
        {
            "groups_sidecar": {"path": str(path), "sha256": sha256_file(path)},
            "candidates_parquet": {"path": str(candidates), "sha256": sha256_file(candidates)},
        }
    )
    weak = pd.DataFrame([weak_row(source_artifacts=artifacts)])
    prepared, audit = prepare_verified_weak_context(weak, {}, [])
    assert len(prepared) == 1
    assert prepared.iloc[0]["n_share_tgt"] == 2  # B was context, not a negative target
    assert prepared.iloc[0]["conf_rel_mean"] == pytest.approx(0.1)
    assert prepared.iloc[0][TRAIN_LABEL_COLUMN] == 2 / 3
    assert audit["counts"]["eligible_training_rows"] == 1
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="verified source snapshot"):
        prepare_verified_weak_context(weak, {}, [])
