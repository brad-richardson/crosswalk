"""Tests for the round-2 resolver experiment machinery (features + selector)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crosswalk.resolver.extract import build_edge_table
from crosswalk.resolver.round2 import (
    EXTENDED_FEATURE_COLUMNS,
    featurize_extended,
    select_expected_f1,
    select_group_predictions,
)


def _toy_edge_table() -> pd.DataFrame:
    rows = []
    for ref, tgt, conf, g0, g1, l0, l1 in [
        ("r1", "t1", 0.99, 0.0, 0.5, 0.0, 1.0),
        ("r1", "t2", 0.60, 0.4, 0.9, 0.0, 1.0),  # overlaps r1's covered span
        ("r2", "t1", 0.90, 0.0, 1.0, 0.0, 0.5),
    ]:
        rows.append(
            {
                "dataset_id": "toy",
                "group_id": "g",
                "human_group_id": "h",
                "labeler": "t",
                "provenance": "clean",
                "match_type": "M:N",
                "ref_id": ref,
                "target_id": tgt,
                "keep": 1,
                "selected": True,
                "pruned": False,
                "confidence": conf,
                "degree_ref": 1,
                "degree_tgt": 1,
                "is_bridge": False,
                "is_sliver": False,
                "biconnected_block": 0,
                "corridor_ref": 0,
                "corridor_tgt": 0,
                "gers_start_frac": g0,
                "gers_end_frac": g1,
                "local_start_frac": l0,
                "local_end_frac": l1,
                "n_edges": 3,
                "n_corridors": 1,
                "n_assignment_components": 1,
                "largest_biconnected_block": 1,
                "oversized_group": False,
                "num_refs": 2,
                "num_targets": 2,
            }
        )
    return pd.DataFrame(rows)


class TestSelectExpectedF1:
    def test_confident_probs_select_all(self):
        sel = select_expected_f1(np.array([0.95, 0.9, 0.85]))
        assert sel.tolist() == [1, 1, 1]

    def test_low_probs_select_none(self):
        sel = select_expected_f1(np.array([0.05, 0.03]))
        assert sel.sum() == 0

    def test_mixed_selects_high_prefix(self):
        probs = np.array([0.95, 0.9, 0.1, 0.05])
        sel = select_expected_f1(probs)
        assert sel[:2].tolist() == [1, 1]
        assert sel[2:].sum() == 0

    def test_selection_is_prefix_of_sorted_probs(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            probs = rng.uniform(0, 1, size=rng.integers(1, 12))
            sel = select_expected_f1(probs)
            chosen = set(np.flatnonzero(sel))
            k = len(chosen)
            assert chosen == set(np.argsort(-probs)[:k])


class TestSelectGroupPredictions:
    def test_rejects_invalid_probabilities(self):
        df = _toy_edge_table()

        with pytest.raises(ValueError, match="finite one-dimensional"):
            select_group_predictions(
                df,
                np.array([0.9, np.nan, 0.1]),
                selector="ef1",
            )
        with pytest.raises(ValueError, match="between 0 and 1"):
            select_group_predictions(
                df,
                np.array([0.9, 1.1, 0.1]),
                selector="ef1",
            )

    def test_rejects_null_group_key(self):
        df = _toy_edge_table()
        df.loc[0, "group_id"] = None

        with pytest.raises(ValueError, match="group keys"):
            select_group_predictions(df, np.array([0.9, 0.8, 0.7]), selector="ef1")


class TestFeaturizeExtended:
    def test_columns_present_and_finite_shapes(self):
        out = featurize_extended(_toy_edge_table())
        for col in EXTENDED_FEATURE_COLUMNS:
            assert col in out.columns, col
        assert len(out) == 3

    def test_competition_margin_sign(self):
        out = featurize_extended(_toy_edge_table()).set_index(["ref_id", "target_id"])
        # r1/t1 (0.99) is the best claim on r1; r1/t2 (0.60) competes and loses
        assert out.loc[("r1", "t1"), "conf_margin_ref"] > 0
        assert out.loc[("r1", "t2"), "conf_margin_ref"] < 0
        assert out.loc[("r1", "t1"), "is_best_for_ref"] == 1
        assert out.loc[("r1", "t2"), "is_best_for_ref"] == 0

    def test_coverage_overlap_flags_redundant_span(self):
        out = featurize_extended(_toy_edge_table()).set_index(["ref_id", "target_id"])
        # r1/t2's gers span [0.4,0.9] is 20% covered by r1/t1's [0,0.5]
        assert out.loc[("r1", "t2"), "ref_span_overlap_higher"] == pytest.approx(0.2, abs=1e-6)
        # the highest-confidence edge on each segment has zero higher coverage
        assert out.loc[("r1", "t1"), "ref_span_overlap_higher"] == 0.0


class TestExtractPrunedColumn:
    def test_pruned_flag_survives_extraction(self):
        group = {
            "group_id": "abc",
            "match_type": "M:N",
            "n_edges": 1,
            "ref_ids": ["r1"],
            "target_ids": ["t1", "t2"],
            "edges": [{"ref_id": "r1", "target_id": "t1", "confidence": 0.99, "selected": True}],
            "rejected_edges": [
                {
                    "ref_id": "r1",
                    "target_id": "t2",
                    "confidence": 0.4,
                    "selected": False,
                    "pruned": True,
                }
            ],
        }
        human = pd.DataFrame(
            [
                {
                    "group_id": "abc",
                    "labeler": "t",
                    "selected_edges": '[{"ref_id": "r1", "target_id": "t1"}]',
                }
            ]
        )
        df = build_edge_table([group], human, "toy")
        assert len(df) == 2
        by = df.set_index("target_id")
        assert not by.loc["t1", "pruned"]
        assert by.loc["t2", "pruned"]
        assert by.loc["t2", "keep"] == 0
        assert not by.loc["t2", "selected"]


def test_pair_feature_families_are_cumulative_and_deduplicated():
    from crosswalk.config import FEATURE_COLUMNS as pair_columns
    from crosswalk.resolver.round2 import (
        EXTENDED_FEATURE_COLUMNS,
        resolver_feature_columns,
    )

    base = resolver_feature_columns(pair_family="none")
    geometry = resolver_feature_columns(pair_family="geometry")
    nonsemantic = resolver_feature_columns(pair_family="nonsemantic")
    all_features = resolver_feature_columns(pair_family="all")

    assert base == EXTENDED_FEATURE_COLUMNS
    assert set(base) < set(geometry) < set(nonsemantic) < set(all_features)
    assert set(pair_columns) <= set(all_features)
    assert len(all_features) == len(set(all_features))


def test_unknown_pair_feature_family_rejected():
    from crosswalk.resolver.round2 import resolver_feature_columns

    with pytest.raises(ValueError, match="unknown pair feature family"):
        resolver_feature_columns(pair_family="magic")
