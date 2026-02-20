"""Unit tests for stitching review modules.

Tests alternatives generation, batch selection, stitching label store,
and compute_group_id.
"""

import json

import pandas as pd
import pytest

from matcher.matching.alternatives import generate_top_k_alternatives
from matcher.matching.batch_selection import select_stitching_batch
from matcher.matching.optimizer import compute_group_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _edge(ref, tgt, conf):
    return {"ref_id": ref, "target_id": tgt, "confidence": conf}


@pytest.fixture
def edges_1_to_n():
    """1:N component — one ref matched to three targets."""
    return [
        _edge("r1", "t1", 0.9),
        _edge("r1", "t2", 0.8),
        _edge("r1", "t3", 0.3),
    ]


@pytest.fixture
def edges_m_to_n():
    """M:N component — two refs matched to two targets."""
    return [
        _edge("r1", "t1", 0.9),
        _edge("r1", "t2", 0.4),
        _edge("r2", "t1", 0.3),
        _edge("r2", "t2", 0.85),
    ]


def _make_group(group_id, edges, alternatives=None, match_type="N:1"):
    """Build a minimal group dict for batch selection tests."""
    return {
        "group_id": group_id,
        "match_type": match_type,
        "edges": edges,
        "alternatives": alternatives or [],
    }


# ---------------------------------------------------------------------------
# compute_group_id
# ---------------------------------------------------------------------------


class TestComputeGroupId:
    def test_deterministic(self):
        a = compute_group_id({"r1", "r2"}, {"t1"})
        b = compute_group_id({"r2", "r1"}, {"t1"})
        assert a == b

    def test_length_is_8(self):
        assert len(compute_group_id({"r1"}, {"t1", "t2"})) == 8

    def test_different_ids_differ(self):
        a = compute_group_id({"r1"}, {"t1"})
        b = compute_group_id({"r1"}, {"t2"})
        assert a != b

    def test_ref_target_order_matters(self):
        """Swapping ref and target sets should produce a different ID."""
        a = compute_group_id({"r1"}, {"t1"})
        b = compute_group_id({"t1"}, {"r1"})
        assert a != b


# ---------------------------------------------------------------------------
# generate_top_k_alternatives
# ---------------------------------------------------------------------------


class TestGenerateTopKAlternatives:
    def test_empty_input(self):
        assert generate_top_k_alternatives([]) == []

    def test_1_to_n_returns_sorted_by_confidence(self, edges_1_to_n):
        alts = generate_top_k_alternatives(edges_1_to_n, k=5)
        confidences = [a["total_confidence"] for a in alts]
        assert confidences == sorted(confidences, reverse=True)

    def test_1_to_n_top_alternative_is_greedy(self, edges_1_to_n):
        alts = generate_top_k_alternatives(edges_1_to_n, k=5)
        top = alts[0]
        # Greedy picks all three targets assigned to r1
        edge_pairs = {(e["ref_id"], e["target_id"]) for e in top["edges"]}
        assert ("r1", "t1") in edge_pairs
        assert ("r1", "t2") in edge_pairs
        assert ("r1", "t3") in edge_pairs

    def test_m_to_n_has_multiple_alternatives(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=5)
        assert len(alts) > 1

    def test_option_index_sequential(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=5)
        indices = [a["option_index"] for a in alts]
        assert indices == list(range(len(alts)))

    def test_k_limits_results(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=2)
        assert len(alts) <= 2

    def test_no_duplicate_edge_sets(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=10)
        edge_sets = []
        for a in alts:
            es = frozenset((e["ref_id"], e["target_id"]) for e in a["edges"])
            edge_sets.append(es)
        assert len(edge_sets) == len(set(edge_sets))

    def test_summary_present(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=1)
        assert "summary" in alts[0]
        assert "->" in alts[0]["summary"]

    @pytest.mark.parametrize(
        "edges",
        [
            [_edge("r1", "t1", 0.5)],
            [_edge("r1", "t1", 0.5), _edge("r1", "t2", 0.3)],
        ],
        ids=["single_edge", "two_edges"],
    )
    def test_always_returns_at_least_one(self, edges):
        alts = generate_top_k_alternatives(edges, k=5)
        assert len(alts) >= 1

    def test_duplicate_edges_keeps_highest_confidence(self):
        edges = [
            _edge("r1", "t1", 0.5),
            _edge("r1", "t1", 0.9),
        ]
        alts = generate_top_k_alternatives(edges, k=5)
        # Only one unique edge set possible
        assert len(alts) == 1
        assert alts[0]["edges"][0]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# select_stitching_batch
# ---------------------------------------------------------------------------


class TestSelectStitchingBatch:
    def _alt(self, conf):
        return {"total_confidence": conf, "edges": [], "summary": ""}

    def test_empty_groups(self):
        result = select_stitching_batch([], pd.DataFrame(), set(), k=10)
        assert result == []

    def test_skips_already_reviewed(self):
        groups = [
            _make_group("g1", [_edge("r1", "t1", 0.9)], [self._alt(0.9)]),
            _make_group("g2", [_edge("r2", "t2", 0.8)], [self._alt(0.8)]),
        ]
        result = select_stitching_batch(groups, pd.DataFrame(), {"g1"}, k=10)
        ids = {g["group_id"] for g in result}
        assert "g1" not in ids
        assert "g2" in ids

    def test_all_reviewed_returns_empty(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, pd.DataFrame(), {"g1"}, k=10)
        assert result == []

    def test_tier_balancing_with_enough_groups(self):
        """With 30 groups, k=20 should produce roughly 8+8+4."""
        groups = []
        for i in range(30):
            conf = 0.5 + i * 0.01
            groups.append(
                _make_group(
                    f"g{i}",
                    [_edge(f"r{i}", f"t{i}", conf)],
                    [self._alt(conf), self._alt(conf - 0.01)],
                )
            )
        result = select_stitching_batch(groups, pd.DataFrame(), set(), k=20)
        assert len(result) == 20

        tiers = {g["review_tier"] for g in result}
        assert tiers == {"label_overlap", "borderline", "clear_winner"}

    @pytest.mark.parametrize("k", [5, 10, 20])
    def test_respects_k(self, k):
        groups = [
            _make_group(f"g{i}", [_edge("r1", "t1", 0.5)], [self._alt(0.5)]) for i in range(50)
        ]
        result = select_stitching_batch(groups, pd.DataFrame(), set(), k=k)
        assert len(result) == k

    def test_label_overlap_scoring(self):
        """Groups with labeled edges should score higher on label_overlap tier."""
        labels_df = pd.DataFrame({"gers_id": ["r1"], "target_id": ["t1"]})
        groups = [
            _make_group("overlap", [_edge("r1", "t1", 0.9)], [self._alt(0.9)]),
            _make_group("no_overlap", [_edge("r2", "t2", 0.9)], [self._alt(0.9)]),
        ]
        result = select_stitching_batch(groups, labels_df, set(), k=2)
        # The overlap group should appear as label_overlap tier
        overlap_group = next(g for g in result if g["group_id"] == "overlap")
        assert overlap_group["review_tier"] == "label_overlap"

    def test_review_tier_and_score_present(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, pd.DataFrame(), set(), k=5)
        assert "review_tier" in result[0]
        assert "review_score" in result[0]

    def test_no_internal_keys_leaked(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, pd.DataFrame(), set(), k=5)
        for g in result:
            assert "_label_overlap_score" not in g
            assert "_borderline_score" not in g
            assert "_review_value" not in g


# ---------------------------------------------------------------------------
# StitchingLabelStore
# ---------------------------------------------------------------------------


class TestStitchingLabelStore:
    @pytest.fixture
    def store(self, tmp_path):
        from matcher.labeling.stitching_store import StitchingLabelStore

        return StitchingLabelStore("test_dataset", labels_dir=tmp_path / "stitching")

    def test_empty_store(self, store):
        assert len(store.df) == 0
        assert store.get_reviewed_group_ids("test_dataset") == set()

    def test_add_and_load(self, store):
        store.add(
            group_id="abc123",
            dataset_id="test_dataset",
            selected_option_index=0,
            selected_edges=[{"ref_id": "r1", "target_id": "t1"}],
            match_type="1:N",
            num_refs=1,
            num_targets=2,
            labeler="tester",
            session_id="sess1",
        )
        assert len(store.df) == 1
        assert store.df.iloc[0]["group_id"] == "abc123"
        assert store.df.iloc[0]["match_type"] == "1:N"

    def test_dedup_replaces_on_same_group_id(self, store):
        for i in range(3):
            store.add(
                group_id="abc123",
                dataset_id="test_dataset",
                selected_option_index=i,
                selected_edges=[],
                match_type="N:1",
                num_refs=2,
                num_targets=1,
                labeler="tester",
                session_id=f"sess{i}",
            )
        assert len(store.df) == 1
        assert store.df.iloc[0]["selected_option_index"] == 2

    def test_get_reviewed_group_ids(self, store):
        store.add("g1", "test_dataset", 0, [], "1:N", 1, 2, "tester", "s1")
        store.add("g2", "test_dataset", 1, [], "N:1", 2, 1, "tester", "s2")
        reviewed = store.get_reviewed_group_ids("test_dataset")
        assert reviewed == {"g1", "g2"}

    def test_get_reviewed_filters_by_dataset(self, store):
        store.add("g1", "test_dataset", 0, [], "1:N", 1, 2, "tester", "s1")
        store.add("g2", "other_dataset", 0, [], "1:N", 1, 2, "tester", "s2")
        assert store.get_reviewed_group_ids("test_dataset") == {"g1"}
        assert store.get_reviewed_group_ids("other_dataset") == {"g2"}

    def test_selected_edges_stored_as_json(self, store):
        edges = [{"ref_id": "r1", "target_id": "t1"}, {"ref_id": "r1", "target_id": "t2"}]
        store.add("g1", "test_dataset", 0, edges, "1:N", 1, 2, "tester", "s1")
        raw = store.df.iloc[0]["selected_edges"]
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["ref_id"] == "r1"

    def test_persistence_across_instances(self, store):
        from matcher.labeling.stitching_store import StitchingLabelStore

        store.add("g1", "test_dataset", 0, [], "1:N", 1, 2, "tester", "s1")

        # New instance reads from disk
        store2 = StitchingLabelStore("test_dataset", labels_dir=store.labels_dir)
        assert len(store2.df) == 1
        assert store2.df.iloc[0]["group_id"] == "g1"

    def test_atomic_backup_exists_after_save(self, store):
        store.add("g1", "test_dataset", 0, [], "1:N", 1, 2, "tester", "s1")
        # First save creates no backup (no prior file)
        assert store.csv_path.exists()
        # Second save creates backup
        store.add("g2", "test_dataset", 0, [], "N:1", 2, 1, "tester", "s2")
        backup = store.csv_path.with_suffix(".csv.bak")
        assert backup.exists()
