"""Unit tests for integration QA module."""

from matcher.integration_qa.decision_store import (
    MergedDecisionStore,
    OrphanDecisionStore,
)


class TestOrphanDecisionStore:
    """Tests for OrphanDecisionStore."""

    def test_add_decision(self, tmp_path):
        """Can add orphan decision."""
        store = OrphanDecisionStore(tmp_path / "orphans.csv")

        store.add_decision(
            edge_id=123,
            original_id="t_1",
            dataset_id="boston_streets",
            component_id=5,
            decision="correct",
            reason="legitimate_new",
            reviewer="test_user",
            session_id="abc123",
            length_m=50.0,
            road_class="residential",
            nearest_main_dist_m=10.0,
            component_size=3,
        )

        assert len(store.df) == 1
        assert store.df.iloc[0]["edge_id"] == 123
        assert store.df.iloc[0]["decision"] == "correct"

    def test_get_reviewed_edges(self, tmp_path):
        """Can get set of reviewed edge IDs."""
        store = OrphanDecisionStore(tmp_path / "orphans.csv")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            dataset_id="test",
            component_id=1,
            decision="correct",
            reason="",
            reviewer="user1",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            dataset_id="test",
            component_id=1,
            decision="incorrect",
            reason="",
            reviewer="user2",
            session_id="abc",
        )

        # All reviewed
        all_reviewed = store.get_reviewed_edges()
        assert all_reviewed == {1, 2}

        # By reviewer
        user1_reviewed = store.get_reviewed_edges("user1")
        assert user1_reviewed == {1}

    def test_undo(self, tmp_path):
        """Can undo last decision."""
        store = OrphanDecisionStore(tmp_path / "orphans.csv")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            dataset_id="test",
            component_id=1,
            decision="correct",
            reason="",
            reviewer="user",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            dataset_id="test",
            component_id=1,
            decision="incorrect",
            reason="",
            reviewer="user",
            session_id="abc",
        )

        assert len(store.df) == 2

        removed = store.remove_last()
        assert removed["edge_id"] == 2
        assert len(store.df) == 1

    def test_stats(self, tmp_path):
        """Can get decision statistics."""
        store = OrphanDecisionStore(tmp_path / "orphans.csv")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            dataset_id="test",
            component_id=1,
            decision="correct",
            reason="",
            reviewer="user",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            dataset_id="test",
            component_id=1,
            decision="incorrect",
            reason="",
            reviewer="user",
            session_id="abc",
        )

        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["correct"] == 1
        assert stats["incorrect"] == 1


class TestMergedDecisionStore:
    """Tests for MergedDecisionStore."""

    def test_add_decision(self, tmp_path):
        """Can add merged edge decision."""
        store = MergedDecisionStore(tmp_path / "merged.csv")

        store.add_decision(
            edge_id=456,
            original_id="t_1",
            dataset_id="boston_streets",
            source_type="target_matched",
            match_ref_id="gers_123",
            decision="correct",
            reason="",
            reviewer="test_user",
            session_id="abc123",
            match_confidence=0.85,
            length_m=75.0,
            road_class="primary",
        )

        assert len(store.df) == 1
        assert store.df.iloc[0]["edge_id"] == 456
        assert store.df.iloc[0]["decision"] == "correct"
        assert store.df.iloc[0]["match_ref_id"] == "gers_123"

    def test_stats(self, tmp_path):
        """Can get decision statistics."""
        store = MergedDecisionStore(tmp_path / "merged.csv")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            dataset_id="test",
            source_type="target_matched",
            match_ref_id="ref_1",
            decision="correct",
            reason="",
            reviewer="user",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            dataset_id="test",
            source_type="target_new",
            match_ref_id=None,
            decision="incorrect",
            reason="matching_error",
            reviewer="user",
            session_id="abc",
        )

        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["correct"] == 1
        assert stats["incorrect"] == 1


