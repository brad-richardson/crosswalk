"""Unit tests for integration QA module."""

from matcher.integration_qa.decision_store import (
    MergedDecisionStore,
    OrphanDecisionStore,
)
from matcher.integration_qa.state import QASession


class TestOrphanDecisionStore:
    """Tests for OrphanDecisionStore."""

    def test_add_decision(self, tmp_path):
        """Can add orphan decision."""
        store = OrphanDecisionStore(tmp_path / "orphans.parquet")

        store.add_decision(
            edge_id=123,
            original_id="t_1",
            source_dataset="boston_streets",
            component_id=5,
            decision="keep",
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
        assert store.df.iloc[0]["decision"] == "keep"

    def test_get_reviewed_edges(self, tmp_path):
        """Can get set of reviewed edge IDs."""
        store = OrphanDecisionStore(tmp_path / "orphans.parquet")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            source_dataset="test",
            component_id=1,
            decision="keep",
            reason="",
            reviewer="user1",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            source_dataset="test",
            component_id=1,
            decision="discard",
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
        store = OrphanDecisionStore(tmp_path / "orphans.parquet")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            source_dataset="test",
            component_id=1,
            decision="keep",
            reason="",
            reviewer="user",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            source_dataset="test",
            component_id=1,
            decision="discard",
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
        store = OrphanDecisionStore(tmp_path / "orphans.parquet")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            source_dataset="test",
            component_id=1,
            decision="keep",
            reason="",
            reviewer="user",
            session_id="abc",
        )
        store.add_decision(
            edge_id=2,
            original_id="t_2",
            source_dataset="test",
            component_id=1,
            decision="discard",
            reason="",
            reviewer="user",
            session_id="abc",
        )

        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["keep"] == 1
        assert stats["discard"] == 1


class TestMergedDecisionStore:
    """Tests for MergedDecisionStore."""

    def test_add_decision(self, tmp_path):
        """Can add merged edge decision."""
        store = MergedDecisionStore(tmp_path / "merged.parquet")

        store.add_decision(
            edge_id=456,
            original_id="t_1",
            source_dataset="boston_streets",
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
        store = MergedDecisionStore(tmp_path / "merged.parquet")

        store.add_decision(
            edge_id=1,
            original_id="t_1",
            source_dataset="test",
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
            source_dataset="test",
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


class TestQASession:
    """Tests for QASession."""

    def test_session_initialization(self):
        """Session initializes with defaults."""
        session = QASession()

        assert session.session_id is not None
        assert len(session.session_id) == 8
        assert session.current_view == "orphans"
        assert session.current_index == 0
        assert session.undo_stack == []

    def test_undo_stack(self):
        """Undo stack works correctly."""
        session = QASession()

        session.push_undo({"type": "orphan", "edge_id": 1})
        session.push_undo({"type": "merged", "edge_id": 2})

        assert len(session.undo_stack) == 2

        action = session.pop_undo()
        assert action["edge_id"] == 2
        assert len(session.undo_stack) == 1

    def test_undo_stack_limit(self):
        """Undo stack is limited to 50 items."""
        session = QASession()

        for i in range(60):
            session.push_undo({"type": "orphan", "edge_id": i})

        assert len(session.undo_stack) == 50
        # Oldest should be dropped
        assert session.undo_stack[0]["edge_id"] == 10
