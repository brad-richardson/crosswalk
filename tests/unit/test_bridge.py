"""Tests for bridge file generation, including confidence filtering."""

import pyarrow.parquet as pq
import pytest

from crosswalk.matching.types import MatchDecision, MatchResult
from crosswalk.resolution.bridge import generate_bridge_file


def _make_match(
    ref_id: str, target_id: str, confidence: float, decision: MatchDecision
) -> MatchResult:
    """Helper to create a MatchResult for testing."""
    return MatchResult(
        ref_id=ref_id,
        target_id=target_id,
        decision=decision,
        confidence=confidence,
        score_breakdown={},
        features={},
    )


@pytest.fixture
def sample_matches():
    """A mix of MATCH, REVIEW, and NO_MATCH at various confidence levels."""
    return [
        _make_match("r1", "t1", 0.9, MatchDecision.MATCH),
        _make_match("r2", "t2", 0.5, MatchDecision.MATCH),
        _make_match("r3", "t3", 0.3, MatchDecision.REVIEW),
        _make_match("r4", "t4", 0.1, MatchDecision.REVIEW),
        _make_match("r5", "t5", 0.8, MatchDecision.NO_MATCH),
    ]


class TestBridgeMinConfidence:
    """Tests for bridge_min_confidence filtering."""

    def test_none_preserves_all_non_nomatch_edges(self, sample_matches, tmp_path):
        """bridge_min_confidence=None should include all MATCH + REVIEW edges."""
        out = tmp_path / "bridge.parquet"
        generate_bridge_file(sample_matches, out, bridge_min_confidence=None)

        table = pq.read_table(out)
        # 4 edges: 2 MATCH + 2 REVIEW (NO_MATCH always excluded)
        assert len(table) == 4

    def test_filters_below_threshold(self, sample_matches, tmp_path):
        """Edges below bridge_min_confidence should be excluded."""
        out = tmp_path / "bridge.parquet"
        generate_bridge_file(sample_matches, out, bridge_min_confidence=0.5)

        table = pq.read_table(out)
        confidences = table.column("confidence").to_pylist()
        # Only r1 (0.9) and r2 (0.5) should remain; r3 (0.3) and r4 (0.1) filtered
        assert len(table) == 2
        assert all(c >= 0.5 for c in confidences)

    def test_boundary_value_included(self, sample_matches, tmp_path):
        """Edge exactly at bridge_min_confidence should be included (not strictly less than)."""
        out = tmp_path / "bridge.parquet"
        # r3 has confidence=0.3, so threshold=0.3 should include it
        generate_bridge_file(sample_matches, out, bridge_min_confidence=0.3)

        table = pq.read_table(out)
        confidences = table.column("confidence").to_pylist()
        # r1 (0.9), r2 (0.5), r3 (0.3) included; r4 (0.1) filtered
        assert len(table) == 3
        assert all(c >= 0.3 for c in confidences)

    def test_high_threshold_filters_most(self, sample_matches, tmp_path):
        """High threshold should only keep high-confidence edges."""
        out = tmp_path / "bridge.parquet"
        generate_bridge_file(sample_matches, out, bridge_min_confidence=0.7)

        table = pq.read_table(out)
        # Only r1 (0.9) survives
        assert len(table) == 1
        assert table.column("confidence").to_pylist() == [0.9]

    def test_threshold_above_all_produces_empty(self, sample_matches, tmp_path):
        """Threshold higher than all edges produces empty bridge."""
        out = tmp_path / "bridge.parquet"
        generate_bridge_file(sample_matches, out, bridge_min_confidence=0.95)

        table = pq.read_table(out)
        assert len(table) == 0

    def test_nomatch_always_excluded(self, tmp_path):
        """NO_MATCH edges are excluded regardless of confidence."""
        matches = [
            _make_match("r1", "t1", 0.99, MatchDecision.NO_MATCH),
        ]
        out = tmp_path / "bridge.parquet"
        generate_bridge_file(matches, out, bridge_min_confidence=None)

        table = pq.read_table(out)
        assert len(table) == 0


class TestStitchProfiles:
    """Tests for stitch profile configuration."""

    def test_profiles_exist(self):
        """All expected profiles are defined."""
        from crosswalk.config import STITCH_PROFILES

        assert "recall" in STITCH_PROFILES
        assert "balanced" in STITCH_PROFILES
        assert "precision" in STITCH_PROFILES

    def test_recall_profile_disables_filtering(self):
        """Recall profile should have None (no filtering)."""
        from crosswalk.config import STITCH_PROFILES

        assert STITCH_PROFILES["recall"] is None

    def test_balanced_profile_value(self):
        """Balanced profile should use 0.5 threshold."""
        from crosswalk.config import STITCH_PROFILES

        assert STITCH_PROFILES["balanced"] == 0.5

    def test_precision_profile_value(self):
        """Precision profile should use 0.7 threshold."""
        from crosswalk.config import STITCH_PROFILES

        assert STITCH_PROFILES["precision"] == 0.7

    def test_default_settings_bridge_min_confidence(self):
        """Default settings should use balanced (0.5) bridge_min_confidence."""
        from crosswalk.config import MatcherSettings

        s = MatcherSettings()
        assert s.bridge_min_confidence == 0.5
