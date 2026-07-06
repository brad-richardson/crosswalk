"""Tests for segment-aware train/test splitting.

These tests verify that the segment-aware splitting logic prevents data leakage
by ensuring no segment appears in both train and test sets.
"""

import numpy as np
import pandas as pd
import pytest

from crosswalk.matching.ml import create_segment_groups, segment_aware_split


class TestCreateSegmentGroups:
    """Tests for the Union-Find based segment grouping."""

    def test_isolated_pairs_get_different_groups(self):
        """Pairs with no shared segments should be in different groups."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C"],
                "target_id": ["X", "Y", "Z"],
            }
        )
        groups = create_segment_groups(df)

        # Each pair should have a unique group
        assert groups.nunique() == 3

    def test_shared_gers_id_same_group(self):
        """Pairs sharing a gers_id should be in the same group."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "A", "B"],
                "target_id": ["X", "Y", "Z"],
            }
        )
        groups = create_segment_groups(df)

        # First two pairs share gers_id "A", should be same group
        assert groups.iloc[0] == groups.iloc[1]
        # Third pair is isolated
        assert groups.iloc[2] != groups.iloc[0]
        assert groups.nunique() == 2

    def test_shared_target_id_same_group(self):
        """Pairs sharing a target_id should be in the same group."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C"],
                "target_id": ["X", "X", "Z"],
            }
        )
        groups = create_segment_groups(df)

        # First two pairs share target_id "X", should be same group
        assert groups.iloc[0] == groups.iloc[1]
        # Third pair is isolated
        assert groups.iloc[2] != groups.iloc[0]
        assert groups.nunique() == 2

    def test_transitive_grouping(self):
        """Pairs connected through a chain of shared segments should be grouped."""
        # A-X, B-X (share X), B-Y (share B with previous)
        # All three should be in the same group
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "B"],
                "target_id": ["X", "X", "Y"],
            }
        )
        groups = create_segment_groups(df)

        # All pairs are transitively connected
        assert groups.nunique() == 1

    def test_preserves_index(self):
        """Group series should have the same index as input DataFrame."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B"],
                "target_id": ["X", "Y"],
            },
            index=[10, 20],
        )
        groups = create_segment_groups(df)

        assert list(groups.index) == [10, 20]


class TestSegmentAwareSplit:
    """Tests for the segment-aware train/test split utility."""

    def test_no_segment_overlap_gers_id(self):
        """No gers_id should appear in both train and test sets."""
        # Create data where some gers_ids appear in multiple pairs
        df = pd.DataFrame(
            {
                "gers_id": ["A", "A", "B", "B", "C", "D", "E", "F"],
                "target_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
                "label": ["match"] * 8,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.3, random_state=42)

        train_gers = set(df.iloc[train_idx]["gers_id"])
        test_gers = set(df.iloc[test_idx]["gers_id"])

        overlap = train_gers & test_gers
        assert len(overlap) == 0, f"gers_id overlap: {overlap}"

    def test_no_segment_overlap_target_id(self):
        """No target_id should appear in both train and test sets."""
        # Create data where some target_ids appear in multiple pairs
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "target_id": ["1", "1", "2", "2", "3", "4", "5", "6"],
                "label": ["match"] * 8,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.3, random_state=42)

        train_target = set(df.iloc[train_idx]["target_id"])
        test_target = set(df.iloc[test_idx]["target_id"])

        overlap = train_target & test_target
        assert len(overlap) == 0, f"target_id overlap: {overlap}"

    def test_no_overlap_with_transitive_connections(self):
        """Complex transitive connections should not cause overlap."""
        # Create a chain: A-1, B-1, B-2, C-2, C-3
        # Plus isolated: D-4, E-5, F-6
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "B", "C", "C", "D", "E", "F"],
                "target_id": ["1", "1", "2", "2", "3", "4", "5", "6"],
                "label": ["match"] * 8,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.3, random_state=42)

        train_gers = set(df.iloc[train_idx]["gers_id"])
        test_gers = set(df.iloc[test_idx]["gers_id"])
        train_target = set(df.iloc[train_idx]["target_id"])
        test_target = set(df.iloc[test_idx]["target_id"])

        assert len(train_gers & test_gers) == 0
        assert len(train_target & test_target) == 0

    def test_indices_partition_data(self):
        """Train and test indices should partition all data."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C", "D", "E"],
                "target_id": ["1", "2", "3", "4", "5"],
                "label": ["match"] * 5,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.4, random_state=42)

        # No overlap between train and test
        assert len(set(train_idx) & set(test_idx)) == 0
        # Union covers all indices
        assert set(train_idx) | set(test_idx) == set(range(len(df)))

    def test_reproducible_with_same_seed(self):
        """Same random_state should produce same split."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C", "D", "E", "F"],
                "target_id": ["1", "2", "3", "4", "5", "6"],
                "label": ["match"] * 6,
            }
        )

        train1, test1 = segment_aware_split(df, test_size=0.3, random_state=123)
        train2, test2 = segment_aware_split(df, test_size=0.3, random_state=123)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

    def test_different_seed_different_split(self):
        """Different random_state should produce different split."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "target_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
                "label": ["match"] * 8,
            }
        )

        train1, test1 = segment_aware_split(df, test_size=0.3, random_state=42)
        train2, test2 = segment_aware_split(df, test_size=0.3, random_state=99)

        # At least one of train or test should differ
        # (with enough data points, different seeds should give different splits)
        assert not (np.array_equal(train1, train2) and np.array_equal(test1, test2)), (
            "Different seeds produced identical splits"
        )

    def test_respects_test_size_approximately(self):
        """Test set should be approximately the requested size."""
        # Use enough isolated pairs to get close to requested ratio
        df = pd.DataFrame(
            {
                "gers_id": [f"G{i}" for i in range(100)],
                "target_id": [f"T{i}" for i in range(100)],
                "label": ["match"] * 100,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.2, random_state=42)

        # With 100 isolated pairs, test size should be close to 20%
        test_ratio = len(test_idx) / len(df)
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio {test_ratio} too far from 0.2"


class TestSegmentAwareSplitEdgeCases:
    """Edge case tests for segment-aware splitting."""

    def test_single_large_group(self):
        """When all pairs are connected, all go to training (can't split 1 group)."""
        # All pairs share at least one segment through the chain
        df = pd.DataFrame(
            {
                "gers_id": ["A", "A", "B", "B"],
                "target_id": ["1", "2", "2", "3"],
                "label": ["match"] * 4,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.5, random_state=42)

        # All in training since there's only 1 group
        assert len(train_idx) == 4
        assert len(test_idx) == 0

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty indices."""
        df = pd.DataFrame({"gers_id": [], "target_id": [], "label": []})

        train_idx, test_idx = segment_aware_split(df, test_size=0.2, random_state=42)

        assert len(train_idx) == 0
        assert len(test_idx) == 0

    def test_single_pair(self):
        """Single pair goes to training (can't split 1 group)."""
        df = pd.DataFrame(
            {
                "gers_id": ["A"],
                "target_id": ["1"],
                "label": ["match"],
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.5, random_state=42)

        # Single pair = single group, all goes to training
        assert len(train_idx) == 1
        assert len(test_idx) == 0

    def test_test_size_zero_returns_all_train(self):
        """test_size=0.0 should return all data in training set."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C"],
                "target_id": ["1", "2", "3"],
                "label": ["match"] * 3,
            }
        )

        train_idx, test_idx = segment_aware_split(df, test_size=0.0, random_state=42)

        assert len(train_idx) == 3
        assert len(test_idx) == 0

    def test_invalid_test_size_raises_error(self):
        """Invalid test_size values should raise ValueError."""
        df = pd.DataFrame(
            {
                "gers_id": ["A"],
                "target_id": ["1"],
                "label": ["match"],
            }
        )

        with pytest.raises(ValueError, match="test_size must be between"):
            segment_aware_split(df, test_size=-0.1, random_state=42)

        with pytest.raises(ValueError, match="test_size must be between"):
            segment_aware_split(df, test_size=1.5, random_state=42)

    def test_null_gers_id_raises_error(self):
        """Null gers_id values should raise ValueError."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", None, "C"],
                "target_id": ["1", "2", "3"],
                "label": ["match"] * 3,
            }
        )

        with pytest.raises(ValueError, match="must not contain null"):
            segment_aware_split(df, test_size=0.3, random_state=42)

    def test_null_target_id_raises_error(self):
        """Null target_id values should raise ValueError."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C"],
                "target_id": ["1", None, "3"],
                "label": ["match"] * 3,
            }
        )

        with pytest.raises(ValueError, match="must not contain null"):
            segment_aware_split(df, test_size=0.3, random_state=42)


class TestSegmentAwareSplitReturnGroups:
    """Tests for the return_groups parameter."""

    def test_return_groups_false_returns_two_values(self):
        """Without return_groups, should return (train_idx, test_idx)."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C", "D"],
                "target_id": ["1", "2", "3", "4"],
                "label": ["match"] * 4,
            }
        )

        result = segment_aware_split(df, test_size=0.3, random_state=42)
        assert len(result) == 2
        train_idx, test_idx = result
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

    def test_return_groups_true_returns_three_values(self):
        """With return_groups=True, should return (train_idx, test_idx, groups)."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C", "D"],
                "target_id": ["1", "2", "3", "4"],
                "label": ["match"] * 4,
            }
        )

        result = segment_aware_split(df, test_size=0.3, random_state=42, return_groups=True)
        assert len(result) == 3
        train_idx, test_idx, groups = result
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert isinstance(groups, pd.Series)

    def test_returned_groups_match_create_segment_groups(self):
        """Returned groups should match what create_segment_groups produces."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "A", "B", "C"],
                "target_id": ["1", "2", "2", "3"],
                "label": ["match"] * 4,
            }
        )

        _, _, groups = segment_aware_split(df, test_size=0.3, random_state=42, return_groups=True)
        expected_groups = create_segment_groups(df)

        pd.testing.assert_series_equal(groups, expected_groups)

    def test_return_groups_empty_dataframe(self):
        """Empty DataFrame with return_groups=True should return empty groups."""
        df = pd.DataFrame({"gers_id": [], "target_id": [], "label": []})

        train_idx, test_idx, groups = segment_aware_split(
            df, test_size=0.2, random_state=42, return_groups=True
        )

        assert len(train_idx) == 0
        assert len(test_idx) == 0
        assert len(groups) == 0

    def test_return_groups_test_size_zero(self):
        """test_size=0.0 with return_groups=True should still return groups."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "B", "C"],
                "target_id": ["1", "2", "3"],
                "label": ["match"] * 3,
            }
        )

        train_idx, test_idx, groups = segment_aware_split(
            df, test_size=0.0, random_state=42, return_groups=True
        )

        assert len(train_idx) == 3
        assert len(test_idx) == 0
        assert len(groups) == 3
        # Each pair is isolated, so 3 unique groups
        assert groups.nunique() == 3

    def test_return_groups_single_group(self):
        """Single large group with return_groups=True should return groups."""
        df = pd.DataFrame(
            {
                "gers_id": ["A", "A", "B", "B"],
                "target_id": ["1", "2", "2", "3"],
                "label": ["match"] * 4,
            }
        )

        train_idx, test_idx, groups = segment_aware_split(
            df, test_size=0.5, random_state=42, return_groups=True
        )

        # All in training since there's only 1 group
        assert len(train_idx) == 4
        assert len(test_idx) == 0
        # But groups should still be returned
        assert len(groups) == 4
        assert groups.nunique() == 1
