"""Tests for atomic writes and backup recovery in label_store.py."""

import pandas as pd
import pytest

from crosswalk.labeling.label_store import (
    HUMAN_LABEL_COLUMNS,
    LabelLoadError,
    LabelStore,
)


@pytest.fixture
def temp_labels_dir(tmp_path):
    """Create a temporary labels directory."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    return labels_dir


@pytest.fixture
def sample_features():
    """Return sample feature dict for testing."""
    return {
        "hausdorff_distance_m": 1.5,
        "buffer_iou_5m": 0.8,
        "buffer_iou_15m": 0.9,
        "heading_delta": 5.0,
        "name_levenshtein": 0.9,
        "name_jaro_winkler": 0.95,
        "class_similarity": 1.0,
    }


class TestAtomicWrites:
    """Tests for atomic write functionality."""

    def test_save_creates_backup(self, temp_labels_dir, sample_features):
        """Test that saving creates a backup of the previous file."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Add first label
        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Verify primary file exists
        assert store.csv_path.exists()
        backup_path = store.csv_path.with_suffix(".csv.bak")

        # Add second label - should create backup
        store.add(
            gers_id="ref_2",
            target_id="target_2",
            label="no_match",
            labeler="tester",
            session_id="session_1",
            original_decision="no_match",
            original_confidence=0.2,
            features=sample_features,
        )

        # Verify backup was created
        assert backup_path.exists()

        # Verify backup has only 1 row (previous state)
        backup_df = pd.read_csv(backup_path)
        assert len(backup_df) == 1
        assert backup_df.iloc[0]["gers_id"] == "ref_1"

        # Verify primary has 2 rows (current state)
        primary_df = pd.read_csv(store.csv_path)
        assert len(primary_df) == 2

    def test_save_does_not_leave_tmp_file(self, temp_labels_dir, sample_features):
        """Test that temporary files are cleaned up after save."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Verify no .tmp file remains
        tmp_path = store.csv_path.with_suffix(".csv.tmp")
        assert not tmp_path.exists()

    def test_save_overwrites_old_backup(self, temp_labels_dir, sample_features):
        """Test that old backup is overwritten, not accumulated."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Add three labels
        for i in range(3):
            store.add(
                gers_id=f"ref_{i}",
                target_id=f"target_{i}",
                label="match",
                labeler="tester",
                session_id="session_1",
                original_decision="match",
                original_confidence=0.9,
                features=sample_features,
            )

        # Backup should have 2 rows (state before last save)
        backup_path = store.csv_path.with_suffix(".csv.bak")
        backup_df = pd.read_csv(backup_path)
        assert len(backup_df) == 2


class TestBackupRecovery:
    """Tests for backup recovery on load."""

    def test_recovery_from_backup_when_primary_corrupted(self, temp_labels_dir, sample_features):
        """Test recovery from backup when primary file is corrupted."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Add labels to create a valid backup
        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )
        store.add(
            gers_id="ref_2",
            target_id="target_2",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Corrupt the primary file
        store.csv_path.write_text("invalid,csv,data\n\x00\x01\x02garbage")

        # Create a new store instance and load
        new_store = LabelStore("test_dataset", labels_dir=temp_labels_dir)
        df = new_store.df

        # Should recover from backup (which has 1 row from before last save)
        assert len(df) == 1
        assert df.iloc[0]["gers_id"] == "ref_1"

    def test_error_when_both_files_corrupted(self, temp_labels_dir, sample_features):
        """Test that LabelLoadError is raised when both files are corrupted."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Add labels
        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )
        store.add(
            gers_id="ref_2",
            target_id="target_2",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Corrupt both files
        store.csv_path.write_text("invalid,csv,data\n\x00\x01\x02garbage")
        backup_path = store.csv_path.with_suffix(".csv.bak")
        backup_path.write_text("also,invalid\n\x00garbage")

        # Create a new store instance and try to load
        new_store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        with pytest.raises(LabelLoadError) as exc_info:
            _ = new_store.df

        assert "corrupted" in str(exc_info.value).lower()

    def test_error_when_primary_corrupted_no_backup(self, temp_labels_dir, sample_features):
        """Test that LabelLoadError is raised when primary is corrupted and no backup."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Add one label (no backup created yet)
        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Corrupt primary file
        store.csv_path.write_text("invalid\x00data")

        # Remove backup if it exists
        backup_path = store.csv_path.with_suffix(".csv.bak")
        if backup_path.exists():
            backup_path.unlink()

        # Create a new store instance and try to load
        new_store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        with pytest.raises(LabelLoadError) as exc_info:
            _ = new_store.df

        assert "no backup" in str(exc_info.value).lower()

    def test_empty_dataframe_when_no_files_exist(self, temp_labels_dir):
        """Test that empty dataframe is returned when no files exist."""
        store = LabelStore("fresh_dataset", labels_dir=temp_labels_dir)

        # Should return empty dataframe, not raise error
        df = store.df
        assert len(df) == 0
        assert list(df.columns) == HUMAN_LABEL_COLUMNS

    def test_backup_only_loads_when_primary_missing(self, temp_labels_dir, sample_features):
        """Test that backup is loaded when primary is missing."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Create labels
        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )
        store.add(
            gers_id="ref_2",
            target_id="target_2",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Delete primary file
        store.csv_path.unlink()

        # Create new store and load
        new_store = LabelStore("test_dataset", labels_dir=temp_labels_dir)
        df = new_store.df

        # Should load from backup
        assert len(df) == 1  # Backup has 1 row (state before last save)


class TestRemoveLastWithAtomic:
    """Tests for remove_last with atomic writes."""

    def test_remove_last_creates_backup(self, temp_labels_dir, sample_features):
        """Test that remove_last also uses atomic writes."""
        store = LabelStore("test_dataset", labels_dir=temp_labels_dir)

        # Add two labels
        store.add(
            gers_id="ref_1",
            target_id="target_1",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )
        store.add(
            gers_id="ref_2",
            target_id="target_2",
            label="match",
            labeler="tester",
            session_id="session_1",
            original_decision="match",
            original_confidence=0.9,
            features=sample_features,
        )

        # Remove last
        removed = store.remove_last()

        assert removed is not None
        assert removed["gers_id"] == "ref_2"

        # Primary should have 1 row
        primary_df = pd.read_csv(store.csv_path)
        assert len(primary_df) == 1

        # Backup should have 2 rows (state before remove)
        backup_path = store.csv_path.with_suffix(".csv.bak")
        backup_df = pd.read_csv(backup_path)
        assert len(backup_df) == 2
