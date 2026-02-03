"""Schema validation tests for label stores.

These tests validate that the label storage architecture is consistent:
- Feature store has correct schema (all FEATURE_COLUMNS)
- Data store has correct schema (required columns and geometry types)
- Labels have corresponding features (parity check)

These tests run against real label data if available, or skip if not present.
They are intended for CI to catch schema drift or missing features.
"""

from pathlib import Path

import pytest

from matcher.config import FEATURE_COLUMNS
from matcher.labeling.data_store import DATA_COLUMNS, DataStore
from matcher.labeling.feature_store import FEATURE_KEY_COLUMNS, FeatureStore
from matcher.labeling.label_store import HUMAN_LABEL_COLUMNS, LabelStore

# Default paths for label storage
LABELS_DIR = Path("labels")
FEATURES_DIR = LABELS_DIR / "features"
DATA_DIR = LABELS_DIR / "data"
HUMAN_DIR = LABELS_DIR / "human"
AGENT_DIR = LABELS_DIR / "agent"


class TestFeatureStoreSchema:
    """Validate feature store Parquet has correct schema."""

    @pytest.fixture
    def features_df(self):
        """Load all features, skip if no data."""
        if not FEATURES_DIR.exists():
            pytest.skip("No features directory found")
        df = FeatureStore.load_all(FEATURES_DIR)
        if len(df) == 0:
            pytest.skip("No features data found")
        return df

    def test_key_columns_present(self, features_df):
        """Feature store has required key columns."""
        for col in FEATURE_KEY_COLUMNS:
            assert col in features_df.columns, f"Missing key column: {col}"

    def test_all_feature_columns_present(self, features_df):
        """Feature store has all FEATURE_COLUMNS."""
        for col in FEATURE_COLUMNS:
            assert col in features_df.columns, f"Missing feature column: {col}"

    def test_key_column_types(self, features_df):
        """Key columns have correct types."""
        assert features_df["gers_id"].dtype == "object"  # string
        assert features_df["target_id"].dtype == "object"  # string

    def test_feature_column_types(self, features_df):
        """Feature columns are numeric (float64 or int64)."""
        import numpy as np

        for col in FEATURE_COLUMNS:
            if col in features_df.columns:
                assert np.issubdtype(features_df[col].dtype, np.number), (
                    f"Wrong type for {col}: {features_df[col].dtype} (expected numeric)"
                )

    def test_no_null_keys(self, features_df):
        """Key columns have no null values."""
        assert features_df["gers_id"].notna().all(), "gers_id has null values"
        assert features_df["target_id"].notna().all(), "target_id has null values"

    def test_dataset_column_added(self, features_df):
        """load_all adds dataset column from partition path."""
        assert "dataset" in features_df.columns, "Missing dataset column from partitioning"


class TestDataStoreSchema:
    """Validate data store GeoParquet has correct schema."""

    @pytest.fixture
    def data_gdf(self):
        """Load all data, skip if no data."""
        if not DATA_DIR.exists():
            pytest.skip("No data directory found")
        gdf = DataStore.load_all(DATA_DIR)
        if len(gdf) == 0:
            pytest.skip("No data found")
        return gdf

    def test_key_columns_present(self, data_gdf):
        """Data store has required key columns."""
        assert "gers_id" in data_gdf.columns
        assert "target_id" in data_gdf.columns

    def test_geometry_columns_present(self, data_gdf):
        """Data store has geometry columns."""
        assert "ref_geometry" in data_gdf.columns
        assert "target_geometry" in data_gdf.columns

    def test_attribute_columns_present(self, data_gdf):
        """Data store has all attribute columns (missing filled with None)."""
        for col in DATA_COLUMNS:
            assert col in data_gdf.columns, f"Missing data column: {col}"

    def test_geometries_are_valid(self, data_gdf):
        """Geometry columns contain valid geometries."""
        # Check ref_geometry
        ref_valid = data_gdf["ref_geometry"].apply(lambda g: g is not None and g.is_valid)
        assert ref_valid.all(), f"Invalid ref_geometry in {(~ref_valid).sum()} rows"

        # Check target_geometry
        target_valid = data_gdf["target_geometry"].apply(lambda g: g is not None and g.is_valid)
        assert target_valid.all(), f"Invalid target_geometry in {(~target_valid).sum()} rows"

    def test_dataset_column_added(self, data_gdf):
        """load_all adds dataset column from partition path."""
        assert "dataset" in data_gdf.columns, "Missing dataset column from partitioning"


class TestHumanLabelSchema:
    """Validate human label CSV has correct schema."""

    @pytest.fixture
    def human_labels_df(self):
        """Load human labels, skip if no data."""
        if not HUMAN_DIR.exists():
            pytest.skip("No human labels directory found")
        df = LabelStore.load_human_labels(HUMAN_DIR)
        if len(df) == 0:
            pytest.skip("No human labels found")
        return df

    def test_key_columns_present(self, human_labels_df):
        """Human labels have required key columns."""
        assert "gers_id" in human_labels_df.columns
        assert "target_id" in human_labels_df.columns

    def test_label_column_present(self, human_labels_df):
        """Human labels have label column."""
        assert "label" in human_labels_df.columns

    def test_label_values_valid(self, human_labels_df):
        """Label values are from expected set."""
        valid_labels = {"match", "no_match", "unsure", "associated", "skip"}
        actual_labels = set(human_labels_df["label"].dropna().unique())
        invalid = actual_labels - valid_labels
        assert not invalid, f"Invalid label values: {invalid}"

    def test_metadata_columns_present(self, human_labels_df):
        """Human labels have metadata columns."""
        expected = ["labeler", "labeled_at", "session_id"]
        for col in expected:
            assert col in human_labels_df.columns, f"Missing metadata column: {col}"


class TestAgentLabelSchema:
    """Validate agent label CSV has correct schema."""

    @pytest.fixture
    def agent_labels_df(self):
        """Load agent labels, skip if no data."""
        if not AGENT_DIR.exists():
            pytest.skip("No agent labels directory found")
        df = LabelStore.load_agent_labels(AGENT_DIR)
        if len(df) == 0:
            pytest.skip("No agent labels found")
        return df

    def test_key_columns_present(self, agent_labels_df):
        """Agent labels have required key columns."""
        assert "gers_id" in agent_labels_df.columns
        assert "target_id" in agent_labels_df.columns

    def test_label_column_present(self, agent_labels_df):
        """Agent labels have label column."""
        assert "label" in agent_labels_df.columns

    def test_confidence_column_present(self, agent_labels_df):
        """Agent labels have confidence column."""
        assert "confidence" in agent_labels_df.columns

    def test_confidence_values_valid(self, agent_labels_df):
        """Confidence values are in [0, 1] range."""
        if "confidence" in agent_labels_df.columns:
            confidence = agent_labels_df["confidence"].dropna()
            if len(confidence) > 0:
                assert confidence.min() >= 0.0, f"Confidence below 0: {confidence.min()}"
                assert confidence.max() <= 1.0, f"Confidence above 1: {confidence.max()}"


class TestLabelFeatureParity:
    """Ensure labels have corresponding features."""

    def test_human_labels_have_features(self):
        """Every human label has corresponding features (normalized format)."""
        if not HUMAN_DIR.exists() or not FEATURES_DIR.exists():
            pytest.skip("Normalized label directories not found")

        labels = LabelStore.load_human_labels(HUMAN_DIR)
        features = FeatureStore.load_all(FEATURES_DIR)

        if len(labels) == 0:
            pytest.skip("No human labels found")
        if len(features) == 0:
            pytest.skip("No features found")

        # Create key sets
        label_dataset = (
            labels["dataset"] if "dataset" in labels.columns else ["unknown"] * len(labels)
        )
        feature_dataset = (
            features["dataset"] if "dataset" in features.columns else ["unknown"] * len(features)
        )

        label_keys = set(zip(labels["gers_id"], labels["target_id"], label_dataset))
        feature_keys = set(zip(features["gers_id"], features["target_id"], feature_dataset))

        missing = label_keys - feature_keys
        if missing:
            missing_pct = len(missing) / len(label_keys) * 100
            sample = list(missing)[:5]

            # Tolerate small number of orphaned labels (< 1%) since source data
            # files get updated and some labels may reference old IDs
            if missing_pct >= 1.0:
                pytest.fail(
                    f"{len(missing)} labels ({missing_pct:.1f}%) without features. "
                    f"Sample: {sample}\n"
                    f"Run 'matcher labels backfill' to compute missing features."
                )
            else:
                import warnings

                warnings.warn(
                    f"{len(missing)} labels ({missing_pct:.2f}%) without features "
                    f"(orphaned references to updated source data). Sample: {sample}",
                    stacklevel=2,
                )


class TestSchemaConsistency:
    """Cross-store schema consistency checks."""

    def test_feature_columns_match_config(self):
        """FeatureStore uses same columns as config.FEATURE_COLUMNS."""
        # Create empty store and check columns
        import tempfile

        from matcher.labeling.feature_store import FeatureStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore("test", features_dir=Path(tmpdir))
            empty_df = store._empty_dataframe()

            for col in FEATURE_COLUMNS:
                assert col in empty_df.columns, f"FeatureStore missing column: {col}"

    def test_label_store_human_columns_defined(self):
        """LabelStore defines expected human label columns."""
        expected_base = ["gers_id", "target_id", "label", "labeler", "labeled_at"]
        for col in expected_base:
            assert col in HUMAN_LABEL_COLUMNS, f"Missing from HUMAN_LABEL_COLUMNS: {col}"

    def test_data_store_columns_defined(self):
        """DataStore defines expected columns."""
        expected = ["gers_id", "target_id", "ref_geometry", "target_geometry"]
        for col in expected:
            assert col in DATA_COLUMNS, f"Missing from DATA_COLUMNS: {col}"
