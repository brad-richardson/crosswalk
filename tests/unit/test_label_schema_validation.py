"""Schema validation tests for label stores.

These tests validate that the label storage architecture is consistent:
- Feature store has correct schema (all FEATURE_COLUMNS)
- Data store has correct schema (required columns and geometry types)
- Labels have corresponding features (parity check)
- Cross-store referential integrity (labels ↔ features ↔ data)
- No duplicate or conflicting label pairs
- Dataset config coverage for all label partitions
- Geometry coordinate bounds (WGS84 sanity)

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


# Features that have been added to FEATURE_COLUMNS but not yet backfilled into
# stored feature parquets (see `matcher backfill`). Remove an entry here once a
# coordinated backfill has run and stored data includes the column.
PENDING_BACKFILL_FEATURES: set[str] = {
    "max_coverage",  # Added in feat/max-coverage; backfill planned post-merge.
}


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
        """Feature store has all FEATURE_COLUMNS (except those pending backfill)."""
        for col in FEATURE_COLUMNS:
            if col in PENDING_BACKFILL_FEATURES:
                continue
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

    def test_names_fields_valid_schema(self, data_gdf):
        """ref_names and target_names are either null or valid names structs.

        Valid names struct: dict with 'primary' key (str), optional 'common' and 'rules'.
        """
        import json

        for col in ("ref_names", "target_names"):
            if col not in data_gdf.columns:
                continue
            for idx, val in data_gdf[col].items():
                if val is None:
                    continue
                # Deserialize if stored as JSON string
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pytest.fail(f"{col} row {idx}: invalid JSON string: {val!r}")
                assert isinstance(val, dict), (
                    f"{col} row {idx}: expected dict or None, got {type(val).__name__}"
                )
                assert "primary" in val, f"{col} row {idx}: missing 'primary' key in names struct"
                primary = val["primary"]
                assert primary is None or isinstance(primary, str), (
                    f"{col} row {idx}: 'primary' must be str or None, got {type(primary).__name__}"
                )
                # common must be dict or list-of-pairs if present
                if "common" in val and val["common"] is not None:
                    common = val["common"]
                    assert isinstance(common, (dict, list)), (
                        f"{col} row {idx}: 'common' must be dict or list, got {type(common).__name__}"
                    )
                # rules must be list if present
                if "rules" in val and val["rules"] is not None:
                    rules = val["rules"]
                    assert isinstance(rules, list), (
                        f"{col} row {idx}: 'rules' must be list, got {type(rules).__name__}"
                    )


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
                    f"Run 'matcher backfill' to compute missing features."
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


class TestTargetIdFormat:
    """Validate that all target IDs use the H3-suffixed format.

    After the H3 spatial suffix migration, all target IDs should be in the
    format: {prefix}_{upstreamID}_{h3suffix} where h3suffix is 10 hex chars.

    Exception: OSM datasets use the format w{number}@{index}.
    """

    H3_ID_PATTERN = r"^.+_[0-9a-f]{10}$"
    OSM_ID_PATTERN = r"^w\d+@\d+$"
    OSM_DATASET_SUFFIX = "_osm"

    def _check_ids(self, df, source_name):
        """Check target_id format, return list of bad IDs."""
        import re

        bad_ids = []
        for _, row in df.iterrows():
            tid = str(row["target_id"])
            ds = str(row.get("dataset", ""))

            if ds.endswith(self.OSM_DATASET_SUFFIX):
                if not re.match(self.OSM_ID_PATTERN, tid):
                    bad_ids.append((ds, tid))
            else:
                if not re.match(self.H3_ID_PATTERN, tid):
                    bad_ids.append((ds, tid))
        return bad_ids

    def test_human_label_ids_h3_format(self):
        """Human label target_ids use H3-suffixed format."""
        if not HUMAN_DIR.exists():
            pytest.skip("No human labels directory found")
        df = LabelStore.load_human_labels(HUMAN_DIR)
        if len(df) == 0:
            pytest.skip("No human labels found")

        bad = self._check_ids(df, "human labels")
        if bad:
            by_ds = {}
            for ds, tid in bad:
                by_ds.setdefault(ds, []).append(tid)
            summary = {ds: len(ids) for ds, ids in by_ds.items()}
            sample = bad[:5]
            pytest.fail(
                f"{len(bad)} human labels with non-H3 target_ids: {summary}\nSample: {sample}"
            )

    def test_agent_label_ids_h3_format(self):
        """Agent label target_ids use H3-suffixed format."""
        if not AGENT_DIR.exists():
            pytest.skip("No agent labels directory found")
        df = LabelStore.load_agent_labels(AGENT_DIR)
        if len(df) == 0:
            pytest.skip("No agent labels found")

        bad = self._check_ids(df, "agent labels")
        if bad:
            by_ds = {}
            for ds, tid in bad:
                by_ds.setdefault(ds, []).append(tid)
            summary = {ds: len(ids) for ds, ids in by_ds.items()}
            sample = bad[:5]
            pytest.fail(
                f"{len(bad)} agent labels with non-H3 target_ids: {summary}\nSample: {sample}"
            )

    def test_feature_ids_h3_format(self):
        """Feature store target_ids use H3-suffixed format."""
        if not FEATURES_DIR.exists():
            pytest.skip("No features directory found")
        df = FeatureStore.load_all(FEATURES_DIR)
        if len(df) == 0:
            pytest.skip("No features found")

        bad = self._check_ids(df, "features")
        if bad:
            by_ds = {}
            for ds, tid in bad:
                by_ds.setdefault(ds, []).append(tid)
            summary = {ds: len(ids) for ds, ids in by_ds.items()}
            sample = bad[:5]
            pytest.fail(f"{len(bad)} features with non-H3 target_ids: {summary}\nSample: {sample}")

    def test_data_ids_h3_format(self):
        """Data store target_ids use H3-suffixed format."""
        if not DATA_DIR.exists():
            pytest.skip("No data directory found")
        gdf = DataStore.load_all(DATA_DIR)
        if len(gdf) == 0:
            pytest.skip("No data found")

        bad = self._check_ids(gdf, "data")
        if bad:
            by_ds = {}
            for ds, tid in bad:
                by_ds.setdefault(ds, []).append(tid)
            summary = {ds: len(ids) for ds, ids in by_ds.items()}
            sample = bad[:5]
            pytest.fail(f"{len(bad)} data rows with non-H3 target_ids: {summary}\nSample: {sample}")

    def test_target_parquets_have_h3_ids(self):
        """Re-fetched target parquets use H3-suffixed IDs."""
        import re

        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            pytest.skip("No data/raw directory found")

        # Only check target dataset parquets, not OSM/overture segment files
        parquets = [
            pq
            for pq in raw_dir.glob("*_v*.parquet")
            if "_osm_" not in pq.name and "overture_" not in pq.name
        ]
        if not parquets:
            pytest.skip("No target parquets found")

        import pandas as pd

        pattern = re.compile(self.H3_ID_PATTERN)
        bad_files = {}
        for pq in parquets:
            df = pd.read_parquet(pq, columns=["id"])
            non_h3 = df[~df["id"].apply(lambda x: bool(pattern.match(str(x))))]
            if len(non_h3) > 0:
                bad_files[pq.name] = len(non_h3)

        if bad_files:
            pytest.fail(f"Target parquets with non-H3 IDs: {bad_files}")


class TestFeatureDataQuality:
    """Validate stored features have plausible values.

    These tests catch corrupted features (e.g., stale geometry lookups during backfill)
    before they reach training. If these fail, re-backfill the affected dataset.
    """

    @pytest.fixture
    def features_df(self):
        """Load all features, skip if no data."""
        if not FEATURES_DIR.exists():
            pytest.skip("No features directory found")
        df = FeatureStore.load_all(FEATURES_DIR)
        if len(df) == 0:
            pytest.skip("No features data found")
        return df

    def test_hausdorff_distance_plausible(self, features_df):
        """Zero match-labeled pairs with hausdorff_distance_m > 1000m."""
        if "hausdorff_distance_m" not in features_df.columns:
            pytest.skip("hausdorff_distance_m not in features")
        labels = LabelStore.load_human_labels(HUMAN_DIR)
        matches = features_df.merge(
            labels[labels["label"] == "match"][["gers_id", "target_id"]],
            on=["gers_id", "target_id"],
        )
        bad = matches[matches["hausdorff_distance_m"] > 1000.0]
        if len(bad) > 0:
            by_dataset = bad.groupby("dataset").size().to_dict()
            pytest.fail(f"{len(bad)} match pairs with hausdorff_distance_m > 1000m: {by_dataset}")

    def test_no_all_nan_feature_rows(self, features_df):
        """No rows where every feature is NaN."""
        feature_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        all_nan = features_df[feature_cols].isna().all(axis=1)
        if all_nan.any():
            count = all_nan.sum()
            by_dataset = features_df[all_nan].groupby("dataset").size().to_dict()
            pytest.fail(f"{count} rows with all-NaN features: {by_dataset}")


class TestCrossStoreReferentialIntegrity:
    """Validate that labels, features, and data stores reference the same pairs.

    Catches orphaned records across stores that could cause silent training issues.
    """

    def _make_keys(self, df):
        """Build a set of (gers_id, target_id, dataset) keys from a DataFrame."""
        dataset = df["dataset"] if "dataset" in df.columns else ["unknown"] * len(df)
        return set(zip(df["gers_id"], df["target_id"], dataset))

    def test_agent_labels_have_features(self):
        """Every agent label has corresponding features."""
        if not AGENT_DIR.exists() or not FEATURES_DIR.exists():
            pytest.skip("Agent or features directory not found")

        labels = LabelStore.load_agent_labels(AGENT_DIR)
        features = FeatureStore.load_all(FEATURES_DIR)

        if len(labels) == 0:
            pytest.skip("No agent labels found")
        if len(features) == 0:
            pytest.skip("No features found")

        label_keys = self._make_keys(labels)
        feature_keys = self._make_keys(features)

        missing = label_keys - feature_keys
        if missing:
            missing_pct = len(missing) / len(label_keys) * 100
            if missing_pct >= 1.0:
                by_ds = {}
                for _, _, ds in missing:
                    by_ds[ds] = by_ds.get(ds, 0) + 1
                pytest.fail(
                    f"{len(missing)} agent labels ({missing_pct:.1f}%) without features: {by_ds}\n"
                    f"Run 'matcher backfill' to compute missing features."
                )

    def test_labels_have_backing_data(self):
        """Every label (human + agent) has geometry backing in the data store.

        Uses pandas directly since some data parquets may lack geopandas geo metadata
        but still contain the correct key columns for referential integrity checks.
        """
        import pandas as pd

        if not DATA_DIR.exists():
            pytest.skip("No data directory found")

        # Load all data parquets using pandas (not geopandas) for key-only check
        data_frames = []
        for d in DATA_DIR.iterdir():
            if not d.is_dir() or not d.name.startswith("dataset="):
                continue
            pq = d / "data.parquet"
            if pq.exists():
                try:
                    df = pd.read_parquet(pq, columns=["gers_id", "target_id"])
                    df["dataset"] = d.name.removeprefix("dataset=")
                    data_frames.append(df)
                except Exception:
                    continue

        if not data_frames:
            pytest.skip("No data found")

        data_df = pd.concat(data_frames, ignore_index=True)
        data_keys = self._make_keys(data_df)

        # Check human labels
        missing_human = 0
        total_labels = 0
        if HUMAN_DIR.exists():
            human = LabelStore.load_human_labels(HUMAN_DIR)
            if len(human) > 0:
                total_labels += len(human)
                human_keys = self._make_keys(human)
                missing_human = len(human_keys - data_keys)

        # Check agent labels
        missing_agent = 0
        if AGENT_DIR.exists():
            agent = LabelStore.load_agent_labels(AGENT_DIR)
            if len(agent) > 0:
                total_labels += len(agent)
                agent_keys = self._make_keys(agent)
                missing_agent = len(agent_keys - data_keys)

        total_missing = missing_human + missing_agent
        if total_missing > 0:
            missing_pct = total_missing / max(total_labels, 1) * 100
            if missing_pct >= 1.0:
                pytest.fail(
                    f"{total_missing} labels ({missing_pct:.1f}%) without backing data "
                    f"(human: {missing_human}, agent: {missing_agent})"
                )


class TestNoDuplicatePairs:
    """Validate no duplicate (gers_id, target_id) pairs within a store per dataset.

    Duplicates cause double-counting during training and corrupt feature lookups.
    """

    def test_no_duplicate_human_labels(self):
        """No duplicate pairs in human labels per dataset."""
        if not HUMAN_DIR.exists():
            pytest.skip("No human labels directory found")
        df = LabelStore.load_human_labels(HUMAN_DIR)
        if len(df) == 0:
            pytest.skip("No human labels found")

        dupes = df.groupby(["dataset", "gers_id", "target_id"]).size()
        dupes = dupes[dupes > 1]
        if len(dupes) > 0:
            by_ds = dupes.reset_index().groupby("dataset").size().to_dict()
            pytest.fail(f"Duplicate human label pairs: {by_ds}")

    def test_no_duplicate_agent_labels(self):
        """No duplicate pairs in agent labels per dataset."""
        if not AGENT_DIR.exists():
            pytest.skip("No agent labels directory found")
        df = LabelStore.load_agent_labels(AGENT_DIR)
        if len(df) == 0:
            pytest.skip("No agent labels found")

        dupes = df.groupby(["dataset", "gers_id", "target_id"]).size()
        dupes = dupes[dupes > 1]
        if len(dupes) > 0:
            by_ds = dupes.reset_index().groupby("dataset").size().to_dict()
            pytest.fail(f"Duplicate agent label pairs: {by_ds}")

    def test_no_duplicate_feature_pairs(self):
        """No duplicate pairs in feature store per dataset."""
        if not FEATURES_DIR.exists():
            pytest.skip("No features directory found")
        df = FeatureStore.load_all(FEATURES_DIR)
        if len(df) == 0:
            pytest.skip("No features found")

        dupes = df.groupby(["dataset", "gers_id", "target_id"]).size()
        dupes = dupes[dupes > 1]
        if len(dupes) > 0:
            by_ds = dupes.reset_index().groupby("dataset").size().to_dict()
            pytest.fail(f"Duplicate feature pairs: {by_ds}")

    def test_no_duplicate_data_pairs(self):
        """No duplicate pairs in data store per dataset."""
        if not DATA_DIR.exists():
            pytest.skip("No data directory found")
        gdf = DataStore.load_all(DATA_DIR)
        if len(gdf) == 0:
            pytest.skip("No data found")

        dupes = gdf.groupby(["dataset", "gers_id", "target_id"]).size()
        dupes = dupes[dupes > 1]
        if len(dupes) > 0:
            by_ds = dupes.reset_index().groupby("dataset").size().to_dict()
            pytest.fail(f"Duplicate data pairs: {by_ds}")


class TestNoConflictingLabels:
    """Validate no conflicting labels for the same pair within a source.

    Same (gers_id, target_id) should not have different label values from the
    same source (human or agent). This catches data corruption or merge errors.
    """

    def test_no_conflicting_human_labels(self):
        """Same pair should not have different human label values."""
        if not HUMAN_DIR.exists():
            pytest.skip("No human labels directory found")
        df = LabelStore.load_human_labels(HUMAN_DIR)
        if len(df) == 0:
            pytest.skip("No human labels found")

        # Group by pair and count unique labels
        label_counts = df.groupby(["dataset", "gers_id", "target_id"])["label"].nunique()
        conflicts = label_counts[label_counts > 1]
        if len(conflicts) > 0:
            by_ds = conflicts.reset_index().groupby("dataset").size().to_dict()
            sample = conflicts.head(5).index.tolist()
            pytest.fail(f"Conflicting human labels for same pair: {by_ds}\nSample: {sample}")

    def test_no_conflicting_agent_labels(self):
        """Same pair should not have different agent label values."""
        if not AGENT_DIR.exists():
            pytest.skip("No agent labels directory found")
        df = LabelStore.load_agent_labels(AGENT_DIR)
        if len(df) == 0:
            pytest.skip("No agent labels found")

        label_counts = df.groupby(["dataset", "gers_id", "target_id"])["label"].nunique()
        conflicts = label_counts[label_counts > 1]
        if len(conflicts) > 0:
            by_ds = conflicts.reset_index().groupby("dataset").size().to_dict()
            sample = conflicts.head(5).index.tolist()
            pytest.fail(f"Conflicting agent labels for same pair: {by_ds}\nSample: {sample}")


class TestDatasetConfigCoverage:
    """Validate every label dataset partition has a matching YAML config.

    Catches orphaned partitions from deleted or renamed datasets that would
    add noise to training without a valid config to contextualize them.
    """

    DATASETS_DIR = Path("datasets")

    def _get_config_names(self):
        """Get set of dataset names from YAML configs."""
        if not self.DATASETS_DIR.exists():
            return set()
        return {p.stem for p in self.DATASETS_DIR.glob("*.yaml")}

    def _get_partition_names(self, store_dir):
        """Get set of dataset names from partition directories."""
        if not store_dir.exists():
            return set()
        return {
            d.name.removeprefix("dataset=")
            for d in store_dir.iterdir()
            if d.is_dir() and d.name.startswith("dataset=")
        }

    def test_human_label_datasets_have_configs(self):
        """Every human label dataset has a YAML config."""
        config_names = self._get_config_names()
        partition_names = self._get_partition_names(HUMAN_DIR)
        if not partition_names:
            pytest.skip("No human label partitions found")

        # OSM datasets (e.g., us_boston_streets_osm) don't need their own config —
        # they derive from the base dataset config
        non_osm = {n for n in partition_names if not n.endswith("_osm")}
        orphaned = non_osm - config_names
        if orphaned:
            pytest.fail(f"Human label partitions without dataset config: {sorted(orphaned)}")

    def test_agent_label_datasets_have_configs(self):
        """Every agent label dataset has a YAML config."""
        config_names = self._get_config_names()
        partition_names = self._get_partition_names(AGENT_DIR)
        if not partition_names:
            pytest.skip("No agent label partitions found")

        non_osm = {n for n in partition_names if not n.endswith("_osm")}
        orphaned = non_osm - config_names
        if orphaned:
            pytest.fail(f"Agent label partitions without dataset config: {sorted(orphaned)}")

    def test_feature_datasets_have_configs(self):
        """Every feature dataset has a YAML config."""
        config_names = self._get_config_names()
        partition_names = self._get_partition_names(FEATURES_DIR)
        if not partition_names:
            pytest.skip("No feature partitions found")

        non_osm = {n for n in partition_names if not n.endswith("_osm")}
        orphaned = non_osm - config_names
        if orphaned:
            pytest.fail(f"Feature partitions without dataset config: {sorted(orphaned)}")

    def test_data_datasets_have_configs(self):
        """Every data dataset has a YAML config."""
        config_names = self._get_config_names()
        partition_names = self._get_partition_names(DATA_DIR)
        if not partition_names:
            pytest.skip("No data partitions found")

        non_osm = {n for n in partition_names if not n.endswith("_osm")}
        orphaned = non_osm - config_names
        if orphaned:
            pytest.fail(f"Data partitions without dataset config: {sorted(orphaned)}")


class TestMatchLabelFeatureQuality:
    """Validate that 'match' labels have plausible feature values.

    Catches computation failures that produce error features on match labels,
    or pairs with extreme values that should never have been labeled as matches
    (indicating a labeling or candidate generation bug).
    """

    @pytest.fixture
    def match_features_df(self):
        """Load match labels joined with features."""
        if not HUMAN_DIR.exists() or not FEATURES_DIR.exists():
            pytest.skip("Labels or features directory not found")

        labels = LabelStore.load_human_labels(HUMAN_DIR)
        features = FeatureStore.load_all(FEATURES_DIR)

        if len(labels) == 0:
            pytest.skip("No human labels found")
        if len(features) == 0:
            pytest.skip("No features found")

        match_labels = labels[labels["label"] == "match"]
        if len(match_labels) == 0:
            pytest.skip("No match labels found")

        merged = match_labels.merge(
            features,
            on=["gers_id", "target_id", "dataset"],
            how="inner",
        )
        if len(merged) == 0:
            pytest.skip("No match labels with features found")
        return merged

    def test_no_error_features_on_match_labels(self, match_features_df):
        """Match labels should not have error-default feature values.

        Error features have hausdorff=10000, buffer_iou=0,
        indicating a computation failure that returned all defaults.
        """
        from matcher.config import MAX_DISTANCE_METERS

        df = match_features_df
        error_mask = (
            (df["hausdorff_distance_m"] >= MAX_DISTANCE_METERS)
            & (df["buffer_iou_5m"] == 0.0)
            & (df["buffer_iou_15m"] == 0.0)
        )
        bad = df[error_mask]
        if len(bad) > 0:
            by_dataset = bad.groupby("dataset").size().to_dict()
            pytest.fail(
                f"{len(bad)} match labels have error-default features "
                f"(hausdorff={MAX_DISTANCE_METERS}, buffer_iou=0): {by_dataset}\n"
                f"This indicates feature computation failures. "
                f"Run 'matcher backfill' to recompute."
            )

    def test_match_labels_have_nonzero_overlap(self, match_features_df):
        """Match labels should have some geometric overlap at 15m buffer.

        If buffer_iou_15m is exactly 0.0, the geometries don't overlap
        at all even with a generous buffer, suggesting either a feature
        computation error or an incorrect match label.
        """
        df = match_features_df
        bad = df[df["buffer_iou_15m"] == 0.0]
        if len(bad) > 0:
            pct = len(bad) / len(df) * 100
            by_dataset = bad.groupby("dataset").size().to_dict()
            if pct >= 1.0:
                pytest.fail(
                    f"{len(bad)} match labels ({pct:.1f}%) with zero buffer_iou_15m: "
                    f"{by_dataset}\n"
                    f"These pairs have no geometric overlap and may have "
                    f"error features or incorrect labels."
                )

    def test_match_labels_hausdorff_reasonable(self, match_features_df):
        """Match labels should not have hausdorff at error default (10000m)."""
        from matcher.config import MAX_DISTANCE_METERS

        df = match_features_df
        bad = df[df["hausdorff_distance_m"] >= MAX_DISTANCE_METERS]
        if len(bad) > 0:
            by_dataset = bad.groupby("dataset").size().to_dict()
            pytest.fail(
                f"{len(bad)} match labels with hausdorff_distance >= "
                f"{MAX_DISTANCE_METERS}m: {by_dataset}\n"
                f"This indicates error features on match labels."
            )


class TestGeometryCoordinateBounds:
    """Validate stored geometries have valid WGS84 coordinates.

    Catches CRS projection bugs where coordinates are in projected meters
    instead of degrees, or other spatial transform errors.
    """

    def test_geometries_within_wgs84_bounds(self):
        """All geometry coordinates fall within valid WGS84 range."""
        if not DATA_DIR.exists():
            pytest.skip("No data directory found")

        gdf = DataStore.load_all(DATA_DIR)
        if len(gdf) == 0:
            pytest.skip("No data found")

        bad_rows = []
        for geom_col in ["ref_geometry", "target_geometry"]:
            if geom_col not in gdf.columns:
                continue
            for idx, geom in gdf[geom_col].items():
                if geom is None:
                    continue
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
                    ds = gdf.at[idx, "dataset"] if "dataset" in gdf.columns else "unknown"
                    bad_rows.append((ds, geom_col, bounds))

        if bad_rows:
            by_ds = {}
            for ds, col, _ in bad_rows:
                key = f"{ds}/{col}"
                by_ds[key] = by_ds.get(key, 0) + 1
            sample = bad_rows[:3]
            pytest.fail(
                f"{len(bad_rows)} geometries outside WGS84 bounds: {by_ds}\nSample: {sample}"
            )
