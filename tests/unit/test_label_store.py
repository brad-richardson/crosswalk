"""Tests for label store and feature parity."""

from shapely.geometry import LineString

from matcher.features.compute import ALL_FEATURE_COLUMNS, compute_pair_features
from matcher.labeling.label_store import LABEL_COLUMNS, LabelStore
from matcher.matching.ml import FEATURE_COLUMNS


class TestFeatureParity:
    """Ensure computed features match what gets saved to labels.

    This is a critical invariant: any feature computed during matching
    must also be saved to labels, otherwise ML training can't use it.
    """

    def test_all_computed_features_are_in_label_columns(self):
        """Every feature in ALL_FEATURE_COLUMNS must be in LABEL_COLUMNS.

        This prevents the bug where a new feature is added to compute.py
        but forgotten in label_store.py, causing labels to be incomplete.
        """
        computed_features = set(ALL_FEATURE_COLUMNS)
        label_features = set(LABEL_COLUMNS)

        missing_from_labels = computed_features - label_features
        assert not missing_from_labels, (
            f"Features computed but not saved to labels: {sorted(missing_from_labels)}\n"
            f"Add these to LABEL_COLUMNS in label_store.py and the add() method."
        )

    def test_compute_pair_features_returns_all_declared_features(self):
        """compute_pair_features() should return all features in ALL_FEATURE_COLUMNS.

        This catches bugs where a feature is declared but not actually computed.
        """
        from shapely import LineString

        # Create simple test geometries
        ref_geom = LineString([(0, 0), (100, 0)])
        target_geom = LineString([(0, 5), (100, 5)])

        features = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name="Main Street",
            target_name="Main St",
            ref_class="residential",
            target_class="residential",
        )

        declared_features = set(ALL_FEATURE_COLUMNS)
        computed_features = set(features.keys())

        missing_from_output = declared_features - computed_features
        assert not missing_from_output, (
            f"Features declared in ALL_FEATURE_COLUMNS but not returned by "
            f"compute_pair_features: {sorted(missing_from_output)}"
        )

        extra_in_output = computed_features - declared_features
        assert not extra_in_output, (
            f"Features returned by compute_pair_features but not declared in "
            f"ALL_FEATURE_COLUMNS: {sorted(extra_in_output)}\n"
            f"Add these to ALL_FEATURE_COLUMNS in compute.py."
        )

    def test_label_store_add_saves_all_computed_features(self):
        """The add() method should explicitly handle all computed features.

        This is a documentation/reminder test - if this fails, it means
        a feature exists in LABEL_COLUMNS but isn't being explicitly
        saved in the add() method (relying on defaults instead).
        """
        # Features that should be explicitly handled in add()
        # (not metadata columns like gers_id, label, versioning, etc.)
        feature_columns = [
            col
            for col in LABEL_COLUMNS
            if col
            not in {
                "gers_id",
                "target_id",
                "label",
                "labeler",
                "labeled_at",
                "session_id",
                "original_decision",
                "original_confidence",
                "ref_start_pct",
                "ref_end_pct",
                "target_start_pct",
                "target_end_pct",
                "is_subsegment",
                # Data versioning columns (metadata, not ML features)
                "ref_data_version",
                "target_data_version",
                "feature_version",
            }
        ]

        # All feature columns should be in ALL_FEATURE_COLUMNS
        # (which defines what compute_pair_features returns)
        for col in feature_columns:
            assert col in ALL_FEATURE_COLUMNS, (
                f"LABEL_COLUMNS has '{col}' but it's not in ALL_FEATURE_COLUMNS.\n"
                f"Either add it to ALL_FEATURE_COLUMNS or remove from LABEL_COLUMNS."
            )

    def test_ml_feature_columns_match_computed_features(self):
        """ML FEATURE_COLUMNS must match ALL_FEATURE_COLUMNS.

        This ensures the ML model uses the same features that are computed
        and saved to labels. A mismatch would cause training failures or
        incorrect predictions.
        """
        ml_features = set(FEATURE_COLUMNS)
        computed_features = set(ALL_FEATURE_COLUMNS)

        missing_from_ml = computed_features - ml_features
        assert not missing_from_ml, (
            f"Features computed but not used by ML model: {sorted(missing_from_ml)}\n"
            f"Add these to FEATURE_COLUMNS in ml.py."
        )

        extra_in_ml = ml_features - computed_features
        assert not extra_in_ml, (
            f"ML model uses features that are not computed: {sorted(extra_in_ml)}\n"
            f"Either add these to ALL_FEATURE_COLUMNS in compute.py or remove from ml.py."
        )


class TestGeometryPersistence:
    """Test that LabelStore.add() with geometry params creates companion file."""

    def test_add_with_geometry_creates_companion_file(self, tmp_path):
        """Adding a label with geometry params persists to label_geometries/."""
        import shutil

        from matcher.labeling.geometry_store import DEFAULT_GEOMETRIES_DIR, GeometryStore

        labels_dir = tmp_path / "labels"

        store = LabelStore("test_dataset_geo_persist", labels_dir=labels_dir)

        ref_geom = LineString([(0.0, 0.0), (1.0, 1.0)])
        target_geom = LineString([(0.0, 0.1), (1.0, 1.1)])

        try:
            store.add(
                gers_id="ref-001",
                target_id="target-001",
                label="match",
                labeler="tester",
                session_id="sess-001",
                original_decision="review",
                original_confidence=0.75,
                features={col: 0.5 for col in ALL_FEATURE_COLUMNS},
                ref_geometry=ref_geom,
                target_geometry=target_geom,
                ref_name_raw="Main St",
                target_name_raw="Main Street",
                ref_class_raw="residential",
                target_class_raw="residential",
                ref_subclass="urban",
                target_subclass="urban",
            )

            # Label should be saved
            assert len(store.df) == 1
            assert store.df.iloc[0]["gers_id"] == "ref-001"

            # Companion file should exist in the default geometry dir
            geo_store = GeometryStore("test_dataset_geo_persist")
            geo_path = DEFAULT_GEOMETRIES_DIR / "dataset=test_dataset_geo_persist" / "data.csv"
            assert geo_path.exists(), f"Companion file not created at {geo_path}"

            # Verify geometry was persisted correctly
            result = geo_store.get_pair("ref-001", "target-001")
            assert result is not None
            assert result["ref_name"] == "Main St"
            assert isinstance(result["ref_geometry"], LineString)
        finally:
            # Clean up companion file created in CWD
            geo_partition = DEFAULT_GEOMETRIES_DIR / "dataset=test_dataset_geo_persist"
            if geo_partition.exists():
                shutil.rmtree(geo_partition)
            # Remove parent dir if empty
            if DEFAULT_GEOMETRIES_DIR.exists() and not any(DEFAULT_GEOMETRIES_DIR.iterdir()):
                DEFAULT_GEOMETRIES_DIR.rmdir()

    def test_add_without_geometry_no_error(self, tmp_path):
        """Adding a label without geometry params does not raise an error."""
        labels_dir = tmp_path / "labels"

        store = LabelStore("test_dataset", labels_dir=labels_dir)

        store.add(
            gers_id="ref-001",
            target_id="target-001",
            label="match",
            labeler="tester",
            session_id="sess-001",
            original_decision="review",
            original_confidence=0.75,
            features={col: 0.5 for col in ALL_FEATURE_COLUMNS},
        )

        # No error should be raised, label should be saved
        assert len(store.df) == 1
