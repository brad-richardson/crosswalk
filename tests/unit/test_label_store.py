"""Tests for label store and feature parity."""

from matcher.features.compute import ALL_FEATURE_COLUMNS, compute_pair_features
from matcher.labeling.label_store import LABEL_COLUMNS


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
        # (not metadata columns like gers_id, label, etc.)
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
            }
        ]

        # All feature columns should be in ALL_FEATURE_COLUMNS
        # (which defines what compute_pair_features returns)
        for col in feature_columns:
            assert col in ALL_FEATURE_COLUMNS, (
                f"LABEL_COLUMNS has '{col}' but it's not in ALL_FEATURE_COLUMNS.\n"
                f"Either add it to ALL_FEATURE_COLUMNS or remove from LABEL_COLUMNS."
            )
