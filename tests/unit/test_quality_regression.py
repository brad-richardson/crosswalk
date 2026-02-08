"""Tests for quality regression detection."""

import geopandas as gpd
from shapely.geometry import LineString

from matcher.datasets.schema import QualityFingerprintConfig
from matcher.quality.regression import check_quality_regression, compute_quick_fingerprint


def _make_fetched_gdf(
    n: int = 100,
    name_ratio: float = 0.8,
    class_ratio: float = 0.9,
) -> gpd.GeoDataFrame:
    """Create a synthetic GeoDataFrame mimicking fetched data.

    Args:
        n: Number of segments
        name_ratio: Fraction with names
        class_ratio: Fraction with non-unknown class
    """
    names = []
    for i in range(n):
        if i < int(n * name_ratio):
            names.append({"primary": f"Street {i}"})
        else:
            names.append(None)

    classes = []
    for i in range(n):
        if i < int(n * class_ratio):
            classes.append("residential")
        else:
            classes.append("unknown")

    return gpd.GeoDataFrame(
        {
            "id": [f"test_{i}" for i in range(n)],
            "names": names,
            "class": classes,
        },
        geometry=[LineString([(i, 0), (i + 0.01, 0.01)]) for i in range(n)],
        crs="EPSG:4326",
    )


def _make_fingerprint(
    total_segments: int = 100,
    name_coverage_ratio: float = 0.8,
    class_coverage_ratio: float = 0.9,
) -> QualityFingerprintConfig:
    """Create a quality fingerprint for testing."""
    return QualityFingerprintConfig(
        total_segments=total_segments,
        name_coverage_ratio=name_coverage_ratio,
        class_coverage_ratio=class_coverage_ratio,
    )


class TestQualityRegression:
    """Tests for check_quality_regression()."""

    def test_no_violations_when_metrics_match(self):
        """No violations when fetched data matches fingerprint."""
        gdf = _make_fetched_gdf(n=100, name_ratio=0.8, class_ratio=0.9)
        fp = _make_fingerprint(
            total_segments=100, name_coverage_ratio=0.8, class_coverage_ratio=0.9
        )

        violations = check_quality_regression(gdf, fp, "test_dataset")
        assert violations == []

    def test_name_coverage_drop_fails(self):
        """Name coverage dropping > 30pp should produce a violation."""
        # Fingerprint says 81% names, actual has 0%
        gdf = _make_fetched_gdf(n=100, name_ratio=0.0)
        fp = _make_fingerprint(name_coverage_ratio=0.81)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        name_violations = [v for v in violations if v.metric == "name_coverage_ratio"]
        assert len(name_violations) == 1
        assert name_violations[0].expected == 0.81
        assert name_violations[0].actual == 0.0

    def test_moderate_name_drop_passes(self):
        """Name coverage dropping < 30pp should not produce a violation."""
        gdf = _make_fetched_gdf(n=100, name_ratio=0.6)
        fp = _make_fingerprint(name_coverage_ratio=0.8)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        name_violations = [v for v in violations if v.metric == "name_coverage_ratio"]
        assert len(name_violations) == 0

    def test_class_coverage_drop_fails(self):
        """Class coverage dropping > 30pp should produce a violation."""
        gdf = _make_fetched_gdf(n=100, class_ratio=0.0)
        fp = _make_fingerprint(class_coverage_ratio=0.9)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        class_violations = [v for v in violations if v.metric == "class_coverage_ratio"]
        assert len(class_violations) == 1

    def test_moderate_segment_change_passes(self):
        """10% change in segment count should pass."""
        gdf = _make_fetched_gdf(n=110)
        fp = _make_fingerprint(total_segments=100)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        seg_violations = [v for v in violations if v.metric == "total_segments"]
        assert len(seg_violations) == 0

    def test_large_segment_change_fails(self):
        """60% change in segment count should fail."""
        gdf = _make_fetched_gdf(n=160)
        fp = _make_fingerprint(total_segments=100)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        seg_violations = [v for v in violations if v.metric == "total_segments"]
        assert len(seg_violations) == 1

    def test_large_segment_decrease_fails(self):
        """70% decrease in segment count should also fail."""
        gdf = _make_fetched_gdf(n=30)
        fp = _make_fingerprint(total_segments=100)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        seg_violations = [v for v in violations if v.metric == "total_segments"]
        assert len(seg_violations) == 1

    def test_no_fingerprint_means_no_check(self):
        """When fingerprint has zero segments, skip segment count check."""
        gdf = _make_fetched_gdf(n=100)
        fp = _make_fingerprint(total_segments=0)

        violations = check_quality_regression(gdf, fp, "test_dataset")
        seg_violations = [v for v in violations if v.metric == "total_segments"]
        assert len(seg_violations) == 0

    def test_multiple_violations(self):
        """Multiple metrics failing at once should produce multiple violations."""
        gdf = _make_fetched_gdf(n=200, name_ratio=0.0, class_ratio=0.0)
        fp = _make_fingerprint(
            total_segments=100,
            name_coverage_ratio=0.8,
            class_coverage_ratio=0.9,
        )

        violations = check_quality_regression(gdf, fp, "test_dataset")
        assert len(violations) == 3  # segments, names, classes


class TestComputeQuickFingerprint:
    """Tests for compute_quick_fingerprint()."""

    def test_computes_correct_metrics(self):
        """Quick fingerprint should capture segment count, name and class coverage."""
        gdf = _make_fetched_gdf(n=100, name_ratio=0.75, class_ratio=0.6)
        fp = compute_quick_fingerprint(gdf)

        assert fp.total_segments == 100
        assert fp.name_coverage_ratio == 0.75
        assert fp.class_coverage_ratio == 0.6
        assert fp.computed_at is not None

    def test_empty_gdf_fingerprint(self):
        """Empty GDF should produce zero metrics."""
        gdf = gpd.GeoDataFrame(
            {"id": [], "names": [], "class": []},
            geometry=[],
            crs="EPSG:4326",
        )
        fp = compute_quick_fingerprint(gdf)

        assert fp.total_segments == 0
        assert fp.name_coverage_ratio == 0.0
        assert fp.class_coverage_ratio == 0.0

    def test_fingerprint_matches_regression_check(self):
        """Quick fingerprint should produce values consistent with regression check."""
        gdf = _make_fetched_gdf(n=200, name_ratio=0.5, class_ratio=0.8)
        fp = compute_quick_fingerprint(gdf)

        # Using the computed fingerprint as baseline should produce no violations
        violations = check_quality_regression(gdf, fp, "test")
        assert violations == []


class TestLastFetchMigration:
    """Test migration from old flat LastFetch to new nested format."""

    def test_old_flat_format_migrates_to_target(self, tmp_path):
        """Old flat last_fetch format should be auto-migrated to target sub-field."""
        import yaml

        from matcher.datasets.schema import load_dataset_config

        # Write YAML with old flat format
        old_config = {
            "name": "test_dataset",
            "last_fetch": {
                "fetched_at": "2024-01-15T10:00:00+00:00",
                "feature_count": 500,
                "geometry_types": ["LineString"],
                "output_path": "data/raw/test.parquet",
            },
        }
        config_path = tmp_path / "test_dataset.yaml"
        with open(config_path, "w") as f:
            yaml.dump(old_config, f)

        # Load and verify migration
        config = load_dataset_config(config_path)
        assert config.last_fetch is not None
        assert config.last_fetch.target is not None
        assert config.last_fetch.target.feature_count == 500
        assert config.last_fetch.reference is None
        assert config.last_fetch.osm is None

    def test_new_nested_format_loads_correctly(self, tmp_path):
        """New nested format should load without migration."""
        import yaml

        from matcher.datasets.schema import load_dataset_config

        new_config = {
            "name": "test_dataset",
            "last_fetch": {
                "target": {
                    "fetched_at": "2024-01-15T10:00:00+00:00",
                    "feature_count": 500,
                },
                "reference": {
                    "fetched_at": "2024-01-16T10:00:00+00:00",
                    "feature_count": 1000,
                },
            },
        }
        config_path = tmp_path / "test_dataset.yaml"
        with open(config_path, "w") as f:
            yaml.dump(new_config, f)

        config = load_dataset_config(config_path)
        assert config.last_fetch is not None
        assert config.last_fetch.target is not None
        assert config.last_fetch.target.feature_count == 500
        assert config.last_fetch.reference is not None
        assert config.last_fetch.reference.feature_count == 1000
        assert config.last_fetch.osm is None
