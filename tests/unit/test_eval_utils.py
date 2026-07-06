"""Tests for eval_utils module (LOO-by-type CV dataset classification)."""

from unittest.mock import patch

from crosswalk.datasets.schema import DatasetConfig, QualityFingerprintConfig
from crosswalk.eval_utils import (
    TYPE_OVERRIDES,
    build_type_groups,
    classify_dataset_type_group,
)


def _make_config(name: str, type_: str = "road", name_cov: float = 0.8, class_cov: float = 0.8):
    """Create a DatasetConfig with optional quality fingerprint."""
    return DatasetConfig(
        name=name,
        type=type_,
        quality_fingerprint=QualityFingerprintConfig(
            name_coverage_ratio=name_cov,
            class_coverage_ratio=class_cov,
        ),
    )


class TestClassifyDatasetTypeGroup:
    def test_road_good(self):
        config = _make_config("us_boston_streets", name_cov=0.9, class_cov=0.7)
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("us_boston_streets") == "road_good"

    def test_road_poor_low_name_coverage(self):
        config = _make_config("hk_hongkong_roads", name_cov=0.3, class_cov=0.9)
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("hk_hongkong_roads") == "road_poor"

    def test_road_poor_low_class_coverage(self):
        config = _make_config("some_road", name_cov=0.9, class_cov=0.2)
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("some_road") == "road_poor"

    def test_sidewalk(self):
        config = _make_config("us_boston_sidewalks", type_="sidewalk")
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("us_boston_sidewalks") == "sidewalk"

    def test_bike_type(self):
        config = _make_config("some_bike", type_="bike")
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("some_bike") == "other"

    def test_trail_type(self):
        config = _make_config("some_trail", type_="trail")
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("some_trail") == "other"

    def test_type_override_reclassification(self):
        """us_boston_bike_network is typed as 'road' in YAML but overridden to 'bike'."""
        assert "us_boston_bike_network" in TYPE_OVERRIDES
        # Even with a road config, the override should classify it as 'other'
        config = _make_config("us_boston_bike_network", type_="road", name_cov=0.9, class_cov=0.9)
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("us_boston_bike_network") == "other"

    def test_missing_config_returns_other(self):
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=None):
            assert classify_dataset_type_group("nonexistent_dataset") == "other"

    def test_missing_quality_fingerprint_returns_road_poor(self):
        config = DatasetConfig(name="some_road", type="road")
        assert config.quality_fingerprint is None
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            assert classify_dataset_type_group("some_road") == "road_poor"

    def test_custom_quality_threshold(self):
        config = _make_config("edge_case", name_cov=0.4, class_cov=0.4)
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            # Default threshold 0.5 -> road_poor
            assert classify_dataset_type_group("edge_case", quality_threshold=0.5) == "road_poor"
            # Lower threshold -> road_good
            assert classify_dataset_type_group("edge_case", quality_threshold=0.3) == "road_good"

    def test_exact_threshold_boundary(self):
        config = _make_config("boundary", name_cov=0.5, class_cov=0.5)
        with patch("crosswalk.eval_utils.get_dataset_config", return_value=config):
            # min(0.5, 0.5) = 0.5 >= 0.5 -> road_good
            assert classify_dataset_type_group("boundary", quality_threshold=0.5) == "road_good"


class TestBuildTypeGroups:
    def test_mixed_datasets(self):
        configs = {
            "road_a": _make_config("road_a", name_cov=0.9, class_cov=0.9),
            "road_b": _make_config("road_b", name_cov=0.1, class_cov=0.1),
            "sw_a": _make_config("sw_a", type_="sidewalk"),
            "trail_a": _make_config("trail_a", type_="trail"),
        }

        def mock_get(name):
            return configs.get(name)

        with patch("crosswalk.eval_utils.get_dataset_config", side_effect=mock_get):
            groups = build_type_groups(["road_a", "road_b", "sw_a", "trail_a"])

        assert groups["road_good"] == ["road_a"]
        assert groups["road_poor"] == ["road_b"]
        assert groups["sidewalk"] == ["sw_a"]
        assert groups["other"] == ["trail_a"]

    def test_empty_input(self):
        groups = build_type_groups([])
        assert groups == {}

    def test_preserves_order_within_group(self):
        configs = {
            "a_road": _make_config("a_road", name_cov=0.9, class_cov=0.9),
            "b_road": _make_config("b_road", name_cov=0.9, class_cov=0.9),
            "c_road": _make_config("c_road", name_cov=0.9, class_cov=0.9),
        }

        def mock_get(name):
            return configs.get(name)

        with patch("crosswalk.eval_utils.get_dataset_config", side_effect=mock_get):
            groups = build_type_groups(["a_road", "b_road", "c_road"])

        assert groups["road_good"] == ["a_road", "b_road", "c_road"]

    def test_only_populated_groups_returned(self):
        configs = {
            "road_a": _make_config("road_a", name_cov=0.9, class_cov=0.9),
        }

        def mock_get(name):
            return configs.get(name)

        with patch("crosswalk.eval_utils.get_dataset_config", side_effect=mock_get):
            groups = build_type_groups(["road_a"])

        assert "road_good" in groups
        assert "road_poor" not in groups
        assert "sidewalk" not in groups
        assert "other" not in groups
