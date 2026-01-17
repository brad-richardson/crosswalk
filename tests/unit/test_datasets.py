"""Tests for the datasets configuration and discovery module."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from matcher.datasets import (
    ClassMappingRule,
    DatasetConfig,
    apply_class_mapping,
    list_dataset_configs,
    load_dataset_config,
)
from matcher.datasets.config import (
    PhysicalAttributes,
    SourceClassification,
    load_dataset_config_from_file,
    save_dataset_config,
)
from matcher.datasets.discover import (
    _analyze_source_tags,
    _find_column_by_patterns,
    discover_dataset,
)


class TestClassMappingRule:
    """Tests for ClassMappingRule.matches() method."""

    def test_matches_simple_value(self):
        """Rule matches when source value equals expected value."""
        rule = ClassMappingRule(source_value=1, target_class="motorway")
        assert rule.matches({"class": 1}, "class") is True
        assert rule.matches({"class": 2}, "class") is False

    def test_matches_list_of_values(self):
        """Rule matches when source value is in list."""
        rule = ClassMappingRule(source_value=[1, 2, 3], target_class="primary")
        assert rule.matches({"class": 1}, "class") is True
        assert rule.matches({"class": 3}, "class") is True
        assert rule.matches({"class": 5}, "class") is False

    def test_matches_greater_than_condition(self):
        """Rule matches with > condition."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"AADT": ">10000"},
        )
        assert rule.matches({"class": 1, "AADT": 15000}, "class") is True
        assert rule.matches({"class": 1, "AADT": 10000}, "class") is False
        assert rule.matches({"class": 1, "AADT": 5000}, "class") is False

    def test_matches_greater_equal_condition(self):
        """Rule matches with >= condition."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"AADT": ">=10000"},
        )
        assert rule.matches({"class": 1, "AADT": 10000}, "class") is True
        assert rule.matches({"class": 1, "AADT": 9999}, "class") is False

    def test_matches_less_than_condition(self):
        """Rule matches with < condition."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="residential",
            conditions={"lanes": "<3"},
        )
        assert rule.matches({"class": 1, "lanes": 2}, "class") is True
        assert rule.matches({"class": 1, "lanes": 3}, "class") is False

    def test_matches_less_equal_condition(self):
        """Rule matches with <= condition."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="residential",
            conditions={"lanes": "<=2"},
        )
        assert rule.matches({"class": 1, "lanes": 2}, "class") is True
        assert rule.matches({"class": 1, "lanes": 3}, "class") is False

    def test_matches_equality_condition_string(self):
        """Rule matches with == condition for strings."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="motorway",
            conditions={"road_type": "==HIGHWAY"},
        )
        assert rule.matches({"class": 1, "road_type": "HIGHWAY"}, "class") is True
        assert rule.matches({"class": 1, "road_type": "LOCAL"}, "class") is False

    def test_matches_equality_condition_numeric(self):
        """Rule matches with == condition for numeric values."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"lanes": "==4"},
        )
        assert rule.matches({"class": 1, "lanes": 4}, "class") is True
        assert rule.matches({"class": 1, "lanes": 4.0}, "class") is True
        assert rule.matches({"class": 1, "lanes": 3}, "class") is False

    def test_matches_direct_string_condition(self):
        """Rule matches with direct string equality."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"status": "ACTIVE"},
        )
        assert rule.matches({"class": 1, "status": "ACTIVE"}, "class") is True
        assert rule.matches({"class": 1, "status": "INACTIVE"}, "class") is False

    def test_matches_non_string_condition(self):
        """Rule matches with non-string condition value."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"lanes": 4},
        )
        assert rule.matches({"class": 1, "lanes": 4}, "class") is True
        assert rule.matches({"class": 1, "lanes": 3}, "class") is False

    def test_matches_missing_condition_value(self):
        """Rule doesn't match when condition column is missing."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"AADT": ">10000"},
        )
        assert rule.matches({"class": 1}, "class") is False

    def test_matches_invalid_numeric_condition(self):
        """Rule handles invalid numeric condition gracefully."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"AADT": ">invalid"},
        )
        # Should return False and log warning for invalid condition
        assert rule.matches({"class": 1, "AADT": 10000}, "class") is False


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_get_target_class_no_rules(self):
        """Returns default class when no rules defined."""
        config = DatasetConfig(name="test", default_class="residential")
        assert config.get_target_class({"class": 1}) == "residential"

    def test_get_target_class_no_source_classification(self):
        """Returns default class when no source classification."""
        config = DatasetConfig(
            name="test",
            default_class="residential",
            class_mapping_rules=[ClassMappingRule(source_value=1, target_class="motorway")],
        )
        assert config.get_target_class({"class": 1}) == "residential"

    def test_get_target_class_matches_rule(self):
        """Returns mapped class when rule matches."""
        config = DatasetConfig(
            name="test",
            default_class="residential",
            source_classification=SourceClassification(column="F_CLASS"),
            class_mapping_rules=[
                ClassMappingRule(source_value=1, target_class="motorway"),
                ClassMappingRule(source_value=2, target_class="trunk"),
            ],
        )
        assert config.get_target_class({"F_CLASS": 1}) == "motorway"
        assert config.get_target_class({"F_CLASS": 2}) == "trunk"
        assert config.get_target_class({"F_CLASS": 99}) == "residential"

    def test_get_target_class_priority_order(self):
        """Higher priority rules are checked first."""
        config = DatasetConfig(
            name="test",
            default_class="residential",
            source_classification=SourceClassification(column="F_CLASS"),
            class_mapping_rules=[
                ClassMappingRule(source_value=3, target_class="secondary", priority=80),
                ClassMappingRule(
                    source_value=3,
                    target_class="primary",
                    conditions={"AADT": ">10000"},
                    priority=85,
                ),
            ],
        )
        # High AADT matches higher priority rule
        assert config.get_target_class({"F_CLASS": 3, "AADT": 15000}) == "primary"
        # Low AADT falls through to lower priority rule
        assert config.get_target_class({"F_CLASS": 3, "AADT": 5000}) == "secondary"


class TestYAMLLoading:
    """Tests for YAML configuration loading."""

    def test_load_valid_config(self):
        """Load a valid YAML configuration."""
        # boston_streets.yaml should be available
        config = load_dataset_config("boston_streets")
        assert config is not None
        assert config.name == "boston_streets"
        assert config.source_classification is not None
        assert config.source_classification.column == "F_F_CLASS"
        assert len(config.class_mapping_rules) > 0

    def test_load_nonexistent_config(self):
        """Return None for nonexistent config."""
        config = load_dataset_config("nonexistent_dataset_xyz")
        assert config is None

    def test_load_invalid_yaml_schema(self):
        """Raise error for invalid YAML schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("[]")  # List instead of dict
            f.flush()
            with pytest.raises(ValueError, match="must be a dict"):
                load_dataset_config_from_file(Path(f.name))

    def test_load_empty_yaml(self):
        """Raise error for empty YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(ValueError, match="Empty or invalid"):
                load_dataset_config_from_file(Path(f.name))

    def test_load_rule_missing_source_value(self):
        """Raise error for rule missing source_value."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test\nclass_mapping_rules:\n  - target_class: primary\n")
            f.flush()
            with pytest.raises(ValueError, match="missing 'source_value'"):
                load_dataset_config_from_file(Path(f.name))

    def test_load_rule_missing_target_class(self):
        """Raise error for rule missing target_class."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test\nclass_mapping_rules:\n  - source_value: 1\n")
            f.flush()
            with pytest.raises(ValueError, match="missing 'target_class'"):
                load_dataset_config_from_file(Path(f.name))


class TestYAMLSaving:
    """Tests for YAML configuration saving."""

    def test_save_and_load_roundtrip(self):
        """Config survives save/load roundtrip."""
        config = DatasetConfig(
            name="test_dataset",
            description="Test description",
            source_classification=SourceClassification(
                column="CLASS",
                description="Test classification",
                values={1: {"description": "Type 1"}},
            ),
            physical_attributes=PhysicalAttributes(
                lanes_column="NUM_LANES",
                speed_column="SPEED_LIM",
            ),
            class_mapping_rules=[
                ClassMappingRule(source_value=1, target_class="motorway", priority=100),
                ClassMappingRule(
                    source_value=2,
                    target_class="primary",
                    conditions={"AADT": ">10000"},
                ),
            ],
            default_class="residential",
            confidence="high",
            notes="Test notes",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        try:
            save_dataset_config(config, path)
            loaded = load_dataset_config_from_file(path)

            assert loaded.name == config.name
            assert loaded.description == config.description
            assert loaded.default_class == config.default_class
            assert loaded.confidence == config.confidence
            assert loaded.notes == config.notes
            assert loaded.source_classification.column == config.source_classification.column
            assert loaded.physical_attributes.lanes_column == "NUM_LANES"
            assert len(loaded.class_mapping_rules) == 2
        finally:
            path.unlink()


class TestApplyClassMapping:
    """Tests for apply_class_mapping function."""

    def test_apply_mapping_basic(self):
        """Apply basic class mapping to GeoDataFrame."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b", "c"],
                "F_CLASS": [1, 2, 3],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            }
        )

        config = DatasetConfig(
            name="test",
            source_classification=SourceClassification(column="F_CLASS"),
            class_mapping_rules=[
                ClassMappingRule(source_value=1, target_class="motorway"),
                ClassMappingRule(source_value=2, target_class="primary"),
            ],
            default_class="residential",
        )

        result = apply_class_mapping(gdf, config, class_column="road_class")
        assert result["road_class"].tolist() == ["motorway", "primary", "residential"]

    def test_apply_mapping_with_source_tags(self):
        """Apply mapping using source_tags column."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b"],
                "source_tags": [{"F_CLASS": 1}, {"F_CLASS": 2}],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            }
        )

        config = DatasetConfig(
            name="test",
            source_classification=SourceClassification(column="F_CLASS"),
            class_mapping_rules=[
                ClassMappingRule(source_value=1, target_class="motorway"),
                ClassMappingRule(source_value=2, target_class="trunk"),
            ],
            default_class="residential",
        )

        result = apply_class_mapping(gdf, config)
        assert result["class"].tolist() == ["motorway", "trunk"]

    def test_apply_mapping_no_rules(self):
        """Return original GeoDataFrame when no rules defined."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )
        config = DatasetConfig(name="test")
        result = apply_class_mapping(gdf, config)
        assert "class" not in result.columns or result.equals(gdf)


class TestListDatasetConfigs:
    """Tests for list_dataset_configs function."""

    def test_list_configs_includes_boston(self):
        """Boston streets config should be listed."""
        configs = list_dataset_configs()
        assert "boston_streets" in configs


class TestDiscovery:
    """Tests for dataset discovery functions."""

    def test_find_column_by_patterns(self):
        """Find column matching pattern."""
        columns = ["ID", "F_CLASS", "NAME", "GEOMETRY"]
        assert _find_column_by_patterns(columns, ["class", "f_class"]) == "F_CLASS"
        assert _find_column_by_patterns(columns, ["nonexistent"]) is None

    def test_analyze_source_tags(self):
        """Analyze source_tags column."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b", "c"],
                "source_tags": [
                    {"CLASS": 1, "LANES": 2},
                    {"CLASS": 2, "LANES": 4},
                    {"CLASS": 1, "SPEED": 35},
                ],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            }
        )

        result = _analyze_source_tags(gdf)
        assert "keys" in result
        assert "CLASS" in result["keys"]
        assert result["detected_class_key"] == "CLASS"
        assert result["detected_physical_attrs"].get("lanes") == "LANES"

    def test_analyze_source_tags_missing_column(self):
        """Return empty dict when source_tags missing."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )
        result = _analyze_source_tags(gdf)
        assert result == {}

    def test_discover_dataset_basic(self):
        """Discover dataset schema and classification."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            gdf = gpd.GeoDataFrame(
                {
                    "id": ["a", "b"],
                    "class": ["primary", "secondary"],
                    "lanes": [2, 4],
                    "geometry": [
                        LineString([(0, 0), (1, 1)]),
                        LineString([(1, 1), (2, 2)]),
                    ],
                },
                crs="EPSG:4326",
            )
            gdf.to_parquet(f.name)

            try:
                report = discover_dataset(Path(f.name))
                assert report.total_rows == 2
                assert report.detected_class_column == "class"
                assert "lanes" in report.detected_physical_attrs
                assert report.suggested_config is not None
            finally:
                Path(f.name).unlink()

    def test_discover_empty_dataset(self):
        """Handle empty dataset gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            gdf = gpd.GeoDataFrame(
                {
                    "id": pd.Series([], dtype=str),
                    "class": pd.Series([], dtype=str),
                    "geometry": gpd.GeoSeries([], crs="EPSG:4326"),
                },
                crs="EPSG:4326",
            )
            gdf.to_parquet(f.name)

            try:
                report = discover_dataset(Path(f.name))
                assert report.total_rows == 0
                assert report.suggested_config is not None
            finally:
                Path(f.name).unlink()
