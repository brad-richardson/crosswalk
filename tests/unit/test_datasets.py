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

    @pytest.mark.parametrize(
        "source_value,row_value,expected",
        [
            (1, 1, True),  # exact match
            (1, 2, False),  # no match
            ([1, 2, 3], 1, True),  # list match - first
            ([1, 2, 3], 3, True),  # list match - last
            ([1, 2, 3], 5, False),  # list no match
        ],
        ids=["exact_match", "no_match", "list_first", "list_last", "list_no_match"],
    )
    def test_source_value_matching(self, source_value, row_value, expected):
        """Rule matches based on source value (single or list)."""
        rule = ClassMappingRule(source_value=source_value, target_class="primary")
        assert rule.matches({"class": row_value}, "class") is expected

    @pytest.mark.parametrize(
        "condition,test_value,expected",
        [
            # Greater than
            (">10000", 15000, True),
            (">10000", 10000, False),
            (">10000", 5000, False),
            # Greater or equal
            (">=10000", 10000, True),
            (">=10000", 9999, False),
            # Less than
            ("<3", 2, True),
            ("<3", 3, False),
            # Less or equal
            ("<=2", 2, True),
            ("<=2", 3, False),
            # Equality (numeric)
            ("==4", 4, True),
            ("==4", 4.0, True),
            ("==4", 3, False),
            # Equality (string)
            ("==HIGHWAY", "HIGHWAY", True),
            ("==HIGHWAY", "LOCAL", False),
        ],
        ids=[
            "gt_above",
            "gt_equal",
            "gt_below",
            "gte_equal",
            "gte_below",
            "lt_below",
            "lt_equal",
            "lte_equal",
            "lte_above",
            "eq_int",
            "eq_float",
            "eq_no_match",
            "eq_str_match",
            "eq_str_no_match",
        ],
    )
    def test_condition_operators(self, condition, test_value, expected):
        """Condition operators should evaluate correctly."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions={"cond_col": condition},
        )
        assert rule.matches({"class": 1, "cond_col": test_value}, "class") is expected

    @pytest.mark.parametrize(
        "conditions,row,expected",
        [
            # Direct string equality (no operator)
            ({"status": "ACTIVE"}, {"class": 1, "status": "ACTIVE"}, True),
            ({"status": "ACTIVE"}, {"class": 1, "status": "INACTIVE"}, False),
            # Non-string direct value
            ({"lanes": 4}, {"class": 1, "lanes": 4}, True),
            ({"lanes": 4}, {"class": 1, "lanes": 3}, False),
            # Missing condition column
            ({"AADT": ">10000"}, {"class": 1}, False),
            # Invalid numeric condition
            ({"AADT": ">invalid"}, {"class": 1, "AADT": 10000}, False),
        ],
        ids=[
            "direct_str_match",
            "direct_str_no_match",
            "direct_int_match",
            "direct_int_no_match",
            "missing_column",
            "invalid_numeric",
        ],
    )
    def test_condition_edge_cases(self, conditions, row, expected):
        """Edge cases for condition matching."""
        rule = ClassMappingRule(
            source_value=1,
            target_class="primary",
            conditions=conditions,
        )
        assert rule.matches(row, "class") is expected


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


def _load_all_dataset_yamls():
    """Load raw YAML data for all dataset configs."""
    import yaml

    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(datasets_dir.glob("*.yaml"))}


# Overture Maps transportation segment schema enums
# https://github.com/OvertureMaps/schema/blob/dev/packages/overture-schema-transportation-theme/src/overture/schema/transportation/enums.py
VALID_CLASSES = {
    "motorway", "primary", "secondary", "tertiary", "residential",
    "living_street", "trunk", "unclassified", "service", "pedestrian",
    "footway", "steps", "path", "track", "cycleway", "bridleway", "unknown",
}  # fmt: skip

VALID_SUBCLASSES = {
    "link", "sidewalk", "crosswalk", "parking_aisle",
    "driveway", "alley", "cycle_crossing",
}  # fmt: skip


class TestOvertureSchemaMappings:
    """Validate that all dataset class/subclass mappings use valid Overture values."""

    ALL_YAMLS = _load_all_dataset_yamls()

    # Datasets with class_mapping in fetch config
    CLASS_MAPPED = [
        (name, data)
        for name, data in ALL_YAMLS.items()
        if data.get("fetch", {}).get("class_mapping")
    ]

    # Datasets with subclass_mapping in fetch config
    SUBCLASS_MAPPED = [
        (name, data)
        for name, data in ALL_YAMLS.items()
        if data.get("fetch", {}).get("subclass_mapping")
    ]

    # Datasets with class_mapping_rules in classification config
    RULES_MAPPED = [
        (name, data)
        for name, data in ALL_YAMLS.items()
        if data.get("classification", {}).get("class_mapping_rules")
    ]

    @pytest.mark.parametrize("name,data", CLASS_MAPPED, ids=[n for n, _ in CLASS_MAPPED])
    def test_class_mapping_values(self, name, data):
        """class_mapping targets must be valid Overture RoadClass values."""
        for src, tgt in data["fetch"]["class_mapping"].items():
            assert tgt in VALID_CLASSES, f"{name}: '{src}' maps to invalid class '{tgt}'"

    @pytest.mark.parametrize("name,data", SUBCLASS_MAPPED, ids=[n for n, _ in SUBCLASS_MAPPED])
    def test_subclass_mapping_values(self, name, data):
        """subclass_mapping targets must be valid Overture Subclass values."""
        for src, tgt in data["fetch"]["subclass_mapping"].items():
            assert tgt in VALID_SUBCLASSES, f"{name}: '{src}' maps to invalid subclass '{tgt}'"

    @pytest.mark.parametrize("name,data", RULES_MAPPED, ids=[n for n, _ in RULES_MAPPED])
    def test_class_mapping_rules_values(self, name, data):
        """class_mapping_rules target_class must be valid Overture RoadClass values."""
        for rule in data["classification"]["class_mapping_rules"]:
            tgt = rule.get("target_class")
            if tgt:
                assert tgt in VALID_CLASSES, f"{name}: rule target_class '{tgt}' is invalid"


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
