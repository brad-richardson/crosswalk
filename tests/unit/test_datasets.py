"""Tests for the datasets configuration and discovery module."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from crosswalk.datasets import (
    ClassMappingRule,
    DatasetConfig,
    apply_class_mapping,
    list_dataset_configs,
    load_dataset_config,
)
from crosswalk.datasets.config import (
    PhysicalAttributes,
    SourceClassification,
    load_dataset_config_from_file,
    save_dataset_config,
)
from crosswalk.datasets.discover import (
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

    def test_quality_hold_survives_schema_roundtrip(self, tmp_path):
        """The persisted publishing hold (quality_hold) must not be silently
        dropped when a fetch/fingerprint update re-saves the YAML through the
        pydantic schema — that is what makes the hold *persisted*."""
        from crosswalk.datasets.schema import (
            DatasetConfig as SchemaDatasetConfig,
        )
        from crosswalk.datasets.schema import (
            QualityHoldConfig,
        )
        from crosswalk.datasets.schema import (
            load_dataset_config as schema_load,
        )
        from crosswalk.datasets.schema import (
            save_dataset_config as schema_save,
        )

        config = SchemaDatasetConfig(
            name="held_ds",
            quality_hold=QualityHoldConfig(reason="cross-mode defect", since="2026-07-06"),
        )
        path = tmp_path / "held_ds.yaml"
        schema_save(config, path)
        loaded = schema_load(path)
        assert loaded.quality_hold is not None
        assert loaded.quality_hold.reason == "cross-mode defect"
        assert loaded.quality_hold.since == "2026-07-06"
        # A second save/load (what update_last_fetch / update_quality_fingerprint
        # do) must keep it too.
        schema_save(loaded, path)
        assert schema_load(path).quality_hold.since == "2026-07-06"


# A hand-curated config: full-line comments, inline comments, a comment
# directly above the machine-owned block, keys both before and after it,
# and a blank-line separator that must survive the surgical update.
COMMENTED_CONFIG = """\
# Curation notes for the Bogotá surface-code mapping — do not lose me.
name: commented_ds
display_name: Commented Dataset  # inline: display comment
type: road
fetch:
  id_column: RID_8  # inline: RID_8 prefix derivation notes
  # full-line comment inside a human-authored block
  class_column: strassenklasse2
# comment immediately above the machine-owned block
last_fetch:
  target:
    fetched_at: '2026-01-01T00:00:00+00:00'
    feature_count: 10

# hold placed 2026-07-06 pending cross-mode fix
quality_hold:
  reason: cross-mode defect  # inline: trailing comment on held reason
  since: '2026-07-06'
notes: trailing human notes
"""


class TestSurgicalYamlUpdate:
    """Fetch-style updates must not strip comments or reflow human YAML (#339)."""

    def _datasets_dir(self, monkeypatch, tmp_path):
        from crosswalk.datasets import schema as schema_module

        monkeypatch.setattr(schema_module, "get_datasets_dir", lambda: tmp_path)
        return tmp_path

    def _run_update(self, name):
        from datetime import UTC, datetime

        from crosswalk.datasets.schema import update_last_fetch

        return update_last_fetch(
            name,
            fetch_type="reference",
            fetched_at=datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC),
            bbox=(-71.2, 42.2, -70.9, 42.4),
            feature_count=42,
            geometry_types=["LineString"],
            output_path="data/raw",
        )

    def test_update_preserves_all_human_bytes(self, monkeypatch, tmp_path):
        """Everything outside the owned last_fetch block stays byte-identical."""
        from crosswalk.datasets.schema import load_dataset_config

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "commented_ds.yaml"
        path.write_text(COMMENTED_CONFIG)

        result = self._run_update("commented_ds")
        assert result is not None
        after = path.read_text()

        # Bytes before the owned block are untouched
        prefix = COMMENTED_CONFIG.split("last_fetch:\n")[0]
        assert after.startswith(prefix)
        # Bytes after the owned block — including the blank-line separator and
        # the comment lines around quality_hold — are untouched
        suffix = (
            "\n# hold placed 2026-07-06 pending cross-mode fix\n"
            + COMMENTED_CONFIG.split("\n# hold placed 2026-07-06 pending cross-mode fix\n")[1]
        )
        assert after.endswith(suffix)
        # Every comment survives byte-identical
        for line in COMMENTED_CONFIG.splitlines():
            if "#" in line:
                assert line in after.splitlines()

        # The owned block itself carries the new values (and keeps target)
        reloaded = load_dataset_config(path)
        assert reloaded.last_fetch.reference.feature_count == 42
        assert reloaded.last_fetch.reference.bbox == (-71.2, 42.2, -70.9, 42.4)
        assert reloaded.last_fetch.target.feature_count == 10
        assert reloaded.quality_hold.since == "2026-07-06"
        assert reloaded.notes == "trailing human notes"

    def test_update_appends_block_when_absent(self, monkeypatch, tmp_path):
        """Missing owned block is appended; file without trailing newline is fine."""
        from crosswalk.datasets.schema import load_dataset_config

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "no_fetch_ds.yaml"
        original = "# keep me\nname: no_fetch_ds\ntype: road  # inline"
        path.write_text(original)  # no trailing newline, no last_fetch

        assert self._run_update("no_fetch_ds") is not None
        after = path.read_text()
        assert after.startswith(original + "\n")
        assert "\nlast_fetch:\n" in after
        assert "# keep me" in after and "# inline" in after
        assert load_dataset_config(path).last_fetch.reference.feature_count == 42

    def test_update_when_block_is_last_in_file(self, monkeypatch, tmp_path):
        """Owned block at EOF is replaced in place; preceding bytes untouched."""
        from crosswalk.datasets.schema import load_dataset_config

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "tail_ds.yaml"
        head = "name: tail_ds  # inline comment\ntype: road\n"
        path.write_text(
            head + "last_fetch:\n  target:\n    fetched_at: '2026-01-01T00:00:00+00:00'\n"
        )

        assert self._run_update("tail_ds") is not None
        after = path.read_text()
        assert after.startswith(head)
        reloaded = load_dataset_config(path)
        assert reloaded.last_fetch.reference.feature_count == 42
        assert reloaded.last_fetch.target is not None

    def test_comment_free_file_roundtrips(self, monkeypatch, tmp_path):
        """A machine-written (comment-free) config still updates correctly."""
        from crosswalk.datasets.schema import (
            DatasetConfig as SchemaDatasetConfig,
        )
        from crosswalk.datasets.schema import (
            load_dataset_config,
            save_dataset_config,
        )

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "plain_ds.yaml"
        save_dataset_config(SchemaDatasetConfig(name="plain_ds", type="road"), path)
        before = path.read_text()

        assert self._run_update("plain_ds") is not None
        after = path.read_text()
        assert after.startswith(before)  # original keys byte-identical, block appended
        assert load_dataset_config(path).last_fetch.reference.output_path == "data/raw"

    def test_quality_fingerprint_update_preserves_comments(self, monkeypatch, tmp_path):
        """The other machine-owned block (quality_fingerprint) is surgical too."""
        from datetime import UTC, datetime

        from crosswalk.datasets.schema import (
            QualityFingerprintConfig,
            load_dataset_config,
            update_quality_fingerprint,
        )

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "fp_ds.yaml"
        content = (
            "# provenance comment\n"
            "name: fp_ds\n"
            "quality_fingerprint:\n"
            "  computed_at: '2026-01-01T00:00:00+00:00'\n"
            "  total_segments: 1\n"
            "quality_hold:\n"
            "  reason: keep me  # inline hold comment\n"
        )
        path.write_text(content)

        fp = QualityFingerprintConfig(
            computed_at=datetime(2026, 7, 7, tzinfo=UTC), total_segments=99
        )
        assert update_quality_fingerprint("fp_ds", fp) is not None

        after = path.read_text()
        assert after.startswith("# provenance comment\nname: fp_ds\n")
        assert after.endswith("quality_hold:\n  reason: keep me  # inline hold comment\n")
        reloaded = load_dataset_config(path)
        assert reloaded.quality_fingerprint.total_segments == 99
        assert reloaded.quality_hold.reason == "keep me"

    def test_mid_block_column0_comment_does_not_corrupt(self, monkeypatch, tmp_path):
        """A column-0 comment splitting the owned block must not orphan a stale
        indented tail that shadows the freshly written values."""
        import yaml as yaml_module

        from crosswalk.datasets.schema import load_dataset_config

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "midcomment_ds.yaml"
        path.write_text(
            "name: midcomment_ds  # keep this inline comment\n"
            "last_fetch:\n"
            "  target:\n"
            "    fetched_at: '2026-01-01T00:00:00+00:00'\n"
            "# stray column-0 comment inside the machine-owned block\n"
            "    feature_count: 1\n"
            "quality_hold:\n"
            "  reason: keep me\n"
        )

        assert self._run_update("midcomment_ds") is not None
        after = path.read_text()

        # No duplicate/shadowed owned block: exactly one last_fetch key, and a
        # plain parse of the file sees the NEW values, not a stale tail.
        assert after.count("last_fetch:") == 1
        parsed = yaml_module.safe_load(after)
        assert parsed["last_fetch"]["reference"]["feature_count"] == 42
        assert parsed["last_fetch"]["target"]["feature_count"] == 1  # carried over
        reloaded = load_dataset_config(path)
        assert reloaded.last_fetch.reference.feature_count == 42
        assert reloaded.last_fetch.target.feature_count == 1
        # Human bytes outside the owned block are untouched
        assert after.startswith("name: midcomment_ds  # keep this inline comment\n")
        assert after.endswith("quality_hold:\n  reason: keep me\n")

    def test_update_is_idempotent(self, monkeypatch, tmp_path):
        """Applying the identical update twice leaves the file byte-identical."""
        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "idem_ds.yaml"
        path.write_text(COMMENTED_CONFIG.replace("commented_ds", "idem_ds"))

        assert self._run_update("idem_ds") is not None
        first = path.read_text()
        assert self._run_update("idem_ds") is not None
        second = path.read_text()
        assert second == first

    def test_safety_net_falls_back_to_full_save(self, monkeypatch, tmp_path, caplog):
        """If the splice does not verify (bad text), fall back to full
        re-serialization: comments are lost but values are correct."""
        import logging

        from crosswalk.datasets import schema as schema_module
        from crosswalk.datasets.schema import load_dataset_config

        datasets_dir = self._datasets_dir(monkeypatch, tmp_path)
        path = datasets_dir / "broken_splice_ds.yaml"
        path.write_text(COMMENTED_CONFIG.replace("commented_ds", "broken_splice_ds"))

        # Force a pathological splice result (unparsable YAML)
        monkeypatch.setattr(
            schema_module,
            "_replace_top_level_block",
            lambda text, key, block_yaml: ":::[ not yaml",
        )
        with caplog.at_level(logging.WARNING, logger="crosswalk.datasets.schema"):
            assert self._run_update("broken_splice_ds") is not None
        assert any("did not verify" in rec.getMessage() for rec in caplog.records)

        # Fallback wrote a full, correct re-serialization
        reloaded = load_dataset_config(path)
        assert reloaded.last_fetch.reference.feature_count == 42
        assert reloaded.last_fetch.target.feature_count == 10
        assert reloaded.quality_hold.since == "2026-07-06"


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

    def test_bogota_bike_cictsuperf_codes_resolve(self):
        """Bogota bike CICTSUPERF surface codes must map to cycleway at fetch time.

        Regression test for the class-vocab bug where raw integer surface codes
        (CICTSUPERF is 'Tipo de superficie', not a road class) leaked into the
        semantic `class` field, making class_similarity 100% NaN. Exercises the
        actual fetch-time path (map_column) with the int-typed column values the
        ArcGIS layer returns, so YAML int-key coercion is covered too.
        """
        from crosswalk.fetch.normalize import map_column

        mapping = self.ALL_YAMLS["co_bogota_bike_network"]["fetch"]["class_mapping"]
        # Full ArcGIS coded-value domain for CICTSUPERF
        domain_codes = [0, 1, 2, 3, 4, 5]
        assert set(mapping.keys()) == set(domain_codes), (
            "class_mapping must cover the full CICTSUPERF domain"
        )
        # Int-typed values (as fetched) and their float/string forms all resolve
        for series in (
            pd.Series(domain_codes),
            pd.Series([float(c) for c in domain_codes]),
            pd.Series([str(c) for c in domain_codes]),
        ):
            mapped = map_column(series, mapping, fallback="unknown")
            assert list(mapped) == ["cycleway"] * len(domain_codes)


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
