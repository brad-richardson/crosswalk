"""Dataset configuration schema and utilities.

Provides structured configuration for dataset-specific settings including
class mapping rules that transform source classification systems to Overture's
road hierarchy.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from loguru import logger

# Environment variable for custom config directories
DATASET_CONFIG_DIR_ENV = "MATCHER_DATASET_CONFIG_DIR"


@dataclass
class ClassMappingRule:
    """A single class mapping rule.

    Attributes:
        source_value: Value(s) in source classification to match
        target_class: Overture class to map to
        conditions: Optional dict of additional conditions (e.g., {"AADT": ">15000"})
        priority: Rule priority (higher = checked first)
    """

    source_value: str | int | list
    target_class: str
    conditions: dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def matches(self, row: dict, source_column: str) -> bool:
        """Check if this rule matches a data row.

        The rule matches when:
        - ``row[source_column]`` matches ``source_value`` (or is in the list when
          ``source_value`` is a list), and
        - every condition in ``self.conditions`` is satisfied.

        String-based condition values support comparison operators:

        * ``">N"`` / ``">=N"`` / ``"<N"`` / ``"<=N"``: ``N`` is parsed as float and
          compared numerically. Example: ``{"AADT": ">15000"}`` matches rows where
          ``row["AADT"] > 15000``.
        * ``"==X"``: String equality comparison. Example: ``{"ROAD_TYPE": "==HIGHWAY"}``
          matches when ``row["ROAD_TYPE"] == "HIGHWAY"``.
        * Any other string is compared by converting both sides to str.
        * Non-string condition values are compared directly with ``==``.

        Args:
            row: Dict-like row from DataFrame.
            source_column: Name of the source classification column.

        Returns:
            True if the rule matches the given row.
        """
        # Check source value
        source_val = row.get(source_column)
        if isinstance(self.source_value, list):
            if source_val not in self.source_value:
                return False
        elif source_val != self.source_value:
            return False

        # Check additional conditions
        for col, condition in self.conditions.items():
            val = row.get(col)
            if val is None:
                return False

            if isinstance(condition, str):
                # Parse comparison operators
                if condition.startswith(">="):
                    try:
                        threshold = float(condition[2:])
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid numeric condition for '{col}': {condition}")
                        return False
                    if not (val >= threshold):
                        return False
                elif condition.startswith("<="):
                    try:
                        threshold = float(condition[2:])
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid numeric condition for '{col}': {condition}")
                        return False
                    if not (val <= threshold):
                        return False
                elif condition.startswith(">"):
                    try:
                        threshold = float(condition[1:])
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid numeric condition for '{col}': {condition}")
                        return False
                    if not (val > threshold):
                        return False
                elif condition.startswith("<"):
                    try:
                        threshold = float(condition[1:])
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid numeric condition for '{col}': {condition}")
                        return False
                    if not (val < threshold):
                        return False
                elif condition.startswith("=="):
                    # String equality comparison
                    cond_val = condition[2:]
                    if isinstance(val, (int, float)):
                        try:
                            if val != float(cond_val):
                                return False
                        except ValueError:
                            if str(val) != cond_val:
                                return False
                    else:
                        if str(val) != cond_val:
                            return False
                else:
                    # Direct comparison
                    if str(val) != str(condition):
                        return False
            else:
                if val != condition:
                    return False

        return True


@dataclass
class PhysicalAttributes:
    """Column mappings for physical road attributes."""

    lanes_column: str | None = None
    speed_column: str | None = None
    width_column: str | None = None
    traffic_column: str | None = None


@dataclass
class SourceClassification:
    """Information about the source classification system."""

    column: str
    description: str | None = None
    values: dict[Any, dict] = field(default_factory=dict)
    documentation_url: str | None = None


@dataclass
class DatasetConfig:
    """Complete configuration for a dataset.

    Attributes:
        name: Dataset identifier (e.g., "boston_streets")
        description: Human-readable description
        source_classification: Info about the source class system
        physical_attributes: Column mappings for road attributes
        class_mapping_rules: List of rules for class transformation
        confidence: Confidence level in the mapping (low/medium/high)
        notes: Additional notes about the mapping
    """

    name: str
    description: str | None = None
    source_classification: SourceClassification | None = None
    physical_attributes: PhysicalAttributes | None = None
    class_mapping_rules: list[ClassMappingRule] = field(default_factory=list)
    default_class: str = "unclassified"
    confidence: str = "medium"
    notes: str | None = None

    def get_target_class(self, row: dict) -> str:
        """Apply mapping rules to get target Overture class.

        Args:
            row: Dict-like row from DataFrame

        Returns:
            Mapped Overture class
        """
        if not self.source_classification or not self.class_mapping_rules:
            return self.default_class

        source_col = self.source_classification.column

        # Sort rules by priority (descending)
        sorted_rules = sorted(self.class_mapping_rules, key=lambda r: -r.priority)

        for rule in sorted_rules:
            if rule.matches(row, source_col):
                return rule.target_class

        return self.default_class


def _get_configs_dir() -> Path:
    """Get the datasets config directory.

    Returns the directory from MATCHER_DATASET_CONFIG_DIR environment variable
    if set, otherwise returns the package's datasets directory.
    """
    custom_dir = os.environ.get(DATASET_CONFIG_DIR_ENV)
    if custom_dir:
        return Path(custom_dir)
    return Path(__file__).parent


def load_dataset_config(name: str) -> DatasetConfig | None:
    """Load a dataset configuration by name.

    Args:
        name: Dataset name (without .yaml extension)

    Returns:
        DatasetConfig if found, None otherwise
    """
    config_path = _get_configs_dir() / f"{name}.yaml"
    if not config_path.exists():
        logger.debug(f"No config found for dataset: {name}")
        return None

    return load_dataset_config_from_file(config_path)


def load_dataset_config_from_file(path: Path) -> DatasetConfig:
    """Load a dataset configuration from a YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        DatasetConfig

    Raises:
        ValueError: If the YAML file has invalid schema
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Validate schema
    if data is None:
        raise ValueError(f"Empty or invalid YAML file: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict, got {type(data).__name__}: {path}")

    # Validate required fields
    if "name" not in data and path.stem == "":
        raise ValueError(f"Config must have 'name' field or valid filename: {path}")

    # Validate mapping rules structure if present
    for i, rule in enumerate(data.get("class_mapping_rules", [])):
        if not isinstance(rule, dict):
            raise ValueError(f"Rule {i} must be a dict: {path}")
        if "source_value" not in rule:
            raise ValueError(f"Rule {i} missing 'source_value': {path}")
        if "target_class" not in rule:
            raise ValueError(f"Rule {i} missing 'target_class': {path}")

    # Parse source classification
    source_class = None
    if "source_classification" in data:
        sc = data["source_classification"]
        source_class = SourceClassification(
            column=sc.get("column", ""),
            description=sc.get("description"),
            values=sc.get("values", {}),
            documentation_url=sc.get("documentation_url"),
        )

    # Parse physical attributes
    phys_attrs = None
    if "physical_attributes" in data:
        pa = data["physical_attributes"]
        phys_attrs = PhysicalAttributes(
            lanes_column=pa.get("lanes_column"),
            speed_column=pa.get("speed_column"),
            width_column=pa.get("width_column"),
            traffic_column=pa.get("traffic_column"),
        )

    # Parse mapping rules
    rules = []
    for rule_data in data.get("class_mapping_rules", []):
        rules.append(
            ClassMappingRule(
                source_value=rule_data.get("source_value"),
                target_class=rule_data.get("target_class", "unclassified"),
                conditions=rule_data.get("conditions", {}),
                priority=rule_data.get("priority", 0),
            )
        )

    return DatasetConfig(
        name=data.get("name", path.stem),
        description=data.get("description"),
        source_classification=source_class,
        physical_attributes=phys_attrs,
        class_mapping_rules=rules,
        default_class=data.get("default_class", "unclassified"),
        confidence=data.get("confidence", "medium"),
        notes=data.get("notes"),
    )


def list_dataset_configs() -> list[str]:
    """List available dataset configurations.

    Returns:
        List of dataset names (without .yaml extension)
    """
    configs_dir = _get_configs_dir()
    return [p.stem for p in configs_dir.glob("*.yaml")]


def apply_class_mapping(
    gdf: gpd.GeoDataFrame,
    config: DatasetConfig,
    source_tags_column: str = "source_tags",
    class_column: str = "class",
) -> gpd.GeoDataFrame:
    """Apply class mapping rules to a GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame
        config: Dataset configuration with mapping rules
        source_tags_column: Column containing source attributes dict
        class_column: Column to write mapped class to

    Returns:
        GeoDataFrame with updated class column

    Note:
        This function uses DataFrame.apply() with axis=1 which may be slow
        for large datasets (>100k rows). For performance-critical applications
        with simple mapping rules, consider using vectorized operations instead.
    """
    if not config.class_mapping_rules:
        logger.warning(f"No mapping rules in config for {config.name}")
        return gdf

    gdf = gdf.copy()

    def map_row(row):
        # Build a dict with source tags expanded
        row_dict = row.to_dict()

        # Extract source tags if present
        source_tags = row_dict.get(source_tags_column)
        if source_tags and isinstance(source_tags, dict):
            row_dict.update(source_tags)

        return config.get_target_class(row_dict)

    gdf[class_column] = gdf.apply(map_row, axis=1)

    # Log mapping results
    class_counts = gdf[class_column].value_counts()
    logger.info(f"Applied class mapping for {config.name}:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count}")

    return gdf


def save_dataset_config(config: DatasetConfig, path: Path | None = None) -> Path:
    """Save a dataset configuration to YAML.

    Args:
        config: Configuration to save
        path: Output path (default: datasets/{name}.yaml)

    Returns:
        Path to saved file
    """
    if path is None:
        path = _get_configs_dir() / f"{config.name}.yaml"

    data = {
        "name": config.name,
        "description": config.description,
        "confidence": config.confidence,
        "default_class": config.default_class,
    }

    if config.notes:
        data["notes"] = config.notes

    if config.source_classification:
        sc = config.source_classification
        data["source_classification"] = {
            "column": sc.column,
            "description": sc.description,
            "values": sc.values,
        }
        if sc.documentation_url:
            data["source_classification"]["documentation_url"] = sc.documentation_url

    if config.physical_attributes:
        pa = config.physical_attributes
        data["physical_attributes"] = {}
        if pa.lanes_column:
            data["physical_attributes"]["lanes_column"] = pa.lanes_column
        if pa.speed_column:
            data["physical_attributes"]["speed_column"] = pa.speed_column
        if pa.width_column:
            data["physical_attributes"]["width_column"] = pa.width_column
        if pa.traffic_column:
            data["physical_attributes"]["traffic_column"] = pa.traffic_column

    if config.class_mapping_rules:
        data["class_mapping_rules"] = []
        for rule in config.class_mapping_rules:
            rule_data = {
                "source_value": rule.source_value,
                "target_class": rule.target_class,
            }
            if rule.conditions:
                rule_data["conditions"] = rule.conditions
            if rule.priority != 0:
                rule_data["priority"] = rule.priority
            data["class_mapping_rules"].append(rule_data)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Saved dataset config to {path}")
    return path
