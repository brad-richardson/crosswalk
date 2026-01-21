"""Unified dataset configuration schema.

This module defines the Pydantic models for the consolidated dataset configuration
that combines:
- Display metadata (from datasets.csv)
- Fetch configuration (from dataset_configs.py)
- Fetch provenance (from .meta.yaml sidecar files)
- Classification mappings (from discover-classes)

Dataset configs are stored as YAML files in the `datasets/` directory at repo root.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    """Source data configuration - where to fetch the data from."""

    type: str = "arcgis"  # arcgis, osm, overture, download, manual
    url: str | None = None  # ArcGIS FeatureServer URL or download URL
    portal_url: str | None = None  # Human-readable portal/documentation URL
    file_format: str | None = None  # For downloads: shp, gpkg, geojson
    where_clause: str | None = None  # SQL WHERE filter for ArcGIS
    api_key_env_var: str | None = None  # Environment variable for API key
    api_key_header: str | None = None  # HTTP header name for API key


class FetchConfig(BaseModel):
    """Configuration for how to process fetched data."""

    id_prefix: str | None = None  # Prefix for generated IDs
    name_column: str | None = None  # Column containing road names
    class_column: str | None = None  # Column for classification
    class_mapping: dict[str | int, str] | None = None  # Source value -> Overture class
    subclass_column: str | None = None  # Optional subclass column
    subclass_mapping: dict[str | int, str] | None = None  # Subclass value -> subclass
    level_column: str | None = None  # Column for z-level (bridges/tunnels)
    bbox: tuple[float, float, float, float] | None = None  # xmin, ymin, xmax, ymax
    crs: str = "EPSG:4326"  # Coordinate reference system


class LastFetch(BaseModel):
    """Provenance information about the last data fetch."""

    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    bbox: tuple[float, float, float, float] | None = None  # Original requested bbox
    bbox_buffered: tuple[float, float, float, float] | None = None  # Expanded bbox
    bbox_buffer_m: float | None = None  # Buffer distance in meters
    feature_count: int = 0
    geometry_types: list[str] = Field(default_factory=list)
    output_path: str | None = None  # Path to the fetched parquet file
    notes: str | None = None


class SourceClassification(BaseModel):
    """Information about the source classification system."""

    column: str
    description: str | None = None
    values: dict[Any, dict] = Field(default_factory=dict)
    documentation_url: str | None = None


class ClassMappingRule(BaseModel):
    """A single class mapping rule for complex mappings."""

    source_value: str | int | list
    target_class: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class ClassificationConfig(BaseModel):
    """Classification discovery and mapping configuration."""

    source_classification: SourceClassification | None = None
    class_mapping_rules: list[ClassMappingRule] = Field(default_factory=list)
    default_class: str = "unclassified"
    confidence: str = "medium"  # low, medium, high


class DatasetConfig(BaseModel):
    """Complete unified dataset configuration.

    Combines display info, fetch config, provenance, and classification
    into a single YAML file per dataset.
    """

    # Identity & display
    name: str  # Dataset identifier (e.g., "boston_streets")
    display_name: str | None = None  # Human-readable name
    type: str = "road"  # road, bike, sidewalk, trail, transit
    description: str | None = None

    # Source & fetch configuration
    source: SourceConfig | None = None
    fetch: FetchConfig | None = None

    # Fetch provenance (auto-updated by fetch commands)
    last_fetch: LastFetch | None = None

    # Classification (from discover-classes)
    classification: ClassificationConfig | None = None

    # Additional metadata
    notes: str | None = None


def _convert_tuples_to_lists(data: dict) -> dict:
    """Recursively convert tuples to lists for YAML serialization."""
    result = {}
    for key, value in data.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = _convert_tuples_to_lists(value)
        elif isinstance(value, list):
            result[key] = [
                _convert_tuples_to_lists(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


def _convert_lists_to_tuples(data: dict, tuple_keys: set[str]) -> dict:
    """Recursively convert specific list fields back to tuples."""
    result = {}
    for key, value in data.items():
        if key in tuple_keys and isinstance(value, list) and len(value) == 4:
            result[key] = tuple(value)
        elif isinstance(value, dict):
            result[key] = _convert_lists_to_tuples(value, tuple_keys)
        elif isinstance(value, list):
            result[key] = [
                _convert_lists_to_tuples(item, tuple_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def save_dataset_config(config: DatasetConfig, path: Path) -> Path:
    """Save a dataset configuration to YAML.

    Args:
        config: Configuration to save
        path: Output path

    Returns:
        Path to saved file
    """
    # Convert to dict, handling datetime and tuple serialization
    data = config.model_dump(exclude_none=True, exclude_unset=True)

    # Handle datetime serialization
    if "last_fetch" in data and "fetched_at" in data["last_fetch"]:
        data["last_fetch"]["fetched_at"] = data["last_fetch"]["fetched_at"].isoformat()

    # Convert tuples to lists for YAML
    data = _convert_tuples_to_lists(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load a dataset configuration from YAML.

    Args:
        path: Path to YAML config file

    Returns:
        DatasetConfig

    Raises:
        ValueError: If the YAML file has invalid schema
        FileNotFoundError: If the file doesn't exist
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        raise ValueError(f"Empty or invalid YAML file: {path}")

    # Parse datetime
    if "last_fetch" in data and isinstance(data["last_fetch"].get("fetched_at"), str):
        data["last_fetch"]["fetched_at"] = datetime.fromisoformat(data["last_fetch"]["fetched_at"])

    # Convert bbox lists to tuples
    tuple_keys = {"bbox", "bbox_buffered"}
    data = _convert_lists_to_tuples(data, tuple_keys)

    return DatasetConfig(**data)


def get_datasets_dir() -> Path:
    """Get the datasets config directory (datasets/ at repo root)."""
    # Walk up from this file to find repo root (identified by pyproject.toml)
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            datasets_dir = current / "datasets"
            datasets_dir.mkdir(exist_ok=True)
            return datasets_dir
        current = current.parent

    # Fallback to current working directory
    datasets_dir = Path.cwd() / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir


def list_dataset_configs() -> list[str]:
    """List available dataset configurations.

    Returns:
        List of dataset names (without .yaml extension)
    """
    datasets_dir = get_datasets_dir()
    return [p.stem for p in datasets_dir.glob("*.yaml")]


def get_dataset_config(name: str) -> DatasetConfig | None:
    """Load a dataset configuration by name.

    Args:
        name: Dataset name (without .yaml extension)

    Returns:
        DatasetConfig if found, None otherwise
    """
    config_path = get_datasets_dir() / f"{name}.yaml"
    if not config_path.exists():
        return None
    return load_dataset_config(config_path)


def update_last_fetch(
    name: str,
    fetched_at: datetime | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_buffered: tuple[float, float, float, float] | None = None,
    bbox_buffer_m: float | None = None,
    feature_count: int = 0,
    geometry_types: list[str] | None = None,
    output_path: str | None = None,
    notes: str | None = None,
) -> DatasetConfig | None:
    """Update the last_fetch section of a dataset config.

    Args:
        name: Dataset name
        **kwargs: LastFetch fields to update

    Returns:
        Updated DatasetConfig if found, None if dataset doesn't exist
    """
    config = get_dataset_config(name)
    if config is None:
        return None

    # Update last_fetch
    config.last_fetch = LastFetch(
        fetched_at=fetched_at or datetime.now(UTC),
        bbox=bbox,
        bbox_buffered=bbox_buffered,
        bbox_buffer_m=bbox_buffer_m,
        feature_count=feature_count,
        geometry_types=geometry_types or [],
        output_path=output_path,
        notes=notes,
    )

    # Save back
    config_path = get_datasets_dir() / f"{name}.yaml"
    save_dataset_config(config, config_path)

    return config
