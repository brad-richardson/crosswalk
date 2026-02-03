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

    type: str = "arcgis"  # arcgis, osm, overture, download, manual, os_downloads
    url: str | None = None  # ArcGIS FeatureServer URL or download URL
    portal_url: str | None = None  # Human-readable portal/documentation URL
    file_format: str | None = None  # For downloads: shp, gpkg, geojson, gml
    where_clause: str | None = None  # SQL WHERE filter for ArcGIS
    api_key_env_var: str | None = None  # Environment variable for API key
    api_key_header: str | None = None  # HTTP header name for API key
    product_id: str | None = None  # OS Data Hub product ID (e.g., "OpenRoads")
    cache_download: bool = False  # Cache large downloads to ~/.cache/matcher/downloads
    cache_ttl_hours: int = 168  # Cache TTL in hours (default: 7 days)


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
    source_crs: str | None = None  # Source data CRS if different (e.g., "EPSG:5179")
    encoding: str | None = None  # File encoding if non-UTF8 (e.g., "EUC-KR")
    # Non-road feature detection
    non_road_type_codes: list[str] | None = (
        None  # Type codes to filter as non-roads (e.g., ['PC', 'PQ'])
    )
    filter_closed_loops: bool = False  # Enable geometry-based closed loop filtering
    exclude: dict[str, list[str]] | None = None  # Column -> values to exclude
    # One-way direction
    oneway_column: str | None = None  # Column for one-way direction
    # Speed limit
    speed_limit_column: str | None = None  # Column for speed limit
    speed_limit_unit: str = "kph"  # Unit: "kph" or "mph" (normalized to kph internally)


class MatchingConfig(BaseModel):
    """Configuration for matching behavior."""

    block_cross_tier: bool = False  # Hard block vehicle↔pedestrian candidate pairs


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


class QualityFingerprintConfig(BaseModel):
    """Quality fingerprint stored in dataset YAML.

    Captures key metrics about dataset quality, computed by `matcher quality fingerprint`.
    See QualityFingerprint in matcher.quality for full metric details.
    """

    # When fingerprint was computed
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Basic statistics
    total_segments: int = 0
    total_length_km: float = 0.0

    # Length distribution (meters)
    length_min_m: float = 0.0
    length_max_m: float = 0.0
    length_median_m: float = 0.0
    length_p5_m: float = 0.0
    length_p95_m: float = 0.0

    # Geometry quality
    vertex_density_mean: float = 0.0
    invalid_geometry_count: int = 0
    sharp_angle_ratio: float = 0.0  # Ratio of segments with sharp turns (>150°)
    mean_sinuosity: float = 1.0  # 1.0 = straight, higher = curvy
    high_sinuosity_ratio: float = 0.0

    # GPS artifacts
    drift_affected_ratio: float = 0.0  # Ratio affected by zigzag/spike/loop

    # Duplicates
    near_duplicate_ratio: float = 0.0

    # Topology
    island_count: int = 0  # Disconnected components
    dead_end_ratio: float = 0.0
    connected_components: int = 0
    largest_component_ratio: float = 0.0  # Main network coverage

    # Attributes
    name_coverage_ratio: float = 0.0  # Ratio with names
    class_coverage_ratio: float = 0.0  # Ratio with road class
    class_distribution: dict[str, int] = Field(default_factory=dict)


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

    # Matching configuration
    matching: MatchingConfig | None = None

    # Fetch provenance (auto-updated by fetch commands)
    last_fetch: LastFetch | None = None

    # Classification (from discover-classes)
    classification: ClassificationConfig | None = None

    # Quality fingerprint (from matcher quality fingerprint)
    quality_fingerprint: QualityFingerprintConfig | None = None

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

    if "quality_fingerprint" in data and "computed_at" in data["quality_fingerprint"]:
        data["quality_fingerprint"]["computed_at"] = data["quality_fingerprint"][
            "computed_at"
        ].isoformat()

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

    # Parse datetime fields
    if "last_fetch" in data and isinstance(data["last_fetch"].get("fetched_at"), str):
        data["last_fetch"]["fetched_at"] = datetime.fromisoformat(data["last_fetch"]["fetched_at"])

    if "quality_fingerprint" in data and isinstance(
        data["quality_fingerprint"].get("computed_at"), str
    ):
        data["quality_fingerprint"]["computed_at"] = datetime.fromisoformat(
            data["quality_fingerprint"]["computed_at"]
        )

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


def update_quality_fingerprint(
    name: str,
    fingerprint: "QualityFingerprintConfig",
) -> DatasetConfig | None:
    """Update the quality_fingerprint section of a dataset config.

    Args:
        name: Dataset name
        fingerprint: QualityFingerprintConfig to save

    Returns:
        Updated DatasetConfig if found, None if dataset doesn't exist
    """
    config = get_dataset_config(name)
    if config is None:
        return None

    # Update quality_fingerprint
    config.quality_fingerprint = fingerprint

    # Save back
    config_path = get_datasets_dir() / f"{name}.yaml"
    save_dataset_config(config, config_path)

    return config


def fingerprint_from_quality(
    quality_fp: "QualityFingerprint",  # noqa: F821
) -> QualityFingerprintConfig:
    """Convert a QualityFingerprint to QualityFingerprintConfig for YAML storage.

    Args:
        quality_fp: Full QualityFingerprint from quality module

    Returns:
        QualityFingerprintConfig suitable for YAML
    """
    return QualityFingerprintConfig(
        computed_at=quality_fp.timestamp,
        total_segments=quality_fp.total_segments,
        total_length_km=round(quality_fp.total_length_m / 1000, 2),
        length_min_m=quality_fp.length_min_m,
        length_max_m=quality_fp.length_max_m,
        length_median_m=quality_fp.length_median_m,
        length_p5_m=quality_fp.length_p5_m,
        length_p95_m=quality_fp.length_p95_m,
        vertex_density_mean=quality_fp.vertex_density_mean,
        invalid_geometry_count=quality_fp.invalid_geometry_count,
        sharp_angle_ratio=quality_fp.sharp_angle_ratio,
        mean_sinuosity=quality_fp.mean_segment_sinuosity,
        high_sinuosity_ratio=quality_fp.high_sinuosity_ratio,
        drift_affected_ratio=quality_fp.drift_affected_ratio,
        near_duplicate_ratio=quality_fp.near_duplicate_ratio,
        island_count=quality_fp.island_count,
        dead_end_ratio=quality_fp.dead_end_ratio,
        connected_components=quality_fp.connected_components,
        largest_component_ratio=quality_fp.largest_component_ratio,
        name_coverage_ratio=quality_fp.name_coverage_ratio,
        class_coverage_ratio=quality_fp.class_coverage_ratio,
        class_distribution=quality_fp.class_distribution,
    )
