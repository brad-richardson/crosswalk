"""Fetch metadata tracking for data provenance.

Stores metadata about fetched datasets in YAML sidecar files alongside
GeoParquet outputs. This enables tracking data freshness, reproducibility,
and provenance.

Metadata file naming: <parquet_file>.meta.yaml
Example: overture_segments.parquet.meta.yaml
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class FetchMetadata(BaseModel):
    """Metadata for a fetched dataset."""

    # When the data was fetched
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Bounding box used for fetch (xmin, ymin, xmax, ymax)
    bbox: tuple[float, float, float, float] | None = None

    # Buffered bounding box if expansion was applied
    bbox_buffered: tuple[float, float, float, float] | None = None
    bbox_buffer_m: float | None = None

    # Filter mode for validation fetches
    filter_mode: str | None = None  # "fully_inside" when validation mode is used

    # Source information
    source: str  # e.g., "overture", "osm", "arcgis"
    source_url: str | None = None

    # Source-specific details
    release: str | None = None  # For Overture: release version
    region: str | None = None  # For OSM: Geofabrik region name

    # Filters applied
    filters: dict[str, Any] = Field(default_factory=dict)

    # Output statistics
    feature_count: int = 0
    geometry_types: list[str] = Field(default_factory=list)

    # Additional notes
    notes: str | None = None

    # Version tracking (added for data versioning)
    transform_version: str | None = None  # TRANSFORM_VERSION at fetch time
    schema_version: str | None = None  # SCHEMA_VERSION at fetch time
    data_version: str | None = None  # Combined DATA_VERSION (e.g., "v1.0")

    # ID column tracking
    id_column: str | None = None  # Source column used (OBJECTID, fid, etc.)
    id_prefix: str | None = None  # Prefix applied to IDs


def save_metadata(output_path: Path, metadata: FetchMetadata) -> Path:
    """Save fetch metadata to a sidecar YAML file.

    Args:
        output_path: Path to the GeoParquet file
        metadata: Fetch metadata to save

    Returns:
        Path to the metadata file
    """
    meta_path = Path(str(output_path) + ".meta.yaml")

    # Convert to dict, handling datetime serialization
    data = metadata.model_dump()
    data["fetched_at"] = data["fetched_at"].isoformat()

    # Convert tuples to lists for safe YAML serialization
    # (yaml.dump writes tuples with !!python/tuple tag which safe_load can't parse)
    if data.get("bbox"):
        data["bbox"] = list(data["bbox"])
    if data.get("bbox_buffered"):
        data["bbox_buffered"] = list(data["bbox_buffered"])

    with open(meta_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return meta_path


def load_metadata(output_path: Path) -> FetchMetadata | None:
    """Load fetch metadata from a sidecar YAML file.

    Args:
        output_path: Path to the GeoParquet file

    Returns:
        FetchMetadata if the sidecar file exists, None otherwise
    """
    meta_path = Path(str(output_path) + ".meta.yaml")

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        data = yaml.safe_load(f)

    # Handle empty or invalid YAML files
    if not data or not isinstance(data, dict):
        return None

    # Parse datetime
    if isinstance(data.get("fetched_at"), str):
        data["fetched_at"] = datetime.fromisoformat(data["fetched_at"])

    # Handle tuple conversion for bbox fields
    if data.get("bbox") and isinstance(data["bbox"], list):
        data["bbox"] = tuple(data["bbox"])
    if data.get("bbox_buffered") and isinstance(data["bbox_buffered"], list):
        data["bbox_buffered"] = tuple(data["bbox_buffered"])

    return FetchMetadata(**data)


def get_metadata_path(output_path: Path) -> Path:
    """Get the metadata sidecar file path for a GeoParquet file.

    Args:
        output_path: Path to the GeoParquet file

    Returns:
        Path to the metadata sidecar file
    """
    return Path(str(output_path) + ".meta.yaml")


def update_metadata(output_path: Path, **updates: Any) -> FetchMetadata:
    """Update existing metadata or create new if not exists.

    Args:
        output_path: Path to the GeoParquet file
        **updates: Fields to update

    Returns:
        Updated FetchMetadata
    """
    metadata = load_metadata(output_path)

    if metadata is None:
        # Create new with required source field
        if "source" not in updates:
            updates["source"] = "unknown"
        metadata = FetchMetadata(**updates)
    else:
        # Update existing
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

    save_metadata(output_path, metadata)
    return metadata
