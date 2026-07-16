"""Unified dataset configuration schema.

This module defines the Pydantic models for the consolidated dataset configuration
that combines:
- Display metadata (from datasets.csv)
- Fetch configuration (from dataset_configs.py)
- Fetch provenance (from .meta.yaml sidecar files)
- Classification mappings (from discover-classes)

Dataset configs are stored as YAML files in the `datasets/` directory at repo root.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    file_pattern: str | None = None  # Glob pattern to select file within ZIP
    cache_download: bool = False  # Cache large downloads to ~/.cache/matcher/downloads
    cache_ttl_hours: int = 168  # Cache TTL in hours (default: 7 days)


class FetchConfig(BaseModel):
    """Configuration for how to process fetched data."""

    id_prefix: str | None = None  # Prefix for generated IDs
    id_column: str | None = None  # Column to use as stable ID (required for data integrity)
    name_column: str | None = None  # Column containing road names
    name_columns: dict[str, str] | None = None  # Language code -> source column mapping
    class_column: str | None = None  # Column for classification
    class_mapping: dict[str | int, str] | None = None  # Source value -> Overture class
    subclass_column: str | None = None  # Optional subclass column
    subclass_mapping: dict[str | int, str] | None = None  # Subclass value -> subclass
    level_column: str | None = None  # Column for z-level (bridges/tunnels)
    bridge_column: str | None = None  # Column indicating a physical bridge
    tunnel_column: str | None = None  # Column indicating a physical tunnel
    bridge_values: list[str | int | float] | None = None  # Explicit bridge-coded values
    tunnel_values: list[str | int | float] | None = None  # Explicit tunnel-coded values
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
    # Polygon-to-centerline conversion
    polygon_to_centerline: bool = False  # Convert polygon geometries to centerline LineStrings

    def physical_flag_domains(self) -> frozenset[str]:
        """Physical flag domains (``is_bridge``/``is_tunnel``) this source surveys.

        Derived from fetch provenance (the configured bridge/tunnel columns), NOT
        inferred from observed positive flags: a source that configures only a
        tunnel column has not surveyed bridges, so its bridge domain stays
        unknown rather than false. This is the single source of truth for the
        ``target_flag_domains`` provenance that
        ``features.physical.compute_physical_pair_features`` requires (see that
        function's docstring and research/physical_feature_experiment_2026-07-15.md).
        Level provenance is separate (see ``level_column``); this returns only the
        flag domains, so a level-only source yields an empty set.
        """
        domains: set[str] = set()
        if self.bridge_column:
            domains.add("is_bridge")
        if self.tunnel_column:
            domains.add("is_tunnel")
        return frozenset(domains)


class MatchingConfig(BaseModel):
    """Configuration for matching behavior."""

    block_cross_tier: bool = False  # Hard block vehicle↔pedestrian candidate pairs


class LastFetchInfo(BaseModel):
    """Provenance for a single fetch (target, reference, or OSM)."""

    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    bbox: tuple[float, float, float, float] | None = None  # Original requested bbox
    bbox_buffered: tuple[float, float, float, float] | None = None  # Expanded bbox
    bbox_buffer_m: float | None = None  # Buffer distance in meters
    feature_count: int = 0
    geometry_types: list[str] = Field(default_factory=list)
    output_path: str | None = None  # Path to the fetched parquet file
    notes: str | None = None


class LastFetch(BaseModel):
    """Provenance information about data fetches, tracked per source type."""

    target: LastFetchInfo | None = None
    reference: LastFetchInfo | None = None
    osm: LastFetchInfo | None = None


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

    Captures key metrics about dataset quality, computed by `crosswalk quality fingerprint`.
    See QualityFingerprint in crosswalk.quality for full metric details.
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


class QualityHoldConfig(BaseModel):
    """Persisted quality hold — blocks the dataset from being published.

    Declares that the dataset's factory output is known-defective (e.g. a
    systematic matching error) and must not ship even when its license is
    approved. The publisher (``crosswalk.factory.publish``) excludes any
    dataset carrying this block. Remove the block once the defect is fixed
    and the output re-verified.
    """

    reason: str  # Human-readable defect description (shown on the credibility page)
    since: str | None = None  # ISO date the hold was placed


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

    # Quality fingerprint (from crosswalk quality fingerprint)
    quality_fingerprint: QualityFingerprintConfig | None = None

    # Quality hold (blocks publishing until removed; see factory/publish.py)
    quality_hold: QualityHoldConfig | None = None

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


def _config_to_yaml_data(config: DatasetConfig) -> dict:
    """Convert a DatasetConfig to a plain dict ready for yaml.dump."""
    data = config.model_dump(exclude_none=True, exclude_unset=True)

    # Handle datetime serialization in last_fetch sub-fields
    if "last_fetch" in data:
        for fetch_type in ("target", "reference", "osm"):
            sub = data["last_fetch"].get(fetch_type)
            if isinstance(sub, dict) and "fetched_at" in sub:
                sub["fetched_at"] = sub["fetched_at"].isoformat()

    if "quality_fingerprint" in data and "computed_at" in data["quality_fingerprint"]:
        data["quality_fingerprint"]["computed_at"] = data["quality_fingerprint"][
            "computed_at"
        ].isoformat()

    # Convert tuples to lists for YAML
    return _convert_tuples_to_lists(data)


def save_dataset_config(config: DatasetConfig, path: Path) -> Path:
    """Save a dataset configuration to YAML.

    Full re-serialization: this strips any comments present in an existing
    file, so it is only appropriate for genuinely new configs (discovery,
    classification export). Machine updates to existing, human-curated
    configs must go through :func:`_update_owned_blocks` instead.

    Args:
        config: Configuration to save
        path: Output path

    Returns:
        Path to saved file
    """
    data = _config_to_yaml_data(config)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path


def _replace_top_level_block(text: str, key: str, block_yaml: str) -> str:
    """Textually replace one top-level YAML block, leaving all other bytes intact.

    The block starts at the column-0 line ``key:`` and extends through all
    following indented (or blank) lines. A column-0 ``#`` comment is part of
    the block when indented content still follows it (otherwise splicing would
    leave an orphaned indented tail that shadows the new values); it is a
    boundary — and preserved — only when the next non-blank, non-comment line
    is another column-0 key, or at EOF. Blank lines immediately before the
    next section are treated as human formatting and preserved. If the key is
    absent, the block is appended at the end of the file.

    Args:
        text: Full YAML file text
        key: Top-level key to replace (must be a plain, unquoted key)
        block_yaml: Replacement YAML text for the block (must end with a newline)

    Returns:
        Updated file text
    """
    lines = text.splitlines(keepends=True)
    prefix = f"{key}:"
    start = None
    for i, line in enumerate(lines):
        if line.startswith(prefix) and (
            len(line.rstrip("\r\n")) == len(prefix) or line[len(prefix)] in " \t"
        ):
            start = i
            break

    if start is None:
        # Block absent — append at end (normalizing a missing trailing newline)
        if text and not text.endswith("\n"):
            text += "\n"
        return text + block_yaml

    # Scan past the block: indented continuation lines and blank lines
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() == "" or line[0] in (" ", "\t"):
            end += 1
            continue
        if line[0] == "#":
            # Column-0 comment: block-interior if indented content still
            # follows (past blanks and further column-0 comments) — leaving it
            # in place would orphan that indented tail as a stale duplicate of
            # the owned block. Boundary only before another column-0 key/EOF.
            lookahead = end + 1
            interior = False
            while lookahead < len(lines):
                nxt = lines[lookahead]
                if nxt.strip() == "" or nxt[0] == "#":
                    lookahead += 1
                    continue
                interior = nxt[0] in (" ", "\t")
                break
            if interior:
                end = lookahead
                continue
            break
        break
    # Blank lines between the owned block and the next section belong to the
    # human formatting of what follows — back them out of the replaced region.
    while end > start + 1 and lines[end - 1].strip() == "":
        end -= 1

    return "".join(lines[:start]) + block_yaml + "".join(lines[end:])


def _update_owned_blocks(config: DatasetConfig, config_path: Path, keys: tuple[str, ...]) -> None:
    """Surgically rewrite only machine-owned top-level blocks of a config file.

    ``yaml.dump`` of a full ``model_dump()`` strips human comments and rewraps
    long scalars (issue #339). Fetch-style updates therefore re-serialize only
    the machine-owned block(s) — e.g. ``last_fetch``, ``quality_fingerprint`` —
    and splice them into the existing file text, leaving every human-authored
    byte untouched.

    Safety net: before writing, the spliced text is re-parsed and the owned
    keys' values are compared against the intended subtrees. If the splice
    produced anything unexpected (unparsable YAML, stale/shadowed values),
    the update falls back to full re-serialization — comments are lost in
    that case, but correctness beats comments.

    Args:
        config: Updated configuration (source of the new block values)
        config_path: Existing YAML file to update
        keys: Top-level keys owned by the machine to replace/append
    """
    if not config_path.exists():
        save_dataset_config(config, config_path)
        return

    data = _config_to_yaml_data(config)
    text = config_path.read_text(encoding="utf-8")
    for key in keys:
        if key not in data:
            continue
        block_yaml = yaml.dump(
            {key: data[key]},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        text = _replace_top_level_block(text, key, block_yaml)

    # Verify the surgical splice before writing it
    try:
        parsed = yaml.safe_load(text)
        splice_ok = isinstance(parsed, dict) and all(
            parsed.get(key) == data[key] for key in keys if key in data
        )
    except yaml.YAMLError:
        splice_ok = False

    if not splice_ok:
        logger.warning(
            "Surgical YAML update of %s for %s did not verify; falling back to "
            "full re-serialization (comments in the file will be lost)",
            keys,
            config_path,
        )
        save_dataset_config(config, config_path)
        return

    config_path.write_text(text, encoding="utf-8")


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

    # Migrate old flat last_fetch format to new nested format
    # Old format has fetched_at directly under last_fetch; new format has target/reference/osm
    if "last_fetch" in data and isinstance(data["last_fetch"], dict):
        if "fetched_at" in data["last_fetch"]:
            # Old flat format — migrate to target sub-field
            old_fetch = data["last_fetch"]
            data["last_fetch"] = {"target": old_fetch}

    # Parse datetime fields in last_fetch sub-fields
    if "last_fetch" in data and isinstance(data["last_fetch"], dict):
        for fetch_type in ("target", "reference", "osm"):
            sub = data["last_fetch"].get(fetch_type)
            if isinstance(sub, dict) and isinstance(sub.get("fetched_at"), str):
                sub["fetched_at"] = datetime.fromisoformat(sub["fetched_at"])

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
    fetch_type: str = "target",
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
        fetch_type: Which fetch to update: "target", "reference", or "osm"
        fetched_at: When the fetch occurred
        bbox: Original requested bounding box
        bbox_buffered: Expanded bounding box
        bbox_buffer_m: Buffer distance in meters
        feature_count: Number of features fetched
        geometry_types: List of geometry types in the data
        output_path: Path to the output file
        notes: Additional notes

    Returns:
        Updated DatasetConfig if found, None if dataset doesn't exist
    """
    if fetch_type not in ("target", "reference", "osm"):
        raise ValueError(f"fetch_type must be 'target', 'reference', or 'osm', got '{fetch_type}'")

    config = get_dataset_config(name)
    if config is None:
        return None

    # Build the fetch info
    fetch_info = LastFetchInfo(
        fetched_at=fetched_at or datetime.now(UTC),
        bbox=bbox,
        bbox_buffered=bbox_buffered,
        bbox_buffer_m=bbox_buffer_m,
        feature_count=feature_count,
        geometry_types=geometry_types or [],
        output_path=output_path,
        notes=notes,
    )

    # Initialize last_fetch if needed
    if config.last_fetch is None:
        config.last_fetch = LastFetch()

    # Update the correct sub-field
    setattr(config.last_fetch, fetch_type, fetch_info)

    # Save back surgically: only the machine-owned last_fetch block is
    # rewritten, so human comments/formatting elsewhere survive (issue #339)
    config_path = get_datasets_dir() / f"{name}.yaml"
    _update_owned_blocks(config, config_path, ("last_fetch",))

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

    # Save back surgically: only the machine-owned quality_fingerprint block
    # is rewritten, so human comments/formatting elsewhere survive (issue #339)
    config_path = get_datasets_dir() / f"{name}.yaml"
    _update_owned_blocks(config, config_path, ("quality_fingerprint",))

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
