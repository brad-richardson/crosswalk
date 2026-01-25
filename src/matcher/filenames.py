"""Centralized filename generation for data files.

Single source of truth for all filename patterns used in the matcher pipeline.
This module ensures consistent naming across fetch, pipeline, and labeling code.
"""

from pathlib import Path

from matcher.config import DATA_VERSION, FEATURE_VERSION

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Project root (src/matcher/filenames.py -> project root)
PROJECT_ROOT = Path(__file__).parents[2]

# Cache directory for labeling UI
LABELING_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "labeling"

# ============================================================================
# FILENAME PATTERNS (with version suffix)
# ============================================================================


def target_filename(dataset_name: str) -> str:
    """Target/local dataset filename.

    Example: us_boston_streets -> us_boston_streets_v1.0.parquet
    """
    return f"{dataset_name}_{DATA_VERSION}.parquet"


def overture_segments_filename(region: str) -> str:
    """Overture segments filename.

    Example: us_boston -> us_boston_overture_segments_v1.0.parquet
    """
    return f"{region}_overture_segments_{DATA_VERSION}.parquet"


def overture_connectors_filename(region: str) -> str:
    """Overture connectors filename.

    Example: us_boston -> us_boston_overture_connectors_v1.0.parquet
    """
    return f"{region}_overture_connectors_{DATA_VERSION}.parquet"


def osm_segments_filename(name: str) -> str:
    """OSM segments filename.

    Example: us_boston_streets -> us_boston_streets_osm_segments_v1.0.parquet
    """
    return f"{name}_osm_segments_{DATA_VERSION}.parquet"


def osm_connectors_filename(name: str) -> str:
    """OSM connectors filename.

    Example: us_boston_streets -> us_boston_streets_osm_connectors_v1.0.parquet
    """
    return f"{name}_osm_connectors_{DATA_VERSION}.parquet"


def bridge_filename(dataset_name: str) -> str:
    """Bridge file filename (no version suffix - output files)."""
    return f"{dataset_name}_bridge.parquet"


# ============================================================================
# VERSION EXTRACTION
# ============================================================================


def extract_version_from_filename(path: Path) -> str | None:
    """Extract version from filename like 'us_boston_streets_v1.0.parquet'.

    Args:
        path: Path to the data file

    Returns:
        Version string without 'v' prefix (e.g., '1.0'), or None if no version found
    """
    stem = path.stem  # 'us_boston_streets_v1.0'
    if "_v" in stem:
        # Get the part after the last '_v'
        version_part = stem.split("_v")[-1]
        # Validate it looks like a version (digits and dots)
        if version_part and all(c.isdigit() or c == "." for c in version_part):
            return version_part
    return None


# ============================================================================
# FILE DISCOVERY (versioned files only - no legacy support)
# ============================================================================


def find_overture_segments(data_dir: Path, dataset_name: str) -> Path | None:
    """Find Overture segments file for a dataset.

    Tries progressively shorter prefixes to find the matching Overture file
    with the current DATA_VERSION. Does NOT fall back to glob matching to
    ensure version consistency.

    Args:
        data_dir: Directory containing data files
        dataset_name: Dataset name (e.g., "us_boston_streets")

    Returns:
        Path to Overture segments file, or None if not found

    Examples:
        us_boston_streets -> us_boston_overture_segments_v1.0.parquet
        us_fort_collins_sidewalks -> us_fort_collins_overture_segments_v1.0.parquet
    """
    parts = dataset_name.split("_")

    # Try progressively shorter prefixes with exact version match only
    for i in range(len(parts), 0, -1):
        region = "_".join(parts[:i])
        path = data_dir / overture_segments_filename(region)
        if path.exists():
            return path

    return None


def find_osm_segments(data_dir: Path, dataset_name: str) -> Path | None:
    """Find OSM segments file for a dataset.

    Args:
        data_dir: Directory containing data files
        dataset_name: Dataset name (e.g., "us_boston_streets")

    Returns:
        Path to OSM segments file, or None if not found
    """
    path = data_dir / osm_segments_filename(dataset_name)
    return path if path.exists() else None


def find_target_file(data_dir: Path, dataset_name: str) -> Path | None:
    """Find target/local dataset file.

    Args:
        data_dir: Directory containing data files
        dataset_name: Dataset name (e.g., "us_boston_streets")

    Returns:
        Path to target file, or None if not found
    """
    path = data_dir / target_filename(dataset_name)
    return path if path.exists() else None


# ============================================================================
# CACHE PATHS (labeling UI)
# ============================================================================


def scored_cache_path(dataset_id: str) -> Path:
    """Get path to scored candidates cache file.

    The scored cache contains candidates with ML predictions (decision, confidence).

    Args:
        dataset_id: Dataset identifier (e.g., "us_boston_streets")

    Returns:
        Path to cache file (may not exist)

    Example:
        us_boston_streets -> data/cache/labeling/us_boston_streets_candidates.parquet
    """
    return LABELING_CACHE_DIR / f"{dataset_id}_candidates.parquet"


def feature_cache_path(dataset_id: str) -> Path:
    """Get path to versioned feature cache file.

    The feature cache contains computed features WITHOUT ML predictions,
    allowing fast re-scoring when the ML model changes.

    Args:
        dataset_id: Dataset identifier (e.g., "us_boston_streets")

    Returns:
        Path to cache file (may not exist)

    Example:
        us_boston_streets -> data/cache/labeling/us_boston_streets_features_v2026-01-24.parquet
    """
    return LABELING_CACHE_DIR / f"{dataset_id}_features_v{FEATURE_VERSION}.parquet"
