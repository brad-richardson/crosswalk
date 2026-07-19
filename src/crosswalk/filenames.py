"""Centralized filename generation for data files.

Single source of truth for all filename patterns used in the crosswalk pipeline.
This module ensures consistent naming across fetch, pipeline, and labeling code.
"""

import hashlib
from pathlib import Path

from crosswalk.config import DATA_VERSION, FEATURE_VERSION, settings

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Project root (src/crosswalk/filenames.py -> project root)
PROJECT_ROOT = Path(__file__).parents[2]

# Cache directory for labeling UI
LABELING_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "labeling"

# Cache directory for integration results
INTEGRATION_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "integration"

# Cache directory for stitching review batches
STITCH_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "stitch"

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


def unmatched_filename(dataset_name: str) -> str:
    """Unmatched segments filename (no version suffix - output files).

    Example: us_boston_streets -> us_boston_streets_unmatched.parquet
    """
    return f"{dataset_name}_unmatched.parquet"


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

    Handles special case for OSM target datasets (ending in '_osm'),
    which use the osm_segments naming convention instead of the
    standard target file naming.

    Args:
        data_dir: Directory containing data files
        dataset_name: Dataset name (e.g., "us_boston_streets" or "us_boston_streets_osm")

    Returns:
        Path to target file, or None if not found

    Examples:
        us_boston_streets -> us_boston_streets_v1.0.parquet
        us_boston_streets_osm -> us_boston_streets_osm_segments_v1.0.parquet
    """
    # Special case: OSM target datasets use osm_segments naming
    if dataset_name.endswith("_osm"):
        # Strip '_osm' suffix and use osm_segments naming
        base_name = dataset_name[:-4]  # Remove '_osm'
        path = data_dir / osm_segments_filename(base_name)
        return path if path.exists() else None

    # Standard target file naming
    path = data_dir / target_filename(dataset_name)
    return path if path.exists() else None


# ============================================================================
# CACHE PATHS (labeling UI)
# ============================================================================


def _model_fingerprint() -> str:
    """Compute a short fingerprint of the current ML model file.

    Uses file size + mtime to detect when the model has been retrained.
    Returns a stable 8-char hex string. If the model file doesn't exist,
    returns "nomodel" so the cache path is still valid.
    """
    # Labeling caches intentionally follow the locally trained advisory model.
    # Production stitch/factory scoring defaults to the bundled artifact via
    # settings.model_path and requires an explicit override.
    model_path = settings.local_model_path
    if not model_path.exists():
        return "nomodel"
    stat = model_path.stat()
    key = f"{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def scored_cache_path(dataset_id: str) -> Path:
    """Get path to versioned scored candidates cache file.

    The scored cache contains candidates with ML predictions (decision, confidence)
    and all computed features. It is versioned by both FEATURE_VERSION and a model
    fingerprint, so that retraining the model automatically invalidates stale
    predictions while the feature cache (which has no model dependency) stays valid.

    Args:
        dataset_id: Dataset identifier (e.g., "us_boston_streets")

    Returns:
        Path to cache file (may not exist)

    Example:
        us_boston_streets -> data/cache/labeling/us_boston_streets_candidates_v2026-02-01_m3a4b5c6d.parquet
    """
    fingerprint = _model_fingerprint()
    return LABELING_CACHE_DIR / f"{dataset_id}_candidates_v{FEATURE_VERSION}_m{fingerprint}.parquet"


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


# ============================================================================
# CACHE PATHS (integration QA)
# ============================================================================


def integration_cache_dir(dataset_name: str) -> Path:
    """Get integration cache directory for a dataset.

    Args:
        dataset_name: Dataset identifier (e.g., "us_boston_streets")

    Returns:
        Path to cache directory (may not exist)

    Example:
        us_boston_streets -> data/cache/integration/us_boston_streets/
    """
    return INTEGRATION_CACHE_DIR / dataset_name


# ============================================================================
# GROUPS SIDECAR (stitching review)
# ============================================================================


def groups_sidecar_path(bridge_path: Path) -> Path:
    """Get path to the groups sidecar JSON alongside a bridge file.

    Args:
        bridge_path: Path to the bridge parquet file

    Returns:
        Path to groups sidecar JSON

    Example:
        data/output/us_boston_streets_bridge.parquet
        -> data/output/us_boston_streets_groups.json
    """
    stem = bridge_path.stem  # us_boston_streets_bridge
    # Replace _bridge suffix with _groups
    if stem.endswith("_bridge"):
        stem = stem[: -len("_bridge")] + "_groups"
    else:
        stem = stem + "_groups"
    return bridge_path.parent / f"{stem}.json"


def candidates_sidecar_path(bridge_path: Path) -> Path:
    """Get the typed resolver-candidate parquet alongside a bridge file.

    Example:
        data/output/us_boston_streets_bridge.parquet
        -> data/output/us_boston_streets_candidates.parquet
    """
    stem = bridge_path.stem
    if stem.endswith("_bridge"):
        stem = stem[: -len("_bridge")] + "_candidates"
    else:
        stem = stem + "_candidates"
    return bridge_path.parent / f"{stem}.parquet"


# Sentinel "dataset" id for the combined cross-dataset stitching review queue
# (crosswalk data stitch-batch-all). It is not a real dataset — its batch file
# aggregates every per-dataset queue, and each group inside carries its own
# ``dataset_id`` so the review UI can route labels back to the owning partition.
STITCH_ALL_QUEUE = "__all__"

# Separate cross-dataset queue for upgrading prior stitch decisions with
# pair-identity dispositions.  Keeping it distinct prevents already-reviewed
# work from reappearing in the ordinary ``__all__`` queue.
STITCH_PAIRWISE_QUEUE = "__pairwise__"


def stitch_batch_path(dataset_id: str) -> Path:
    """Get path to stitching review batch file.

    Args:
        dataset_id: Dataset identifier (e.g., "us_boston_streets"), or
            ``STITCH_ALL_QUEUE`` for the combined cross-dataset queue.

    Returns:
        Path to batch JSON file

    Example:
        us_boston_streets -> data/cache/stitch/us_boston_streets_batch.json
        __all__           -> data/cache/stitch/__all___batch.json
    """
    return STITCH_CACHE_DIR / f"{dataset_id}_batch.json"
