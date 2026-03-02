"""Utilities for leave-one-out by type cross-validation.

Classifies labeled datasets into type groups (road_good, road_poor, sidewalk, other)
for LOO-by-type CV that tests cross-dataset generalization.
"""

from __future__ import annotations

from .datasets.schema import get_dataset_config

# Manual type reclassifications for datasets whose YAML `type` field
# doesn't reflect their actual category for CV purposes.
TYPE_OVERRIDES: dict[str, str] = {
    "us_boston_bike_network": "bike",
}

# Minimum number of valid labels for a dataset to be included in LOO CV.
MIN_LOO_LABELS = 10

# Default threshold for splitting road datasets into good/poor quality.
DEFAULT_QUALITY_THRESHOLD = 0.5


def classify_dataset_type_group(
    dataset_name: str,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> str:
    """Classify a dataset into a type group for LOO CV.

    Groups:
        road_good: type=road AND min(name_cov, class_cov) >= threshold
        road_poor: type=road AND min(name_cov, class_cov) < threshold
        sidewalk:  type=sidewalk
        other:     type=bike/trail or anything else (merged)

    Args:
        dataset_name: Dataset name (matches YAML filename without extension)
        quality_threshold: Threshold for splitting road into good/poor

    Returns:
        One of: "road_good", "road_poor", "sidewalk", "other"
    """
    # Check for manual override first
    effective_type = TYPE_OVERRIDES.get(dataset_name)

    if effective_type is None:
        config = get_dataset_config(dataset_name)
        if config is None:
            return "other"
        effective_type = config.type
    else:
        config = get_dataset_config(dataset_name)

    # Non-road types
    if effective_type == "sidewalk":
        return "sidewalk"
    if effective_type in ("bike", "trail"):
        return "other"

    # For road type, split by quality fingerprint
    if effective_type == "road":
        if config is None or config.quality_fingerprint is None:
            return "road_poor"

        qf = config.quality_fingerprint
        min_cov = min(qf.name_coverage_ratio, qf.class_coverage_ratio)
        if min_cov >= quality_threshold:
            return "road_good"
        else:
            return "road_poor"

    # Unknown type -> other
    return "other"


def build_type_groups(
    dataset_names: list[str],
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> dict[str, list[str]]:
    """Group datasets by type for LOO CV.

    Args:
        dataset_names: List of dataset names to classify
        quality_threshold: Threshold for road good/poor split

    Returns:
        Dict mapping group name to list of dataset names in that group.
        Only groups with at least one dataset are included.
    """
    groups: dict[str, list[str]] = {}
    for name in dataset_names:
        group = classify_dataset_type_group(name, quality_threshold)
        groups.setdefault(group, []).append(name)
    return groups
