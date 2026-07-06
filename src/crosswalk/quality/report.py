"""Quality report generation and output.

Generates JSON reports from quality fingerprints.
"""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from loguru import logger

from .fingerprint import QualityFingerprint
from .metrics import compute_quality_metrics


def generate_quality_report(
    data_path: Path,
    dataset_name: str | None = None,
    name_column: str | None = None,
    class_column: str | None = None,
) -> QualityFingerprint:
    """Generate a quality report for a dataset file.

    Args:
        data_path: Path to GeoParquet file with road edges
        dataset_name: Name for the dataset (defaults to filename)
        name_column: Column containing road names
        class_column: Column containing road class

    Returns:
        QualityFingerprint with computed metrics
    """
    logger.info(f"Generating quality report for {data_path}")

    # Load data
    gdf = gpd.read_parquet(data_path)

    # Use filename as dataset name if not provided
    if dataset_name is None:
        dataset_name = data_path.stem

    # Compute metrics
    fingerprint = compute_quality_metrics(
        edges_gdf=gdf,
        dataset_name=dataset_name,
        name_column=name_column,
        class_column=class_column,
    )

    return fingerprint


def save_quality_report(
    fingerprint: QualityFingerprint,
    output_path: Path,
    indent: int = 2,
) -> Path:
    """Save a quality fingerprint to a JSON file.

    Args:
        fingerprint: QualityFingerprint to save
        output_path: Path for output JSON file
        indent: JSON indentation level

    Returns:
        Path to the saved file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = fingerprint.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved quality report to {output_path}")
    return output_path


def load_quality_report(path: Path) -> QualityFingerprint:
    """Load a quality fingerprint from a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        QualityFingerprint loaded from file
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return QualityFingerprint.from_dict(data)


def compare_fingerprints(
    fp1: QualityFingerprint,
    fp2: QualityFingerprint,
) -> dict[str, Any]:
    """Compare two quality fingerprints.

    Args:
        fp1: First fingerprint (typically "before")
        fp2: Second fingerprint (typically "after")

    Returns:
        Dictionary with comparison metrics
    """
    return {
        "datasets": [fp1.dataset_name, fp2.dataset_name],
        "segment_count_delta": fp2.total_segments - fp1.total_segments,
        "length_delta_m": fp2.total_length_m - fp1.total_length_m,
        "name_coverage_delta": fp2.name_coverage_ratio - fp1.name_coverage_ratio,
        "island_count_delta": fp2.island_count - fp1.island_count,
        "dead_end_ratio_delta": fp2.dead_end_ratio - fp1.dead_end_ratio,
        "component_ratio_delta": fp2.largest_component_ratio - fp1.largest_component_ratio,
        "invalid_geometry_delta": fp2.invalid_geometry_count - fp1.invalid_geometry_count,
    }
