"""Context generation for agent labeling pipeline.

Generates YAML metadata files and candidate packages for AI agent labeling.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyproj
import yaml
from loguru import logger
from shapely.ops import transform

from .image_renderer import render_candidate_images
from .sampler import SampledCandidate


def _calculate_length_meters(geom: Any) -> float:
    """Calculate geometry length in meters.

    Projects WGS84 geometry to appropriate UTM zone for accurate length calculation.
    """
    if geom is None or geom.is_empty:
        return 0.0

    try:
        # Get centroid for UTM zone calculation
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y

        # Determine UTM zone
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"

        # Create transformer
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        # Project and calculate length
        projected = transform(transformer.transform, geom)
        return projected.length
    except Exception:
        # Fallback: rough approximation using degrees
        # 1 degree ≈ 111km at equator
        return geom.length * 111000


def generate_metadata_yaml(
    candidate: SampledCandidate,
    batch_id: str,
) -> str:
    """Generate YAML metadata for a candidate.

    Args:
        candidate: Sampled candidate with all metadata
        batch_id: Batch identifier

    Returns:
        YAML string
    """
    # Get geometry info
    ref_geom = candidate.ref_geometry
    target_geom = candidate.target_geometry

    # Calculate lengths in meters (project from WGS84 to UTM)
    ref_length = _calculate_length_meters(ref_geom)
    target_length = _calculate_length_meters(target_geom)

    # Calculate bbox
    if ref_geom and target_geom:
        combined = ref_geom.union(target_geom)
        bbox = list(combined.bounds)
    elif ref_geom:
        bbox = list(ref_geom.bounds)
    elif target_geom:
        bbox = list(target_geom.bounds)
    else:
        bbox = [0, 0, 0, 0]

    # Organize features by category
    features = candidate.features
    geometric_features = {
        "hausdorff_distance": _round_value(features.get("hausdorff_distance")),
        "mean_hausdorff_distance": _round_value(features.get("mean_hausdorff_distance")),
        "buffer_iou": _round_value(features.get("buffer_iou")),
        "overlap_ratio": _round_value(features.get("overlap_ratio")),
        "heading_delta": _round_value(features.get("heading_delta")),
        "length_ratio": _round_value(features.get("length_ratio")),
        "centroid_distance": _round_value(features.get("centroid_distance")),
    }

    semantic_features = {
        "name_levenshtein": _round_value(features.get("name_levenshtein")),
        "name_jaro_winkler": _round_value(features.get("name_jaro_winkler")),
        "name_token_sort": _round_value(features.get("name_token_sort")),
        "class_similarity": _round_value(features.get("class_similarity")),
    }

    topological_features = {
        "degree_match_score": _round_value(features.get("degree_match_score")),
        "dead_end_match": bool(features.get("dead_end_match", False)),
        "intersection_match": bool(features.get("intersection_match", False)),
    }

    # Build metadata structure
    metadata = {
        "candidate": {
            "ref_id": candidate.ref_id,
            "target_id": candidate.target_id,
            "dataset": candidate.dataset,
            "batch": batch_id,
        },
        "names": {
            "reference": candidate.ref_name,
            "target": candidate.target_name,
        },
        "classes": {
            "reference": candidate.ref_class,
            "target": candidate.target_class,
        },
        "ml_prediction": {
            "decision": candidate.ml_decision,
            "confidence": _round_value(candidate.ml_confidence),
        },
        "geometry": {
            "ref_length_m": _round_value(ref_length),
            "target_length_m": _round_value(target_length),
            "bbox": [_round_value(v) for v in bbox],
        },
        "features": {
            "geometric": geometric_features,
            "semantic": semantic_features,
            "topological": topological_features,
        },
        "images": {
            "satellite": "satellite.png",
            "geometry": "geometry.png",
        },
    }

    return yaml.dump(metadata, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _round_value(value: Any, decimals: int = 4) -> Any:
    """Round numeric values for cleaner YAML output."""
    if value is None:
        return None
    # Handle numpy types
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, int):
        return int(value)
    return value


def write_candidate_package(
    output_dir: Path,
    candidate: SampledCandidate,
    batch_id: str,
    fetch_satellite: bool = True,
) -> Path:
    """Write complete candidate package (YAML + images).

    Args:
        output_dir: Base output directory (will create subdirectory for candidate)
        candidate: Sampled candidate
        batch_id: Batch identifier
        fetch_satellite: Whether to fetch satellite imagery

    Returns:
        Path to candidate directory
    """
    # Create candidate directory
    candidate_dir_name = f"{candidate.ref_id}__{candidate.target_id}"
    # Sanitize directory name (remove problematic characters)
    candidate_dir_name = candidate_dir_name.replace("/", "_").replace("\\", "_")
    candidate_dir = output_dir / candidate_dir_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Generate and write metadata
    yaml_content = generate_metadata_yaml(candidate, batch_id)
    metadata_path = candidate_dir / "metadata.yaml"
    metadata_path.write_text(yaml_content)

    # Generate and write images
    satellite_img, geometry_img = render_candidate_images(
        ref_geom=candidate.ref_geometry,
        target_geom=candidate.target_geometry,
        fetch_satellite=fetch_satellite,
    )

    geometry_path = candidate_dir / "geometry.png"
    geometry_img.save(geometry_path)

    if satellite_img:
        satellite_path = candidate_dir / "satellite.png"
        satellite_img.save(satellite_path)

    return candidate_dir


def write_batch_manifest(
    batch_dir: Path,
    candidates: list[SampledCandidate],
    batch_id: str,
    dataset_name: str,
    config_info: dict[str, Any] | None = None,
) -> Path:
    """Write batch manifest with metadata about the batch.

    Args:
        batch_dir: Batch directory
        candidates: List of sampled candidates
        batch_id: Batch identifier
        dataset_name: Source dataset name
        config_info: Additional configuration info

    Returns:
        Path to manifest file
    """
    # Count by bucket
    bucket_counts: dict[str, int] = {}
    for c in candidates:
        bucket = c.confidence_bucket
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # Count by decision
    decision_counts: dict[str, int] = {}
    for c in candidates:
        decision = c.ml_decision
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "total_candidates": len(candidates),
        "by_confidence_bucket": bucket_counts,
        "by_ml_decision": decision_counts,
        "config": config_info or {},
        "candidates": [
            {
                "ref_id": c.ref_id,
                "target_id": c.target_id,
                "ml_confidence": _round_value(c.ml_confidence),
                "ml_decision": c.ml_decision,
                "confidence_bucket": c.confidence_bucket,
            }
            for c in candidates
        ],
    }

    manifest_path = batch_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )

    return manifest_path


def generate_batch(
    candidates: list[SampledCandidate],
    output_dir: Path,
    batch_id: str | None = None,
    dataset_name: str | None = None,
    fetch_satellite: bool = True,
    config_info: dict[str, Any] | None = None,
) -> Path:
    """Generate a complete batch of candidate packages.

    Args:
        candidates: List of sampled candidates
        output_dir: Base output directory (e.g., agent_labels)
        batch_id: Batch identifier (auto-generated if not provided)
        dataset_name: Source dataset name
        fetch_satellite: Whether to fetch satellite imagery
        config_info: Additional configuration info for manifest

    Returns:
        Path to batch directory
    """
    # Generate batch ID if not provided
    if batch_id is None:
        batch_id = f"batch_{datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')}"

    # Infer dataset name if not provided
    if dataset_name is None and candidates:
        dataset_name = candidates[0].dataset

    # Create batch directory structure
    batch_dir = output_dir / "batches" / batch_id
    candidates_dir = batch_dir / "candidates"
    labels_dir = batch_dir / "labels"

    batch_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating batch {batch_id} with {len(candidates)} candidates")

    # Write candidate packages
    for i, candidate in enumerate(candidates):
        if i % 10 == 0:
            logger.info(f"Processing candidate {i + 1}/{len(candidates)}")

        write_candidate_package(
            output_dir=candidates_dir,
            candidate=candidate,
            batch_id=batch_id,
            fetch_satellite=fetch_satellite,
        )

    # Write manifest
    write_batch_manifest(
        batch_dir=batch_dir,
        candidates=candidates,
        batch_id=batch_id,
        dataset_name=dataset_name or "unknown",
        config_info=config_info,
    )

    logger.info(f"Batch generated at {batch_dir}")
    return batch_dir
