#!/usr/bin/env python3
"""Backfill alignment-based features into existing label files.

This script computes alignment features (coverage, aligned similarity features)
for labeled segment pairs and updates the label CSV files with the new features.

The script:
1. Loads labeled pairs from labels/dataset=*/data.csv
2. Loads geometries from source data (gers_id -> Overture, target_id -> target dataset)
3. Runs linestring_alignment() on each pair
4. Recomputes similarity features on aligned sublines
5. Adds new coverage features
6. Updates CSV in place (preserving label, labeler, timestamp)

Usage:
    python scripts/backfill_aligned_features.py
    python scripts/backfill_aligned_features.py --labels-dir labels --data-dir data/raw
    python scripts/backfill_aligned_features.py --dataset boston_streets
    python scripts/backfill_aligned_features.py --dry-run
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

from matcher.features.alignment import (
    AlignmentResult,
    compute_coverage_features,
    create_subline,
    linestring_alignment,
)
from matcher.features.geometric import compute_geometric_features

# Dataset name to (target file, reference file) mapping
# Reference file is the Overture segments file to use for this dataset
DATASET_CONFIG = {
    "boston_bikes": ("boston_bike_network.parquet", "overture_segments.parquet"),
    "boston_sidewalks": ("boston_sidewalks.parquet", "overture_segments.parquet"),
    "boston_sidewalks_full": ("boston_sidewalks.parquet", "overture_segments.parquet"),
    "boston_sidewalks_relational": ("boston_sidewalks.parquet", "overture_segments.parquet"),
    "boston_streets": ("boston_streets.parquet", "overture_segments.parquet"),
    "osm": ("osm_segments.parquet", "overture_segments.parquet"),
    "fort_collins_streets": ("fort_collins_streets.parquet", "overture_fort_collins_segments.parquet"),
    "fort_collins_sidewalks": ("fort_collins_sidewalks.parquet", "overture_fort_collins_segments.parquet"),
    "frisco_trails": ("frisco_trails.parquet", "overture_frisco_segments.parquet"),
    "frisco_roads": ("frisco_roads.parquet", "overture_frisco_segments.parquet"),
}

# For backward compatibility
DATASET_TARGET_MAP = {k: v[0] for k, v in DATASET_CONFIG.items()}

# Alignment coverage feature columns that will be added
ALIGNMENT_FEATURE_COLUMNS = [
    "ref_coverage",
    "target_coverage",
    "min_coverage",
    "coverage_ratio",
]

# Geometric features that should be recomputed on aligned sublines
SIMILARITY_FEATURE_COLUMNS = [
    "hausdorff_distance",
    "mean_hausdorff_distance",
    "buffer_iou",
    "overlap_ratio",
    "heading_delta",
    "length_ratio",
    "projection_distance",
    "centroid_distance",
    "collinear_gap_ratio",
]


def _get_utm_crs_for_geometry(geom) -> CRS | None:
    """Get appropriate UTM CRS for a geometry based on its centroid.

    Args:
        geom: Shapely geometry (assumed in WGS84)

    Returns:
        CRS for appropriate UTM zone, or None if not needed
    """
    centroid = geom.centroid
    lon, lat = centroid.x, centroid.y

    # Check if looks like geographic coordinates
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None  # Already projected

    # Check for coordinates that look too large for lat/lon
    if lon > 1000 or lon < -1000:
        return None

    # Calculate UTM zone
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone

    return CRS.from_epsg(epsg)


def _project_geometry(geom, transformer):
    """Project a geometry using a transformer."""
    if geom is None or geom.is_empty:
        return geom
    return shapely_transform(transformer.transform, geom)


def compute_aligned_features(
    ref_geom, target_geom
) -> tuple[AlignmentResult | None, dict[str, float]]:
    """Compute alignment and aligned similarity features for a pair.

    Args:
        ref_geom: Reference (GERS) geometry
        target_geom: Target geometry

    Returns:
        Tuple of (AlignmentResult or None, dict of features)
    """
    if ref_geom is None or target_geom is None:
        return None, {}

    if ref_geom.is_empty or target_geom.is_empty:
        return None, {}

    try:
        # Project to UTM for accurate alignment (if in geographic CRS)
        utm_crs = _get_utm_crs_for_geometry(ref_geom)
        if utm_crs is not None:
            transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
            ref_proj = _project_geometry(ref_geom, transformer)
            target_proj = _project_geometry(target_geom, transformer)
        else:
            ref_proj = ref_geom
            target_proj = target_geom

        # Compute alignment on projected geometries
        alignment = linestring_alignment(ref_proj, target_proj)

        # Compute coverage features
        coverage_feats = compute_coverage_features(alignment)

        # Extract aligned sublines
        ref_subline = create_subline(
            ref_geom, alignment.overture_start_frac, alignment.overture_end_frac
        )
        target_subline = create_subline(
            target_geom, alignment.dataset_start_frac, alignment.dataset_end_frac
        )

        # Use sublines if valid, otherwise fall back to full geometry
        geom_for_similarity_ref = ref_subline if ref_subline else ref_geom
        geom_for_similarity_target = target_subline if target_subline else target_geom

        # Recompute similarity features on aligned sublines
        geom_features = compute_geometric_features(
            geom_for_similarity_ref, geom_for_similarity_target
        )

        # Build feature dict
        features = {
            # Coverage features
            "ref_coverage": coverage_feats["ref_coverage"],
            "target_coverage": coverage_feats["target_coverage"],
            "min_coverage": coverage_feats["min_coverage"],
            "coverage_ratio": coverage_feats["coverage_ratio"],
            # Similarity features (recomputed on aligned sublines)
            "hausdorff_distance": geom_features.hausdorff_distance,
            "mean_hausdorff_distance": geom_features.mean_hausdorff_distance,
            "buffer_iou": geom_features.buffer_iou,
            "overlap_ratio": geom_features.overlap_ratio,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
            "projection_distance": geom_features.projection_distance,
            "centroid_distance": geom_features.centroid_distance,
            "collinear_gap_ratio": geom_features.collinear_gap_ratio,
        }

        return alignment, features

    except Exception as e:
        logger.warning(f"Failed to compute alignment: {e}")
        return None, {}


def backfill_dataset(
    dataset_name: str,
    labels_dir: Path,
    data_dir: Path,
    ref_gdf: gpd.GeoDataFrame,
    recompute_similarity: bool = True,
    dry_run: bool = False,
) -> int:
    """Backfill alignment features for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'boston_streets')
        labels_dir: Path to labels directory
        data_dir: Path to raw data directory
        ref_gdf: Pre-loaded reference GeoDataFrame (Overture segments)
        recompute_similarity: If True, also recompute similarity features on aligned sublines
        dry_run: If True, don't write changes

    Returns:
        Number of labels processed
    """
    label_path = labels_dir / f"dataset={dataset_name}" / "data.csv"
    if not label_path.exists():
        logger.warning(f"No labels found for {dataset_name}")
        return 0

    target_file = DATASET_TARGET_MAP.get(dataset_name)
    if not target_file:
        logger.warning(f"Unknown dataset: {dataset_name}")
        return 0

    target_path = data_dir / target_file
    if not target_path.exists():
        logger.warning(f"Target file not found: {target_path}")
        return 0

    logger.info(f"Processing {dataset_name}...")

    # Load labels
    df = pd.read_csv(label_path)
    logger.info(f"  Loaded {len(df)} labels")

    # Load target data
    logger.info(f"  Loading target data: {target_file}")
    target_gdf = gpd.read_parquet(target_path)
    target_gdf = target_gdf.set_index("id")

    # Build lookup dicts for geometries
    ref_geom_lookup = {str(idx): row.geometry for idx, row in ref_gdf.iterrows()}
    target_geom_lookup = {str(idx): row.geometry for idx, row in target_gdf.iterrows()}

    # Compute alignment features for each label
    new_features = []
    missing_ref = 0
    missing_target = 0
    successful_alignments = 0

    for _, row in df.iterrows():
        gers_id = str(row["gers_id"])
        target_id = str(row["target_id"])

        ref_geom = ref_geom_lookup.get(gers_id)
        target_geom = target_geom_lookup.get(target_id)

        if ref_geom is None:
            missing_ref += 1
        if target_geom is None:
            missing_target += 1

        alignment, features = compute_aligned_features(ref_geom, target_geom)

        if alignment is not None:
            successful_alignments += 1
        else:
            # Use default values for failed alignments
            features = {
                "ref_coverage": 0.0,
                "target_coverage": 0.0,
                "min_coverage": 0.0,
                "coverage_ratio": 0.0,
            }
            # Keep existing similarity features if not recomputing
            if recompute_similarity:
                for col in SIMILARITY_FEATURE_COLUMNS:
                    if col in row:
                        features[col] = row[col]

        new_features.append(features)

    logger.info(f"  Successful alignments: {successful_alignments}/{len(df)}")
    if missing_ref > 0:
        logger.warning(f"  {missing_ref} labels with missing reference segments")
    if missing_target > 0:
        logger.warning(f"  {missing_target} labels with missing target segments")

    # Add new columns to dataframe
    features_df = pd.DataFrame(new_features)

    # Determine which columns to update
    columns_to_update = ALIGNMENT_FEATURE_COLUMNS.copy()
    if recompute_similarity:
        columns_to_update.extend(SIMILARITY_FEATURE_COLUMNS)

    # Drop existing columns if they exist (for re-runs)
    for col in columns_to_update:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Only add columns that exist in features_df
    cols_to_add = [c for c in columns_to_update if c in features_df.columns]
    df = pd.concat([df, features_df[cols_to_add]], axis=1)

    if not dry_run:
        df.to_csv(label_path, index=False)
        logger.info(f"  Saved {len(df)} labels with alignment features")
    else:
        logger.info(f"  [DRY RUN] Would save {len(df)} labels with alignment features")

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill alignment-based features into label files"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("labels"),
        help="Path to labels directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only this dataset (default: all)",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only add coverage features, don't recompute similarity features",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just show what would be done",
    )
    args = parser.parse_args()

    # Resolve paths
    labels_dir = args.labels_dir.resolve()
    data_dir = args.data_dir.resolve()

    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return 1

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(DATASET_CONFIG.keys())

    # Group datasets by reference file
    ref_file_to_datasets = {}
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIG:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue
        _, ref_file = DATASET_CONFIG[dataset_name]
        if ref_file not in ref_file_to_datasets:
            ref_file_to_datasets[ref_file] = []
        ref_file_to_datasets[ref_file].append(dataset_name)

    # Process datasets grouped by reference file
    total_processed = 0
    for ref_file, dataset_names in ref_file_to_datasets.items():
        ref_path = data_dir / ref_file
        if not ref_path.exists():
            logger.warning(f"Reference file not found: {ref_path}, skipping datasets: {dataset_names}")
            continue

        logger.info(f"Loading reference data from {ref_file}...")
        ref_gdf = gpd.read_parquet(ref_path)
        ref_gdf = ref_gdf.set_index("id")
        logger.info(f"Loaded {len(ref_gdf)} reference segments")

        for dataset_name in dataset_names:
            count = backfill_dataset(
                dataset_name,
                labels_dir,
                data_dir,
                ref_gdf,
                recompute_similarity=not args.coverage_only,
                dry_run=args.dry_run,
            )
            total_processed += count

    logger.info(f"\nBackfill complete: {total_processed} labels processed")
    return 0


if __name__ == "__main__":
    exit(main())
