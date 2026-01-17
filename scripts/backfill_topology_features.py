#!/usr/bin/env python3
"""Backfill topology features into existing label files.

This script computes topology features (endpoint degrees, dead ends, intersections)
for labeled segment pairs and adds them to the label CSV files.

Usage:
    python scripts/backfill_topology_features.py
    python scripts/backfill_topology_features.py --labels-dir labels --data-dir data/raw
    python scripts/backfill_topology_features.py --dataset boston_streets
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from matcher.features.spatial_context import (
    SpatialContextIndex,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    compute_topology_features,
)

# Dataset name to target file mapping
DATASET_TARGET_MAP = {
    "boston_bikes": "boston_bike_network.parquet",
    "boston_sidewalks": "boston_sidewalks.parquet",
    "boston_sidewalks_full": "boston_sidewalks.parquet",
    "boston_sidewalks_relational": "boston_sidewalks.parquet",
    "boston_streets": "boston_streets.parquet",
    "osm": "osm_segments.parquet",
}

# Topology feature columns that will be added
TOPOLOGY_FEATURE_COLUMNS = [
    "from_degree_ref",
    "to_degree_ref",
    "from_degree_target",
    "to_degree_target",
    "degree_match_score",
    "degree_signature_similarity",
    "is_dead_end_ref",
    "is_dead_end_target",
    "dead_end_match",
    "is_intersection_ref",
    "is_intersection_target",
    "intersection_match",
]


def load_and_index_geodataframe(
    path: Path, id_column: str = "id", ids_to_compute: set | None = None
) -> tuple[gpd.GeoDataFrame, SpatialContextIndex, dict]:
    """Load a GeoDataFrame and build spatial index with pre-computed topology.

    Args:
        path: Path to parquet file
        id_column: Column to use as ID
        ids_to_compute: If provided, only compute topology for these IDs

    Returns:
        Tuple of (GeoDataFrame, SpatialContextIndex, topology dict)
    """
    gdf = gpd.read_parquet(path)
    gdf = gdf.set_index(id_column)
    gdf_reset = gdf.reset_index()
    gdf_reset[id_column] = gdf_reset[id_column].astype(str)

    # Build spatial index with snap_tolerance=0 to skip expensive clustering
    # For backfilling, exact endpoint matches are sufficient
    index = SpatialContextIndex()
    index.build_from_gdf(gdf_reset, id_column=id_column, snap_tolerance=0)

    # Pre-compute topology only for requested segments
    topology = {}
    if ids_to_compute:
        # Create a lookup for fast access
        id_to_idx = {row[id_column]: idx for idx, row in gdf_reset.iterrows()}
        for seg_id in ids_to_compute:
            if seg_id in id_to_idx:
                idx = id_to_idx[seg_id]
                geom = gdf_reset.iloc[idx].geometry
                if geom is not None and not geom.is_empty:
                    topology[seg_id] = compute_topology_features(geom, index)
    else:
        # Compute for all segments
        for _, row in gdf_reset.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                topology[row[id_column]] = compute_topology_features(geom, index)

    return gdf, index, topology


def compute_topology_for_pair(
    ref_topo: dict, target_topo: dict
) -> dict[str, float]:
    """Compute topology match features for a reference-target pair.

    Args:
        ref_topo: Topology features for reference segment
        target_topo: Topology features for target segment

    Returns:
        Dictionary of topology match features
    """
    # Degree match score
    degree_match = compute_degree_match_score(
        ref_topo["from_degree"],
        ref_topo["to_degree"],
        target_topo["from_degree"],
        target_topo["to_degree"],
    )

    # Signature similarity
    sig_sim = compute_degree_signature_similarity(
        ref_topo["degree_signature"], target_topo["degree_signature"]
    )

    # Topology flags
    is_dead_end_ref = 1.0 if ref_topo["is_dead_end"] else 0.0
    is_dead_end_target = 1.0 if target_topo["is_dead_end"] else 0.0
    dead_end_match = 1.0 if is_dead_end_ref == is_dead_end_target else 0.0

    is_intersection_ref = 1.0 if ref_topo["is_intersection"] else 0.0
    is_intersection_target = 1.0 if target_topo["is_intersection"] else 0.0
    intersection_match = 1.0 if is_intersection_ref == is_intersection_target else 0.0

    return {
        "from_degree_ref": ref_topo["from_degree"],
        "to_degree_ref": ref_topo["to_degree"],
        "from_degree_target": target_topo["from_degree"],
        "to_degree_target": target_topo["to_degree"],
        "degree_match_score": degree_match,
        "degree_signature_similarity": sig_sim,
        "is_dead_end_ref": is_dead_end_ref,
        "is_dead_end_target": is_dead_end_target,
        "dead_end_match": dead_end_match,
        "is_intersection_ref": is_intersection_ref,
        "is_intersection_target": is_intersection_target,
        "intersection_match": intersection_match,
    }


def get_default_topology() -> dict:
    """Return default topology for missing segments."""
    return {
        "from_degree": 1,
        "to_degree": 1,
        "is_dead_end": True,
        "is_intersection": False,
        "degree_signature": (1,),
    }


def backfill_dataset(
    dataset_name: str,
    labels_dir: Path,
    data_dir: Path,
    ref_topology: dict,
    dry_run: bool = False,
) -> int:
    """Backfill topology features for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'boston_streets')
        labels_dir: Path to labels directory
        data_dir: Path to raw data directory
        ref_topology: Pre-computed reference topology dict
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

    # Get unique target IDs needed
    target_ids = set(df["target_id"].astype(str).unique())
    logger.info(f"  Need topology for {len(target_ids)} unique target segments")

    # Load and index target data (only compute for needed IDs)
    logger.info(f"  Loading target data: {target_file}")
    _, _, target_topology = load_and_index_geodataframe(
        target_path, ids_to_compute=target_ids
    )
    logger.info(f"  Computed topology for {len(target_topology)} target segments")

    # Compute topology features for each label
    new_features = []
    missing_ref = 0
    missing_target = 0

    for _, row in df.iterrows():
        gers_id = str(row["gers_id"])
        target_id = str(row["target_id"])

        ref_topo = ref_topology.get(gers_id, get_default_topology())
        if gers_id not in ref_topology:
            missing_ref += 1

        tgt_topo = target_topology.get(target_id, get_default_topology())
        if target_id not in target_topology:
            missing_target += 1

        new_features.append(compute_topology_for_pair(ref_topo, tgt_topo))

    if missing_ref > 0:
        logger.warning(f"  {missing_ref} labels with missing reference segments")
    if missing_target > 0:
        logger.warning(f"  {missing_target} labels with missing target segments")

    # Add new columns to dataframe
    features_df = pd.DataFrame(new_features)

    # Drop existing topology columns if they exist (for re-runs)
    for col in TOPOLOGY_FEATURE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = pd.concat([df, features_df], axis=1)

    if not dry_run:
        df.to_csv(label_path, index=False)
        logger.info(f"  Saved {len(df)} labels with topology features")
    else:
        logger.info(f"  [DRY RUN] Would save {len(df)} labels with topology features")

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill topology features into label files"
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
        datasets = list(DATASET_TARGET_MAP.keys())

    # First pass: collect all reference IDs needed across all datasets
    logger.info("Collecting reference IDs from labels...")
    all_ref_ids = set()
    for dataset_name in datasets:
        label_path = labels_dir / f"dataset={dataset_name}" / "data.csv"
        if label_path.exists():
            df = pd.read_csv(label_path, usecols=["gers_id"])
            all_ref_ids.update(df["gers_id"].astype(str).unique())
    logger.info(f"Found {len(all_ref_ids)} unique reference IDs")

    # Load reference data with only needed IDs
    ref_path = data_dir / "overture_segments.parquet"
    if not ref_path.exists():
        logger.error(f"Reference file not found: {ref_path}")
        return 1

    logger.info("Loading reference data and building spatial index...")
    _, _, ref_topology = load_and_index_geodataframe(ref_path, ids_to_compute=all_ref_ids)
    logger.info(f"Computed topology for {len(ref_topology)} reference segments")

    # Process datasets
    total_processed = 0
    for dataset_name in datasets:
        count = backfill_dataset(
            dataset_name, labels_dir, data_dir, ref_topology, args.dry_run
        )
        total_processed += count

    logger.info(f"\nBackfill complete: {total_processed} labels processed")
    return 0


if __name__ == "__main__":
    exit(main())
