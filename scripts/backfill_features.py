#!/usr/bin/env python3
"""Unified backfill script for alignment and topology features.

This script computes alignment and/or topology features for labeled segment pairs
and updates the label CSV files with the new features.

Usage:
    # Backfill all features (both alignment and topology)
    python scripts/backfill_features.py

    # Alignment features only
    python scripts/backfill_features.py --alignment-only

    # Topology features only
    python scripts/backfill_features.py --topology-only

    # Specific dataset
    python scripts/backfill_features.py --dataset boston_streets

    # Dry run (show what would be done)
    python scripts/backfill_features.py --dry-run
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
from matcher.features.relational import compute_perpendicular_offset
from matcher.features.semantic import compute_name_similarity
from matcher.features.spatial_context import (
    SpatialContextIndex,
    build_inferred_graph,
    compute_all_topology,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    compute_endpoint_features,
    compute_road_graphlet_features,
    graphlet_segment_similarity,
)

# Dataset name to (target file, reference file) mapping
# Reference file is the Overture segments file to use for this dataset
# Note: Dataset names have us_ prefix but data files use shorter names
# Reference file naming: {region}_overture_segments.parquet (e.g., boston_overture_segments.parquet)
DATASET_CONFIG = {
    "us_boston_bikes": ("boston_bike_network.parquet", "boston_overture_segments.parquet"),
    "us_boston_sidewalks": ("boston_sidewalks.parquet", "boston_overture_segments.parquet"),
    "us_boston_streets": ("boston_streets.parquet", "boston_overture_segments.parquet"),
    "us_boston_osm": ("boston_osm_segments.parquet", "boston_overture_segments.parquet"),
    "us_fort_collins_streets": (
        "fort_collins_streets.parquet",
        "fort_collins_overture_segments.parquet",
    ),
    "us_fort_collins_sidewalks": (
        "fort_collins_sidewalks.parquet",
        "fort_collins_overture_segments.parquet",
    ),
    "us_frisco_trails": ("frisco_trails.parquet", "frisco_overture_segments.parquet"),
    "us_frisco_roads": ("frisco_roads.parquet", "frisco_overture_segments.parquet"),
}

# Alignment coverage feature columns
ALIGNMENT_FEATURE_COLUMNS = [
    "ref_coverage",
    "target_coverage",
    "min_coverage",
    "coverage_ratio",
]

# Geometric features recomputed on aligned sublines
# Distance features use _m suffix to indicate meters (matching config.py)
SIMILARITY_FEATURE_COLUMNS = [
    "hausdorff_distance_m",
    "mean_hausdorff_distance_m",
    "hausdorff_p95_m",
    "buffer_iou_5m",
    "buffer_iou_15m",
    "overlap_ratio",
    "heading_delta",
    "length_ratio",
    "projection_distance_m",
    "centroid_distance_m",
    "collinear_gap_ratio",
]

# Topology feature columns
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

# Semantic feature columns (name similarity)
SEMANTIC_FEATURE_COLUMNS = [
    "name_soundex",
    "name_metaphone",
    "has_name_ref",
    "has_name_target",
    "name_is_generic",
]

# Endpoint proximity feature columns (direction-invariant)
# Distance features use _m suffix to indicate meters (matching config.py)
ENDPOINT_FEATURE_COLUMNS = [
    "min_endpoint_proximity_m",
    "max_endpoint_proximity_m",
    "shared_endpoint_count",
]

# Lateral offset feature columns (IQR and P95 instead of consistency)
# Distance features use _m suffix to indicate meters (matching config.py)
LATERAL_FEATURE_COLUMNS = [
    "lateral_offset_m",
    "lateral_offset_iqr_m",
    "lateral_offset_p95_m",
]

# Graphlet feature columns
GRAPHLET_FEATURE_COLUMNS = [
    "graphlet_similarity",
    "endpoint_degree_similarity",
]


def _get_local_equidistant_crs(geom) -> CRS | None:
    """Get local azimuthal equidistant CRS centered on geometry.

    Args:
        geom: Shapely geometry (assumed in WGS84)

    Returns:
        CRS for local projection, or None if not needed
    """
    centroid = geom.centroid
    lon, lat = centroid.x, centroid.y

    # Check if looks like geographic coordinates
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None  # Already projected

    # Check for coordinates that look too large for lat/lon
    if lon > 1000 or lon < -1000:
        return None

    # Create local azimuthal equidistant CRS
    proj_string = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    return CRS.from_proj4(proj_string)


def _project_geometry(geom, transformer):
    """Project a geometry using a transformer."""
    if geom is None or geom.is_empty:
        return geom
    return shapely_transform(transformer.transform, geom)


def get_default_topology() -> dict:
    """Return default topology for missing segments."""
    return {
        "from_degree": 1,
        "to_degree": 1,
        "is_dead_end": True,
        "is_intersection": False,
        "degree_signature": (1, 1),
    }


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
        # Project to local equidistant CRS for accurate alignment
        local_crs = _get_local_equidistant_crs(ref_geom)
        if local_crs is not None:
            transformer = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)
            ref_proj = _project_geometry(ref_geom, transformer)
            target_proj = _project_geometry(target_geom, transformer)
        else:
            ref_proj = ref_geom
            target_proj = target_geom

        # Compute alignment on projected geometries
        alignment = linestring_alignment(ref_proj, target_proj)

        # Compute coverage features
        coverage_feats = compute_coverage_features(alignment)

        # Extract aligned sublines from original WGS84 geometries
        # (both backfill and scoring should use WGS84 for consistency)
        ref_subline = create_subline(
            ref_geom, alignment.overture_start_frac, alignment.overture_end_frac
        )
        target_subline = create_subline(
            target_geom, alignment.dataset_start_frac, alignment.dataset_end_frac
        )

        # Use sublines if valid, otherwise fall back to full geometry
        geom_for_similarity_ref = ref_subline if ref_subline else ref_geom
        geom_for_similarity_target = target_subline if target_subline else target_geom

        # Recompute similarity features on aligned sublines (in WGS84)
        geom_features = compute_geometric_features(
            geom_for_similarity_ref, geom_for_similarity_target
        )

        # Build feature dict (using _m suffix for distance features to match config.py)
        features = {
            # Coverage features
            "ref_coverage": coverage_feats["ref_coverage"],
            "target_coverage": coverage_feats["target_coverage"],
            "min_coverage": coverage_feats["min_coverage"],
            "coverage_ratio": coverage_feats["coverage_ratio"],
            # Similarity features (recomputed on aligned sublines)
            "hausdorff_distance_m": geom_features.hausdorff_distance,
            "mean_hausdorff_distance_m": geom_features.mean_hausdorff_distance,
            "hausdorff_p95_m": geom_features.hausdorff_p95_distance,
            "buffer_iou_5m": geom_features.buffer_iou_5m,
            "buffer_iou_15m": geom_features.buffer_iou_15m,
            "overlap_ratio": geom_features.overlap_ratio,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
            "projection_distance_m": geom_features.projection_distance,
            "centroid_distance_m": geom_features.centroid_distance,
            "collinear_gap_ratio": geom_features.collinear_gap_ratio,
        }

        return alignment, features

    except Exception as e:
        logger.warning(f"Failed to compute alignment: {e}")
        return None, {}


def compute_topology_for_pair(ref_topo: dict, target_topo: dict) -> dict[str, float]:
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


def load_and_compute_topology(
    path: Path,
    id_column: str = "id",
    ids_to_compute: set | None = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Load a GeoDataFrame and compute topology features.

    Args:
        path: Path to parquet file
        id_column: Column to use as ID
        ids_to_compute: If provided, only compute topology for these IDs

    Returns:
        Tuple of (GeoDataFrame, topology dict)
    """
    gdf = gpd.read_parquet(path)
    gdf = gdf.set_index(id_column)
    gdf_reset = gdf.reset_index()
    gdf_reset[id_column] = gdf_reset[id_column].astype(str)

    # Use explicit connector-based topology if available, otherwise geometry inference
    connectors_col = "connectors" if "connectors" in gdf_reset.columns else None
    topology = compute_all_topology(
        gdf_reset,
        id_column=id_column,
        tolerance_m=5.0,
        ids_to_compute=ids_to_compute,
        connectors_column=connectors_col,
    )

    logger.debug(f"Computed topology for {len(topology)} segments")
    return gdf_reset, topology


def backfill_dataset(
    dataset_name: str,
    labels_dir: Path,
    data_dir: Path,
    ref_gdf: gpd.GeoDataFrame | None = None,
    ref_topology: dict | None = None,
    ref_graphlet_data: tuple | None = None,
    compute_alignment: bool = True,
    compute_topology: bool = True,
    compute_semantic: bool = True,
    compute_endpoint: bool = True,
    compute_lateral: bool = True,
    compute_graphlet: bool = True,
    recompute_similarity: bool = True,
    dry_run: bool = False,
) -> int:
    """Backfill features for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'boston_streets')
        labels_dir: Path to labels directory
        data_dir: Path to raw data directory
        ref_gdf: Pre-loaded reference GeoDataFrame (for alignment)
        ref_topology: Pre-computed reference topology dict
        ref_graphlet_data: Pre-computed reference graphlet data (G, seg_to_start, seg_to_end, features)
        compute_alignment: Whether to compute alignment features
        compute_topology: Whether to compute topology features
        compute_semantic: Whether to compute semantic features (name_soundex, name_metaphone)
        compute_endpoint: Whether to compute endpoint proximity features
        compute_lateral: Whether to compute lateral offset features
        compute_graphlet: Whether to compute graphlet features
        recompute_similarity: If True, also recompute similarity features on aligned sublines
        dry_run: If True, don't write changes

    Returns:
        Number of labels processed
    """
    label_path = labels_dir / f"dataset={dataset_name}" / "data.csv"
    if not label_path.exists():
        logger.warning(f"No labels found for {dataset_name}")
        return 0

    if dataset_name not in DATASET_CONFIG:
        logger.warning(f"Unknown dataset: {dataset_name}")
        return 0

    target_file, _ = DATASET_CONFIG[dataset_name]
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

    # Build geometry lookup
    target_geom_lookup = {str(idx): row.geometry for idx, row in target_gdf.iterrows()}

    # Initialize feature containers
    alignment_features = []
    topology_features = []

    # Compute alignment features if requested
    if compute_alignment and ref_gdf is not None:
        ref_geom_lookup = {str(idx): row.geometry for idx, row in ref_gdf.iterrows()}

        missing_ref = 0
        missing_target = 0
        successful_alignments = 0
        failed_alignments = []

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
                failed_alignments.append((gers_id, target_id))
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

            alignment_features.append(features)

        logger.info(f"  Successful alignments: {successful_alignments}/{len(df)}")
        if missing_ref > 0:
            logger.warning(f"  {missing_ref} labels with missing reference segments")
        if missing_target > 0:
            logger.warning(f"  {missing_target} labels with missing target segments")
        if failed_alignments:
            logger.warning(f"  {len(failed_alignments)} pairs had failed alignments")

    # Compute topology features if requested
    if compute_topology and ref_topology is not None:
        # Get unique target IDs needed
        target_ids = set(df["target_id"].astype(str).unique())
        logger.info(f"  Need topology for {len(target_ids)} unique target segments")

        # Compute topology for target data
        _, target_topology = load_and_compute_topology(target_path, ids_to_compute=target_ids)
        logger.info(f"  Computed topology for {len(target_topology)} target segments")

        missing_ref_topo = 0
        missing_target_topo = 0

        for _, row in df.iterrows():
            gers_id = str(row["gers_id"])
            target_id = str(row["target_id"])

            ref_topo = ref_topology.get(gers_id, get_default_topology())
            if gers_id not in ref_topology:
                missing_ref_topo += 1

            tgt_topo = target_topology.get(target_id, get_default_topology())
            if target_id not in target_topology:
                missing_target_topo += 1

            topology_features.append(compute_topology_for_pair(ref_topo, tgt_topo))

        if missing_ref_topo > 0:
            logger.warning(f"  {missing_ref_topo} labels with missing reference topology")
        if missing_target_topo > 0:
            logger.warning(f"  {missing_target_topo} labels with missing target topology")

    # Compute semantic features (name_soundex, name_metaphone)
    semantic_features = []
    if compute_semantic and ref_gdf is not None:
        logger.info("  Computing semantic features...")
        # Get name columns
        ref_name_col = "name" if "name" in ref_gdf.columns else "names"
        target_name_col = "name" if "name" in target_gdf.columns else "names"

        ref_gdf_reset = ref_gdf.reset_index()
        ref_name_lookup = {}
        if ref_name_col in ref_gdf_reset.columns:
            for _, row in ref_gdf_reset.iterrows():
                ref_name_lookup[str(row["id"])] = row.get(ref_name_col)

        target_gdf_reset = target_gdf.reset_index()
        target_name_lookup = {}
        if target_name_col in target_gdf_reset.columns:
            for _, row in target_gdf_reset.iterrows():
                target_name_lookup[str(row["id"])] = row.get(target_name_col)

        for _, row in df.iterrows():
            gers_id = str(row["gers_id"])
            target_id = str(row["target_id"])

            ref_name = ref_name_lookup.get(gers_id)
            target_name = target_name_lookup.get(target_id)

            name_sim = compute_name_similarity(ref_name, target_name)
            semantic_features.append(
                {
                    "name_soundex": name_sim.get("soundex_match", 0.5),
                    "name_metaphone": name_sim.get("metaphone_similarity", 0.5),
                    "has_name_ref": name_sim.get("has_name_ref", 0.0),
                    "has_name_target": name_sim.get("has_name_target", 0.0),
                    "name_is_generic": name_sim.get("name_is_generic", 0.0),
                }
            )

        logger.info(f"  Computed semantic features for {len(semantic_features)} pairs")

    # Compute endpoint proximity features
    endpoint_features = []
    if compute_endpoint:
        logger.info("  Computing endpoint proximity features...")
        # Build spatial index for target
        target_gdf_reset = target_gdf.reset_index()
        target_gdf_reset["id"] = target_gdf_reset["id"].astype(str)
        target_index = SpatialContextIndex()
        target_index.build_from_gdf(target_gdf_reset, id_column="id")

        # Create index lookup for target
        target_idx_lookup = {str(row["id"]): idx for idx, row in target_gdf_reset.iterrows()}

        for _, row in df.iterrows():
            target_id = str(row["target_id"])
            target_idx = target_idx_lookup.get(target_id)

            if target_idx is not None:
                target_geom = target_gdf_reset.geometry.iloc[target_idx]
                if target_geom is not None and not target_geom.is_empty:
                    ep_feats = compute_endpoint_features(
                        target_geom, target_index, exclude_segment_idx=target_idx
                    )
                    endpoint_features.append(ep_feats)
                else:
                    endpoint_features.append(
                        {
                            "min_endpoint_proximity_m": 10000.0,
                            "max_endpoint_proximity_m": 10000.0,
                            "shared_endpoint_count": 0,
                        }
                    )
            else:
                endpoint_features.append(
                    {
                        "min_endpoint_proximity_m": 10000.0,
                        "max_endpoint_proximity_m": 10000.0,
                        "shared_endpoint_count": 0,
                    }
                )

        logger.info(f"  Computed endpoint features for {len(endpoint_features)} pairs")

    # Compute lateral offset features
    lateral_features = []
    if compute_lateral and ref_gdf is not None:
        logger.info("  Computing lateral offset features...")
        ref_geom_lookup = {str(idx): row.geometry for idx, row in ref_gdf.iterrows()}

        for _, row in df.iterrows():
            gers_id = str(row["gers_id"])
            target_id = str(row["target_id"])

            ref_geom = ref_geom_lookup.get(gers_id)
            target_geom = target_geom_lookup.get(target_id)

            if ref_geom is not None and target_geom is not None:
                try:
                    lateral_offset, lateral_iqr, lateral_p95 = compute_perpendicular_offset(
                        target_geom, ref_geom
                    )
                    lateral_features.append(
                        {
                            "lateral_offset_m": min(lateral_offset, 10000.0),
                            "lateral_offset_iqr_m": min(lateral_iqr, 10000.0),
                            "lateral_offset_p95_m": min(lateral_p95, 10000.0),
                        }
                    )
                except Exception:
                    lateral_features.append(
                        {
                            "lateral_offset_m": 10000.0,
                            "lateral_offset_iqr_m": 10000.0,
                            "lateral_offset_p95_m": 10000.0,
                        }
                    )
            else:
                lateral_features.append(
                    {
                        "lateral_offset_m": 10000.0,
                        "lateral_offset_iqr_m": 10000.0,
                        "lateral_offset_p95_m": 10000.0,
                    }
                )

        logger.info(f"  Computed lateral features for {len(lateral_features)} pairs")

    # Compute graphlet features
    graphlet_features = []
    if compute_graphlet and ref_graphlet_data is not None:
        logger.info("  Computing graphlet features...")
        ref_G, ref_seg_to_start, ref_seg_to_end, ref_node_features = ref_graphlet_data

        # Build graphlet data for target
        target_gdf_reset = target_gdf.reset_index()
        target_gdf_reset["id"] = target_gdf_reset["id"].astype(str)
        target_G, target_seg_to_start, target_seg_to_end = build_inferred_graph(
            target_gdf_reset, id_column="id", tolerance_m=5.0
        )
        target_node_features = compute_road_graphlet_features(target_G)

        for _, row in df.iterrows():
            gers_id = str(row["gers_id"])
            target_id = str(row["target_id"])

            try:
                graphlet_sim = graphlet_segment_similarity(
                    gers_id,
                    target_id,
                    ref_node_features,
                    target_node_features,
                    (ref_seg_to_start, ref_seg_to_end),
                    (target_seg_to_start, target_seg_to_end),
                )
                graphlet_features.append(graphlet_sim)
            except Exception:
                graphlet_features.append(
                    {
                        "graphlet_similarity": 0.5,
                        "endpoint_degree_similarity": 0.5,
                    }
                )

        logger.info(f"  Computed graphlet features for {len(graphlet_features)} pairs")

    # Update dataframe with new features
    columns_to_update = []

    if alignment_features:
        alignment_df = pd.DataFrame(alignment_features)
        columns_to_update.extend(ALIGNMENT_FEATURE_COLUMNS)
        if recompute_similarity:
            columns_to_update.extend(SIMILARITY_FEATURE_COLUMNS)

        # Drop existing columns if they exist
        for col in columns_to_update:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Only add columns that exist in features_df
        cols_to_add = [c for c in columns_to_update if c in alignment_df.columns]
        df = pd.concat([df, alignment_df[cols_to_add]], axis=1)

    if topology_features:
        topology_df = pd.DataFrame(topology_features)

        # Drop existing topology columns
        for col in TOPOLOGY_FEATURE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.concat([df, topology_df], axis=1)

    if semantic_features:
        semantic_df = pd.DataFrame(semantic_features)

        # Drop existing semantic columns
        for col in SEMANTIC_FEATURE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.concat([df, semantic_df], axis=1)

    if endpoint_features:
        endpoint_df = pd.DataFrame(endpoint_features)

        # Drop existing endpoint columns
        for col in ENDPOINT_FEATURE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.concat([df, endpoint_df], axis=1)

    if lateral_features:
        lateral_df = pd.DataFrame(lateral_features)

        # Drop existing lateral columns
        for col in LATERAL_FEATURE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.concat([df, lateral_df], axis=1)

    if graphlet_features:
        graphlet_df = pd.DataFrame(graphlet_features)

        # Drop existing graphlet columns
        for col in GRAPHLET_FEATURE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.concat([df, graphlet_df], axis=1)

    if not dry_run:
        df.to_csv(label_path, index=False, float_format=lambda x: f"{x:.10g}")
        features_added = []
        if compute_alignment:
            features_added.append("alignment")
        if compute_topology:
            features_added.append("topology")
        if compute_semantic:
            features_added.append("semantic")
        if compute_endpoint:
            features_added.append("endpoint")
        if compute_lateral:
            features_added.append("lateral")
        if compute_graphlet:
            features_added.append("graphlet")
        logger.info(f"  Saved {len(df)} labels with {', '.join(features_added)} features")
    else:
        logger.info(f"  [DRY RUN] Would save {len(df)} labels")

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Unified backfill script for alignment and topology features"
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
        "--alignment-only",
        action="store_true",
        help="Only compute alignment features (skip topology)",
    )
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Only compute topology features (skip alignment)",
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

    # Determine what to compute
    compute_alignment = not args.topology_only
    compute_topology = not args.alignment_only

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

    # If computing topology, collect all reference IDs first
    all_ref_ids = set()
    if compute_topology:
        logger.info("Collecting reference IDs from labels...")
        for dataset_name in datasets:
            label_path = labels_dir / f"dataset={dataset_name}" / "data.csv"
            if label_path.exists():
                df = pd.read_csv(label_path, usecols=["gers_id"])
                all_ref_ids.update(df["gers_id"].astype(str).unique())
        logger.info(f"Found {len(all_ref_ids)} unique reference IDs")

    # Process datasets grouped by reference file
    total_processed = 0
    for ref_file, dataset_names in ref_file_to_datasets.items():
        ref_path = data_dir / ref_file
        if not ref_path.exists():
            logger.warning(
                f"Reference file not found: {ref_path}, skipping datasets: {dataset_names}"
            )
            continue

        # Load reference data if needed
        ref_gdf = None
        ref_topology = None

        if compute_alignment:
            logger.info(f"Loading reference data from {ref_file}...")
            ref_gdf = gpd.read_parquet(ref_path)
            ref_gdf = ref_gdf.set_index("id")
            logger.info(f"Loaded {len(ref_gdf)} reference segments")

        if compute_topology:
            logger.info(f"Computing reference topology from {ref_file}...")
            # Only compute topology for IDs we need
            ref_ids_for_file = set()
            for dataset_name in dataset_names:
                label_path = labels_dir / f"dataset={dataset_name}" / "data.csv"
                if label_path.exists():
                    df = pd.read_csv(label_path, usecols=["gers_id"])
                    ref_ids_for_file.update(df["gers_id"].astype(str).unique())

            _, ref_topology = load_and_compute_topology(ref_path, ids_to_compute=ref_ids_for_file)
            logger.info(f"Computed topology for {len(ref_topology)} reference segments")

        # Compute graphlet data for reference file
        ref_graphlet_data = None
        if ref_gdf is not None:
            logger.info(f"Computing reference graphlet features from {ref_file}...")
            ref_gdf_reset = ref_gdf.reset_index()
            ref_gdf_reset["id"] = ref_gdf_reset["id"].astype(str)
            ref_G, ref_seg_to_start, ref_seg_to_end = build_inferred_graph(
                ref_gdf_reset, id_column="id", tolerance_m=5.0
            )
            ref_node_features = compute_road_graphlet_features(ref_G)
            ref_graphlet_data = (ref_G, ref_seg_to_start, ref_seg_to_end, ref_node_features)
            logger.info(f"Computed graphlet features for {len(ref_node_features)} reference nodes")

        for dataset_name in dataset_names:
            count = backfill_dataset(
                dataset_name,
                labels_dir,
                data_dir,
                ref_gdf=ref_gdf,
                ref_topology=ref_topology,
                ref_graphlet_data=ref_graphlet_data,
                compute_alignment=compute_alignment,
                compute_topology=compute_topology,
                compute_semantic=True,
                compute_endpoint=True,
                compute_lateral=True,
                compute_graphlet=True,
                recompute_similarity=not args.coverage_only,
                dry_run=args.dry_run,
            )
            total_processed += count

    logger.info(f"\nBackfill complete: {total_processed} labels processed")
    return 0


if __name__ == "__main__":
    exit(main())
