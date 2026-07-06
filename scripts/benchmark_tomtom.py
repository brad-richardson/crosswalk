#!/usr/bin/env python3
"""Benchmark: Match TomTom-only Overture segments to non-TomTom Overture segments.

This script evaluates how well TomTom-sourced segments in Overture match back to
non-TomTom Overture segments. This tests the matcher's ability to link premerged data.

Key insight: Overture data contains "premerged" segments from multiple sources.
The `sources` column is an array of dicts like:
    [{"dataset": "OpenStreetMap", "record_id": "w123"},
     {"dataset": "TomTom", "record_id": "tt456"}]

We split Overture into:
- Reference: Segments where TomTom is NOT a source (OSM-only or other sources)
- Target: Segments where TomTom IS the sole source (TomTom-only)

Then run matching to see how well TomTom-only segments link back.
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Add matcher to path for script execution
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release

from crosswalk.fetch.overture import BoundingBox, extract_lr_attributes
from crosswalk.pipeline.runner import run_pipeline
from crosswalk.utils.geometry import filter_to_linestrings

# India bbox with good TomTom coverage (Mumbai area)
# TomTom has significant coverage in urban India
DEFAULT_BBOX = (72.7, 18.85, 73.1, 19.25)


def is_tomtom_only(sources) -> bool:
    """Check if segment has ONLY TomTom as source.

    Args:
        sources: Array of source dicts from Overture

    Returns:
        True if TomTom is the sole source
    """
    if sources is None:
        return False
    # Handle numpy arrays
    if hasattr(sources, "tolist"):
        sources = sources.tolist()
    if not isinstance(sources, list) or len(sources) == 0:
        return False
    return all(isinstance(s, dict) and s.get("dataset") == "TomTom" for s in sources)


def has_tomtom(sources) -> bool:
    """Check if segment has ANY TomTom source.

    Args:
        sources: Array of source dicts from Overture

    Returns:
        True if any source is TomTom
    """
    if sources is None:
        return False
    # Handle numpy arrays
    if hasattr(sources, "tolist"):
        sources = sources.tolist()
    if not isinstance(sources, list):
        return False
    return any(isinstance(s, dict) and s.get("dataset") == "TomTom" for s in sources)


def prepare_for_matching(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Prepare a GeoDataFrame for matching pipeline.

    Ensures required columns and extracts linear-referenced attributes.

    Args:
        gdf: Input GeoDataFrame from Overture

    Returns:
        GeoDataFrame ready for matching
    """
    # Filter to LineString geometries
    gdf = filter_to_linestrings(gdf, source_name="overture")

    # Extract linear-referenced attributes
    gdf = extract_lr_attributes(gdf)

    # Ensure CRS is set
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Ensure id column exists
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index.astype(str)

    # Drop bbox column if present (conflicts with write_covering_bbox)
    if "bbox" in gdf.columns:
        gdf = gdf.drop(columns=["bbox"])

    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TomTom-only vs non-TomTom Overture matching"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        default=DEFAULT_BBOX,
        help=f"Bounding box in WGS84 (default: Mumbai area {DEFAULT_BBOX})",
    )
    parser.add_argument(
        "--release",
        type=str,
        help="Overture release version (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save intermediate files (default: temp dir)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    bbox = BoundingBox(
        xmin=args.bbox[0],
        ymin=args.bbox[1],
        xmax=args.bbox[2],
        ymax=args.bbox[3],
    )

    # Get release version
    release = args.release or get_latest_release()
    logger.info(f"Using Overture release: {release}")

    # Fetch Overture road segments
    logger.info(f"Fetching Overture segments for bbox: {bbox}")
    overture = geodataframe("segment", bbox=bbox.to_tuple(), release=release)

    # Filter to roads only
    if "subtype" in overture.columns:
        overture = overture[overture["subtype"] == "road"]

    initial_count = len(overture)
    logger.info(f"Total road segments: {initial_count}")

    # Check sources column exists
    if "sources" not in overture.columns:
        logger.error("No 'sources' column found in Overture data. Cannot split by source.")
        sys.exit(1)

    # Analyze source distribution
    tomtom_only_mask = overture["sources"].apply(is_tomtom_only)
    has_tomtom_mask = overture["sources"].apply(has_tomtom)
    non_tomtom_mask = ~has_tomtom_mask

    n_tomtom_only = tomtom_only_mask.sum()
    n_has_tomtom = has_tomtom_mask.sum()
    n_non_tomtom = non_tomtom_mask.sum()
    n_mixed = n_has_tomtom - n_tomtom_only

    logger.info("Source distribution:")
    logger.info(f"  TomTom-only: {n_tomtom_only}")
    logger.info(f"  Mixed (has TomTom + other): {n_mixed}")
    logger.info(f"  Non-TomTom: {n_non_tomtom}")

    if n_tomtom_only == 0:
        logger.warning("No TomTom-only segments found in this bbox.")
        logger.info("Try a different region with TomTom coverage (India, Western Europe, etc.)")
        sys.exit(1)

    if n_non_tomtom == 0:
        logger.warning("No non-TomTom segments found - cannot create reference set.")
        sys.exit(1)

    # Split into reference (non-TomTom) and target (TomTom-only)
    target = overture[tomtom_only_mask].copy()
    reference = overture[non_tomtom_mask].copy()

    logger.info("Split:")
    logger.info(f"  Reference (non-TomTom): {len(reference)}")
    logger.info(f"  Target (TomTom-only): {len(target)}")

    # Prepare for matching
    logger.info("Preparing data for matching...")
    reference = prepare_for_matching(reference)
    target = prepare_for_matching(target)

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="tomtom_benchmark_")
        output_dir = Path(temp_dir)
        cleanup = True

    try:
        # Save to parquet for pipeline
        ref_path = output_dir / "reference.parquet"
        target_path = output_dir / "target.parquet"
        bridge_path = output_dir / "bridge.parquet"

        logger.info(f"Saving intermediate files to {output_dir}")
        reference.to_parquet(ref_path, write_covering_bbox=True)
        target.to_parquet(target_path, write_covering_bbox=True)

        # Run matching pipeline
        logger.info("Running crosswalk pipeline...")
        result = run_pipeline(
            reference_path=ref_path,
            target_path=target_path,
            output_path=bridge_path,
            method="xgboost",
            buffer_distance_m=50.0,
        )

        # Report results
        print("\n" + "=" * 60)
        print("TomTom Self-Matching Benchmark Results")
        print("=" * 60)
        print(f"Region: bbox={bbox.to_tuple()}")
        print(f"Release: {release}")
        print()
        print("Data split:")
        print(f"  Reference (non-TomTom):  {result.n_reference:,}")
        print(f"  Target (TomTom-only):    {result.n_target:,}")
        print()
        print(f"Candidates generated:      {result.n_candidates:,}")
        print()
        print("Results:")
        print(f"  Matched:   {result.n_matched:,}")
        print(f"  Review:    {result.n_review:,}")
        print(f"  Unmatched: {result.n_unmatched:,}")
        print()

        if result.n_target > 0:
            match_rate = result.n_matched / result.n_target * 100
            match_or_review_rate = (result.n_matched + result.n_review) / result.n_target * 100
            print(f"Match rate:           {match_rate:.1f}%")
            print(f"Match+Review rate:    {match_or_review_rate:.1f}%")
        print("=" * 60)

        if args.output_dir:
            print(f"\nOutput files saved to: {output_dir}")
            print(f"  - {ref_path}")
            print(f"  - {target_path}")
            print(f"  - {bridge_path}")

    finally:
        # Cleanup temp dir if we created one
        if cleanup:
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
