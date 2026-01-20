"""Bridge file generation linking local IDs to GERS.

The bridge file is the primary output of the conflation pipeline,
providing a mapping between local dataset IDs and Overture GERS IDs.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ..matching.rules import MatchDecision, MatchResult

# Bridge file schema
BRIDGE_SCHEMA = pa.schema(
    [
        ("local_id", pa.string()),
        ("gers_id", pa.string()),
        ("confidence", pa.float64()),
        ("match_type", pa.string()),  # "1:1", "1:N", "N:1"
        ("match_method", pa.string()),  # "rule", "xgboost", "gnn"
        ("match_decision", pa.string()),  # "match", "review", "no_match"
        ("matched_at", pa.timestamp("us", tz="UTC")),
        ("pipeline_version", pa.string()),
        # Linear reference fields from alignment (0-1 fractions)
        # These indicate where on each geometry the match alignment starts/ends
        ("gers_start_frac", pa.float64()),  # Where match starts on GERS segment
        ("gers_end_frac", pa.float64()),  # Where match ends on GERS segment
        ("local_start_frac", pa.float64()),  # Where match starts on local segment
        ("local_end_frac", pa.float64()),  # Where match ends on local segment
    ]
)


def generate_bridge_file(
    matches: list[MatchResult],
    output_path: Path,
    pipeline_version: str = "0.1.0",
    match_method: str = "rule",
) -> Path:
    """Generate bridge file from match results.

    Args:
        matches: List of MatchResult objects
        output_path: Path for output Parquet file
        pipeline_version: Version string for provenance
        match_method: Method used for matching

    Returns:
        Path to generated file
    """
    logger.info(f"Generating bridge file with {len(matches)} matches...")

    now = datetime.now(UTC)

    records = []
    for match in matches:
        # Only include matches and reviews (not no_match)
        if match.decision == MatchDecision.NO_MATCH:
            continue

        records.append(
            {
                "local_id": str(match.target_id),
                "gers_id": str(match.ref_id),
                "confidence": float(match.confidence),
                "match_type": match.features.get("match_type", "1:1"),
                "match_method": match_method,
                "match_decision": match.decision.value,
                "matched_at": now,
                "pipeline_version": pipeline_version,
                # Linear reference fields (may be None if alignment not computed)
                "gers_start_frac": match.gers_start_frac,
                "gers_end_frac": match.gers_end_frac,
                "local_start_frac": match.local_start_frac,
                "local_end_frac": match.local_end_frac,
            }
        )

    if not records:
        logger.warning("No matches to write to bridge file")
        # Create empty file with schema
        table = pa.Table.from_pylist([], schema=BRIDGE_SCHEMA)
    else:
        table = pa.Table.from_pylist(records, schema=BRIDGE_SCHEMA)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write parquet file
    pq.write_table(table, output_path)

    logger.info(f"Saved bridge file to {output_path}")
    logger.info(f"  Total matches: {len(records)}")

    return output_path


def generate_unmatched_report(
    target: gpd.GeoDataFrame,
    matched_ids: set,
    output_path: Path,
    id_column: str = "local_id",
) -> Path:
    """Generate report of unmatched target features.

    Args:
        target: Target GeoDataFrame
        matched_ids: Set of matched target IDs
        output_path: Path for output file
        id_column: Column name for target IDs

    Returns:
        Path to generated file
    """
    logger.info("Generating unmatched report...")

    # Find unmatched features
    if id_column in target.columns:
        unmatched_mask = ~target[id_column].isin(matched_ids)
    else:
        unmatched_mask = ~target.index.isin(matched_ids)

    unmatched = target[unmatched_mask].copy()

    logger.info(f"  {len(unmatched)} unmatched features")

    # Add reason column
    unmatched["unmatched_reason"] = "no_match_found"

    # Select relevant columns
    columns_to_keep = ["geometry"]
    for col in ["local_id", "name", "road_class", "unmatched_reason"]:
        if col in unmatched.columns:
            columns_to_keep.append(col)

    unmatched = unmatched[columns_to_keep]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to parquet
    unmatched.to_parquet(output_path)

    logger.info(f"Saved unmatched report to {output_path}")

    return output_path


def generate_review_file(
    matches: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    output_path: Path,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> Path:
    """Generate file with matches needing review.

    Includes geometry and features for manual inspection.

    Args:
        matches: List of MatchResult objects
        reference: Reference GeoDataFrame
        target: Target GeoDataFrame
        output_path: Path for output file
        ref_id_column: ID column in reference
        target_id_column: ID column in target

    Returns:
        Path to generated file
    """
    logger.info("Generating review file...")

    # Filter to review decisions only
    review_matches = [m for m in matches if m.decision == MatchDecision.REVIEW]

    if not review_matches:
        logger.info("  No matches need review")
        # Create empty GeoDataFrame
        review_gdf = gpd.GeoDataFrame()
        review_gdf.to_parquet(output_path)
        return output_path

    logger.info(f"  {len(review_matches)} matches need review")

    # Build review records with geometry
    records = []

    # Create lookup for geometries
    ref_lookup = {}
    for idx, row in reference.iterrows():
        rid = row.get(ref_id_column, idx)
        ref_lookup[rid] = row.geometry

    target_lookup = {}
    for idx, row in target.iterrows():
        tid = row.get(target_id_column, idx)
        target_lookup[tid] = row.geometry

    for match in review_matches:
        ref_geom = ref_lookup.get(match.ref_id)
        target_geom = target_lookup.get(match.target_id)

        records.append(
            {
                "local_id": str(match.target_id),
                "gers_id": str(match.ref_id),
                "confidence": match.confidence,
                "ref_geometry": ref_geom,
                "target_geometry": target_geom,
                "hausdorff_distance": match.features.get("hausdorff_distance"),
                "name_similarity": match.features.get("name_token_sort"),
            }
        )

    # Create GeoDataFrame with target geometry as primary
    review_gdf = gpd.GeoDataFrame(
        records,
        geometry="target_geometry",
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    review_gdf.to_parquet(output_path)

    logger.info(f"Saved review file to {output_path}")

    return output_path


def load_bridge_file(path: Path) -> list[dict[str, Any]]:
    """Load bridge file from parquet.

    Args:
        path: Path to bridge file

    Returns:
        List of match records
    """
    import pandas as pd

    df = pd.read_parquet(path)
    return df.to_dict("records")


def merge_bridge_files(
    paths: list[Path],
    output_path: Path,
    dedup_strategy: str = "highest_confidence",
) -> Path:
    """Merge multiple bridge files into one.

    Args:
        paths: List of bridge file paths
        output_path: Output path for merged file
        dedup_strategy: How to handle duplicates:
            - "highest_confidence": Keep match with highest confidence
            - "first": Keep first occurrence
            - "all": Keep all (may have duplicate local_ids)

    Returns:
        Path to merged file
    """
    import pandas as pd

    logger.info(f"Merging {len(paths)} bridge files...")

    dfs = []
    for path in paths:
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        logger.warning("No bridge files to merge")
        return output_path

    merged = pd.concat(dfs, ignore_index=True)

    if dedup_strategy == "highest_confidence":
        # Keep highest confidence match per local_id
        merged = merged.sort_values("confidence", ascending=False)
        merged = merged.drop_duplicates(subset=["local_id"], keep="first")
    elif dedup_strategy == "first":
        merged = merged.drop_duplicates(subset=["local_id"], keep="first")
    # "all" strategy keeps all rows

    # Write merged file
    table = pa.Table.from_pandas(merged, schema=BRIDGE_SCHEMA)
    pq.write_table(table, output_path)

    logger.info(f"Saved merged bridge file to {output_path}")
    logger.info(f"  Total matches: {len(merged)}")

    return output_path
