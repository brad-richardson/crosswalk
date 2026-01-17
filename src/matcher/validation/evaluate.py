"""Evaluation of validation experiments against ground truth.

Compares matcher results to known dropped record_ids to compute
recall and other metrics.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from loguru import logger


def get_osm_way_id(osm_id: str) -> str:
    """Extract base OSM way ID from various formats.

    Handles:
    - w123456 (plain OSM way ID from fetch)
    - w123456@5 (versioned record_id from Overture)

    Args:
        osm_id: OSM identifier string

    Returns:
        Base way ID (e.g., "w123456")
    """
    if not osm_id:
        return ""

    osm_id = str(osm_id)

    # Strip version suffix if present
    if "@" in osm_id:
        osm_id = osm_id.split("@")[0]

    return osm_id


def evaluate_by_record_id(
    bridge: gpd.GeoDataFrame,
    unmatched: gpd.GeoDataFrame,
    fresh_osm: gpd.GeoDataFrame,
    dropped_record_ids: set[str],
    osm_id_column: str = "id",
) -> pd.DataFrame:
    """Evaluate match results against ground truth.

    For each fresh OSM segment with a dropped record_id:
    - Check if it was matched (appears in bridge) or orphaned (in unmatched)

    Args:
        bridge: Match results from matcher (target_id -> gers_id)
        unmatched: Orphaned target segments
        fresh_osm: Fresh OSM data with way IDs
        dropped_record_ids: Record IDs that were dropped from reference
        osm_id_column: Column name for OSM ID in fresh_osm

    Returns:
        DataFrame with per-segment evaluation:
            osm_id, record_id, should_match, matched, confidence
    """
    # Normalize dropped record_ids (strip version suffixes)
    dropped_ids_normalized = {get_osm_way_id(rid) for rid in dropped_record_ids}

    logger.info(f"Evaluating against {len(dropped_ids_normalized)} dropped record_ids")

    # Get OSM IDs from fresh data
    if osm_id_column not in fresh_osm.columns:
        raise ValueError(f"OSM ID column '{osm_id_column}' not found in fresh_osm")

    # Build evaluation records
    eval_records = []

    for idx, row in fresh_osm.iterrows():
        osm_id = str(row[osm_id_column])
        base_id = get_osm_way_id(osm_id)

        # Should this segment have matched? (was its Overture equivalent dropped?)
        should_match = base_id in dropped_ids_normalized

        # Was it actually matched?
        # Check if osm_id appears in bridge's target_id column
        matched = False
        confidence = None

        if "target_id" in bridge.columns:
            bridge_match = bridge[bridge["target_id"] == osm_id]
            if len(bridge_match) > 0:
                matched = True
                if "confidence" in bridge_match.columns:
                    confidence = bridge_match["confidence"].iloc[0]

        # Also check local_id (alternative column name)
        if not matched and "local_id" in bridge.columns:
            bridge_match = bridge[bridge["local_id"] == osm_id]
            if len(bridge_match) > 0:
                matched = True
                if "confidence" in bridge_match.columns:
                    confidence = bridge_match["confidence"].iloc[0]

        eval_records.append({
            "osm_id": osm_id,
            "record_id": base_id,
            "should_match": should_match,
            "matched": matched,
            "confidence": confidence,
        })

    eval_df = pd.DataFrame(eval_records)

    # Log summary
    should_match_count = eval_df["should_match"].sum()
    matched_of_should = eval_df[eval_df["should_match"]]["matched"].sum()
    logger.info(f"  Fresh OSM segments: {len(eval_df)}")
    logger.info(f"  Should have matched: {should_match_count}")
    logger.info(f"  Actually matched: {matched_of_should}")

    return eval_df


def compute_metrics(eval_df: pd.DataFrame) -> dict:
    """Compute recall and other metrics from evaluation.

    For validation, we primarily care about recall:
    "Of the segments we dropped, how many got matched back?"

    Args:
        eval_df: Evaluation DataFrame from evaluate_by_record_id

    Returns:
        Dictionary with metrics:
            total_dropped, matched_back, orphaned, recall, mean_confidence
    """
    # Filter to segments that should have matched
    should_match_df = eval_df[eval_df["should_match"]]

    total_dropped = len(should_match_df)
    matched_back = should_match_df["matched"].sum()
    orphaned = total_dropped - matched_back

    recall = matched_back / total_dropped if total_dropped > 0 else 0.0

    # Mean confidence of matched segments
    matched_df = should_match_df[should_match_df["matched"]]
    mean_confidence = matched_df["confidence"].mean() if len(matched_df) > 0 else None

    # Also compute metrics for segments that should NOT have matched
    should_not_match_df = eval_df[~eval_df["should_match"]]
    unexpected_matches = should_not_match_df["matched"].sum()

    metrics = {
        "total_dropped": int(total_dropped),
        "matched_back": int(matched_back),
        "orphaned": int(orphaned),
        "recall": float(recall),
        "mean_confidence": float(mean_confidence) if mean_confidence is not None else None,
        # Additional metrics
        "total_fresh_osm": len(eval_df),
        "unexpected_matches": int(unexpected_matches),
    }

    logger.info("Validation Metrics:")
    logger.info(f"  Total dropped: {metrics['total_dropped']}")
    logger.info(f"  Matched back: {metrics['matched_back']}")
    logger.info(f"  Orphaned: {metrics['orphaned']}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    if metrics["mean_confidence"] is not None:
        logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")

    return metrics


def analyze_failures(
    eval_df: pd.DataFrame,
    fresh_osm: gpd.GeoDataFrame,
    osm_id_column: str = "id",
) -> gpd.GeoDataFrame:
    """Analyze segments that should have matched but didn't (false negatives).

    Args:
        eval_df: Evaluation DataFrame
        fresh_osm: Fresh OSM GeoDataFrame
        osm_id_column: Column name for OSM ID

    Returns:
        GeoDataFrame of failed matches with evaluation info
    """
    # Find false negatives
    fn_mask = eval_df["should_match"] & ~eval_df["matched"]
    fn_ids = set(eval_df[fn_mask]["osm_id"])

    # Filter fresh_osm to false negatives
    failures = fresh_osm[fresh_osm[osm_id_column].isin(fn_ids)].copy()

    # Add evaluation columns
    eval_lookup = eval_df.set_index("osm_id")
    failures["should_match"] = True
    failures["matched"] = False

    logger.info(f"Found {len(failures)} false negatives (should have matched but didn't)")

    return failures
