"""Evaluation of validation experiments against ground truth.

Compares crosswalk results to known dropped record_ids to compute
recall and other metrics.
"""

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
    bridge: pd.DataFrame,
    unmatched: pd.DataFrame,
    fresh_osm: gpd.GeoDataFrame,
    dropped_record_ids: set[str],
    osm_id_column: str = "id",
) -> pd.DataFrame:
    """Evaluate match results against ground truth.

    For each fresh OSM segment with a dropped record_id:
    - Check if it was matched (appears in bridge) or orphaned (in unmatched)

    Args:
        bridge: Match results from crosswalk (target_id -> reference_id)
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

    # Build evaluation DataFrame using vectorized operations
    osm_ids = fresh_osm[osm_id_column].astype(str)
    record_ids = osm_ids.map(get_osm_way_id)

    eval_df = pd.DataFrame(
        {
            "osm_id": osm_ids.values,
            "record_id": record_ids.values,
        }
    )

    # Should this segment have matched? (was its Overture equivalent dropped?)
    eval_df["should_match"] = eval_df["record_id"].isin(dropped_ids_normalized)

    # Initialize matched/confidence columns
    eval_df["matched"] = False
    eval_df["confidence"] = None

    # Build sets of matched IDs for O(1) lookup
    has_confidence = "confidence" in bridge.columns

    # Check target_id matches
    if "target_id" in bridge.columns:
        target_matches = set(bridge["target_id"].astype(str))
        target_matched_mask = eval_df["osm_id"].isin(target_matches)
        eval_df.loc[target_matched_mask, "matched"] = True

        if has_confidence:
            # Merge to get confidence for target_id matches
            bridge_target = bridge[["target_id", "confidence"]].copy()
            bridge_target["target_id"] = bridge_target["target_id"].astype(str)
            bridge_target = bridge_target.drop_duplicates(subset=["target_id"], keep="first")
            eval_df = eval_df.merge(
                bridge_target.rename(columns={"target_id": "osm_id", "confidence": "conf_target"}),
                on="osm_id",
                how="left",
            )
            eval_df["confidence"] = eval_df["conf_target"].combine_first(eval_df["confidence"])
            eval_df = eval_df.drop(columns=["conf_target"])

    # Check local_id matches (for segments not matched via target_id)
    if "local_id" in bridge.columns:
        local_matches = set(bridge["local_id"].astype(str))
        # Only mark as matched if not already matched
        local_matched_mask = eval_df["osm_id"].isin(local_matches) & ~eval_df["matched"]
        eval_df.loc[local_matched_mask, "matched"] = True

        if has_confidence:
            # Merge to get confidence for local_id matches (only where not already set)
            bridge_local = bridge[["local_id", "confidence"]].copy()
            bridge_local["local_id"] = bridge_local["local_id"].astype(str)
            bridge_local = bridge_local.drop_duplicates(subset=["local_id"], keep="first")
            eval_df = eval_df.merge(
                bridge_local.rename(columns={"local_id": "osm_id", "confidence": "conf_local"}),
                on="osm_id",
                how="left",
            )
            eval_df["confidence"] = eval_df["confidence"].combine_first(eval_df["conf_local"])
            eval_df = eval_df.drop(columns=["conf_local"])

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
    failures["should_match"] = True
    failures["matched"] = False

    logger.info(f"Found {len(failures)} false negatives (should have matched but didn't)")

    return failures
