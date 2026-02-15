"""Label backfill - add LR data to existing labels and recompute features.

This module provides functionality to backfill existing labeled pairs with
linear-referenced attribute data from current Overture data, and recompute
semantic features using correct majority-covering attributes.

Usage:
    from matcher.labeling.backfill import backfill_lr_data, recompute_label_features

    # Add LR data to labels/data/
    backfill_lr_data("my_dataset")

    # Recompute features with LR-aware extraction
    recompute_label_features("my_dataset")
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger

from ..fetch.overture import extract_lr_attributes
from .data_store import DataStore

# Default directory for data store
DEFAULT_DATA_DIR = Path("labels/data")


def backfill_lr_data(
    dataset_id: str,
    overture_path: Path,
    target_path: Path | None = None,
    data_dir: Path | None = None,
    dry_run: bool = False,
    target_is_osm: bool = False,
) -> dict[str, int]:
    """Backfill LR data to existing data store entries.

    Looks up each labeled pair's reference segment by gers_id in the current
    Overture data and adds the LR attributes (names_lr, subclass_lr, etc.)
    to the data store.

    For Overture → OSM datasets (where target_is_osm=True):
    - Reference LR comes from Overture data (looked up by gers_id)
    - Target LR comes from OSM data (looked up by target_id which is an OSM ID)
    - The OSM data has LR-capable structure and will have extract_lr_attributes run

    Args:
        dataset_id: Dataset identifier for the label partition
        overture_path: Path to current Overture segments parquet file
        target_path: Optional path to target data (for target LR attributes)
        data_dir: Directory for data stores (default: labels/data)
        dry_run: If True, don't save changes, just report counts
        target_is_osm: If True, target is OSM data with LR-capable structure.
            Use this for Overture → OSM matching datasets.

    Returns:
        Dict with counts: {"updated": n, "not_found": n, "total": n}
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Auto-detect OSM datasets based on naming convention
    if not target_is_osm and dataset_id.endswith("_osm"):
        logger.info(f"Auto-detected OSM dataset from name: {dataset_id}")
        target_is_osm = True

    logger.info(f"Backfilling LR data for dataset: {dataset_id}")

    # Load data store
    store = DataStore(dataset_id, data_dir=data_dir)
    if len(store.gdf) == 0:
        logger.warning(f"No data found for dataset {dataset_id}")
        return {"updated": 0, "not_found": 0, "total": 0}

    logger.info(f"Found {len(store.gdf)} data entries")

    # Load Overture data with LR attributes
    logger.info(f"Loading Overture data from {overture_path}")
    ref_gdf = gpd.read_parquet(overture_path)

    # Extract LR attributes if not already present
    if "names_lr" not in ref_gdf.columns:
        logger.info("Extracting LR attributes from Overture data...")
        ref_gdf = extract_lr_attributes(ref_gdf)

    # Create ID lookup
    ref_id_col = "id" if "id" in ref_gdf.columns else ref_gdf.columns[0]
    ref_gdf[ref_id_col] = ref_gdf[ref_id_col].astype(str)
    ref_by_id = ref_gdf.set_index(ref_id_col)

    # Load target data if provided
    target_by_id = None
    if target_path is not None and target_path.exists():
        logger.info(f"Loading target data from {target_path}")
        target_gdf = gpd.read_parquet(target_path)
        if "names_lr" not in target_gdf.columns:
            if target_is_osm:
                # OSM data has LR-capable structure (names.rules, etc.)
                logger.info("Extracting LR attributes from OSM target data...")
                target_gdf = extract_lr_attributes(target_gdf)
            else:
                # Regular target data typically doesn't have native LR
                logger.info("Target data doesn't have LR columns - using trivial LR")
        target_id_col = "id" if "id" in target_gdf.columns else target_gdf.columns[0]
        target_gdf[target_id_col] = target_gdf[target_id_col].astype(str)
        target_by_id = target_gdf.set_index(target_id_col)

    # Process each data entry
    updated = 0
    not_found = 0

    for _idx, row in store.gdf.iterrows():
        gers_id = str(row["gers_id"])
        target_id = str(row["target_id"])

        # Get LR data from reference
        ref_names_lr = None
        ref_subclass_lr = None
        ref_level_lr = None
        ref_road_flags_lr = None
        ref_oneway_lr = None
        ref_speed_limit_kph_lr = None

        if gers_id in ref_by_id.index:
            ref_row = ref_by_id.loc[gers_id]
            ref_names_lr = ref_row.get("names_lr")
            ref_subclass_lr = ref_row.get("subclass_lr")
            ref_level_lr = ref_row.get("level_lr")
            ref_road_flags_lr = ref_row.get("road_flags_lr")
            ref_oneway_lr = ref_row.get("oneway_lr")
            ref_speed_limit_kph_lr = ref_row.get("speed_limit_kph_lr")
            ref_names_raw = ref_row.get("names")  # Full Overture names dict
        else:
            not_found += 1
            continue

        # Get LR data from target (if available)
        target_names_lr = None
        target_subclass_lr = None
        target_level_lr = None
        target_road_flags_lr = None
        target_oneway_lr = None
        target_speed_limit_kph_lr = None
        target_names_raw = None

        if target_by_id is not None and target_id in target_by_id.index:
            target_row = target_by_id.loc[target_id]
            target_names_lr = target_row.get("names_lr")
            target_subclass_lr = target_row.get("subclass_lr")
            target_level_lr = target_row.get("level_lr")
            target_road_flags_lr = target_row.get("road_flags_lr")
            target_oneway_lr = target_row.get("oneway_lr")
            target_speed_limit_kph_lr = target_row.get("speed_limit_kph_lr")
            target_names_raw = target_row.get("names")  # Full target names dict

        # Update the data store entry
        if not dry_run:
            success = store.update_lr_attributes(
                gers_id=gers_id,
                target_id=target_id,
                ref_names_lr=ref_names_lr,
                target_names_lr=target_names_lr,
                ref_subclass_lr=ref_subclass_lr,
                target_subclass_lr=target_subclass_lr,
                ref_level_lr=ref_level_lr,
                target_level_lr=target_level_lr,
                ref_road_flags_lr=ref_road_flags_lr,
                target_road_flags_lr=target_road_flags_lr,
                ref_oneway_lr=ref_oneway_lr,
                target_oneway_lr=target_oneway_lr,
                ref_speed_limit_kph_lr=ref_speed_limit_kph_lr,
                target_speed_limit_kph_lr=target_speed_limit_kph_lr,
            )
            # Also persist raw names structs
            ref_dict = ref_names_raw if isinstance(ref_names_raw, dict) else None
            target_dict = target_names_raw if isinstance(target_names_raw, dict) else None
            if ref_dict is not None or target_dict is not None:
                store.update_names_raw(
                    gers_id=gers_id,
                    target_id=target_id,
                    ref_names=ref_dict,
                    target_names=target_dict,
                )
            if success:
                updated += 1
        else:
            updated += 1

    # Save changes
    if not dry_run and updated > 0:
        store.save()
        logger.info(f"Saved {updated} updated data entries")

    result = {
        "updated": updated,
        "not_found": not_found,
        "total": len(store.gdf),
    }

    if dry_run:
        logger.info(f"Dry run complete: would update {updated}, not found {not_found}")
    else:
        logger.info(f"Backfill complete: updated {updated}, not found {not_found}")

    return result


def backfill_all_datasets(
    overture_dir: Path,
    target_dir: Path | None = None,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, int]]:
    """Backfill LR data for all datasets with data entries.

    Args:
        overture_dir: Directory containing Overture parquet files
        target_dir: Optional directory containing target parquet files
        data_dir: Directory for data stores (default: labels/data)
        dry_run: If True, don't save changes, just report counts

    Returns:
        Dict mapping dataset_id to counts
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return {}

    # Find all dataset partitions
    partitions = list(data_dir.glob("dataset=*/data.parquet"))
    if not partitions:
        logger.warning("No data partitions found")
        return {}

    results = {}
    for partition in partitions:
        dataset_id = partition.parent.name.replace("dataset=", "")
        logger.info(f"\n=== Processing {dataset_id} ===")

        # Find Overture file
        overture_path = overture_dir / "overture_segments.parquet"
        if not overture_path.exists():
            logger.warning(f"Overture file not found: {overture_path}")
            continue

        # Find target file (try common naming patterns)
        target_path = None
        if target_dir is not None:
            for pattern in [f"{dataset_id}.parquet", f"{dataset_id}_*.parquet"]:
                matches = list(target_dir.glob(pattern))
                if matches:
                    target_path = matches[0]
                    break

        try:
            results[dataset_id] = backfill_lr_data(
                dataset_id=dataset_id,
                overture_path=overture_path,
                target_path=target_path,
                data_dir=data_dir,
                dry_run=dry_run,
            )
        except Exception as e:
            logger.error(f"Failed to backfill {dataset_id}: {e}")
            results[dataset_id] = {"error": str(e)}

    return results
