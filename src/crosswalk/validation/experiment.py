"""Validation experiment orchestration.

Runs full validation experiments: create holdout, fetch fresh OSM,
run crosswalk, evaluate results.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..fetch.osm import fetch_osm_data
from ..fetch.overture import BoundingBox
from ..pipeline import run_pipeline
from .evaluate import analyze_failures, compute_metrics, evaluate_by_record_id
from .holdout import (
    drop_by_bbox,
    drop_by_class,
    drop_by_source,
    drop_random_osm,
)


@dataclass
class ExperimentConfig:
    """Configuration for a validation experiment."""

    overture_path: str
    output_dir: str
    bbox: tuple[float, float, float, float]
    strategy: str  # "random", "bbox", "source", "class"
    matcher_method: str = "rule"
    # Strategy-specific parameters
    fraction: float = 0.1  # For "random" strategy
    drop_bbox: tuple[float, float, float, float] | None = None  # For "bbox" strategy
    source_dataset: str = "OpenStreetMap"  # For "source" strategy
    road_class: str = "residential"  # For "class" strategy
    seed: int = 42
    fast_mode: bool = False  # Only match dropped segments

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Result of a validation experiment."""

    config: ExperimentConfig
    metrics: dict
    n_overture: int
    n_dropped: int
    n_fresh_osm: int
    n_matched: int
    n_unmatched: int


def run_validation_experiment(
    overture_path: Path,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    strategy: str = "random",
    matcher_method: str = "rule",
    fraction: float = 0.1,
    drop_bbox: tuple[float, float, float, float] | None = None,
    source_dataset: str = "OpenStreetMap",
    road_class: str = "residential",
    seed: int = 42,
    fast_mode: bool = False,
) -> ExperimentResult:
    """Run a full validation experiment.

    Steps:
    1. Load Overture segments
    2. Create holdout (reduced reference + dropped record_ids)
    3. Fetch fresh OSM for bbox
    4. Run crosswalk (fresh OSM as target, reduced Overture as reference)
    5. Evaluate by record_id
    6. Save results and metrics

    Args:
        overture_path: Path to full Overture segments parquet
        output_dir: Directory for experiment outputs
        bbox: Bounding box for fresh OSM fetch (xmin, ymin, xmax, ymax)
        strategy: Drop strategy ("random", "bbox", "source", "class")
        matcher_method: Matching method ("rule" or "xgboost")
        fraction: Fraction to drop for "random" strategy
        drop_bbox: Bounding box for "bbox" strategy (defaults to main bbox)
        source_dataset: Dataset to drop for "source" strategy
        road_class: Road class to drop for "class" strategy
        seed: Random seed for reproducibility
        fast_mode: If True, only match dropped segments instead of the full set

    Returns:
        ExperimentResult with metrics and statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = ExperimentConfig(
        overture_path=str(overture_path),
        output_dir=str(output_dir),
        bbox=bbox,
        strategy=strategy,
        matcher_method=matcher_method,
        fraction=fraction,
        drop_bbox=drop_bbox,
        source_dataset=source_dataset,
        road_class=road_class,
        seed=seed,
        fast_mode=fast_mode,
    )

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info("=" * 60)
    logger.info("VALIDATION EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Output: {output_dir}")

    # Step 1: Load Overture
    logger.info("Step 1: Loading Overture segments...")
    overture = gpd.read_parquet(overture_path)
    logger.info(f"  Loaded {len(overture)} Overture segments")

    # Step 2: Create holdout
    logger.info(f"Step 2: Creating holdout with strategy '{strategy}'...")

    if strategy == "random":
        reduced_ref, dropped_ids = drop_random_osm(overture, fraction=fraction, seed=seed)
    elif strategy == "bbox":
        target_bbox = drop_bbox or bbox
        reduced_ref, dropped_ids = drop_by_bbox(overture, bbox=target_bbox)
    elif strategy == "source":
        reduced_ref, dropped_ids = drop_by_source(overture, source_dataset=source_dataset)
    elif strategy == "class":
        reduced_ref, dropped_ids = drop_by_class(
            overture, road_class=road_class, source_dataset="OpenStreetMap"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_dropped = len(overture) - len(reduced_ref)
    logger.info(f"  Dropped {n_dropped} segments, {len(dropped_ids)} record_ids")

    # Save dropped record_ids
    dropped_ids_path = output_dir / "dropped_record_ids.json"
    with open(dropped_ids_path, "w") as f:
        json.dump(list(dropped_ids), f)

    # Save reduced reference
    reduced_ref_path = output_dir / "reduced_reference.parquet"
    reduced_ref.to_parquet(reduced_ref_path)
    logger.info(f"  Saved reduced reference to {reduced_ref_path}")

    # Step 3: Fetch fresh OSM
    logger.info("Step 3: Fetching fresh OSM data...")
    bbox_obj = BoundingBox(xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3])

    osm_segments_path, _ = fetch_osm_data(
        bbox=bbox_obj,
        output_dir=output_dir,
    )

    fresh_osm_all = gpd.read_parquet(osm_segments_path)
    logger.info(f"  Fetched {len(fresh_osm_all)} fresh OSM segments")

    # Identify which OSM segments correspond to dropped record_ids
    # Pre-compute normalized dropped IDs once for efficiency
    normalized_dropped = {rid.split("@")[0] if "@" in str(rid) else str(rid) for rid in dropped_ids}

    def matches_dropped_id(osm_id):
        base_id = osm_id.split("@")[0] if "@" in str(osm_id) else str(osm_id)
        return base_id in normalized_dropped

    should_match_mask = fresh_osm_all["id"].apply(matches_dropped_id)
    n_should_match = should_match_mask.sum()
    logger.info(f"  Of these, {n_should_match} correspond to dropped record_ids")

    if fast_mode:
        # Fast mode: only match segments that should match
        fresh_osm = fresh_osm_all[should_match_mask].copy()
        logger.info(f"  FAST MODE: Using only {len(fresh_osm)} segments that should match")

        if len(fresh_osm) == 0:
            logger.warning("No OSM segments match dropped record_ids - cannot run validation")
            raise ValueError(
                "No OSM segments correspond to dropped record_ids. "
                "This may indicate a mismatch between Overture provenance and current OSM data."
            )
    else:
        fresh_osm = fresh_osm_all

    # Save the OSM data used for matching
    fresh_osm_path = output_dir / "fresh_osm.parquet"
    fresh_osm.to_parquet(fresh_osm_path)

    # Also save all OSM for reference
    if fast_mode:
        fresh_osm_all_path = output_dir / "fresh_osm_all.parquet"
        fresh_osm_all.to_parquet(fresh_osm_all_path)

    # Step 4: Run crosswalk
    logger.info("Step 4: Running matcher...")
    bridge_path = output_dir / "bridge.parquet"

    pipeline_result = run_pipeline(
        reference_path=reduced_ref_path,
        target_path=fresh_osm_path,
        output_path=bridge_path,
        method=matcher_method,
        target_id_column="id",  # Fresh OSM uses "id" column
        target_class_column="class",  # Fresh OSM uses "class" not "road_class"
    )

    logger.info(f"  Matched: {pipeline_result.n_matched}")
    logger.info(f"  Unmatched: {pipeline_result.n_unmatched}")

    # Load results
    bridge = pd.read_parquet(bridge_path)
    unmatched_path = output_dir / "unmatched.parquet"
    unmatched = pd.read_parquet(unmatched_path) if unmatched_path.exists() else pd.DataFrame()

    # Step 5: Evaluate
    logger.info("Step 5: Evaluating results...")

    eval_df = evaluate_by_record_id(
        bridge=bridge,
        unmatched=unmatched,
        fresh_osm=fresh_osm,
        dropped_record_ids=dropped_ids,
        osm_id_column="id",
    )

    # Save evaluation
    eval_path = output_dir / "evaluation.parquet"
    eval_df.to_parquet(eval_path)

    # Compute metrics
    metrics = compute_metrics(eval_df)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 6: Analyze failures
    logger.info("Step 6: Analyzing failures...")
    failures = analyze_failures(eval_df, fresh_osm, osm_id_column="id")
    if len(failures) > 0:
        failures_path = output_dir / "failures.parquet"
        failures.to_parquet(failures_path)
        logger.info(f"  Saved {len(failures)} failure cases to {failures_path}")

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  Output: {output_dir}")

    return ExperimentResult(
        config=config,
        metrics=metrics,
        n_overture=len(overture),
        n_dropped=n_dropped,
        n_fresh_osm=len(fresh_osm),
        n_matched=pipeline_result.n_matched,
        n_unmatched=pipeline_result.n_unmatched,
    )


def compare_experiments(
    experiment_dirs: list[Path],
) -> pd.DataFrame:
    """Compare metrics across multiple experiments.

    Args:
        experiment_dirs: List of experiment output directories

    Returns:
        DataFrame comparing metrics across experiments
    """
    records = []
    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)

        # Load config
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            logger.warning(f"No config found in {exp_dir}")
            continue

        with open(config_path) as f:
            config = json.load(f)

        # Load metrics
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"No metrics found in {exp_dir}")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        records.append(
            {
                "experiment": exp_dir.name,
                "strategy": config.get("strategy"),
                "matcher_method": config.get("matcher_method"),
                **metrics,
            }
        )

    return pd.DataFrame(records)
