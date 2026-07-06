"""Candidate sampling for agent labeling pipeline.

Samples diverse candidates across confidence ranges to create balanced
labeling batches for AI agents.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger

from ..blocking import generate_candidates
from ..matching.types import MatchResult


@dataclass
class SamplingConfig:
    """Configuration for candidate sampling.

    Attributes:
        n_candidates: Total number of candidates to sample
        confidence_buckets: Mapping of bucket names to (min, max) confidence ranges
        bucket_proportions: Mapping of bucket names to sampling proportions
        seed: Random seed for reproducibility
        buffer_distance_m: Search radius for candidate generation (meters)
    """

    n_candidates: int = 100
    # Focus sampling on the uncertain zone (0.1-0.9) where the model struggles
    # and agent labels add the most value. Based on error analysis:
    # - <0.1 confident no-match: 9.2% error (model reliable)
    # - 0.1-0.9 uncertain: 37.1% error (coin-flip zone, most valuable)
    # - >0.9 confident match: 15.3% error (model reliable)
    confidence_buckets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "uncertain": (0.1, 0.9),
        }
    )
    bucket_proportions: dict[str, float] = field(
        default_factory=lambda: {
            "uncertain": 1.0,
        }
    )
    seed: int = 42
    buffer_distance_m: float = 50.0


@dataclass
class SampledCandidate:
    """A sampled candidate with all metadata for labeling.

    Attributes:
        ref_id: Reference segment ID (GERS ID)
        target_id: Target segment ID
        ref_geometry: Reference geometry (WGS84)
        target_geometry: Target geometry (WGS84)
        ref_name: Reference segment name
        target_name: Target segment name
        ref_class: Reference road class
        target_class: Target road class
        ml_confidence: ML model confidence score
        ml_decision: ML model decision (match/review/no_match)
        features: Computed feature values
        dataset: Source dataset name
        confidence_bucket: Which bucket this was sampled from
    """

    ref_id: str
    target_id: str
    ref_geometry: Any  # shapely geometry
    target_geometry: Any
    ref_name: str | None
    target_name: str | None
    ref_class: str | None
    target_class: str | None
    ml_confidence: float
    ml_decision: str
    features: dict[str, float]
    dataset: str
    confidence_bucket: str
    ref_start_frac: float = 0.0
    ref_end_frac: float = 1.0
    target_start_frac: float = 0.0
    target_end_frac: float = 1.0


def load_geodataframe(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame from parquet or other formats."""
    if path.suffix == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)

    # Ensure CRS is set (default to WGS84 if missing)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf


def _project_to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project GeoDataFrame to appropriate UTM zone."""
    if gdf.crs and gdf.crs.is_geographic:
        centroid = gdf.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
        return gdf.to_crs(utm_crs)
    return gdf


def sample_candidates(
    reference_path: Path,
    target_path: Path,
    config: SamplingConfig,
    dataset_name: str | None = None,
    model_path: Path | None = None,
    exclude_labeled: set[tuple[str, str]] | None = None,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
) -> list[SampledCandidate]:
    """Sample diverse candidates across confidence ranges.

    Args:
        reference_path: Path to reference segments (Overture)
        target_path: Path to target segments (local data)
        config: Sampling configuration
        dataset_name: Name of the target dataset (inferred from path if not provided)
        model_path: Path to trained ML model (uses rule-based if not provided)
        exclude_labeled: Set of (ref_id, target_id) pairs to exclude
        ref_id_column: ID column name in reference data
        target_id_column: ID column name in target data
        ref_name_column: Name column in reference data
        target_name_column: Name column in target data
        ref_class_column: Class column in reference data
        target_class_column: Class column in target data

    Returns:
        List of SampledCandidate objects
    """
    # Infer dataset name from path if not provided
    if dataset_name is None:
        dataset_name = target_path.stem

    logger.info(f"Loading data for sampling: {dataset_name}")

    # Load data
    reference = load_geodataframe(reference_path)
    target = load_geodataframe(target_path)

    logger.info(f"Loaded {len(reference)} reference segments, {len(target)} target segments")

    # Project to metric CRS
    reference_proj = _project_to_utm(reference)
    target_proj = _project_to_utm(target)

    # Generate candidates
    candidates = generate_candidates(
        reference=reference_proj,
        target=target_proj,
        buffer_distance_m=config.buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    if not candidates:
        logger.warning("No candidates generated")
        return []

    logger.info(f"Generated {len(candidates)} candidate pairs")

    # Score candidates using ML model
    if not model_path or not model_path.exists():
        raise FileNotFoundError(
            f"ML model not found at {model_path}. Run 'crosswalk train' to train the model."
        )

    scored = _score_with_ml(
        candidates,
        reference_proj,
        target_proj,
        model_path,
        ref_name_column,
        target_name_column,
        ref_class_column,
        target_class_column,
    )

    # Exclude already-labeled pairs
    if exclude_labeled:
        scored = [r for r in scored if (str(r.ref_id), str(r.target_id)) not in exclude_labeled]
        logger.info(f"After excluding labeled: {len(scored)} candidates")

    # Stratified sampling by confidence bucket
    rng = np.random.default_rng(config.seed)
    sampled = _stratified_sample(scored, config, rng)

    # Build SampledCandidate objects with full metadata
    ref_lookup = reference.set_index(ref_id_column)
    target_lookup = target.set_index(target_id_column)

    has_ref_name = ref_name_column in reference.columns
    has_target_name = target_name_column in target.columns
    has_ref_class = ref_class_column in reference.columns
    has_target_class = target_class_column in target.columns

    def get_row(lookup, id_val):
        result = lookup.loc[[id_val]]
        if len(result) == 0:
            raise KeyError(f"ID {id_val} not found")
        return result.iloc[0]

    results = []
    for result, bucket in sampled:
        ref_row = get_row(ref_lookup, result.ref_id)
        target_row = get_row(target_lookup, result.target_id)

        ref_name_raw = ref_row.get(ref_name_column) if has_ref_name else None
        ref_name = ref_name_raw.get("primary") if isinstance(ref_name_raw, dict) else ref_name_raw
        target_name_raw = target_row.get(target_name_column) if has_target_name else None
        target_name = (
            target_name_raw.get("primary") if isinstance(target_name_raw, dict) else target_name_raw
        )
        ref_class = ref_row.get(ref_class_column) if has_ref_class else None
        target_class = target_row.get(target_class_column) if has_target_class else None

        results.append(
            SampledCandidate(
                ref_id=str(result.ref_id),
                target_id=str(result.target_id),
                ref_geometry=ref_row.geometry,
                target_geometry=target_row.geometry,
                ref_name=ref_name,
                target_name=target_name,
                ref_class=ref_class,
                target_class=target_class,
                ml_confidence=result.confidence,
                ml_decision=result.decision.value,
                features=result.features,
                dataset=dataset_name,
                confidence_bucket=bucket,
            )
        )

    logger.info(f"Sampled {len(results)} candidates across confidence buckets")
    return results


def _stratified_sample(
    scored: list,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> list[tuple[Any, str]]:
    """Perform stratified sampling by confidence bucket.

    Returns list of (MatchResult, bucket_name) tuples.
    """
    # Assign each candidate to a bucket
    bucketed: dict[str, list] = {name: [] for name in config.confidence_buckets}

    for result in scored:
        for bucket_name, (min_conf, max_conf) in config.confidence_buckets.items():
            if min_conf <= result.confidence < max_conf or (
                max_conf == 1.0 and result.confidence == 1.0
            ):
                bucketed[bucket_name].append(result)
                break

    # Log bucket sizes
    for bucket_name, items in bucketed.items():
        logger.info(f"Bucket '{bucket_name}': {len(items)} candidates")

    # Sample from each bucket according to proportions
    sampled = []
    for bucket_name, proportion in config.bucket_proportions.items():
        bucket_items = bucketed.get(bucket_name, [])
        n_to_sample = int(config.n_candidates * proportion)

        if len(bucket_items) == 0:
            logger.warning(f"Bucket '{bucket_name}' is empty, skipping")
            continue

        if len(bucket_items) < n_to_sample:
            logger.warning(
                f"Bucket '{bucket_name}' has {len(bucket_items)} items, "
                f"requested {n_to_sample}, using all available"
            )
            selected = bucket_items
        else:
            indices = rng.choice(len(bucket_items), size=n_to_sample, replace=False)
            selected = [bucket_items[i] for i in indices]

        sampled.extend((item, bucket_name) for item in selected)

    return sampled


def _score_with_ml(
    candidates: list,
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    model_path: Path,
    ref_name_column: str,
    target_name_column: str,
    ref_class_column: str,
    target_class_column: str,
) -> list[MatchResult]:
    """Score candidates using ML model.

    Args:
        candidates: List of CandidatePair objects
        reference: Reference GeoDataFrame (projected CRS)
        target: Target GeoDataFrame (projected CRS)
        model_path: Path to trained ML model
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target

    Returns:
        List of MatchResult objects with ML-based scores
    """
    from ..matching.ml import MLMatcher

    # Sampling scores are advisory (candidate triage): stale model warns only.
    matcher = MLMatcher(model_path=str(model_path), allow_version_mismatch=True)
    logger.info(f"Loaded ML model from {model_path}")

    return matcher.score_candidates(
        candidates=candidates,
        reference=reference,
        target=target,
        ref_name_column=ref_name_column,
        target_name_column=target_name_column,
        ref_class_column=ref_class_column,
        target_class_column=target_class_column,
    )
