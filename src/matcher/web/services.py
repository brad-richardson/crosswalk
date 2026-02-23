"""Service layer for the matcher web UI.

Provides business logic functions for dataset loading, candidate management,
label recording, configuration access, and integration QA.
"""

import csv
import json
import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ..config import FEATURE_COLUMNS, FEATURE_VERSION, settings
from ..datasets.loader import DatasetLoader
from ..datasets.schema import list_dataset_configs
from ..features.semantic import display_name
from ..filenames import (
    LABELING_CACHE_DIR,
    find_overture_segments,
    find_target_file,
    integration_cache_dir,
    stitch_batch_path,
)
from ..integration_qa.decision_store import MergedDecisionStore, OrphanDecisionStore
from ..labeling.data_loader import (
    CandidatePairView,
    _compute_score_breakdown_from_features,
    build_views_from_feature_df,
    filter_candidates,
    generate_scored_candidates_with_cache,
    get_cache_info,
    get_cached_matcher,
    get_feature_cache_info,
    get_feature_cache_path,
    load_feature_cache,
    load_geodataframe,
)
from ..labeling.data_store import DataStore
from ..labeling.feature_store import FeatureStore
from ..labeling.label_store import LabelStore, get_data_version

logger = logging.getLogger(__name__)

# Project root: src/matcher/web/services.py -> project root is 3 levels up
PROJECT_ROOT = Path(__file__).parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_FILE = Path.home() / ".matcher_labeler_config.json"

# Shared loading state — prevents duplicate background work across routes
# (e.g., labeling and batch both trying to load the same dataset)
loading_lock = threading.Lock()
loading_tasks: dict[str, threading.Thread] = {}
loading_errors: dict[str, str] = {}


def list_datasets() -> list[str]:
    """List available dataset IDs.

    Returns:
        Sorted list of dataset identifiers that have both
        reference and target files on disk.
    """
    loader = DatasetLoader(DATA_DIR)
    return loader.list_available()


def is_dataset_cached(dataset_id: str) -> bool:
    """Check if feature cache exists for this dataset (file existence only)."""
    return get_feature_cache_path(dataset_id).exists()


def load_candidates(dataset_id: str) -> list[CandidatePairView]:
    """Load and score candidate pairs for a dataset.

    Uses the two-stage caching flow: feature cache for expensive computation,
    then ML scoring on top.

    Args:
        dataset_id: Dataset identifier (e.g., "us_boston_streets")

    Returns:
        List of CandidatePairView objects sorted by confidence (review first).
    """
    loader = DatasetLoader(DATA_DIR)
    ref_path = loader.find_reference_path(dataset_id)
    target_path = loader.find_target_path(dataset_id)

    if ref_path is None or target_path is None:
        return []

    reference = load_geodataframe(ref_path)
    target = load_geodataframe(target_path)

    return generate_scored_candidates_with_cache(
        reference=reference,
        target=target,
        dataset_id=dataset_id,
    )


def get_unlabeled_candidates(
    dataset_id: str,
    candidates: list[CandidatePairView],
) -> list[CandidatePairView]:
    """Filter candidates to only unlabeled pairs.

    Args:
        dataset_id: Dataset identifier
        candidates: Full list of candidates

    Returns:
        Candidates that have not yet been labeled.
    """
    store = LabelStore(dataset_id)
    labeled_pairs = store.get_labeled_pairs()
    return filter_candidates(candidates, labeled_pairs=labeled_pairs, show_labeled=False)


def get_labeler_name() -> str:
    """Read the labeler name from the config file.

    Returns:
        Labeler name string, or "unknown" if not configured.
    """
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
            return config.get("labeler_name", "unknown")
        except (json.JSONDecodeError, OSError):
            pass
    return "unknown"


def get_session_id() -> str:
    """Generate a short session ID for labeling.

    Returns:
        8-character UUID string.
    """
    return str(uuid.uuid4())[:8]


def record_label(
    dataset_id: str,
    pair: CandidatePairView,
    label: str,
) -> None:
    """Record a label for a candidate pair.

    Writes to the LabelStore with all fields from the CandidatePairView.

    Args:
        dataset_id: Dataset identifier
        pair: The candidate pair being labeled
        label: Label value ("match", "no_match", "unsure")
    """
    store = LabelStore(dataset_id)
    labeler = get_labeler_name()
    session_id = get_session_id()

    # Compute data versions from source files
    loader = DatasetLoader(DATA_DIR)
    ref_path = loader.find_reference_path(dataset_id)
    target_path = loader.find_target_path(dataset_id)
    ref_data_version = get_data_version(ref_path) if ref_path else None
    target_data_version = get_data_version(target_path) if target_path else None

    store.add(
        gers_id=pair.ref_id,
        target_id=pair.target_id,
        label=label,
        labeler=labeler,
        session_id=session_id,
        original_decision=pair.decision,
        original_confidence=pair.confidence,
        features=pair.features,
        ref_start_pct=pair.ref_start_frac,
        ref_end_pct=pair.ref_end_frac,
        target_start_pct=pair.target_start_frac,
        target_end_pct=pair.target_end_frac,
        ref_data_version=ref_data_version,
        target_data_version=target_data_version,
        ref_geometry=pair.ref_geometry,
        target_geometry=pair.target_geometry,
        ref_class_raw=pair.ref_class,
        target_class_raw=pair.target_class,
        ref_subclass=pair.ref_subclass,
        target_subclass=pair.target_subclass,
        ref_names=pair.ref_names_raw,
        target_names=pair.target_names_raw,
        ref_topology=pair.ref_topology,
        target_topology=pair.target_topology,
    )


def get_labels_for_review(
    dataset_id: str,
    filter_type: str | None = None,
    page: int = 0,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    """Get paginated labels for review.

    Args:
        dataset_id: Dataset identifier
        filter_type: Filter by label type (match, no_match, unsure) or None/all for all
        page: Zero-based page number
        page_size: Number of labels per page

    Returns:
        Tuple of (list of label dicts, total count after filtering).
    """
    store = LabelStore(dataset_id)
    df = store.df
    if df.empty:
        return [], 0
    if filter_type and filter_type != "all":
        df = df[df["label"] == filter_type]
    total = len(df)
    if "labeled_at" in df.columns:
        df = df.sort_values("labeled_at", ascending=False)
    start = page * page_size
    page_df = df.iloc[start : start + page_size]
    return page_df.to_dict("records"), total


def update_review_label(dataset_id: str, gers_id: str, target_id: str, new_label: str) -> bool:
    """Update an existing label's value.

    Args:
        dataset_id: Dataset identifier
        gers_id: Overture reference segment ID
        target_id: Target segment ID
        new_label: New label value (match, no_match, unsure)

    Returns:
        True if found and updated, False if pair not found.
    """
    store = LabelStore(dataset_id)
    labeler = get_labeler_name()
    return store.update_label(gers_id, target_id, new_label, labeler)


def delete_review_label(dataset_id: str, gers_id: str, target_id: str) -> bool:
    """Delete an existing label.

    Args:
        dataset_id: Dataset identifier
        gers_id: Overture reference segment ID
        target_id: Target segment ID

    Returns:
        True if found and deleted, False if pair not found.
    """
    store = LabelStore(dataset_id)
    return store.delete_label(gers_id, target_id)


def undo_last_label(dataset_id: str) -> dict | None:
    """Undo the last label for a dataset.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Dict of the removed label row, or None if nothing to undo.
    """
    store = LabelStore(dataset_id)
    return store.remove_last()


# --- Integration QA service functions ---

EDGE_FILES = [
    "edges",
    "net_new",
    "disconnected",
    "filtered",
    "bridges",
]


def load_qa_edges(dataset_id: str) -> dict[str, gpd.GeoDataFrame | None]:
    """Load integration edges for QA review.

    Uses integration_cache_dir(dataset_id) to find parquet files.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Dict with keys: edges, net_new_edges, disconnected_edges,
        filtered_edges, bridge_edges. Each value is a GeoDataFrame
        or None if the file doesn't exist.
    """
    cache_dir = integration_cache_dir(dataset_id)
    result: dict[str, gpd.GeoDataFrame | None] = {}

    for name in EDGE_FILES:
        path = cache_dir / f"{name}.parquet"
        if path.exists():
            try:
                result[name] = gpd.read_parquet(path)
            except Exception:
                logger.exception("Failed to load %s for dataset %s", path, dataset_id)
                result[name] = None
        else:
            result[name] = None

    return result


def record_qa_decision(
    edge_id: int,
    original_id: str,
    dataset_id: str,
    edge_type: str,
    decision: str,
    reason: str,
    note: str = "",
    **kwargs,
) -> None:
    """Record a QA accept/reject decision.

    Args:
        edge_id: Edge identifier
        original_id: Original edge ID from source dataset
        dataset_id: Dataset identifier
        edge_type: Either "orphan" or "merged"
        decision: Decision value ("correct" or "incorrect")
        reason: Reason for the decision
        note: Optional reviewer note (currently stored as part of reason)
        **kwargs: Additional fields passed to the decision store
    """
    reviewer = get_labeler_name()
    session_id = get_session_id()

    full_reason = f"{reason}: {note}" if note else reason

    if edge_type == "orphan":
        store = OrphanDecisionStore()
        store.add_decision(
            edge_id=edge_id,
            original_id=original_id,
            dataset_id=dataset_id,
            component_id=kwargs.get("component_id", 0),
            decision=decision,
            reason=full_reason,
            reviewer=reviewer,
            session_id=session_id,
            length_m=kwargs.get("length_m", 0.0),
            road_class=kwargs.get("road_class", ""),
            nearest_main_dist_m=kwargs.get("nearest_main_dist_m", 0.0),
            component_size=kwargs.get("component_size", 0),
        )
    else:
        store = MergedDecisionStore()
        store.add_decision(
            edge_id=edge_id,
            original_id=original_id,
            dataset_id=dataset_id,
            source_type=kwargs.get("source_type", ""),
            match_ref_id=kwargs.get("match_ref_id"),
            decision=decision,
            reason=full_reason,
            reviewer=reviewer,
            session_id=session_id,
            match_confidence=kwargs.get("match_confidence", 0.0),
            length_m=kwargs.get("length_m", 0.0),
            road_class=kwargs.get("road_class", ""),
        )


# --- Dashboard service functions ---

CV_RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "ml_cv_results.csv"


def _read_cv_results() -> list[dict]:
    """Read CV results CSV and return list of row dicts."""
    if not CV_RESULTS_PATH.exists():
        return []
    with open(CV_RESULTS_PATH, newline="") as f:
        return list(csv.DictReader(f))


def get_overall_metrics() -> dict | None:
    """Latest overall CV metrics from benchmarks/ml_cv_results.csv."""
    rows = _read_cv_results()
    overall_rows = [r for r in rows if r.get("dataset") == "overall"]
    if not overall_rows:
        return None
    latest = overall_rows[-1]
    return {
        "f1_mean": float(latest["f1_mean"]),
        "f1_std": float(latest["f1_std"]),
        "accuracy_mean": float(latest["accuracy_mean"]),
        "precision_mean": float(latest["precision_mean"]),
        "recall_mean": float(latest["recall_mean"]),
        "n_samples": int(latest["n_samples"]),
        "run_date": latest["run_date"],
    }


def get_dataset_metrics() -> list[dict]:
    """Per-dataset metrics combining labels + CV results + cache status.

    Includes ALL configured datasets (from YAML configs), not just those
    with fetched data. Each entry indicates whether raw data, feature cache,
    and candidate cache are available.
    """
    available = set(list_datasets())
    all_configs = set(list_dataset_configs())
    all_dataset_ids = sorted(available | all_configs)

    rows = _read_cv_results()

    # Build lookup: dataset -> latest CV row
    cv_latest: dict[str, dict] = {}
    for r in rows:
        ds = r.get("dataset", "")
        if ds and ds != "overall":
            cv_latest[ds] = r

    result = []
    for ds in all_dataset_ids:
        has_data = ds in available

        # Check for reference and target files individually
        has_reference = find_overture_segments(DATA_DIR, ds) is not None
        has_target = find_target_file(DATA_DIR, ds) is not None

        # Labels (only check if data exists, since LabelStore may still have labels)
        store = LabelStore(ds)
        df = store.df
        label_count = len(df)
        label_dist = {"match": 0, "no_match": 0, "unsure": 0}
        if not df.empty and "label" in df.columns:
            counts = df["label"].value_counts().to_dict()
            for k in label_dist:
                label_dist[k] = int(counts.get(k, 0))

        cv = cv_latest.get(ds)
        f1 = float(cv["f1_mean"]) if cv else None
        accuracy = float(cv["accuracy_mean"]) if cv else None
        n_samples = int(cv["n_samples"]) if cv else None

        feature_info = get_feature_cache_info(ds)
        candidate_info = get_cache_info(ds)

        result.append(
            {
                "dataset_id": ds,
                "has_data": has_data,
                "has_reference": has_reference,
                "has_target": has_target,
                "label_count": label_count,
                "label_dist": label_dist,
                "f1": f1,
                "accuracy": accuracy,
                "n_samples": n_samples,
                "cache_exists": feature_info["exists"],
                "cache_age_hours": feature_info.get("age_hours"),
                "cache_is_fresh": feature_info.get("is_fresh"),
                "candidates_cached": candidate_info["exists"],
            }
        )

    result.sort(key=lambda x: x["dataset_id"])
    return result


def get_cv_trends() -> dict:
    """CV eval time series for Chart.js."""
    rows = _read_cv_results()
    overall_rows = [r for r in rows if r.get("dataset") == "overall"]
    dates = []
    f1_values = []
    accuracy_values = []
    for r in overall_rows:
        date_str = r["run_date"]
        # Truncate to date portion for display
        if "T" in date_str:
            date_str = date_str.split("T")[0]
        dates.append(date_str)
        f1_values.append(float(r["f1_mean"]))
        accuracy_values.append(float(r["accuracy_mean"]))
    return {"dates": dates, "f1": f1_values, "accuracy": accuracy_values}


def get_dataset_detail(dataset_id: str) -> dict | None:
    """Detailed info for one dataset."""
    datasets = list_datasets()
    if dataset_id not in datasets:
        return None

    store = LabelStore(dataset_id)
    df = store.df
    label_count = len(df)
    label_dist = {"match": 0, "no_match": 0, "unsure": 0}
    if not df.empty and "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        for k in label_dist:
            label_dist[k] = int(counts.get(k, 0))

    # CV metrics
    rows = _read_cv_results()
    cv_rows = [r for r in rows if r.get("dataset") == dataset_id]
    cv = cv_rows[-1] if cv_rows else None

    cache_info = get_feature_cache_info(dataset_id)

    return {
        "dataset_id": dataset_id,
        "label_count": label_count,
        "label_dist": label_dist,
        "f1": float(cv["f1_mean"]) if cv else None,
        "f1_std": float(cv["f1_std"]) if cv else None,
        "accuracy": float(cv["accuracy_mean"]) if cv else None,
        "precision": float(cv["precision_mean"]) if cv else None,
        "recall": float(cv["recall_mean"]) if cv else None,
        "n_samples": int(cv["n_samples"]) if cv else None,
        "run_date": cv["run_date"] if cv else None,
        "cache_exists": cache_info["exists"],
        "cache_age_hours": cache_info.get("age_hours"),
        "cache_is_fresh": cache_info.get("is_fresh"),
        "cache_version": cache_info.get("version"),
        "cache_candidate_count": cache_info.get("candidate_count"),
    }


# --- Context tile cache ---

CONTEXT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tiles"


def _context_cache_path(dataset_id: str) -> Path:
    """Return path to the lightweight context cache parquet for a dataset."""
    return CONTEXT_CACHE_DIR / f"{dataset_id}_context.parquet"


def build_context_cache(dataset_id: str, target_gdf: gpd.GeoDataFrame) -> Path:
    """Build a lightweight context cache parquet for MVT tile serving.

    Extracts id, name, class, and simplified geometry from the target GeoDataFrame.
    Only builds if the cache doesn't already exist.

    Args:
        dataset_id: Dataset identifier
        target_gdf: Target GeoDataFrame with full geometries

    Returns:
        Path to the context cache parquet file
    """
    cache_path = _context_cache_path(dataset_id)
    if cache_path.exists():
        return cache_path

    logger.info(f"Building context cache for {dataset_id} ({len(target_gdf)} features)")

    # Build lightweight dataframe
    cols = {}
    if "id" in target_gdf.columns:
        cols["id"] = target_gdf["id"].values
    else:
        cols["id"] = range(len(target_gdf))

    # Extract primary name from names dict
    if "names" in target_gdf.columns:
        cols["name"] = target_gdf["names"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )
    elif "name" in target_gdf.columns:
        cols["name"] = target_gdf["name"].values

    if "class" in target_gdf.columns:
        cols["class"] = target_gdf["class"].values

    # Ensure WGS84 and round coordinates to match pair GeoJSON precision
    from .utils import UI_GEOM_PRECISION

    src_gdf = target_gdf
    if src_gdf.crs is None:
        src_gdf = src_gdf.set_crs("EPSG:4326")
    elif src_gdf.crs.to_epsg() != 4326:
        src_gdf = src_gdf.to_crs("EPSG:4326")

    rounded = shapely.set_precision(src_gdf.geometry.values, grid_size=10**-UI_GEOM_PRECISION)
    context_gdf = gpd.GeoDataFrame(cols, geometry=rounded, crs="EPSG:4326")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    context_gdf.to_parquet(cache_path)

    logger.info(f"Context cache saved: {cache_path} ({len(context_gdf)} features)")
    return cache_path


# --- Batch Label service functions ---

# Cache for dataset label counts (expires after 30 seconds)
_label_counts_cache: dict[str, int] | None = None
_label_counts_cache_time: float = 0.0
_LABEL_COUNTS_TTL = 30.0


def get_dataset_label_counts() -> dict[str, int]:
    """Get human label counts per dataset, cached for 30 seconds.

    Returns:
        Dict mapping dataset_id to number of human labels.
    """
    global _label_counts_cache, _label_counts_cache_time

    now = time.monotonic()
    if _label_counts_cache is not None and (now - _label_counts_cache_time) < _LABEL_COUNTS_TTL:
        return _label_counts_cache

    datasets = list_datasets()
    counts = {}
    for ds in datasets:
        store = LabelStore(ds)
        counts[ds] = len(store.df)

    _label_counts_cache = counts
    _label_counts_cache_time = now
    return counts


def _batch_manifest_path(dataset_id: str) -> Path:
    """Get path to batch manifest JSON file."""
    return LABELING_CACHE_DIR / f"{dataset_id}_batch.json"


def has_batch(dataset_id: str) -> bool:
    """Check if a batch manifest exists on disk."""
    return _batch_manifest_path(dataset_id).exists()


def save_batch_manifest(
    dataset_id: str,
    pairs: list[dict],
    feature_version: str,
) -> Path:
    """Save batch manifest to disk.

    Args:
        dataset_id: Dataset identifier
        pairs: List of dicts with ref_id, target_id, bucket keys
        feature_version: Feature cache version string

    Returns:
        Path to saved manifest file
    """
    manifest = {
        "dataset_id": dataset_id,
        "feature_version": feature_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_total": len(pairs),
        "pairs": pairs,
    }

    path = _batch_manifest_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved batch manifest with {len(pairs)} pairs to {path}")
    return path


def load_batch_manifest(dataset_id: str) -> dict | None:
    """Load batch manifest from disk.

    Returns:
        Manifest dict or None if not found.
    """
    path = _batch_manifest_path(dataset_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load batch manifest for %s", dataset_id)
        return None


def delete_batch_manifest(dataset_id: str) -> bool:
    """Delete batch manifest from disk."""
    path = _batch_manifest_path(dataset_id)
    if path.exists():
        path.unlink()
        return True
    return False


def generate_batch(
    dataset_id: str,
    n: int = 50,
    seed: int | None = None,
) -> list[CandidatePairView]:
    """Generate a stratified batch of candidates for labeling.

    Samples from three confidence buckets:
    - likely_match (>0.75): 20% of n
    - borderline (0.50-0.75): 60% of n
    - likely_no_match (<0.50): 20% of n

    If a bucket has fewer candidates than requested, redistributes
    remaining slots to other buckets.

    Args:
        dataset_id: Dataset identifier
        n: Total number of candidates to sample (default 100)
        seed: Random seed for reproducibility

    Returns:
        List of CandidatePairView objects for the batch
    """
    # Load feature cache
    feature_df = load_feature_cache(dataset_id)
    if feature_df is None:
        raise ValueError(f"No feature cache found for {dataset_id}")

    # Run ML prediction on all candidates
    feature_cols = [col for col in feature_df.columns if col in FEATURE_COLUMNS]
    features_list = feature_df[feature_cols].to_dict("records")

    matcher = get_cached_matcher()
    if matcher is None:
        raise ValueError("No ML model available. Train a model first with 'matcher train'.")

    probs = matcher.predict(features_list)
    probs_arr = np.array(probs)

    # Exclude already-labeled pairs
    store = LabelStore(dataset_id)
    labeled_pairs = store.get_labeled_pairs()
    ref_ids = feature_df["ref_id"].values
    target_ids = feature_df["target_id"].values

    # Build mask for unlabeled pairs
    unlabeled_mask = np.array(
        [(str(r), str(t)) not in labeled_pairs for r, t in zip(ref_ids, target_ids)]
    )

    # Apply unlabeled filter
    available_indices = np.where(unlabeled_mask)[0]
    available_probs = probs_arr[available_indices]

    logger.info(
        f"Batch generation: {len(available_indices)} unlabeled candidates "
        f"(excluded {unlabeled_mask.size - len(available_indices)} labeled)"
    )

    # Stratified sampling
    rng = np.random.default_rng(seed)

    buckets = {
        "likely_match": np.where(available_probs > 0.75)[0],
        "borderline": np.where((available_probs >= 0.50) & (available_probs <= 0.75))[0],
        "likely_no_match": np.where(available_probs < 0.50)[0],
    }

    target_counts = {
        "likely_match": n // 5,  # 20%
        "borderline": n * 3 // 5,  # 60%
        "likely_no_match": n // 5,  # 20%
    }

    sampled_indices = []
    remaining = 0

    # First pass: sample up to target from each bucket
    for bucket_name in ["likely_match", "borderline", "likely_no_match"]:
        bucket_indices = buckets[bucket_name]
        target_count = target_counts[bucket_name]

        if len(bucket_indices) <= target_count:
            sampled_indices.extend(bucket_indices.tolist())
            remaining += target_count - len(bucket_indices)
        else:
            chosen = rng.choice(bucket_indices, size=target_count, replace=False)
            sampled_indices.extend(chosen.tolist())

    # Second pass: redistribute remaining slots
    if remaining > 0:
        already_sampled = set(sampled_indices)
        all_available = set(range(len(available_indices)))
        unsampled = sorted(all_available - already_sampled)
        if unsampled:
            extra_count = min(remaining, len(unsampled))
            extra = rng.choice(unsampled, size=extra_count, replace=False)
            sampled_indices.extend(extra.tolist())

    # Map back to feature_df indices
    selected_feature_indices = available_indices[sampled_indices]

    # Build batch manifest
    batch_pairs = []
    for _local_idx, feat_idx in zip(sampled_indices, selected_feature_indices):
        prob = probs_arr[feat_idx]
        if prob > 0.75:
            bucket = "likely_match"
        elif prob >= 0.50:
            bucket = "borderline"
        else:
            bucket = "likely_no_match"

        batch_pairs.append(
            {
                "ref_id": str(ref_ids[feat_idx]),
                "target_id": str(target_ids[feat_idx]),
                "bucket": bucket,
            }
        )

    save_batch_manifest(dataset_id, batch_pairs, FEATURE_VERSION)

    # Build views for sampled candidates
    sampled_feature_df = feature_df.iloc[selected_feature_indices].reset_index(drop=True)

    loader = DatasetLoader(DATA_DIR)
    ref_path = loader.find_reference_path(dataset_id)
    target_path = loader.find_target_path(dataset_id)

    if ref_path is None or target_path is None:
        return []

    reference = load_geodataframe(ref_path)
    target_gdf = load_geodataframe(target_path)

    # Build context tile cache if it doesn't exist yet
    build_context_cache(dataset_id, target_gdf)

    views = build_views_from_feature_df(
        feature_df=sampled_feature_df,
        reference=reference,
        target=target_gdf,
        filter_to_review_band=False,
    )

    if len(views) < n:
        logger.warning(
            f"Batch for {dataset_id} has only {len(views)} candidates (requested {n}) — "
            f"not enough unlabeled candidates to fill the batch"
        )

    logger.info(
        f"Generated batch of {len(views)} candidates for {dataset_id} "
        f"(likely_match={len(buckets['likely_match'])}, "
        f"borderline={len(buckets['borderline'])}, "
        f"likely_no_match={len(buckets['likely_no_match'])})"
    )

    return views


def load_batch(dataset_id: str) -> list[CandidatePairView]:
    """Load a persisted batch from disk.

    Reads the batch manifest, filters the feature cache to batch pair IDs,
    and builds CandidatePairView objects.

    Args:
        dataset_id: Dataset identifier

    Returns:
        List of CandidatePairView objects for the batch
    """
    manifest = load_batch_manifest(dataset_id)
    if manifest is None:
        raise ValueError(f"No batch manifest found for {dataset_id}")

    # Validate feature version
    if manifest.get("feature_version") != FEATURE_VERSION:
        logger.warning(
            f"Batch feature version mismatch: {manifest.get('feature_version')} != {FEATURE_VERSION}"
        )

    # Get batch pair IDs
    batch_pair_ids = {(p["ref_id"], p["target_id"]) for p in manifest["pairs"]}

    # Load feature cache and filter to batch pairs
    feature_df = load_feature_cache(dataset_id)
    if feature_df is None:
        raise ValueError(f"No feature cache found for {dataset_id}")

    mask = pd.Series(
        [
            (str(r), str(t)) in batch_pair_ids
            for r, t in zip(feature_df["ref_id"], feature_df["target_id"])
        ]
    )
    batch_feature_df = feature_df[mask.values].reset_index(drop=True)

    # Build views
    loader = DatasetLoader(DATA_DIR)
    ref_path = loader.find_reference_path(dataset_id)
    target_path = loader.find_target_path(dataset_id)

    if ref_path is None or target_path is None:
        return []

    reference = load_geodataframe(ref_path)
    target_gdf = load_geodataframe(target_path)

    # Build context tile cache if it doesn't exist yet
    build_context_cache(dataset_id, target_gdf)

    views = build_views_from_feature_df(
        feature_df=batch_feature_df,
        reference=reference,
        target=target_gdf,
        filter_to_review_band=False,
    )

    logger.info(f"Loaded batch of {len(views)} candidates for {dataset_id}")
    return views


def generate_batch_from_pairs(
    dataset_id: str,
    pair_ids: list[tuple[str, str]],
) -> list[CandidatePairView]:
    """Generate a batch from a specific list of pair IDs.

    Used for label auditing — accepts a pre-selected list of suspect pairs
    (e.g., match-labeled pairs with low post_node_continuation_m) and creates
    a batch for review in the labeling UI.

    Falls back to label store data (FeatureStore + DataStore) when no feature
    cache exists. This works for already-labeled pairs without requiring
    the expensive full-dataset feature computation.

    Args:
        dataset_id: Dataset identifier
        pair_ids: List of (ref_id, target_id) tuples to include

    Returns:
        List of CandidatePairView objects for the specified pairs
    """
    feature_df = load_feature_cache(dataset_id)
    if feature_df is None:
        # Fall back to label store for already-labeled pairs
        logger.info(f"No feature cache for {dataset_id}, falling back to label store data")
        return _generate_batch_from_label_store(dataset_id, pair_ids)

    # Filter to requested pairs via merge (vectorized)
    pair_df = pd.DataFrame(pair_ids, columns=["ref_id", "target_id"]).astype(str)
    batch_feature_df = feature_df.merge(pair_df, on=["ref_id", "target_id"]).reset_index(drop=True)

    if len(batch_feature_df) == 0:
        logger.warning(f"No matching pairs found in feature cache for {dataset_id}")
        return []

    # Save manifest for persistence
    batch_pairs = [
        {"ref_id": str(r), "target_id": str(t), "bucket": "audit"}
        for r, t in zip(batch_feature_df["ref_id"], batch_feature_df["target_id"])
    ]
    save_batch_manifest(dataset_id, batch_pairs, FEATURE_VERSION)

    # Build views
    loader = DatasetLoader(DATA_DIR)
    ref_path = loader.find_reference_path(dataset_id)
    target_path = loader.find_target_path(dataset_id)

    if ref_path is None or target_path is None:
        return []

    reference = load_geodataframe(ref_path)
    target_gdf = load_geodataframe(target_path)

    views = build_views_from_feature_df(
        feature_df=batch_feature_df,
        reference=reference,
        target=target_gdf,
        filter_to_review_band=False,
    )

    logger.info(
        f"Generated audit batch of {len(views)} candidates for {dataset_id} "
        f"from {len(pair_ids)} requested pairs"
    )
    return views


def _generate_batch_from_label_store(
    dataset_id: str,
    pair_ids: list[tuple[str, str]],
) -> list[CandidatePairView]:
    """Generate batch views from label store data (FeatureStore + DataStore).

    Used as fallback when no feature cache exists. Works for already-labeled
    pairs only, since it reads pre-computed features and geometries from the
    label store's normalized architecture.

    Args:
        dataset_id: Dataset identifier
        pair_ids: List of (ref_id, target_id) tuples to include

    Returns:
        List of CandidatePairView objects for the specified pairs
    """
    from ..features.alignment import create_subline

    feature_store = FeatureStore(dataset_id)
    data_store = DataStore(dataset_id)

    # Load label metadata to get alignment fractions
    label_store = LabelStore(dataset_id)
    labels_df = label_store.df
    alignment_lookup = {}
    for _, row in labels_df.iterrows():
        key = (str(row["gers_id"]), str(row["target_id"]))
        alignment_lookup[key] = {
            "ref_start_frac": float(row.get("ref_start_pct", 0.0)),
            "ref_end_frac": float(row.get("ref_end_pct", 1.0)),
            "target_start_frac": float(row.get("target_start_pct", 0.0)),
            "target_end_frac": float(row.get("target_end_pct", 1.0)),
        }

    # Collect features and geometry data for requested pairs
    features_for_prediction = []
    pair_data = []

    for ref_id, target_id in pair_ids:
        features = feature_store.get(str(ref_id), str(target_id))
        data = data_store.get_pair(str(ref_id), str(target_id))

        if features is None:
            logger.debug(f"No features in label store for {ref_id}/{target_id}")
            continue
        if data is None:
            logger.debug(f"No geometry data in label store for {ref_id}/{target_id}")
            continue

        feature_dict = {col: features.get(col, float("nan")) for col in FEATURE_COLUMNS}
        features_for_prediction.append(feature_dict)
        pair_data.append((str(ref_id), str(target_id), feature_dict, data))

    if not pair_data:
        logger.warning(f"No pairs found in label store for {dataset_id}")
        return []

    # Batch ML prediction
    matcher = get_cached_matcher()
    if matcher and features_for_prediction:
        probs = matcher.predict(features_for_prediction)
    else:
        probs = [0.5] * len(features_for_prediction)

    # Build CandidatePairView objects
    views = []
    for i, (ref_id, target_id, feature_dict, data) in enumerate(pair_data):
        prob = probs[i]

        if prob >= settings.optimizer_match_threshold:
            decision = "match"
        elif prob >= settings.optimizer_review_threshold:
            decision = "review"
        else:
            decision = "no_match"

        score_breakdown = _compute_score_breakdown_from_features(feature_dict)

        # Get alignment fractions from label metadata
        alignment = alignment_lookup.get(
            (ref_id, target_id),
            {
                "ref_start_frac": 0.0,
                "ref_end_frac": 1.0,
                "target_start_frac": 0.0,
                "target_end_frac": 1.0,
            },
        )
        ref_start = alignment["ref_start_frac"]
        ref_end = alignment["ref_end_frac"]
        target_start = alignment["target_start_frac"]
        target_end = alignment["target_end_frac"]

        # Build aligned sub-geometries from fractions
        ref_geom = data["ref_geometry"]
        target_geom = data["target_geometry"]
        ref_aligned = create_subline(ref_geom, ref_start, ref_end)
        target_aligned = create_subline(target_geom, target_start, target_end)

        views.append(
            CandidatePairView(
                ref_id=ref_id,
                target_id=target_id,
                ref_geometry=ref_geom,
                target_geometry=target_geom,
                ref_name=display_name(data.get("ref_names")),
                target_name=display_name(data.get("target_names")),
                ref_class=data.get("ref_class"),
                target_class=data.get("target_class"),
                decision=decision,
                confidence=prob,
                score_breakdown=score_breakdown,
                features=feature_dict,
                ref_topology=data.get("ref_topology"),
                target_topology=data.get("target_topology"),
                ref_aligned_geometry=ref_aligned,
                target_aligned_geometry=target_aligned,
                ref_start_frac=ref_start,
                ref_end_frac=ref_end,
                target_start_frac=target_start,
                target_end_frac=target_end,
            )
        )

    # Save batch manifest for persistence
    batch_pairs = [{"ref_id": v.ref_id, "target_id": v.target_id, "bucket": "audit"} for v in views]
    save_batch_manifest(dataset_id, batch_pairs, FEATURE_VERSION)

    logger.info(
        f"Generated audit batch of {len(views)} from label store for {dataset_id} "
        f"({len(pair_ids)} requested, {len(pair_ids) - len(views)} missing)"
    )
    return views


# --- Stitching Review service functions ---


def load_stitch_batch(dataset_id: str) -> dict | None:
    """Load a stitching review batch from disk.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Batch dict or None if not found
    """
    path = stitch_batch_path(dataset_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load stitch batch for %s", dataset_id)
        return None


def get_unreviewed_stitch_groups(dataset_id: str, groups: list[dict]) -> list[dict]:
    """Filter groups to only those not yet reviewed.

    Args:
        dataset_id: Dataset identifier
        groups: List of group dicts from the batch

    Returns:
        List of unreviewed group dicts
    """
    from ..labeling.stitching_store import StitchingLabelStore

    store = StitchingLabelStore(dataset_id)
    reviewed_ids = store.get_reviewed_group_ids(dataset_id)
    return [g for g in groups if g.get("group_id") not in reviewed_ids]


def record_stitching_label(
    dataset_id: str,
    group_id: str,
    selected_edges: list[dict],
    match_type: str,
    num_refs: int,
    num_targets: int,
) -> None:
    """Record a stitching review label.

    Args:
        dataset_id: Dataset identifier
        group_id: Group identifier
        selected_edges: List of {ref_id, target_id} dicts
        match_type: "1:N", "N:1", or "M:N"
        num_refs: Number of ref segments
        num_targets: Number of target segments
    """
    from ..labeling.stitching_store import StitchingLabelStore

    store = StitchingLabelStore(dataset_id)
    labeler = get_labeler_name()
    session_id = get_session_id()

    store.add(
        group_id=group_id,
        selected_edges=selected_edges,
        match_type=match_type,
        num_refs=num_refs,
        num_targets=num_targets,
        labeler=labeler,
        session_id=session_id,
    )
