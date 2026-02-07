"""Service layer for the matcher web UI.

Provides business logic functions for dataset loading, candidate management,
label recording, configuration access, and integration QA.
"""

import json
import logging
import uuid
from pathlib import Path

import geopandas as gpd

from ..datasets.loader import DatasetLoader
from ..filenames import integration_cache_dir
from ..integration_qa.decision_store import MergedDecisionStore, OrphanDecisionStore
from ..labeling.data_loader import (
    CandidatePairView,
    filter_candidates,
    generate_scored_candidates_with_cache,
    load_geodataframe,
)
from ..labeling.label_store import LabelStore

logger = logging.getLogger(__name__)

# Project root: src/matcher/web/services.py -> project root is 3 levels up
PROJECT_ROOT = Path(__file__).parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_FILE = Path.home() / ".matcher_labeler_config.json"


def list_datasets() -> list[str]:
    """List available dataset IDs.

    Returns:
        Sorted list of dataset identifiers that have both
        reference and target files on disk.
    """
    loader = DatasetLoader(DATA_DIR)
    return loader.list_available()


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
        ref_geometry=pair.ref_geometry,
        target_geometry=pair.target_geometry,
        ref_name_raw=pair.ref_name,
        target_name_raw=pair.target_name,
        ref_class_raw=pair.ref_class,
        target_class_raw=pair.target_class,
        ref_subclass=pair.ref_subclass,
        target_subclass=pair.target_subclass,
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
