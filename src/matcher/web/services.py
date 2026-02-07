"""Service layer for the matcher web UI.

Provides business logic functions for dataset loading, candidate management,
label recording, and configuration access.
"""

import json
import uuid
from pathlib import Path

from ..datasets.loader import DatasetLoader
from ..labeling.data_loader import (
    CandidatePairView,
    filter_candidates,
    generate_scored_candidates_with_cache,
    load_geodataframe,
)
from ..labeling.label_store import LabelStore

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


def undo_last_label(dataset_id: str) -> dict | None:
    """Undo the last label for a dataset.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Dict of the removed label row, or None if nothing to undo.
    """
    store = LabelStore(dataset_id)
    return store.remove_last()
