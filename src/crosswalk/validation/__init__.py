"""Validation system for matcher using ground-truth experiments.

This module provides tools to validate crosswalk by:
1. Dropping segments from Overture by record_id (holdout)
2. Fetching fresh OSM data with the same way IDs
3. Running crosswalk to see if dropped segments get matched back
4. Evaluating results against known ground truth
"""

from .evaluate import compute_metrics, evaluate_by_record_id
from .experiment import run_validation_experiment
from .holdout import (
    create_holdout,
    drop_by_bbox,
    drop_by_class,
    drop_by_source,
    drop_random_osm,
    extract_record_ids,
)

__all__ = [
    # Holdout creation
    "extract_record_ids",
    "create_holdout",
    "drop_random_osm",
    "drop_by_bbox",
    "drop_by_source",
    "drop_by_class",
    # Evaluation
    "evaluate_by_record_id",
    "compute_metrics",
    # Experiment
    "run_validation_experiment",
]
