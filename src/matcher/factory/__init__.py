"""Bridge-table factory: batch, versioned, resumable dataset stitching.

Milestone M4 of ``docs/SCALING_ROADMAP.md``. Discovers stitchable dataset pairs
from a raw-data directory, runs the full stitch pipeline per dataset in parallel
worker processes, and writes a versioned, quality-annotated output layout under
``data/factory/release=<overture-release>/dataset=<name>/`` with a per-dataset
manifest for incremental/resume and a churn-delta report across releases.

See ``docs/FACTORY.md`` for usage and the box-deployment runbook.
"""

from .discovery import DatasetPair, discover_pairs, resolve_release
from .manifest import Manifest
from .runner import FactoryPaths, reoptimize_dataset, run_dataset

__all__ = [
    "DatasetPair",
    "discover_pairs",
    "resolve_release",
    "Manifest",
    "FactoryPaths",
    "run_dataset",
    "reoptimize_dataset",
]
