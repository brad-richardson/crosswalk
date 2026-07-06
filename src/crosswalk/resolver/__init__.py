"""Experimental learned group resolver (prototype).

This package builds a per-edge keep/drop training dataset from the stitching
group sidecars + curated stitching labels, and prototypes a classifier that
would replace/augment the hand-written optimizer edge selection inside M:N
stitching groups.

STATUS: EXPERIMENTAL. Nothing in this package is imported by the production
pipeline (``matching/optimizer.py``, ``features/pipeline.py``, ``cli/``). It is
a research harness only, exercised via ``scripts/build_resolver_dataset.py`` and
``tests/unit/test_resolver_extract.py``. Wiring it into the optimizer is a
future milestone (see ``research/learned_group_resolver_prototype.md``).
"""

from crosswalk.resolver.extract import (
    EDGE_LABEL_COL,
    build_edge_table,
    load_sidecar_groups,
)
from crosswalk.resolver.features import FEATURE_COLUMNS, featurize

__all__ = [
    "EDGE_LABEL_COL",
    "FEATURE_COLUMNS",
    "build_edge_table",
    "featurize",
    "load_sidecar_groups",
]
