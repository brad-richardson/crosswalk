"""Blocking-stage recall metric.

Measures whether the blocking stage (``generate_candidates``) retains labeled
true-match pairs. Blocking is the first stage of the pipeline: any true match
lost there never reaches the ML scorer, so it is invisible to every downstream
metric (model eval, bridge-file precision/recall, etc.). This module closes
that blind spot by replaying the real blocking path over labeled data.

The computation intentionally reuses the exact inference-time setup:

1. ``filter_to_linestrings`` (same ingest filtering as ``pipeline/runner.py``)
2. ``ensure_projected_crs`` (same metric-CRS projection)
3. ``generate_candidates`` (the actual STRtree + buffer blocking)

Labeled match pairs are then checked for membership in the candidate set.
Pairs whose IDs cannot be resolved in the loaded datasets are counted
separately as ``unresolvable`` and do not count against recall. Pairs whose
target geometry is a MultiLineString are reported in their own bucket, since
the real pipeline drops those at ingest and they can never match.
"""

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger

from ..blocking import generate_candidates
from ..config import settings
from ..utils import ensure_projected_crs
from ..utils.geometry import filter_to_linestrings

# Alternative buffers to report recall at, in addition to the primary buffer.
# Shows what a larger search radius would recover.
DEFAULT_ALT_BUFFERS_M: tuple[float, ...] = (100.0, 150.0)


@dataclass
class MissedPair:
    """A labeled true-match pair that blocking failed to generate."""

    gers_id: str
    target_id: str
    distance_m: float  # Minimum distance between the two geometries (meters)


@dataclass
class BlockingRecallResult:
    """Result of a blocking-recall computation.

    Attributes:
        buffer_distance_m: The buffer distance the real blocking pass used
        total_match_labels: Number of distinct labeled match pairs considered
        blocked: Labeled match pairs present in the candidate set
        missed: Labeled match pairs absent from the candidate set (with the
            minimum geometry-to-geometry distance so the user can see how far
            outside the buffer they were)
        unresolvable: Labeled pairs whose IDs are not present in the loaded
            datasets (do not count against recall)
        multilinestring_dropped: Labeled pairs whose target geometry is a
            MultiLineString; the pipeline drops these at ingest so they can
            never match regardless of buffer
        recall_at_buffer: Recall at the primary and alternative buffers,
            computed by thresholding minimum pair distances (one blocking-
            equivalent measurement, no repeated passes)
    """

    buffer_distance_m: float
    total_match_labels: int
    blocked: int
    missed: list[MissedPair] = field(default_factory=list)
    unresolvable: list[tuple[str, str]] = field(default_factory=list)
    multilinestring_dropped: list[tuple[str, str]] = field(default_factory=list)
    recall_at_buffer: dict[float, float] = field(default_factory=dict)

    @property
    def n_resolvable(self) -> int:
        """Labeled match pairs that could be checked against blocking."""
        return self.blocked + len(self.missed)

    @property
    def recall(self) -> float:
        """Fraction of resolvable labeled match pairs retained by blocking."""
        if self.n_resolvable == 0:
            return float("nan")
        return self.blocked / self.n_resolvable


def _pair_tuples(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Extract (gers_id, target_id) tuples from a labels DataFrame."""
    return list(df[["gers_id", "target_id"]].itertuples(index=False, name=None))


def compute_blocking_recall(
    reference_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    labels_df: pd.DataFrame,
    buffer_distance_m: float | None = None,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    alt_buffer_distances: tuple[float, ...] = DEFAULT_ALT_BUFFERS_M,
) -> BlockingRecallResult:
    """Compute blocking-stage recall against labeled true matches.

    Runs the SAME ``generate_candidates`` blocking used at inference (same
    ingest filtering and metric-CRS projection) and measures what fraction of
    labeled match pairs survive it.

    Args:
        reference_gdf: Reference (Overture) GeoDataFrame, as loaded from disk
        target_gdf: Target (local data) GeoDataFrame, as loaded from disk
        labels_df: Labels with at least gers_id, target_id, label columns
        buffer_distance_m: Blocking search radius (None = settings default)
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        alt_buffer_distances: Extra buffers to report recall at

    Returns:
        BlockingRecallResult with recall, missed pairs (and their distances),
        unresolvable pairs, MultiLineString-dropped pairs, and recall at
        alternative buffers
    """
    buffer_distance_m = buffer_distance_m or settings.buffer_distance_m

    # Distinct labeled true-match pairs
    matches = labels_df[labels_df["label"] == "match"][["gers_id", "target_id"]]
    matches = matches.astype(str).drop_duplicates()
    total_match_labels = len(matches)

    buffers = sorted({float(buffer_distance_m), *(float(b) for b in alt_buffer_distances)})

    if total_match_labels == 0:
        logger.warning("No labeled match pairs found - nothing to measure")
        return BlockingRecallResult(
            buffer_distance_m=buffer_distance_m,
            total_match_labels=0,
            blocked=0,
            recall_at_buffer={b: float("nan") for b in buffers},
        )

    # Bucket pairs whose target is a MultiLineString: the real pipeline drops
    # these at ingest (filter_to_linestrings), so they can never match.
    target_ids_str = target_gdf[target_id_column].astype(str)
    mls_target_ids = set(target_ids_str[target_gdf.geometry.geom_type == "MultiLineString"])
    mls_mask = matches["target_id"].isin(mls_target_ids)
    multilinestring_dropped = _pair_tuples(matches[mls_mask])
    matches = matches[~mls_mask]

    # Same ingest filtering as the inference pipeline
    reference = filter_to_linestrings(reference_gdf, source_name="reference")
    target = filter_to_linestrings(target_gdf, source_name="target")

    # Pairs whose IDs are not in the loaded datasets cannot be checked
    ref_ids = set(reference[ref_id_column].astype(str))
    target_ids = set(target[target_id_column].astype(str))
    resolvable_mask = matches["gers_id"].isin(ref_ids) & matches["target_id"].isin(target_ids)
    unresolvable = _pair_tuples(matches[~resolvable_mask])
    matches = matches[resolvable_mask]

    if len(matches) == 0:
        logger.warning("No resolvable labeled match pairs - recall is undefined")
        return BlockingRecallResult(
            buffer_distance_m=buffer_distance_m,
            total_match_labels=total_match_labels,
            blocked=0,
            unresolvable=unresolvable,
            multilinestring_dropped=multilinestring_dropped,
            recall_at_buffer={b: float("nan") for b in buffers},
        )

    # Same metric-CRS projection as the inference pipeline
    projection_result = ensure_projected_crs(reference, target)
    reference_proj = projection_result.reference
    target_proj = projection_result.target
    if projection_result.was_reprojected:
        logger.info(f"Projected to {projection_result.projected_crs} for meter-based blocking")

    # The REAL blocking pass (the code path we are measuring)
    candidates = generate_candidates(
        reference=reference_proj,
        target=target_proj,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    # Membership check: restrict the candidate set to labeled target IDs
    # first so we never materialize a Python set over millions of pairs.
    if len(candidates) > 0:
        cand_target_ids = candidates.target_ids.astype(str)
        keep = np.isin(cand_target_ids, matches["target_id"].to_numpy())
        candidate_pairs = set(zip(candidates.ref_ids[keep].astype(str), cand_target_ids[keep]))
    else:
        candidate_pairs = set()

    # Minimum geometry-to-geometry distances for every resolvable labeled pair
    # (vectorized; one distance per labeled pair). These drive both the missed
    # pair report and recall at alternative buffers - blocking with buffer B
    # retains a pair iff min distance <= B, so a single measurement covers all
    # buffers without repeated blocking passes.
    ref_geom_by_id = dict(
        zip(reference_proj[ref_id_column].astype(str), reference_proj.geometry.values)
    )
    target_geom_by_id = dict(
        zip(target_proj[target_id_column].astype(str), target_proj.geometry.values)
    )
    ref_geoms = np.array([ref_geom_by_id[g] for g in matches["gers_id"]], dtype=object)
    target_geoms = np.array([target_geom_by_id[t] for t in matches["target_id"]], dtype=object)
    distances = shapely.distance(ref_geoms, target_geoms)

    blocked = 0
    missed: list[MissedPair] = []
    for (gers_id, target_id), dist in zip(_pair_tuples(matches), distances):
        if (gers_id, target_id) in candidate_pairs:
            blocked += 1
        else:
            missed.append(MissedPair(gers_id=gers_id, target_id=target_id, distance_m=float(dist)))
    missed.sort(key=lambda p: p.distance_m)

    recall_at_buffer = {b: float(np.mean(distances <= b)) for b in buffers}

    result = BlockingRecallResult(
        buffer_distance_m=buffer_distance_m,
        total_match_labels=total_match_labels,
        blocked=blocked,
        missed=missed,
        unresolvable=unresolvable,
        multilinestring_dropped=multilinestring_dropped,
        recall_at_buffer=recall_at_buffer,
    )
    logger.info(
        f"Blocking recall @ {buffer_distance_m:g}m: {result.recall:.4f} "
        f"({blocked}/{result.n_resolvable} resolvable pairs; "
        f"{len(missed)} missed, {len(unresolvable)} unresolvable, "
        f"{len(multilinestring_dropped)} MultiLineString-dropped)"
    )
    return result
