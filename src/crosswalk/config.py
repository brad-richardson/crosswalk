"""Configuration settings for the crosswalk pipeline."""

import math
import multiprocessing
from pathlib import Path

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

# Maximum distance value for features (used instead of infinity to avoid XGBoost issues)
# 10km represents "very far" for road segment matching
MAX_DISTANCE_METERS = 10000.0

# Metric averaging mode for binary classification evaluation
# "binary" reports precision/recall/F1 for the positive class (match=1),
# giving distinct, actionable metrics. "weighted" makes F1 ≈ accuracy.
METRIC_AVERAGE = "binary"
METRIC_SCORING = "f1"  # sklearn scoring string for cross_val_score

# Default topology features for genuinely unknown topology.
# Uses NaN so XGBoost learns optimal split direction for missing values,
# and NaN is clearly distinguishable from real data (unlike degree=1 which
# looks identical to a real dead-end segment).
DEFAULT_TOPOLOGY_FEATURES = {
    "from_degree": float("nan"),
    "to_degree": float("nan"),
    "is_dead_end": float("nan"),
    "is_intersection": float("nan"),
    "degree_signature": (),
}

# Tolerance for determining if alignment is "full" vs "partial"
# If fractions are within this tolerance of 0.0 or 1.0, treat as full alignment
# Uses 1% tolerance (0.01) consistently across UI display and label metadata
ALIGNMENT_FULL_TOLERANCE = 0.01

# Default snap tolerance for endpoint clustering, topology computation, etc.
# 5 meters is appropriate for road network matching where GPS/digitization error
# typically ranges from 1-5 meters
DEFAULT_SNAP_TOLERANCE_M = 5.0

# Physical overlap filter for candidate pairs (meters)
# The buffer corridor width stays fixed at PHYSICAL_OVERLAP_MIN_M, but the
# acceptance threshold adapts to segment length:
#   threshold = max(FLOOR, min(MIN, shorter_segment_length * 0.5))
# For segments >=10m this behaves identically to the old fixed 5m threshold.
# For shorter segments (sidewalks, footpaths) it scales down, with a 1m floor.
PHYSICAL_OVERLAP_MIN_M = 5.0
PHYSICAL_OVERLAP_FLOOR_M = 1.0

# Minimum number of labels per dataset before it's considered "done" for labeling
MIN_LABELS_PER_DATASET = 50

# Minimum stitching labels per dataset for progress tracking in the UI
MIN_STITCHING_LABELS_PER_DATASET = 15

# Tolerance for matching Overture connectors to target segments.
# How close a target segment must pass to an Overture connector position
# to be considered "at" that junction. 5m matches DEFAULT_SNAP_TOLERANCE_M
# and the typical GPS/digitization error budget.
OVERTURE_ANCHOR_TOLERANCE_M = 5.0

# Alignment divergence detection thresholds
# Used to truncate alignment at points where roads diverge significantly
DIVERGENCE_DISTANCE_MULTIPLIER = 3.0  # Multiple of buffer_distance for distance threshold
DIVERGENCE_MIN_DISTANCE_M = 20.0  # Minimum absolute distance threshold (meters)
DIVERGENCE_PARALLELNESS_THRESHOLD = 0.5  # dot2 < this = diverging (>45 degrees)

# Junction overlap sanity check: when two segments meet at a junction, the
# alignment can report a small spurious overlap near the shared endpoint.
# Re-derive fractions via endpoint projection for overlaps shorter than this.
JUNCTION_MAX_OVERLAP_M = 20.0

# Maximum accepted overlap (meters) for alignment-based grouping and conflict detection.
# Two matches overlapping by more than this on the shared segment are considered incompatible.
MAX_ALIGNMENT_OVERLAP_M = 5.0

# Maximum uncovered distance on the shared segment when optimizer grouping
# rescues complementary, same-name alignment fragments.  The endpoint snap
# tolerance remains the primary contiguity rule; this narrow secondary rule
# spans short connector segments that split an otherwise continuous named
# corridor (the Boston regression cases are 8-12 m).
OPTIMIZER_ALIGNMENT_RESCUE_MAX_GAP_M = 15.0

# Same-side endpoint-distance cap for that rescue. This covers the audited
# 15.15 m and 21.24 m Boston connector gaps while preventing alignment noise
# from joining distant fragments of a repeated street name.
OPTIMIZER_ALIGNMENT_RESCUE_MAX_ENDPOINT_GAP_M = 25.0

# ---------------------------------------------------------------------------
# Junction "sliver" edge classification (single source of truth)
# ---------------------------------------------------------------------------
#
# A junction "sliver" is a candidate edge where two segments of (usually
# different) streets barely overlap — typically where a road end clips the side
# of another road at an intersection (measured examples: 0.2-1.1 m overlap spans,
# ML confidence 0.11-0.38). Slivers pollute stitching labels and can weld
# otherwise-independent groups into monster connected components.
#
# HYBRID rule (both conditions must hold for an edge to be a sliver):
#   1. FRACTION test:  max(ref_span_frac, tgt_span_frac) < SLIVER_SPAN_THRESHOLD
#      Using the max (not min) is deliberate: legitimate asymmetric matches (a
#      10 m local segment against a 1 km ref) have one tiny span but the other
#      near 1.0, so their max stays high. A true sliver substantially covers
#      NEITHER segment, so its max span is small.
#   2. ABSOLUTE test:  max(ref_span_frac*ref_len_m, tgt_span_frac*tgt_len_m)
#                        < SLIVER_ABS_OVERLAP_M
#      The absolute overlap length in meters. This is what a fraction-only test
#      gets wrong: 9% of a 2 km ref = 180 m of real road that a fraction-only
#      test would misclassify as a sliver. The absolute gate keeps it real.
#
# The two tests are AND-ed. This means the fraction test is a NECESSARY gate, so
# there is a residual limitation the hybrid rule does NOT catch: a very short
# stub with a LARGE coverage fraction but tiny absolute overlap (e.g. 0.6 m of a
# 4 m stub = 15% span) passes the fraction gate (0.15 is not < 0.10) and is
# therefore classified as NOT a sliver even though only 0.6 m physically
# overlaps. Catching that case would require an OR / absolute-only rule, which
# was rejected here because it risks dropping legitimate short-segment matches.
#
# Edges with missing/unknown alignment fractions default to a full [0,1] span
# (1.0) and edges with missing/unknown lengths default to +inf meters, so an
# unmeasurable edge is NEVER classified as a sliver (we never drop what we
# cannot measure).
SLIVER_SPAN_THRESHOLD = 0.10  # fraction of segment length (dimensionless)
SLIVER_ABS_OVERLAP_M = 5.0  # absolute overlap floor (meters)

# DISPLAY-ONLY near-sliver band. Not consumed by the optimizer or any label
# gate: it exists purely so evidence packs can flag edges the strict hybrid rule
# does NOT tag but which sit in the same junction-kiss regime the panel argues
# over. An edge is BORDERLINE (see crosswalk.matching.sliver.edge_is_borderline)
# when it is NOT a sliver yet its larger coverage fraction is still below this
# band. The band (1.5x the sliver span threshold) captures two cases:
#   1. Edges that fail the sliver test ONLY on the 5 m absolute floor — a tiny
#      span fraction (< SLIVER_SPAN_THRESHOLD) that maps to >= 5 m on a long
#      urban segment (e.g. 2.9% of a 200 m ref). These are exactly the edges the
#      strict rule leaves untagged on dense datasets.
#   2. Edges sitting just above the fraction threshold
#      (SLIVER_SPAN_THRESHOLD <= max span frac < this band) — "near the
#      boundary" — where inclusion/exclusion is genuinely contested.
SLIVER_BORDERLINE_SPAN_THRESHOLD = 1.5 * SLIVER_SPAN_THRESHOLD  # 0.15


def _sliver_frac(value: float | None, default: float) -> float:
    """Normalize an alignment span fraction, defaulting missing/NaN to ``default``."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return abs(v)


def _sliver_len(value: float | None) -> float:
    """Normalize a segment length (meters), defaulting missing/NaN/<=0 to +inf.

    An unknown length makes the absolute-overlap product large so the edge fails
    the sliver test — we never classify an unmeasurable edge as a sliver.
    """
    if value is None:
        return math.inf
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.inf
    if math.isnan(v) or v <= 0:
        return math.inf
    return v


def sliver_overlap_m(
    ref_span_frac: float | None,
    tgt_span_frac: float | None,
    ref_len_m: float | None = None,
    tgt_len_m: float | None = None,
) -> float:
    """Absolute aligned-overlap length (meters) the hybrid rule's absolute gate uses.

    Returns ``max(ref_span_frac*ref_len_m, tgt_span_frac*tgt_len_m)`` with the
    exact same span/length normalization as :func:`is_sliver_edge`. This is the
    SINGLE definition of an edge's absolute overlap — the sliver classifier and
    any evidence-pack display both read it, so they can never drift. Missing
    lengths normalize to +inf, so the result is +inf for unmeasurable edges.
    """
    rf = _sliver_frac(ref_span_frac, 1.0)
    tf = _sliver_frac(tgt_span_frac, 1.0)
    rl = _sliver_len(ref_len_m)
    tl = _sliver_len(tgt_len_m)
    return max(rf * rl, tf * tl)


def is_sliver_edge(
    ref_span_frac: float | None,
    tgt_span_frac: float | None,
    ref_len_m: float | None = None,
    tgt_len_m: float | None = None,
) -> bool:
    """Classify a candidate edge as a junction sliver using the hybrid rule.

    An edge is a sliver iff BOTH:
      - ``max(ref_span_frac, tgt_span_frac) < SLIVER_SPAN_THRESHOLD`` and
      - ``max(ref_span_frac*ref_len_m, tgt_span_frac*tgt_len_m) < SLIVER_ABS_OVERLAP_M``.

    Args:
        ref_span_frac: Ref-side aligned span as a fraction of ref length (0-1).
        tgt_span_frac: Target-side aligned span as a fraction of target length.
        ref_len_m: Full ref segment length in meters (optional).
        tgt_len_m: Full target segment length in meters (optional).

    Missing/NaN fractions default to a full span (1.0) and missing/NaN lengths
    default to +inf, so an unmeasurable edge is never a sliver. See the module
    comment above for the residual limitation of the AND rule.
    """
    rf = _sliver_frac(ref_span_frac, 1.0)
    tf = _sliver_frac(tgt_span_frac, 1.0)

    frac_test = max(rf, tf) < SLIVER_SPAN_THRESHOLD
    abs_test = sliver_overlap_m(ref_span_frac, tgt_span_frac, ref_len_m, tgt_len_m) < (
        SLIVER_ABS_OVERLAP_M
    )
    return frac_test and abs_test


# Parallel sibling detection thresholds
# Used to detect split carriageway representation (dual highways)
PARALLEL_SIBLING_MIN_OFFSET_M = 5.0  # Minimum lateral offset for sibling
PARALLEL_SIBLING_MAX_OFFSET_M = 30.0  # Maximum lateral offset for sibling
PARALLEL_SIBLING_MIN_ALIGNMENT = 0.9  # Minimum parallel alignment score (0-1)

# Positive same-road evidence thresholds for the *unnamed* sibling path.
# A "sibling" means "same road split into carriageways", not "any parallel
# neighbor". When neither segment is named we cannot rely on name evidence, so
# we demand positive geometric evidence that the two lines are the same road
# rather than accepting any parallel same-class neighbor in the offset band:
#   1. Exact road-class match (not just class tolerance).
#   2. High parallel fraction: a real twin runs parallel along most of the
#      shared stretch, whereas incidental neighbors are only briefly parallel.
#   3. Comparable extent: a twin spans roughly the same stretch, so the two
#      segment lengths must be similar.
PARALLEL_SIBLING_UNNAMED_MIN_PARALLEL_FRACTION = 0.6  # Fraction parallel (0-1)
PARALLEL_SIBLING_UNNAMED_MIN_LENGTH_RATIO = 0.5  # min(len)/max(len) extent match

# Expected half-width by road class (meters)
# Derived from OSM wiki/taginfo typical paved widths
# Used to normalize lateral offset by road type
#
# Source data (typical paved width):
#   motorway:    22-35m (4-8 lanes, almost always dual carriageway)
#   trunk:       14-25m (2-6 lanes, often divided)
#   primary:     10-20m (2-4 lanes, mix of divided/undivided)
#   secondary:   8-15m  (2-3 lanes, mostly undivided)
#   tertiary:    7-12m  (2 lanes, rarely divided)
#   residential: 5-9m   (1-2 lanes, centerline dominant)
#   service:     3-6m   (1 lane, alleys/driveways)
#
EXPECTED_HALF_WIDTH_BY_CLASS_M: dict[str, float] = {
    "motorway": 14.0,  # (22+35)/2 / 2 ≈ 14m
    "trunk": 10.0,  # (14+25)/2 / 2 ≈ 10m
    "primary": 7.5,  # (10+20)/2 / 2 = 7.5m
    "secondary": 5.75,  # (8+15)/2 / 2 ≈ 5.75m
    "tertiary": 4.75,  # (7+12)/2 / 2 ≈ 4.75m
    "residential": 3.5,  # (5+9)/2 / 2 = 3.5m
    "service": 2.25,  # (3+6)/2 / 2 = 2.25m
    "unclassified": 4.0,  # Default, similar to tertiary
    "living_street": 3.0,  # Narrow, pedestrian priority
    "pedestrian": 2.0,  # Pedestrian-only
    "track": 3.0,  # Unpaved/agricultural
    "path": 1.5,  # Footpaths
    "cycleway": 2.0,  # Bike paths
}
DEFAULT_EXPECTED_HALF_WIDTH_M = 4.0  # Fallback for unknown classes

# Standardized column names for parquet files (Overture format)
# The fetch step transforms source columns (e.g., "name_1", "road_classification")
# to these standardized names. Use these constants when reading parquet files.
NAMES_COLUMN = "names"
CLASS_COLUMN = "class"
SUBCLASS_COLUMN = "subclass"

# Linear-referenced attribute columns (stores JSON-serialized LR data)
# These columns store attributes that vary along the segment's length
# Each column contains a list of dicts: [{"start": 0.0, "end": 0.5, "value": "..."}]
NAMES_LR_COLUMN = "names_lr"
SUBCLASS_LR_COLUMN = "subclass_lr"
LEVEL_LR_COLUMN = "level_lr"
ROAD_FLAGS_LR_COLUMN = "road_flags_lr"
ONEWAY_LR_COLUMN = "oneway_lr"
SPEED_LIMIT_KPH_LR_COLUMN = "speed_limit_kph_lr"

# All LR columns for convenience
LR_ATTRIBUTE_COLUMNS = [
    NAMES_LR_COLUMN,
    SUBCLASS_LR_COLUMN,
    LEVEL_LR_COLUMN,
    ROAD_FLAGS_LR_COLUMN,
    ONEWAY_LR_COLUMN,
    SPEED_LIMIT_KPH_LR_COLUMN,
]

# ============================================================================
# DATA AND FEATURE VERSIONING
# ============================================================================

# Schema version (major) - tracks structural/breaking changes to DATA FILES
# Bump when: Parquet schema changes, column renames in data files, ID format changes
# NOTE: Feature column changes go in FEATURE_VERSION, not here
SCHEMA_VERSION = "1"

# Transform version (minor) - tracks data transformation logic
# Bump when: CRS handling, ID mapping, geometry processing changes
TRANSFORM_VERSION = "0"

# Combined data version for filename suffix: v{SCHEMA}.{TRANSFORM}
DATA_VERSION = f"v{SCHEMA_VERSION}.{TRANSFORM_VERSION}"  # e.g., "v1.0"

# Version string for feature computation. Bump this when feature computation
# logic changes to track which features were computed with which code version.
# Format: YYYY-MM-DD or semantic version (e.g., "1.0.0")
FEATURE_VERSION = "2026-07-07.2"

# Version of the post-scoring optimizer/export decision contract. Bump whenever
# grouping, assignment, review demotion, pruning, or bridge-publication logic
# changes in a way that must invalidate factory optimize caches. Runtime knobs
# are snapshotted separately; this token covers algorithmic/code-path changes
# that cannot be represented by a setting value alone.
OPTIMIZER_VERSION = "2026-07-12.1"


def bundled_model_path() -> Path:
    """Path to the pretrained model shipped inside the package.

    This artifact is committed under ``src/crosswalk/_model/`` and ships in the
    wheel, so a fresh clone / ``pip install`` can ``crosswalk stitch`` with zero
    training. Its ``feature_version`` is kept in lockstep with ``FEATURE_VERSION``
    by a CI test (``tests/unit/test_shipped_model.py``) that fails whenever the
    two diverge — forcing a retrain + reship in the same PR that bumps features.
    """
    return Path(__file__).parent / "_model" / "matcher_model_combined.joblib"


def bundled_spark_model_path() -> Path:
    """Path to the Spark-portable XGBoost model shipped inside the package.

    A committed XGBoost-native JSON booster (28 SPARK_PORTABLE_FEATURES) that
    Spark consumers (the tf-data-platform sister project) import straight from
    the wheel instead of hand-copying files. Its ``feature_version`` is kept in
    lockstep with ``FEATURE_VERSION`` by ``tests/unit/test_shipped_spark_model.py``.
    Reship with ``crosswalk export-spark-model`` (see docs/RELEASING.md).
    """
    return Path(__file__).parent / "_model" / "spark_model.json"


def bundled_spark_manifest_path() -> Path:
    """Path to the Spark-portable model manifest shipped inside the package.

    JSON sidecar for :func:`bundled_spark_model_path`: feature list (order
    matters), feature_version, hyperparams, and isotonic calibration knots.
    """
    return Path(__file__).parent / "_model" / "spark_manifest.json"


# ============================================================================
# FEATURE COLUMNS - Single source of truth for ML pipeline
# ============================================================================
# These dicts/lists define all features computed during matching and used for ML.
# Import these in ml.py, compute.py, label_store.py, and feature_panel.py.
#
# IMPORTANT: When adding new features, add them to FEATURE_CATEGORIES below.
# FEATURE_COLUMNS is derived automatically from FEATURE_CATEGORIES.

# Feature categories - organized for display and documentation
# Distance/length features use _m suffix to indicate meters
#
# GEOMETRY PROVENANCE (which geometry each feature is computed from):
#   Aligned portion (34): hausdorff_*, buffer_iou_*, heading_delta,
#       lateral_offset_*, sinuosity_*, min_length_m, edge_distance_rmse_m,
#       collinear_gap_ratio, angle_histogram_similarity, shape_complexity_*,
#       heading_consistency_*, vertex_density_*, has_parallel_sibling_ref,
#       parallel_fraction_ref, crossing_angle_*, transverse_neighbor_fraction_*
#   Full geometry (11): aligned_length_m, coverage_*, intersection_overlap_*
#   Alignment-aware via connector snapping (8): endpoint proximity, graphlet, clustering
#   Alignment-aware via connectors (13): topology features — both ref (Overture explicit
#       connectors) and target (synthetic connectors sampled from full-network spatial index)
#       are computed via compute_aligned_topology_features() with connector data.
#       shared_anchor_count uses Overture connectors projected onto both sides.
#       Fallback to full-segment topology only in labeling UI edge cases.
#   Semantic (11): name_*, has_name_*, class_similarity, route_prefix_match
#   Removed: see docs/RESEARCH_GRAVEYARD.md
FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Geometric": [
        "hausdorff_distance_m",
        "mean_hausdorff_distance_m",
        "hausdorff_p95_m",  # 95th percentile of min-distances (robust to outliers)
        "buffer_iou_5m",  # Tight alignment (exact centerline matches)
        "buffer_iou_15m",  # Offset alignment (sidewalks, bike lanes parallel to roads)
        "heading_delta",
        "collinear_gap_ratio",
        "angle_histogram_similarity",  # Shape fingerprint via turn angle distribution
        "edge_distance_rmse_m",  # RMSE of sampled point distances (Hootenanny)
    ],
    "Name Similarity": [
        "name_levenshtein",
        "name_jaro_winkler",
        "name_token_sort",
        "name_soundex",
        "name_metaphone",
        "has_name_ref",  # 1.0 if ref has non-empty name, else 0.0
        "has_name_target",  # 1.0 if target has non-empty name, else 0.0
        "name_is_generic",  # 1.0 if either name matches generic pattern
        "name_numeric_match",  # Better matching for numbered routes (I-90, US-101)
        "route_prefix_match",  # Compare route types (interstate vs us_route vs state_route)
    ],
    "Class": [
        "class_similarity",
    ],
    "Endpoint/Connectivity": [
        "min_endpoint_proximity_m",  # Min of start/end proximities
        "max_endpoint_proximity_m",  # Max of start/end proximities
        "shared_endpoint_count",
    ],
    "Lateral Offset": [
        "lateral_offset_m",
        "lateral_offset_iqr_m",  # IQR (p75 - p25) - robust to outliers
        "lateral_offset_p95_m",  # 95th percentile of lateral offsets
    ],
    "Topology": [
        "from_degree_ref",
        "to_degree_ref",
        "from_degree_target",
        "to_degree_target",
        # Target-native topology: the target segment's intrinsic endpoint-cluster
        # structure (full-segment, Union-Find on the target network), kept SEPARATE
        # from the unified comparability features above (which project Overture
        # connectors onto the aligned sub-portion for cross-dataset degree parity).
        # The unification (#252) dropped the target's own structure signal — old
        # is_dead_end_target scored AUC 0.583. See PR #256 follow-up.
        "from_degree_target_native",
        "to_degree_target_native",
        "degree_match_score",
        "degree_signature_similarity",
        "is_dead_end_ref",
        "is_dead_end_target",
        "is_dead_end_target_native",
        "dead_end_match",
        "is_intersection_ref",
        "is_intersection_target",
        "is_intersection_target_native",
        "intersection_match",
        "interior_junction_count_ref",
        "interior_junction_count_target",
        "interior_junction_count_delta",
        "interior_connector_jaccard",
        "interior_junction_position_sim",
        "shared_anchor_count",
    ],
    "Alignment Coverage": [
        "ref_coverage",
        "target_coverage",
        "min_coverage",
        "coverage_ratio",
        "max_coverage",
    ],
    "Graphlet": [
        "graphlet_similarity",
        "endpoint_degree_similarity",
    ],
    "Clustering": [
        "clustering_coef_ref",  # Local clustering coefficient at ref endpoints
        "clustering_coef_target",  # Local clustering coefficient at target endpoints
        "clustering_coef_delta",  # Absolute difference in clustering coefficients
    ],
    "Sinuosity": [
        "sinuosity_ref",
        "sinuosity_target",
        "sinuosity_delta",
    ],
    "Heading Consistency": [
        "heading_consistency_ref",
        "heading_consistency_target",
        "heading_consistency_delta",
    ],
    "Vertex Density": [
        "vertex_density_ref",
        "vertex_density_target",
        "vertex_density_ratio",
    ],
    "Length": [
        "min_length_m",
        "aligned_length_m",  # Absolute length of aligned overlap on ref side (meters)
    ],
    "Shape Complexity": [
        "shape_complexity_ref",
        "shape_complexity_target",
        "shape_complexity_delta",
    ],
    "Parallel Sibling": [
        "has_parallel_sibling_ref",  # Whether ref segment has a parallel sibling
        "parallel_fraction_ref",  # Fraction of ref segment with nearby parallel sibling (0-1)
        "offset_vs_half_corridor_ratio",  # Normalized offset for dual carriageway detection
        "offset_over_expected_halfwidth",  # Offset normalized by road class width
        "likely_representation_mismatch",  # Flag when ref/target have different representation
    ],
    "Crossing Angle": [
        "crossing_angle_min_ref",  # Min angle to nearby different-tier corridor, ref side (0-90°)
        "transverse_neighbor_fraction_ref",  # Fraction of nearby different-tier segments >60°, ref side (0-1)
        "crossing_angle_min_target",  # Min angle to nearby different-tier corridor, target side (0-90°)
        "transverse_neighbor_fraction_target",  # Fraction of nearby different-tier segments >60°, target side (0-1)
    ],
    "Intersection Overlap": [
        "post_node_continuation_m",  # How far target continues past alignment boundary along ref heading (meters)
        "endpoint_heading_divergence",  # Max heading difference at alignment boundaries (0-90°)
    ],
    # Road Properties features (oneway_match, speed_limit_similarity) moved to graveyard
    # - Data is still fetched (oneway_lr, speed_limit_kph_lr columns) for future use
    # - See docs/RESEARCH_GRAVEYARD.md for ablation results
}

# Flattened list of all feature columns (derived from FEATURE_CATEGORIES)
FEATURE_COLUMNS: list[str] = [
    feature for features in FEATURE_CATEGORIES.values() for feature in features
]

# Features declared in FEATURE_CATEGORIES but not yet present in the stored
# label feature parquets (labels/features/). Training tolerates these being
# missing from labels (filled with NaN — XGBoost handles missing values
# natively) instead of raising. Remove an entry once a coordinated
# `crosswalk backfill` has run and the updated parquets are committed.
PENDING_BACKFILL_FEATURES: set[str] = set()

# Semantic features - excluded when training geometry-only models
SEMANTIC_FEATURES = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_soundex",
    "name_metaphone",
    "has_name_ref",
    "has_name_target",
    "name_is_generic",
    "class_similarity",
    "name_numeric_match",
    "route_prefix_match",
]


# Features included in Spark-portable models for Overture matching.
# Used by `crosswalk export-spark-model`. Inclusive list (won't break with feature drift).
#
# Spark-portability is a *necessary* condition for membership, not a sufficient
# one: a feature qualifies if it needs nothing but the two aligned geometries and
# the two Overture name structs — no graph topology, no spatial index, no
# connector data. But 45 of the 83 FEATURE_COLUMNS clear that bar, not 28. This
# list is the value-selected subset of them.
#
# Do NOT read an omission here as "infeasible in Spark". The 17 feasible-but-
# omitted features are enumerated and proven computable from a bare pair (bit-for-
# bit against `compute_pair_features`) in tests/test_spark_feature_expansion.py,
# and measured for F1 / size / latency in
# research/spark_feature_expansion_2026-08-07.md. That doc is the reason any given
# one of them is out; `docs/SPARK_MODEL_CARD.md` carries the per-category verdicts.
SPARK_PORTABLE_FEATURES = [
    # Geometry (distance/overlap)
    "hausdorff_distance_m",
    "mean_hausdorff_distance_m",
    "hausdorff_p95_m",
    "buffer_iou_5m",
    "buffer_iou_15m",
    "heading_delta",
    "collinear_gap_ratio",
    "edge_distance_rmse_m",
    # Name similarity (top 3 by importance)
    "name_levenshtein",
    "name_token_sort",
    "name_numeric_match",
    # Class
    "class_similarity",
    # Lateral offset
    "lateral_offset_m",
    "lateral_offset_iqr_m",
    "lateral_offset_p95_m",
    # Coverage
    "ref_coverage",
    "target_coverage",
    "min_coverage",
    "coverage_ratio",
    # Sinuosity
    "sinuosity_ref",
    "sinuosity_target",
    # Heading consistency (target-side only)
    "heading_consistency_target",
    # Length
    "min_length_m",
    "aligned_length_m",
    # Shape complexity (target-side only)
    "shape_complexity_target",
    # Parallel sibling (offset ratio only)
    "offset_over_expected_halfwidth",
    # Intersection overlap
    "post_node_continuation_m",
    "endpoint_heading_divergence",
]  # fmt: skip

# XGBoost hyperparams tuned for the 28-feature Spark-portable model.
# Tuned 2026-07-03 via `scripts/tune_model.py --feature-set spark` (Optuna,
# 100 trials, TPESampler seed=42) with the leakage-free protocol: the seed-42
# holdout was discarded before tuning and the search used inner GroupKFold CV
# on the training portion only, with a size penalty of 0.00001 F1 per tree
# above 100 n_estimators. Epsilon-compact selection (inference speed matters
# for Spark): cheapest trial by n_estimators * max_depth within 0.003 raw
# CV F1 of the best — selected 224 trees x depth 10 (CV F1 0.9216) over the
# best-F1 310 x 10 (CV F1 0.9242).
SPARK_PORTABLE_XGB_PARAMS: dict[str, float | int] = {
    "n_estimators": 224,
    "learning_rate": 0.01275299313255589,
    "max_depth": 10,
    "min_child_weight": 2,
    "subsample": 0.8019037612739637,
    "colsample_bytree": 0.9661600548038851,
    "gamma": 0.6021730351738508,
    "reg_alpha": 1.5439549237262677,
    "reg_lambda": 2.1882487406505136,
    "max_bin": 343,
}


def default_worker_count() -> int:
    """Number of parallel workers, reserving ~10% of cores for other processes."""
    total = multiprocessing.cpu_count()
    return max(1, int(total * 0.9) - 1)


class MatcherSettings(BaseSettings):
    """Global settings for the crosswalk pipeline."""

    model_config = ConfigDict(env_prefix="MATCHER_", env_file=".env", extra="ignore")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    output_dir: Path = Field(default=Path("data/output"), description="Output directory")
    model_path: Path = Field(
        default_factory=bundled_model_path,
        description="Active production ML model. Defaults to the bundled artifact; "
        "set MATCHER_MODEL_PATH or pass stitch --model-path for an explicit override.",
    )
    local_model_path: Path = Field(
        default=Path("data/models/matcher_model_combined.joblib"),
        description="Locally trained full model used by advisory labeling workflows",
    )
    model_geom_only_path: Path = Field(
        default=Path("data/models/matcher_model_geom_only.joblib"),
        description="Path to geometry-only ML model",
    )

    # Overture settings
    overture_release: str | None = Field(
        default=None,
        description="Overture Maps release version (None = use latest)",
    )

    # OSM PBF settings
    pbf_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "matcher" / "pbf",
        description="Cache directory for downloaded PBF files",
    )
    pbf_cache_ttl_hours: int = Field(
        default=24,
        description="Cache TTL for PBF files in hours",
    )

    # Topology settings
    snap_tolerance_m: float = Field(
        default=2.0,
        description="Snap tolerance for undershoots/overshoots (meters)",
    )
    node_cluster_tolerance_m: float = Field(
        default=0.5,
        description="Tolerance for clustering nearby nodes (meters)",
    )
    respect_z_levels: bool = Field(
        default=True,
        description="Respect bridge/tunnel z-levels when detecting intersections",
    )

    # Blocking settings
    buffer_distance_m: float = Field(
        default=50.0,
        description="Candidate search radius (meters)",
    )
    # Probability calibration. When True (default) and the loaded model carries
    # an isotonic calibrator, MLMatcher.predict() returns calibrated P(match).
    # All five confidence thresholds below (scoring_*, optimizer_*, and
    # bridge_min_confidence) are therefore applied to genuine probabilities.
    # Set False to fall back to raw XGBoost scores (e.g. to reproduce
    # pre-calibration behaviour for A/B comparison).
    enable_calibration: bool = Field(
        default=True,
        description="Apply the model's isotonic probability calibrator at inference. "
        "When True, all confidence thresholds operate on calibrated P(match).",
    )

    # Scoring thresholds (per-candidate, used by ML scorer for bridge file output).
    # NOTE: applied to CALIBRATED P(match) when enable_calibration is True.
    scoring_match_threshold: float = Field(
        default=0.5,
        description="Confidence threshold for MATCH decision in ML scoring",
    )
    scoring_review_threshold: float = Field(
        default=0.1,
        description="Confidence threshold for REVIEW decision in ML scoring (below = NO_MATCH)",
    )

    # Optimizer/labeling thresholds (used by 1:N optimizer and labeling UI)
    optimizer_match_threshold: float = Field(
        default=0.75,
        description="Confidence threshold for automatic match in optimizer and labeling UI",
    )
    optimizer_review_threshold: float = Field(
        default=0.5,
        description="Confidence threshold for review in optimizer and labeling UI "
        "(below this = no match)",
    )
    optimizer_memory_limit_gb: float = Field(
        default=8.0,
        description="Memory limit for sparse match optimization in GB. "
        "If estimated memory exceeds this, greedy algorithm is used instead.",
    )

    # Corridor-aware grouping (group-splitting design).
    # When enabled, endpoint-proximity contiguity in 1:N, N:1, M:N, and greedy
    # post-expansion only chains segments that are collinear continuations
    # (turn angle <= max_turn_deg) OR share a normalized name. This stops
    # perpendicular junction kisses from welding independent corridors.
    optimizer_corridor_aware: bool = Field(
        default=True,
        description="Gate optimizer contiguity on collinear continuation or same name.",
    )
    optimizer_corridor_max_turn_deg: float = Field(
        default=40.0,
        description="Max deflection (deg) from straight at a shared endpoint for two "
        "segments to count as a collinear corridor continuation.",
    )
    # Grouping-only confidence prune (secondary): candidate edges below this
    # confidence do not GLUE components together (so low-confidence tangles do
    # not weld into monsters), but they remain scored candidates and stay in a
    # group's edge list when their endpoints already co-land there via stronger
    # edges. They are still eligible for 1:1 greedy assignment, so coverage is
    # unaffected. Distinct from ``min_confidence`` (the candidate floor).
    #
    # CALIBRATION-EQUIVALENT OPERATING POINT. This prune is applied to
    # ``MatchResult.confidence``, which is a CALIBRATED P(match) whenever the
    # active model carries an isotonic calibrator and ``enable_calibration`` is
    # True (the default; see calibration.py). The corridor-aware grouping design
    # (#267) tuned and validated this prune against RAW XGBoost scores at p=0.5
    # (research/group_splitting_design.md: Boston monster residue 23->6, Seattle
    # 85->8, orphaning only 3.7%/5.9% of labeled selected edges). Isotonic
    # calibration maps the mid-range raw 0.5 to ~0.575, so a naive p=0.5 prune on
    # calibrated scores is effectively a weaker raw~0.42 prune that welds more
    # weak edges and regroups the M:N components. Setting the calibrated default
    # to the calibrated image of raw 0.5 keeps the EFFECTIVE prune population
    # equal to what #267 validated (measured regrouping vs raw baseline drops:
    # Boston 7.0%->4.4%, Seattle 15.3%->11.5% edge-membership churn; monster
    # counts identical). The pipeline selects between these two based on whether
    # calibration is actually active on the loaded model (see
    # runner.py::_effective_glue_min_confidence), so an UNcalibrated model still
    # prunes at the raw-0.5 point and does not silently over-prune.
    optimizer_glue_min_confidence: float = Field(
        default=0.575,
        description="Grouping-only glue prune applied to CALIBRATED P(match). "
        "0.575 is the calibrated image of the raw-0.5 point #267 validated.",
    )
    optimizer_glue_min_confidence_raw: float = Field(
        default=0.5,
        description="Grouping-only glue prune used when the active model applies no "
        "calibration (raw XGBoost scores). This is the operating point #267 validated.",
    )
    # Structural export gate (replaces the flat max_edges cap in stitch_export).
    # A group is auto-exportable when it is a single corridor-pair OR has few
    # assignment-components and stays within a soft edge budget; a hard backstop
    # ceiling blocks anything larger regardless (defence against a
    # structure-detection bug auto-exporting a monster).
    stitch_export_max_assignment_components: int = Field(
        default=2,
        description="Max assignment-components for a structurally-simple exportable group.",
    )
    stitch_export_soft_max_edges: int = Field(
        default=30,
        description="Soft edge budget for a structurally-simple exportable group.",
    )
    stitch_export_backstop_max_edges: int = Field(
        default=40,
        description="Hard backstop edge ceiling; no group above this auto-exports.",
    )
    # Panel option pruning (evidence packs only). On monster M:N groups the
    # greedy-perturbation alternatives are ~20 near-duplicate variations of one
    # assignment; the panel gets more signal from a small, maximally-distinct
    # option set. Pruning happens at the metadata level in the evidence-pack
    # path only — the web review UI keeps the full one-click option set.
    stitch_panel_max_options: int = Field(
        default=8,
        description="Max options presented to the LLM stitching panel after diversity "
        "pruning (protected options — optimizer proposal + whole-group seeds — are "
        "always kept, even beyond this cap).",
    )
    stitch_panel_prune_min_distinct_edges: int = Field(
        default=200,
        description="Diversity pruning of panel options only triggers on groups whose "
        "options span MORE than this many distinct candidate edges; smaller groups "
        "keep the full option set (byte-identical packs).",
    )
    # Panel option-order shuffling (evidence packs only; OFF by default). With the
    # flag off, packs list options in canonical order — optimizer proposal first,
    # so option A IS the optimizer's pick — and the POSITION_ANCHOR monitor below
    # carries the anchoring signal (Brad's original monitoring-over-shuffling
    # decision, unchanged by default). Turning the flag ON deterministically
    # shuffles the letter assignment per pack (content-seeded; see
    # stitch_evidence.shuffle_options_for_panel), which breaks the "A = optimizer"
    # anchor but makes letters content-free: POSITION_ANCHOR cannot trip on
    # shuffled-era ballots (the monitor excludes them) and the OPTIMIZER_ANCHOR
    # monitor — which joins ballots to each pack's recorded optimizer letter —
    # becomes the anchoring signal in both modes. Flipping the flag re-mints
    # option_menu_sha256 for regenerated packs, so enable it only at a wave
    # boundary (mid-wave, --resume seat salvage correctly rejects the reordered
    # menus).
    stitch_panel_shuffle_options: bool = Field(
        default=False,
        description="Shuffle panel option presentation order per evidence pack "
        "(deterministic, content-seeded). OFF by default: canonical optimizer-first "
        "order, monitored by POSITION_ANCHOR. Turning it on breaks the 'A = "
        "optimizer' position anchor; OPTIMIZER_ANCHOR then carries the anchoring "
        "signal. Enable only at a wave boundary (packs regenerate with new "
        "option_menu_sha256).",
    )
    # Per-voter bias monitoring for the stitch panel (crosswalk.agent_labeling.
    # panel_monitor). Makes voter defects LOUD instead of found by accident.
    # Motivating evidence: voter `agy` (Gemini Flash via CLI) voted the first-listed
    # option "A" in 11/12 valid ballots at a CONSTANT 0.95 confidence — a
    # position-anchored rubber stamp that inflated unanimity and drove ~1/3 of panel
    # failures in its waves, with nothing surfacing it. Brad originally chose
    # MONITORING over option-letter shuffling (the monitor exposes the anchor;
    # shuffling hides it), and monitoring remains the default. Shuffling is now
    # additionally available as an opt-in mitigation (stitch_panel_shuffle_options
    # above); because shuffled letters are content-free, the OPTIMIZER_ANCHOR
    # monitor keys on the pack's recorded optimizer letter instead of the letter
    # slot and works in BOTH modes. Defaults are conservative so only a genuine
    # anchor / rubber-stamp trips, not ordinary agreement.
    panel_monitor_position_anchor_share: float = Field(
        default=0.6,
        description="POSITION_ANCHOR alarm: fraction of a voter's valid ballots landing on "
        "its single most-common choice POSITION (letter slot; NONE/ABSTAIN excluded). Above "
        "this the voter is picking by slot, not merit.",
    )
    panel_monitor_position_anchor_min_n: int = Field(
        default=10,
        description="Minimum valid ballots before POSITION_ANCHOR can trip (aggregate / "
        "offline monitoring floor). Also the aggregate floor for OPTIMIZER_ANCHOR "
        "(minimum ballots with a known optimizer letter).",
    )
    panel_monitor_optimizer_anchor_share: float = Field(
        default=0.8,
        description="OPTIMIZER_ANCHOR alarm: fraction of a voter's letter ballots (on "
        "groups whose pack records an optimizer letter) that agree with the optimizer's "
        "proposed option. Above this the voter is rubber-stamping the optimizer rather "
        "than judging the geometry. Unlike POSITION_ANCHOR this works whether or not "
        "option order is shuffled. The threshold is calibrated to committed provenance "
        "base rates (the optimizer is genuinely right most of the time): healthy seats "
        "sit at ~0.72-0.73 agreement, the anchoring-suspect codex seat at ~0.81, and "
        "the retired rubber-stamp agy seat at ~0.92 — 0.8 separates them; the "
        "POSITION_ANCHOR-style 0.6 would trip every honest seat.",
    )
    panel_monitor_constant_confidence_std: float = Field(
        default=0.02,
        description="CONSTANT_CONFIDENCE alarm: population std of a voter's confidence over "
        "its cast ballots (letter choices and reject-all NONE verdicts) below this flags a "
        "rubber-stamped constant (e.g. agy's flat 0.95).",
    )
    panel_monitor_constant_confidence_min_n: int = Field(
        default=10,
        description="Minimum scored (finite-confidence) cast ballots before CONSTANT_CONFIDENCE "
        "can trip.",
    )
    panel_monitor_wave_min_n: int = Field(
        default=8,
        description="Lower valid-ballot floor used when surfacing POSITION_ANCHOR at wave time "
        "(per-batch), where a voter has fewer ballots than the aggregate view.",
    )
    # Panel low-confidence routing gate. A unanimous auto_accept whose MINIMUM
    # confidence across valid (non-abstaining) votes falls below this floor is
    # demoted to human_review (route_reason="low_confidence"). Empirically the
    # two Gemini-based voters (agy, opencode) report near-constant inflated
    # confidence (agy pinned at 0.95; opencode median 1.0), so the panel minimum
    # is effectively the calibrated voter's (claude / codex) self-report. Across
    # the committed ballots that minimum cleanly separates flagged-wrong
    # unanimous verdicts (min <= 0.72, e.g. the wrong Boston 25724c6c at 0.66)
    # from clean accepts (min >= 0.78, clustered at 0.90+); 0.75 sits in the
    # empty 0.72–0.78 gap. Applied AFTER the size gate (size_gated wins if both).
    # A blank/NaN confidence on a valid vote counts as BELOW the floor. Set to
    # 0.0 to disable.
    stitch_min_voter_confidence: float = Field(
        default=0.75,
        description="Panel low-confidence gate floor: a unanimous auto_accept whose "
        "minimum valid-vote confidence is below this is demoted to human_review. The "
        "minimum tracks the calibrated voter (the Gemini voters are pinned high), so "
        "0.75 separates flagged-wrong verdicts (<=0.72) from clean accepts (>=0.78). "
        "Set to 0.0 to disable.",
    )
    auto_select_model: bool = Field(
        default=True,
        description="Automatically select between full and geometry-only models based on "
        "target dataset attributes. If target has >50% name coverage, uses full model. "
        "Otherwise, uses geometry-only model if available.",
    )

    # --- Candidate-graph persistence (M2 resolver prerequisite) ---------------
    # The groups sidecar records, per group, the optimizer's SELECTED assignment
    # in ``edges`` (unchanged) PLUS a separate ``rejected_edges`` list carrying
    # the non-selected candidate pairs the optimizer saw for the group's nodes
    # (each with ``selected: false`` + the same structural layer). This makes
    # UNDER-selection learnable for the resolver track and surfaces the extra
    # plausible edges the review UI previously discarded. ``rejected_edges`` is a
    # SIBLING key: every existing consumer of ``edges`` is byte-unaffected, so the
    # stitch gate is provably invariant to it (only resolver extract + a future
    # review UI opt in). Bounded by a per-group cap to keep sidecars tractable
    # (Tunis-scale groups.json is already ~145 MB); truncation is recorded.
    stitch_persist_rejected_edges: bool = Field(
        default=True,
        description="Persist non-selected candidate edges per group in the sidecar "
        "(separate ``rejected_edges`` list). Resolver-track prerequisite; additive.",
    )
    stitch_rejected_edges_max_per_group: int = Field(
        default=64,
        description="Cap on persisted rejected edges per group (highest-confidence "
        "kept). Records ``n_rejected_total`` + ``rejected_truncated`` when exceeded.",
    )
    # Full candidate-graph persistence (learned-resolver flip condition #1, see
    # docs/SCALING_ROADMAP.md). Per group, ``candidate_edges`` records EVERY
    # candidate pair in the group's connected component that passed the optimizer
    # candidate floor (min_confidence), each with its ML confidence and a
    # ``selected`` flag (True iff the pair is in THIS group's optimizer
    # assignment). Unlike ``rejected_edges`` it is uncapped, includes pairs
    # selected elsewhere (marked ``selected_elsewhere``), and uses one uniform
    # minimal schema — the complete pre-selection graph the resolver trains on.
    # Additive sibling key: no existing consumer reads it.
    stitch_persist_candidate_graph: bool = Field(
        default=True,
        description="Persist the FULL per-component candidate graph per group in the "
        "sidecar (``candidate_edges``: every floor-passing pair with confidence + "
        "selected flag). Learned-resolver flip condition #1; additive.",
    )
    stitch_persist_candidates: bool = Field(
        default=True,
        description="Persist a typed ``*_candidates.parquet`` row for every resolver "
        "candidate edge, including all runtime pair features, structural context, "
        "optimizer status, and feature/model provenance. Additive; does not change "
        "matching or the groups JSON.",
    )

    # --- Confidence-drop prune (M2 / resolver Phase 1) -------------------------
    # Post-optimizer prune of group (M:N/1:N/N:1) selections: drop a selected
    # group edge whose (calibrated) confidence is below an absolute threshold.
    # This is the one-parameter model the #272 eval validated (evaluate.py
    # ``baseline_conf`` — an absolute confidence threshold, NOT group-relative —
    # beat both keep-all and the learned per-edge model on the clean slice).
    #
    # PER-DATASET OPT-IN (ALLOWLIST). The optimal floor is dataset-dependent and a
    # dataset over-prunes at the wrong floor (#284's own sweep showed the
    # Boston-tuned 0.96 regresses sidewalk-like sets below keep-all). So the prune
    # applies ONLY to datasets with an explicit, validated threshold in
    # ``resolver_prune_overrides`` — the allowlist, keyed on DATASET IDENTITY (the
    # dataset name the runner is told, never the output filename — #348). A dataset
    # NOT in the map is NOT pruned (effective threshold 0.0, a no-op byte-identical
    # to the pre-prune pipeline); ``runner.py::_effective_prune_threshold`` logs
    # every run's prune decision (on/off + why) so the state is never silent.
    # This replaces the previous "global default 0.96
    # for every dataset" behaviour, which silently over-pruned the ~30 never-tuned
    # datasets. Tune a new dataset via the #284 sweep recipe (see SCALING_ROADMAP
    # M2) BEFORE adding its floor to the allowlist.
    #
    # Shipped allowlist (validated under the #271 stitch gate):
    #   us_boston_streets    0.96  (117-label clean slice: filtered edge-F1
    #                               0.8671 -> 0.8790, group exact 0.5093 -> 0.5833)
    #   us_seattle_sidewalks 0.90  (27-label slice: 0.8665 -> 0.8913, 0.40 -> 0.50;
    #                               0.96 regresses this set below keep-all)
    # Set an allowlist value <= 0 to keep a dataset explicitly disabled.
    #
    # CALIBRATED-ONLY OPERATING POINTS. Every validated floor was tuned on
    # CALIBRATED ``MatchResult.confidence``; unlike the glue prune there is NO
    # validated raw-score point. So the prune is applied only when the active model
    # actually calibrates — ``_effective_prune_threshold`` skips it (returns 0.0)
    # when calibration is inactive, so an uncalibrated model does not silently
    # over-prune raw scores.
    #
    # CONFIG MIGRATION (allowlist cutover): the former global floor field
    # ``resolver_prune_min_confidence`` (0.96, applied to every dataset without an
    # override) has been REMOVED. Its behaviour is now expressed per dataset in
    # ``resolver_prune_overrides``; there is no global fallback floor. Any external
    # config still setting ``resolver_prune_min_confidence`` is silently ignored by
    # pydantic-settings (extra field) — move the value into ``resolver_prune_overrides``
    # keyed by the specific dataset(s) it was validated for.
    resolver_prune_enabled: bool = Field(
        default=True,
        description="Master switch for the post-optimizer confidence-drop prune of "
        "group edges. When True, the prune applies to datasets present in "
        "``resolver_prune_overrides`` (the validated allowlist); datasets absent "
        "there are never pruned. When False, the prune is off for every dataset.",
    )
    resolver_prune_overrides: dict[str, float] = Field(
        default_factory=lambda: {"us_boston_streets": 0.96, "us_seattle_sidewalks": 0.90},
        description="Allowlist of per-dataset confidence-drop prune thresholds, keyed "
        "by DATASET IDENTITY — the dataset name passed to the runner (``crosswalk "
        "stitch`` dataset argument / factory pair name), NEVER derived from the "
        "output filename (#348). A run without a dataset identity (raw -r/-t path "
        "mode) is never pruned. ONLY datasets "
        "listed here are pruned (and only when ``resolver_prune_enabled``); a dataset "
        "absent from the map is never pruned. The value is the absolute (calibrated) "
        "confidence floor for a SELECTED group edge — edges below it are dropped, "
        "except each group always retains its single highest-confidence edge (never "
        "emptied). A value <= 0 keeps a listed dataset explicitly disabled. Tune a "
        "new dataset (see SCALING_ROADMAP M2) before adding it here.",
    )

    # Training data validation thresholds
    training_max_hausdorff_m: float = Field(
        default=1000.0,
        description="Max Hausdorff distance for valid training pairs (meters)",
    )

    # Error handling settings
    error_hard_fail_threshold: float = Field(
        default=0.50,
        description="Fail if any phase exceeds this error rate (0.50 = 50%)",
    )
    error_log_samples: int = Field(
        default=5,
        description="Maximum number of sample errors to log (one per phase:type)",
    )
    # Relational feature settings
    anchor_search_radius_m: float = Field(
        default=30.0,
        description="Max distance to search for anchor road (meters)",
    )
    anchor_min_alignment: float = Field(
        default=0.7,
        description="Minimum parallel alignment to consider as anchor (0-1)",
    )
    endpoint_snap_tolerance_m: float = Field(
        default=5.0,
        description="Tolerance for considering endpoints connected (meters)",
    )
    neighbor_context_radius_m: float = Field(
        default=100.0,
        description="Radius for finding neighboring segments for context propagation (meters)",
    )

    # CRS settings
    default_crs: str = Field(
        default="EPSG:4326",
        description="Default CRS for input data",
    )
    working_crs: str | None = Field(
        default=None,
        description="Working CRS for metric calculations (auto-detected if None)",
    )

    # Bridge output settings
    bridge_min_confidence: float | None = Field(
        default=0.5,
        description="Post-optimization per-edge confidence filter for bridge output. "
        "Edges below this threshold are excluded from the bridge file. "
        "None disables filtering (high recall). Default 0.5 balances precision/recall.",
    )

    # Score propagation settings (EXPERIMENTAL, default off).
    # Structure-aware post-scoring / pre-optimizer step: boost pairs whose
    # topological neighbors are confident consistent matches, dampen pairs that
    # compete with confident non-adjacent alternatives. See
    # matching/score_propagation.py.
    enable_score_propagation: bool = Field(
        default=False,
        description="Enable experimental structure-aware score propagation "
        "between per-pair scoring and the optimizer. Default off (byte-identical).",
    )
    score_propagation_rounds: int = Field(
        default=2,
        description="Number of damped propagation rounds.",
    )
    score_propagation_alpha: float = Field(
        default=0.6,
        description="Boost strength (logit units at full neighbor agreement).",
    )
    score_propagation_beta: float = Field(
        default=0.6,
        description="Dampen strength (logit units at full competitor confidence).",
    )
    score_propagation_damping: float = Field(
        default=0.5,
        description="Per-round contraction applied to the propagated signal.",
    )
    score_propagation_delta_cap: float = Field(
        default=1.5,
        description="Max absolute logit drift from the original score (bounds adjustment).",
    )
    score_propagation_junction_m: float = Field(
        default=20.0,
        description="Grid size (meters) for ref/target junction coincidence.",
    )
    score_propagation_boost_only: bool = Field(
        default=False,
        description="Ablation: apply boost only, disable the dampen term.",
    )

    # Integration settings
    min_segment_length_m: float = Field(
        default=3.0,
        description="Minimum segment length to include in integration (meters). Filters noise.",
    )
    overlap_iou_threshold: float = Field(
        default=0.8,
        description="IoU threshold for detecting overlapping segments during integration",
    )
    overlap_buffer_m: float = Field(
        default=10.0,
        description="Buffer distance for overlap detection (meters)",
    )
    near_duplicate_tolerance_m: float = Field(
        default=2.0,
        description="Distance to consider segments near-duplicates (meters). "
        "Intentionally tight since near-duplicates should nearly overlay.",
    )
    near_duplicate_overlap: float = Field(
        default=0.8,
        description="Minimum overlap ratio to consider as near-duplicate",
    )


# Global settings instance
settings = MatcherSettings()

# Stitch profiles: named configurations for bridge_min_confidence.
# Determined by parameter sweep on us_boston_streets (33 labeled stitch groups).
#
# | Profile   | bridge_min_confidence | Stitch P | Stitch R | Stitch F1 |
# |-----------|----------------------|----------|----------|-----------|
# | recall    | None                 | 0.82     | 0.98     | 0.89      |
# | balanced  | 0.5                  | 0.95     | 0.91     | 0.93      |
# | precision | 0.7                  | 0.99     | 0.90     | 0.94      |
STITCH_PROFILES: dict[str, float | None] = {
    "recall": None,
    "balanced": 0.5,
    "precision": 0.7,
}
