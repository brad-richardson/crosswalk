"""Configuration settings for the matcher pipeline."""

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
    rl = _sliver_len(ref_len_m)
    tl = _sliver_len(tgt_len_m)

    frac_test = max(rf, tf) < SLIVER_SPAN_THRESHOLD
    abs_overlap = max(rf * rl, tf * tl)
    abs_test = abs_overlap < SLIVER_ABS_OVERLAP_M
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
FEATURE_VERSION = "2026-07-04.1"

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
        "degree_match_score",
        "degree_signature_similarity",
        "is_dead_end_ref",
        "is_dead_end_target",
        "dead_end_match",
        "is_intersection_ref",
        "is_intersection_target",
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
# `matcher backfill` has run and the updated parquets are committed.
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
# These are computable from aligned geometry pairs alone — no graph topology,
# no spatial indexes, no connector data required.
# Used by `matcher export-spark-model`. Inclusive list (won't break with feature drift).
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
    """Global settings for the matcher pipeline."""

    model_config = ConfigDict(env_prefix="MATCHER_", env_file=".env", extra="ignore")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    output_dir: Path = Field(default=Path("data/output"), description="Output directory")
    model_path: Path = Field(
        default=Path("data/models/matcher_model_combined.joblib"),
        description="Path to trained ML model",
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
    # Scoring thresholds (per-candidate, used by ML scorer for bridge file output)
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
    auto_select_model: bool = Field(
        default=True,
        description="Automatically select between full and geometry-only models based on "
        "target dataset attributes. If target has >50% name coverage, uses full model. "
        "Otherwise, uses geometry-only model if available.",
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
