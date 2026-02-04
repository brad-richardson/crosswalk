"""Configuration settings for the matcher pipeline."""

from pathlib import Path

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

# Maximum distance value for features (used instead of infinity to avoid XGBoost issues)
# 10km represents "very far" for road segment matching
MAX_DISTANCE_METERS = 10000.0

# Default topology features for empty/missing geometries
# Represents an isolated dead-end segment (degree 1 at both endpoints)
DEFAULT_TOPOLOGY_FEATURES = {
    "from_degree": 1,
    "to_degree": 1,
    "is_dead_end": True,
    "is_intersection": False,
    "degree_signature": (1,),
}

# Tolerance for determining if alignment is "full" vs "partial"
# If fractions are within this tolerance of 0.0 or 1.0, treat as full alignment
# Uses 1% tolerance (0.01) consistently across UI display and label metadata
ALIGNMENT_FULL_TOLERANCE = 0.01

# Default snap tolerance for endpoint clustering, topology computation, etc.
# 5 meters is appropriate for road network matching where GPS/digitization error
# typically ranges from 1-5 meters
DEFAULT_SNAP_TOLERANCE_M = 5.0

# Minimum physical overlap for candidate pairs (meters)
# Pairs with less actual geometric intersection (without alignment translation)
# are rejected early. Based on label analysis: 5m gives 5.9:1 no_match:match
# filter ratio (removes 46.5% of no_match while only losing 6.7% of match).
PHYSICAL_OVERLAP_MIN_M = 5.0

# Alignment divergence detection thresholds
# Used to truncate alignment at points where roads diverge significantly
DIVERGENCE_DISTANCE_MULTIPLIER = 3.0  # Multiple of buffer_distance for distance threshold
DIVERGENCE_MIN_DISTANCE_M = 20.0  # Minimum absolute distance threshold (meters)
DIVERGENCE_PARALLELNESS_THRESHOLD = 0.5  # dot2 < this = diverging (>45 degrees)

# Parallel sibling detection thresholds
# Used to detect split carriageway representation (dual highways)
PARALLEL_SIBLING_MIN_OFFSET_M = 5.0  # Minimum lateral offset for sibling
PARALLEL_SIBLING_MAX_OFFSET_M = 30.0  # Maximum lateral offset for sibling
PARALLEL_SIBLING_MIN_ALIGNMENT = 0.9  # Minimum parallel alignment score (0-1)

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
FEATURE_VERSION = "2026-02-04"

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
FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Geometric": [
        "hausdorff_distance_m",
        "mean_hausdorff_distance_m",
        "hausdorff_p95_m",  # 95th percentile of min-distances (robust to outliers)
        "buffer_iou_5m",  # Tight alignment (exact centerline matches)
        "buffer_iou_15m",  # Offset alignment (sidewalks, bike lanes parallel to roads)
        "heading_delta",
        "length_ratio",
        "centroid_distance_m",
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
    ],
    "Alignment Coverage": [
        "ref_coverage",
        "target_coverage",
        "min_coverage",
        "coverage_ratio",
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
    # Road Properties features (oneway_match, speed_limit_similarity) moved to graveyard
    # - Data is still fetched (oneway_lr, speed_limit_kph_lr columns) for future use
    # - See docs/RESEARCH_GRAVEYARD.md for ablation results
}

# Flattened list of all feature columns (derived from FEATURE_CATEGORIES)
FEATURE_COLUMNS: list[str] = [
    feature for features in FEATURE_CATEGORIES.values() for feature in features
]

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
    max_heading_diff: float = Field(
        default=45.0,
        description="Maximum heading difference for candidates (degrees)",
    )
    max_length_ratio: float = Field(
        default=5.0,
        description="Maximum length ratio for candidates",
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

    # Error handling settings
    error_hard_fail_threshold: float = Field(
        default=0.50,
        description="Fail if any phase exceeds this error rate (0.50 = 50%)",
    )
    error_log_samples: int = Field(
        default=5,
        description="Maximum number of sample errors to log (one per phase:type)",
    )
    matching_weights: dict[str, float] = Field(
        default={
            "hausdorff_norm": 0.10,
            "mean_hausdorff_norm": 0.10,
            "buffer_iou": 0.30,
            "heading_norm": 0.10,
            "length_ratio": 0.10,
            "projection_norm": 0.10,
            "name_similarity": 0.15,
            "class_similarity": 0.05,
        },
        description="Feature weights for match scoring (must sum to 1.0)",
    )

    @field_validator("matching_weights")
    @classmethod
    def validate_weights_sum(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that matching weights sum to 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point tolerance
            raise ValueError(f"matching_weights must sum to 1.0, got {total:.4f}")
        return v

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
