"""Configuration settings for the matcher pipeline."""

from pathlib import Path

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

# Maximum distance value for features (used instead of infinity to avoid XGBoost issues)
# 10km represents "very far" for road segment matching
MAX_DISTANCE_METERS = 10000.0

# Tolerance for determining if alignment is "full" vs "partial"
# If fractions are within this tolerance of 0.0 or 1.0, treat as full alignment
# Uses 1% tolerance (0.01) consistently across UI display and label metadata
ALIGNMENT_FULL_TOLERANCE = 0.01

# ============================================================================
# FEATURE COLUMNS - Single source of truth for ML pipeline
# ============================================================================
# These lists define all features computed during matching and used for ML.
# Import these in ml.py, compute.py, and label_store.py to ensure consistency.

# All feature columns computed by the matcher
# Distance/length features use _m suffix to indicate meters
FEATURE_COLUMNS = [
    # Geometric features (11)
    "hausdorff_distance_m",
    "mean_hausdorff_distance_m",
    "hausdorff_p95_m",  # 95th percentile of min-distances (robust to outliers)
    "buffer_iou_5m",  # Tight alignment (exact centerline matches)
    "buffer_iou_15m",  # Offset alignment (sidewalks, bike lanes parallel to roads)
    "overlap_ratio",  # TODO: Remove - always 1.0 due to blocking bias (candidates are already geometrically close)
    "heading_delta",
    "length_ratio",
    "projection_distance_m",
    "centroid_distance_m",
    "collinear_gap_ratio",
    # Semantic features - name (8)
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_soundex",
    "name_metaphone",
    "has_name_ref",  # 1.0 if ref has non-empty name, else 0.0
    "has_name_target",  # 1.0 if target has non-empty name, else 0.0
    "name_is_generic",  # 1.0 if either name matches generic pattern
    # Semantic features - class (1)
    "class_similarity",
    # Endpoint/connectivity (3) - direction-invariant
    "min_endpoint_proximity_m",  # Min of start/end proximities
    "max_endpoint_proximity_m",  # Max of start/end proximities
    "shared_endpoint_count",
    # Lateral offset (3)
    "lateral_offset_m",
    "lateral_offset_iqr_m",  # IQR (p75 - p25) - robust to outliers
    "lateral_offset_p95_m",  # 95th percentile of lateral offsets
    # Topology features (12)
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
    # Alignment coverage features (4)
    "ref_coverage",
    "target_coverage",
    "min_coverage",
    "coverage_ratio",
    # Graphlet features (2) - network topology similarity
    "graphlet_similarity",
    "endpoint_degree_similarity",
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

    # Matching settings
    match_threshold: float = Field(
        default=0.75,
        description="Confidence threshold for automatic match",
    )
    review_threshold: float = Field(
        default=0.5,
        description="Confidence threshold for review (below this = no match)",
    )
    optimizer_memory_limit_gb: float = Field(
        default=8.0,
        description="Memory limit for sparse match optimization in GB. "
        "If estimated memory exceeds this, greedy algorithm is used instead.",
    )
    alignment_enabled: bool = Field(
        default=True,
        description="Enable pre-match linestring alignment for computing features on "
        "aligned sublines. When enabled, similarity features (hausdorff, buffer_iou, "
        "etc.) are computed on comparable portions of geometries rather than full "
        "geometries. Coverage features are always computed regardless of this setting.",
    )
    auto_select_model: bool = Field(
        default=True,
        description="Automatically select between full and geometry-only models based on "
        "target dataset attributes. If target has >50% name coverage, uses full model. "
        "Otherwise, uses geometry-only model if available.",
    )
    matching_weights: dict[str, float] = Field(
        default={
            "hausdorff_norm": 0.10,
            "mean_hausdorff_norm": 0.10,
            "buffer_iou": 0.15,
            "overlap_ratio": 0.15,
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
