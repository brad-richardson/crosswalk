"""Configuration settings for the matcher pipeline."""

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class MatcherSettings(BaseSettings):
    """Global settings for the matcher pipeline."""

    model_config = ConfigDict(env_prefix="MATCHER_", env_file=".env")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    output_dir: Path = Field(default=Path("data/output"), description="Output directory")

    # Overture settings
    overture_release: str = Field(
        default="2024-12-18.0",
        description="Overture Maps release version",
    )
    overture_s3_region: str = Field(
        default="us-west-2",
        description="AWS region for Overture S3 bucket",
    )

    # Topology settings
    snap_tolerance: float = Field(
        default=2.0,
        description="Snap tolerance for undershoots/overshoots (meters)",
    )
    node_cluster_tolerance: float = Field(
        default=0.5,
        description="Tolerance for clustering nearby nodes (meters)",
    )
    respect_z_levels: bool = Field(
        default=True,
        description="Respect bridge/tunnel z-levels when detecting intersections",
    )

    # Blocking settings
    buffer_distance: float = Field(
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
    matching_weights: dict[str, float] = Field(
        default={
            "hausdorff_norm": 0.20,
            "frechet_norm": 0.10,
            "buffer_iou": 0.20,
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

    # CRS settings
    default_crs: str = Field(
        default="EPSG:4326",
        description="Default CRS for input data",
    )
    working_crs: Optional[str] = Field(
        default=None,
        description="Working CRS for metric calculations (auto-detected if None)",
    )


# Global settings instance
settings = MatcherSettings()
