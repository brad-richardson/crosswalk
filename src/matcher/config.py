"""Configuration settings for the matcher pipeline."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class MatcherSettings(BaseSettings):
    """Global settings for the matcher pipeline."""

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

    # CRS settings
    default_crs: str = Field(
        default="EPSG:4326",
        description="Default CRS for input data",
    )
    working_crs: Optional[str] = Field(
        default=None,
        description="Working CRS for metric calculations (auto-detected if None)",
    )

    class Config:
        env_prefix = "MATCHER_"
        env_file = ".env"


# Global settings instance
settings = MatcherSettings()
