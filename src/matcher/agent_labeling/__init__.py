"""Agent labeling pipeline for AI-assisted road segment matching.

This package provides tools for generating labeling tasks that can be
processed by AI agents (Claude Code, Codex, Gemini CLI, etc.) with
support for future consensus-based human review.

Components:
    - sampler: Sample diverse candidates across confidence ranges
    - image_renderer: Generate satellite and geometry overlay images
    - context_generator: Generate YAML metadata for candidates
    - agent_store: Store and manage agent labels separately from training data
"""

from .agent_store import AgentLabelStore
from .context_generator import (
    generate_metadata_yaml,
    write_candidate_package,
    write_candidate_sweep_package,
)
from .image_renderer import (
    fetch_satellite_tile,
    render_candidate_variant,
    render_geometry_only,
    render_subline_geometry_only,
    render_subline_road_context,
    render_with_overlay,
)
from .runner import run_agent_batch
from .sampler import SamplingConfig, sample_candidates

__all__ = [
    "SamplingConfig",
    "sample_candidates",
    "fetch_satellite_tile",
    "render_with_overlay",
    "render_geometry_only",
    "render_subline_geometry_only",
    "render_subline_road_context",
    "render_candidate_variant",
    "generate_metadata_yaml",
    "write_candidate_package",
    "write_candidate_sweep_package",
    "AgentLabelStore",
    "run_agent_batch",
]
