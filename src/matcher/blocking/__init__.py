"""Candidate generation via spatial blocking."""

from .spatial_index import CandidatePair, generate_candidates, generate_candidates_duckdb

__all__ = [
    "generate_candidates",
    "generate_candidates_duckdb",
    "CandidatePair",
]
