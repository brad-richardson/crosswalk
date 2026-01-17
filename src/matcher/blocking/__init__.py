"""Candidate generation via spatial blocking."""

from .spatial_index import CandidatePair, generate_candidates

__all__ = [
    "generate_candidates",
    "CandidatePair",
]
