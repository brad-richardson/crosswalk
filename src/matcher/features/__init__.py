"""Feature engineering module for candidate pair comparison."""

from .geometric import compute_geometric_features, GeometricFeatures
from .semantic import compute_name_similarity, compute_class_similarity
from .topological import compute_topological_features

__all__ = [
    "compute_geometric_features",
    "GeometricFeatures",
    "compute_name_similarity",
    "compute_class_similarity",
    "compute_topological_features",
]
