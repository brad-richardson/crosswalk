"""Feature engineering module for candidate pair comparison."""

from .geometric import GeometricFeatures, compute_geometric_features
from .relational import (
    RelationalFeatures,
    compute_endpoint_proximity,
    compute_neighbor_agreement,
    compute_parallel_alignment,
    compute_perpendicular_offset,
    compute_relational_features,
    compute_side_of_street,
)
from .semantic import compute_class_similarity, compute_name_similarity
from .spatial_context import (
    AnchorMatch,
    AnchorRoadMatcher,
    SpatialContextIndex,
    compute_endpoint_features,
)
from .topological import compute_topological_features

__all__ = [
    "compute_geometric_features",
    "GeometricFeatures",
    "compute_name_similarity",
    "compute_class_similarity",
    "compute_topological_features",
    # Relational features
    "compute_relational_features",
    "compute_perpendicular_offset",
    "compute_side_of_street",
    "compute_parallel_alignment",
    "compute_endpoint_proximity",
    "compute_neighbor_agreement",
    "RelationalFeatures",
    # Spatial context
    "AnchorRoadMatcher",
    "AnchorMatch",
    "SpatialContextIndex",
    "compute_endpoint_features",
]
