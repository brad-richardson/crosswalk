"""Feature engineering module for candidate pair comparison."""

from .geometric import compute_geometric_features, GeometricFeatures
from .semantic import compute_name_similarity, compute_class_similarity
from .topological import compute_topological_features
from .relational import (
    compute_relational_features,
    compute_perpendicular_offset,
    compute_side_of_street,
    compute_parallel_alignment,
    compute_endpoint_proximity,
    compute_neighbor_agreement,
    RelationalFeatures,
)
from .spatial_context import (
    AnchorRoadMatcher,
    AnchorMatch,
    SpatialContextIndex,
    compute_endpoint_features,
)

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
