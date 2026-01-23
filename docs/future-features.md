# Future Feature Ideas

This document captures potential ML features for road matching that were identified during feature review but are not yet implemented. These are candidates for future iterations based on feature importance analysis after the current model is trained.

## Geometric Features

### Sinuosity Ratio / Delta
- **Feature**: `sinuosity_ratio`, `sinuosity_delta`
- **Purpose**: Distinguish curvy roads from straight roads
- **Computation**: `sinuosity = line_length / straight_distance` for each segment, then compare
- **Use case**: Prevent matching a curved residential street to a straight highway even if they have similar endpoints

### Fréchet Distance
- **Feature**: `frechet_distance_m`
- **Purpose**: Order-preserving distance metric that considers both position and traversal order
- **Trade-off**: Heavier computation than Hausdorff (~O(n²) with Douglas-Peucker simplification)
- **Use case**: Distinguish segments that have similar point sets but different shapes

### Vertex Density
- **Feature**: `vertex_density_m`
- **Purpose**: Quality signal - high-quality data tends to have consistent vertex spacing
- **Computation**: `num_vertices / line_length`
- **Use case**: Weight matching decisions based on data quality

### Short Segment Flag
- **Feature**: `short_segment_flag`, `min_length_m`
- **Purpose**: Short segments (<10m) may need different matching logic
- **Use case**: Improve matching for ramps, driveways, and connection segments

## Semantic Features

### Cardinal Direction Mismatch
- **Feature**: `cardinal_direction_mismatch`
- **Purpose**: Catch "North Main St" vs "South Main St" false positives
- **Computation**: Extract direction prefix (N/S/E/W/NE/NW/SE/SW) and compare
- **Use case**: Prevent matching different ends of the same named road

### Name Numeric Match
- **Feature**: `name_numeric_match`
- **Purpose**: Better matching for numbered routes (I-90, US-101, Route 66)
- **Computation**: Extract numeric suffix and compare equality
- **Use case**: Numbered highways often have similar names across datasets

## Topology Features

### Junction Angle Similarity
- **Feature**: `junction_angle_similarity`
- **Purpose**: Compare intersection geometry patterns
- **Computation**: For each endpoint, compute angles to connected segments and compare patterns
- **Use case**: Resolve complex urban intersections where multiple segments meet

### Network Continuity Score
- **Feature**: `network_continuity_score`
- **Purpose**: Penalize matches that would create disconnected subgraphs
- **Computation**: Check if match would maintain network connectivity
- **Use case**: Prevent matching isolated segments when better-connected alternatives exist

## Implementation Priority

After training with the current 42 features, feature importance analysis will inform which of these to implement next:

1. **High priority** (if current features show gaps):
   - Cardinal direction mismatch (semantic)
   - Short segment flag (geometric)

2. **Medium priority** (performance optimization):
   - Sinuosity ratio (geometric)
   - Junction angle similarity (topology)

3. **Lower priority** (specialized use cases):
   - Fréchet distance (expensive to compute)
   - Network continuity score (requires global graph analysis)

## Infrastructure & Tooling

### Unified Dataset Fetch/Load/Parse Utility

**Problem**: Dataset loading logic is duplicated across multiple modules (backfill, ML scorer, labeling UI, etc.) with inconsistent handling of:
- Dataset-specific vs shared Overture reference files
- OSM variant datasets (e.g., `us_boston_streets_osm` using base `us_boston_streets` Overture)
- Column naming conventions (id, gers_id, ref_id)
- CRS projection and coordinate systems
- Missing data file detection and fallbacks

**Proposed Solution**: Create a unified `DatasetLoader` class that:
1. **Auto-discovers** data files based on dataset config and naming conventions
2. **Handles variants**: OSM datasets automatically use base dataset Overture files
3. **Provides consistent projections**: Always returns data in appropriate CRS for the task
4. **Validates data**: Fails fast with clear errors when data is missing or malformed
5. **Caches loaded data**: Avoids redundant parquet reads within a session

```python
# Proposed API
from matcher.datasets import DatasetLoader

loader = DatasetLoader(data_dir="data/raw")
ref_gdf, target_gdf = loader.load_pair("us_boston_streets")  # Returns projected GDFs
ref_gdf, osm_gdf = loader.load_pair("us_boston_streets_osm")  # Auto-uses base Overture

# For backfill/batch processing
with loader.session():
    for dataset in loader.list_available():
        ref, target = loader.load_pair(dataset)
        # Data cached within session
```

**Current workarounds**:
- `backfill_features()` has its own `get_reference_data()` helper with caching
- `ml.py` pre-loads data in `score_candidates()`
- Labeling UI's `data_loader.py` has separate loading logic

**Priority**: Medium-High (reduces code duplication and bug surface area)

## References

- Feedback from Codex and Gemini model reviews confirmed the current feature set
- P95 quantile chosen over P90 based on external feedback (top 1-2% are true outliers)
- Generic name patterns expanded to include non-vehicular infrastructure terms
