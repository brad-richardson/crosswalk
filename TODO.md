# Matcher TODO & Future Features

This document consolidates all future feature ideas, technical debt, and improvement opportunities for the road network matcher.

---

## Table of Contents

1. [Geometric Features](#geometric-features)
2. [Semantic Features](#semantic-features)
3. [Attribute Features](#attribute-features)
4. [Topology Features](#topology-features)
5. [Graph Embeddings (Research)](#graph-embeddings-research)
6. [Blocking & Candidate Generation](#blocking--candidate-generation-optimization)
7. [Sub-segment Matching](#sub-segment-matching)
8. [Infrastructure & Tooling](#infrastructure--tooling)
9. [Label Data Management](#label-data-management)
10. [Other Ideas](#other-ideas)
11. [Known Issues & Technical Debt](#known-issues--technical-debt)

---

## Geometric Features

### Fréchet Distance
- **Feature**: `frechet_distance_m`
- **Purpose**: Order-preserving distance metric that considers both position and traversal order
- **Trade-off**: Heavier computation than Hausdorff (~O(n^2) with Douglas-Peucker simplification)
- **Use case**: Distinguish segments that have similar point sets but different shapes
- **Priority**: Lower (expensive to compute)

### Short Segment Flag
- **Feature**: `short_segment_flag`, `min_length_m`
- **Purpose**: Short segments (<10m) may need different matching logic
- **Use case**: Improve matching for ramps, driveways, and connection segments

### Turn-Angle Histogram
- **Feature**: `turn_angle_histogram`
- **Purpose**: Capture shape characteristics via distribution of turn angles
- **Computation**: Compute histogram of angles between consecutive segments
- **Academic basis**: Hootenanny "shape distance" and conflation literature
- **Use case**: Compare shape fingerprints between candidate pairs

---

## Semantic Features

### Name Abbreviation Normalization
- **Feature**: Enhancement to existing name similarity
- **Purpose**: Better matching when datasets use different abbreviations
- **Computation**: Pre-process with abbreviation dictionary (St->Street, Ave->Avenue, Blvd->Boulevard)
- **Use case**: "Main St" vs "Main Street" should be near-identical match

### Route Number Normalization
- **Feature**: `route_number_similarity`
- **Purpose**: Handle different representations of numbered routes
- **Computation**: Normalize "I-5", "Interstate 5", "I 5" to canonical form; compare shield prefixes
- **Use case**: Highway datasets often have inconsistent route number formatting

### Reference/Alt Name Token Overlap
- **Feature**: `ref_alt_name_overlap`
- **Purpose**: Use alternative names and ref tags for matching
- **Computation**: Token overlap between ref/alt_name fields when primary names don't match
- **Use case**: Roads with multiple official names or abbreviations

### Language-Aware Name Comparison
- **Feature**: `name_language_similarity`
- **Purpose**: Handle international datasets with non-English names
- **Computation**: Language detection, choose appropriate phonetic algorithm
- **Use case**: International deployments (co_bogota, etc.)
- **Priority**: Medium for international datasets, LOW for English-only

---

## Attribute Features

Features derived from road attributes beyond names and classes.

### Lane Count Similarity
- **Feature**: `lane_count_match`, `lane_count_diff`
- **Purpose**: Use lane count as matching signal
- **Computation**: Compare lane counts; penalize large differences
- **Data sources**: Already exposed in several dataset configs
- **Use case**: Multi-lane highways vs. single-lane residential

### Speed Limit Similarity
- **Feature**: `speed_limit_diff`
- **Purpose**: Speed limits correlate with road class and should match
- **Computation**: Absolute difference in speed limits (when available)
- **Use case**: Distinguish frontage roads from highways

### Infrastructure Flags
- **Features**: `oneway_match`, `surface_match`, `bridge_tunnel_match`
- **Purpose**: Binary flags that should agree for matching segments
- **Computation**: Direct comparison of oneway, surface, bridge/tunnel attributes
- **Use case**: Avoid matching one-way street to two-way street

---

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
- **Priority**: Lower (requires global graph analysis)

### Junction Angle Signature
- **Feature**: `junction_angle_signature`
- **Purpose**: Compare intersection geometry beyond simple degree count (more detailed than `junction_angle_similarity` above)
- **Computation**: For each endpoint, extract bearings of all incident edges; compare "bearing fingerprints" (sorted list of angles)
- **Gap**: Currently have `heading_delta` but no feature comparing angles of incident roads at intersection nodes
- **Use case**: Distinguish T-intersections (90 degree branch) from Y-splits

### Local Clustering Coefficient
- **Feature**: `clustering_coefficient`
- **Purpose**: Distinguish grid intersections from branching patterns
- **Computation**: For each endpoint, count triangles involving node v, normalized by possible edges
- **Use case**: Urban grid areas vs. suburban cul-de-sac patterns
- **Implementation**: NetworkX has built-in `nx.clustering()` function

### K-Hop Path Continuity
- **Feature**: `path_continuity_k`
- **Purpose**: Check if segments continue into matching neighbors at k hops
- **Computation**: Follow connected segments k hops; compare continuity patterns
- **Academic basis**: Hootenanny network-style propagation
- **Use case**: Validate matches by checking network context

### Seed-and-Grow Neighbor Agreement
- **Feature**: `neighbor_agreement_score`
- **Purpose**: Reinforce confidence when nearby segments also have strong matches
- **Computation**: Start from high-confidence "seed" matches; propagate scores to neighbors
- **Academic basis**: Hootenanny-style propagation, Volz et al. MRF conflation
- **Use case**: Catch erroneous matches that would isolate segments
- **Priority**: Medium (requires iterative post-processing)

### Enhanced Degree Signature Similarity
- **Feature**: Improvement to existing `degree_signature_similarity`
- **Problem**: Current implementation compares raw degree tuples - (1,2,3,4) vs (1,2,3,5) treated very differently
- **Proposed**: Use Earth Mover's Distance (EMD) or optimal matching instead of simple comparison
- **Use case**: More nuanced comparison of "4-way intersection" vs "5-way intersection"

### Investigate Graphlet Performance
- **Status**: Enabled full graphlet features (Jan 2026)
- **Background**: Disabled `degrees_only` mode to compute full 6-dimensional graphlet vectors:
  - degree, triangles, squares, clustering, two_hop_count, is_articulation
- **Observation**: CV F1 improved slightly (0.905 vs 0.901), holdout metrics unchanged (99.6%)
- **Open questions**:
  - Are graphlet features providing signal or just noise?
  - Which graphlet components are most predictive (triangles? clustering?)?
  - Does performance vary by dataset type (urban grid vs suburban)?
  - Memory/compute trade-off worth the marginal improvement?
- **Suggested experiments**:
  - Feature importance analysis for graphlet components
  - Ablation study: train with/without each graphlet component
  - Per-dataset breakdown of graphlet feature impact
- **Priority**: Low-Medium (incremental improvement)

### Graphlet Signature / Lateral Offset (Partial)
- **Status**: PARTIALLY IMPLEMENTED (Jan 2026)
- **Implemented**: `lateral_offset_m`, `lateral_offset_iqr_m`, `lateral_offset_p95_m` features
- **Measures**: Perpendicular distance between candidate segments
- **Use case**: Same-side sidewalks have low lateral offset (< 5m), opposite-side have high (10-30m)
- **Gap**: Junction angles and intersection connectivity signature not yet implemented

---

## Graph Embeddings (Research)

**Priority:** Low (exploratory research)
**Status:** Idea

### Problem

Current topology features are hand-crafted and limited to local neighborhood. Graph neural networks could learn richer representations of segment context.

### Proposed Approaches

#### Node2Vec / GraphSAGE Embeddings
- **Feature**: `graph_embedding_similarity`
- **Purpose**: Use learned embeddings as segment context representation
- **Computation**: Train node2vec or GraphSAGE on road network; use embeddings as feature vectors
- **Trade-off**: Requires pre-training on full network; adds complexity

#### Siamese GNN
- **Purpose**: Learn to directly compare two segments' network contexts
- **Computation**: Feed both segments through shared GNN; compare output embeddings
- **Academic basis**: Graph neural networks for entity matching

### Implementation Notes

- Could use PyTorch Geometric or DGL
- Pre-train on OSM or Overture network
- Use as additional similarity signal, not replacement for interpretable features
- May be overkill for current dataset sizes

---

## Blocking & Candidate Generation Optimization

### Adaptive Buffer Distance
- **Problem**: Fixed 50m buffer may be too loose or too tight for different datasets
- **Proposed**: Auto-detect optimal buffer via alignment statistics on sample
  - If dataset has poor alignment: increase buffer to 100m
  - If dataset is well-aligned: decrease to 25m
- **Impact**: 10-15% reduction in false candidates

### Heading-Based Candidate Pruning
- **Problem**: Currently generates ALL spatial neighbors, then computes features
- **Proposed**: Add max_heading_diff filter (e.g., < 45 degrees) BEFORE feature computation
- **Impact**: 30-40% fewer features to compute, no accuracy loss (weak candidates already filtered by ML)

---

## Sub-segment Matching

**Priority:** Medium
**Status:** Deferred until initial ML model is trained

### Problem

Current matching is whole-segment only. When segments from different datasets have different segmentation (split at different intersections), we can identify that they match but not *which portions* overlap.

Example: Overture segment A (Tremont to Head Place) partially overlaps Boston segment B (Tremont to Tamworth). They match, but only ~30% of each segment actually corresponds.

### Why It Matters

For actual conflation/merging of datasets, we need to know:
- Which portion of segment A corresponds to segment B
- Whether segment A also matches segment C (for the non-overlapping portion)
- How to transfer attributes between partially-matching segments

### Potential Approaches

1. **Linear Referencing in Labels**
   - Store match as: `(ref_id, target_id, ref_start_pct, ref_end_pct, target_start_pct, target_end_pct)`
   - Requires UI changes to select sub-portions
   - More precise training data

2. **Post-ML Geometric Alignment**
   - Run ML model first to identify candidate matches
   - Then use geometric algorithms (e.g., Fréchet matching, point projection) to find exact correspondence
   - Simpler labeling, alignment handled algorithmically

3. **Segment Pre-processing**
   - Re-segment both datasets at common break points before matching
   - Could use intersection detection to find natural break points
   - Makes matching easier but adds preprocessing complexity

### Recommendation

Start with **Approach 2** (post-ML geometric alignment):
- Train ML model with current whole-segment labels
- Add alignment step in conflation pipeline after matching
- Evaluate if sub-segment labels (Approach 1) improve model quality

---

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

---

## Label Data Management

### Stable ID Strategy

**Problem**: Labels reference segments by ID, but IDs can change when data is re-fetched (e.g., `boston_streets_123` becomes `us_boston_streets_123`). This breaks the link between labels and their source data.

**Best Practices**:
1. **Never re-fetch data that has labels** without checking label compatibility
2. **Use source-provided IDs when available**: Many datasets include stable FIDs (e.g., ArcGIS `source_tags.FID`) that persist across fetches
3. **Consider geometry hashes**: For datasets without stable IDs, a hash of the geometry provides a reproducible identifier
4. **Add `--id-column` option**: Allow users to specify which column to use for IDs during fetch

**Proposed Improvements**:
- Add `--id-column` flag to `matcher fetch` commands
- Store source FID in parquet metadata for auditing
- Create migration tool for updating label IDs when prefixes change

### Feature Version Management

**Current State**: Labels now track `feature_version` to identify which computation logic was used.

**Future Improvements**:
- Semantic versioning for features (e.g., `2.1.0`)
- Automatic feature version bumps in CI when `features/*.py` changes
- Deprecation warnings when loading labels with outdated feature versions
- Migration tooling to backfill features when versions differ

### Label Archive & History

**Problem**: When labels become orphaned (IDs no longer match), they're either lost or manually fixed.

**Proposed Solution**:
- Archive orphaned labels to `labels/archived/` instead of deleting
- Track label history (when labeled, when backfilled, version changes)
- Provide recovery tooling to re-link archived labels when data issues are fixed

### Data Lineage Documentation

**Problem**: It's hard to trace which data version was used for which model.

**Proposed Solution**:
- Store data versions in model metadata
- Add `matcher model-info` command to show training data provenance
- Include data lineage in MLflow/experiment tracking

### Data File Versioning / Schema Checksums

**Priority:** Medium
**Status:** Idea

**Problem**: Data files in `data/raw/` can become stale when:
- Feature computation logic changes (new features added, formulas updated)
- Schema changes in the fetch pipeline (new columns, renamed fields)
- Overture/OSM schema updates break assumptions

**Proposed Solution**:

1. **Schema Version in Config**
   ```python
   # config.py
   DATA_SCHEMA_VERSION = "1.0"  # Bump when schema changes
   ```

2. **Version in Meta Files**
   - Store `schema_version` in `.meta.yaml` files when fetching

3. **Version Check on Load**
   - When loading parquet files, verify schema version matches
   - Fail fast with clear error

4. **CLI Integration**
   - `matcher validate-data` - Check all data files for version compatibility
   - `matcher fetch --force` - Refetch even if data exists

---

## Other Ideas

### Multi-dataset Support
- Currently: Overture <-> single local dataset
- Future: Support multiple local datasets, or chained matching

### Active Learning
- Use model uncertainty to prioritize labeling
- Surface candidates the model is least confident about

### Quality Metrics Dashboard
- Track inter-labeler agreement
- Identify systematic disagreements by edge case type

### Bike/Sidewalk Networks

**Status:** Deferred - decide whether to:
1. Label in next round alongside roads
2. Train separate model for non-road networks
3. Wait until road model is more mature

Considerations:
- Bike/sidewalk have different geometry patterns (narrower, more curves)
- Often lack names entirely
- May benefit from geometry-only model approach

### LLM-Assisted Labeling

**Priority:** Low-Medium (exploratory)
**Status:** Idea

**Problem**: Manual labeling is time-consuming and doesn't scale.

**Potential Approaches**:

1. **ASCII Art Visualization**
   ```
   Reference: ===============> (Main St, 150m, heading E)
   Target:    ----------------> (Main Street, 145m, heading E)
              |-- 3m offset --|

   Features: hausdorff=2.1m, name_sim=0.95, class=residential/residential
   ```

2. **Two-Tone Image Encoding**
   - Generate minimal 64x64 or 128x128 images
   - Reference in one color, target in another
   - LLMs with vision can process these

3. **Structured Text Description**
   - Describe geometry relationships in natural language

4. **Coordinate Sequence Encoding**
   - Normalize coordinates to relative units
   - Express as coordinate pairs

**Experiment Plan**:
1. Create 50 sample pairs with known labels
2. Implement ASCII visualization
3. Prompt Claude/GPT to label with explanation
4. Compare to human labels
5. If >85% agreement, consider scaling up

### Improve Geometry-Only Model

**Priority:** Medium
**Status:** Planned

**Problem**: Geometry-only model is worse than full model. Could be improved with:

1. **New features**: Labels collected before `mean_hausdorff_distance` was added don't have it
2. **More no_match examples**: Especially cases with similar names but different geometry
3. **Relabeling pass**: Focus on geometry alignment, ignore names during labeling

---

## Integration: Connectivity-Based Gating

**Priority:** Medium
**Status:** Designed and prototyped (branch: `feature/connectivity-gating-and-debug-logging`)

### Problem

Short segments (< 20m) are currently rejected during integration even if they provide valuable network connectivity. This leads to gaps in the integrated network where small connector segments would bridge disconnected components.

### Proposed Solution

Allow segments below `min_merge_length_m` but above a new `min_connectivity_length_m` threshold if they add network connectivity value.

#### Connectivity Check Logic

A segment "adds connectivity" if it:
1. **Bridges two disconnected components** in the main network, OR
2. **Creates a meaningful shortcut** (existing graph path > `connectivity_path_threshold_m`)

#### Implementation

```python
def _check_adds_connectivity(
    candidate_segments: gpd.GeoDataFrame,
    main_network: gpd.GeoDataFrame,
    tolerance_m: float,
    path_threshold_m: float,
) -> pd.Series:
    """Check if segments add connectivity to the network."""
    # Build graph from main network
    # For each candidate:
    #   1. Find nearest nodes to endpoints
    #   2. Check if bridges disconnected components (no path exists)
    #   3. Check if creates shortcut (existing path > threshold AND > 3x segment length)
```

#### CLI Options

- `--enable-connectivity-gating` (default: False)
- `--min-connectivity-length-m` (default: 5m) - Minimum length when gating applies
- `--connectivity-path-threshold-m` (default: 500m) - Path threshold for shortcut detection

### Related: Debug Logging for Transitive Connectivity

The prototype also includes enhanced diagnostic logging for debugging transitive connectivity issues:

- Log counts and tolerance at start of propagation
- Log top 10 closest orphan distances per hop
- Log distance distribution stats (min, median, max)
- Suggest tolerance adjustments when orphans are within 2x/3x of current tolerance
- Enable with `--debug-connectivity` flag

### Location

`src/matcher/integration/orphan_detector.py`

---

## Known Issues & Technical Debt

### HIGH: Scalability - Large Dataset Support

- **Problem**: `runner.py` uses `geopandas.read_parquet` which loads entire dataset into memory
- **Impact**: Will fail on state-sized or larger datasets
- **Location**: `src/matcher/pipeline/runner.py`
- **Solution**: Migrate to Spark/Sedona/GraphFrames (see research below)

---

## Spark Migration Research (Jan 2026)

**Status**: Research complete, implementation deferred pending labeling improvements

**Summary**: Feature computation pipeline can be migrated from GDF/NetworkX to Spark/Sedona/GraphFrames in 5 stages. This is NOT one large refactor - stages have clear boundaries and can be delivered independently.

### Migration Stages

| Stage | Scope | Sedona/GraphFrames Coverage |
|-------|-------|----------------------------|
| 1. Infrastructure | Session setup, GeoParquet I/O | Direct Sedona support |
| 2. Blocking | STRtree → Sedona spatial join | `ST_Intersects(ST_Buffer(...))` |
| 3. Geometric features | 11 features | 6 direct SQL, 5 Pandas UDFs |
| 4. Topology features | 12 features via GraphFrames | `g.degrees`, `connectedComponents()` |
| 5. Integration | Wire to ML scoring | Collect to driver for XGBoost |

### Key Findings

**Sedona coverage**: ~60% direct SQL equivalents, ~25% Pandas UDFs, ~15% custom
- Direct: `ST_HausdorffDistance`, `ST_Buffer`, `ST_Distance`, `ST_Centroid`, `ST_Length`
- UDFs needed: mean_hausdorff, perpendicular_offset, collinear_gap (vertex sampling)
- Numba works in Pandas UDFs (existing alignment code ported FROM Spark originally)

**GraphFrames coverage**: Most algorithms available
- Direct: `g.degrees`, `connectedComponents()`, `triangleCount()`, `pageRank()`
- Skip: articulation points, edge betweenness (already disabled for >10K nodes/edges)
- Graph construction: `ST_StartPoint`/`ST_EndPoint` → `ST_DWithin` proximity → connected components

**Local dev**: Docker-based Spark recommended to avoid Java dependency issues
- Use `apache/sedona:latest` base image with GraphFrames JAR
- Mount source/data volumes, set `SPARK_DRIVER_MEMORY=8g`

**Cloud deployment** (future): AWS Glue/EMR
- Session factory pattern to abstract local vs Glue SparkSession
- S3 GeoParquet with bbox pushdown
- Sedona/GraphFrames JARs via `--extra-jars`

### Files Affected

**Replace** (wholesale, no legacy paths):
- `blocking/spatial_index.py` → `spark/blocking.py`
- `features/geometric.py` → `spark/features/geometric.py`
- `features/spatial_context.py` → `spark/features/topology.py`
- `pipeline/runner.py` → `spark/pipeline.py`

**Keep**: `config.py` (FEATURE_COLUMNS source of truth), `matching/ml.py` (adapt interface)

### Risks

| Risk | Mitigation |
|------|-----------|
| Sedona function gaps | Pandas UDFs wrapping existing Shapely/Numba code |
| GraphFrames missing algorithms | Skip (already disabled for large graphs) |
| Memory during collect | Filter to scored candidates before driver collect |

Full plan: `/home/brad/.claude/plans/eventual-prancing-koala.md`

### LOW: Datasets with Polygon Geometries (Need Re-fetch)

- **Problem**: Some target datasets have Polygon geometries instead of LineStrings
- **Affected datasets** (files deleted, need re-fetch with correct data):
  - `ca_toronto_roads` - 57,345 Polygons (wrong layer from source?)
  - `co_bogota_bike_network` - 6,082 Polygons (wrong layer from source?)
  - `co_bogota_sidewalks` - 164,868 Polygons (sidewalk polygons, not centerlines)
- **Action needed**: Check source data portals for LineString road centerline layers

### Medium: Robustness Issues

#### Overly Broad Exception Handling
- **Problem**: `except Exception: return None` silently swallows errors
- **Location**: `blocking/spatial_index.py`
- **Solution**: Catch specific exceptions and log warnings

#### Race Condition in Model Selection
- **Problem**: Checks file existence but doesn't validate model is loadable/valid
- **Location**: `ml.py:135-197`
- **Solution**: Attempt to load model to verify validity before selection

#### CRS Validation Gap
- **Problem**: No check for null/invalid geometries after reprojection
- **Location**: `pipeline/runner.py:127-130`
- **Solution**: Validate geometries after CRS transformation

#### Incomplete Config Migration
- **Problem**: TODO indicates `discover-classes` falls back to old config format
- **Location**: `cli.py:1308`
- **Solution**: Complete migration to YAML-only config

### Low Priority

#### Version Tracking Allows None
- **Problem**: Can't distinguish pre-version labels from missing versions
- **Location**: `label_store.py:100-105`
- **Solution**: Use sentinel value or require version

#### Alignment Fraction Tolerance
- **Problem**: 0.01 (1%) may be too loose for some segments
- **Location**: `label_store.py:56-58`
- **Solution**: Make tolerance configurable

#### Memory Inefficiency
- **Problem**: Deduplication after list building, should use set during build
- **Location**: `ml.py:495-498`
- **Solution**: Use set comprehension

---

## Implementation Priority Summary

### Medium Priority

| Feature/Fix | Category | Effort |
|-------------|----------|--------|
| Junction angle distribution | Topology | Medium |
| Local clustering coefficient | Topology | Low |

### Lower Priority (Research)

| Feature/Fix | Category | Effort |
|-------------|----------|--------|
| Fréchet distance | Geometric | High |
| Neighbor consistency (MRF) | Context | High |
| Language-aware names | Semantic | Medium |
| Graph embeddings | Research | High |

---

## References

- **Ruiz-Lendinez et al. (2021)** - "Road Network Conflation Using Semantics and Geometry"
- **Juhasz et al. (2012)** - "Road Network Conflation Based on Iterative Hausdorff Distance Calculation"
- **Volz et al. (2011)** - "Map Conflation Using MRFs"
- **Hootenanny** (open-source conflation tool) - Junction angle distribution algorithms
