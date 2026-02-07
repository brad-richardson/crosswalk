# Research Ideas

Exploratory feature ideas and research directions for the road network matcher. These are longer-term investigations that haven't been prioritized for implementation.

For tried-and-removed features, see [RESEARCH_GRAVEYARD.md](RESEARCH_GRAVEYARD.md).
For actionable backlog items, see [TODO.md](../TODO.md).

---

## Potential Feature Ideas

### Fréchet Distance
- **Feature**: `frechet_distance_m`
- **Purpose**: Order-preserving distance metric that considers both position and traversal order
- **Critical issue**: Direction-sensitive. Datasets with inconsistent digitization direction (common) give poor scores for identical geometry. Would require `min(frechet(A, B), frechet(A, reverse(B)))` which doubles cost.
- **Why Hausdorff is preferred**: Direction-agnostic by design
- **Priority**: Low

### Route Number Normalization
- **Feature**: `route_number_similarity`
- **Purpose**: Normalize "I-5", "Interstate 5", "I 5" to canonical form; compare shield prefixes
- **Note**: `name_numeric_match` and `route_prefix_match` partially address this already

### Reference/Alt Name Token Overlap
- **Feature**: `ref_alt_name_overlap`
- **Purpose**: Token overlap between ref/alt_name fields when primary names don't match

### Attribute Features (Lane Count, Speed Limit, etc.)
- Lane count similarity, speed limit difference, surface/bridge/tunnel flags
- Data is already fetched for some of these (`oneway_lr`, `speed_limit_kph_lr` columns)
- Previous ablation showed oneway/speed features hurt model performance (see RESEARCH_GRAVEYARD.md)
- May need different formulation or more training data

---

## Topology Research

### Network Continuity Score
- Penalize matches that would create disconnected subgraphs
- Requires global graph analysis (lower priority)

### K-Hop Path Continuity
- Check if segments continue into matching neighbors at k hops
- Academic basis: Hootenanny network-style propagation

### Seed-and-Grow Neighbor Agreement (MRF)
- Reinforce confidence when nearby segments also have strong matches
- Start from high-confidence "seed" matches; propagate scores to neighbors
- Academic basis: Hootenanny-style propagation, Volz et al. MRF conflation
- Requires iterative post-processing

### Enhanced Degree Signature Similarity
- Current implementation compares raw degree tuples
- Proposed: Use Earth Mover's Distance (EMD) for more nuanced comparison

### Graphlet Performance Investigation
- Full graphlet features enabled Jan 2026 (degree, triangles, squares, clustering, two_hop_count, is_articulation)
- CV F1 improved slightly (0.905 vs 0.901), holdout metrics unchanged (99.6%)
- Open questions: Which components are most predictive? Does impact vary by dataset type?
- Suggested: Feature importance analysis, ablation study, per-dataset breakdown

---

## Graph Embeddings

**Priority:** Low (exploratory)

### Node2Vec / GraphSAGE Embeddings
- Train embeddings on road network; use as feature vectors for segment context
- Could use PyTorch Geometric or DGL
- Pre-train on OSM or Overture network

### Siamese GNN
- Learn to directly compare two segments' network contexts via shared GNN
- Academic basis: Graph neural networks for entity matching

**Note**: May be overkill for current dataset sizes.

---

## Spark Migration Research (Jan 2026)

**Status**: Research complete, implementation deferred

Feature computation pipeline can be migrated from GDF/NetworkX to Spark/Sedona/GraphFrames in 5 stages:

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

**GraphFrames coverage**: Most algorithms available
- Direct: `g.degrees`, `connectedComponents()`, `triangleCount()`, `pageRank()`
- Skip: articulation points, edge betweenness (already disabled for >10K nodes/edges)

**Local dev**: Docker-based Spark (`apache/sedona:latest` with GraphFrames JAR)

**Cloud deployment**: AWS Glue/EMR with session factory pattern

### Files Affected

**Replace**: `blocking/spatial_index.py`, `features/geometric.py`, `features/spatial_context.py`, `pipeline/runner.py`
**Keep**: `config.py` (source of truth), `matching/ml.py` (adapt interface)

### Risks

| Risk | Mitigation |
|------|-----------|
| Sedona function gaps | Pandas UDFs wrapping existing Shapely/Numba code |
| GraphFrames missing algorithms | Skip (already disabled for large graphs) |
| Memory during collect | Filter to scored candidates before driver collect |
