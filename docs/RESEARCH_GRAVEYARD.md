# Research Graveyard

Features that were implemented, tested, and removed or parked due to insufficient contribution to model performance.

---

## Junction Angle Similarity (Removed 2026-02-03)

**Commit:** `81c0d55`

### What It Was
Three features comparing the geometric pattern of junctions (inter-edge angles) at segment endpoints:
- `junction_angle_similarity_from` - Similarity at 'from' endpoint
- `junction_angle_similarity_to` - Similarity at 'to' endpoint
- `junction_angle_similarity_avg` - Average of from and to

### Implementation
1. At each endpoint node, computed bearing to each neighboring node
2. Sorted bearings and computed angles between consecutive edges (junction signature)
3. Compared signatures using:
   - Direct element-wise comparison for same-degree junctions
   - Histogram comparison (30° bins) for different degrees
4. Tried both forward/reverse orientations, picked best match

### Ablation Results
| Feature | F1 Delta | Classification |
|---------|----------|----------------|
| `junction_angle_similarity_from` | -0.00001% | redundant (noise-level) |
| `junction_angle_similarity_to` | -0.00001% | redundant (noise-level) |
| `junction_angle_similarity_avg` | -0.00001% | redundant (noise-level) |

### Why Removed
- **Negligible contribution**: F1 delta was essentially zero (within measurement noise)
- **Moderate computation cost**: Required graph traversal at each endpoint, bearing calculations, angle sorting
- **Poor ROI**: The complexity didn't justify the ~0% improvement
- **Existing coverage**: Existing topology features (`degree_match_score`, `degree_signature_similarity`) already capture junction complexity at lower cost

### Lessons Learned
1. Degree-based features may already capture junction patterns sufficiently
2. Angle-based features add geometric precision but at computation cost
3. In urban networks with regular grids, junction angles are often similar (90°) making comparison less discriminative

---

## Spectral Embedding Similarity (Parked 2026-01-27)

**Branch:** `feature/spectral-embeddings` (commit `0628bde`, never merged)

### What It Was
A single feature using Laplacian eigenvector embeddings to capture global network structure:
- `spectral_embedding_similarity`

### Implementation
1. Computed normalized graph Laplacian for ref and target networks
2. Extracted 32 smallest non-trivial eigenvectors using `scipy.sparse.linalg.eigsh`
3. Used algebraic trick (eigsh on A_norm with `which='LM'`) for ~14x speedup over naive approach
4. Compared embedding vectors between matched endpoints

### Ablation Results
| Feature | Importance Rank | Importance Score |
|---------|-----------------|------------------|
| `spectral_embedding_similarity` | #43/57 | 0.0045 |

### Why Parked
- **Low importance**: Ranked near bottom of feature importance
- **High computation cost**: Eigenvector decomposition is expensive, even with optimizations
- **Marginal benefit**: Global network structure less useful than local topology for segment matching

### Lessons Learned
1. Local topology features (graphlets, degree signatures) outperform global embeddings for segment-level matching
2. Spectral methods may be more useful for graph-level tasks (entire network comparison) rather than element-level matching
3. The 14x speedup wasn't enough to justify the still-expensive computation for minimal gain

---

## Zero-Importance Features (Removed 2026-01-25)

**Commit:** `890c34b`

### What They Were
Five features removed after showing zero or near-zero importance in model evaluation:
- `projection_distance_m` - Distance along projection axis
- `cardinal_direction_mismatch` - 1.0 if N/S or E/W conflict in street names
- `length_bin_ref` - Categorical length bin for reference segment
- `length_bin_target` - Categorical length bin for target segment
- `length_bin_match` - Whether length bins match

### Why Removed
- **Zero feature importance**: XGBoost assigned effectively zero importance to all five features
- **Redundancy**: `projection_distance_m` redundant with other distance metrics; length bins redundant with continuous `length_ratio`
- **Rare signal**: `cardinal_direction_mismatch` rarely triggered (most streets don't have cardinal prefixes)

### Lessons Learned
1. Categorical binning of continuous features (length bins) often loses information vs. keeping the continuous version
2. Features based on rare string patterns (cardinal directions) may not have enough signal
3. Multiple distance metrics can be redundant - keep the most discriminative ones

---

## Traffic Tier Features (Removed 2026-01-31)

**Commit:** `3886dcd`

### What They Were
Two features for comparing road classification compatibility:
- `same_traffic_tier` - 1.0 if same tier (vehicle/pedestrian/cycling), 0.0 otherwise, 0.5 if unknown
- `tier_incompatible` - 1.0 if vehicle↔pedestrian mismatch, 0.0 otherwise

### Implementation
Road classes grouped into tiers:
- **Vehicle**: motorway, trunk, primary, secondary, tertiary, residential, service, unclassified
- **Pedestrian**: footway, path, pedestrian, steps
- **Cycling**: cycleway

### Why Removed
- **Backfill required**: Features weren't in existing labels, would need full backfill
- **Uncertain value**: `class_similarity` already captures class relationships
- **Deferred**: Functions kept in `semantic.py` for potential future use

### Status
Parked in TODO.md - may revisit if class_similarity proves insufficient for cross-tier matching scenarios.

---

## Overlap Ratio (Removed 2026-01-23)

**Commit:** `fe4540d`

### What It Was
- `overlap_ratio` - Ratio of geometric overlap between candidate segments

### Why Removed
- **Always 1.0**: Due to blocking/candidate generation, all candidates already have significant geometric overlap
- **No discriminative power**: Feature had zero variance in practice

### Lessons Learned
1. Features derived from blocking criteria are often useless - the blocking already filtered for them
2. Always check feature variance before adding to model

---

## Template for Future Entries

### Feature Name (Removed/Parked YYYY-MM-DD)

**Commit/Branch:** `hash` or `branch-name`

#### What It Was
Brief description of the feature(s).

#### Implementation
How it was computed.

#### Ablation Results
| Feature | F1 Delta / Importance | Classification |
|---------|----------------------|----------------|
| `feature_name` | X.XX% | classification |

#### Why Removed
- Reason 1
- Reason 2

#### Lessons Learned
- Insight 1
- Insight 2
