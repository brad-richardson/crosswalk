# Future Features & Enhancements

Tracked ideas for future development.

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

---

## Automatic Model Routing

**Priority:** High
**Status:** Planned

### Problem

The full model relies heavily on name similarity (47% feature importance). For datasets without names/attributes, it performs poorly.

### Solution

Implement automatic routing based on attribute availability:

```python
def get_matcher(has_names: bool, has_classes: bool) -> MLMatcher:
    if has_names:
        return MLMatcher("matcher_model.joblib")  # Full model
    else:
        return MLMatcher("matcher_model_geom_only.joblib")  # Geometry-only
```

### Models Available

- `matcher_model.joblib` - Full model with names (85.6% accuracy)
- `matcher_model_geom_only.joblib` - Geometry-only (81.4% accuracy)

---

## Class Similarity Investigation

**Priority:** Medium
**Status:** Addressed (Jan 2026)

### Problem

Class similarity has low feature importance and may not be working correctly. Suspect the class values aren't being transformed/normalized properly from the original datasets.

### Resolution

1. Expanded `ROAD_CLASS_HIERARCHY` in `semantic.py` to include:
   - Link roads (motorway_link, trunk_link, etc.)
   - Pedestrian/bike infrastructure (footway, sidewalk, cycleway, path, etc.)
   - Living streets, pedestrian areas
2. Updated default rank for unknown classes (now 6/residential instead of 5)
3. Added `get_class_info()` diagnostic function to help debug class value issues

Low feature importance may be inherent to the training data (most matches are same-class anyway). The YAML dataset configs help ensure consistent class normalization for new datasets.

---

## Improve Geometry-Only Model

**Priority:** Medium
**Status:** Planned

### Problem

Geometry-only model (81.4%) is 4% worse than full model. Could be improved with:

1. **New features**: Labels collected before `mean_hausdorff_distance` was added don't have it
2. **More no_match examples**: Especially cases with similar names but different geometry
3. **Relabeling pass**: Focus on geometry alignment, ignore names during labeling

---

## Other Ideas

### Multi-dataset Support
- Currently: Overture ↔ single local dataset
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

---

## Graphlet Signature / Topology Features

**Priority:** Medium-High (for sidewalk/bike matching)
**Status:** Partially addressed (Jan 2026)

### Problem

Sidewalks on opposite sides of a street have nearly identical:
- Geometry (parallel lines, same length)
- Heading (same direction)
- Class (both footway/sidewalk)
- Name (if inherited from adjacent road)

Current features cannot disambiguate between them. This leads to false positive matches where left-side sidewalk matches right-side sidewalk.

### Implemented: Lateral Offset Feature

Added `lateral_offset` and `lateral_offset_consistency` features in `ml.py`:
- Measures perpendicular distance between candidate segments
- Same-side sidewalks have low lateral offset (< 5m typically)
- Opposite-side sidewalks have high lateral offset (10-30m, road width)
- Uses `compute_perpendicular_offset()` from `relational.py`

### "Side of Street" as a Feature

The key differentiator is **topological relationship to the road network**:
- Left sidewalk is connected to road network from one side
- Right sidewalk is connected from the opposite side
- At intersections, they connect to different corners

### Potential Further Approaches

1. **Offset Direction from Road Centerline** (infrastructure exists in `relational.py`)
   - `AnchorRoadMatcher` already finds anchor roads
   - `compute_side_of_street()` returns left/right/unknown
   - Challenge: Need to compare sides for both reference AND target

2. **Intersection Connectivity Signature**
   - At endpoints, which other segments connect?
   - Build a local "graphlet" (small subgraph pattern)
   - Encode connectivity pattern as feature vector
   - Same-side sidewalks connect to same intersection corners

3. **Cross-Street Segment Detection**
   - Look for perpendicular segments (crosswalks) that connect sidewalks
   - Sidewalks connected by crosswalk are on opposite sides
   - Could use as hard constraint in matching

### Implementation Notes

- `lateral_offset` should handle most cases (geometric difference is usually clear)
- More complex topology features may be needed for ambiguous cases
- Overture has `road_surface` and connector data that might help

### Related Work

- OSM models sidewalks with `sidewalk=left|right|both` on road ways
- Some datasets associate sidewalks with their parent road via attributes
- Graph neural networks for road network embedding (but may be overkill)

---

## LLM-Assisted Labeling

**Priority:** Low-Medium (exploratory)
**Status:** Idea

### Problem

Manual labeling is time-consuming and doesn't scale. Could LLM agents (Claude Code, Gemini, Codex) help with labeling road segment matches?

### Challenge

LLMs are text-based and don't natively understand geometry. Need a way to represent road geometries in a format LLMs can reason about.

### Potential Approaches

1. **ASCII Art Visualization**
   - Project segment pairs to local coordinate system
   - Render as ASCII art showing relative positions
   - Include distance markers and direction indicators
   ```
   Reference: ═══════════════► (Main St, 150m, heading E)
   Target:    ────────────────► (Main Street, 145m, heading E)
              └── 3m offset ──┘

   Features: hausdorff=2.1m, name_sim=0.95, class=residential/residential
   ```

2. **Two-Tone Image Encoding**
   - Generate minimal 64x64 or 128x128 images
   - Reference in one color, target in another
   - Include a north arrow and scale bar
   - LLMs with vision can process these

3. **Structured Text Description**
   - Describe geometry relationships in natural language
   - "Target segment runs parallel to reference, 3m to the left, same direction"
   - Include computed feature values for context

4. **Coordinate Sequence Encoding**
   - Normalize coordinates to relative units
   - Express as coordinate pairs: "ref: (0,0)→(100,0); target: (2,3)→(98,5)"
   - LLMs can reason about simple coordinate geometry

### Implementation Notes

- Could be a CLI command: `matcher label-llm --batch 50 --output labels.csv`
- Would need prompt engineering to teach LLM the task
- Human review of LLM labels before training
- Could use confidence scores to route uncertain cases to humans

### Benefits

- Scale labeling without hiring annotators
- 24/7 availability
- Consistent application of labeling guidelines
- Could explain reasoning for each label

### Concerns

- LLM accuracy on geometric reasoning
- Cost at scale (API calls)
- Need validation against human labels
- May not handle edge cases well

### Experiment Plan

1. Create 50 sample pairs with known labels
2. Implement ASCII visualization
3. Prompt Claude/GPT to label with explanation
4. Compare to human labels
5. If >85% agreement, consider scaling up

---

## Data File Versioning / Schema Checksums

**Priority:** Medium
**Status:** Idea

### Problem

Data files in `data/raw/` can become stale when:
- Feature computation logic changes (new features added, formulas updated)
- Schema changes in the fetch pipeline (new columns, renamed fields)
- Overture/OSM schema updates break assumptions

Currently there's no automated way to detect when data files need to be refetched. This leads to:
- Subtle bugs from schema mismatches
- Wasted debugging time when old data causes failures
- Training on stale features

### Proposed Solution

Add a versioning system to data files:

1. **Schema Version in Config**
   ```python
   # config.py
   DATA_SCHEMA_VERSION = "1.0"  # Bump when schema changes
   ```

2. **Version in Meta Files**
   - Store `schema_version` in `.meta.yaml` files when fetching
   - Example:
     ```yaml
     fetched_at: "2026-01-23T15:30:00Z"
     schema_version: "1.0"
     feature_count: 125769
     ```

3. **Version Check on Load**
   - When loading parquet files, verify schema version matches
   - Fail fast with clear error: "Data file us_boston_streets.parquet has schema v0.9, expected v1.0. Please refetch."

4. **CLI Integration**
   - `matcher validate-data` - Check all data files for version compatibility
   - `matcher fetch --force` - Refetch even if data exists

### Benefits

- Early failure with clear messaging
- Prevents subtle bugs from stale data
- Documents when schema changes require data refresh

### Alternatives Considered

- Git LFS for data versioning (too heavy for this use case)
- Checksums of entire files (doesn't capture semantic changes)
- DVC (overkill for current project size)
