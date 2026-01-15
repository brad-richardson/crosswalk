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
**Status:** Needs investigation

### Problem

Class similarity has low feature importance and may not be working correctly. Suspect the class values aren't being transformed/normalized properly from the original datasets.

### TODO

1. Check Overture class values vs Boston class values
2. Verify the class hierarchy mapping in `semantic.py`
3. May need dataset-specific class normalization

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
**Status:** Research needed

### Problem

Sidewalks on opposite sides of a street have nearly identical:
- Geometry (parallel lines, same length)
- Heading (same direction)
- Class (both footway/sidewalk)
- Name (if inherited from adjacent road)

Current features cannot disambiguate between them. This leads to false positive matches where left-side sidewalk matches right-side sidewalk.

### "Side of Street" as a Feature

The key differentiator is **topological relationship to the road network**:
- Left sidewalk is connected to road network from one side
- Right sidewalk is connected from the opposite side
- At intersections, they connect to different corners

### Potential Approaches

1. **Offset Direction from Road Centerline**
   - For each sidewalk, find nearest road segment
   - Compute perpendicular offset direction (left vs right of road)
   - Encode as signed distance or binary left/right
   - Challenge: Requires road network as context, not just segment pairs

2. **Intersection Connectivity Signature**
   - At endpoints, which other segments connect?
   - Build a local "graphlet" (small subgraph pattern)
   - Encode connectivity pattern as feature vector
   - Same-side sidewalks connect to same intersection corners

3. **Cross-Street Segment Detection**
   - Look for perpendicular segments (crosswalks) that connect sidewalks
   - Sidewalks connected by crosswalk are on opposite sides
   - Could use as hard constraint in matching

4. **Relative Position to Named Road**
   - If sidewalk is associated with "Main St", compute offset
   - Left-side = negative offset, right-side = positive offset
   - Requires road-sidewalk association (from network or spatial proximity)

5. **Bearing Delta to Nearest Road**
   - Sidewalk heading vs road heading
   - Plus perpendicular offset direction
   - Combined gives unique "signature" for each side

### Implementation Notes

- May require loading road network as context during matching
- Could be expensive to compute for all candidates
- Consider computing only for high-confidence geometric matches that need disambiguation
- Overture has `road_surface` and connector data that might help

### Related Work

- OSM models sidewalks with `sidewalk=left|right|both` on road ways
- Some datasets associate sidewalks with their parent road via attributes
- Graph neural networks for road network embedding (but may be overkill)
