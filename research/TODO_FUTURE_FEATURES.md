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
