# Road Segment Labeling Guide

Guidelines for labeling road segment matches between Overture (reference) and local datasets (target).

## Recommended Approach

### Current Priority: Collecting No-Match Examples

The training data is imbalanced (86% match, only 1.5% no_match). To build a robust model:

1. **Run with expanded buffer** to find more no-match candidates:
   ```bash
   MATCHER_BUFFER_DISTANCE=150 streamlit run src/matcher/labeling/app.py
   ```

2. **Use the "No Match" filter** first - these are rule-based NO_MATCH predictions, easiest to confirm

3. **Then use "Review" filter** - borderline cases, many will be no_match

4. **Target:** ~100 no_match examples (currently have 7)

### General Labeling Tips

- Focus on geometric alignment as the primary indicator
- Name similarity helps but isn't required for a match
- When in doubt, use "unsure" - better than mislabeling
- Use keyboard shortcuts for speed (M, N, I, U, Z)

---

## Label Definitions

| Label | Meaning |
|-------|---------|
| **match** | Same physical road segment (or same road with different segmentation) |
| **no_match** | Different roads, not the same feature (includes related-but-different like sidewalks, split highways) |
| **unsure** | Ambiguous case, needs review |

---

## Edge Cases

### 1. Opposite Carriageways of Split Road

**Scenario:** Both segments are carriageways of the same divided road, but different sides (e.g., eastbound vs westbound).

![Split carriageway example](example_split_carriageway.png)

**Label:** `no_match`

**Reasoning:** They're physically different lanes. There should be another segment in the other dataset that correctly matches. Labeling opposite carriageways as "match" would teach the model to incorrectly conflate parallel roads.

---

### 2. Centerline vs Split Road

**Scenario:** One dataset represents the road as a single centerline, the other has separate segments for each direction.

**Label:** `match`

**Reasoning:** They represent the same physical road, just modeled differently. The model should learn that centerlines can match to split road segments.

---

### 3. Segmentation Mismatch (Partial Overlap)

**Scenario:** Same road, but datasets split it at different points. Segments may only have a small overlap.

![Segmentation mismatch example](example_segmentation_mismatch.png)

**Label:** `match`

**Reasoning:** They represent the same physical road corridor. The model will learn from features like `overlap_ratio` and `length_ratio` that partial overlaps can still be matches. The goal is "do these represent the same road?" not "do they have identical geometry?"

---

### 4. Road vs Adjacent Sidewalk

**Scenario:** A road segment is being compared to a sidewalk/footpath that runs parallel to it.

**Label:** `no_match`

**Reasoning:** They're spatially related but not the same feature. For a binary match classifier, we want the model to reject these as non-matches. Teaching the model to distinguish parallel-but-different features (sidewalks, bike lanes, service roads) is valuable training signal.

**Note:** The labeling app filters out non-road classes (footway, cycleway, pedestrian, path, steps, track) from Overture by default to reduce these cases.

---

### 5. Same Name, Different Road

**Scenario:** Two segments share a name but are clearly different roads (e.g., two different "Main Street" segments in different locations).

**Label:** `no_match`

**Reasoning:** Name similarity alone doesn't make a match. Geometric features should differentiate these cases.

---

### 6. Different Names, Same Road

**Scenario:** Segments are clearly the same road but have different names (e.g., name missing in one dataset, or alternate name used).

**Label:** `match`

**Reasoning:** Geometric alignment is the primary indicator. The model should learn that name mismatch doesn't preclude a match.

---

## Future Enhancements

### Sub-segment Matching

**Problem:** Whole-segment labeling can't capture cases where only a portion of each segment actually matches (e.g., segmentation mismatch, partial overlaps).

**Solution:** The labeling UI now supports sub-segment selection:
- Enable "Subsegment" checkbox to activate sub-segment mode
- Estimate is auto-applied based on endpoint projection
- Use sliders to fine-tune the matching portions (0-100% for each segment)
- Labels store `ref_start_pct`, `ref_end_pct`, `target_start_pct`, `target_end_pct`

**When to use:** When segments clearly represent the same road but have different extents. Specify which portion of each segment actually overlaps.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| M | Match |
| N | No Match |
| U | Unsure |
| Z | Undo |
| Arrow Left/Right | Navigate |
