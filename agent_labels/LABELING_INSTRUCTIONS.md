# Transportation Segment Matching: Labeling Context for AI Agents

This document provides context for AI agents tasked with labeling transportation segment matching candidates.

## Task Overview

You are labeling whether two segments from different datasets represent the **same physical feature**.

- **Reference segment**: From Overture Maps (a standardized global transportation dataset)
- **Target segment**: From a local municipal dataset (e.g., Boston Streets, Boston Sidewalks, Boston Bike Network)

**Important**: These datasets include various transportation feature types:
- Roads and streets (vehicle traffic)
- Sidewalks and footpaths (pedestrian)
- Bike lanes and cycle tracks (bicycle)
- Park paths and trails
- Service roads and alleys

Your job is to determine if two segments represent the **same physical feature** - not just same location, but same feature type. A road and an adjacent sidewalk are NOT a match even if they run parallel.

## Label Definitions

### `match`
The reference and target segments represent the **same physical feature**.

Use `match` when:
- The geometries clearly align (following the same path)
- They are the same feature type (road-to-road, sidewalk-to-sidewalk, bike path-to-bike path)
- Any name differences are minor (abbreviations, spelling variations)
- The segments may have different lengths (segmentation differences)
- One segment may be a subset of the other

### `no_match`
The segments represent **different features**.

Use `no_match` when:
- Different feature types (road vs sidewalk, road vs bike path, etc.)
- The segments are on parallel but separate features (e.g., opposite sides of a divided highway)
- One is a sidewalk/path and the other is a road centerline
- The segments are clearly offset from each other
- They represent intersecting features (perpendicular)

### `unsure`
You cannot confidently determine if they match.

Use `unsure` when:
- The images are unclear or missing
- The geometry is ambiguous (could be either)
- Edge cases where reasonable people would disagree

## Decision Framework

### Primary: Geometric Alignment

The most important factor is whether the geometries align:

1. **Overlay alignment**: In the satellite image, do the blue (reference) and red (target) lines follow the same path?
2. **Offset**: A few meters of offset is acceptable (GPS/digitization error). Large offsets (>10m) suggest different roads.
3. **Direction**: Lines should follow the same direction. Perpendicular lines are NOT matches.

### Secondary: Name Matching

Names provide supporting evidence but are NOT decisive:

- **Same name**: Strong evidence of match, but verify geometry
- **Similar names**: "Main St" vs "MAIN STREET" - still a match if geometry aligns
- **Different names**: Could still be a match (one dataset may be outdated or use different naming)
- **Missing names**: Many roads are unnamed - rely on geometry

### Tertiary: Road Class

Road classes provide weak evidence:

- Similar classes (residential ↔ local street) support match
- Different classes don't preclude match (datasets classify differently)

## Common Edge Cases

### 1. Segmentation Differences and Overlap Threshold

**Core principle**: Do these two geometries cover the **same physical movement space** for at least **10% of either segment's length**?

**Scenario**: Reference is 200m long, target is 80m long, but they overlap perfectly for 80m.

**Decision**: `match` - Datasets segment roads differently. The overlap (80m = 40% of reference, 100% of target) far exceeds the 10% threshold. The goal is "do these represent the same physical space?" not "do they have identical geometry?"

**Overlap threshold examples**:
- 200m ref, 80m target, 80m overlap → `match` (80m is 40% of ref, 100% of target)
- 200m ref, 200m target, 15m overlap → `no_match` (15m is only 7.5% of either)
- 100m ref, 50m target, 10m overlap → `match` (10m is 10% of ref, 20% of target)

**Adjacent but non-overlapping segments**: If two segments are end-to-end consecutive on the same road but don't actually overlap spatially, they are `no_match`. They represent different physical spans of the road.

### 2. Opposite Carriageways of Split Road

**Scenario**: A divided highway with separate lanes. Reference is the southbound lane, target is the northbound lane. Both are carriageways of the same conceptual road, but on different sides.

**Decision**: `no_match` - These are physically different lanes. There should be another segment in the other dataset that correctly matches. Labeling opposite carriageways as "match" would teach the model to incorrectly conflate parallel roads.

### 3. Centerline vs Split Road

**Scenario**: One dataset represents the road as a single centerline down the middle, the other dataset has separate segments for each direction (split carriageways).

**Decision**: `match` - They represent the same physical road, just modeled differently. The model should learn that centerlines can match to split road segments. This is different from case #2 because here one segment is the centerline, not another carriageway.

### 4. Road vs Adjacent Sidewalk

**Scenario**: A road segment is being compared to a sidewalk/footpath that runs parallel to it.

**Decision**: `no_match` - Sidewalks are physically separate features—different surface, different grade (raised curb). They warrant their own stable ID in Overture/OSM. Match sidewalks to footway segments, not roads.

### 5. Bike Lanes and Cycle Facilities

**Core principle**: Would Overture/OSM create a new feature with a stable ID for this, or is it an attribute on the road?

| Facility Type | Same feature as road? | Label vs Road | Label vs Cycleway |
|---------------|----------------------|---------------|-------------------|
| Painted bike lane (sharrows, bike arrows) | Yes - same surface/grade | `match` | `no_match` |
| Flexpost-separated lane (same pavement) | Yes - still same surface | `match` | `no_match` |
| Buffered lane with paint only | Yes - same surface | `match` | `no_match` |
| Raised/curbed bike lane | No - different grade | `no_match` | `match` |
| Separated cycle track | No - physically separate | `no_match` | `match` |

**How to decide**:
- `cycleway=lane` in OSM → attribute on road way (same feature) → `match` to road
- `highway=cycleway` in OSM → separate way (new feature) → `match` to cycleway, `no_match` to road

**When facility type is unknown**: If you can't determine from the data or satellite whether a bike facility is painted (same surface) or separated (different surface), use `unsure`. This prevents incorrect class relationships.

**Check satellite imagery**:
- Same asphalt, same grade → same feature as road → `match` to road
- Raised, curbed, or separate surface → different feature → `match` to cycleway, `no_match` to road

### 6. Service Roads / Alleys

**Scenario**: Target is a small service road parallel to the main reference road.

**Decision**: `no_match` - Service roads are distinct from main roads.

### 7. Different Digitization Accuracy

**Scenario**: Lines mostly align but have 3-5m offset, consistent along the length.

**Decision**: `match` - This offset is typical digitization error. As long as they follow the same path, it's a match.

### 8. Same Name, Different Road

**Scenario**: Two segments share a name but are clearly different roads (e.g., two "Main Street" segments in different locations).

**Decision**: `no_match` - Name similarity alone doesn't make a match. Geometric features should differentiate these cases.

### 9. Intersection Overlap

**Scenario**: Reference ends at an intersection, target starts at the same intersection but continues in a different direction.

**Decision**: `no_match` - They share an endpoint but represent different road segments.

## Reading the Metadata YAML

Each candidate has a `metadata.yaml` file with:

```yaml
candidate:
  ref_id: "overture_abc123"      # Reference segment ID
  target_id: "boston_12345"       # Target segment ID
  dataset: "boston_streets"       # Source dataset

names:
  reference: "Main Street"        # Reference road name (may be null)
  target: "MAIN ST"               # Target road name (may be null)

classes:
  reference: "residential"        # Overture road class
  target: "Local Street"          # Local dataset class

ml_prediction:
  decision: "review"              # ML model's decision
  confidence: 0.72                # ML confidence (0-1)

geometry:
  ref_length_m: 142.5             # Reference length in meters
  target_length_m: 138.2          # Target length in meters
  bbox: [-71.08, 42.35, ...]      # Bounding box [minx, miny, maxx, maxy]

features:
  geometric:
    hausdorff_distance: 8.5       # Max distance between geometries (lower = better)
    buffer_iou: 0.89              # Overlap quality (higher = better)
    overlap_ratio: 0.94           # How much overlaps (higher = better)
    heading_delta: 2.1            # Direction difference in degrees (lower = better)
    length_ratio: 0.97            # Length similarity (closer to 1.0 = better)
  semantic:
    name_levenshtein: 0.85        # Name similarity (higher = better)
    name_jaro_winkler: 0.92       # Name similarity (higher = better)
  topological:
    degree_match_score: 0.88      # Endpoint connectivity match (higher = better)
```

### Key Metrics to Consider

| Metric | Good Match | Likely No Match |
|--------|------------|-----------------|
| `buffer_iou` | > 0.7 | < 0.3 |
| `overlap_ratio` | > 0.8 | < 0.3 |
| `hausdorff_distance` | < 15m | > 50m |
| `heading_delta` | < 10° | > 45° |

## Reading the Images

### Satellite Image (`satellite.png`)
- **Blue line (solid)**: Reference segment (Overture)
- **Red line (dashed)**: Target segment (local data)
- Background: Aerial/satellite imagery

### Geometry Image (`geometry.png`)
- Same color coding, white background
- Cleaner view without satellite imagery noise
- Use this when satellite is unclear

## Output Format

Provide your labels as a CSV with these columns:

```csv
ref_id,target_id,label,confidence,reasoning
overture_abc123,boston_12345,match,0.95,"Lines clearly overlap, same road"
overture_def456,boston_67890,no_match,0.90,"Parallel roads, 15m offset"
```

- `label`: One of `match`, `no_match`, `unsure`
- `confidence`: Your confidence in the label (0.0 to 1.0)
- `reasoning`: Brief explanation (1 sentence)

## Tips for Consistent Labeling

1. **Geometry first**: Always start with the images. If geometries clearly align, it's likely a match.
2. **Trust your eyes**: ML metrics are helpful but not definitive. Visual inspection is most reliable.
3. **When in doubt, use `unsure`**: It's better to mark ambiguous cases for human review.
4. **Be consistent**: Apply the same standards to all candidates.
5. **Consider the road type**: Highway segments need very precise alignment; residential streets have more tolerance.
