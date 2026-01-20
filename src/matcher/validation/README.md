# Road Existence Validation - Research Document

## Status: Research/Documentation Only

This document captures research on road existence validation approaches for future implementation. No immediate implementation planned.

**Decision**: When implemented, validation scores should be added as columns (merge with warning) rather than auto-filtering, allowing downstream systems to make filtering decisions.

---

## Overview

Validation checks to verify that roads actually exist before merging them into the integrated network. Two complementary approaches:

1. **Address-based validation** - Check if nearby Overture addresses have matching street names
2. **Satellite imagery validation** - Use imagery-derived road detection to confirm road existence

---

## Current Architecture Context

The integration pipeline (`src/matcher/integration/pipeline.py`) has 6 stages:
```
Stage 1: Load reference network
Stage 2: Load and prepare target datasets (includes filter_short_segments, detect_near_duplicates)
Stage 3: Combine networks with provenance
Stage 4: Detect orphans by proximity
Stage 5: Build result structure
Stage 6: Write outputs
```

**Best insertion point**: Between Stage 2 (after filters) and Stage 3 (before combining), or as a new filter in Stage 2.

---

## Approach 1: Address-Based Validation

### Concept
Use Overture's address dataset to validate road names. If a road segment has nearby addresses with matching street names, it's more likely to be a real road.

### Implementation

**New module**: `src/matcher/validation/address_validator.py`

```python
from pyproj import Geod
import geopandas as gpd
from rapidfuzz import fuzz

def validate_roads_by_addresses(
    roads_gdf: gpd.GeoDataFrame,
    addresses_gdf: gpd.GeoDataFrame,
    buffer_m: float = 50.0,
    name_threshold: float = 0.6,
) -> gpd.GeoDataFrame:
    """
    Validate road segments by checking for nearby addresses with matching names.

    Returns roads_gdf with added columns:
    - address_match_count: Number of nearby addresses with matching street name
    - address_validation_score: Confidence score (0-1)
    """
```

### Data Requirements
- Overture addresses dataset (fetch via DuckDB or similar)
- Road segments with `name` column

### Key Differences from PySpark Version
- Use GeoPandas/Shapely instead of Sedona (matches existing codebase)
- Use rapidfuzz instead of PySpark's levenshtein (faster, more options)
- Use STRtree for spatial indexing (existing pattern in codebase)

---

## Approach 2: Satellite Imagery Validation

### Available Options (Ranked by Feasibility)

| Option | Imagery Source | Detection | Cost | Effort |
|--------|---------------|-----------|------|--------|
| **A. SAM3 (Segment Anything 3)** | Any imagery | Text-prompted ("paved road") | $0 | Medium |
| **B. EODT4Crises API** | User-provided | SAM-based (free) | $0 | Low |
| **C. Overture Building Footprints** | Overture | Proxy (no roads between buildings) | $0 | Low |
| **D. NAIP + Custom Model** | NAIP (US only) | D-LinkNet/RoadFormer | $0-500 | High |
| **E. OpenAerialMap + Model** | OAM (spotty coverage) | Custom | $0 | High |
| **F. Nearmap API** | Nearmap | Pre-built | $1000+/mo | Low |

### Promising: SAM3 (Segment Anything with Concepts)

**SAM3** (github.com/facebookresearch/sam3) is Meta's foundation model for segmentation:
- **Text-prompted segmentation**: Query with "paved road", "highway", "dirt road", etc.
- **848M parameters**: DETR detector + SAM2 tracker architecture
- **75-80% human performance** on concept recognition benchmark
- **Interactive refinement**: Can add point/box corrections

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
image = Image.open("satellite_tile.jpg")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="paved road")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

**Advantages for road validation:**
- No fine-tuning needed - works out of the box with text prompts
- Can distinguish road types ("highway" vs "residential street" vs "dirt road")
- Supports interactive refinement for edge cases
- Combines well with any imagery source (NAIP, OAM, Sentinel)

### Other Options

**Option B: EODT4Crises** (ESA-funded, free)
- Submit imagery, get road vectors back
- Uses SAM-based model internally
- Best for spot-checking specific areas

**Option C: Overture Buildings as Proxy** (Simplest)
- If a "road" passes through multiple building footprints, it's likely invalid
- Already have Overture data access
- Quick to implement, catches obvious errors

---

## Proposed Implementation Phases

### Phase 1: Address Validation (MVP)
1. Create `src/matcher/validation/` module
2. Implement `address_validator.py` with:
   - `fetch_addresses_for_bbox()` - Get Overture addresses
   - `validate_by_address_proximity()` - Core validation logic
3. Add as optional filter in integration pipeline
4. Output: `address_match_count`, `address_validation_score` columns

### Phase 2: Building Footprint Check
1. Implement `building_validator.py`:
   - `fetch_buildings_for_bbox()` - Get Overture buildings
   - `check_building_intersection()` - Flag roads crossing buildings
2. Add to validation module
3. Output: `building_intersection_count`, `building_validation_pass` columns

### Phase 3: Satellite Imagery with SAM3 (Future)
1. Set up SAM3 model (848M params, needs GPU)
2. Implement tile-based imagery fetching (NAIP for US, Sentinel for global)
3. Run SAM3 with text prompt "paved road" per tile
4. Cache segmentation masks by tile coordinates
5. Compare road candidates against imagery masks:
   - Compute overlap between road geometry and detected road pixels
   - Score based on coverage percentage
6. Add `imagery_validation_score` column to output

**SAM3 Pipeline Sketch:**
```python
def validate_by_imagery(roads_gdf, bbox, imagery_source="naip"):
    """Validate roads against satellite imagery using SAM3."""
    # 1. Fetch imagery tiles covering bbox
    tiles = fetch_imagery_tiles(bbox, source=imagery_source)

    # 2. Run SAM3 on each tile
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    road_masks = {}
    for tile_id, tile_img in tiles.items():
        state = processor.set_image(tile_img)
        output = processor.set_text_prompt(state, prompt="paved road")
        road_masks[tile_id] = output["masks"]

    # 3. Compare road geometries against masks
    scores = []
    for idx, road in roads_gdf.iterrows():
        overlap = compute_mask_overlap(road.geometry, road_masks)
        scores.append(overlap)

    roads_gdf["imagery_validation_score"] = scores
    return roads_gdf
```

---

## Integration Points

### Option A: Pre-Integration Filter (PREFERRED)

Validate unmatched segments before combining with reference network.

```python
# In pipeline.py, Stage 2
unmatched_gdf = filter_short_segments(unmatched_gdf, min_length)
unmatched_gdf = detect_near_duplicates(unmatched_gdf, matched_gdf)
unmatched_gdf = validate_road_existence(unmatched_gdf, bbox)  # NEW
# Adds existence_confidence, address_match_count, etc. columns
# Low-confidence roads still included but flagged
```

**Advantages:**
- Catches invalid roads early, before they affect orphan detection
- Validation scores available in all downstream outputs
- Can be run in parallel with other filters

### Option B: Post-Integration Validation

Validate after orphan detection, on the final main_edges.

```python
# In pipeline.py, after Stage 4
main_edges, orphans = detect_orphans_by_proximity(...)
main_edges = validate_road_existence(main_edges, bbox)  # NEW
# Flag low-confidence roads for QA review
```

**Use case:** When you only want to validate roads that passed connectivity checks.

### Option C: QA Enhancement

Add validation scores to `orphans.parquet` for the QA app to display, letting humans make final decisions with more context.

**Use case:** Human-in-the-loop validation, not automated.

---

## Output Schema

New columns added to road segments:

```python
# Address validation
"address_match_count": int,      # Nearby addresses with matching name
"address_validation_score": float,  # 0-1 confidence

# Building validation
"building_intersection_count": int,  # Buildings intersected (should be 0)
"building_validation_pass": bool,    # True if no intersections

# Combined
"existence_confidence": float,   # Combined score from all validators
"existence_flags": list[str],    # ["no_matching_addresses", "crosses_building"]
```

---

## Data Sources Summary

| Data | Source | Access Method |
|------|--------|---------------|
| Overture Addresses | Overture Maps | DuckDB/Parquet |
| Overture Buildings | Overture Maps | DuckDB/Parquet |
| NAIP Imagery | USGS | WMTS/GEE API |
| OpenAerialMap | OAM | REST API |
| Sentinel-2 | Copernicus | sentinelsat |

---

## Design Decisions (Resolved)

1. **Threshold behavior**: Merge with warning column
   - Add `existence_confidence` and related columns to output
   - Let downstream systems decide filtering thresholds
   - Preserves all data while providing validation signal

## Open Questions (For Future Implementation)

1. **Scope**: Should validation run on ALL unmatched segments, or only those about to be merged (connected, non-fringe)?

2. **Imagery priority**: Which satellite imagery approach to pursue first?
   - EODT4Crises API (free but requires imagery sourcing)
   - Custom model with NAIP (US-only but high resolution)
   - Building footprint proxy (simplest, no imagery needed)

3. **Performance**: Should validation be:
   - Synchronous (validate during pipeline run)?
   - Async batch job (validate after pipeline, flag for review)?

---

## Verification

1. **Unit tests**: Mock address/building data, verify scoring logic
2. **Integration tests**: Run validation on known good/bad road samples
3. **Manual QA**: Check validation scores correlate with human judgments
4. **Metrics**: Track false positive/negative rates for automated filtering

---

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/matcher/validation/__init__.py` | NEW - validation module |
| `src/matcher/validation/address_validator.py` | NEW - address-based validation |
| `src/matcher/validation/building_validator.py` | NEW - building intersection check |
| `src/matcher/integration/pipeline.py` | Add validation step |
| `src/matcher/integration/filters.py` | Optional: add validation filter |

---

## Dependencies

```toml
# Already in project
geopandas
shapely
rapidfuzz

# May need to add
duckdb  # For Overture data access (if not using existing fetch)
```
