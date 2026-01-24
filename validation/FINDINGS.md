# Validation System Findings

## Overview

Built a validation system that uses Overture's known provenance to create ground-truth experiments. Since Overture segments have `record_id` linking back to source OSM ways, we can:
1. Drop specific OSM-sourced segments from Overture
2. Fetch fresh OSM data (same way IDs)
3. Run matcher and check if the right segments get matched back

## Initial Results

**Random 5% Drop Strategy (Fast Mode)**

| Metric | Value |
|--------|-------|
| Recall | **74.5%** |
| Dropped segments | 4,429 |
| Fresh OSM segments | 5,503 |
| Matched back | 4,097 |
| Orphaned (not matched) | 1,406 |
| Mean confidence | 0.401 |

Fast mode only matches segments that correspond to dropped record_ids, reducing candidates from ~3M to ~178k.

## Key Observations

1. **74.5% recall is reasonable but not great** - Over 25% of segments that should have matched were orphaned
2. **Low mean confidence (0.401)** - Matched segments have relatively low confidence scores
3. **More fresh OSM segments than dropped** - 5,503 fresh OSM vs 4,429 dropped Overture segments, likely due to Overture's merging of multiple OSM ways into single segments

## Things to Explore

### 1. Analyze Failure Cases
The 1,406 false negatives are saved to `validation/random_5pct_fast_v2/failures.parquet`. Questions to investigate:
- Are failures concentrated in certain road classes?
- Are failures related to geometry complexity (many vertices, sharp angles)?
- Are failures due to Overture's merging/splitting affecting matching?
- What's the distribution of nearest-reference distances for failures?

### 2. Compare Strategies
Run experiments with different drop strategies:
- **TomTom source**: Drop all TomTom segments - tests matching with genuinely different geometry
- **Road class**: Drop residential roads - tests performance on a specific class
- **Bbox**: Drop a geographic region - tests spatial consistency

### 3. Compare Matchers
Run same experiments with:
- Rule-based matcher (current)
- XGBoost matcher
- Compare recall, confidence distribution, and failure patterns

### 4. Full Mode vs Fast Mode
Fast mode only matches dropped segments. Full mode matches all OSM segments against reduced reference. Compare:
- Does full mode have different recall?
- Are there unexpected matches (non-dropped segments matching)?

## Implementation Gaps

### High Priority

1. **Name column handling**: Fresh OSM has `names` as dict (`{'primary': 'Beacon Street'}`), but pipeline expects simple string `name` column. Need to extract primary name for proper name-based matching.

2. **Confidence calibration**: Mean confidence of 0.401 is low. Need to investigate:
   - Is this appropriate for the match quality?
   - Should confidence thresholds be adjusted?

3. **Stratified metrics**: Break down recall by:
   - Road class (primary, secondary, residential, etc.)
   - Road length
   - Geometric complexity

### Medium Priority

4. **Parallel scoring**: Scoring is single-threaded. Could parallelize for faster full-mode experiments.

5. **Geometry perturbation**: Add option to perturb holdout geometry before matching to simulate real-world coordinate differences between datasets.

6. **Cross-validation**: Run multiple random seeds and aggregate metrics for more robust estimates.

### Low Priority

7. **Visualization**: Generate maps showing:
   - Matched vs unmatched segments
   - Confidence heatmaps
   - Failure case locations

8. **CI integration**: Add validation experiments to CI to catch matcher regressions.

## Files Created

```
src/matcher/validation/
├── __init__.py          # Module exports
├── holdout.py           # Create holdout datasets by dropping segments
├── evaluate.py          # Evaluate by record_id matching
└── experiment.py        # Experiment orchestration

validation/
├── FINDINGS.md          # This file
└── random_5pct_fast_v2/ # Experiment output
    ├── config.json
    ├── dropped_record_ids.json
    ├── reduced_reference.parquet
    ├── fresh_osm.parquet
    ├── bridge.parquet
    ├── unmatched.parquet
    ├── evaluation.parquet
    ├── metrics.json
    └── failures.parquet
```

## CLI Usage

```bash
# Random 5% drop with fast mode
matcher validate-matching data/raw/overture_segments.parquet \
    --bbox "-71.19,42.21,-70.92,42.40" \
    --strategy random --fraction 0.05 \
    --output validation/random_5pct/ \
    --fast

# TomTom holdout
matcher validate-matching data/raw/overture_segments.parquet \
    --bbox "-71.19,42.21,-70.92,42.40" \
    --strategy source --source-dataset TomTom \
    --output validation/tomtom/

# Residential roads holdout
matcher validate-matching data/raw/overture_segments.parquet \
    --bbox "-71.19,42.21,-70.92,42.40" \
    --strategy class --road-class residential \
    --output validation/residential/
```

## Next Steps

1. Run failure analysis on current results
2. Run TomTom strategy experiment
3. Fix name column extraction for better name matching
4. Add stratified metrics by road class
