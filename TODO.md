# Matcher TODO

Actionable backlog for the road network matcher.

- For tried-and-removed features, see [docs/RESEARCH_GRAVEYARD.md](docs/RESEARCH_GRAVEYARD.md).
- For exploratory research ideas, see [docs/RESEARCH_IDEAS.md](docs/RESEARCH_IDEAS.md).

---

## Known Issues & Technical Debt

### HIGH: Scalability - Large Dataset Support

- **Problem**: `runner.py` uses `geopandas.read_parquet` which loads entire dataset into memory
- **Impact**: Will fail on state-sized or larger datasets
- **Location**: `src/matcher/pipeline/runner.py`
- **Solution**: Migrate to Spark/Sedona/GraphFrames (see [docs/RESEARCH_IDEAS.md](docs/RESEARCH_IDEAS.md#spark-migration-research-jan-2026))

### HIGH: Robust Feature Backfill Validation

**Problem**: Backfill now uses stored geometries from `labels/data/` (stable), but fallback to raw data lookup can still produce wrong features if data has been re-fetched with different filtering or extent.

**Needed**:
1. Candidate validation after resolving geometries (centroids within buffer distance)
2. Fallback rejection when stored geometry isn't available
3. Audit trail logging when backfill uses fallback lookup

**Location**: `src/matcher/cli/labels.py:backfill_features()`

### Medium: Robustness Issues

- **Overly broad exception handling** in `blocking/spatial_index.py` — `except Exception: return None` silently swallows errors
- **Race condition in model selection** in `ml.py` — checks file existence but doesn't validate model is loadable
- **CRS validation gap** in `pipeline/runner.py` — no check for null/invalid geometries after reprojection

### Low: Datasets with Polygon Geometries

Some target datasets have Polygon geometries instead of LineStrings (files deleted, need re-fetch):
- `ca_toronto_roads`, `co_bogota_bike_network`, `co_bogota_sidewalks`

---

## Feature Ideas

### Dual Carriageway / Centerline Handling

**Priority:** Medium
**Status:** Partially addressed via parallel sibling features

Parallel sibling features (`has_parallel_sibling_ref`, `parallel_fraction_ref`, `offset_vs_half_corridor_ratio`, `offset_over_expected_halfwidth`, `likely_representation_mismatch`) partially address dual carriageway detection. Remaining work:
- Detect split carriageway start/end points (Y-junction patterns)
- Pre-filter dual carriageway cases with specialized logic

---

## Integration

### Conflict Detector
- Detect duplicate matches in integration output (deferred)

---

## Agent Labeling

### Manually Curate Few-Shot Examples
- Current few-shot selection is automatic (random balanced sample from ground truth in other batches)
- Manually curate a set of high-quality examples covering key edge cases: split carriageways, parallel sidewalks, bike lanes, short overlaps, name mismatches
- Store in a dedicated directory (e.g. `data/agents/few_shot/`) so they're reused across batches

---

## Label Data Management

### Label Archive & History
- Archive orphaned labels to `labels/archived/` instead of losing them
- Provide recovery tooling to re-link archived labels

### Data Lineage
- Store data versions in model metadata
- Add `matcher model-info` command to show training data provenance

---

## Other Ideas

### Adaptive Buffer Distance
- Pipeline default is 75m with relaxed heading (90°) and length ratio (20.0) filters
- Could auto-detect optimal buffer per dataset via alignment statistics on sample

### Active Learning
- Use model uncertainty to prioritize labeling candidates

### Bike/Sidewalk Networks
- May need separate model or geometry-only approach
- Bike lane vs cycleway classification issue (PR #111)

---

## References

- **Ruiz-Lendinez et al. (2021)** - "Road Network Conflation Using Semantics and Geometry"
- **Juhasz et al. (2012)** - "Road Network Conflation Based on Iterative Hausdorff Distance Calculation"
- **Volz et al. (2011)** - "Map Conflation Using MRFs"
- **Hootenanny** (open-source conflation tool) - Junction angle distribution algorithms
