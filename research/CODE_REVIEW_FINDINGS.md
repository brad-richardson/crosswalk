# Code Review Findings

Combined findings from self-review and external feedback on `feature/ml-matcher-training` branch.

---

## Status Summary

| Issue | Status |
|-------|--------|
| #1 Data Leakage | **FIXED** |
| #2 Train/Inference Mismatch | **FIXED** |
| #3 Duplicate Features | **FIXED** |
| #4 Hardcoded Binary | **FIXED** |
| #5 Insecure Model Loading | Deferred (low risk for local use) |
| #6 Slow Feature Extraction | Deferred (performance optimization) |
| #7 Memory Risk | Deferred |
| #8 Redundant Functions | **FIXED** |
| #9 Import Ordering | **FIXED** |
| #10-18 | See details below |

**Additional fixes from external review:**
- Label filtering now excludes skip/unexpected labels (not just unsure)
- Empty candidates edge case handled in score_candidates
- Empty dataframe edge case handled in _extract_features_and_labels
- Label storage schema updated for mean_hausdorff_distance/overlap_ratio
- projection_distance removed from ML features (duplicate of mean_hausdorff)

---

## Fixed Issues

### 1. Data Leakage in ML Training [FIXED]
**Location:** `src/matcher/matching/ml.py`

Imputation now computed on training data only, then applied to both train and test sets.

### 2. Train/Inference Feature Mismatch [FIXED]
**Location:** `src/matcher/matching/ml.py`

`feature_medians` dict stored during training and used during inference.

### 3. Duplicate Feature Columns [FIXED]
**Location:** `src/matcher/matching/ml.py`

`_extract_from_columns` now properly deduplicates features and skips proxied columns.

### 4. Hardcoded Binary Assumption [FIXED]
**Location:** `src/matcher/matching/ml.py`

`predict()` now dynamically finds match class index from `self.model.classes_`.

### 8. Redundant Functions [FIXED]
**Location:** `src/matcher/features/geometric.py`

`_avg_projection_distance` now delegates to `_mean_hausdorff_distance`. `projection_distance` removed from ML FEATURE_COLUMNS to avoid double-weighting.

### 9. Import Ordering [FIXED]
**Location:** `src/matcher/labeling/app.py`

Moved import to top of file.

### NEW: Multiclass Label Filtering [FIXED]
**Location:** `src/matcher/matching/ml.py:146-154`

Training now filters to only valid labels (match, no_match, associated), excluding skip/unsure/unexpected values that would cause NaN in stratified split.

### NEW: Label Storage Schema [FIXED]
**Location:** `src/matcher/labeling/label_store.py`

Schema updated to include `mean_hausdorff_distance` and `overlap_ratio` (replacing `frechet_distance`).

### NEW: Empty Candidates Edge Case [FIXED]
**Location:** `src/matcher/matching/ml.py:445`

`score_candidates` now returns empty list early if no candidates.

### NEW: Empty DataFrame Edge Case [FIXED]
**Location:** `src/matcher/matching/ml.py:268`

`_extract_features_and_labels` now raises ValueError for empty dataframe and handles null first row gracefully.

---

## Deferred Issues

### 5. Insecure Model Loading
**Location:** `src/matcher/matching/ml.py` → `load_model()`

`joblib.load()` on untrusted paths can execute arbitrary code (pickle vulnerability).

**Deferred because:** Models are local files, not downloaded from untrusted sources.

**Future fix:** Validate model paths, consider ONNX format for sharing.

---

### 6. Python-Loop Feature Extraction (SLOW)
**Location:** `src/matcher/features/geometric.py`

`_mean_hausdorff_distance` iterates in Python and creates Shapely Point objects:
```python
dists_a_to_b = [line_b.distance(Point(coord)) for coord in line_a.coords]
```

**Deferred because:** Performance optimization for scale-up phase.

**Future fix:** Vectorize with numpy or use numba.

---

### 7. Memory Risk in Candidate Generation
**Location:** `src/matcher/blocking/spatial_index.py`

Creates full copies of GeoDataFrames, doubling memory.

**Deferred because:** Current datasets fit in memory.

---

## Remaining Code Quality Issues

### 10. Hardcoded Non-Road Class Filter
**Location:** `src/matcher/labeling/app.py:554`

```python
non_road_classes = {'footway', 'steps', 'cycleway', 'pedestrian', 'path', 'track'}
```

**Status:** Low priority, works for current use case.

### 11. Weight Mismatch Between Files
**Location:** `src/matcher/matching/rules.py` vs `src/matcher/config.py`

**Status:** Low priority, ML model doesn't use these weights.

### 12. Unused Variable
**Location:** Previously in ml.py, now removed via refactoring.

### 13. XGBoost Dependency Check [FIXED]
Added try/except with helpful error message.

---

## Existing Codebase Issues (Not from this PR)

### 14. Projection Logic Assumption
**Location:** `src/matcher/pipeline/runner.py`

UTM zone calculation assumes WGS84 input.

### 15. Hardcoded OSM Flags
**Location:** `src/matcher/fetch/osm.py`

Valid bridge values hardcoded.

### 16. Missing Evaluation Logic
**Location:** `src/matcher/cli.py`

`--ground-truth` flag is non-functional.

### 17. Unused 1:N Matching
**Location:** `src/matcher/pipeline/runner.py`

`optimize_with_one_to_many` exists but unused.

### 18. No Tests for New Functionality
ML training, new geometric features, and labeling filters lack test coverage.

---

## Final Model Performance

After fixes:
- Test accuracy: 86.4%
- CV F1 (5-fold): 0.823 ± 0.043
- Features: 9 (deduplicated, projection_distance removed)
- Training samples: 471 (14 unsure filtered)
