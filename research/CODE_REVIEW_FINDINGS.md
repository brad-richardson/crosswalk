# Code Review Findings

Combined findings from self-review and external feedback on `feature/ml-matcher-training` branch.

---

## 🔴 Critical Issues (Bugs & Data Integrity)

### 1. Data Leakage in ML Training
**Location:** `src/matcher/matching/ml.py` → `_prepare_features_from_columns()`

**Issue:** Median imputation is performed on the entire dataset before train/test split:
```python
# BUG: This leaks test data statistics into training
for i in range(X.shape[1]):
    col_median = np.nanmedian(X[:, i])  # Computed on ALL data
    X[np.isnan(X[:, i]), i] = col_median
```

**Impact:** Inflates evaluation metrics, model may perform worse in production than reported.

**Fix:** Calculate imputation statistics only on `X_train`, then apply to both `X_train` and `X_test`.

---

### 2. Train/Inference Feature Mismatch
**Location:** `src/matcher/matching/ml.py`

**Issue:**
- Training uses median imputation for missing values
- Inference (`_features_to_array`) fills missing values with `0.0`:
```python
row = [feat_dict.get(col, 0.0) for col in self.feature_names]  # 0.0, not median!
```

**Impact:** Model trained on median-imputed data sees zeros in production → degraded performance.

**Fix:** Store imputation values during training, apply same values during inference.

---

### 3. Duplicate Feature Columns in Training
**Location:** `src/matcher/matching/ml.py:258-267`

**Issue:** When `mean_hausdorff_distance` or `overlap_ratio` aren't in labels, the mapping creates duplicates:
```
"Using features: ['hausdorff_distance', 'hausdorff_distance', 'buffer_iou', 'buffer_iou', ...]"
```

**Impact:** Model trains on redundant features, wasting capacity.

**Fix:** Track which features are actually available vs proxied, avoid duplicates in feature list.

---

### 4. Hardcoded Binary Assumption in predict()
**Location:** `src/matcher/matching/ml.py:293`

```python
match_idx = 1  # In binary, class 1 is match
```

**Issue:** Breaks if multiclass model is loaded.

**Fix:** Dynamically find match index from `self.label_decoder`.

---

### 5. Insecure Model Loading
**Location:** `src/matcher/matching/ml.py` → `load_model()`

**Issue:** `joblib.load()` on untrusted paths can execute arbitrary code (pickle vulnerability).

**Recommendation:**
- Validate model paths are within expected directories
- Consider ONNX format for external model sharing
- Add checksum verification for model files

---

## 🟠 Performance Issues

### 6. Python-Loop Feature Extraction (SLOW)
**Location:** `src/matcher/features/geometric.py`

**Issue:** `_mean_hausdorff_distance` and `_avg_projection_distance` iterate in Python and create Shapely Point objects in loops:
```python
dists_a_to_b = [line_b.distance(Point(coord)) for coord in line_a.coords]
```

**Impact:** Extremely slow for large networks (millions of segments).

**Fix:** Vectorize using numpy on coordinate arrays, or use numba (already in dependencies but unused).

---

### 7. Memory Risk in Candidate Generation
**Location:** `src/matcher/blocking/spatial_index.py` → `generate_candidates()`

**Issue:** Creates full copies of GeoDataFrames (`reference_prep`, `target_prep`), doubling memory.

**Impact:** OOM errors on large datasets.

**Fix:** Pass only necessary columns, or use `generate_candidates_iter()` (currently unused).

---

### 8. Redundant Functions
**Location:** `src/matcher/features/geometric.py`

**Issue:** `_mean_hausdorff_distance` and `_avg_projection_distance` now compute the exact same thing.

**Fix:** Consolidate into one function, or differentiate their behavior.

---

## 🟡 Code Quality Issues

### 9. Import Statement in Wrong Location
**Location:** `src/matcher/labeling/app.py:35`

```python
def save_config(config: dict) -> None:
    ...

from matcher.config import settings  # <- Should be at top
```

---

### 10. Hardcoded Non-Road Class Filter
**Location:** `src/matcher/labeling/app.py:554`

```python
non_road_classes = {'footway', 'steps', 'cycleway', 'pedestrian', 'path', 'track'}
```

**Fix:** Move to config or make it a parameter.

---

### 11. Weight Mismatch Between Files
**Location:** `src/matcher/matching/rules.py` vs `src/matcher/config.py`

**Issue:** `DEFAULT_WEIGHTS` and `matching_weights` have diverged.

**Fix:** Single source of truth - either config.py or rules.py, not both.

---

### 12. Unused Variable
**Location:** `src/matcher/matching/ml.py` → `train()`

```python
df_valid = df[valid_mask].copy()  # Never used
```

---

### 13. No Dependency Check for XGBoost
**Location:** `src/matcher/matching/ml.py`

**Issue:** `import xgboost` inside `train()` fails with unclear error if not installed.

**Fix:** Add try/except with helpful error message.

---

## 🔵 Existing Codebase Issues (Not from this PR)

### 14. Projection Logic Assumption
**Location:** `src/matcher/pipeline/runner.py`

**Issue:** UTM zone calculation assumes WGS84 input. May fail silently for other CRS.

---

### 15. Hardcoded OSM Flags
**Location:** `src/matcher/fetch/osm.py` → `_build_road_flags`

**Issue:** Valid bridge values (viaduct, trestle, etc.) are hardcoded.

**Fix:** Move to config.py.

---

### 16. Missing Evaluation Logic
**Location:** `src/matcher/cli.py` → `evaluate`

**Issue:** Contains TODO for precision/recall, `--ground-truth` flag is non-functional.

---

### 17. Unused 1:N Matching Capability
**Location:** `src/matcher/pipeline/runner.py`

**Issue:** `optimize_with_one_to_many` exists in `optimizer.py` but pipeline only uses 1:1 matching.

---

### 18. No Tests for New Functionality

**Issue:** ML training, new geometric features, and labeling filters have no test coverage.

---

## Priority Order for Fixes

| Priority | Issue | Effort |
|----------|-------|--------|
| P0 | #1 Data Leakage | Medium |
| P0 | #2 Train/Inference Mismatch | Medium |
| P1 | #3 Duplicate Features | Low |
| P1 | #4 Hardcoded Binary | Low |
| P1 | #6 Slow Feature Extraction | High |
| P2 | #7 Memory Risk | Medium |
| P2 | #8 Redundant Functions | Low |
| P2 | #11 Weight Mismatch | Low |
| P3 | Others | Low-Medium |

---

## Recommendations

1. **Immediate (before merge):** Fix #1, #2, #3, #4 - these affect model correctness
2. **Soon:** Address #6 performance before scaling to larger datasets
3. **Later:** Clean up code quality issues, add tests
