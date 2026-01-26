# Matcher Architecture

Technical details of the ML pipeline, feature computation, and decision thresholds.

For usage instructions, see [README.md](../README.md). For development workflow, see [CLAUDE.md](../CLAUDE.md).

## ML Model

- **Algorithm**: XGBoost binary classifier
- **Features**: 56 features across 14 categories (defined in `src/matcher/config.py::FEATURE_COLUMNS`)
- **Location**: `data/models/matcher_model_combined.joblib`
- **Training**: `matcher train` (trains on all labeled data in `labels/`)
- **Parallelization**: Uses `ProcessPoolExecutor` with worker initialization for feature computation
- **Auto Model Selection**: When `settings.auto_select_model=True`, automatically uses geometry-only model for datasets with low name coverage (< 50%)

The trained model is not committed to git. After cloning, run `matcher train` before using `-m xgboost`.

## Decision Thresholds

Two separate threshold systems exist:

### ML Scoring (hardcoded in `ml.py`)

Applied per-candidate during scoring:

| Confidence | Decision |
|------------|----------|
| `>= 0.5` | MATCH |
| `>= 0.1` | REVIEW |
| `< 0.1` | NO_MATCH |

### Optimizer Settings (configurable in `config.py`)

Applied during 1:N group optimization:

| Setting | Default | Purpose |
|---------|---------|---------|
| `match_threshold` | 0.75 | Confidence for automatic match in optimizer |
| `review_threshold` | 0.5 | Below this = no match in optimizer |

1:N groups: `avg_confidence >= review_threshold` -> MATCH

## Model Evaluation

Always use holdout evaluation for unbiased metrics:

```bash
# Default: 20% holdout with seed=42 (recommended)
matcher eval-model data/models/matcher_model_combined.joblib

# Custom seed for different split
matcher eval-model data/models/matcher_model_combined.joblib --seed 123

# Evaluate on ALL data (may include training data - use with caution)
matcher eval-model data/models/matcher_model_combined.joblib --no-holdout
```

**Why holdout matters:**
- Evaluating on training data gives artificially inflated accuracy (~99%)
- Holdout evaluation gives realistic generalization metrics (~95-96%)
- Use consistent seed (default: 42) for comparable results across experiments
- When comparing models or feature sets, always use the same holdout split

## Feature Categories

56 features across 14 categories. `config.py::FEATURE_COLUMNS` is the single source of truth.

| Category | Count | Features |
|----------|-------|----------|
| Geometric | 9 | hausdorff_distance_m, mean_hausdorff_distance_m, hausdorff_p95_m, buffer_iou_5m, buffer_iou_15m, heading_delta, length_ratio, centroid_distance_m, collinear_gap_ratio |
| Semantic - Name | 8 | name_levenshtein, name_jaro_winkler, name_token_sort, name_soundex, name_metaphone, has_name_ref, has_name_target, name_is_generic |
| Semantic - Class | 1 | class_similarity |
| Endpoint/Connectivity | 3 | min_endpoint_proximity_m, max_endpoint_proximity_m, shared_endpoint_count |
| Lateral Offset | 3 | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m |
| Topology | 12 | from/to_degree_ref/target, degree_match_score, degree_signature_similarity, is_dead_end_ref/target, dead_end_match, is_intersection_ref/target, intersection_match |
| Alignment Coverage | 4 | ref_coverage, target_coverage, min_coverage, coverage_ratio |
| Graphlet | 2 | graphlet_similarity, endpoint_degree_similarity |
| Sinuosity | 3 | sinuosity_ref, sinuosity_target, sinuosity_delta |
| Heading Consistency | 3 | heading_consistency_ref, heading_consistency_target, heading_consistency_delta |
| Vertex Density | 3 | vertex_density_ref, vertex_density_target, vertex_density_ratio |
| Length | 1 | min_length_m |
| Shape Complexity | 3 | shape_complexity_ref, shape_complexity_target, shape_complexity_delta |
| Numeric Route | 1 | name_numeric_match |

## Feature Computation Paths

Understanding the computation paths is critical for preventing training/inference skew.

### Single Source of Truth

```
config.py::FEATURE_COLUMNS (56 features)
         |
         |---> compute.py::compute_pair_features()  <-- AUTHORITATIVE computation
         |           |
         |           |---> ml.py::_compute_single_feature() (inference)
         |           |
         |           +---> labeling UI (training data generation)
         |
         +---> label_store.py::LABEL_COLUMNS (storage schema)
```

### Computation Paths

**Path 1: ML Inference (scoring candidates)**
```
ml.py::score_candidates()
    |
    |---> Pre-compute endpoint features: compute_endpoint_features()
    |---> Pre-compute topology features: compute_all_topology()
    |---> Pre-compute graphlet features: precompute_graphlet_features()
    |---> Pre-compute alignments: compute_alignment_batch()
    |
    +---> Parallel workers call _compute_single_feature()
            |
            +---> compute_pair_features(..., endpoint_features=pre_computed, ...)
```

**Path 2: Labeling UI (training data generation)**
```
labeling UI
    |
    +---> compute_pair_features() directly
            |
            +---> LabelStore.add(features=computed_features)
```

**Path 3: Training (loading labels)**
```
ml.py::train()
    |
    +---> LabelStore.load_all()
            |
            +---> Features already stored in CSV (from labeling)
```

### Pre-computation Table

The ML scorer pre-computes certain features **before** parallelization for efficiency:

| Feature Type | Pre-computed? | Why |
|--------------|---------------|-----|
| Endpoint proximity | Yes | Requires spatial index over all segments |
| Topology (degrees) | Yes | Requires Union-Find over full network |
| Graphlet features | Yes | Requires building road graph |
| Alignments | Yes | Expensive geometry operations |
| Geometric/semantic | No | Computed per-pair in workers |

**Critical invariant**: Pre-computed features must produce the same values as direct computation. This is tested in `tests/unit/test_ml_pipeline_consistency.py`.

### Imputation Consistency

Missing feature values are imputed using medians computed from **training data only**:

```python
# During training (ml.py::train)
for feat in features:
    median = np.nanmedian(X_train[:, feat_idx])  # Training data only!
    self.feature_medians[feat] = median

# During inference (ml.py::_impute_missing)
fill_value = self.feature_medians.get(feat_name, 0.0)  # Uses stored median
```

**Risk**: If a new feature is added but not in `feature_medians`, inference falls back to 0.0, which may not be appropriate.

## Test Coverage for Consistency

| Test File | What It Catches |
|-----------|----------------|
| `test_label_store.py` | Features computed but not saved to labels |
| `test_feature_consistency.py` | Error defaults, naming conventions |
| `test_ml_pipeline_consistency.py` | Pre-computation vs direct computation, imputation consistency |
