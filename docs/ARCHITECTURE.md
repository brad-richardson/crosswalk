# Matcher Architecture

Technical details of the ML pipeline, feature computation, and decision thresholds.

For usage instructions, see [README.md](../README.md). For development workflow, see [CLAUDE.md](../CLAUDE.md).

## ML Model

- **Algorithm**: XGBoost binary classifier
- **Features**: 72 features across 16 categories (defined in `src/matcher/config.py::FEATURE_COLUMNS`)
- **Location**: `data/models/matcher_model_combined.joblib`
- **Training**: `matcher train` (trains on all labeled data in `labels/`)
- **Parallelization**: Uses `ProcessPoolExecutor` with worker initialization for feature computation
- **Auto Model Selection**: When `settings.auto_select_model=True`, automatically uses geometry-only model for datasets with low name coverage (< 50%)

The trained model is not committed to git. After cloning, run `matcher train` before using `-m xgboost`.

## Decision Thresholds

All thresholds are configurable in `config.py`.

### Scoring Thresholds (per-candidate, bridge file output)

Applied by the ML scorer when classifying each candidate pair:

| Setting | Default | Decision |
|---------|---------|----------|
| `scoring_match_threshold` | 0.5 | `>= this` -> MATCH |
| `scoring_review_threshold` | 0.1 | `>= this` -> REVIEW, below -> NO_MATCH |

### Optimizer/Labeling Thresholds (1:N groups and labeling UI)

1:N group optimization uses the Hungarian algorithm to resolve cases where a single Overture segment corresponds to multiple local segments (e.g., split carriageways).

Applied during 1:N group optimization and to define the labeling UI review band:

| Setting | Default | Purpose |
|---------|---------|---------|
| `optimizer_match_threshold` | 0.75 | Confident match in optimizer; upper bound of labeling review band |
| `optimizer_review_threshold` | 0.5 | Below this = no match in optimizer; lower bound of labeling review band |

1:N groups: `avg_confidence >= optimizer_review_threshold` -> MATCH

## Model Evaluation

Use cross-validation or holdout evaluation for unbiased metrics:

```bash
# Cross-validation (default: 5-fold, segment-aware splitting)
matcher ml eval

# Evaluate an existing model on 20% holdout
matcher ml eval --model data/models/matcher_model_combined.joblib

# Custom folds or seed
matcher ml eval --cv-folds 10 --seed 123

# Evaluate on specific dataset(s)
matcher ml eval --model data/models/matcher_model_combined.joblib -d us_frisco_trails
```

**Why holdout/CV matters:**
- Evaluating on training data gives artificially inflated accuracy (~99%)
- Cross-validation gives realistic generalization metrics with variance estimates
- Segment-aware splitting prevents data leakage (the same segment ID—either gers_id or target_id—never appears in both train and test)
- Use consistent seed (default: 42) for comparable results across experiments

## Feature Categories

72 features across 16 categories. `config.py::FEATURE_COLUMNS` is the single source of truth.

| Category | Count | Features |
|----------|-------|----------|
| Geometric | 11 | hausdorff_distance_m, mean_hausdorff_distance_m, hausdorff_p95_m, buffer_iou_5m, buffer_iou_15m, heading_delta, length_ratio, centroid_distance_m, collinear_gap_ratio, angle_histogram_similarity, edge_distance_rmse_m |
| Name Similarity | 10 | name_levenshtein, name_jaro_winkler, name_token_sort, name_soundex, name_metaphone, has_name_ref, has_name_target, name_is_generic, name_numeric_match, route_prefix_match |
| Class | 1 | class_similarity |
| Endpoint/Connectivity | 3 | min_endpoint_proximity_m, max_endpoint_proximity_m, shared_endpoint_count |
| Lateral Offset | 3 | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m |
| Topology | 12 | from/to_degree_ref/target, degree_match_score, degree_signature_similarity, is_dead_end_ref/target, dead_end_match, is_intersection_ref/target, intersection_match |
| Alignment Coverage | 4 | ref_coverage, target_coverage, min_coverage, coverage_ratio |
| Graphlet | 2 | graphlet_similarity, endpoint_degree_similarity |
| Clustering | 3 | clustering_coef_ref, clustering_coef_target, clustering_coef_delta |
| Sinuosity | 3 | sinuosity_ref, sinuosity_target, sinuosity_delta |
| Heading Consistency | 3 | heading_consistency_ref, heading_consistency_target, heading_consistency_delta |
| Vertex Density | 3 | vertex_density_ref, vertex_density_target, vertex_density_ratio |
| Length | 2 | min_length_m, aligned_length_m |
| Shape Complexity | 3 | shape_complexity_ref, shape_complexity_target, shape_complexity_delta |
| Parallel Sibling | 5 | has_parallel_sibling_ref, parallel_fraction_ref, offset_vs_half_corridor_ratio, offset_over_expected_halfwidth, likely_representation_mismatch |
| Crossing Angle | 4 | crossing_angle_min_ref, transverse_neighbor_fraction_ref, crossing_angle_min_target, transverse_neighbor_fraction_target |

## Feature Computation Paths

Understanding the computation paths is critical for preventing training/inference skew.

### Single Source of Truth

```
config.py::FEATURE_COLUMNS (72 features)
         |
         |---> compute.py::compute_pair_features()  <-- AUTHORITATIVE computation
         |           |
         |           |---> ml.py::_compute_single_feature() (inference)
         |           |
         |           +---> labeling UI (training data generation)
         |
         +---> src/matcher/labeling/feature_store.py (Parquet storage, keyed by gers_id + target_id)
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
            +---> FeatureStore.add(features=computed_features)
            +---> LabelStore.add(label metadata only)
```

**Path 3: Training (loading labels)**
```
ml.py::train()
    |
    +---> LabelStore.load_all()
            |
            +---> Joins human labels (CSV) with features (Parquet)
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
