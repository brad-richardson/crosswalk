# Spark-Portable 28-Feature XGBoost Model

## Purpose

Road network conflation model for scoring candidate geometry pairs as match/no_match.
Designed for distributed inference in Spark via broadcast booster + pandas_udf.

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Trees:** 200 (tuned from 834 via Optuna with size penalty)
- **CV F1 (match class):** 0.932 ± 0.004 (5-fold segment-aware cross-validation)
- **Training data:** 5,429 labeled pairs across 34 datasets (after filtering hausdorff > 1000m)
- **Feature version:** 2026-02-16.1
- **Exported:** 2026-03-27

## Why 28 Features (not 78)

The full matcher model uses 78 features including topology (18), graphlet (2), clustering (3),
and additional name/shape features. The 28-feature subset was selected for Spark portability:

**Included (28):** Features computable from aligned geometry pairs alone, without graph topology
or spatial index queries:

| Category | Features | Count |
|----------|----------|-------|
| Hausdorff variants | hausdorff_distance_m, hausdorff_p95_m, mean_hausdorff_distance_m | 3 |
| Buffer IoU | buffer_iou_5m, buffer_iou_15m | 2 |
| Heading/Alignment | heading_delta, collinear_gap_ratio, edge_distance_rmse_m | 3 |
| Name similarity | name_levenshtein, name_token_sort, name_numeric_match | 3 |
| Class | class_similarity | 1 |
| Lateral offset | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m | 3 |
| Coverage | ref_coverage, target_coverage, min_coverage, coverage_ratio | 4 |
| Sinuosity | sinuosity_ref, sinuosity_target | 2 |
| Heading consistency | heading_consistency_target | 1 |
| Length | min_length_m, aligned_length_m | 2 |
| Shape complexity | shape_complexity_target | 1 |
| Parallel sibling | offset_over_expected_halfwidth | 1 |
| Intersection overlap | post_node_continuation_m, endpoint_heading_divergence | 2 |

**Excluded (50):** Features requiring infrastructure not available in the Spark matching pipeline:

- **Topology (18):** Require Overture connector graph + synthetic connector computation for target side
- **Graphlet (2) + Clustering (3):** Require local graph neighborhood construction
- **Endpoint proximity (3):** Require spatial index of connectors (could be added later — best candidate for next improvement)
- **Crossing angle (4):** Require spatial index of neighboring segments
- **Parallel sibling (4 of 5):** Require spatial index for nearby same-name segments
- **Additional name (7 of 10):** jaro_winkler, soundex, metaphone, has_name_*, name_is_generic, route_prefix_match — low marginal value given levenshtein + token_sort
- **Additional shape/heading (5):** ref-side variants and deltas — low feature importance
- **Vertex density (3):** Low discriminative power
- **Angle histogram (1):** Correlated with heading_delta + buffer_iou

## Performance Comparison

| Model | Features | Match F1 | Accuracy |
|-------|----------|----------|----------|
| Spark formula (hand-tuned) | 6 | 0.859 | 80.9% |
| **This model (28-feat)** | **28** | **0.932** | **89.8%** |
| Full matcher model | 78 | 0.937 | 93.2% |

The 28-feature model captures ~95% of the full model's quality with ~36% of the features.

## Hyperparameters (Optuna-tuned)

```
n_estimators: 200
learning_rate: 0.038
max_depth: 8
min_child_weight: 1
subsample: 0.731
colsample_bytree: 0.938
gamma: 0.742
reg_alpha: 0.682
reg_lambda: 2.052
max_bin: 206
scale_pos_weight: 0.650
```

Tuned with 80 Optuna trials using a slight penalty for tree count (0.00001 F1 per tree above 100).
Best threshold found at 200 trees — actually slightly higher F1 than 834 trees (less overfitting
with the higher learning rate and stronger regularization).

## How to Reproduce

```bash
# Single command: train + export
uv run matcher export-spark-model
```

This uses `SPARK_PORTABLE_FEATURES` from `config.py` (inclusive list of the 28 features)
to train a model excluding all topology/graph/spatial-index features, then exports as
XGBoost-native JSON + manifest.

Note: The Optuna hyperparameters above were used for the shipped model. To retune,
see `scripts/tune_model.py`. The default hyperparams in `ml.py` are for the full 78-feature
model — the 28-feature model benefits from a higher learning rate and fewer trees.

> **Caveat (2026-07-02):** these Spark-model hyperparameters were tuned before
> `tune_model.py` adopted the leakage-free protocol, so they saw the seed-42
> holdout and the F1 figures above are mildly optimistic. The 78-feature
> `DEFAULT_XGB_PARAMS` have been retuned leakage-free; the Spark params have not.

## Known Limitations

- **class_similarity:** The Spark v1 job computes class similarity using a subtype-tier approach
  (base.py) while the matcher uses a continuous rank-based scorer (semantic.py). This is a known
  train/serve skew that may affect predictions for cross-class matches. Consider aligning the
  Spark implementation to the matcher's approach.
- **Name normalization:** Requires the same STREET_ABBREVIATIONS expansion as the matcher.
  The Spark v2 job includes this, but any alternative implementation must match exactly.
- **Sinuosity:** Must be capped at 10.0 to match training data.
