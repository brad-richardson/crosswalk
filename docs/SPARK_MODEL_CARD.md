# Spark-Portable 28-Feature XGBoost Model

## Purpose

Road network conflation model for scoring candidate geometry pairs as match/no_match.
Designed for distributed inference in Spark via broadcast booster + pandas_udf.

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Trees:** 310 (Optuna with size penalty; the full 78-feature model uses 170)
- **Holdout match F1:** 0.912 (seed-42 segment-aware holdout, never seen during tuning)
- **CV F1 (match class):** 0.924 ± 0.007 (5-fold segment-aware cross-validation, training rows only)
- **Training data:** 5,430 labeled pairs across 34 datasets (after filtering hausdorff > 1000m)
- **Feature version:** 2026-02-16.1
- **Exported:** 2026-07-02

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
| **This model (28-feat)** | **28** | **0.912** | **89.5%** |
| Full matcher model | 78 | 0.930 | 91.7% |

The 28- and 78-feature rows are honest seed-42 holdout metrics (hyperparameters tuned
leakage-free, holdout never seen during tuning). Earlier versions of this card quoted
0.932 for the 28-feature model — that figure came from hyperparameters tuned with the
holdout included, so it was mildly optimistic; the drop to 0.912 is bias removal, not
a regression. The 28-feature model captures ~98% of the full model's match F1 with
~36% of the features.

## Hyperparameters (Optuna-tuned)

```
n_estimators: 310
learning_rate: 0.011
max_depth: 10
min_child_weight: 3
subsample: 0.816
colsample_bytree: 0.943
gamma: 0.714
reg_alpha: 1.007
reg_lambda: 2.010
max_bin: 400
scale_pos_weight: 0.635  (computed from training labels, not tuned)
```

Tuned 2026-07-02 with `scripts/tune_model.py --feature-set spark` (100 Optuna trials,
TPESampler seed=42) using the leakage-free protocol: the seed-42 holdout is discarded
before tuning and the objective is mean match F1 over an inner GroupKFold (segment-aware)
cross-validation on the training portion only, minus a size penalty of 0.00001 F1 per
tree above 100 to favor compact models for Spark deployment. Source of truth:
`SPARK_PORTABLE_XGB_PARAMS` in `config.py`.

## How to Reproduce

```bash
# Single command: train + export
uv run matcher export-spark-model
```

This uses `SPARK_PORTABLE_FEATURES` from `config.py` (inclusive list of the 28 features)
to train a model excluding all topology/graph/spatial-index features, then exports as
XGBoost-native JSON + manifest.

Note: The Optuna hyperparameters above were used for the shipped model. To retune, run
`uv run python scripts/tune_model.py --feature-set spark`. The default hyperparams in
`ml.py` (`DEFAULT_XGB_PARAMS`) are for the full 78-feature model; both param sets are
now tuned with the same leakage-free protocol.

## Known Limitations

- **class_similarity:** The Spark v1 job computes class similarity using a subtype-tier approach
  (base.py) while the matcher uses a continuous rank-based scorer (semantic.py). This is a known
  train/serve skew that may affect predictions for cross-class matches. Consider aligning the
  Spark implementation to the matcher's approach.
- **Name normalization:** Requires the same STREET_ABBREVIATIONS expansion as the matcher.
  The Spark v2 job includes this, but any alternative implementation must match exactly.
- **Sinuosity:** Must be capped at 10.0 to match training data.
