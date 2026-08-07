# Spark-Portable 28-Feature XGBoost Model

## Purpose

Road network conflation model for scoring candidate geometry pairs as match/no_match.
Designed for distributed inference in Spark via broadcast booster + pandas_udf.

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Trees:** 224, max_depth 10 (Optuna + epsilon-compact selection; the full local model uses 170)
- **Holdout match F1:** 0.909 (seed-42 segment-aware holdout, never seen during tuning)
- **CV F1 (match class):** 0.922 ± 0.008 (5-fold segment-aware cross-validation, training rows only)
- **Predict throughput:** ~1.8M rows/sec single-node (see Inference Latency below)
- **Training data:** 5,430 labeled pairs across 34 datasets (after filtering hausdorff > 1000m)
- **Feature version:** 2026-07-07.2
- **Exported:** 2026-07-03

## Why 28 Features (not 83)

The full matcher model uses 83 features including topology (22), graphlet (2), clustering (3),
and additional name/shape features. The 28-feature subset was selected for Spark portability:

**Included (28):** Computable from aligned geometry pairs alone, without graph topology or
spatial index queries. 45 of the 83 clear that bar; these 28 are the subset selected for
inclusion. Note what is *not* claimed: no tier in
[research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md)
ablates a member of the 28 — every tier is `base + additions` — so their individual
contributions have never been measured. The 17 exclusions below are backed by
measurement; these 28 inherit their membership from the original feasibility cut.

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

**Excluded (55)**, for two different reasons. Read the split carefully — it is the
part of this card that is easiest to get wrong, and `config.py` got it wrong until
2026-08-07.

*Excluded because Spark cannot compute them (38).* These need network-wide
structure that does not exist per candidate row:

- **Topology (22):** Require Overture connector graph + synthetic connector computation for target side
- **Graphlet (2) + Clustering (3):** Require local graph neighborhood construction
- **Endpoint proximity (3):** Require spatial index of connectors (could be added later — best candidate for next improvement)
- **Crossing angle (4):** Require spatial index of neighboring segments
- **Parallel sibling (4 of 5):** Require spatial index for nearby same-name segments

*Excluded on measured value, not feasibility (17).* These need nothing but the two
aligned geometries and the two name structs — the Spark job already holds both.
Proven computable bit-for-bit from a bare pair in
`tests/test_spark_feature_expansion.py`; measured for F1 / size / latency in
[research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md):

- **Additional name (7 of 10):** jaro_winkler, soundex, metaphone, has_name_*, name_is_generic, route_prefix_match. 6 of the 7 are already computed by `compute_name_similarity()` and discarded by the exporter — marginal cost 0.00 µs/pair. The "low marginal value given levenshtein + token_sort" verdict holds for the block as a whole (+0.0033 LOO F1), but **not** for `has_name_target`, which is worth +0.0043 on its own.
- **Additional shape/heading (5):** ref-side variants and deltas — low feature importance. Not individually ablated; the geometry block *as a whole* (all 10 below) measures −0.0034 LOO F1, and the solo deltas for these five span −0.0017 to +0.0004.
- **Vertex density (3):** Low discriminative power. Confirmed (`vertex_density_target` is the worst of all 17 at −0.0026).
- **Angle histogram (1):** Correlated with heading_delta + buffer_iou. Confirmed (−0.0001).
- **max_coverage (1):** `max(ref_coverage, target_coverage)` — derivable in SQL from columns the model already carries, and 4th by XGBoost gain in a 45-feature model. Still measures −0.0007: the splits do not transfer across datasets. A trap; see §3 of the research doc.

## Performance Comparison

| Model | Features | Match F1 | Accuracy |
|-------|----------|----------|----------|
| Spark formula (hand-tuned) | 6 | 0.859 | 80.9% |
| **This model (28-feat)** | **28** | **0.909** | **89.2%** |
| Full matcher model | 78 | 0.930 | 91.7% |

The full-model row is the 78-feature model as it stood when this table was measured;
the local model has since grown to 83 features, and the row has not been re-measured.
For a same-protocol comparison against the current 83-feature model, see the tier table
in [research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md).

The 28- and 78-feature rows are honest seed-42 holdout metrics (hyperparameters tuned
leakage-free, holdout never seen during tuning). Earlier versions of this card quoted
0.932 for the 28-feature model — that figure came from hyperparameters tuned with the
holdout included, so it was mildly optimistic; most of the drop to 0.909 is bias removal
(the best leakage-free trial scores 0.912 on the holdout), plus ~0.002 traded away by the
epsilon-compact selection for ~1.3x faster inference. The 28-feature model captures ~98%
of the full model's match F1 with ~36% of the features.

## Hyperparameters (Optuna-tuned)

```
n_estimators: 224
learning_rate: 0.013
max_depth: 10
min_child_weight: 2
subsample: 0.802
colsample_bytree: 0.966
gamma: 0.602
reg_alpha: 1.544
reg_lambda: 2.188
max_bin: 343
scale_pos_weight: 0.635  (computed from training labels, not tuned)
```

Tuned 2026-07-03 with `scripts/tune_model.py --feature-set spark` (100 Optuna trials,
TPESampler seed=42) using the leakage-free protocol: the seed-42 holdout is discarded
before tuning and the objective is mean match F1 over an inner GroupKFold (segment-aware)
cross-validation on the training portion only, minus a size penalty of 0.00001 F1 per
tree above 100. Because inference speed is critical for the Spark deployment, the final
params come from **epsilon-compact selection**: among all trials within 0.003 raw CV F1
of the best, the one with the lowest traversal cost (`n_estimators * max_depth`) is
selected — here 224 trees x depth 10 (CV F1 0.9216) over the best-F1 trial's
310 x 10 (CV F1 0.9242). Source of truth: `SPARK_PORTABLE_XGB_PARAMS` in `config.py`.

## Inference Latency

`.predict` on the seed-42 holdout features tiled to 1M rows (float32, 28 features),
median of 5 runs, single node (Apple Silicon, XGBoost `hist`):

| Params | Trees x depth | 1M rows | Throughput |
|--------|--------------|---------|------------|
| Old (leaky-tuned) | 200 x 8 | 0.44 s | ~2.3M rows/sec |
| **Selected (this model)** | **224 x 10** | **0.56 s** | **~1.8M rows/sec** |

Absolute numbers are machine-specific; the ratio is what matters. The best-F1
leakage-free trial (310 x 10, 0.73 s, ~1.4M rows/sec) was rejected by the
epsilon-compact selection as ~1.3x slower for +0.002 holdout F1.

## How to Reproduce

```bash
# Single command: train + export
uv run crosswalk export-spark-model
```

This uses `SPARK_PORTABLE_FEATURES` from `config.py` (inclusive list of the 28 features)
to train a model excluding all topology/graph/spatial-index features, then exports as
XGBoost-native JSON + manifest.

Note: The Optuna hyperparameters above were used for the shipped model. To retune, run
`uv run python scripts/tune_model.py --feature-set spark`. The default hyperparams in
`ml.py` (`DEFAULT_XGB_PARAMS`) are for the full 83-feature model; both param sets are
now tuned with the same leakage-free protocol.

## Known Limitations

- **class_similarity:** The Spark v1 job computes class similarity using a subtype-tier approach
  (base.py) while the matcher uses a continuous rank-based scorer (semantic.py). This is a known
  train/serve skew that may affect predictions for cross-class matches. Consider aligning the
  Spark implementation to the matcher's approach.
- **Name normalization:** Requires the same STREET_ABBREVIATIONS expansion as the matcher.
  The Spark v2 job includes this, but any alternative implementation must match exactly.
- **Sinuosity:** Must be capped at 10.0 to match training data.
