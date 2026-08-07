# Spark-Portable 35-Feature XGBoost Model

## Purpose

Road network conflation model for scoring candidate geometry pairs as match/no_match.
Designed for distributed inference in Spark via broadcast booster + pandas_udf.

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Trees:** 209, max_depth 10 (Optuna + epsilon-compact selection; the full local model uses 170)
- **Holdout match F1:** 0.923 raw / 0.927 production (seed-42 segment-aware holdout, never seen during tuning)
- **CV F1 (match class):** 0.925 ± 0.007 (5-fold segment-aware cross-validation, training rows only)
- **LOO-by-type F1:** 0.880 ± 0.001 (33-fold leave-one-dataset-out — the cross-dataset metric, see the research doc)
- **Predict throughput:** ~1.1M rows/sec single-node (see Inference Latency below)
- **Training data:** 5,428 labeled pairs across 34 datasets (5,457 loaded, 29 dropped by validation)
- **Feature version:** 2026-07-07.2
- **Exported:** 2026-08-07

## Why 35 Features (not 83)

The full matcher model uses 83 features including topology (22), graphlet (2), clustering (3),
and additional shape features. The 35-feature subset was selected for Spark portability:

**Included (35):** Computable from aligned geometry pairs alone, without graph topology or
spatial index queries — *and* carrying their weight on measured F1. 45 of the 83 clear the
first bar; these 35 clear both. See the Excluded split below:

| Category | Features | Count |
|----------|----------|-------|
| Hausdorff variants | hausdorff_distance_m, hausdorff_p95_m, mean_hausdorff_distance_m | 3 |
| Buffer IoU | buffer_iou_5m, buffer_iou_15m | 2 |
| Heading/Alignment | heading_delta, collinear_gap_ratio, edge_distance_rmse_m | 3 |
| Name similarity | name_levenshtein, name_jaro_winkler, name_token_sort, name_soundex, name_metaphone, has_name_ref, has_name_target, name_is_generic, name_numeric_match, route_prefix_match | 10 |
| Class | class_similarity | 1 |
| Lateral offset | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m | 3 |
| Coverage | ref_coverage, target_coverage, min_coverage, coverage_ratio | 4 |
| Sinuosity | sinuosity_ref, sinuosity_target | 2 |
| Heading consistency | heading_consistency_target | 1 |
| Length | min_length_m, aligned_length_m | 2 |
| Shape complexity | shape_complexity_target | 1 |
| Parallel sibling | offset_over_expected_halfwidth | 1 |
| Intersection overlap | post_node_continuation_m, endpoint_heading_divergence | 2 |

**Excluded (48)**, for two different reasons. Read the split carefully — it is the
part of this card that is easiest to get wrong, and `config.py` got it wrong until
2026-08-07.

*Excluded because Spark cannot compute them (38).* These need network-wide
structure that does not exist per candidate row:

- **Topology (22):** Require Overture connector graph + synthetic connector computation for target side
- **Graphlet (2) + Clustering (3):** Require local graph neighborhood construction
- **Endpoint proximity (3):** Require spatial index of connectors (could be added later — best candidate for next improvement)
- **Crossing angle (4):** Require spatial index of neighboring segments
- **Parallel sibling (4 of 5):** Require spatial index for nearby same-name segments

*Excluded on measured value, not feasibility (10).* These need nothing but the two
aligned geometries — the Spark job already holds them. Proven computable bit-for-bit
from a bare pair in `tests/test_spark_feature_expansion.py`; measured for F1 / size /
latency in
[research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md).
As a tier the block measures **−0.0034 LOO F1** for **16.24 µs/pair**, so it is out on
its own numbers, not on portability:

- **Additional shape/heading (5):** ref-side variants and deltas — low feature importance. Confirmed.
- **Vertex density (3):** Low discriminative power. Confirmed (`vertex_density_target` is the worst of all 17 at −0.0026).
- **Angle histogram (1):** Correlated with heading_delta + buffer_iou. Confirmed (−0.0001).
- **max_coverage (1):** `max(ref_coverage, target_coverage)` — derivable in SQL from columns the model already carries, and 4th by XGBoost gain in a 45-feature model. Still measures −0.0007: the splits do not transfer across datasets. A trap; see §3 of the research doc.

Re-check as the label base grows. The geometry block losing is a 5,487-label result,
not a permanent one.

### The name block, added 2026-08-07

The other 7 of the original 17 — `jaro_winkler`, `soundex`, `metaphone`, `has_name_ref`,
`has_name_target`, `name_is_generic`, `route_prefix_match` — **now ship**. This card
previously excluded them as "low marginal value given levenshtein + token_sort". Measured,
the full name block is worth **+0.0033 LOO F1** for **1.18 µs/pair**: 6 of the 7 are keys
in the `compute_name_similarity()` dict that the exporter was already building and throwing
away, so the only new computation is one `compute_route_prefix_match()` call.

The largest single contributor is `has_name_target` (+0.0043 solo, 90% of the gap to the
full 83-feature model). Its three biggest wins — `ke_nairobi_roads` +0.093,
`tn_tunis_ml_roads` +0.050, `co_bogota_bike_network` +0.039 — are all target layers with
`name_coverage_ratio = 0.0`: the flag lets the model read "this name comparison is
uninformative" instead of treating a neutral-filled `name_levenshtein` as weak evidence.
That is the regime a Spark job scoring arbitrary layers spends most of its time in.
`ch_geneva_pedestrian_network` also has zero name coverage and *regressed* by 0.056, so
this is a strong pattern with a real counterexample, not a law.

The block was shipped whole rather than as `has_name_target` alone: the two score the same
within noise, and a lone indicator flag with none of its companion name context is the more
overfit selection, not the safer one.

**Caveat on `route_prefix_match` — the one member that is not free.** It returns NaN unless
*both* names canonicalize to a recognized route designation (I-90, US-101, SR-520), which
street and sidewalk layers essentially never carry. Measured over the whole feature store it
is non-NaN on **1 of 5,532 labelled pairs (0.02%)** — a column XGBoost cannot split on. It is
also the *only* member of the block that costs new computation: the entire 1.18 µs/pair
marginal cost of this widening is that one call, since the other 6 are dict keys the exporter
was already discarding. Dropping it would make the name block literally free at no measurable
F1 cost (solo lift +0.0002, i.e. noise). It ships because the block was taken whole, which is
a deliberate call rather than an oversight; pinned by
`test_route_prefix_match_is_almost_always_nan`. Revisit if the label base gains highway data.

### LOO before/after

5 seeds, 33 folds, same harness as the research doc, isolating the feature change from the
retune:

| Config | Features | Params | LOO F1 |
|---|---|---|---|
| Before | 28 | 28-tuned | 0.8740 ± 0.0010 |
| Feature change only | 35 | 28-tuned | 0.8770 ± 0.0003 |
| **This model** | **35** | **35-tuned** | **0.8797 ± 0.0007** |

The retune contributes +0.0027 on top of the name block's +0.0030, for +0.0057 total — which
puts the Spark model slightly *above* the full 83-feature local model's 0.8788 on this metric.
Do not over-read that: the two are within ~0.001 of each other and the local model is not
tuned for LOO.

## Performance Comparison

| Model | Features | Match F1 | Accuracy |
|-------|----------|----------|----------|
| Spark formula (hand-tuned) | 6 | 0.859 | 80.9% |
| Previous Spark model (28-feat) | 28 | 0.909 | 89.2% |
| **This model (35-feat)** | **35** | **0.923** | **90.7%** |
| Full matcher model | 78 | 0.930 | 91.7% |

**These rows are not all on the same footing — read the caveats before quoting the deltas.**

* The 28-feature row was measured 2026-07-03. The label base has since been re-keyed and
  fully backfilled (#473), so part of the 0.909 → 0.923 gap is corrected input data
  rather than the added features. For a like-for-like comparison where the feature set is
  the *only* moving part, use the 5-seed tier table in
  [research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md):
  28 features scores 0.9217 holdout / 0.8740 LOO and 35 scores 0.9245 / 0.8773 under
  identical hyperparameters.
* The full-model row is the 78-feature model as it stood when this table was measured;
  the local model has since grown to 83 features, and the row has not been re-measured.
  The research doc's reference tier puts the current 83-feature model at 0.9268 holdout /
  0.8788 LOO.

All rows are honest seed-42 holdout metrics (hyperparameters tuned leakage-free, holdout
never seen during tuning). Earlier versions of this card quoted 0.932 for the 28-feature
model — that figure came from hyperparameters tuned with the holdout included, so it was
mildly optimistic. The 35-feature model captures ~99% of the full model's match F1 with
~42% of the features.

## Hyperparameters (Optuna-tuned)

```
n_estimators: 209
learning_rate: 0.019
max_depth: 10
min_child_weight: 3
subsample: 0.808
colsample_bytree: 0.899
gamma: 0.954
reg_alpha: 0.594
reg_lambda: 0.324
max_bin: 149
scale_pos_weight: 0.648  (computed from training labels, not tuned)
```

Retuned 2026-08-07 with `scripts/tune_model.py --feature-set spark` (100 Optuna trials,
TPESampler seed=42) using the leakage-free protocol: the seed-42 holdout is discarded
before tuning and the objective is mean match F1 over an inner GroupKFold (segment-aware)
cross-validation on the training portion only, minus a size penalty of 0.00001 F1 per
tree above 100. Because inference speed is critical for the Spark deployment, the final
params come from **epsilon-compact selection**: among all trials within 0.003 raw CV F1
of the best, the one with the lowest traversal cost (`n_estimators * max_depth`) is
selected — here 209 trees x depth 10 (CV F1 0.9262) over the best-F1 trial's
509 x 10 (CV F1 0.9276). Source of truth: `SPARK_PORTABLE_XGB_PARAMS` in `config.py`.

The retune was **required** by the widening, not an optimization pass: the prior values
were fitted to 28 features on 2026-07-03. It happened to land on a point that is both
better and cheaper than the old one (0.9262 vs 0.9216 raw CV F1, cost 2090 vs 2240), so
the name block was not paid for with either accuracy or latency headroom.

## Inference Latency

`.predict` on 1M random float32 rows, median of 5 runs, XGBoost `hist`. The 28- and
35-feature models were re-benchmarked head-to-head on the same (Linux) box, so this
table supersedes the earlier Apple Silicon numbers:

| Model | Features | Trees x depth | 1M rows | Throughput | Artifact |
|-------|----------|--------------|---------|------------|----------|
| Previous | 28 | 224 x 10 | 0.84 s | ~1.18M rows/sec | 1176 KB |
| **This model** | **35** | **209 x 10** | **0.90 s** | **~1.11M rows/sec** | **1059 KB** |

Absolute numbers are machine-specific (this box was under concurrent load); the ratio is
what matters. The widening costs **~6% throughput** — 7 more columns to marshal into the
DMatrix, partly offset by 15 fewer trees — and *saves* 10% on artifact size. Feature
computation on the consumer side adds 1.18 µs/pair, 0.5% of the ~224 µs a candidate pair
already costs.

## How to Reproduce

```bash
# Single command: train + export
uv run crosswalk export-spark-model
```

This uses `SPARK_PORTABLE_FEATURES` from `config.py` (inclusive list of the 35 features)
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
