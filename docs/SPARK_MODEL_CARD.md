# Spark-Portable 34-Feature XGBoost Model

## Purpose

Road network conflation model for scoring candidate geometry pairs as match/no_match.
Designed for distributed inference in Spark via broadcast booster + pandas_udf.

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Trees:** 168, max_depth 9 (Optuna + epsilon-compact selection; the full local model uses 170)
- **Holdout match F1:** 0.924 raw / 0.922 production (seed-42 segment-aware holdout, never seen during tuning)
- **CV F1 (match class):** 0.927 ± 0.011 (5-fold segment-aware cross-validation, training rows only)
- **LOO-by-type F1:** 0.884 ± 0.001 (33-fold leave-one-dataset-out — the cross-dataset metric, and the one this model is selected on; see below)
- **Predict throughput:** ~3.2M rows/sec single-node (see Inference Latency below)
- **Training data:** 5,428 labeled pairs across 34 datasets (5,457 loaded, 29 dropped by validation)
- **Feature version:** 2026-07-07.2
- **Exported:** 2026-08-07

## Why 34 Features (not 83)

The full matcher model uses 83 features including topology (22), graphlet (2), clustering (3),
and additional shape features. The 34-feature subset was selected for Spark portability:

**Included (34):** Computable from aligned geometry pairs alone, without graph topology or
spatial index queries — *and* carrying their weight on measured F1. 45 of the 83 clear the
first bar; these 34 clear both. See the Excluded split below:

| Category | Features | Count |
|----------|----------|-------|
| Hausdorff variants | hausdorff_distance_m, hausdorff_p95_m, mean_hausdorff_distance_m | 3 |
| Buffer IoU | buffer_iou_5m, buffer_iou_15m | 2 |
| Heading/Alignment | heading_delta, collinear_gap_ratio, edge_distance_rmse_m | 3 |
| Name similarity | name_levenshtein, name_jaro_winkler, name_token_sort, name_soundex, name_metaphone, has_name_ref, has_name_target, name_is_generic, name_numeric_match | 9 |
| Class | class_similarity | 1 |
| Lateral offset | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m | 3 |
| Coverage | ref_coverage, target_coverage, min_coverage, coverage_ratio | 4 |
| Sinuosity | sinuosity_ref, sinuosity_target | 2 |
| Heading consistency | heading_consistency_target | 1 |
| Length | min_length_m, aligned_length_m | 2 |
| Shape complexity | shape_complexity_target | 1 |
| Parallel sibling | offset_over_expected_halfwidth | 1 |
| Intersection overlap | post_node_continuation_m, endpoint_heading_divergence | 2 |

**Excluded (49)**, for two different reasons. Read the split carefully — it is the
part of this card that is easiest to get wrong, and `config.py` got it wrong until
2026-08-07.

*Excluded because Spark cannot compute them (38).* These need network-wide
structure that does not exist per candidate row:

- **Topology (22):** Require Overture connector graph + synthetic connector computation for target side
- **Graphlet (2) + Clustering (3):** Require local graph neighborhood construction
- **Endpoint proximity (3):** Require spatial index of connectors (could be added later — best candidate for next improvement)
- **Crossing angle (4):** Require spatial index of neighboring segments
- **Parallel sibling (4 of 5):** Require spatial index for nearby same-name segments

*Excluded on measured value, not feasibility (11).* These need nothing but the two
aligned geometries and the two name structs — the Spark job already holds both. Proven
computable bit-for-bit from a bare pair in `tests/test_spark_feature_expansion.py`;
measured for F1 / size / latency in
[research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md).

The **geometry block (10)** measures **−0.0034 LOO F1** for **16.24 µs/pair** as a tier,
so it is out on its own numbers, not on portability:

- **Additional shape/heading (5):** ref-side variants and deltas — low feature importance. Confirmed.
- **Vertex density (3):** Low discriminative power. Confirmed (`vertex_density_target` is the worst of all 17 at −0.0026).
- **Angle histogram (1):** Correlated with heading_delta + buffer_iou. Confirmed (−0.0001).
- **max_coverage (1):** `max(ref_coverage, target_coverage)` — derivable in SQL from columns the model already carries, and 4th by XGBoost gain in a 45-feature model. Still measures −0.0007: the splits do not transfer across datasets. A trap; see §3 of the research doc.

**`route_prefix_match` (1)** is out for a different reason again — it is **non-NaN on 1 of
5,532 labelled pairs (0.02%)**. It returns NaN unless *both* names canonicalize to a
recognized route designation (I-90, US-101, SR-520), which street and sidewalk layers do
not carry; XGBoost cannot split on a column missing in 5,531 of 5,532 rows. It is also the
only name feature needing a call of its own, so excluding it is precisely what makes the
name block cost **0.00 µs/pair**. Solo lift given up: +0.0002 (noise). Pinned by
`test_route_prefix_match_is_almost_always_nan`; revisit if the label base gains highway data.

Re-check as the label base grows. The geometry block losing is a 5,487-label result,
not a permanent one.

### The name block, added 2026-08-07

Six of the original 17 — `jaro_winkler`, `soundex`, `metaphone`, `has_name_ref`,
`has_name_target`, `name_is_generic` — **now ship**. This card previously excluded them as
"low marginal value given levenshtein + token_sort". Measured, the name block is worth
**+0.0033 LOO F1** at **0.00 µs/pair**: every one is a key in the `compute_name_similarity()`
dict that the exporter was already building and throwing away. A consumer already calling
that function just stops discarding six values.

The largest single contributor is `has_name_target` (+0.0043 solo, 90% of the gap to the
full 83-feature model). Its three biggest wins — `ke_nairobi_roads` +0.093,
`tn_tunis_ml_roads` +0.050, `co_bogota_bike_network` +0.039 — are all target layers with
`name_coverage_ratio = 0.0`: the flag lets the model read "this name comparison is
uninformative" instead of treating a neutral-filled `name_levenshtein` as weak evidence.
That is the regime a Spark job scoring arbitrary layers spends most of its time in.
`ch_geneva_pedestrian_network` also has zero name coverage and *regressed* by 0.056, so
this is a strong pattern with a real counterexample, not a law.

The block was taken as a block rather than as `has_name_target` alone: the two score the same
within noise, and a lone indicator flag with none of its companion name context is the more
overfit selection, not the safer one. `route_prefix_match` is the one member left out, on the
fill-rate grounds described in the Excluded section above — dropping it is what takes the
block's marginal cost to zero.

### LOO before/after

5 seeds, 33 folds, same harness as the research doc, isolating the feature change from the
retune:

| Config | Features | Params | LOO F1 | Δ |
|---|---|---|---|---|
| Before | 28 | 28-tuned | 0.8740 ± 0.0010 | — |
| Feature change only | 34 | 28-tuned | 0.8775 ± 0.0008 | +0.0035 |
| **This model** | **34** | **34-tuned** | **0.8839 ± 0.0014** | **+0.0099** |

Per type group, before → after: `road_poor` 0.8767 → 0.9042, `road_good` 0.8803 → 0.8911,
`other` 0.8499 → 0.8568, `sidewalk` 0.8700 → 0.8692 (flat). For reference the full
83-feature local model scores 0.8788 on this metric.

The retune contributes +0.0027 on top of the name block's +0.0030, for +0.0057 total — which
puts the Spark model slightly *above* the full 83-feature local model's 0.8788 on this metric.
Do not over-read that: the two are within ~0.001 of each other and the local model is not
tuned for LOO.

## Performance Comparison

| Model | Features | Match F1 | Accuracy |
|-------|----------|----------|----------|
| Spark formula (hand-tuned) | 6 | 0.859 | 80.9% |
| Previous Spark model (28-feat) | 28 | 0.909 | 89.2% |
| **This model (34-feat)** | **34** | **0.924** | **90.8%** |
| Full matcher model | 78 | 0.930 | 91.7% |

**These rows are not all on the same footing — read the caveats before quoting the deltas.**

* The 28-feature row was measured 2026-07-03. The label base has since been re-keyed and
  fully backfilled (#473), so part of the 0.909 → 0.924 gap is corrected input data
  rather than the added features, and part is the hyperparameter retune. For a
  like-for-like comparison where the feature set is the *only* moving part, use the LOO
  table below (0.8740 → 0.8775 for the features alone) or the 5-seed tier table in
  [research/spark_feature_expansion_2026-08-07.md](../research/spark_feature_expansion_2026-08-07.md).
* The full-model row is the 78-feature model as it stood when this table was measured;
  the local model has since grown to 83 features, and the row has not been re-measured.
  The research doc's reference tier puts the current 83-feature model at 0.9268 holdout /
  0.8788 LOO.

All rows are honest seed-42 holdout metrics (hyperparameters tuned leakage-free, holdout
never seen during tuning). Earlier versions of this card quoted 0.932 for the 28-feature
model — that figure came from hyperparameters tuned with the holdout included, so it was
mildly optimistic. The 34-feature model captures ~99% of the full model's match F1 with
~41% of the features, and on the cross-dataset LOO metric it scores *above* the full
model (0.8839 vs 0.8788) — which says more about the full model not being tuned for LOO
than about the Spark model being better, and should not be over-read.

## Hyperparameters (Optuna-tuned)

```
n_estimators: 168
learning_rate: 0.023
max_depth: 9
min_child_weight: 1
subsample: 0.671
colsample_bytree: 0.802
gamma: 0.233
reg_alpha: 0.595
reg_lambda: 0.032
max_bin: 164
scale_pos_weight: 0.648  (computed from training labels, not tuned)
```

Retuned 2026-08-07 with `scripts/tune_model.py --feature-set spark` (**300** Optuna trials,
TPESampler seed=42) using the leakage-free protocol: the seed-42 holdout is discarded
before tuning and the objective is mean match F1 over an inner GroupKFold (segment-aware)
cross-validation on the training portion only, minus a size penalty of 0.00001 F1 per
tree above 100. Because inference speed is critical for the Spark deployment, the final
params come from **epsilon-compact selection**: among all trials within 0.003 raw CV F1
of the best, the one with the lowest traversal cost (`n_estimators * max_depth`) is
selected — here 168 trees x depth 9 (CV F1 0.9270) over the best-F1 trial's
216 x 10 (CV F1 0.9292). Source of truth: `SPARK_PORTABLE_XGB_PARAMS` in `config.py`.

The retune was **required** by the widening, not an optimization pass: the prior values
were fitted to 28 features on 2026-07-03. It landed on a point better and cheaper than
the old one on every axis — CV F1 0.9216 → 0.9270, traversal cost 2240 → 1512 — so the
name block was not paid for with accuracy or model complexity.

### Why 300 trials, and why CV F1 alone is not enough

The **100-trial** run of this identical search selected 353 x 7 (CV F1 0.9257, cost
2471), and that point was **worse than doing nothing**: LOO F1 0.8763 against 0.8777 for
simply reusing the old 28-feature hyperparameters on this feature set, and ~30% slower to
score. The rule's arithmetic was correct — 353 x 7 really was the cheapest eligible trial
in that run — but epsilon-compact optimizes *inner-CV F1*, a within-distribution metric,
and breaks ties on `n_estimators * max_depth`, a proxy that predicted +10% traversal cost
for a point that measured ~40% slower under load. A shallow-wide shape can therefore win
the rule while generalizing worse across datasets.

At 300 trials the search found a compact-and-general point, and the rule's own pick is
also the best of its eligible set on LOO, so no LOO-based override was applied and no
selection optimism was banked:

| Candidate | LOO F1 | Shape | Cost |
|---|---|---|---|
| Old 28-tuned params (control) | 0.8777 ± 0.0010 | 224 × 10 | 2240 |
| **Selected (trial 139)** | **0.8837 ± 0.0018** | **168 × 9** | **1512** |
| 2nd cheapest eligible | 0.8811 ± 0.0014 | 157 × 10 | 1570 |
| 4th cheapest eligible | 0.8815 ± 0.0010 | 163 × 10 | 1630 |
| 3rd cheapest eligible | 0.8801 ± 0.0017 | 179 × 9 | 1611 |
| Best-CV-F1 trial | 0.8823 ± 0.0007 | 216 × 10 | 2160 |

**Retune at ≥300 trials and sanity-check the selected point on LOO before shipping it.**
CV F1 alone does not catch this failure mode.

## Inference Latency

`.predict` on 1M random float32 rows, median of 7 runs, XGBoost `hist`, both models
benchmarked head-to-head on the same idle Linux box. This table supersedes the earlier
Apple Silicon numbers:

| Model | Features | Trees x depth | 1M rows | Throughput | Artifact |
|-------|----------|--------------|---------|------------|----------|
| Previous | 28 | 224 x 10 | 0.299 s | ~3.35M rows/sec | 1176 KB |
| **This model** | **34** | **168 × 9** | **0.310 s** | **~3.22M rows/sec** | **1268 KB** |

**Net cost: ~4% throughput, +8% artifact size.** Two effects cancel most of the way out —
6 more columns to marshal into the DMatrix costs a few percent, while the smaller tree
ensemble (traversal cost 1512 vs 2240) gives most of it back.

Consumer-side feature computation adds **0.00 µs/pair**: all 6 added features are keys
`compute_name_similarity()` already returns.

⚠️ **Benchmark under load and you will get nonsense.** Interim measurements taken on this
box while an Optuna search and a LOO sweep were running reported 1.18M and 0.93M rows/sec
for these same two artifacts — a 3x understatement that also inverted the ranking's
magnitude. Only quote throughput measured on an idle machine.

## How to Reproduce

```bash
# Single command: train + export
uv run crosswalk export-spark-model
```

This uses `SPARK_PORTABLE_FEATURES` from `config.py` (inclusive list of the 34 features)
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
