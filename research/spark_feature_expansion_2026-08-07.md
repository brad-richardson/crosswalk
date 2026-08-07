# Widening the Spark-portable feature set: what is actually addable, and what it buys

**Date:** 2026-08-07 · **Status:** research + demo, **now applied.**

The measurements below were taken with nothing shipped — `src/crosswalk/_model/*`,
`config.py`, and `SPARK_PORTABLE_FEATURES` were untouched throughout, so every
number is a clean read of the feature set as the only moving part. The name block
minus `route_prefix_match` was subsequently added, taking
`SPARK_PORTABLE_FEATURES` from 28 to 34 with a hyperparameter retune and a reship
of the Spark artifacts. Nothing in §§1-3 has been re-run against that change;
treat those tiers as measurements relative to the **28-feature** baseline, which
is what they are.

## Recommendation in one line

Add **`has_name_target`** — one feature, zero marginal compute, **+0.0043 LOO F1**
(4× the seed noise floor), which recovers **90% of the entire gap** between the
28-feature Spark model and the full 83-feature local model. Optionally add
`has_name_ref` alongside it for symmetry. Add **nothing else**: the other 15
addable features are collectively worth nothing, and the 10 geometry ones are
net negative.

## Decision (2026-08-07): ship the name block minus `route_prefix_match`

**Not taken.** The call is to add the **name block** — 6 features, 34 total —
rather than the single-feature tier this analysis recommends. The 10 geometry
features are **not** added, and neither is `route_prefix_match` (see the fill-rate
finding below). The measurements below were all taken with nothing shipped. What follows is the
reasoning for overriding the ranking they produce.

The shipped set is tier t2 minus one feature. t2 measured +0.0033 LOO F1 at
1.18 µs/pair; dropping the one member that carries **all** of that cost and is
NaN on 99.98% of pairs takes it to **0.00 µs/pair**.

* **The gap between t2 and t2a is inside the noise.** t2a is 0.8783 ± 0.0011 and
  t2 is 0.8773 ± 0.0010. A 0.0010 difference against those bands is not a result.
  This analysis is entitled to say "the name block beats the 28-feature baseline";
  it is not entitled to say "one of its features beats the other six".
* **A lone `has_name_target` is the more overfit choice, not the safer one.**
  §2 already concedes the +0.0043 "rests on a handful of datasets" and has a real
  counterexample (`ch_geneva_pedestrian_network`, −0.056). Selecting the single
  best-scoring feature out of 17 candidates measured on 33 folds is a selection
  procedure with its own optimism, and the winner here is a bare indicator flag
  with none of the name context that would let the model condition on it. The
  full name block gives the flag its companions; it scores the same and rests on
  less.
* **The 6 shipped name features are free.** Every one is a key in the
  `compute_name_similarity()` dict that the exporter already builds and discards —
  **0.00 µs/pair marginal**. A consumer already calling that function just stops
  dropping them. (`route_prefix_match` is the 7th and the only one needing a call
  of its own, at 1.18 µs/pair; excluding it is what makes this figure zero.)
* **The geometry block is excluded on measurement, and that is the difference
  from t4.** t3 (geometry alone) is −0.0034; t4 (all 17) is +0.0034, statistically
  identical to t2's +0.0033. So the 10 geometry features contribute nothing while
  costing 16.24 µs/pair — 93% of the marginal compute of "all 17" for 0.0001 of
  F1. Cheap is not the same as worth adding, and this is the tier where that
  distinction has teeth. Revisit per §4.6 as the label base grows: the block
  losing is a 5,457-label result, not a permanent one.

### Why `route_prefix_match` is excluded: it is 99.98% NaN

This analysis measured its *value* (+0.0002 solo, noise) and its *cost* (1.18
µs/pair) but never checked how often it produces a value at all. It returns NaN
unless **both** names canonicalize to a recognized route designation (I-90,
US-101, SR-520), and street/sidewalk layers do not carry those. Measured across
the whole feature store: **non-NaN on 1 of 5,532 stored labelled pairs (0.02%)**,
the single hit in `ca_toronto_roads`. XGBoost cannot split on that.

It is also the only member of the block that costs new computation, so excluding
it takes the widening from "close to free" to **actually free** — 34 features at
0.00 µs/pair. Pinned by `test_route_prefix_match_is_almost_always_nan` in the
follow-up PR, which fails in either direction.

The same sparsity makes it degenerate in this PR's parity fixture: it is NaN on
all 180 sample pairs, so `test_addable_features_match_authoritative_computation`
compares it against a constant and a stub returning NaN would pass. Pinned by
`test_degenerate_fixture_features_are_exactly_the_known_two`, together with
`name_is_generic` (constant 0.0 on the same fixture). The "bit-for-bit" claim is
therefore strong for 15 of the 17 and weak for those two.

**Method lesson for §3's ranking:** a solo LOO delta measures whether a feature
*helps* and says nothing about whether it is *populated*. A column that is NaN
everywhere scores exactly like one that is present and useless. The other 16
candidates should be fill-rate checked before any future tier ships on that
ranking alone.

### Also found during implementation: the retune needs ≥300 trials, checked on LOO

§4.2 says to retune `SPARK_PORTABLE_XGB_PARAMS`, which is right, but understates how
carefully. The first retune (100 trials, the count the 2026-07-03 tune used) selected a
point that was **worse than not retuning at all**: LOO 0.8763 against 0.8777 for reusing
the old 28-feature hyperparameters on the same 34 features, and ~30% slower to score.

The cause is that `scripts/tune_model.py`'s epsilon-compact selection optimizes *inner-CV
F1* — a within-distribution metric — and breaks ties on `n_estimators * max_depth`, a
proxy that predicted +10% traversal cost for a point that measured ~40% slower. A
shallow-wide shape (353 × 7) can win that rule while generalizing worse across datasets.
The rule was not misapplied; the 100-trial search simply had not found a
compact-and-general point.

At 300 trials it did (168 × 9, CV F1 0.9270, cost 1512 — cheaper than the 28-feature
model's 2240), and the rule's own pick was also the best of its eligible set on LOO
(0.8837 vs 0.8811 / 0.8815 / 0.8801 for the next three cheapest, and 0.8823 for the
best-CV-F1 trial). So the selection rule was **not** overridden and no LOO selection
optimism was banked — the LOO scoring was a check, not a chooser.

The measured end state, 5 seeds:

| Config | Features | Params | LOO F1 | Δ |
|---|---|---|---|---|
| Before | 28 | 28-tuned | 0.8740 ± 0.0010 | — |
| Feature change only | 34 | 28-tuned | 0.8775 ± 0.0008 | +0.0035 |
| Shipped | 34 | 34-tuned (300 trials) | **0.8839 ± 0.0014** | **+0.0099** |

Note the retune is worth more than the features (+0.0064 vs +0.0035) — but that split is
**confounded**. The comparison is 34 features + a fresh 300-trial search against 28 features
+ a five-week-old 100-trial one. A fresh search on the 28-feature set was never run, so "the
features helped" and "any fresh search helps" are not separated. §3's recommended tier (t2a,
29 features, LOO 0.8783) was likewise never retuned — §5 lists that as explicitly not
measured. The experiment that would settle it is a 300-trial retune of the 28-feature set,
and it has not been run. Standing guidance, now in `config.py`: retune at >= 300 trials and
sanity-check the selected point on LOO before shipping. CV F1 alone does not catch this.

Consequences for the follow-up PR, on top of §4: `SPARK_PORTABLE_XGB_PARAMS` was
Optuna-tuned for 28 features, so the retune (§4.2) is a required step rather than
an optional one. The tf-data-platform coordination in §4.5 is 6 new columns rather
than one, but all 6 are pure string ops on the resolved name pair — no new geometry
work on the consumer side, no new call at all if the job already invokes
`compute_name_similarity()`, and no new dependency (`jellyfish` and `rapidfuzz` are
already core `crosswalk-py` deps).

---

## 1. Feasibility: the classification, verified by computation

The starting hypothesis (17 excluded features "look like they need nothing but
the two geometries or the two name strings") is **correct**, and it is now
enforced by a test rather than argued.

`tests/test_spark_feature_expansion.py::compute_spark_addable_features` takes
exactly four inputs — `ref_geom`, `target_geom`, `ref_names`, `target_names` —
and reproduces the authoritative `compute_pair_features()` output for all 17
columns **bit-for-bit** on 180 real labelled pairs from three datasets
(`us_boston_streets`, `de_berlin_roads`, `us_seattle_sidewalks`). No STRtree, no
connector graph, no topology dict, no native-target degrees, no corridor model.

The complement is enforced too: withholding that context NaNs out all 38
remaining excluded features, the shipped 28 all still compute (so the NaN test
is not passing on a broken pipeline), and the three buckets are asserted to
partition the 55 exclusions exactly — a new feature cannot land in
`FEATURE_COLUMNS` without a Spark-feasibility verdict.

### The 55 excluded features, by why they are excluded

| Bucket | n | Verdict |
|---|---|---|
| Name similarity | 7 | **Addable.** Pure string ops on the resolved variant pair. |
| Per-geometry / pair-geometry | 10 | **Addable.** One extra pass over the aligned sublines. |
| Topology | 22 | Not addable — Overture connectors projected onto both sides + the target network's own endpoint-cluster Union-Find. |
| Endpoint/Connectivity | 3 | Not addable — `scipy.cKDTree` over **every** endpoint in the target layer (`spatial_context.py`, `SpatialContextIndex.kdtree`). |
| Crossing angle | 4 | Not addable — STRtree over the whole Overture layer. |
| Parallel sibling | 4 | Not addable — corridor/sibling search over an STRtree. |
| Graphlet | 2 | Not addable — connector graph. |
| Clustering | 3 | Not addable — connector graph node features. |

### Where the framing needs correcting

1. **The alignment is free, and it is the expensive part.**
   `linestring_alignment(reference, target)` is a pure pairwise function
   (`alignment.py:627`). The Spark job already pays it — `ref_coverage`,
   `aligned_length_m`, `sinuosity_ref` and `min_length_m` all need the aligned
   sublines. Every addable feature rides on work already done.

2. **Two of the 10 "geometry" features need no computation at all.**
   `max_coverage` is `max(ref_coverage, target_coverage)` and `sinuosity_delta`
   is `abs(sinuosity_ref - sinuosity_target)` — both over columns already in the
   28-feature vector. A consumer could add them in SQL. (Measured, asserted, and
   still not worth adding — see §3.)

3. **6 of the 7 name features are already computed and thrown away.**
   `compute_name_similarity()` (`semantic.py:316`) returns 12 keys in one call,
   covering **8 of the 10** `Name Similarity` features. The exporter keeps
   `name_levenshtein` and `name_token_sort` and discards `jaro_winkler`,
   `soundex`, `metaphone`, `has_name_ref`, `has_name_target`, `name_is_generic`.
   The other **two** name features are separate calls:
   `compute_name_numeric_match()` (`compute.py:354`, already shipped) and
   `compute_route_prefix_match()` (`compute.py:358`). So the free-by-construction
   set is the 6 discarded keys, not all 7 excluded name features. **If the Spark job calls into `crosswalk-py`** (which is how the
   booster and manifest are already consumed — `crosswalk/spark.py`), the
   recommended feature costs literally zero: it is a dict key currently dropped
   on the floor.

4. **No new dependency.** `jellyfish>=1.0` (soundex/metaphone) and
   `rapidfuzz>=3.5` (Jaro-Winkler) are already **core** `crosswalk-py`
   dependencies, not extras (`pyproject.toml:37-38`). (Not needed for the
   recommendation, but it removes the portability objection for the rest of the
   name block. Note Spark SQL's built-in `soundex()` is *not* a drop-in — the
   repo applies it to a "key content word" after `_normalize_street_name` +
   road-type-word filtering, `_soundex_key_word` at `semantic.py:223`.)

5. **`offset_over_expected_halfwidth` is a category-name trap.** It sits in the
   Parallel Sibling category but is `lateral_offset / class-expected half width`
   — no sibling search — which is why it already ships while the other four
   Parallel Sibling features cannot. Pinned by a test.

6. **The repo already disagrees with itself about *why* these 17 are out.**
   `cli/main.py:1279` and `config.py:549-550` claim infeasibility ("no graph
   topology, no spatial indexes, no connector data required").
   `docs/SPARK_MODEL_CARD.md:50-53` gives *value* reasons for exactly these 17
   ("low marginal value given levenshtein + token_sort", "low feature
   importance", "low discriminative power", "correlated with heading_delta +
   buffer_iou"). The model card is the honest one; the CLI/config comment is
   over-broad. The measurements below settle the value question — and largely
   **vindicate the card**, except on `has_name_*`.

---

## 2. Measured tradeoffs

All tiers trained with `SPARK_PORTABLE_XGB_PARAMS` and the same grouped-holdout
protocol `crosswalk export-spark-model` uses, so the **only** moving part is the
feature set. Every number is mean ± population std over **5 seeds**
(42, 1, 2, 3, 4), full label set. `LabelStore.load_all` returns 5,487 rows /
34 datasets; after dropping 30 `unsure` labels the trained and evaluated set is
**5,457 match/no_match rows across 33 datasets**, which is why the LOO run has 33
folds rather than 34.

* `cv_f1` — GroupKFold CV over training rows (what `export-spark-model` prints).
* `test_f1` — production holdout F1 (calibrated, `settings.scoring_match_threshold`).
* `loo_f1` — LOO-by-type CV with `eval_utils.py` semantics: 33 folds, one dataset
  held out per fold, trained on all others. **This is the metric that matters**
  for a Spark job scoring layers it has never seen, and it has the tightest
  noise floor (±0.0010) because the folds are deterministic — only XGBoost's
  row/column sampling moves between seeds.

| Tier | n | cv_f1 | test_f1 | **loo_f1** | ΔLOO | model KB | infer µs/row | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| **t0 baseline (shipped)** | **28** | 0.9182 ± 0.0019 | 0.9217 ± 0.0022 | **0.8740 ± 0.0010** | — | 1167 | 0.57 ± 0.11 | 403 |
| t1 + free derived (`max_coverage`, `sinuosity_delta`) | 30 | 0.9189 ± 0.0015 | 0.9233 ± 0.0040 | 0.8726 ± 0.0009 | **−0.0014** | 1164 | 0.67 ± 0.19 | 413 |
| **t2a + `has_name_target`** | **29** | 0.9189 ± 0.0024 | **0.9260 ± 0.0035** | **0.8783 ± 0.0011** | **+0.0043** | 1164 | 0.53 ± 0.03 | 413 |
| t2b + `has_name_ref/target` | 30 | 0.9201 ± 0.0023 | 0.9245 ± 0.0024 | **0.8783 ± 0.0017** | **+0.0043** | 1158 | 0.61 ± 0.07 | 414 |
| t2 + all 7 name features | 35 | 0.9193 ± 0.0024 | 0.9245 ± 0.0034 | 0.8773 ± 0.0010 | +0.0033 | 1159 | 0.55 ± 0.01 | 420 |
| t3 + all 10 geometry features | 38 | 0.9186 ± 0.0016 | 0.9223 ± 0.0028 | 0.8706 ± 0.0013 | **−0.0034** | 1192 | 0.69 ± 0.08 | 428 |
| t4 + all 17 feasible | 45 | 0.9208 ± 0.0018 | 0.9246 ± 0.0024 | 0.8774 ± 0.0016 | +0.0034 | 1188 | 0.68 ± 0.01 | 439 |
| *ref: full local model (not Spark-feasible)* | *83* | *0.9220 ± 0.0017* | *0.9268 ± 0.0024* | *0.8788 ± 0.0003* | *+0.0048* | *1225* | *0.91 ± 0.09* | *447* |

Reproduce (~15 min, `OMP_NUM_THREADS=8`):

```bash
uv run python research/spark_feature_expansion.py --seeds 42,1,2,3,4 \
    --out research/results/spark_feature_expansion_2026-08-07.json
```

### Reading the table

* **One feature does the whole job.** `has_name_target` alone (+0.0043) beats
  the full 7-feature name block (+0.0033) and the full 17-feature set (+0.0034),
  and lands within 0.0005 of the *entire* 83-feature local model on LOO
  (0.8783 vs 0.8788) — **90% of the gap for one free column**. It also gives the
  best holdout F1 of any tier except the 83-feature reference (0.9260).
* **More is worse.** Every tier that adds features on top of `has_name_target`
  scores lower on LOO. At 5,457 evaluated labels the model does not have the data to use
  them, and they dilute.
* **The geometry block is a net negative** (−0.0034 LOO) and is the part that
  actually costs inference time and model size. The model card's "low
  discriminative power / correlated with heading_delta + buffer_iou" verdict
  holds up under measurement.
* **`max_coverage` and `sinuosity_delta` — the two literally free ones — also
  lose** (−0.0014). See the trap in §3.
* **Speed and memory are never the constraint.** Even the 45-feature tier runs
  at 0.68 µs/row (~1.5M rows/sec single node) on a 1.2 MB artifact. The
  recommended 29-feature tier is indistinguishable from the baseline on all
  three of size, latency, and RSS. The µs/row column carries ±0.03–0.19 of noise
  because the box was under concurrent load; the ordering (more features →
  slower) is stable, the absolute numbers are not precise.

### Marginal feature-computation cost (per candidate pair)

From `tests/test_spark_feature_expansion.py::test_addable_feature_marginal_cost`
(180 real pairs, median of 7 sweeps, single-pair Python loop — *not* the batched
`_compute_feature_chunk` path, so absolute numbers are pessimistic and only the
ratios are claims):

| Stage | µs/pair |
|---|---|
| align + subline + coords (already paid by the 28) | 194.71 |
| `compute_name_similarity` call (already paid) | 11.93 |
| target-side sinuosity / heading / complexity (already paid) | 17.39 |
| **= shipped-28 per-pair subtotal** | **224.03** |
| + `route_prefix_match` → unlocks all 7 name features | **1.18** |
| + 5 new geometry calls → unlocks all 10 geometry features | 16.24 |
| **= marginal cost of all 17** | **17.43 (7.8%)** |

The recommended feature (`has_name_target`) is inside the already-paid 11.93 µs
row: **0.00 µs marginal**. The 10 geometry features cost 16.24 µs/pair for a
negative F1 return.

### Where the lift comes from

Paired per-dataset LOO deltas for `has_name_target` (mean over 5 seeds):
**18 datasets better / 13 worse / 2 unchanged, median +0.0008**, so the mean is
positive on the bulk *and* carries three large wins:

| Dataset | baseline LOO F1 | +`has_name_target` | Δ | target `name_coverage_ratio` |
|---|---|---|---|---|
| `ke_nairobi_roads` | 0.8487 | 0.9418 | **+0.0930** | 0.0 |
| `tn_tunis_ml_roads` | 0.9495 | 0.9993 | **+0.0499** | 0.0 |
| `co_bogota_bike_network` | 0.9434 | 0.9823 | **+0.0389** | 0.0 |
| `us_usfs_flathead` | 0.8281 | 0.8400 | +0.0119 | — |
| `fi_helsinki_roads` | 0.8888 | 0.8960 | +0.0072 | 0.508 |
| … | | | | |
| `us_usfs_lolo` | 0.8740 | 0.8626 | −0.0114 | — |
| `ch_geneva_pedestrian_network` | 0.7898 | 0.7333 | **−0.0564** | 0.0 |

**Mechanism:** the three biggest winners are all target layers with
`name_coverage_ratio = 0.0` — no names at all. `has_name_target` gives the model
an explicit "this name comparison is uninformative" flag instead of letting it
read a neutral-filled `name_levenshtein` as weak evidence. That is precisely the
regime a Spark job scoring arbitrary layers against Overture spends most of its
time in, which makes a +0.0043 mean more valuable than its size suggests.

**Stated plainly: `ch_geneva_pedestrian_network` also has zero name coverage and
regressed by 0.056.** So this is a strong observed pattern with a real
counterexample, not a law. The effect is robust to seed (±0.0011) but rests on a
handful of datasets; it should be re-checked as the label base grows.

---

## 3. Ranked recommendation

Solo LOO-F1 lift over the 28-feature baseline, one feature at a time, mean over
3 seeds (baseline 0.8741) — from
`research/spark_feature_expansion.py --per-feature`. Marginal cost is per
candidate pair, on top of what the shipped 28 already compute.

### Tier A — add

| # | Feature | Solo ΔLOO | What it needs | Marginal cost |
|---|---|---|---|---|
| 1 | **`has_name_target`** | **+0.0035** (+0.0043 as a full tier, 5 seeds) | `bool(target name non-empty after strip)` | **0.00 µs** — already a key in the `compute_name_similarity` dict the exporter discards |

### Tier A′ — optional, take with #1 or not at all

| # | Feature | Solo ΔLOO | Marginal cost |
|---|---|---|---|
| 2 | `has_name_ref` | −0.0003 | 0.00 µs |

Solo it is neutral, but paired with `has_name_target` the tier measures the same
+0.0043 with no cost. The pair encodes *which side* is unnamed, which neither
flag does alone — a robustness argument for datasets not yet in the label base,
not something these 33 folds can confirm. Take it if you want the symmetry;
skipping it loses nothing measurable.

### Tier B — feasible and nearly free, but do not add

| # | Feature | Solo ΔLOO | Why not |
|---|---|---|---|
| 3 | `vertex_density_ratio` | +0.0007 | within noise; the block it belongs to is net negative |
| 4 | `shape_complexity_ref` | +0.0004 | within noise |
| 5 | `name_is_generic` | +0.0002 | within noise |
| 6 | `route_prefix_match` | +0.0002 | within noise; the only genuinely new call (1.18 µs) |
| 7 | `name_jaro_winkler` | +0.0001 | within noise; redundant with `name_levenshtein` |
| 8 | `name_soundex` | −0.0000 | |
| 9 | `angle_histogram_similarity` | −0.0001 | model card's "correlated with heading_delta + buffer_iou" confirmed |
| 10 | `shape_complexity_delta` | −0.0002 | |
| 11 | `heading_consistency_ref` | −0.0003 | |
| 12 | `vertex_density_ref` | −0.0006 | |
| 13 | **`max_coverage`** | **−0.0007** | see the trap below |
| 14 | `sinuosity_delta` | −0.0012 | |
| 15 | `name_metaphone` | −0.0015 | worst of the name block |
| 16 | `heading_consistency_delta` | −0.0017 | |
| 17 | `vertex_density_target` | −0.0026 | worst of all 17 |

**`max_coverage` is the trap worth naming.** In the 45-feature model it is the
**4th most important feature by XGBoost gain** (0.0687, ahead of
`class_similarity` and `aligned_length_m`), and it costs literally nothing —
`max(ref_coverage, target_coverage)`, derivable in SQL from columns the model
already carries. It is the single feature this investigation was most likely to
recommend on importance alone. It measures **−0.0007 solo** and **−0.0014 as
part of the free-derived tier**. Gain importance says the trees split on it
often; LOO CV says those splits do not transfer to unseen datasets. Its
per-dataset spread is wide (`sg_singapore_footpaths` +0.036, `br_sao_paulo_roads`
−0.043, range 0.079) — consistent with learning dataset-specific segmentation
ratios. **Do not add it.**

**Correction (2026-08-07):** an earlier draft called that spread "the widest in
the set". It is not — it ranks **2nd of 17**. The widest is `has_name_target` at
0.150 (`ke_nairobi_roads` +0.089, `ch_geneva_pedestrian_network` −0.060), 1.9×
wider. So "widest spread ⇒ learning dataset-specific behaviour ⇒ do not add"
cannot be the argument against `max_coverage`, because it would rule out the
feature this analysis recommends. The honest version: spread alone is not a
reason to reject, and `max_coverage` is rejected on its **mean** (−0.0007 solo,
−0.0014 in-tier), not its variance. §2 already discloses `has_name_target`'s
spread and its counterexample; the case for it is mean-plus-mechanism, and the
same standard has to apply here.

This is the general shape of the result: **feasibility was never the binding
constraint — label volume is.** All 17 are computable; only one is worth
computing.

---

## 4. What to change if this is accepted

**Applied 2026-08-07** (steps 1-4); step 5 is outstanding and step 6 stands as
written. Kept in the original proposal voice so the reasoning reads in order,
with outcomes noted per step.

1. `config.py::SPARK_PORTABLE_FEATURES` — add the chosen features **in
   `FEATURE_COLUMNS` order**. Order matters, but not for the reason an earlier
   draft of this section gave ("the Spark scorer broadcasts the manifest list as
   the DMatrix column order, so append rather than insert"), and a second draft
   was wrong the other way ("position is cosmetic"). The mechanism:
   `export-spark-model` passes this list to `MLMatcher` as an *exclusion* set
   (the `exclude_features` line in `cli/main.py`), and `_extract_from_columns`
   rebuilds `feature_names` in `FEATURE_COLUMNS` order, ignoring this list's
   ordering entirely — so the **model** is order-insensitive. But
   `build_spark_model_manifest` writes the manifest from `feature_names`, and
   `tests/unit/test_shipped_spark_model.py` compares
   `manifest["features"] == SPARK_PORTABLE_FEATURES` as an **ordered list**.
   Appending is exactly what breaks it, with a diff that reads like a stale
   export rather than an ordering mistake.
2. Retune `SPARK_PORTABLE_XGB_PARAMS` for the new set:
   `uv run python scripts/tune_model.py --feature-set spark`. The current params
   were Optuna-tuned *for 28 features* on 2026-07-03; the tiers above
   deliberately reuse them so the feature set is the only variable, which makes
   +0.0043 a **lower bound**.
3. Reship in the same PR — `tests/unit/test_shipped_spark_model.py` fails until
   the manifest matches config:
   `uv run crosswalk export-spark-model -o data/models/export`, then copy
   `model.json`/`manifest.json` into `src/crosswalk/_model/` (`docs/RELEASING.md`).
4. Fix the two stale/over-broad rationales:
   * `cli/main.py:1279` and `config.py:548-551` — "no topology, graph, or
     spatial-index features required" is not why the name and geometry features
     are excluded. They are excluded on measured value.
   * `docs/SPARK_MODEL_CARD.md` — says 78 features (now 83) and "Excluded (50)"
     (now 55); its "Additional name (7 of 10) … low marginal value" line is the
     one claim these measurements contradict, and only for `has_name_target`.
5. Coordinate with tf-data-platform: the consumer must emit the new column
   before the wider booster helps. If it calls `crosswalk.features.semantic`,
   this is one dict key; if it reimplements name similarity, it is
   `1.0 if target_name and target_name.strip() else 0.0`.
6. Re-run this when the label base grows. The geometry block losing is a
   small-data result, not a permanent one; `research/spark_feature_expansion.py`
   is parameterised to be re-run as-is.

## 5. What was sampled rather than run in full

* **Tier sweep** — run in full: 8 tiers × 5 seeds, the whole label set (5,457
  evaluated rows / 33 datasets), 33 LOO folds each. Nothing sampled.
* **Per-feature ranking — 3 seeds** (42, 1, 2) rather than 5, to keep the box
  free. Per-dataset pairing keeps the comparison tight, but individual solo
  deltas carry more noise than the tier deltas; the recommendation was confirmed
  at 5 seeds as its own tier (t2a).
* **Parity/demo fixture — 180 pairs** (60 each from 3 datasets), not all 5,457.
  Parity is exact equality, so a mismatch would surface at any sample size; 180
  was enough to exercise reversed alignments, partial coverage, and missing names.
* **Feature-cost timings — single-pair Python loop**, not the batched pipeline.
  Absolute µs/pair are therefore higher than production; the ratios are the claim.
* **Inference timings are noisy** (±0.03–0.19 µs/row) — the machine was running
  an Overture fetch and a factory build concurrently. Differences under ~0.1
  µs/row in that column should not be read as real.
* **Not measured:** stitch-level quality (`mbench run … --gate`). It needs
  `data/output`, absent from this worktree, and `crosswalk stitch` does not use
  the Spark model anyway.
* **Not measured:** whether retuning hyperparameters for 29 features changes the
  ranking. It can only help the recommended tier (§4.2).
* **Not measured:** anything on the tf-data-platform side. Whether the Spark job
  calls `crosswalk-py` or reimplements the features is inferred from
  `crosswalk/spark.py`'s documented consumption pattern, not verified.

## Artifacts

| Path | What |
|---|---|
| `research/spark_feature_expansion.py` | Tier sweep (F1 / size / inference / RSS) + `--per-feature` LOO ranking |
| `tests/test_spark_feature_expansion.py` | Feasibility proof-by-computation, exclusion-partition guard, marginal-cost bench |
| `research/results/spark_feature_expansion_2026-08-07.json` | 5-seed tier results incl. per-dataset LOO rows |
| `research/results/spark_feature_expansion_per_feature_2026-08-07.json` | 3-seed per-feature LOO deltas |

```bash
# Feasibility proof + classification guard (fast, ~2s)
uv run pytest tests/test_spark_feature_expansion.py -m "not slow" -n 0

# Marginal cost + model-size/inference table (prints the numbers above)
uv run pytest tests/test_spark_feature_expansion.py -m slow -s -n 0

# Full tier sweep (~15 min)
OMP_NUM_THREADS=8 uv run python research/spark_feature_expansion.py \
    --seeds 42,1,2,3,4 --out research/results/spark_feature_expansion_2026-08-07.json

# Per-feature ranking (~10 min)
OMP_NUM_THREADS=8 uv run python research/spark_feature_expansion.py --per-feature \
    --seeds 42,1,2 --out research/results/spark_feature_expansion_per_feature_2026-08-07.json
```
