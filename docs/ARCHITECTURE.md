# Crosswalk Architecture

Crosswalk publishes `local_id ↔ GERS id` bridge tables — the rosetta stone that
makes a city's locally-keyed data joinable to the open map. This document covers
the engine behind that: technical details of the ML pipeline, feature computation,
and decision thresholds.

For usage instructions, see [README.md](../README.md). For development workflow, see [CLAUDE.md](../CLAUDE.md).

## ML Model

- **Algorithm**: XGBoost binary classifier
- **Features**: 83 features across 17 categories (defined in `src/crosswalk/config.py::FEATURE_COLUMNS`)
- **Production location**: bundled `src/crosswalk/_model/matcher_model_combined.joblib`
- **Local training output**: `data/models/matcher_model_combined.joblib`
- **Training**: `crosswalk train` (trains on all labeled data in `labels/`)
- **Parallelization**: Uses `ProcessPoolExecutor` with worker initialization for feature computation
- **Auto Model Selection**: the advisory labeling workflow can use a local geometry-only model for datasets with low name coverage (< 50%)

`crosswalk stitch` and the factory default to the committed bundled artifact.
An incidental file under `data/models/` never changes production output. A local
experiment must opt in with `crosswalk stitch --model-path <path>` or
`MATCHER_MODEL_PATH=<path>`; a missing explicit override fails instead of falling
back silently. The labeling UI remains local-first because its scores are
advisory during retraining.

### Shipped models (bundled in the wheel)

Two pretrained artifacts are committed under `src/crosswalk/_model/` and ship in
the wheel, so `pip install crosswalk-py` needs zero training:

| Artifact | What it is | Consumer |
|----------|------------|----------|
| `matcher_model_combined.joblib` | Full-feature (`config.FEATURE_COLUMNS`) calibrated model | Default `crosswalk stitch` and factory artifact (`config.bundled_model_path()`) |
| `spark_model.json` + `spark_manifest.json` | Spark-portable 28-feature (`SPARK_PORTABLE_FEATURES`) XGBoost-native booster + manifest | Spark scoring jobs (tf-data-platform) via `matcher.spark` |

Both are kept in lockstep with `config.FEATURE_VERSION` by CI
(`test_shipped_model.py`, `test_shipped_spark_model.py`) — a feature bump fails
those tests until the artifacts are re-exported in the same PR (see
docs/RELEASING.md). Newly trained native models and Spark manifests also record
`training_metadata.source_commit`: the source SHA plus separate tracked-dirty
and untracked-file counts. Older artifacts remain loadable when this additive
field is absent; the currently bundled model receives it at its next clean
retrain/reship.

#### Spark-portable model shipping and consumption

The Spark-portable model is a first-class shipped artifact: `matcher
export-spark-model` trains the 28-feature geometry-only subset (no topology,
graph, or spatial-index features) and writes `model.json` + `manifest.json`;
those are copied into the package as `spark_model.json` / `spark_manifest.json`
and included in the wheel by `pyproject.toml`'s `[tool.hatch.build.targets.wheel]
artifacts` rule.

A Spark job consumes them with **no heavy imports** — `matcher.spark` touches
only the stdlib at import time (numpy is lazy, imported inside
`apply_calibration`), so `import matcher.spark` never pulls in
shapely/geopandas/xgboost/pandas:

```python
from matcher.spark import spark_model_json, spark_manifest, apply_calibration

manifest = spark_manifest()
features = manifest["features"]            # ordered; broadcast as the column order
booster.load_model(bytearray(spark_model_json().encode()))
# ... score to raw P(match), then (optionally) calibrate:
calibrated = apply_calibration(raw_scores, manifest["calibration"])
```

`apply_calibration` reproduces `IsotonicCalibrator.transform` exactly
(`np.interp` over the manifest's `calibration.x_thresholds`/`y_thresholds` with
endpoint clipping). The manifest ships `calibration.applied = false` — the knots
are emitted so the Spark job *can* remap raw scores to calibrated `P(match)`;
whether to consume them (and re-fit downstream thresholds on calibrated scores)
is a consumer decision. A job that prefers not to depend on the installed
package can instead vendor the two `src/crosswalk/_model/spark_*.json` files
directly out of the wheel.

## Decision Thresholds

All thresholds are configurable in `config.py`.

### Probability Calibration

`MLMatcher.train` fits an isotonic-regression calibrator on the out-of-fold
predictions from the in-training GroupKFold CV (training rows only — the seed-42
holdout never participates, so calibration is leakage-free). The calibrator is
stored in the model artifact as portable piecewise-linear knots
(`calibration.py::IsotonicCalibrator`) and applied by `MLMatcher.predict` when
`enable_calibration` is True (default). **All five confidence thresholds below
therefore operate on calibrated `P(match)`, not raw XGBoost scores.** Set
`enable_calibration=False` (or `predict(..., calibrated=False)`) to fall back to
raw scores for A/B comparison. A single global calibrator is used: per-dataset-
type calibration (road_good/road_poor/sidewalk/other) was measured and rejected
— it overfit the small sidewalk/other groups without beating global overall.
The Spark-portable export emits the knots into `manifest.json`
(`calibration.applied=false`); wiring the Spark job to consume them is a
tf-data-platform follow-up.

#### Confidence consumers (raw vs calibrated)

Every decision threshold that consumes an ML score reads
`MatchResult.confidence` (built from `MLMatcher.predict`, calibrated by default)
or calls `predict()` directly — so all of them operate on calibrated `P(match)`
when `enable_calibration` is True. There is no consumer left on raw scores:

| Consumer | Threshold(s) | Confidence source | Scale |
|----------|--------------|-------------------|-------|
| ML scorer decision (`predict_batch`) | `scoring_match_threshold` 0.5, `scoring_review_threshold` 0.1 | `predict()` | Calibrated |
| Optimizer candidate floor (`find_match_components`) | `min_confidence` (0.1 from runner) | `MatchResult.confidence` | Calibrated |
| Optimizer grouping glue prune | `optimizer_glue_min_confidence` **0.575** (calibrated) / `optimizer_glue_min_confidence_raw` 0.5 (uncalibrated) | `MatchResult.confidence` | Calibrated |
| Resolver confidence-drop prune (post-optimizer, group edges) | per-dataset **allowlist** `resolver_prune_overrides` (Boston streets 0.96, Seattle sidewalks 0.90); `resolver_prune_enabled` master switch; datasets absent from the map are NOT pruned | `MatchResult.confidence` | Calibrated |
| Optimizer 1:N group decision | `optimizer_review_threshold` 0.5 (avg conf) | `MatchResult.confidence` | Calibrated |
| Labeling UI review band | `optimizer_match_threshold` 0.75 / `optimizer_review_threshold` 0.5 | `predict()` | Calibrated |
| Bridge output filter (`generate_bridge_file`) | `bridge_min_confidence` 0.5 | `MatchResult.confidence` | Calibrated |
| Stitch sidecar edge confidences + `alternatives`/`batch_selection`/`stitch_options` | relative ranking only (no absolute gate) | `MatchResult.confidence` | Calibrated |
| `stitch_export` size gate | structural (edge / assignment-component counts) | — | No confidence gate |
| `stitch-run` panel routing | agent-vote confidence (LLM self-report) | LLM output | Not an ML score |
| `score_propagation` (experimental) | — | logit of `MatchResult.confidence` | Default OFF |

**Glue-prune calibration equivalence.** The grouping-only glue prune is the one
threshold that was *empirically tuned* against raw scores: the corridor-aware
design (#267) validated it at raw `p=0.5` (`research/group_splitting_design.md`).
Because isotonic calibration maps the mid-range raw 0.5 to ~0.575, a naive `0.5`
on calibrated scores is a weaker raw~0.42 prune. The prune is therefore
**calibration-aware**: `runner.py::_effective_glue_min_confidence` inspects the
loaded model's `MLMatcher.calibration_active` and passes
`optimizer_glue_min_confidence` (**0.575**, the calibrated image of raw 0.5) when
calibration is active, or `optimizer_glue_min_confidence_raw` (**0.5**) for an
uncalibrated model — so the effective prune population matches what #267
validated in either state and an uncalibrated model never silently over-prunes.
Measured with a calibrated model, keeping the prune at 0.5-calibrated regrouped
7.0% of selected edges (Boston) / 15.3% (Seattle) vs the raw-0.5 baseline; the
0.575-calibrated point cuts that to 4.4% / 11.5% (the residual is the
candidate-floor effect, not the glue prune) while leaving all monster (>20-edge)
groups identical (Boston 6, Seattle 2).

**Resolver confidence-drop prune (per-dataset opt-in / allowlist).** After the
optimizer assigns group edges, a second prune (`apply_confidence_drop_prune`, M2 /
resolver Phase 1) drops any *selected* M:N/1:N/N:1 group edge whose calibrated
confidence is below an absolute floor — the one-parameter filter the #272 resolver
eval validated (it beat both keep-all and the learned per-edge model on the clean
slice). It never touches 1:1 matches and always keeps each group's single
highest-confidence edge (a group is never emptied). Pruned pairs are recorded in
the sidecar (`pruned` per edge, `n_pruned` per group) so the effect is auditable —
`n_pruned` is exact: each pruned pair is attributed to its pre-prune `group_id` and
counted in exactly that owner group — pruned edges are exempt from the per-group
rejected-edge cap, a pruned *pendant* edge (both endpoints leave the owner group) is
recovered and recorded there, and the same pair appearing as an incident alternative
in a *foreign* corridor sub-group (where a surviving endpoint lives) is not
re-counted. With attribution present the count is authoritative (independent of
`stitch_persist_rejected_edges`); `sum(n_pruned)` equals the number of dropped edges.
The optimal floor is **dataset-dependent**, so the prune is **per-dataset opt-in**:
`runner.py::_effective_prune_threshold` applies it ONLY to datasets present in the
`resolver_prune_overrides` **allowlist**, and only while the `resolver_prune_enabled`
master switch (default **True**) is set. The allowlist is keyed on **dataset
identity** — the dataset name the runner is told (`crosswalk stitch`'s dataset
argument / the factory pair name), **never** anything derived from the output
filename (#348: the old filename-stem resolution silently skipped pruning for any
nonstandard output name like `after4_<dataset>_bridge.parquet`, changing match
counts mid-measurement; the output path now plays no part, so any `-o` name prunes
identically). A run with **no dataset identity** (raw `-r`/`-t` path mode without a
dataset name) is never pruned. A dataset absent from the allowlist is **not
pruned**; an allowlist value ≤ 0 keeps a listed dataset explicitly disabled. Every
run logs its prune decision — enabled (dataset @ threshold), not allowlisted, no
dataset identity, master switch off, or calibration guard — so the state is never
silent. There is **no global default floor** — the previous
`resolver_prune_min_confidence` (0.96 applied to every dataset without an override)
has been removed, because #284's own sweep showed that Boston-tuned 0.96 over-prunes
never-tuned / sidewalk-like sets.
Because every validated floor was tuned on **calibrated** confidence and no raw
operating point exists, `_effective_prune_threshold` also skips the prune (returns
0.0) when the active model applies no calibration — so an uncalibrated model does
not silently over-prune raw scores (mirrors the glue prune's raw/calibrated guard).
Shipped allowlist, validated under the #271 stitch gate at the coordinated-retrain
deploy (calibrated model, `PYTHONHASHSEED=0`):

| Dataset | Threshold | filtered edge-F1 | group exact | gate |
|---------|-----------|------------------|-------------|------|
| us_boston_streets (117 labels) | prune OFF | 0.8671 | 0.5093 | PASS |
| us_boston_streets | **0.96** (allowlisted) | **0.8790** | **0.5833** | PASS |
| us_seattle_sidewalks (27 labels) | prune OFF | 0.8665 | 0.40 | unarmed |
| us_seattle_sidewalks | **0.90** (allowlisted) | **0.8913** | **0.50** | unarmed |

Seattle's lower-confidence sidewalks over-prune at 0.96 (filtered F1 regresses to
0.848, below keep-all); 0.90 is its F1/exact-maximizing point, which is why it
carries its own allowlist floor. Both shipped datasets were tuned via the #284
sweep recipe. Adding a new dataset to the allowlist requires the same per-dataset
tuning first (see SCALING_ROADMAP M2) — datasets are never pruned on an untuned
inherited default.

### Scoring Thresholds (per-candidate, bridge file output)

Applied by the ML scorer when classifying each candidate pair:

| Setting | Default | Decision |
|---------|---------|----------|
| `scoring_match_threshold` | 0.5 | `>= this` -> MATCH |
| `scoring_review_threshold` | 0.1 | `>= this` -> REVIEW, below -> NO_MATCH |

### Optimizer/Labeling Thresholds (1:N groups and labeling UI)

Group optimization (`src/crosswalk/matching/optimizer.py::optimize_matches_with_grouping`) resolves cases where a single Overture segment corresponds to multiple local segments (e.g., split carriageways), and vice versa. There is no assignment solver; the pipeline is:

1. **Connected components**: Build bipartite connected components over candidate pairs above `min_confidence` (`find_match_components`)
2. **Classification**: Classify each component as 1:1, 1:N, N:1, or M:N by counting distinct refs/targets (`_classify_and_resolve_component`)
3. **Corridor-aware contiguity**: Within 1:N/N:1/M:N components and post-greedy expansion, cluster endpoint-adjacent segments only when they are collinear continuations or share a normalized name; disconnected chains remain separate groups
4. **Guarded alignment rescue**: Rejoin complementary same-name fragments when their alignment intervals tile one shared segment with only a short gap/overlap and either their endpoints snap directly or a real short connector segment explains the split; ambiguous transitive attachments are rejected
5. **Canonical greedy assignment**: Assign unclaimed candidates by confidence, name agreement, measured alignment coverage, and stable IDs/indices (`optimize_matches_greedy`), so exact calibrated-score ties are input-order invariant
6. **Post-hoc expansion and recomposition**: Expand surviving anchors with corridor-compatible candidates above the glue floor, then recompose connected assignments without duplicating selected pairs; low-confidence/decomposed additions remain REVIEW
7. **Symmetric coverage-conflict demotion**: When two targets claim overlapping portions of one reference, or two references claim overlapping portions of one target (overlap > `MAX_ALIGNMENT_OVERLAP_M` = 5 m), demote the lower-ranked assignment to REVIEW (`_validate_assignment_coverage`)

Applied during group optimization and to define the labeling UI review band:

| Setting | Default | Purpose |
|---------|---------|---------|
| `optimizer_match_threshold` | 0.75 | Confident match in optimizer; upper bound of labeling review band |
| `optimizer_review_threshold` | 0.5 | Below this = no match in optimizer; lower bound of labeling review band |

Low-confidence additions, pruned/decomposed singletons, parallel siblings, and
coverage conflicts are retained only as REVIEW with an explicit review-reason
flag; they are not promoted merely to keep a group nonempty.

## Model Evaluation

Use cross-validation or holdout evaluation for unbiased metrics:

```bash
# Cross-validation (default: 5-fold, segment-aware splitting)
crosswalk eval

# Evaluate an existing model on 20% holdout
crosswalk eval --model data/models/matcher_model_combined.joblib

# Custom folds or seed
crosswalk eval --cv-folds 10 --seed 123

# Evaluate on specific dataset(s)
crosswalk eval --model data/models/matcher_model_combined.joblib -d us_frisco_trails
```

**Why holdout/CV matters:**
- Evaluating on training data gives artificially inflated accuracy (~99%)
- Cross-validation gives realistic generalization metrics with variance estimates
- Segment-aware splitting prevents data leakage (the same segment ID—either gers_id or target_id—never appears in both train and test)
- Use consistent seed (default: 42) for comparable results across experiments

## Feature Categories

83 features across 17 categories. `config.py::FEATURE_COLUMNS` is the single source of truth.

| Category | Count | Features |
|----------|-------|----------|
| Geometric | 9 | hausdorff_distance_m, mean_hausdorff_distance_m, hausdorff_p95_m, buffer_iou_5m, buffer_iou_15m, heading_delta, collinear_gap_ratio, angle_histogram_similarity, edge_distance_rmse_m |
| Name Similarity | 10 | name_levenshtein, name_jaro_winkler, name_token_sort, name_soundex, name_metaphone, has_name_ref, has_name_target, name_is_generic, name_numeric_match, route_prefix_match |
| Class | 1 | class_similarity |
| Endpoint/Connectivity | 3 | min_endpoint_proximity_m, max_endpoint_proximity_m, shared_endpoint_count |
| Lateral Offset | 3 | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m |
| Topology | 22 | from/to_degree_ref/target, from/to_degree_target_native, degree_match_score, degree_signature_similarity, is_dead_end_ref/target, is_dead_end_target_native, dead_end_match, is_intersection_ref/target, is_intersection_target_native, intersection_match, interior_junction_count_ref/target, interior_junction_count_delta, interior_connector_jaccard, interior_junction_position_sim, shared_anchor_count |
| Alignment Coverage | 5 | ref_coverage, target_coverage, min_coverage, coverage_ratio, max_coverage |
| Graphlet | 2 | graphlet_similarity, endpoint_degree_similarity |
| Clustering | 3 | clustering_coef_ref, clustering_coef_target, clustering_coef_delta |
| Sinuosity | 3 | sinuosity_ref, sinuosity_target, sinuosity_delta |
| Heading Consistency | 3 | heading_consistency_ref, heading_consistency_target, heading_consistency_delta |
| Vertex Density | 3 | vertex_density_ref, vertex_density_target, vertex_density_ratio |
| Length | 2 | min_length_m, aligned_length_m |
| Shape Complexity | 3 | shape_complexity_ref, shape_complexity_target, shape_complexity_delta |
| Parallel Sibling | 5 | has_parallel_sibling_ref, parallel_fraction_ref, offset_vs_half_corridor_ratio, offset_over_expected_halfwidth, likely_representation_mismatch |
| Crossing Angle | 4 | crossing_angle_min_ref, transverse_neighbor_fraction_ref, crossing_angle_min_target, transverse_neighbor_fraction_target |
| Intersection Overlap | 2 | post_node_continuation_m, endpoint_heading_divergence |

## Feature Computation Paths

Understanding the computation paths is critical for preventing training/inference skew.

### Single Source of Truth

```
config.py::FEATURE_COLUMNS (83 features)
         |
         |---> compute.py::compute_pair_features()  <-- AUTHORITATIVE computation
         |           |
         |           |---> ml.py::_compute_single_feature() (inference)
         |           |
         |           +---> labeling UI (training data generation)
         |
         +---> src/crosswalk/labeling/feature_store.py (Parquet storage, keyed by gers_id + target_id)
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

### Missing Value Handling

There is **no imputation**. NaN feature values are passed through unchanged to XGBoost, which handles missing values natively (each tree split learns a default direction for missing values). The only sanitization is infinity capping: `ml.py::_cap_infinities` replaces `±inf` with `MAX_DISTANCE_METERS` because XGBoost handles NaN but not inf. This is applied consistently to training data, test data, and inference features (`_features_to_array` also fills missing dict keys with NaN, not 0).

**Risk**: Because NaN is a valid model input, a feature that is systematically NaN at inference but populated during training (or vice versa) fails silently — tree routing changes for those rows instead of raising an error. Guardrails: `train()` raises if labels are missing expected features, `crosswalk backfill` keeps stored features current, and `tests/unit/test_ml_pipeline_consistency.py` verifies NaN preservation and inf capping.

## Test Coverage for Consistency

| Test File | What It Catches |
|-----------|----------------|
| `test_label_store.py` | Features computed but not saved to labels |
| `test_feature_consistency.py` | Error defaults, naming conventions |
| `test_ml_pipeline_consistency.py` | Pre-computation vs direct computation, NaN preservation / inf capping, alignment-aware graphlet computation, label-store round-trip parity |

## Stitching labels

`/stitching-review` curates M:N group edge selections into
`labels/stitching/dataset=*/data.csv` (see `labeling/stitching_store.py`). Schema:

| Column | Meaning |
|--------|---------|
| `group_id` | Deterministic group hash (ref/target id sets) |
| `dataset_id` | Dataset partition |
| `selected_edges` | JSON `[{ref_id, target_id}, ...]` — the endorsed pair set (empty for set rows) |
| `match_type` | `1:1` / `1:N` / `N:1` / `M:N` |
| `num_refs`, `num_targets` | Segment counts (membership sizes for set rows) |
| `labeler` | Reviewer id; `panel_*` = LLM-panel auto-accept (non-human) |
| `labeled_at`, `session_id` | Provenance (`session_id = deanchored_v1` marks de-anchored reviews) |
| `label_semantics` | `pair` (default) or `set` |
| `ref_ids`, `target_ids` | Set-label membership as JSON id arrays (empty for pair rows) |

**Pair vs set semantics.** A **pair** label's `selected_edges` is the authoritative
per-pair truth the reviewer endorsed — used for explicit option-card
ratifications and for the LLM panel's exported consensus. A **set** label records
only that *these refs and these targets form one matched group*: the reviewer
asserted MEMBERSHIP, not individual pairings (the mobile UI makes per-pair
adjudication impractical). Manual and de-anchored submits therefore record a set
label — membership in `ref_ids`/`target_ids`, `selected_edges` empty — rather
than expanding the active pill cross-product into pairs the reviewer never
adjudicated.

Loaders default a missing/blank `label_semantics` to `pair` (NaN-safe), so CSVs
predating these columns read as ordinary pair labels; the columns migrate lazily
on the next save. Historical cross-product manual labels are converted to set
semantics with `crosswalk data stitch-reinterpret-sets` (uses the shared
`agent_labeling.xprod` cross-product detector; panel rows and non-artifact
ratifications are left untouched; idempotent, with a `.csv.bak` backup).

**Eval.** Pair labels are scored on edge-level precision/recall/F1 and exact
match (the stitch gate's enforced metrics). Set labels are excluded from those
pools and scored on membership exact-match / boundary precision / coverage — see
BENCHMARKING.md "Stitch-level quality gate". The scoring core is shared between
`matcher.agent_labeling.stitch_eval` and the matcher-free `mbench.eval.stitch_metrics`,
parity-guarded by `tests/unit/test_mbench_set_metric_parity.py`.

**Drift-aware review queue.** `group_id` is a content hash of the exact
ref/target id sets, so regenerating the optimizer output re-mints ids for
already-reviewed geometry — an exact-id "already reviewed" filter would re-queue
a relabeled group as brand new (the Bogotá `3c3e6853` → `8e32a935` case).
`labeling/stitch_coverage.py` classifies every current group against the
dataset's labels using the SAME drift mapping eval/rekey use
(`stitch_eval.recover_labeled_groups`: pair labels by selected-edge overlap, set
labels by membership overlap, #354 deterministic tie-break). Semantics per
current group G and mapped label L (kept membership = set-label
`ref_ids`/`target_ids`, or pair-label edge endpoints ∪ id columns):

| Coverage | Rule | Queue behavior |
|----------|------|----------------|
| Exact id | L's `group_id` survives verbatim | Reviewed → excluded (pre-drift behavior, id-stable paths unaffected) |
| Full | G.refs ⊆ L.kept_refs AND G.targets ⊆ L.kept_targets | Reviewed → excluded (re-review would be a mechanical re-approve) |
| Partial | maps, but G has members outside L's kept universes | Queued with `prior_label` delta (banner, kept ∩ current prefill, new-member pill flags) |
| None | no label maps | Queued as before |

Applied at batch build (`cli/data.py::_generate_stitch_batch_for_dataset`),
serve time (`web/services.py::get_unreviewed_stitch_groups`, per owning dataset
in the `__all__` queue, recomputed fresh each request), and the agent panel feed
(`crosswalk agent stitch-batch`; `--calibration` lifts the exclusion for
deliberate ground-truth waves). A merged group covered only by the UNION of
several labels is never auto-excluded — one label must fully cover it.

### Panel v5: the quad composition and the quorum consensus rule

The consensus panel (`crosswalk agent stitch-run`, `agent_labeling/stitch_runner.py`)
runs each voter through its own CLI. The blessed **v5 default** (2026-07-10) is the
four-seat quad: `claude`/claude-opus-4-8 (medium) + `codex`/gpt-5.6-terra (medium)
+ `kimi`/Kimi K2.6 + `muse`/Muse Spark 1.1. Three seats drive the **opencode**
transport, each under its OWN provider name: Kimi (provider name
`kimi`, `openrouter/moonshotai/kimi-k2.6`, via opencode's native OpenRouter auth
stored by `opencode auth`); the residual v3-era Qwen voter (provider name
`opencode`, `openrouter/qwen/qwen3-vl-235b-a22b-instruct`; only in the
`v3-candidate`/`no-agy` panels); and the Muse voter (provider name `muse`,
`meta/muse-spark-1.1`, Meta's OpenAI-compatible developer API).

**Quorum consensus rule (v5).** `compute_consensus` auto-accepts when **all
valid (non-abstaining) votes agree and at least 3 are valid**
(`agree == n_valid >= 3`), replacing the pre-v5 `agree == len(votes)` rule under
which a single abstention (e.g. a voter timeout) blocked an otherwise-clean
3-of-4 agreement. The two accept tiers stay distinguishable end-to-end: a full
4/4 accept is `unanimous` (labeler `panel_unanimous_v5`), a 3-of-4 accept over
an abstention is `quorum` (route reason `quorum`; labeler `panel_quorum_v5`).
The analogous `quorum_none` remains human review because NONE can also mean no
exact offered option or insufficient evidence; only a human-confirmed empty
selection becomes reject-all truth. A recomposed decomposed group with ANY
quorum-accepted sub-verdict mints
`panel_quorum_decomposed_v5`). Quorum forgives **abstention only, never
disagreement** — a dissenting valid vote still routes to human review — and the
size / class-consistency / low-confidence gates all still apply on top,
evaluated over the valid votes. For any 3-voter panel (v2/v3/v4 re-runs) the
rule is routing-identical to the old one (all-valid agreement at quorum IS full
unanimity with 3 voters; an abstention drops below quorum), proven by an
exhaustive old-vs-new sweep in `tests/unit/test_stitch_agent.py`. The pre-v5
`abstention` route reason is retired from live routing (its case now
auto-accepts as `quorum`) but kept in `panel_routing` so historical
consensus.csv rows keep deriving and rendering faithfully.

**Each opencode-transport seat carries its own provider name (`kimi`, `muse`, and
the residual `opencode`/Qwen), not the shared transport name.** The provider string
is a keying field in several places (vote-provenance dedupe on
`(source_batch, group_id, provider)`; the panel monitor's per-voter stats; the
`provider=letter` minority strings; the resume-consistency provider-set check; the
`--*-model` CLI overrides). The v5 quad seats **both** Kimi and Muse,
so a shared name would put two indistinguishable voters in one wave (collapsed
provenance rows = silent vote loss, pooled monitor stats, an ambiguous `--*-model`).
Distinct names keep every voter addressable; each invoker is resolved via
`_INVOKERS["kimi" | "muse" | "opencode"] -> invoke_opencode`, and the `--agent`
threading keys on the resolved invoker (not the name) so Muse still gets its
tool-less `vote` agent. `--kimi-model` / `--muse-model` pin Kimi / Muse; the kept
`--opencode-model` now targets only the residual Qwen seat.

Historical panels stay addressable for reproduction: `v4` (the former 3-seat
default: claude + codex/gpt-5.6-terra + kimi), `v3`/`v2` (claude + codex + agy),
`v3-candidate`, `no-agy`, `v4-candidate` (the #397 validation composition), and
`meta-candidate` (the v4 trio with kimi **swapped** for muse — superseded by v5,
which seats both). `quad-candidate` is now an alias of the v5 default: it was
the calibration composition that recorded the 2026-07-10 quad wave (53 groups /
212 ballots) whose replay motivated the quorum rule — muse posted the top exact
accuracy (~67% vs claude 65 / codex 63 / kimi 57) with 0/53 abstains at
~$0.03/vote, with a monitored recall-leaning A-bias that the consensus rules
structurally contain (a dissent only blocks auto-accept; it never mints).
Non-blessed compositions are still refused by the `stitch-export`
`(provider, model)` gate without `--allow-nonstandard-panel` and never mint a
blessed labeler.

Setup (no machine-level config required):

- **Provider**: a project-level [`opencode.json`](../opencode.json) at the repo
  root defines a custom `meta` provider (`@ai-sdk/openai-compatible`,
  `baseURL: https://api.meta.ai/v1`). opencode resolves this config by walking up
  from the working directory to the Git root, so it is picked up automatically
  when `stitch-run` is invoked from the repo. The API key is referenced via
  `{env:META_API_KEY}` — **never inlined**. `META_API_KEY` (in `.env`) is loaded
  into the process environment by `crosswalk`'s `load_dotenv()` and inherited by
  the opencode subprocess.
- **Tool-less `vote` agent**: `opencode.json` also defines a model-agnostic `vote`
  agent with all tools disabled. Under opencode's default `build` agent a voter
  burns its turn on (auto-rejected) `ls`/`cat`/`read` tool calls instead of
  answering — a voter with the evidence-pack PNGs already force-attached needs no
  tools. **Both** the `OPENCODE_KIMI` and `MUSE` specs set `opencode_agent="vote"`,
  threaded to `invoke_opencode` as `--agent vote`. Kimi was moved onto `vote` after
  the 2026-07-10 quad-candidate calibration wave, where it timed out (480s) on 7/30
  groups under `build` — its successful votes ran a median 37s / max 172s (bimodal
  answer-fast-or-stall-forever, not slow thinking) — while Muse on the same
  transport and identical packs had 0/30 timeouts (median 19s) under `vote`. The
  `--agent` threading keys on the resolved invoker, so this is invocation plumbing
  only: the export gate keys voter identity on `(provider, model)` pairs, leaving
  the blessed composition unchanged. The residual v3-era Qwen seat passes no
  agent and is byte-identical to before this knob existed.
- **Output budget**: Muse emits hidden reasoning tokens; a low output budget
  truncates its JSON mid-object (`finish_reason: "length"`, `content: null`). The
  `muse-spark-1.1` model entry sets a generous `limit.output` (32000), and the
  spec carries a 480s timeout (reasoning runs long on large packs; `--timeout`
  still overrides).
