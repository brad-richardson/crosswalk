# Matcher Architecture

Technical details of the ML pipeline, feature computation, and decision thresholds.

For usage instructions, see [README.md](../README.md). For development workflow, see [CLAUDE.md](../CLAUDE.md).

## ML Model

- **Algorithm**: XGBoost binary classifier
- **Features**: 79 features across 17 categories (defined in `src/matcher/config.py::FEATURE_COLUMNS`)
- **Location**: `data/models/matcher_model_combined.joblib`
- **Training**: `matcher train` (trains on all labeled data in `labels/`)
- **Parallelization**: Uses `ProcessPoolExecutor` with worker initialization for feature computation
- **Auto Model Selection**: When `settings.auto_select_model=True`, automatically uses geometry-only model for datasets with low name coverage (< 50%)

The trained model is not committed to git. After cloning, run `matcher train` before using `-m xgboost`.

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
`resolver_prune_overrides` **allowlist** (keyed by the dataset name derived from the
bridge output filename), and only while the `resolver_prune_enabled` master switch
(default **True**) is set. A dataset absent from the allowlist is **not pruned** (a
one-line info log records the skip); an allowlist value ≤ 0 keeps a listed dataset
explicitly disabled. There is **no global default floor** — the previous
`resolver_prune_min_confidence` (0.96 applied to every dataset without an override)
has been removed, because #284's own sweep showed that Boston-tuned 0.96 over-prunes
never-tuned / sidewalk-like sets. Filename resolution is robust to the documented
before/after comparison workflow: `before_<dataset>_bridge.parquet` /
`after_<dataset>_bridge.parquet` resolve to `<dataset>` by exact match after
stripping a known prefix (overlapping names like `us_boston_streets_2` never
collide — only exact post-strip matches count).
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

Group optimization (`src/matcher/matching/optimizer.py::optimize_matches_with_grouping`) resolves cases where a single Overture segment corresponds to multiple local segments (e.g., split carriageways), and vice versa. There is no assignment solver; the pipeline is:

1. **Connected components**: Build bipartite connected components over candidate pairs above `min_confidence` (`find_match_components`)
2. **Classification**: Classify each component as 1:1, 1:N, N:1, or M:N by counting distinct refs/targets (`_classify_and_resolve_component`)
3. **Contiguity clustering**: Within 1:N/N:1/M:N components, cluster segments into contiguous subgroups via cKDTree endpoint proximity (within `contiguity_tolerance`); non-contiguous singletons fall back to the 1:1 pool
4. **Greedy 1:1 assignment**: Assign unclaimed leftover candidates greedily by descending confidence (`optimize_matches_greedy`)
5. **Post-hoc expansion**: Expand 1:1 matches into 1:N/N:1 groups where contiguous candidates exist (`_expand_greedy_matches`); this only adds matches, never removes assignments
6. **Coverage-conflict demotion**: When two targets claim overlapping portions of the same reference segment (overlap > `MAX_ALIGNMENT_OVERLAP_M` = 5 m), the lower-confidence match is demoted to REVIEW (`_validate_assignment_coverage`)

Applied during group optimization and to define the labeling UI review band:

| Setting | Default | Purpose |
|---------|---------|---------|
| `optimizer_match_threshold` | 0.75 | Confident match in optimizer; upper bound of labeling review band |
| `optimizer_review_threshold` | 0.5 | Below this = no match in optimizer; lower bound of labeling review band |

1:N groups: `avg_confidence >= optimizer_review_threshold` -> MATCH

## Model Evaluation

Use cross-validation or holdout evaluation for unbiased metrics:

```bash
# Cross-validation (default: 5-fold, segment-aware splitting)
matcher eval

# Evaluate an existing model on 20% holdout
matcher eval --model data/models/matcher_model_combined.joblib

# Custom folds or seed
matcher eval --cv-folds 10 --seed 123

# Evaluate on specific dataset(s)
matcher eval --model data/models/matcher_model_combined.joblib -d us_frisco_trails
```

**Why holdout/CV matters:**
- Evaluating on training data gives artificially inflated accuracy (~99%)
- Cross-validation gives realistic generalization metrics with variance estimates
- Segment-aware splitting prevents data leakage (the same segment ID—either gers_id or target_id—never appears in both train and test)
- Use consistent seed (default: 42) for comparable results across experiments

## Feature Categories

79 features across 17 categories. `config.py::FEATURE_COLUMNS` is the single source of truth.

| Category | Count | Features |
|----------|-------|----------|
| Geometric | 9 | hausdorff_distance_m, mean_hausdorff_distance_m, hausdorff_p95_m, buffer_iou_5m, buffer_iou_15m, heading_delta, collinear_gap_ratio, angle_histogram_similarity, edge_distance_rmse_m |
| Name Similarity | 10 | name_levenshtein, name_jaro_winkler, name_token_sort, name_soundex, name_metaphone, has_name_ref, has_name_target, name_is_generic, name_numeric_match, route_prefix_match |
| Class | 1 | class_similarity |
| Endpoint/Connectivity | 3 | min_endpoint_proximity_m, max_endpoint_proximity_m, shared_endpoint_count |
| Lateral Offset | 3 | lateral_offset_m, lateral_offset_iqr_m, lateral_offset_p95_m |
| Topology | 18 | from/to_degree_ref/target, degree_match_score, degree_signature_similarity, is_dead_end_ref/target, dead_end_match, is_intersection_ref/target, intersection_match, interior_junction_count_ref/target, interior_junction_count_delta, interior_connector_jaccard, interior_junction_position_sim, shared_anchor_count |
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
config.py::FEATURE_COLUMNS (79 features)
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

### Missing Value Handling

There is **no imputation**. NaN feature values are passed through unchanged to XGBoost, which handles missing values natively (each tree split learns a default direction for missing values). The only sanitization is infinity capping: `ml.py::_cap_infinities` replaces `±inf` with `MAX_DISTANCE_METERS` because XGBoost handles NaN but not inf. This is applied consistently to training data, test data, and inference features (`_features_to_array` also fills missing dict keys with NaN, not 0).

**Risk**: Because NaN is a valid model input, a feature that is systematically NaN at inference but populated during training (or vice versa) fails silently — tree routing changes for those rows instead of raising an error. Guardrails: `train()` raises if labels are missing expected features, `matcher backfill` keeps stored features current, and `tests/unit/test_ml_pipeline_consistency.py` verifies NaN preservation and inf capping.

## Test Coverage for Consistency

| Test File | What It Catches |
|-----------|----------------|
| `test_label_store.py` | Features computed but not saved to labels |
| `test_feature_consistency.py` | Error defaults, naming conventions |
| `test_ml_pipeline_consistency.py` | Pre-computation vs direct computation, NaN preservation / inf capping, alignment-aware graphlet computation, label-store round-trip parity |
