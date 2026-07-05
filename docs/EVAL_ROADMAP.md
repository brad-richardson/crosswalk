# Evaluation Roadmap

Findings from the deep evaluation-methodology review (2026-07-02), what has been
fixed, what remains, and the recommended sequence. This is a living document —
update it as items land.

## Context

A full architectural review of the eval methodology found that reported metrics
systematically overstated model quality (hyperparameter tuning leakage, CV/test
overlap, no cross-dataset holdout in CI, structurally inflated cbench precision)
and that the end product — the optimizer's final assignments — was almost
entirely unmeasured. The first three rounds of fixes landed 2026-07-02/03.

## Fixed (PRs #222–#228)

| Finding | Fix | PR |
|---|---|---|
| Docs described mechanisms the code never had (Hungarian optimizer, median imputation, cbench target-level eval) | Docs corrected to match implementation | #222 |
| cbench `--match-level target` documented but unimplemented; precision inflation invisible | Target-level eval implemented (now the default); `labeled_coverage` / `unlabeled_predictions` / `skipped_unsure` surfaced | #223 |
| Blocking recall unmeasured (matches lost at candidate generation invisible to all metrics) | `matcher blocking-recall` command; replays the real `generate_candidates` path; also fixed falsy-zero buffer bug | #224 |
| In-training CV included the test rows; agent labels concatenated before the split; stale `feature_version` only warned | CV runs on train rows only; agent labels appended post-split (and segment-overlap-excluded); stale features now error (`allow_stale_features` escape hatch) | #225 |
| Only segment-level splits enforced — no cross-dataset generalization gate | LOO-by-type CV promoted to CI gate (`tests/regression/test_loo_cv.py`; per-type-group macro-F1 floors = baseline − 0.05) | #226 |
| Hyperparameters tuned on all labels including the test set (optimistic metrics) | `tune_model.py` holds out the seed-42 test set before Optuna; `DEFAULT_XGB_PARAMS` regenerated (170 trees; honest CV F1 0.9317) | #227 |
| Spark-portable params carried the same tuning leakage; retune initially regressed inference speed | Leakage-free retune with epsilon-compact selection (224 trees × depth 10, ~1.79M rows/s; honest match F1 0.9094); `--feature-set spark` and `--epsilon` in `tune_model.py` | #228 |

Honest baselines after these fixes (seed-42 holdout, 1,087 rows):
full model match F1 ≈ 0.930 / accuracy 91.7%; Spark-portable match F1 ≈ 0.909 /
accuracy 89.2%. LOO-by-type macro-F1 baselines: road_good 0.909, road_poor
0.926, sidewalk 0.871, other 0.932.

## Remaining gaps (ranked by threat to results)

1. **The shipped output is barely evaluated.** All gated metrics measure the
   pairwise classifier. The optimizer's final assignments are evaluated by the
   labeled stitching groups (`labels/stitching/`). Shipped in #258, cbench
   computes a **modernized, non-blocking** stitch-level metric on every run
   (default-on: the `labels/stitching` dir is auto-resolved; skipped silently
   when a dataset has none, and any error is swallowed). It reaches parity with
   the matcher-side `matcher agent stitch-eval`: group mapping robust to
   group_id churn (exact-id + edge-overlap against the `*_groups.json` sidecar),
   raw **and** sliver-filtered edge precision/recall/F1, per-group exact-match
   rate, and a per-labeler breakdown (human vs `panel_*`). The standalone sliver
   rule in `cbench.eval.sliver` is parity-tested against
   `matcher.config.is_sliver_edge` (`tests/unit/test_cbench_sliver_parity.py`).
   **Now gated (2026-07-05).** The metric was promoted from non-blocking to an
   enforced, auto-arming gate. `cbench run[-batch] --gate` exits nonzero when an
   *armed* dataset's sliver-filtered edge-F1 or exact-match falls below a
   per-dataset floor in `cbench/datasets.toml` (`[gate.<dataset>]`). Because the
   gate needs live pipeline outputs (a bridge parquet + its `*_groups.json`
   sidecar) that don't exist in GitHub Actions (`data/output` is untracked), it
   is enforced at **benchmark time** in the pre-merge checklist for
   matching-logic PRs (see `docs/BENCHMARKING.md` and CLAUDE.md Change Tracking),
   not in unit CI. The gate *machinery* (edge-overlap mapping, sliver filtering,
   arming, floor logic) is regression-tested in CI on a committed miniature
   fixture of real Boston groups (`cbench/tests/test_gate.py`,
   `cbench/tests/fixtures/mini_*`), so the code cannot rot even though the
   live-quality check runs out-of-band.

   **Auto-arming.** A dataset's floor is enforced only once ≥ `min_mapped_groups`
   (30) curated labels map to current pipeline groups; below that the gate
   reports `skip_unarmed` (non-blocking). This makes the gate go live as the
   label base grows with no second PR: `us_boston_streets` (73 labels → 67 mapped
   groups) is already **armed**; `us_seattle_sidewalks` (9 labels → 2 mapped) is
   unarmed and ungated until it grows.

   **Re-measured baselines** (2026-07-05, post-#263/#267, against the committed
   `data/output/*_{bridge.parquet,groups.json}`; the #258 snapshot raw F1 ≈ 0.814
   predated both remediations and is superseded):

   | Dataset | Labels→mapped | Raw F1 | Raw exact | Filtered F1 | Filtered exact | Sliver-affected | Floors (F1 / exact) |
   |---|---|---|---|---|---|---|---|
   | us_boston_streets | 73 → 67 | 0.8345 | 0.5373 | 0.8345 | 0.5373 | 0 | 0.78 / 0.45 |
   | us_seattle_sidewalks | 9 → 2 | 0.6939 | 0.5000 | 0.6939 | 0.5000 | 0 | unarmed |

   (Boston per-labeler: human n=22 F1 0.7695 exact 0.409; panel n=45 F1 0.863
   exact 0.60. Floors are baseline − margin, LOO-gate style: F1 −0.055, exact
   −0.087 — the wider exact margin reflects that per-group exact-match is noisier
   on a ~67-group base.) Remaining: grow stitching labels further (arms more
   datasets), then consider a learned group resolver.
2. **~~Uncalibrated probabilities under five hand-set thresholds.~~ (largely
   addressed — isotonic calibration shipped in #266.)** XGBoost scores are not
   guaranteed to be probabilities; `scoring_match/review_threshold` (0.5/0.1),
   `optimizer_match/review_threshold` (0.75/0.5), and `bridge_min_confidence`
   (0.5) are static config defaults. **Fixed:** `MLMatcher.train` now fits an
   isotonic calibrator on out-of-fold training predictions (leakage-free — the
   seed-42 holdout never participates), stored in the artifact as portable
   knots and applied at inference (`enable_calibration`, default on), so all
   five thresholds now gate on calibrated `P(match)`. On the seed-42 holdout,
   holdout ECE dropped 0.0131 -> 0.0096 and Brier held (0.0619 -> 0.0617);
   accuracy/F1 held-or-improved (0.9116/0.9248 -> 0.9144/0.9283). The model was
   already close to calibrated (`scale_pos_weight` ~= 0.64), so gains are
   modest; the main win is that the thresholds are now semantically meaningful.
   **Measured and rejected:** per-dataset-type calibration (road_good/road_poor/
   sidewalk/other) overfits the small sidewalk/other groups and did not beat a
   single global calibrator overall, so a global calibrator is used and the
   thresholds are left unchanged. **Deferred follow-ups:** (a) fitting the
   thresholds themselves per type from the calibrated PR curve (data-thin, not
   yet justified); (b) corridor-aware grouping + a structural export gate shipped
   in #267, but feeding calibrated confidence into its optimizer group gates is
   still open;
   (c) the Spark export emits calibration knots into `manifest.json`
   (`applied=false`) but the Spark job does not yet consume them
   (tf-data-platform work).
3. **The headline F1 is not "fraction of roads matched correctly."** Labels are
   deliberately oversampled near the decision boundary (human batch: 60%
   borderline; agent: 100% in [0.1, 0.9]), so pairwise F1 on this set
   understates easy-case accuracy and says nothing about end-to-end assignment
   quality. Treat 0.93 as a hard-case classifier score, not product accuracy.
4. **Label base is thin, skewed, and single-annotator.** ~5.5K human labels
   (60% match; several datasets near-all-match; Boston dominant), one labeler,
   no inter-annotator agreement, `unsure` dropped everywhere, corrections
   overwrite in place with no history. At this size ±1–2 F1 points is noise —
   label growth beats further tuning. The agent-labeling trust cascade
   (ensemble-agreement routing: ~86% solo accuracy, 89% when image variants
   agree, coin-flip when they disagree — see `research/agent_eval_full_sweep.md`)
   is designed but unbuilt; `agent_weight=0.0` keeps agent labels out of
   training today.
5. **Known train/serve skews.** The graphlet/clustering skew (full network at
   backfill vs. candidate-only subgraph at inference) is **fixed**: post-#253 all
   consumers — inference, labeling, backfill — build graphlet/clustering graphs on
   the full ref/target networks through the shared `prepare_worker_data()`
   (`features/pipeline.py`). The paired backlog idea — *drop graphlet features
   (skew + Spark-speed twofer)* — was **re-evaluated and rejected on the evidence**
   (2026-07-05, `research/graphlet_drop_reevaluation.md`): both drop rationales are
   now void. The Spark-portable feature set excludes graphlets by construction (no
   Spark inference speed to reclaim), and the local graph build costs only ~0.6–0.9 s
   per dataset post-#255 — trivial next to per-pair work. Graphlet signal is
   weak-but-real and US-concentrated (folded-AUC 0.576 US vs. 0.509 — near-random —
   non-US), so the diverse-geography argument that justifies the topology/endpoint
   features does **not** extend to graphlets. Decision: **keep in the local model,
   keep excluded from Spark** (the current state); locally they are
   redundant-but-harmless (category ablation −0.0016, within one CV-F1 std), which
   the project's pruning policy says not to prune until the label base clears ~10K
   diverse labels — revisit only then. The Spark port still has documented feature
   skews unrelated to graphlets (`class_similarity` tiering, name normalization —
   `docs/SPARK_MODEL_CARD.md`).
6. **Blocking-conditioned ground truth.** All labels originate from blocked
   candidates, so recall against never-blocked true matches (>50–75 m apart,
   MultiLineString targets) is structurally unmeasurable. `matcher
   blocking-recall` guards against regressions but cannot see out-of-candidate
   truth; only out-of-band labeling (e.g., labeling from the unmatched report)
   can.

## Architecture assessment: "high recall then prune"

The candidate-generation → pairwise-scoring → prune/assign shape is the
standard, proven architecture for entity resolution and conflation; all-pairs
scoring is intractable and a cheap high-recall filter in front of an expensive
scorer is the only workable decomposition. **Keep the shape.**

The weak link is *where the intelligence lives*: the learned component is the
pairwise scorer, but road matching is a joint problem (the right match for a
segment depends on its neighbors' matches), and all of that global reasoning
currently lives in hand-written heuristics — greedy-by-confidence assignment,
connected components, 5 m contiguity snapping, average-confidence group gates
(`optimizer.py`). Long-term direction, in order:

1. Make the assignment stage measurable (gap #1), because nothing else about
   it can be decided without that.
2. Make it learned: a group resolver trained on stitching labels
   (edge include/exclude within a component), plus structure-aware scoring
   (seed-and-grow / MRF neighbor propagation — already in
   `docs/RESEARCH_IDEAS.md`).
3. Only if entity-level eval shows a formulation ceiling: consider path-based
   map-matching (snap the local network onto Overture as continuous paths,
   HMM/Viterbi style), which handles segmentation mismatch natively and yields
   per-meter linear referencing. This is a v2 bet, not a refactor.

## Feature-audit follow-ups (2026-07-04, this branch)

Two of the three open follow-ups from the July feature audit (see
`memory/project-feature-audit-2026-07.md`) are implemented here; the third is a
re-measurement gated on backfill.

1. **Target-native topology re-added as separate features.** #252 unified target
   degree/dead-end onto Overture-connector projection for cross-dataset
   comparability, but that *replaced* rather than *augmented* the target's own
   endpoint-cluster structure (old `is_dead_end_target` scored AUC 0.583). Added
   four columns — `from_degree_target_native`, `to_degree_target_native`,
   `is_dead_end_target_native`, `is_intersection_target_native` — sourced from the
   full-segment endpoint-cluster topology (`target_topology_full`), alongside the
   unified comparability features. NaN-gated flags (avoids truthy-NaN → 1.0).
   Wired through `_compute_non_geometric_features` (the shared inference/backfill
   path), so backfill materializes them automatically.
2. **Endpoint-proximity de-degenerated.** `min/max_endpoint_proximity_m` were
   87–91% pinned at the `MAX_DISTANCE_METERS` (10 km) sentinel because the batch
   path measured proximity from a bounded `query_ball_point(r = 2·tolerance ≈
   10 m)` list — anything farther collapsed to the cap. Replaced with an unbounded
   cKDTree k-NN query (skipping the segment's own two endpoints), so proximity is
   now a continuous distance. The tolerance radius query is retained solely for
   `shared_endpoint_count` (a connectivity signal, correctly bounded).

`FEATURE_VERSION` bumped `2026-07-04.1 → 2026-07-04.2`. Stored label features are
now stale, so **retraining and the ablation require a coordinated backfill first**
(the stale-version guard from #225 will otherwise error — working as intended).

### Task 3: Re-run the category ablation

The Feb 2026 verdicts in `docs/ablation_and_dataset_coverage_feb2026.md` are stale
against this codebase. `scripts/ablation_study.py` already derives its categories
from `config.FEATURE_CATEGORIES`, so it picks up the new columns with no change.
Resume with:

```bash
# 1. Coordinated backfill — recompute all label features with the new code.
#    Some non-US datasets need Overture S3; retry -D <dataset> for any that fail.
uv run matcher backfill

# 2. Retrain on the refreshed features.
uv run matcher train

# 3. (optional) Refresh the Spark-portable model.
uv run matcher export-spark-model

# 4. Re-run the category ablation and compare against Feb 2026.
uv run python scripts/ablation_study.py --mode category \
    --output benchmarks/ablation_2026_07

# 5. Supersede docs/ablation_and_dataset_coverage_feb2026.md with the new verdicts,
#    paying attention to: Topology (does target-native recover the lost holdout?),
#    Endpoint/Connectivity (is proximity now informative post-redesign?), and the
#    Parallel Sibling / Graphlet categories flagged in the July audit.
```

## Recommended sequence

1. ~~Scale stitching-group ground truth, then promote the cbench stitch-level
   metric to a gate.~~ **Done (2026-07-05):** promoted to an auto-arming,
   benchmark-time gate (`cbench run --gate`, per-dataset floors in
   `datasets.toml`, CI fixture test of the machinery). Boston is armed (67 mapped
   groups); the gate engages on further datasets automatically as labels grow.
   Continue scaling stitching labels (agent-assisted; separate plan doc) to arm
   more datasets and enable a learned group resolver.
2. ~~Isotonic calibration~~ (shipped, #266); per-dataset-type calibration was
   measured and rejected. Remaining: per-type *thresholds* from the calibrated PR
   curve (deferred, data-thin) and Spark-side consumption of the exported knots.
3. Ground-truth trust cascade for pair labels (ensemble-agreement routing,
   provenance-tiered training weights).
4. Learned group resolver once 100+ stitching labels exist.
5. Revisit formulation (path matching) only with entity-level evidence.
