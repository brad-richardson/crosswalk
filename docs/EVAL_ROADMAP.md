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
   pairwise classifier. The optimizer's final assignments are evaluated by ~13
   labeled stitching groups (`labels/stitching/`), ungated. A classifier gain
   that worsens final assignments is invisible. Fix: grow stitching labels to
   100+, gate on stitch-level precision/recall (design already exists in
   `docs/plans/2026-02-21-stitch-eval-design.md`), then consider a learned
   group resolver.
2. **Uncalibrated probabilities under five hand-set thresholds.** XGBoost
   scores are not probabilities; `scoring_match/review_threshold` (0.5/0.1),
   `optimizer_match/review_threshold` (0.75/0.5), and `bridge_min_confidence`
   (0.5) are static config defaults applied uniformly across very different
   dataset types. Fix: isotonic calibration on a held-out fold, then fit
   thresholds per dataset-type group from the calibrated PR curve.
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
5. **Known train/serve skews.** Graphlet/clustering features are computed on
   the full network at backfill time but candidate-only subgraphs at inference
   (`cli/main.py` backfill comments acknowledge this). The Spark port has
   documented feature skews (`class_similarity` tiering, name normalization —
   `docs/SPARK_MODEL_CARD.md`). Either fix the skew or drop the affected
   features (ablation suggests graphlets are near-free to drop).
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

1. Scale stitching-group ground truth (agent-assisted; separate plan doc) and
   gate stitch-level metrics in cbench/CI.
2. Isotonic calibration + per-dataset-type thresholds.
3. Ground-truth trust cascade for pair labels (ensemble-agreement routing,
   provenance-tiered training weights).
4. Learned group resolver once 100+ stitching labels exist.
5. Revisit formulation (path matching) only with entity-level evidence.
