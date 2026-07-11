# Feature ablation strategy — 2026-07-11

## Decision

Do not strip features from the production pair model on the current evidence.
The refreshed ablation shows substantial redundancy masking and dataset-specific
feature dead zones:

- `class` and `name_similarity` have the clearest incremental value in paired
  grouped CV.
- `geometric` is the strongest category by itself, despite slightly improving
  the full model when removed. That is complementary/redundant signal, not proof
  that geometry is useless.
- Several categories reverse sign between the single holdout and grouped CV.
  The previous script classified features from the holdout delta even though it
  already computed CV, making its removal recommendations too brittle.
- A family can be globally useful but entirely missing or constant for a dataset.
  Global removal and per-dataset availability are separate questions.

## Tooling changes

`scripts/ablation_study.py` now treats the mean of fold-paired grouped-CV deltas
as the classification and ranking metric. The existing holdout delta remains in
the CSV as a diagnostic, alongside the standard deviation of paired fold deltas.

Two complementary views were added:

```bash
# Standalone signal: train one model per category using only that family.
uv run python scripts/ablation_study.py --mode category-only \
  --labels labels --output /tmp/ablation-category-only

# Storage/variation audit only; no model training.
uv run python scripts/ablation_study.py --mode coverage \
  --labels labels --output /tmp/ablation-coverage
```

Every mode writes `feature_coverage.csv`, with non-null coverage and lists/counts
of all-missing, constant, and usable features per dataset and category. A feature
is “usable” only when it has at least two distinct non-null values in the dataset
slice.

The summary no longer calls any feature “safe to remove” after one ablation run.
A removal candidate must agree across paired grouped CV, permutation importance,
multi-seed stability, category-only signal, and per-dataset coverage.

## Current-main evidence

Data: 5,487 labeled pairs, 33 datasets, 83 features, 17 categories. Baseline at
seed 999: holdout F1 0.9371; five-fold grouped-CV F1 0.9249 ± 0.0061.

### Removal ablation

Negative delta means removing the category hurts. `holdout Δ` and `paired CV Δ`
come from the same models/data and make the instability visible.

| Removed category | Holdout Δ | Paired CV Δ | Fold-delta std | Read |
|---|---:|---:|---:|---|
| class | -0.0127 | **-0.0092** | 0.0057 | strongest incremental evidence |
| name similarity | -0.0124 | **-0.0054** | 0.0040 | useful, dataset-conditional |
| topology | -0.0047 | -0.0024 | 0.0027 | small incremental value |
| crossing angle | -0.0079 | -0.0020 | 0.0026 | small incremental value |
| intersection overlap | -0.0079 | -0.0016 | 0.0027 | small incremental value |
| alignment coverage | -0.0069 | -0.0013 | 0.0041 | uncertain/small |
| lateral offset | -0.0013 | +0.0015 | 0.0031 | removal candidate, unproven |
| graphlet | **-0.0058** | **+0.0017** | 0.0011 | sign reversal; needs multi-seed/permutation |
| geometric | **-0.0054** | **+0.0021** | 0.0053 | sign reversal; heavily redundant |

Most paired deltas are smaller than their fold variability. Even the two strongest
families do not have a clean “all folds agree” story from five folds. The right
conclusion is a ranking of follow-up effort, not a deletion list.

### Category-only models

Standalone performance asks a different question: whether a family contains useful
signal without help from correlated families.

| Category only | Grouped-CV F1 |
|---|---:|
| geometric | **0.8423** |
| parallel sibling | 0.8132 |
| lateral offset | 0.8015 |
| length | 0.7952 |
| topology | 0.7709 |
| alignment coverage | 0.7693 |
| class | 0.7662 |
| crossing angle | 0.7463 |
| name similarity | 0.7230 |
| intersection overlap | 0.7154 |

Geometry is the clearest redundancy-masking example: best standalone family,
but the all-feature model can replace it with correlated offset/length/coverage
signals. Removing it would reduce diversity and likely increase sensitivity to
dataset-specific missingness unless a smaller combined model proves otherwise.

### Dataset dead zones

The coverage report confirms and expands the early-signal panel observations:

- `class_similarity` is fully missing for Hong Kong roads (208 rows), Tunis ML
  roads (200), and Nairobi roads (50). Class cannot help those datasets.
- Only 1 of 10 name features varies for Philadelphia, Boston, Fort Collins, and
  Seattle sidewalks; Tunis; Kisumu/Nairobi; Singapore footpaths; Geneva pedestrian;
  and Bogotá bike. The string similarity columns are mostly all-missing because
  target names are absent.
- `clustering` has only 1 of 3 usable features in most datasets; target/delta
  coefficients are frequently missing or constant.
- Flathead has only 1 of 5 varying `parallel_sibling` features.
- Small datasets make absence look deceptively definitive: Bogotá bike has only
  29 pair labels and several datasets have 50 or fewer. Coverage diagnoses the
  input, but it does not measure generalization.

These are arguments for explicit missingness/availability handling and better
sampling, not for globally disabling the family. Suppressing a constant family
per dataset may save compute, but XGBoost already ignores splits with no variation;
the main value is observability and publish-readiness expectations.

## Recommended experiment ladder

1. **Repeat the paired category removals across at least five seeds.** Require the
   mean delta direction to be stable and report the distribution of paired fold
   deltas. Do not tune the removal threshold on the same runs.
2. **Run permutation importance on the same held-out folds.** The current permutation
   mode uses one holdout; extend it to grouped out-of-fold permutations so it shares
   the ablation universe. A removal candidate must be weak in both views.
3. **Add leave-one-dataset-out evaluation.** Grouped segment CV prevents segment
   leakage but still mixes geographies. LODO should report both macro dataset F1
   and worst-dataset regression, with minimum class/row gates. This is the key test
   for geometry and other redundant families that may rescue feature dead zones.
4. **Test compact bundles, not one deletion at a time.** Start from geometry, then
   greedily add class, name, topology, coverage, and parallel/offset families using
   nested CV. Compare against the full model on F1, calibration, model size, and
   inference latency. Category-only results provide the starting order; they do not
   prove the optimal bundle.
5. **Acquire labels in dead zones.** Prioritize Tunis/Nairobi/Hong Kong for class
   absence; sidewalks, Bogotá bike, Singapore footpaths, and Geneva pedestrian for
   name absence; and non-US topology/crossing cases. Draw labels independently of
   the current matcher candidates where possible to reduce candidate-selection bias.
6. **Gate removal conservatively.** Require no material macro/LODO regression,
   no dataset worse than -0.01 F1 (or a justified small-N exception), stable paired
   CV across seeds, negligible permutation importance, and unchanged candidate recall.
   Keep the existing 10K-label revisit threshold for graphlets.

## Practical priority

The most valuable near-term work is not pruning a few columns. It is persisting the
same typed feature matrix for every resolver candidate, exposing dataset-level dead
zones, and collecting diverse labels that can adjudicate LODO performance. Once that
exists, a compact feature bundle can be selected with evidence rather than by whichever
single holdout happened to win.
