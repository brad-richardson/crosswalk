# Meili ensemble experiment: can Valhalla map-matching improve matcher?

> _Historical document. The project was renamed `matcher` → `crosswalk` (PyPI `crosswalk-py`) on 2026-07-05; the original name is preserved below unchanged._

**Date:** 2026-07-05 · **Verdict: NO-SHIP (both mechanisms).** The research report
is the deliverable.

## Thesis under test

Valhalla Meili (map-matching) has a **decorrelated error profile** from matcher's
ML scorer:

- **Meili:** perfect recall (1.000 on every labeled slice), precision lost to
  parallel-road / carriageway snapping (it snaps every trace onto *something*).
- **matcher:** precision-first (near-zero false positives), recall bounded — the
  thesis assumed — by candidate generation.

Two ways to absorb Meili as a component were evaluated:

- **(A) Candidate augmentation** — add Meili's matched `(ref, target)` pairs to
  matcher's candidate universe before ML scoring. Does it recover matcher's false
  negatives?
- **(B) Agreement feature** — a `mapmatch_agreement` bit (is this candidate pair
  on Meili's snapped path for that target?) fed to the scorer. Does it improve
  precision / AUC?

## Setup

- Datasets: `us_boston_streets` (roads), `us_fort_collins_sidewalks`,
  `us_seattle_sidewalks` (footways). Raw parquets in `data/raw/`.
- Meili run via the mbench adapter (`pyvalhalla` 3.7.0, Python 3.12, `meili`
  extra), pedestrian costing, densify 10 m, search-radius 25 m, overlap filter
  0.10 / 8 m — the exact `docs/BENCHMARK_RESULTS.md` configuration.
- matcher candidate universe reproduced with the production blocking call
  `generate_candidates(ref, target, ref_id_column="id", target_id_column="id")`
  at the default `buffer_distance_m=50`
  (`src/matcher/blocking/spatial_index.py`) — the same arguments
  `pipeline/runner.py` passes.
- Mechanism B CV is **leakage-aware**: segment-grouped `GroupKFold`
  (`create_segment_groups`, Union-Find over `gers_id`/`target_id`, matching the
  training path in `ml.py`), `DEFAULT_XGB_PARAMS`, threshold 0.5. Meili sees no
  labels, so the agreement bit carries no label leakage; the CV additionally
  guards against segment leakage across folds.
- Scripts: `scratch_mech_a.py`, `scratch_mech_b.py` (kept out of the tree; results
  reproduced below).

## Step 1 — baseline reproduction (target-level, labeled slice)

| Dataset | matcher F1 | Meili F1 | matcher (doc) | Meili (doc) |
|---------|-----------:|---------:|--------------:|------------:|
| us_boston_streets | **0.9963** | 0.9944 | 0.9963 | 0.9944 |
| us_fort_collins_sidewalks | **0.9735** | 0.9615 | 0.9758 | 0.9615 |
| us_seattle_sidewalks | 0.8916 | **0.9444** | — (new) | 0.9444 |

matcher: Boston P/R 0.9963/0.9963 (TP/FP/FN 267/1/1); FC 0.9951/0.9528 (202/1/10);
Seattle 0.9136/0.8706 (74/7/11). Meili and matcher match the recorded baselines
(FC matcher 0.9735 vs doc 0.9758 — one extra FP as the label base has grown since
the doc; immaterial).

**New datapoint:** on **Seattle sidewalks matcher (F1 0.892) trails Meili
(0.944)** — the first dataset in the benchmark where an external baseline beats
matcher on target-level F1. This is the one place the ensemble thesis has real
headroom, and it is examined closely below.

## Step 2 — Mechanism A: candidate augmentation

For each dataset: build matcher's blocking candidate universe, intersect with the
labeled `match` pairs (candidate recall), see what is missing, and check whether
Meili's edge sets recover the misses. Also count wrong pairs (labeled `no_match`)
Meili would inject.

| Dataset | match pairs | **cand recall** | blocking misses | Meili recovers | union recall | `no_match` Meili predicts | of those, **net-new** candidates |
|---------|------------:|----------------:|----------------:|---------------:|-------------:|--------------------------:|----------------------------:|
| us_boston_streets | 280 | **1.000** | 0 | 0 | 1.000 | 3 | 0 |
| us_fort_collins_sidewalks | 212 | **1.000** | 0 | 0 | 1.000 | 17 | 0 |
| us_seattle_sidewalks | 85 | **1.000** | 0 | 0 | 1.000 | 10 | 0 |

**matcher's candidate universe already contains 100 % of the adjudicable true
pairs on all three datasets.** There are zero blocking-stage misses, so Meili has
nothing to recover. matcher's residual false negatives (Boston 1, FC 10, Seattle
11 at final output) are **scoring / optimizer rejections of pairs that are already
candidates**, not candidate-generation gaps. The thesis premise — "matcher recall
limited by candidate generation" — is **false on these datasets**.

Worse for (A): every wrong pair Meili predicts on a labeled `no_match`
(Boston 3 / FC 17 / Seattle 10) is *already* a matcher candidate — **net-new
candidates injected = 0**. So candidate augmentation would add zero recall and,
on the labeled slice, not even any new pairs for the scorer to reject.

> **Honesty caveat (selection bias).** Candidate recall of 1.000 is measured on
> the labeled slice (~3 % coverage), and labels were largely generated *from*
> matcher's own candidates (the labeling UI surfaces matcher candidate pairs), so
> a true match lying outside matcher's blocking would rarely have been labeled —
> the metric is partly circular. Meili's candidate value, if any, is on *unlabeled*
> true matches matcher never surfaces; that cannot be quantified without new
> labels drawn independently of matcher. What we *can* adjudicate says the
> bottleneck is scoring, not blocking. **Mechanism A: no win.**

## Step 3 — Mechanism B: `mapmatch_agreement` feature

Agreement bit set for every labeled pair (1 if Meili matched that `(ref, target)`,
else 0). Pooled contingency over the 3 datasets (n=1064, 577 matches; `unsure`
labels excluded throughout — e.g. Boston has 639 pair labels, 626 after dropping
its 13 `unsure`):

|            | agree=0 | agree=1 |
|------------|--------:|--------:|
| no_match   | 457 | 30 |
| match      | 90 | 487 |

### Feature discrimination (agreement bit alone vs truth)

| Dataset | n | AUC | agree-rate on match | agree-rate on no_match |
|---------|--:|----:|--------------------:|-----------------------:|
| us_boston_streets | 626 | 0.937 | 0.882 | 0.009 |
| us_fort_collins_sidewalks | 238 | **0.605** | 0.863 | **0.654** |
| us_seattle_sidewalks | 200 | 0.792 | 0.671 | 0.087 |
| POOLED | 1064 | 0.891 | — | — |

The bit is a strong signal on roads (Boston 0.94) but **near-useless on Fort
Collins sidewalks (0.60)** — Meili's agreement fires on **65 % of labeled
no_match pairs** there, the parallel-sidewalk snapping tax. This is precisely the
regime (dense parallel footways) the ensemble thesis targeted, and it is where the
feature is weakest.

### Leakage-aware GroupKFold CV (pooled 3 datasets, 5 folds, 894 groups, n=1064)

| Model | F1 | AUC | agreement importance |
|-------|---:|----:|---------------------:|
| baseline (83 feat) | 0.9437 ± 0.0104 | 0.9888 | — |
| + agreement (84 feat) | 0.9495 ± 0.0110 | 0.9889 | 0.339 |
| **delta** | **+0.0058** | **+0.0001** | |

The threshold-independent **pooled AUC delta is +0.0001** — no aggregate
discrimination gain. Two honesty qualifiers on that headline: (1) the pool is
**ceiling-dominated** — Boston is 59 % of it (626/1064) with base AUC 0.995,
where no feature has room to add separation, so "+0.0001 pooled" should be read
as "no gain where the base model is saturated; see the per-dataset table for the
non-saturated case (Seattle)". (2) The F1 comparison against the fold std
(±0.011) is indicative only — that std describes the fold-to-fold *level* of F1,
not the variance of the paired base→augmented *delta* (folds are shared between
the two runs), so it is not a formal noise test; the AUC delta is the
load-bearing evidence. XGBoost assigns the feature nontrivial importance (0.34)
yet gains ~nothing held-out — the classic signature of a feature **largely
redundant with existing geometry features** (the model reshuffles importance
onto it without improving separation).

### Per-dataset out-of-fold delta (base → +agreement)

| Dataset | AUC | F1 |
|---------|----:|---:|
| us_boston_streets | 0.9953 → 0.9950 (−0.0003) | 0.9691 → 0.9710 (+0.0019) |
| us_fort_collins_sidewalks | 0.9771 → 0.9782 (+0.0011) | 0.9714 → 0.9713 (−0.0001) |
| us_seattle_sidewalks | 0.9327 → **0.9407 (+0.0080)** | 0.8022 → **0.8333 (+0.0311)** |

There is a **kernel of truth in the thesis, isolated to Seattle** — the one
dataset where matcher underperforms Meili. Adding the agreement feature nudges
Seattle OOF AUC +0.008 / F1 +0.031. But: (1) it is small-n (200 pairs, 85
positives — a handful of flips), (2) it washes out completely in the pooled AUC
(+0.0001), and (3) Boston and FC are flat-to-negative. The decorrelated signal
helps only where matcher is weakest, and not enough to move the aggregate.

## Step 4 — stitch-level quality gate (unchanged)

Docs-only change; matching logic untouched. For the record, the armed Boston gate
passes at baseline: `mbench run matcher us_boston_streets --gate` →
**PASS** F1 0.8725 ≥ 0.83, exact 0.5135 ≥ 0.50 (111 mapped groups).

## Verdict & recommendation

**No-ship, both mechanisms.**

- **(A) Candidate augmentation — no measurable win, and the upside is
  unmeasurable on current labels.** Candidate recall is already 1.000 on the
  adjudicable slice; matcher's FNs are scoring decisions on in-universe pairs;
  Meili injects zero net-new candidates on the labeled slice. Any real upside
  would live on unlabeled true matches matcher never surfaces, which the current
  (matcher-seeded) label base cannot adjudicate — so on the evidence available
  there is only downside risk (FP injection), no demonstrated upside.
- **(B) Agreement feature — does not clear the bar.** Pooled leakage-aware AUC
  delta +0.0001 (ceiling-dominated pool; see qualifiers above); feature actively
  low-quality on dense sidewalks (FC AUC 0.60). The only movement (Seattle) is
  small-n and aggregate-neutral.

The shipping cost is steep and structural, which makes a ~0 global gain a clear
no: a new feature bumps `FEATURE_VERSION`, forcing **re-export of both bundled
models** (`src/matcher/_model/` joblib + Spark JSON; the `test_shipped_model*`
lockstep tests fail otherwise), an **optional Valhalla dependency + config flag**,
a **full `matcher backfill`**, and NaN-degradation plumbing so matcher stays
functional without the `meili` extra. High maintenance surface, Valhalla
unavailable in Spark, for no measurable aggregate quality gain.

**Why the thesis fails despite Meili's decorrelated errors:** the ensemble bet
assumed matcher's recall was candidate-bound (A) and that Meili's snap carried
signal orthogonal to matcher's geometry features (B). Neither holds — matcher's
blocking already achieves full recall on adjudicable pairs, and Meili's "is it on
the snapped path" signal is largely a re-encoding of geometric proximity /
alignment that matcher's 83 features already capture (hence importance 0.34 with
zero AUC gain), while adding a fresh failure mode (parallel-snap agreement on
no_match sidewalks).

**Narrow future revisit (not now):** the Seattle result is the one live lead. If
(a) many more sidewalk datasets get labeled and Seattle's small-n signal
replicates across them, **and** (b) the Valhalla dependency is already being taken
on for another reason (e.g. a map-matching-native product path from
`docs/EVAL_ROADMAP.md`), then a `mapmatch_agreement` feature gated behind the
optional extra could be worth re-measuring — scoped to footway datasets, where
matcher currently trails. Absent both, the dependency and model-reship cost are
not justified.
