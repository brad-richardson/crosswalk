# Graphlet feature drop — re-evaluation (2026-07-04)

> _Historical document. The project was renamed `matcher` → `crosswalk` (PyPI `crosswalk-py`) on 2026-07-05; the original name is preserved below unchanged._

> **2026-07-11 methodology update:** The category delta quoted below came from a
> single holdout split. The paired grouped-CV harness now reports graphlet removal
> at **+0.0017 CV F1** (fold-delta std 0.0011) while the same run's holdout says
> **-0.0058**—a sign reversal. This strengthens the original “keep for now”
> decision: neither view alone supports removal. See
> `feature_ablation_strategy_2026-07-11.md` for the multi-view removal gate.

**Backlog item:** *"drop graphlet features (train-serve skew + Spark speed twofer)."*

**Verdict: KEEP in the local model; they are ALREADY excluded from the Spark-portable
model. Close the backlog drop item — both of its stated rationales are now void.**
The drop is *not* justified on the stated grounds. Graphlets remain a legitimate
*future* prune candidate on marginal-value grounds, but per the project's own pruning
policy that decision waits for 10K+ labels.

Graphlet features: `graphlet_similarity`, `endpoint_degree_similarity` (category
"Graphlet" in `config.py::FEATURE_CATEGORIES`).

---

## 1. Train-serve skew — RESOLVED (both paths)

The original skew ("full network at backfill, candidate-only subgraph at inference",
documented in `docs/EVAL_ROADMAP.md` point #5) no longer exists.

- **Local model, all consumers share one path.** `prepare_worker_data()`
  (`src/matcher/features/pipeline.py:328-362`) builds graphlet/clustering graphs on the
  **FULL** ref and target networks (not candidate subsets), post-#253. All three
  consumers route through it:
  - inference / scoring — `matching/ml.py:1721`
  - labeling UI — `labeling/data_loader.py:730`
  - backfill (training data) — `cli/main.py:1665`
  There is therefore no train/serve skew for the local model. **`EVAL_ROADMAP.md` #5 is
  stale** and should be updated.
- **Spark production path already excludes graphlets.** `SPARK_PORTABLE_FEATURES`
  (`config.py:464`, 28 features) contains neither graphlet feature. The Overture-scale
  serve model has *zero* graphlet skew because it has zero graphlets. So the skew concern
  is already resolved for production by exclusion, independent of #253.
- **Residual nuance (minor, target-side only):** backfill re-computes the *ref* graphlet
  on the full ref network (`cli/main.py:1683`) but the *target* graphlet is built inside
  `prepare_worker_data` on `augmented_target` (raw data + stored-data segments), which can
  differ from the full target network inference sees. Small, one-sided, and unrelated to
  the drop decision.

**Skew is not a reason to drop.** Fixing it is done.

## 2. Cost — the "Spark speed twofer" is VOID

Graphlets are **not** in the Spark model, so there is no Spark inference speed to reclaim
by dropping them. The "twofer" premise is false.

Local-pipeline cost is now small (measured 2026-07-04, post-#255, warmed numba, cached):

| Network | Segments | Build type | Time |
|---|---|---|---|
| Boston target (local) | 10,844 | inferred connector graph | 0.63 s |
| Boston ref (Overture) | 125,769 | explicit-connector graph | 0.88 s |
| Philadelphia sidewalks target (per code comment) | 204,760 | inferred | ~16 s |

This is a **once-per-dataset** build, run for ref+target concurrently in a 2-thread pool,
and is trivial next to per-pair alignment/feature work over 100K–1M candidate pairs. Cost
is not a compelling reason to drop.

## 3. Value — marginal, and the diverse-geography thesis does NOT extend to graphlets

**Category ablation (July, `benchmarks/ablation_2026_07/`, 5,487 pairs, 83 features):**
`graphlet` F1 delta = **−0.0016**, classified **"redundant"** (`ablation_results.csv`).
That is smaller in magnitude than one CV-F1 std (0.0062) — statistically indistinguishable
from "removing does nothing." It is the 3rd-weakest of 17 categories; only `vertex_density`
(−0.0001) and `lateral_offset` (+0.0005) are weaker. The "16/17 categories carry signal"
framing counts graphlet's negative delta, but its magnitude is within noise.

**Univariate folded-AUC (computed here over 5,400 labeled+feature pairs, 33 datasets;
folded = max(AUC, 1−AUC)):**

| Feature | ALL | US (n≈2,780) | non-US (n≈2,620) |
|---|---|---|---|
| graphlet_similarity | 0.539 | **0.576** | **0.509** |
| endpoint_degree_similarity | 0.534 | 0.565 | 0.506 |
| — clustering_coef_delta (ctx) | 0.506 | 0.507 | 0.504 |
| — degree_match_score (topology, ctx) | 0.563 | 0.547 | 0.574 |
| — buffer_iou_15m (strong, ctx) | 0.784 | 0.828 | 0.732 |
| — class_similarity (strong, ctx) | 0.672 | 0.738 | 0.600 |

The ALL folded-AUC (0.539, i.e. +0.039 over 0.5) reconciles with the reported
"folded-AUC +0.047" (harness/label-version differences aside). It is weak but non-zero.

**Key finding — the diverse-geography argument reverses for graphlets.** The strategic
reason to keep the expanded label set was that topology/endpoint features ablate useful
*only* once non-US pairs are added (doc July takeaway #2). Graphlets do the **opposite**:
folded-AUC is 0.576 on US pairs but **0.509 (near-random) on non-US pairs**. Unlike
`degree_match_score` (0.547 US → 0.574 non-US), graphlet signal is US-concentrated and
evaporates on exactly the diverse geographies that justified keeping those labels. So the
"diverse geography rescues graph features" thesis, which is real for topology/endpoint,
**does not extend to the graphlet category.**

## 4. Verdict

| Original DROP rationale | Status now | Holds? |
|---|---|---|
| Train-serve skew | Fixed for local (#253, shared `prepare_worker_data`); N/A for Spark (excluded) | No |
| Spark speed twofer | Void — graphlets never enter the Spark model | No |

Both stated reasons are gone, so **the backlog item as written is obsolete — do not drop
for those reasons.** On the merits of value, graphlets are *marginal-and-redundant* (−0.0016
ablation within CV std; 0.54 folded-AUC; near-random on non-US), not harmful. The project's
own pruning policy (`docs/ablation...feb2026.md`: redundancy-masking with tree ensembles;
"revisit pruning at 10,000+ labels") says do not prune redundant-but-harmless features at
the current ~5.5K single-annotator label scale.

**Decision: KEEP-but-exclude-from-Spark — which is the current state.** Keep graphlets in
the local model (status quo), keep them out of the Spark-portable model (status quo). Close
the drop backlog item and update `EVAL_ROADMAP.md` #5 (skew is fixed).

**What would make the drop worth revisiting:**
1. Label base clears the ~10K diverse-geography bar (so importance signals stabilize), AND
2. A targeted leakage-free retrain shows the local model's holdout F1 is within one CV-std
   with graphlets removed (expected, given −0.0016), AND
3. Ideally a permutation-importance check (not just ablation) confirms graphlet is not a
   redundancy-masked false negative for the local model.
If all three hold, drop graphlets from the local model too — but on marginal-value grounds,
not the (now void) skew/Spark rationale.
