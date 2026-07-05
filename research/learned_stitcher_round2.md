# Learned Stitcher Round 2: data fixed, architecture tested, verdict quantified

> _Harness renamed `cbench` → `mbench` (2026-07-05). The "cbench" mention below is historical; the harness is now invoked as `mbench`._

**Date:** 2026-07-05 · **Runner:** `scripts/run_stitcher_round2.py` (PYTHONHASHSEED=0)
**Round 1:** `research/learned_group_resolver_prototype.md` (PR #272)

Round 1 ended in a conditional NO-GO with two suspects: **data** (the sidecar
persisted only optimizer-selected edges, so under-selection was unlearnable;
~40 clean groups) and **architecture** (per-edge independent XGBoost, untested
group-level objectives). Both suspects have now been interrogated separately.

## What changed since round 1

| Gap (round 1) | State now |
|---|---|
| Rejected candidates not persisted | `rejected_edges` + `pruned` records in every sidecar (PR #282/#284) |
| Under-selection unlearnable | **71 under-selection positives** in the table (25% of rejected rows; 59 of them are prune drops the human kept) |
| ~40 clean labeled groups | 94 clean (79 Boston + 15 Seattle) of 126 mapped; 895 candidate edges |
| Production = keep-all (recall 1.0 by construction) | Production = optimizer **+ confidence prune** (#284); on-table recall 0.879 — a real, beatable baseline in both directions |

## Results (grouped CV, out-of-fold; production = sidecar `selected`)

Table-internal comparisons only (the external cbench bar of 0.879/0.583 has a
different denominator — label edges outside the candidate set).

### Pooled (895 edges / 126 groups — includes noisy split-provenance labels)

| model | P | R | F1 | grp-exact |
|---|---|---|---|---|
| **production (optimizer+prune)** | 0.850 | 0.879 | 0.864 | 0.675 |
| oracle-tuned conf threshold | 0.796 | 0.928 | 0.857 | 0.611 |
| round-1 repro (per-edge xgb, thr) | 0.866 | 0.843 | 0.854 | 0.675 |
| ext-feats + threshold | 0.870 | 0.872 | 0.871 | 0.659 |
| ext-feats + expected-F1 select | 0.860 | 0.892 | 0.876 | 0.683 |
| **ext-feats + eF1 + panel soft** | 0.872 | 0.885 | **0.879** | **0.698** |

### Clean slice (495 edges / 94 groups — the trustworthy comparison)

| model | P | R | F1 | grp-exact |
|---|---|---|---|---|
| **production (optimizer+prune)** | 0.875 | 0.964 | **0.917** | 0.787 |
| oracle-tuned conf threshold | 0.869 | 0.937 | 0.902 | 0.734 |
| round-1 repro | 0.877 | 0.877 | 0.877 | 0.787 |
| ext-feats + eF1 | 0.880 | 0.901 | 0.890 | 0.787 |
| ext-feats + eF1 + soft | 0.884 | 0.892 | 0.888 | **0.819** |

Group-bootstrap (2000 resamples) on the clean slice: production beats the best
learned config by **ΔF1 +0.026, 95% CI [−0.000, +0.055], P(learned ≥ production)
≈ 0.028**. The production advantage is small but statistically real at current
label scale.

Per-dataset: Boston (765/106) — ext-feats+eF1 **beats** production on F1
(0.882 vs 0.863) with exact tied at 0.679; Seattle (130/20 — too small to
trust) — production wins F1 0.868, soft-label configs win exact (0.70 vs 0.65).

## The verdict: it was BOTH, and fixing them isn't enough (yet)

1. **Data alone does not flip round 1.** The identical round-1 model on the
   new both-directions table still loses to production (clean 0.877 vs 0.917).
2. **Architecture is directionally right.** Expected-F1 structured selection +
   competition/coverage features is the best learned config in every slice,
   closes most of round 1's gap (0.877 → 0.890 clean), wins pooled F1 and
   Boston F1, and panel soft labels again add group-exact (+3.2 pts clean).
   The set-level objective is what finally moved exact-match — the metric
   round 1 could not budge.
3. **Round 1's headline inverted:** the tuned confidence threshold no longer
   beats anything — production absorbed it as the #284 prune. The bar moved up
   exactly as the prototype's Phase-1 plan intended.
4. **Still NO-GO for replacing the optimizer today**, by a quantified margin:
   production's high recall on trustworthy labels (0.964) is what the learned
   models can't match without paying precision.

## The confounder to fix before round 3 (bigger than model choice)

**Label anchoring.** Every curated label was produced by a reviewer looking at
the optimizer's pre-seeded proposal (option menus + pills). Agreement between
labels and production is therefore partly *causal*, not just evaluative —
production is graded on answers it helped write. The 71 under-selection
positives exist only because reviewers actively added edges; drops of rejected
edges (keep=0 on rejected rows, 218 of them) are systematically easier for
production to "get right" because reviewers rarely saw those edges at all
(pre-#282 UIs never displayed rejected candidates). This bias inflates the
production baseline relative to any challenger, and no amount of modeling on
the same labels removes it.

## Flip conditions for round 3 (ordered by expected impact)

1. **De-anchored labels:** review sessions that display the FULL candidate set
   (selected + rejected, now available in the UI data) for a sample of groups,
   so keep/drop truth is elicited independently of production's proposal.
   Even 30–50 such groups would give an unbiased eval slice.
2. **Scale clean labels ~94 → 200+** (panel waves on Berlin/Tunis add
   cross-dataset structure diversity; sidewalk labels are the scarcest at 15).
3. **Per-edge features beyond the sidecar:** the 78 pairwise features exist
   for only ~5% of group edges; batch-computing them for labeled groups'
   edges (a few thousand pairs) is cheap and unlocks name/class/geometry
   signals the sidecar lacks.
4. Keep the eF1 selector and competition/coverage features — they are the
   validated architecture; a GNN remains unjustified at this scale.

## Reproduce

```bash
PYTHONHASHSEED=0 uv run python scripts/run_stitcher_round2.py
```

Caveat: split-label mapping has a known tie-break wobble (±2 groups run-to-run
across code paths; the runner itself is self-consistent). The clean slice is
bit-stable and is the basis for all conclusions above.
