# Structure-Aware Score Propagation — Experiment Record

> _Historical document. The project was renamed `matcher` → `crosswalk` (PyPI `crosswalk-py`) on 2026-07-05; the original name is preserved below unchanged._

> _Harness renamed `cbench` → `mbench` (2026-07-05). Mentions of "cbench" below are historical; the harness is now invoked as `mbench`._

**Status: NEGATIVE result → recommend GRAVEYARD (prototype kept, flag-gated OFF).**
**Date:** 2026-07-03
**Branch:** `experiment/score-propagation`

## Motivation

The pairwise XGBoost scorer is *locally blind*: whether local segment `T` matches
reference segment `R` depends on whether `T`'s topological neighbors match `R`'s
neighbors, yet today all such global reasoning lives in hand-tuned optimizer
heuristics (`optimize_matches_with_grouping`). Score propagation
(RESEARCH_IDEAS.md "Seed-and-Grow Neighbor Agreement (MRF)"; Hootenanny-style
propagation; Volz et al. MRF conflation) is the label-free way to inject structure
between per-pair scoring and the optimizer.

## Design

A post-scoring, pre-optimizer step:
`propagate_scores(results, reference, target) -> adjusted results`
(`src/matcher/matching/score_propagation.py`), wired into `run_pipeline` behind
`settings.enable_score_propagation` (default **False**). With the flag off the
function is never called and the pipeline output is byte-identical (verified).

**Adjacency (reuses endpoint geometry, no new graph):** ref and target segment
endpoints are snapped to a coincidence grid (`junction_coincidence_m`, default 20 m).
A **corner** of a candidate pair `(R,T)` is a grid cell where one of `R`'s endpoints
and one of `T`'s endpoints coincide — the physical intersection where `R` and `T`
enter/leave on the same side ("corresponding side" established purely by geometry).

**Consistent-neighbor (boost) edges:** two pairs `(R,T)`, `(R',T')` that share a
corner cell with `R≠R'` **and** `T≠T'` — i.e. both continue through the same physical
corner. Confident consistent neighbors reinforce a pair.

**Competitor (dampen) edges:** two pairs that share one side (same `T`, or same `R`)
but are **not** adjacent on the other side (their other-side junction cells are
disjoint) — a genuine alternative, not a `1:N` continuation. A confident competitor
dampens a pair. This is the classic parallel-road trap.

**Update (bounded, iterated):** in logit space, each of `n_rounds` damped rounds adds
`alpha · mean(consistent-neighbor agreement) − beta · max(positive competitor
confidence)`, where `agree = 2·s − 1`. The cumulative drift from the original logit is
clipped to `±delta_cap`, so the scorer's calibrated-ish ordering is perturbed, not
destroyed. Decisions are recomputed with the same `scoring_match_threshold` /
`scoring_review_threshold` used by the scorer.

**Parameters** (settings, `MATCHER_`-env overridable):

| knob | default | meaning |
|------|---------|---------|
| `score_propagation_rounds` | 2 | damped propagation rounds |
| `score_propagation_alpha` | 0.6 | boost strength (logit units) |
| `score_propagation_beta` | 0.6 | dampen strength (logit units) |
| `score_propagation_damping` | 0.5 | per-round contraction |
| `score_propagation_delta_cap` | 1.5 | max \|logit drift\| |
| `score_propagation_junction_m` | 20.0 | junction coincidence grid (m) |
| `score_propagation_boost_only` | false | ablation: disable dampen |

**Performance:** fully vectorized (shapely endpoints + numpy `add.at`/`maximum.at`).
Boston (61,880 scored pairs, 96K consistent + 178K competitor edges) propagates in
**0.46 s**; Seattle sidewalks (198,835 pairs, 368K + 450K edges) in **1.7 s**. Meets
the "minutes not hours" bar with wide margin.

## Boston streets — before/after (CLAUDE.md table)

Flag off vs on (default params). Off is **byte-identical** to a `main`-code control
run (verified on all substantive bridge columns).

| Dataset | Metric | Off | On (default) | Delta |
|---------|--------|-----|--------------|-------|
| us_boston_streets | Matched (targets) | 10590 | 10582 | -8 |
| us_boston_streets | Review (targets)  | 520 | 449 | -71 |
| us_boston_streets | Unmatched (approx)| ~203 | ~211 | +8 |
| us_boston_streets | Total bridge edges| 15533 | 15378 | -155 |
| us_boston_streets | 1:1 edges | 1797 | 2027 | +230 |
| us_boston_streets | 1:N groups | 140 | 200 | +60 |
| us_boston_streets | N:1 groups | 829 | 980 | +151 |
| us_boston_streets | M:N edges | 11500 | 10405 | -1095 |

Propagation shifts edges out of tangled `M:N` components toward cleaner
`1:1`/`N:1`/`1:N` — the dampen term prunes cross-links. Whether that is good or bad is
decided by the label-based metrics below.

## Label-based evaluation (cbench vs `labels/human`, stitch vs `labels/stitching`)

### Boston streets (639 labels; 270 positive pairs; 13 stitch groups)

| config | target F1 | target P | target R | pair F1 | pair P | pair R | stitch F1 |
|--------|-----------|----------|----------|---------|--------|--------|-----------|
| **off (baseline)** | **0.9926** | 0.9889 | 0.9963 | **0.9746** | 0.9890 | 0.9607 | **0.9531** |
| default (r2, α=β=0.6) | 0.9944 | 0.9926 | 0.9963 | 0.9708 | 0.9925 | 0.9500 | 0.9429 |
| boost-only (r2) | 0.9926 | 0.9889 | 0.9963 | 0.9746 | 0.9890 | 0.9607 | 0.9438 |
| rounds=1 | 0.9944 | 0.9926 | 0.9963 | 0.9708 | 0.9925 | 0.9500 | 0.9489 |
| rounds=3 | 0.9926 | 0.9888 | 0.9963 | 0.9672 | 0.9888 | 0.9464 | 0.9429 |
| low-beta (β=0.3) | 0.9944 | 0.9926 | 0.9963 | 0.9708 | 0.9925 | 0.9500 | 0.9438 |
| gentle (α=β=0.4, cap 1.0) | 0.9944 | 0.9926 | 0.9963 | 0.9708 | 0.9925 | 0.9500 | 0.9446 |

- **Target-level:** propagation removes **1 false-positive target** (FP 3→2) →
  +0.0018 F1. The only unambiguous win, and it comes from the dampen term.
- **Pair-level:** every propagation config **regresses** (0.9746 → 0.9672–0.9708);
  the dampen term prunes 3 true-positive pairs (269→266). Boost-only is identical to
  off (boost alone removes no pair TPs).
- **Stitch-level:** every config **regresses** (0.9531 → 0.9429–0.9489); recall drops
  from 0.9135 to ~0.90 as legitimate `M:N` continuation edges are pruned.

### Seattle sidewalks (201 labels; 85 positive pairs) — second opinion, different character

Dense, grid-like parallel structure → dampen-dominated (154K dampened vs 14K boosted).
(Stitch labels did not overlap produced groups here → 0 groups evaluated, omitted.)

| config | target F1 | target P | target R | pair F1 | pair P | pair R |
|--------|-----------|----------|----------|---------|--------|--------|
| **off (baseline)** | **0.8750** | 0.9333 | 0.8235 | **0.7919** | 0.9219 | 0.6941 |
| default (r2) | 0.8627 | 0.9706 | 0.7765 | 0.7832 | 0.9655 | 0.6588 |
| boost-only (r2) | 0.8662 | 0.9444 | 0.8000 | 0.7808 | 0.9344 | 0.6706 |
| rounds=1 | 0.8662 | 0.9444 | 0.8000 | 0.7891 | 0.9355 | 0.6824 |
| low-beta (β=0.3) | 0.8645 | 0.9571 | 0.7882 | 0.7862 | 0.9500 | 0.6706 |
| gentle | 0.8627 | 0.9706 | 0.7765 | 0.7832 | 0.9655 | 0.6588 |

- Propagation raises precision (kills FPs) but the heavy dampening prunes **more true
  matches than false ones**: target recall 0.8235 → 0.7765, and **every config
  regresses target F1 and pair F1**. No config recovers the baseline.

## Ablation findings

1. **Dampen vs boost-only.** The dampen term is responsible for both the one target
   win on Boston *and* all of the pair/stitch/Seattle harm. Boost-only is nearly a
   no-op on well-scored corridors (the base model already nails high-confidence
   matches, leaving no headroom) and still slightly hurts stitch grouping.
2. **Rounds.** `rounds=1 < rounds=2 < rounds=3` in harm — more iterations amplify the
   over-pruning. `rounds=3` hits the `delta_cap` (max \|Δlogit\| = 1.5) and is worst.
3. **Beta / gentleness.** Lowering `beta` or `alpha` and tightening `delta_cap`
   monotonically shrinks the effect toward the off baseline but never crosses it into
   net-positive on the composite. There is no interior sweet spot in the swept region.
4. **Bounds respected.** Max \|Δlogit\| stays ≤ `delta_cap` in every run (unit-tested).

## Why it doesn't work here

- The base XGBoost scorer is already ~0.99 pair-F1 on Boston — a **label-free
  structural prior has almost no headroom and can only add noise**. Where there *is*
  headroom (Seattle, ~0.79), the dampen term is **miscalibrated against a competent
  scorer**: many parallel sidewalk/road pairs that "compete" geometrically are in fact
  both correct (dual carriageways, sidewalk-both-sides, service roads), so dampening
  competitors destroys true recall.
- The consistent-neighbor **boost** rarely changes an outcome because confident
  corridors are already confident; it mostly reshuffles `M:N` group formation, which
  the downstream optimizer heuristics were already handling.

## Recommendation: GRAVEYARD

Ship nothing to the default path. The idea, in this unsupervised boost/dampen form, is
net **neutral-to-negative** across two datasets of different character and every knob
setting swept. The single reproducible win (−1 false-positive target on Boston) is not
worth the pair/stitch/Seattle recall regressions, and the labeled eval sets are small
enough (85–270 positives) that these deltas are near noise.

**Kept as a flag-gated (`enable_score_propagation=False`) prototype** because it is
clean, fast, well-tested, and byte-identical when off — a documented dead-end future
work can revisit. What would be needed to make it viable:

- **Learned / validated edge weights** instead of hand-set `alpha`/`beta` — treat
  propagation as an MRF with edge potentials fit against `labels/stitching`, not a
  fixed geometric heuristic.
- **Calibration-aware dampening** that only fires when a competitor is confident *and*
  the pair's own features are weak, rather than on every geometric competitor.
- **A weaker base scorer** (geometry-only model, or noisier datasets) where structural
  priors actually have headroom — this experiment used the strong combined model.

## Reproduction

```bash
# Boston before/after (flag via env; default is off)
uv run matcher stitch -r data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    -t data/raw/us_boston_streets_v1.0.parquet -o data/output/off_boston_bridge.parquet
MATCHER_ENABLE_SCORE_PROPAGATION=1 uv run matcher stitch \
    -r data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    -t data/raw/us_boston_streets_v1.0.parquet -o data/output/on_boston_bridge.parquet

# Ablation (scores once, sweeps params): research/ablate.py
# Label eval (reuses cbench on precomputed bridges): research/eval_bridges.py
# Bridge stat table: research/bridge_stats.py
```
