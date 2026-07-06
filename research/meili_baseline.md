# Valhalla Meili map-matching baseline — result & path-based-formulation pilot

> _Historical document. The project was renamed `matcher` → `crosswalk` (PyPI `crosswalk-py`) on 2026-07-05; the original name is preserved below unchanged._

> _Harness renamed `cbench` → `mbench` (2026-07-05). Mentions of "cbench" below are historical; the harness is now invoked as `mbench`._

This records the first Valhalla **Meili** (map-matching) baseline in `cbench` and
reads it as a **pilot of the path-based-formulation bet** in
`docs/EVAL_ROADMAP.md` (§Architecture assessment #3):

> Only if entity-level eval shows a formulation ceiling: consider path-based
> map-matching (snap the local network onto Overture as continuous paths,
> HMM/Viterbi style), which handles segmentation mismatch natively and yields
> per-meter linear referencing. This is a v2 bet, not a refactor.

Meili *is* that formulation, off the shelf: each local segment is fed to Valhalla
as a synthetic GPS trace and snapped (`shape_match=map_snap`, HMM/Viterbi) onto an
Overture-derived routable graph; the matched edge sequence is the
segment↔GERS correspondence set. So its behavior on our datasets is direct
evidence for/against building matching *around* a path formulation.

## Headline numbers (target-level, vs human labels)

| Dataset | Meili P | Meili R | Meili F1 | matcher F1 | Hoot F1 |
|---------|---------|---------|----------|------------|---------|
| us_boston_streets (roads) | 0.989 | **1.000** | 0.994 | 0.996 | 0.973 |
| us_fort_collins_sidewalks | 0.926 | **1.000** | 0.962 | 0.976 | 0.927 |
| us_seattle_sidewalks | 0.895 | **1.000** | 0.944 | — | — |

Full tables (TP/FP/FN, wall time, stitch metrics) in `docs/BENCHMARK_RESULTS.md`.
Config: `pyvalhalla` 3.7.0 (native ARM, in-process), pedestrian costing, traces
densified to 10 m, `search_radius` 25 m, overlap filter 10 %/8 m.

## What the result implies for the path-based bet

**Verdict: the formulation is viable and complementary, not a free win.** Meili is
the strongest external baseline — beats Hootenanny everywhere, within 0.002–0.014
F1 of matcher — and it does so with a *categorically different* error profile.

### Where Meili wins

1. **Perfect recall (1.000 on all three datasets).** Path-snapping never misses a
   true correspondence on the labeled slice. Concretely, matcher's single Boston
   false-negative — target `us_boston_streets_7943_882a30661d`, a labeled match
   matcher's pairwise scorer dropped — **Meili recovers**, because the trace has
   to snap *somewhere* and the correct edge is the lowest-cost path. This is the
   formulation's core strength: global path continuity catches segments a
   per-pair classifier abstains on.

2. **Segmentation mismatch handled natively — no candidate generation, no
   stitch optimizer.** A single local segment that spans several short Overture
   segments is matched to *all* of them in one trace, in order, for free. This is
   pervasive, not a corner case:
   - Boston: **2,863 / 10,806** matched targets (27 %) split across ≥2 GERS ids.
   - Fort Collins: **2,413 / 38,404** (6 %).
   - Example: Boston `us_boston_streets_4_882a306451` snaps across two GERS ids
     (`dd05d7e8-…` covering 29 % of its length + `59d871bf-…` covering 70 %) —
     the M:N split that matcher needs a whole connected-components + optimizer
     stage (`optimizer.py`) to reconstruct, Meili produces as the raw trace
     output. This is exactly the "handles segmentation mismatch natively" claim,
     confirmed.

3. **Speed (and it's a valid number).** 5–29 s end-to-end, ARM-native, ~6–17×
   faster than matcher's ML pipeline (68–85 s). The dominant cost is the one-time
   Overture→tiles build (~7 s for Boston), amortized across runs by the graph
   cache.

### Where Meili loses

1. **Precision, via parallel geometry — the map-matching tax.** Every false
   positive is a trace snapped onto a *parallel neighbor* of the true feature:
   - **Sidewalks → adjacent road.** Fort Collins FP `fc_sidewalk_3762_88268524c9`
     (a footway) snaps onto a `residential` road centerline **2.4 m** away;
     `fc_sidewalk_35852_8826aa5167` onto a road **3.0 m** away. The sidewalk runs
     parallel to and within a snap radius of the carriageway, so map_snap — which
     has no notion of "this is a sidewalk, not the road" — attaches it to the
     road edge. This is the sidewalk analogue of the classic **divided-highway /
     parallel-carriageway** map-matching failure.
   - Boston road FPs (`us_boston_streets_240_882a306429`,
     `…_4818_882a30663b`, `…_6196_882a306603`) are the same shape on roads:
     a snap onto a parallel/opposing edge that a `no_match` label marks wrong.
   - This is *structural*, not tunable away: tightening `search_radius` to cut
     parallel snaps would start dropping real matches on offset local geometry.
     A path formulation needs an explicit **same-feature vs parallel-feature**
     discriminator — precisely the kind of geometric/semantic signal matcher's
     pairwise scorer already encodes (parallel-sibling gate, name agreement,
     class compatibility). Meili has none of it.

2. **No first-class no-match / abstention.** Map-matching emits a path for every
   trace; "no match" only exists as *below our overlap threshold*. So Meili can't
   natively say "this local segment has no Overture counterpart" — it will snap it
   to the nearest routable thing. On the labeled slice this inflates recall to a
   perfect 1.000 (every labeled-match target does have a counterpart) but it means
   the precision story would worsen on datasets with genuinely-unmatched locals.

3. **Grouping is slightly coarser.** Stitch-edge F1 is high (Boston 0.901) but
   exact-group-match (0.521) trails matcher (0.537): Meili recovers the right
   edges but assembles them into M:N groups a bit less precisely — a direct
   consequence of the same over-snapping.

## Recommendation for the roadmap

The pilot supports the EVAL_ROADMAP framing: a path formulation is a **credible v2
direction** that would (a) eliminate the candidate-generation + stitch-optimizer
stages for the common segmentation-split case and (b) give per-meter linear
referencing for free (Meili already returns `begin/end_shape_index` per edge). But
the result also pins down the **cost of admission**: the intelligence matcher
spends on precision (parallel-sibling rejection, name/class agreement, learned
pair scoring) does not disappear in a path world — it moves into edge-emission /
costing and a same-feature discriminator. The most promising synthesis is
**hybrid**: use map-matching for high-recall candidate *paths* (replacing buffer
candidate-gen and handling splits natively), then keep a learned per-edge
same-feature scorer for precision. Meili alone is a strong recall engine with a
precision ceiling set by parallel geometry.

## Reproduce

See `docs/BENCHMARKING.md` → "Valhalla Meili (map-matching)" for setup, and
`docs/BENCHMARK_RESULTS.md` for the full metric tables and honesty caveats. In
short (from a Python ≥ 3.12 env with `cbench[meili]` installed):

```bash
uv run --python 3.12 cbench run meili us_boston_streets -c cbench/datasets.toml
```
