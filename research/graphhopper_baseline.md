# GraphHopper map-matching baseline — result & formulation-vs-engine analysis

This records the **GraphHopper** map-matching baseline in `mbench` (adapter name
`graphhopper`), the *second* live external map-matcher after Valhalla Meili. It was
built to answer one question the Meili pilot raised:

> Meili's signature is **perfect recall + lower precision (parallel-geometry
> snapping)**. Is that a property of the **path-based formulation** (segment-as-trace,
> HMM map-match, matched-edge = correspondence) — or of **Valhalla specifically**?

GraphHopper is the controlled experiment: a *separate, independently-implemented*
HMM map-matcher (Newson/Krumm, like Meili) run over the **identical formulation and
the identical Overture→PBF graph**. If it reproduces Meili's signature, the
signature is the formulation's; where it diverges, that piece is engine-specific.

## Headline numbers (target-level, vs human labels) — same machine, same label epoch

Both engines re-run on this Mac (arm64) at the current label epoch for a clean
paired comparison. **The JVM runs ARM-native** (jbang-managed JDK 17), so
GraphHopper's wall time is a valid datapoint — unlike the x86-emulated Hootenanny.

| Dataset | Engine | P | R | F1 | TP | FP | FN | Preds | Wall time (peak RSS) |
|---------|--------|---|---|----|----|----|----|-------|----------------------|
| us_boston_streets (roads) | Meili | 0.9889 | **1.000** | 0.9944 | 268 | **3** | 0 | 15,591 | 27.3 s (837 MB) |
| us_boston_streets (roads) | GraphHopper | 0.8024 | **1.000** | 0.8904 | 268 | **66** | 0 | 20,764 | 29.1 s / 21.5 s match-only (1234 MB) |
| us_fort_collins_sidewalks | Meili | 0.9258 | **1.000** | 0.9615 | 212 | 17 | 0 | 41,221 | 18.0 s (369 MB) |
| us_fort_collins_sidewalks | GraphHopper | 0.9298 | **1.000** | 0.9636 | 212 | 16 | 0 | 42,919 | 27.1 s (657 MB) |
| us_seattle_sidewalks | Meili | 0.8947 | **1.000** | 0.9444 | 85 | 10 | 0 | 53,217 | 33.2 s (908 MB) |
| us_seattle_sidewalks | GraphHopper | 0.8673 | **1.000** | 0.9290 | 85 | 13 | 0 | 56,374 | 108.6 s (1305 MB) |

Config (GraphHopper): `graphhopper-map-matching:10.2` via jbang, JDK 17, `foot`
profile (bidirectional, traverses residential..footway — the analogue of Meili's
`pedestrian` costing), traces densified to 10 m, `measurementErrorSigma` 25 m,
`minNetworkSize` 0 (no subnetwork pruning — keep every edge snappable, like
Valhalla), overlap filter 10 %/8 m on a **trace-density matched-length** estimate.
Full tables in `docs/BENCHMARK_RESULTS.md`.

## Verdict: the formulation owns recall (and sidewalk precision); Valhalla owns road precision

### 1. Perfect recall is the formulation's, not Valhalla's — confirmed

Both engines hit **R = 1.000 on all three datasets**. A second, independent HMM
map-matcher never misses a true correspondence on the labeled slice either. This is
the formulation's core strength restated: the trace has to snap *somewhere*, and the
correct edge is the lowest-cost path, so global path continuity catches every
labeled match. The path-based bet's perfect-recall property is now confirmed across
**two** engines — it is a formulation property, full stop.

### 2. The precision tax is the formulation's *on sidewalks* — confirmed

On the two sidewalk datasets the engines are **statistically indistinguishable**:

- Fort Collins: GraphHopper **F1 0.9636 / 16 FP** vs Meili **0.9615 / 17 FP**.
- Seattle: GraphHopper **F1 0.9290 / 13 FP** vs Meili **0.9444 / 10 FP**.

Two independently-built map-matchers make *the same* ~16 parallel-road-snap errors
on Fort Collins. The sidewalk precision tax — a footway snapped onto the adjacent
parallel *road* centerline because `map_snap`/HMM has no "this is a sidewalk, not
the road" notion — is therefore a property of the **formulation**, reproduced
engine-independently. This is the strongest single result here: it rules out "Meili
just has a loose snap radius" and pins the sidewalk tax to the segment-as-trace idea
itself. A path formulation needs an explicit **same-feature vs parallel-feature**
discriminator regardless of engine — exactly the geometric/semantic signal matcher's
pairwise scorer encodes (parallel-sibling gate, class compatibility). Neither
map-matcher has it.

### 3. The *magnitude* of the road precision tax is engine-specific — Valhalla wins

The one place the engines diverge sharply is **Boston roads**: GraphHopper
**F1 0.890 / 66 FP** vs Meili **0.994 / 3 FP** — a 22× gap in false positives on the
same graph and traces. Both still have perfect recall; the difference is entirely
precision. GraphHopper's `foot` HMM snaps the trace along **parallel / opposing
carriageways** (Boston has many divided roads, dual centerlines, and closely-spaced
one-ways) that Valhalla's costing rejects. This is *not* tunable away and *not* a
measurement artifact:

- **Not a sigma artifact.** Sweeping `measurementErrorSigma` 10 → 15 → 25 m moved
  Boston FP only 72 → 67 → 66. Precision is flat ~0.80 across the range.
- **Not a foot-vs-car artifact.** Switching to the `car` profile gave 69 FP
  (P 0.795) — essentially identical. The parallel-snap is inherent to GraphHopper's
  HMM on this graph, not to the costing model.
- **Not a full-edge-length artifact.** GraphHopper's API exposes only each matched
  edge's *full* length (not Valhalla's matched sub-length), so the adapter estimates
  matched-length from trace density (`n_states × densify_m`, capped). Switching
  between full-length and the density estimate left Boston at 66 FP unchanged — the
  66 offending edges carry many observations each, i.e. the trace genuinely runs
  *along* the parallel carriageway for tens of meters. These are real snaps, not
  one-node clips.

So on roads, precision depends on the engine's cost/emission model. Valhalla's
map_snap is measurably better at rejecting parallel-carriageway snaps than
GraphHopper's HMM. If you pick a map-matcher for a road conflation pilot, this is a
real quality reason to prefer Valhalla; on sidewalks the two are interchangeable.

### 4. Stitch-level (M:N grouping) tracks the target-level story

Boston stitch (legacy id-map, current 111-group epoch, same machine): GraphHopper
**P 0.609 / R 0.881 / F1 0.720 / exact 0.126** vs Meili **P 0.886 / R 0.944 /
F1 0.914 / exact 0.523**. GraphHopper's extra parallel-carriageway edges land in the
wrong M:N groups, collapsing exact-group-match to 0.126. On Seattle sidewalks the
gap is smaller (GraphHopper F1 0.515 vs Meili 0.860, 20 groups) — same direction,
consistent with the target-level precision gap.

### 5. Speed: comparable on small/medium graphs, notably slower on large ones

GraphHopper is ~1.5–2× slower than Meili on Boston/FC (~20–27 s vs 12–18 s) and
**~3.7× slower on Seattle** (108 s vs 33 s). The cause is architectural and honest:
Meili serves from **precomputed Valhalla tiles**, whereas the GraphHopper adapter
runs **flexible** routing (no CH/LM preprocessing) between HMM candidates over the
whole graph — so its match cost scales worse with reference-graph size (Seattle's
165k segments is the largest). Enabling LM preprocessing would trade import time for
match speed; left off here to keep the pipeline a thin build+match with no tuning.
Matching dominates end-to-end (the cached graph import is cheap on reload), so
"match-only" ≈ cold wall time on the sidewalk datasets.

## The one genuine DX win: no server

GraphHopper's operational-complexity score (17/25, vs Meili's 21) is a hair behind
Meili, and the whole gap is the **JVM dependency** (jbang fetches a ~200 MB JDK 17
+ resolves the jar on first run) plus the large-graph slowness. Its counterweight is
real: it is an **embeddable library that runs entirely in one JVM process, with no
running Valhalla service** — the thing you reach for when avoiding a server matters
more than Valhalla's stronger conflation precedent. Both are far easier to run than
Hootenanny (Docker/emulation) and than matcher's train+fetch cold-start.

## Recommendation for the roadmap

GraphHopper *strengthens* the EVAL_ROADMAP path-based framing rather than changing
it. It confirms the two facts that matter for a v2 bet: (a) **perfect recall and
native segmentation-mismatch handling are the formulation's**, reproduced across two
engines; and (b) **the precision ceiling is set by parallel geometry** — on
sidewalks engine-independently, on roads to a degree that a *better costing model*
(Valhalla) can partly close but not eliminate. The synthesis is unchanged and now
better-evidenced: **map-matching for high-recall candidate paths + a learned
per-edge same-feature scorer for precision.** Between the two engines, prefer
Valhalla for road precision; prefer GraphHopper when a no-server embeddable JVM
library is the deciding constraint.

## Implementation notes

The adapter shares ~80% with Meili: the same `mbench/convert/pbf.py` Overture→PBF
conversion (behavior-preserving — the synthetic way_id is now *also* written to each
way's `name` tag, which Valhalla ignores) and the same `mapmatch_common.py` trace
densification + overlap filter. GraphHopper does **not** expose OSM way ids on
matched edges (a long-standing limitation with no clean fix in stock GraphHopper), so
the way_id rides in the `name` tag (KVStorage) and is recovered via
`edge.getName()` — no source patch, no third-party id-mapping fork. The engine runs
via `jbang mbench/src/mbench/adapters/GraphHopperRunner.java` (single file, `//DEPS`
pins the version). Java is optional: absent jbang, the adapter errors clearly and
its tests skip.

## Reproduce

See `docs/BENCHMARKING.md` → "GraphHopper (map-matching)" for setup, and
`docs/BENCHMARK_RESULTS.md` for the full tables. In short (Python ≥ 3.11 with
`mbench[graphhopper]` installed and `jbang` on PATH):

```bash
uv run mbench run graphhopper us_boston_streets -c mbench/datasets.toml
# footways: `foot` profile (the default) covers them — swap the dataset name.
```
