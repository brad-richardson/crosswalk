# Early-signal agent stitching panel — 2026-07-06

Cheap per-dataset quality signal: the v2 3-voter consensus panel (claude-opus-4-8 /
codex gpt-5.5 / agy Gemini-3.5-Flash, `--panel default`, #291 defaults) run on **5
tier-selected M:N groups per dataset** for every dataset that has a groups sidecar and
had **not** already been paneled. Boston and Seattle are excluded (already done).

Batch naming: `<dataset>_earlysignal1`. Export rule: unanimous among ≥3 valid voters
(`auto_accept`), labeler stamp `panel_unanimous_v3`, `--max-edges 20`.

**The primary deliverable is the issues list** (§2) — this pass exists to surface data /
optimizer problems cheaply, the way the first partial pass caught the Bogotá class-vocab
bug (#293) and the Singapore empty-`RD_CD` synthetic-id bug (#300). Consensus labels are
the secondary yield.

## 1. Results

7 datasets, 35 groups voted, **20 labels exported**. All panels ran clean 3/3 quorum with
a **single** voter failure across all 35 groups (agy abstained on Missoula's 67-edge
interchange — §2 issue 4).

| Dataset | Voted | Unanimous | Majority | None | Auto-accept | **Exported** | Mean conf | Issues |
|---|---|---|---|---|---|---|---|---|
| us_usfs_flathead | 5 | 5 | 0 | 0 | 5 | **5** | 0.912 | 0 (cleanest) |
| tn_tunis_ml_roads | 5 | 3 | 2 | 0 | 3 | **3** | 0.859 | over-selection |
| us_montana_missoula | 5 | 3 | 1 | 1 | 3 | **3** | 0.804 | over-selection, agy fail |
| sg_singapore_roads | 5 | 3 | 2 | 0 | 3 | **3** | 0.863 | over-selection |
| de_berlin_roads | 5 | 3 | 2 | 0 | 2 | **2** | 0.895 | footway↔service vocab |
| co_bogota_bike_network | 5 | 4 | 1 | 0 | 3 | **2** | 0.885 | **cross-class**, edge-cap skip |
| sg_singapore_footpaths | 5 | 4 | 1 | 0 | 2 | **2** | 0.936 | **cross-class** |
| **Total** | **35** | **25** | **9** | **1** | **21** | **20** | 0.874 | |

Unanimous verdicts (25) exceed auto-accept candidates (21) because unanimous
**reject** verdicts (choice = NONE) route to human_review, not export (4 groups:
Berlin 1, Bogotá 1, SG footpaths 2). Auto-accept (21) exceeds exports (20) because
one Bogotá group was over the 20-edge export cap (issue 5).

### Per-dataset group detail

`opt` = optimizer's option letter; `PRUNE a<b` = panel chose a strictly smaller edge set
(a edges) than the optimizer (b edges); `REJECT` = unanimous/majority NONE.

**us_usfs_flathead** — 5/5 unanimous auto-accept, the cleanest dataset in the sweep.
- `2d0aa21a` N:1 unanimous **PRUNE 1<2** conf 0.92
- `2e9224ee` N:1 unanimous opt conf 0.95 · `bdb2b55e` 1:N unanimous opt 0.95
- `dde73dce` M:N unanimous opt 0.887 (12 edges) · `ffd96faf` N:1 unanimous opt 0.853

**tn_tunis_ml_roads** — targets all class `unknown` (ML-derived, unclassed).
- `26c95038` N:1 unanimous opt 0.983 · `de84375c` 1:N unanimous opt 0.977
- `a2abaa19` N:1 unanimous **PRUNE 1<2** 0.81 (exported — a pruned unanimous verdict)
- `0c574647` 1:N majority opt 0.835 → review
- `b450b752` M:N majority **PRUNE 35<67** 0.69 → review (optimizer nearly doubled edges)

**us_montana_missoula**
- `8b4dfb4e` M:N unanimous opt 0.95 · `07c87a6d` M:N unanimous opt 0.87
- `a6ec6a92` 1:N unanimous **PRUNE 1<2** 0.95 (exported)
- `b4c50f8c` 1:N majority opt 0.83 → review (a *parking lot* target — issue 3)
- `29901bc5` M:N **NONE** conf 0.42 → review (67-edge interchange; claude F / codex B /
  agy abstain — issue 4)

**sg_singapore_roads**
- `45ba7148` N:1 unanimous opt 0.963 · `bde6d341` N:1 unanimous opt 0.95 ·
  `73b5224b` M:N unanimous opt 0.903
- `7cd478a6` 1:N majority **PRUNE 1<2** 0.83 → review
- `5dc2107f` M:N majority **PRUNE 7<8** 0.67 → review

**de_berlin_roads**
- `44c3aa68` N:1 unanimous opt 0.98 · `6073f265` 1:N unanimous opt 0.98 (both exported)
- `297cebae` M:N majority opt 0.85 → review · `407c2be3` M:N majority opt 0.75 → review
- `794558c0` 1:N unanimous **REJECT** 0.917 → review (footway→service, panel says no match)

**co_bogota_bike_network** — targets all class `cycleway`, all names empty.
- `2385d968` 1:N unanimous opt 0.943 · `2eee0397` 1:N unanimous opt 0.927 (exported)
- `50b7da83` 1:N unanimous opt 0.91 — **auto-accept but export-skipped (45 > 20 edges)**
- `4ab5d650` 1:N majority **REJECT** 0.825 → review (footway→cycleway, cross-class)
- `b237074a` M:N unanimous **REJECT** 0.82 → review (secondary→cycleway, cross-class)

**sg_singapore_footpaths**
- `22f7b622` 1:N unanimous opt 0.943 · `f32c87a9` 1:N unanimous opt 0.923 (exported)
- `d4d4a468` 1:N majority opt 0.92 → review
- `5792f2b0` 1:N unanimous **REJECT** 0.953 (residential→footway, cross-class)
- `7c651ad7` 1:N unanimous **REJECT** 0.94 (residential→footway, cross-class)

## 2. Issues found (primary deliverable)

### Issue 1 — Separated infrastructure matched to the parallel vehicular carriageway (BLOCKER)

The optimizer matches dedicated **cycleways** (Bogotá) and **footpaths** (Singapore) to
the adjacent road centerline of a **different class**, and the panel rejects them. Four
independent REJECT verdicts across two datasets, at high confidence:

| Group | Match | Class pairing | Verdict | Conf |
|---|---|---|---|---|
| bogota `b237074a` | M:N | secondary → cycleway | **unanimous NONE** | 0.82 |
| bogota `4ab5d650` | 1:N | footway → cycleway | majority NONE | 0.825 |
| sg_footpaths `5792f2b0` | 1:N | residential → footway | **unanimous NONE** | 0.953 |
| sg_footpaths `7c651ad7` | 1:N | residential → footway | **unanimous NONE** | 0.94 |

This is the operational form of the enriched-A/B "cycleway↔vehicular policy gap #3"
(`panel_enriched_ab.md`), now shown to **also hit Singapore footpaths** (a footpath matched
to a residential-road centerline is the same failure as a cycleway matched to a secondary).
The class-consistency gate treats these pairings as NEUTRAL, so the optimizer emits them and
only the panel catches them. The panel is a reliable detector here, but **the optimizer
should not be generating footpath/cycleway ↔ parallel-road matches in the first place.**
Blocks publishing Bogotá and Singapore footpaths until there is an optimizer-side gate or a
class-policy decision.

### Issue 2 — Optimizer over-selection on M:N groups

**Every** non-optimizer panel pick in the sweep was a *strictly smaller* edge set — the
panel only ever pruned, never added:

| Group | Optimizer → panel edges |
|---|---|
| tunis `b450b752` | 67 → 35 |
| missoula `29901bc5` | 67 → 34 (no consensus) |
| sg_roads `5dc2107f` | 8 → 7 |
| tunis `a2abaa19`, missoula `a6ec6a92`, sg_roads `7cd478a6`, flathead `2d0aa21a` | 2 → 1 |

The two 67-edge groups are large highway interchanges (Tunis dual-carriageway, Missoula
I-90/Hwy-200) where the optimizer proposes ~2× the edges the panel accepts. Confirms the
standing over-selection thesis (`optimizer_underselection_investigation.md`). These land in
review (not label-corrupting), but the optimizer's M:N edge selection is measurably too
greedy on big interchanges, and that is where the low-confidence review cluster lives
(conf 0.42/0.67/0.69/0.75).

### Issue 3 — Per-dataset feature degradation (name / class sparsity)

Not code bugs, but each of these makes a whole feature family dead weight and should
calibrate publish expectations:
- **Bogotá:** every target is class `cycleway` with an **empty name** → name features
  contribute nothing; every match is cross-class by construction.
- **Tunis (ML roads):** every target is class `unknown` → class features useless;
  geometry-only matching.
- **Flathead (USFS):** target names are the official NFSR route names, which read as
  landmarks (`HUCKLEBERRY HILL`, `ESTES LAKE`, `PATRICK BOUNDARY`, `DEVIL'S CORKSCREW
  CAMPGROUND`) and essentially never string-match Overture road names. *Investigated and
  cleared* — verified against `source_tags.name`; these are correct road names, not a data
  bug — but name-similarity is near-useless for Flathead.
- **Berlin:** local data labels footpaths/paths `service` where Overture uses `footway` →
  a systematic `footway↔service` pairing (accepted twice as `residential↔service`, rejected
  once at `794558c0` `footway↔service`).
- **Missoula:** local street names are bare stems (`Pine`, `Woody` vs Overture `West Pine
  Street`), and some targets are **parking lots / trailheads** (`Blue Mountain Parking
  Lot`, `19041-A - BLUE MOUNTAIN TRAIL HEAD`) — `b4c50f8c` matched a parking-lot polygon-
  derived line to a service road (majority, review).

### Issue 4 — agy voter fails on the largest pack

The only voter failure in 105 provider-invocations: on Missoula `29901bc5` (67-edge
interchange, the largest pack in the sweep) **agy exited code 1 / ABSTAIN**, leaving two
disagreeing voters (claude F, codex B) → NONE consensus → review. Min-quorum handled it
correctly (no bad export). Failure correlates with pack size/complexity — worth watching agy
on oversized interchange packs.

### Issue 5 — 20-edge export cap drops a valid long cycleway corridor

Bogotá `50b7da83` (1:N cycleway, **unanimous auto-accept**, conf 0.91) was export-skipped
for `over_max_edges (45 > 20)`. A legitimate long cycleway chain the panel fully endorsed
cannot be exported. Minor: consider a higher cap (or corridor-splitting) for bike-network
datasets, where long single-corridor cycleways are common.

## 3. Berlin — fresh vs the enriched-A/B notes

The tier selector drew **entirely different groups** for `de_berlin_roads_earlysignal1`
(297cebae/407c2be3/44c3aa68/6073f265/794558c0) than the enriched-A/B wave
(1dfc9b52/2b98a74f/5503832a/9c1cd4f7/e1f4821a) — the sidecar was regenerated, so
group-by-group comparison isn't possible; compare aggregate behavior:

- **enriched-A/B Berlin:** 3 auto-accepts, 2 review; dominated by **M:N pruning** verdicts
  (unanimous B, unanimous F — non-optimizer *smaller* sets = the over-selection signal).
- **earlysignal1 Berlin:** 2 auto-accept, 3 review; **every** pick was `choice = A` (the
  optimizer letter) — **zero pruning this round**. The two M:N groups both *agreed* with
  the optimizer at majority, plus one clean unanimous full-REJECT (`794558c0`).

Read: no regression, just selection variance — this fresh draw happened to pull optimizer-
agreeing M:N groups rather than the over-selecting ones enriched-A/B hit, so the Berlin
over-selection signal didn't recur here (it did recur strongly on Tunis/Missoula/SG-roads,
issue 2). The **new** Berlin observation is the `footway↔service` class-vocabulary
divergence (issue 3), not seen called out in the enriched-A/B notes.

## 4. Ranked "what to fix before publishing these datasets"

1. **Cycleway/footpath ↔ parallel-carriageway false matches (issue 1).** BLOCKER for
   `co_bogota_bike_network` and `sg_singapore_footpaths`. Needs an optimizer-side class
   gate or an explicit policy for separated-infrastructure ↔ road-centerline matches. The
   panel already detects them reliably; the optimizer shouldn't emit them.
2. **Optimizer M:N over-selection on large interchanges (issue 2).** Tighten edge
   selection where the optimizer proposes ~2× the accepted edges (Tunis/Missoula 67-edge
   groups). This is the entire low-confidence review cluster.
3. **Suppress dead feature families per dataset (issue 3).** Drop/neutralize name features
   where all target names are empty (Bogotá) and class features where all targets are
   `unknown` (Tunis), so the model isn't fed constant/near-constant columns.
4. **Relax the 20-edge export cap for long corridors (issue 5).** Bogotá `50b7da83` is a
   unanimous accept lost to the cap.
5. **agy robustness on oversized packs (issue 4).** Lowest priority — min-quorum already
   contains the blast radius.

**Publish-readiness by dataset (early signal only, 5 groups each):**
- **Ready-ish:** `us_usfs_flathead` (5/5 clean), `de_berlin_roads`, `sg_singapore_roads`,
  `tn_tunis_ml_roads` — roads-vs-roads; only over-selection review cases, no systemic
  false-match class.
- **Blocked on issue 1:** `co_bogota_bike_network`, `sg_singapore_footpaths`.
- **Watch:** `us_montana_missoula` — parking-lot/trailhead targets in the local data
  (issue 3) plus the interchange over-selection.

## Appendix — reproduction

```bash
# Factory-sourced datasets: stage sidecar into data/output (canonical name), then panel.
cp data/factory/release=2026-01-21.0/dataset=<d>/groups.json data/output/<d>_groups.json
uv run crosswalk agent stitch-batch <d> --name <d>_earlysignal1 -n 5
uv run crosswalk agent stitch-run  --batch data/agents/stitching/batches/<d>_earlysignal1
uv run crosswalk agent stitch-export -b data/agents/stitching/batches/<d>_earlysignal1 -d <d>
```

Datasets: `de_berlin_roads`, `tn_tunis_ml_roads` (data/output sidecars);
`co_bogota_bike_network`, `sg_singapore_footpaths`, `sg_singapore_roads`,
`us_montana_missoula`, `us_usfs_flathead` (factory release=2026-01-21.0).

Inventory note: a `release=2026-06-17.0` refresh of `us_montana_missoula` (1914 groups)
appeared mid-run — same dataset re-matched against a newer Overture release, already covered
by this pass; a fresh re-panel against the new release is deferred (no new dataset).
