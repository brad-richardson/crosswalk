# Alignment-truncation investigation: parallel carriageways

**Status:** research note (no production code change proposed here — see "Recommended fix").
**Trigger:** on the live stitching-review map, pairs of near-identical-length,
cleanly-parallel road segments (American Legion Highway dual carriageway, Boston)
show alignment intervals that stop far short of full coverage.

**Reproduction data:** group `701d491e` in
`data/agents/stitching/batches/us_boston_streets_phase2/batch.json`.
Fixture extracted to `tests/fixtures/alignment_truncation_701d491e.json`.

| Edge | ref → target | ref len | tgt len | ratio | conf | stored `gers` | stored `local` |
|------|--------------|--------:|--------:|------:|-----:|---------------|----------------|
| R4-T2 | `726ce920…` → `…_3833_…` | 162.3 m | 159.8 m | 0.984 | 0.99 | `[0.417, 0.999]` | `[0.0, 1.0]` |
| R7-T6 | `9203993e…` → `…_7722_…` | 178.4 m | 176.0 m | 0.987 | 0.99 | `[0.001, 0.724]` | `[0.0, 1.0]` |

Both store **full coverage on one side and a truncated span on the other** for two
near-equal-length segments — 100% of a ~160 m target mapping onto ~58% (94 m) of a
~162 m ref. That is geometrically impossible for a clean parallel match.

---

## 1. The divergence-truncation hypothesis is disproven

The prime suspect was the divergence post-processing in
`linestring_alignment` (`_detect_divergence_endpoints`, `DIVERGENCE_*` constants
in `config.py`) truncating the interval where the two carriageways' constant
lateral offset trips the distance/parallelness thresholds.

**Direct reproduction** — load the two WGS84 geometries from the batch, project to
a local AEQD CRS (as `compute_alignment_batch` does), and run `linestring_alignment`:

```
R4-T2  COMPUTED ref=[0.008, 0.992]  tgt=[0.000, 1.000]   (stored: ref=[0.417,0.999])
R7-T6  COMPUTED ref=[0.007, 0.993]  tgt=[0.000, 1.000]   (stored: ref=[0.001,0.724])
```

The function returns **full coverage on both sides**, not the stored truncation.
`detect_divergence=False` gives the identical result — divergence never fires.

**Why divergence cannot fire here.** Tracing the 32 divergence samples for R4-T2:

- `buffer_distance = 3.33 m`, so `distance_threshold = max(20, 3×3.33) = 20 m`.
- Every sample's point-to-point distance is **0.17–1.10 m** (the two carriageways
  in these stored geometries are essentially coincident, ~1 m apart — *not* the
  15–20 m offset a dual carriageway would have).
- Every sample's direction `dot2` is **0.997–1.000** (well above the 0.5 threshold).

No sample is divergent; `first_good_from_start = 0`, `last_good_from_end = 31`.

**Historical check.** Re-running the reproduction against the alignment code at
`3252d1e` (#217), `6f80ed9` (#186), `c9c8672` (#169) and `63dfd58` (#166) all
return `ref=[0.008, 0.992]`. No released revision produces the stored truncation
from these geometries. The originally-hypothesized fix (make divergence measure the
*change* in lateral offset rather than absolute offset) would therefore fix nothing
here and only risks regressions — **it was not applied.**

## 2. Actual root cause: geometry-identity mismatch in serialization

The stored fractions are internally consistent only with a reference **~275 m**
long: `local=[0,1]` (full 159.8 m target) over `gers`-span `0.581` implies a scored
ref length of `159.8 / 0.581 ≈ 275 m`. But:

- The **raw** Overture segment `726ce920` is **861.1 m** (single row, unique id).
- The geometry **serialized into the batch** is **162.3 m**, and it is exactly the
  tail `[0.811, 1.0]` of the raw 861 m segment (its start projects onto the raw
  line at 698.7 m, off by 0.01 m).
- The geometry actually **scored** was a **third** version (~275 m) — a sub-edge of
  the raw segment.

So a single GERS id `726ce920` is associated with **multiple geometries** across the
run. The alignment fractions were computed against the ~275 m scored edge (and are
correct for it), but the sidecar serialized a *different* 162 m edge for the same
id. On the 162 m geometry the valid fractions land near the tail and read as a
truncated start.

The code-level mechanism is in `_export_groups_sidecar`
(`src/matcher/pipeline/runner.py`):

```python
ref_geom_lookup = dict(zip(reference[ref_id_column], reference.geometry))
```

`dict(zip(...))` silently keeps only the **last** row when an id repeats. Scoring
keys geometry by **positional index** (`ref_idx` into `reference_proj`) and can
distinguish co-id edges; the sidecar re-looks-up by **id string** and cannot.
`MatchResult` (`types.py`) carries only `ref_id`/`target_id`, not the scored edge's
index or geometry, so the identity of the scored edge is lost by the time the
sidecar is written.

> Provenance note: the current raw `us_boston_streets_overture_segments_v1.0.parquet`
> has unique ids, so this batch was generated from a run whose reference input
> contained duplicate GERS ids (an Overture reference split at connectors). The
> *latent code defect* — unsafe id-keyed geometry lookup + id-only `MatchResult` —
> is present on `main` regardless and will resurface whenever the reference carries
> split/duplicate ids.

## 3. Impact assessment

**Features that consume the fractions** (`config.py`):
`ref_coverage`, `target_coverage`, `min_coverage`, `coverage_ratio`,
`aligned_length_m`, plus every aligned-subline feature (buffer IoU, aligned Hausdorff,
lateral-offset sampling, etc.) via the `create_subline(...)` calls in
`ml.py::prepare_worker_data`.

**Training / inference are NOT distorted.** In `ml.py` the alignment and all
derived features are computed against the *same* positional-indexed geometry
(`ref_geom_full` = `reference_proj[ref_idx]`). Fractions and geometry are consistent
within the feature pipeline, so `ref_coverage` et al. are correct for whatever edge
was scored. No feature backfill or retrain is warranted, and **FEATURE_VERSION was
not touched.**

**The damage is confined to the sidecar / stitching-review + agent-batch display.**
Human and agent reviewers see the wrong ref geometry with fractions that look
truncated, which can drive incorrect stitching decisions / labels on otherwise
high-confidence matches.

**Systematic scan** of all 60 phase-2 groups — high-confidence edges (conf ≥ 0.9)
where the two batch geometries are within 10% length of each other but the `gers`
*or* `local` span is < 0.85:

- **25 of 255** qualifying edges (9.8%) show the asymmetric full-vs-truncated pattern.
- Split roughly evenly between truncated-`gers` (full `local`) and truncated-`local`
  (full `gers`) — the signature of a per-id geometry substitution, not a real
  partial overlap.
- Examples: `701d491e/726ce920` (gers 0.581 / local 1.000),
  `701d491e/9203993e` (0.722 / 1.000), `c657000e/52391076` (1.000 / 0.519),
  `8750dc1e/1862c6ee` (1.000 / 0.420), `07c9a3a8/60c067f9` (1.000 / 0.264).

Full list reproducible via the scan in this branch's PR description.

## 4. Recommended fix (not applied — needs design review)

Preserve the identity of the scored edge end-to-end instead of re-deriving geometry
by id at serialization time. Options, smallest-blast-radius first:

1. **Carry the scored geometry (or `ref_idx`) on `MatchResult`** and have
   `_export_groups_sidecar` serialize *that* geometry, rather than
   `dict(zip(reference[id], reference.geometry))`. This is the correct fix but
   touches `MatchResult`, the scorer, and the sidecar writer.
2. **Detect and refuse to silently collapse duplicate ids** in the lookup: if
   `reference[id]` has duplicates, group geometries per id and disambiguate by
   choosing the edge whose `[gers_start, gers_end]` sub-line best matches the
   target — or at minimum emit a warning and mark the group for review.
3. **Guarantee unique reference ids upstream** (suffix split edges, mirroring the
   H3 suffixing already used for target ids), so id-keyed lookups are always safe.

The change is low-risk in principle but non-trivial in surface area and interacts
with the optimizer's id-based grouping, so it is deferred to a dedicated PR rather
than bundled with this note.

## 5. Deliverables in this branch

- `research/alignment_truncation_investigation.md` — this note.
- `tests/fixtures/alignment_truncation_701d491e.json` — real R4-T2 / R7-T6
  geometries + stored edges.
- `tests/unit/test_alignment_truncation.py`:
  - PASSING guards that `linestring_alignment` gives full coverage on the real
    parallel carriageways (protects the divergence thresholds against a future
    change that would truncate near-coincident parallel roads).
  - PASSING check that the stored truncation is not reproducible from the stored
    geometry (encodes the diagnosis).
  - `xfail(strict=True)` documenting the `dict(zip(...))` duplicate-id collapse
    that is the root of the display truncation.
