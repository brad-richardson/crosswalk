# Reference-side label re-key and backfill (2026-08-08)

Companion to `research/label_feature_staleness_scope_2026-08-07.md`, which covered
the **target** side. This one covers the **reference** (Overture / GERS) side after
the 2026-07-22.0 release bump (#479).

## The situation is not symmetric with the target side

On the target side (#473), 511 orphaned labels were recoverable because the id
*scheme* changed while the geometry persisted — a near-pure re-key.

On the reference side, 195 labelled GERS ids (199 label rows, 203 data-store rows)
are absent from the current release. The first probe tested only exact-WKB and
Hausdorff-identical survival and reported "193 have no geometric successor",
which was read as "the roads are gone". **That reading was wrong.** Splits and
merges fail an identity test by construction; the bucket meant "no 1:1 twin",
not "no successor".

Re-measuring with corridor overlap (8 m buffer, successor must lie ≥60% inside
the old corridor, coverage measured against old segment length):

| fate | count | what happened |
|---|---:|---|
| split | 134 | road still there, re-segmented into N pieces (median 5, mean 9.6) |
| gone | 43 | no successor within the corridor |
| reshape | 17 | one successor, geometry edited |
| merge | 1 | absorbed into a longer segment |

Successor counts have a long tail: 65 of the 134 splits have ≤4 successors, but 9
have >20, and `co_bogota_roads` contributes splits into **158** and **172**
fragments. Those are corridor re-segmentation, not splits in a sense any
automated rule should act on.

## Why stale reference keys cost quality

Unlike target-side orphans, these do **not** silently leave the training set —
`crosswalk backfill` splices stored `ref_geometry` back in via
`stored_ref_overrides` (`cli/main.py`), and the backfill reported 0 skipped.

The cost is subtler. A spliced segment is not in the current release's connector
graph, so it has no topology. Measured over the 199 rows:

| feature | orphan NaN rate | normal NaN rate |
|---|---:|---:|
| `graphlet_similarity` | 1.000 | 0.005 |
| `endpoint_degree_similarity` | 1.000 | 0.005 |
| `clustering_coef_ref` | 1.000 | 0.049 |
| `clustering_coef_delta` | 1.000 | 0.166 |

Mean NaN rate across all features: 0.130 for orphan rows vs 0.079 for the rest.
This is train/serve skew — at inference those features are populated — and the
missingness correlates with dataset (mumbai 44, philadelphia 21, boston 20).

## What was done

`scripts/rekey_orphaned_labels.py` gained `--side ref` rather than a second
script, reusing the existing tiered matching, ambiguity refusal, `.prerekey.bak`
backups and collision guard. Target-side behaviour is unchanged by default.

Tier 3 (reference-only) accepts a re-key when **exactly one** successor survives
in the old corridor, covering ≥ `--cov` of the old length with total length
within `--ratio-tol`. Defaults `--cov 0.95 --ratio-tol 0.15`.

**Splits are refused, deliberately.** A `match` label asserts "this local segment
corresponds to road A". When A becomes A1..A5, deciding which fragments the
target actually covers is a stitching decision. Auto-resolving it would fan 134
pair labels into ~1,280 synthetic pairs — roughly 23% of a 5,457-row labelled
base — unreviewed and concentrated in whichever datasets happened to churn. Bad
labels cost more than missing ones. Those belong in a human re-review queue
feeding `labels/stitching/`.

The tight tier-3 bar cost 8 of the 18 single-successor candidates, including one
retaining only 70% of its length and the lone merge at 173%. Re-keying makes
features recompute against geometry the human never saw, so a loose bar silently
mutates what the label asserts.

Result: **11 label rows / 10 GERS ids re-keyed, 0 collisions.** Refusals:

```
141  split — refused, needs human re-review
 27  gone — no successor in corridor
 16  successor covers too little of the old segment
  5  successor length changed too much
```

Post-backfill, ref-orphaned label rows fell 199 → 189, and those 10 rows
recovered live topology. The remaining 189 stay 100% NaN on the four graph
features — that is the split/gone population, and it is a **141-label re-review
queue**, now a known quantity rather than an unmeasured one.

## Model impact

Fingerprint `ddaecf23…` → `4a46d5b5…`. Retrained and reshipped both artifacts
(combined joblib + Spark-portable booster/manifest) in the same change, per the
lockstep gate.

LOO-by-type, both label states re-run at seed 42 on identical code (the pre-run
reproduced the committed baseline exactly, so this is a clean A/B):

| group | pre | post | delta |
|---|---:|---:|---:|
| road_good | 0.8832 | 0.8902 | +0.0070 |
| road_poor | 0.9244 | 0.9307 | +0.0063 |
| sidewalk | 0.8698 | 0.8767 | +0.0069 |
| other | 0.8891 | 0.8786 | **−0.0105** |

Three groups improved; `other` regressed and breached its 0.88 floor. Per
dataset:

| dataset | n | pre | post | delta |
|---|---:|---:|---:|---:|
| ch_geneva_hiking_routes | 50 | 0.6667 | 0.6667 | 0.0000 |
| co_bogota_bike_network | 29 | 0.9643 | 0.9455 | −0.0188 (= one pair) |
| us_boston_bike_network | 86 | 0.9620 | 0.9487 | −0.0133 |
| us_frisco_trails | 177 | 0.9634 | 0.9534 | −0.0100 |

**None of those four datasets had a label re-keyed** (0 of 7 orphans mapped
across them), so the movement is the global feature recomputation, not the
re-key. At n=29 a single flipped pair is worth ~0.019 F1.

The `other` floor was lowered 0.88 → 0.86 by maintainer decision to unblock the
merge. That floor is fitted to a just-measured number rather than independently
derived, unlike the other three — see the note in `tests/regression/test_loo_cv.py`.
The outstanding fixes are unchanged: label `co_bogota_bike_network` up from 29,
and a data-quality look at `ch_geneva_hiking_routes`.

## Boston stitch, before/after

| metric | before | after | delta |
|---|---:|---:|---:|
| match | 14,281 | 14,229 | −52 |
| review | 331 | 259 | −72 |
| 1:1 | 6,023 | 6,034 | +11 |
| 1:N | 973 | 981 | +8 |
| N:1 | 4,869 | 4,874 | +5 |
| M:N | 2,747 | 2,599 | −148 |
| unique local segments matched | 10,507 | 10,489 | **−18** |

Review shrank, but 18 local segments lost coverage. Not a neutral outcome.

## Follow-ups

1. **Re-review queue for the 141 splits** — surface them in `/stitching-review`
   with candidate fragments pre-computed, producing real 1:N labels.
2. **Record `ref_release` on labels.** Nothing today records which Overture
   release a label was keyed against, so every bump turns this into archaeology
   rather than a routine maintenance pass.
3. **Feed spliced refs through `build_inferred_connector_graph`.** It already
   runs during backfill and infers connectivity geometrically. That would
   populate the four graph features for stale-keyed rows without touching label
   semantics at all — cheaper and lower-risk than any re-key, and it addresses
   the majority (189 rows) that re-keying cannot reach.
