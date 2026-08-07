# Label-feature staleness: scope for a global backfill + retrain

**Date:** 2026-08-07 · **Status:** scoped, not executed · **Decision needed before starting**

## The problem

21.6% of the labeled training set (1,183 of 5,487 pairs) has a stored
`class_similarity` that disagrees with what current code computes from current
raw data. Training and the LOO gate consume the stored values, so the model is
being fit on inputs that no longer describe the data — silent train/test skew.

Measured with `scripts/tier_penalty_evidence.py`'s loader plus an in-memory
recompute (`class_similarity` depends only on class/subclass strings, so it can
be recomputed without the feature pipeline). It is a **lower bound** on total
staleness: geometry-dependent features drift too wherever raw geometry was
re-fetched, and this probe cannot see them.

Found incidentally while ablating `TIER_PENALTIES` — see
[cross_mode_audit_2026-08-06.md](cross_mode_audit_2026-08-06.md) for the chain
that led here.

## Cause: target-side re-fetch, not code drift

**99.2% of stored features already carry the current `FEATURE_VERSION`**
(`2026-07-07.2`; only 45 rows across 10 datasets sit at `2026-07-04.2`). The
features were computed by current feature code — the *inputs* moved underneath
them afterwards.

`labels/data/` snapshots `ref_class`/`target_class`/`target_geometry` as they
were at label time, so the drift can be attributed exactly. Across 5,543
labeled pairs, compared against current `data/raw/`:

| Drift | pairs | % | side |
|---|---:|---:|---|
| target class changed | 206 | 3.7% | target |
| **target id vanished** | **511** | **9.2%** | target |
| ref class changed | 63 | 1.1% | ref |
| ref id vanished | 162 | 2.9% | ref |

**It is overwhelmingly the target side**, from the 2026-07 target re-fetches.
Reference drift is minor and the 162 vanished GERS ids are ordinary
cross-release churn. There are two distinct failure modes, and they need
different treatment:

**(a) Class remap, ids intact — backfill fixes cleanly.** ke_kisumu_roads 107
(all 87 ids still resolve), nl_amsterdam_roads 37, fi_helsinki_roads 27,
co_bogota_roads 18, ca_toronto_roads 11, de_berlin_roads 6.

**(b) Id re-keying, ids gone — backfill CANNOT fix these.** The re-fetch minted
a new id scheme, orphaning every label against it. `backfill --skip-missing`
(the default) will silently skip them, shrinking the effective training set by
~9% without saying so:

| Dataset | orphaned | evidence |
|---|---:|---|
| sg_singapore_roads | 200 | labels are `sg_road_None_…` — the source id column was **null** at label time; the re-fetch fixed it. 0 of 148 ids overlap |
| us_utah_slc_roads | 199 | `…_201491_…` → `…_854_…`, 1 of 200 overlap |
| us_usfs_flathead | 50 | `…_4354511_…` → `…_26364136_…`, 0 of 45 overlap |
| co_bogota_bike_network | 29 | 0 overlap |
| nl_amsterdam_roads | 13 | partial |
| hk_hongkong_roads, us_usfs_lolo, us_montana_missoula, us_fort_collins_streets, sg_singapore_footpaths | 20 | partial |
| **total** | **511** | |

**All 511 orphaned labels retain their stored `target_geometry` (100%)**, so
they are recoverable by geometric re-keying rather than re-labeling — the same
approach as the existing `scripts/rekey_seattle_target.py` /
`scripts/migrate_seattle_compkey.py`.

**Consequence for scope: no `FEATURE_VERSION` bump is required** (feature
computation logic is unchanged), but a **re-key step must precede the backfill**
or the retrain bakes in a 9%-smaller label base. The bundled-model fingerprint
lockstep still applies (below).

## Blast radius

28 of 34 labeled datasets carry some staleness. Concentrated, not uniform:

| Dataset | labels | stale | % |
|---|---:|---:|---:|
| us_utah_slc_roads | 200 | 197 | 98.5 |
| us_boston_sidewalks | 281 | 194 | 69.0 |
| us_philadelphia_sidewalks | 345 | 112 | 32.5 |
| ke_kisumu_roads | 107 | 107 | 100.0 |
| us_usfs_flathead | 50 | 50 | 100.0 |
| us_austin_sidewalks | 94 | 44 | 46.8 |
| nl_amsterdam_roads | 200 | 43 | 21.5 |
| in_mumbai_streets | 257 | 35 | 13.6 |
| us_fort_collins_sidewalks | 238 | 34 | 14.3 |
| co_bogota_bike_network | 29 | 29 | 100.0 |
| fi_helsinki_roads | 84 | 28 | 33.3 |
| co_bogota_roads | 134 | 25 | 18.7 |
| ca_toronto_roads | 91 | 19 | 20.9 |
| *18 more* | | ≤17 each | ≤10 |
| **total** | **5,487** | **1,183** | **21.6** |

The 100% rows (kisumu, flathead, bogota bike, missoula) indicate wholesale
re-fetch or re-key after features were computed. `us_utah_slc_roads` at 98.5%
and `us_boston_sidewalks` at 69% carry the most absolute weight, and
`us_boston_streets` is the single largest labeled dataset (639) but only 2.7%
stale.

## Plan

1. **Baseline capture.** Record current LOO-by-type macro-F1 per group and the
   training-gate metrics *before* touching anything, on the same seed/folds CI
   uses. (Current measured baseline: road_good 0.9007, road_poor 0.9282,
   sidewalk 0.8584, other 0.9090.)
2. ~~**Re-key the 511 orphaned labels FIRST.**~~ **DONE 2026-08-07** —
   `scripts/rekey_orphaned_labels.py`, **493 of 511 rows recovered (96.5%)**.
   See "Re-key result" below.
3. **Global backfill.** `crosswalk backfill --include-agent` across all
   datasets (5,487 human + 1,051 agent). Default `--skip-missing` tolerates
   pairs that still cannot resolve — **count and report skips**, they are
   silent label loss otherwise.
4. **Measure what moved.** Diff stored-vs-new across all 83 features, not just
   `class_similarity`, so the true staleness (including geometry-dependent
   features) is finally quantified rather than inferred.
5. **Re-measure LOO + training metrics** on the backfilled store.
6. **Retrain + reship** the bundled model
   (`crosswalk train -o src/crosswalk/_model/matcher_model_combined.joblib`)
   and re-export the Spark artifact. Required regardless of the version
   question: changing stored features changes the `labeled_data` fingerprint,
   which is exactly what failed CI on the Helsinki-only backfill.
7. **Reset the LOO floors** in `tests/regression/test_loo_cv.py` to the new
   measured baseline, with the numbers recorded in the PR body.
8. **Boston before/after + stitch quality gate** per CLAUDE.md, since a
   retrained model changes matching output.

**Cost estimate:** the backfill is dominated by per-dataset setup (loading raw
data, STRtree/context construction), not per-label work — Helsinki's 84 pairs
took ~90s including context build over 374k ref + 219k target segments. At 34
datasets, several much larger (London 300k, São Paulo, Sydney), expect
**roughly 1–2.5 hours** for step 3, plus the re-key pass (step 2) and minutes each for 6–8.

## Re-key result (step 2, executed 2026-08-07)

`scripts/rekey_orphaned_labels.py --apply`. Two-tier matching against the
`target_geometry` retained in `labels/data/`: exact WKB equality, then Hausdorff
≤ 0.5 m with **exactly one** candidate in range (ambiguity is refused, never
guessed — a mis-keyed label poisons training data, which is worse than a
dropped one).

**493 of 511 orphaned rows recovered (96.5%).** 18 remain:

| Dataset | rows left | why |
|---|---:|---|
| nl_amsterdam_roads | 13 | nearest Hausdorff 22–38 m — genuinely different geometry |
| us_utah_slc_roads | 4 | nearest Hausdorff 1.2–92.8 m |
| sg_singapore_footpaths | 1 | nearest Hausdorff 31.2 m |

Those 18 point at features that no longer exist upstream; they need re-labeling
or retirement, not re-keying.

**Independent validation** (attributes were never used for matching, so they are
a free check): across the three largest re-keyed sets, the new geometry's length
ratio to the stored geometry is **exactly 1.0000 — min, median and max** (388
pairs), and the target class is identical for 100% of Singapore and Flathead and
96.9% of Utah. The 6 Utah class differences are the class-remap drift described
above, on geometry that still matches exactly — not mis-keys.

Row counts are unchanged (5,487 human / 1,051 agent) and the label→feature join
is **5,487/5,487 before and after**, so no label or feature row was lost.

Note the tier-2 threshold matters more than it looks: only 24 of Singapore's 200
matched on exact bytes, but 148 ids matched at Hausdorff ≈ 0 — the re-fetch
rewrote identical geometry with different WKB.

## Risks and open decisions

- **The floors will very likely move, possibly down.** `other` has already
  drifted from 0.9319 (2026-07-02 baseline) to 0.9090 today without anyone
  touching it, and the Helsinki-only backfill pushed it to 0.8794 — *below* the
  0.88 floor, on 84 pairs. A 1,183-pair correction could move groups further.
  **Standing decision (2026-08-07): if a group lands under floor, stop and
  report the numbers rather than lowering the floor to fit.**
- **A drop is not automatically a regression.** If corrected features score
  worse, the honest reading is that some of the model's current performance
  rests on stale inputs. That needs to be stated plainly, not smoothed over.
- **Retraining changes every dataset's output.** Published releases are
  immutable and skipped, never overwritten, without `--force`
  (docs/PUBLISHING.md:305) — so this should land **before** the R2 publish push,
  or the published `2026-06-17.0` bridges will be built from the pre-retrain
  model with no clean way to update them in place.
- **Agent labels** (1,051) are included; they feed the same store. If their
  provenance should be frozen instead, that is a decision to make before step 2.

## Explicitly out of scope

- **`TIER_PENALTIES` / `compute_class_similarity` constant changes.** Ablated
  2026-08-07 across six variants; every one was flat or worse than baseline on
  LOO macro-F1, including the two the empirical P(match) analysis most strongly
  supported. The constants disagree with observed match rates, but correcting
  them does not help — XGBoost has already learned to compensate. Recorded so
  this is not re-derived later. Harness: `scripts/tier_penalty_evidence.py`.
- Overture reference re-fetch / release bump — deliberately pinned at
  `2026-06-17.0` so this change is measurable in isolation.

## Durable follow-up

Nothing detects this drift. Stored features record a `feature_version` but
nothing records *which inputs* produced them, so re-fetching raw data silently
invalidates the store — which is how 21.6% accumulated unnoticed. A cheap fix:
record a raw-input fingerprint alongside `feature_version` and have `backfill
--dry-run` (or CI) report partitions whose inputs have moved.
