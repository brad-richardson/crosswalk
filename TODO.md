# Matcher TODO

Actionable backlog for the road network matcher.

- For tried-and-removed features, see [docs/RESEARCH_GRAVEYARD.md](docs/RESEARCH_GRAVEYARD.md).
- For exploratory research ideas, see [docs/RESEARCH_IDEAS.md](docs/RESEARCH_IDEAS.md).

---

## Known Issues & Technical Debt

### Medium: Make Both Overture Fetch Paths Release-Consistent

**Problem:** The two Overture fetch paths can leave a configured dataset with
mixed-release reference artifacts:

1. `crosswalk data fetch reference|overture` writes the canonical segment and
   connector paths, but its non-`--force` skip gate checks only that both files
   exist. It does not verify that their metadata releases agree with each other
   or with the requested/latest release.
2. The YAML-free `crosswalk fetch-overture` command defaults to segments-only.
   With `--connectors`, it appends `_connectors` to the output stem; an output
   named `*_overture_segments_v1.0.parquet` therefore produces the noncanonical
   `*_overture_segments_v1.0_connectors.parquet` instead of
   `*_overture_connectors_v1.0.parquet`.

**Observed impact (2026-07-13):** A June segment refresh left 15 canonical
connector sidecars on the January release, while the configured dataset's
aggregate `last_fetch.reference` timestamp reflected only the newer segment
fetch. This did not change June GERS ids/geometries, but it made factory
topology features and provenance mixed-release. One redundant Singapore
connector file was also written under the noncanonical name.

**Patch both paths:**

- Add a shared reference-artifact audit that reads segment and connector
  metadata together and checks release, bbox/buffer, schema/data version, and
  canonical paths before a fetch is skipped or a factory run starts.
- Make `crosswalk data fetch reference|overture` refetch (or fail loudly with a
  precise `--force` instruction) when either canonical artifact is missing,
  unreadable, or release-inconsistent; update `last_fetch.reference` only after
  the pair validates.
- Make `crosswalk fetch-overture --connectors` derive the canonical connector
  sibling when the output follows the configured
  `*_overture_segments_v*.parquet` convention. Warn or refuse when a
  segments-only invocation targets a configured canonical raw-data path, while
  preserving segments-only behavior for ordinary YAML-free usage.
- Add regression tests for stale-segment/fresh-connector and
  fresh-segment/stale-connector pairs, missing metadata, the canonical filename
  derivation, and a fully current no-op fetch.

**Location:** `src/crosswalk/cli/data.py`, `src/crosswalk/cli/main.py`,
`src/crosswalk/fetch/overture.py`, and fetch/factory provenance tests.

### HIGH: Scalability - Large Dataset Support

- **Problem**: `runner.py` uses `geopandas.read_parquet` which loads entire dataset into memory
- **Impact**: Will fail on state-sized or larger datasets (target: 300M segments for Overture vs OSM)
- **Location**: `src/crosswalk/pipeline/runner.py`
- **Solution**: Add PySpark + Sedona distributed backend alongside existing GDF path. Dual-backend architecture with `StitchExecutor` protocol, shared feature math, H3 spatial partitioning with halo for boundary correctness. See [design doc](docs/plans/2026-02-22-spark-dual-backend-design.md) for full stage-by-stage analysis, abstraction approach, risks, and implementation phases.

### Medium: Divergence Detection Fails on Winding Roads (PR #81 follow-up)

**Problem**: `_detect_divergence_endpoints` doesn't truncate the ref subline when the reference loops far away from the target and back. The 1D offset model extends the ref subline through the loop because:

1. **No target parameter bounds check**: When `t - offset` falls outside `[0, target_length]`, `_interpolate_along_line` silently clamps to the target endpoint instead of flagging the sample as out-of-bounds. So the ref can extend arbitrarily beyond the target's actual extent.
2. **Coarse inter-sample direction vectors**: Direction parallelness is computed between consecutive samples, not from local line tangents. With 32 samples over a long comparison region, a winding loop within a single sample gap averages out and can pass the dot product check.
3. **20m distance threshold too generous**: `DIVERGENCE_MIN_DISTANCE_M = 20.0` lets winding roads stay "close enough" when they loop back near the target.

**Fix options** (in order of simplicity):
1. ~~**Target parameter bounds check** (simplest): At line 256, if `t - offset < 0` or `t - offset > target_length`, mark the sample as diverged. This directly truncates where the ref extends beyond the target's extent.~~ **Done** (alignment.py lines 268-275).
2. **Monotonicity constraint**: Track the projected target parameter and truncate where it stops increasing (the ref is going somewhere the target doesn't).
3. **Local tangent-based direction**: Compute bearing from each line's own vertex segments instead of inter-sample differences.

**Reproduction pair**:
- GERS: `e720aba9-61d0-410a-b5f2-29eba4ae3048`
- Target: `us_montana_helena_223490_8827927a87`
- Dataset: `us_montana_helena`

<details>
<summary>WKT geometries</summary>

REF:
```
LINESTRING (-112.232657 46.720897, -112.2324275 46.7219862, -112.232585 46.723373, -112.232711 46.724186, -112.232697 46.725128, -112.232742 46.725635, -112.232753 46.726053, -112.232799 46.726247, -112.233405 46.726642, -112.234104 46.726964, -112.234617 46.727295, -112.23492 46.727625, -112.234937 46.727649, -112.235094 46.727867, -112.234941 46.728349, -112.234753 46.728993, -112.234063 46.729484, -112.233782 46.729701, -112.233963 46.729785, -112.235694 46.730586, -112.235963 46.730595, -112.236395 46.730385, -112.236548 46.730087, -112.236968 46.729644, -112.237354 46.729427, -112.238007 46.729564, -112.238556 46.729661, -112.2395 46.729725, -112.2402 46.729701, -112.241156 46.729733, -112.241892 46.72949, -112.242203 46.729526, -112.242523 46.729563, -112.242626 46.730232, -112.242765 46.730539, -112.242717 46.731038, -112.242564 46.731537, -112.242319 46.73222, -112.242166 46.732543, -112.242189 46.732817, -112.242446 46.732793, -112.242622 46.732423, -112.242868 46.732101, -112.24316 46.731828, -112.243394 46.731595, -112.243698 46.731419, -112.243827 46.731113, -112.24377 46.730791, -112.243735 46.730486, -112.243934 46.73026, -112.244075 46.729971, -112.24417 46.729616, -112.244136 46.729303, -112.244114 46.728925, -112.24401 46.72853, -112.24373 46.728264, -112.24331 46.728176, -112.242773 46.728199, -112.242108 46.728087, -112.241571 46.72807, -112.240917 46.728118, -112.240393 46.727964, -112.24009 46.727715, -112.2398836 46.7272639, -112.2402382 46.7274812, -112.2406133 46.7276011, -112.241087 46.7276168, -112.2415235 46.7275282, -112.24377 46.726916, -112.244143 46.72694, -112.244575 46.727093, -112.244878 46.727238, -112.245228 46.727239, -112.245625 46.727135, -112.246104 46.727119, -112.246349 46.72716, -112.246547 46.72733, -112.246721 46.727708, -112.24685 46.727917, -112.247222 46.728425, -112.247875 46.728634, -112.248318 46.729111, -112.2491526 46.7294596)
```

TGT:
```
LINESTRING (-112.23242434877292 46.720523234706526, -112.23247237989453 46.72059746576035, -112.23254796324423 46.720698505678456, -112.23261906939054 46.72080163250172, -112.23266980983105 46.72090344923081, -112.23270514326612 46.72099068387334, -112.23276180450267 46.721092987991256, -112.23279964064412 46.721188267652536, -112.23286240593302 46.721287002613686, -112.23293005715877 46.721378776587564, -112.23293260657752 46.7213799762383, -112.23293429900353 46.721382367533764, -112.2329348335011 46.72138388434046, -112.23293531948968 46.721386212820356, -112.23293550633926 46.72138680094473, -112.23296738485375 46.7214227325603, -112.23303019416011 46.72148877299183, -112.23309338435214 46.721546877690606, -112.23316508338654 46.72161774919121, -112.23323851527113 46.72169181985412, -112.23331373120985 46.7217700817726, -112.23340131695007 46.72185723310831, -112.23348036689842 46.72194909171271, -112.23353880859587 46.72205036188311, -112.23358417351771 46.72216743771035, -112.23363341556833 46.722283518723216, -112.23370323263222 46.722402437825906, -112.23377782873341 46.72251746096316, -112.23384974156684 46.722626063286114, -112.2339367416055 46.72274033710756, -112.2340223061363 46.72283924350553, -112.23411222749624 46.72293624622463, -112.23419907212633 46.723044848936205, -112.23427727855665 46.72314582515257, -112.2343341571855 46.723244609491424, -112.23441453933543 46.723347153813826, -112.23450059883797 46.72345316497114, -112.23458493177853 46.72353805109306, -112.23468447050189 46.72361806970704, -112.23476658101043 46.723679863282996, -112.23489458015865 46.72377099582335, -112.23500722709866 46.72384862721647, -112.23512775406031 46.72392670003219, -112.23524594270731 46.72400257676661, -112.23537322140666 46.724089762675966, -112.23548985417156 46.72415306751405, -112.2356483592085 46.724225904934336, -112.23575852500167 46.72426756841183, -112.2358990852925 46.72430165870298, -112.23605434112305 46.72432602789971, -112.23621906160344 46.72435620410935, -112.23655901195472 46.72444809479117, -112.23672085962284 46.724491308685465, -112.23687690237757 46.72454654913935, -112.23701131819186 46.72461437957305, -112.2371288286108 46.72467162740742, -112.23724765595998 46.72473938132479, -112.23737461845232 46.724828014399726, -112.23748355355356 46.72488663892968, -112.23758985479438 46.724944966582086, -112.23769120362307 46.7250037726447, -112.2379732404863 46.725113084900045, -112.23822578386213 46.72524246747956, -112.2383698511758 46.72542012347371, -112.23855073773805 46.72565732564286, -112.23873771128466 46.725792698522355, -112.23892433089505 46.726098752285175, -112.2390897951805 46.72642918056333, -112.23927186572227 46.726689501323925, -112.23940937084281 46.726871906331276, -112.2395720692155 46.72701659773178, -112.2397829397453 46.72717877417796, -112.23993988890015 46.72728430674189, -112.24006963347495 46.72737154830854, -112.24041584688041 46.72753965167486, -112.24064989035126 46.72762687558708, -112.24088578075835 46.72761660399675, -112.24118142170823 46.72758625042294, -112.24157080185805 46.72753967753705, -112.24172227757788 46.7274664110906, -112.24195976968112 46.727353875127704, -112.24223465864964 46.727314741767174, -112.24274903487964 46.72720481204493, -112.24310100289279 46.727104949352096)
```

</details>

**Location**: `src/crosswalk/features/alignment.py:_detect_divergence_endpoints()` (lines 185-325), config thresholds in `src/crosswalk/config.py:47-49`
**Related PR**: #81 (original divergence detection), #169 (multi-seed offset fix)

### Low: Mid-Alignment Divergence in M:N Groups (Pond Pattern)

**Problem**: When ref and target segments share start/end points but diverge in the middle (e.g., paths around different sides of a pond), `_detect_divergence_endpoints` doesn't catch the divergence because it only scans inward from the edges. Start is good, end is good → no truncation → full alignment reported for the cross-side pair.

**Example geometry:**
```
        ___R1 (long - approach + left side + departure)___
       /                                                   \
  ----*                                                     *----
       \___R2 (short - just right side)___/

        ___T1 (short - just left side)___
       /                                  \
  ----*                                    *----
       \___T2 (long - approach + right side + departure)___/
```

R1↔T2 matches full-length (shared approach/departure, different middle). R2 and T1 are short stubs on the opposite sides. The 2×2 M:N group needs location-dependent assignment but currently can't disambiguate.

**Frequency**: ~10/5000 labeled pairs, trail/path datasets only.

**Current decision**: Accept as-is. These are informative matches (useful for coverage). Similar class of issue as onramp termination differences and digitization quirks. Not worth over-engineering for the current use case.

**Future options if needed** (lightest → heaviest):

1. **Alignment quality profile features** (~50-100 lines): Sample points along the full aligned region, compute summary stats (`divergent_fraction`, `max_contiguous_good_fraction`, `alignment_quality_variance`). Add as ML features so the model can learn that pairs with good endpoints but bad middles are suspicious. No schema changes.

2. **Multi-segment alignment** (~200-300 lines): Modify `AlignmentResult` to support a list of disjoint aligned segments. For the cross-side pair: segment 1 at frac 0.0–0.2, gap, segment 2 at frac 0.8–1.0. New features: `alignment_segment_count`, `total_gap_fraction`. Touches alignment, features, and bridge schema.

3. **Post-hoc M:N disambiguation in optimizer** (~100-150 lines): After scoring, for M:N groups, sample the middle of each alignment and compute local geometric similarity to re-rank or prune edges. Targeted but heuristic.

4. **Label correctly and accept** (current approach): Label cross-side pairs appropriately, accept residual error. Existing features (hausdorff, lateral offset) may provide weak signal for the model.

**Related**: "Divergence Detection Fails on Winding Roads" (above) addresses edge-divergence; this is about middle-divergence.

**Location**: `src/crosswalk/features/alignment.py:_detect_divergence_endpoints()`, `src/crosswalk/matching/optimizer.py`

### Medium: Robustness Issues

- **Overly broad exception handling** in `blocking/spatial_index.py` — `except Exception: return None` silently swallows errors

### Low: Datasets with Polygon Geometries

Some target datasets have Polygon geometries instead of LineStrings (files deleted, need re-fetch):
- `ca_toronto_roads`, `co_bogota_bike_network`, `co_bogota_sidewalks`

---

## Stitch Ground Truth

### DONE (2026-07-17): Targeted Physical/Frontage Stitch Wave — all 65 packs voted

The reviewed v7 production manifest (built from clean merged commit `70957a2`)
**completed 2026-07-17: 65/65 packs voted and consolidated** with full 3-seat
panels (`claude-opus-4-8`/high, `gpt-5.6-sol`/high, `meta/muse-spark-1.1`/high).
The wave resumed from the 7 pre-existing packs at `--group-workers 2` (quota
headroom); no Codex/Claude quota halt occurred. `stitch_runner.py` was untouched
throughout, so every ballot stayed resume-compatible.

**Wave provenance is preserved in git** (2026-07-17): the wave is **195 ballots
= 65 group-runs × 3 seats over 50 unique groups** (0 errors, 0 abstains; every
NONE is a decisive verdict), all archived to the tracked `labels/votes/dataset=*/`
tree via `write_vote_provenance`. (Note: `write_vote_provenance` *accumulates*,
so it reported a post-merge file total of 282 vote / 94 consensus rows for these
8 datasets — that total folds in ~87 prior-wave rows; the v7 wave itself is 195
ballots, confirmed independently by both analysts.) The 660 MB / 3.8k-PNG batch
working dirs under `data/agents/stitching/batches/` stay git-ignored by design —
`labels/votes/` is the durable form.

**⚠️ OTHER PIECE — deferred label minting (do not lose this):** the wave's
**6 auto-accepts** (unanimous/quorum) were **NOT** minted into
`labels/stitching/` as production `panel_*_v7` labels. The v7 composition is
still a nonstandard/unpromoted panel, and minting production labels from it is a
provenance decision that should wait until the wave analysis validates v7. Once
validated, mint per dataset with:

```bash
crosswalk agent stitch-export -d <dataset> \
  -b data/agents/stitching/batches/<dataset>_physical_context_v7_20260715[_variant] \
  --allow-nonstandard-panel --stamp-era v7
```

(Only the 6 auto_accept groups export; the 52 human_review/NONE groups stay in
review by design.)

**New (2026-07-17): the wave driver auto-exports on completion.**
`scripts/run_physical_stitch_wave.py` now calls `_auto_export_wave()` after a
clean (non-paused) run: it **always** archives vote provenance to
`labels/votes/`, and mints `labels/stitching/` labels **only for blessed
panels**. A nonstandard/unpromoted panel (like this v7) has label minting
withheld with an explicit notice + the exact `stitch-export` command above — so
no future wave is lost to a forgotten manual export, while production labels
still require the deliberate provenance decision. Export failures warn, never
crash a completed wave.

**Analysis DONE (2026-07-17):** two independent analysts (Claude Fable + Codex
`gpt-5.6-sol`) reviewed all 195 ballots — reports at
`research/v7_wave_analysis_{fable,codex}_2026-07-17.md` (shared goals doc:
`research/v7_wave_analysis_context_2026-07-17.md`). **Convergent verdict: do NOT
bless v7 yet — the binding constraint is option-menu expressibility, not evidence
or panel quality.** ~76–82% of the 74 NONE ballots are "no exact option offered"
(often all 3 seats naming the same missing set, e.g. `1b90f03b` = "all edges
except e1"); only 11 are genuine reject-alls (Berlin underpasses `422d5d7b`/
`d4d2e782`, Geneva cycleway `e4746a04`, HK `4eed5e80`). The physical/coincidence
enrichment works in the hypothesized direction (control reject rate
enriched/no_coinc 53% → no_phys 33% → minimal 13%, all flips toward over-merge
as context is removed; physical is the dominant factor) but is NOT yet validated
against human truth. Prioritized next steps (both analysts agree): (1) exact-pair
option generation + `none_reason` enum; (2) rubric fixes — anchor-vs-clip,
duplicate-vs-split-carriageway/MI-2-vs-MI-4 precedence, unknown-physical default;
(3) diff-highlighted option images (17 groups called them pixel-identical);
(4) suspected option-generator bugs `750ae089`/`e085519d` (conf-0.99 edges absent
from every option); (5) human-adjudicate the 5 controls (esp. `92c0997f` —
possible over-split/false-edge, and `fb8f359f`), then a small confirmation wave
measuring agreement with truth, not NONE rate. Panel composition is fine — no
seat change warranted. THEN build a deduplicated v7-only 50-group manual-review
pack. Full state, commands, manifest hash, and the post-vote analysis checklist
are in
[`research/physical_feature_experiment_2026-07-15.md`](research/physical_feature_experiment_2026-07-15.md#operational-handoff-targeted-v7-stitching-wave).

To re-run a similar wave (needs `META_API_KEY` in the environment and
`~/.opencode/bin` on `PATH`; smoke-test Muse before spending Claude quota):

```bash
set -a; . ./.env; set +a
PATH=/home/brad/.opencode/bin:$PATH UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/run_physical_stitch_wave.py \
  data/agents/stitching/batches/physical_context_v7_20260715_manifest.json \
  --group-workers 2
```

`--group-workers N` (PR #441) runs N packs concurrently via an order-preserving
dispatcher (~Nx wall-clock). Provider concurrency is N per seat, so
quota/rate-limit halts are more likely at higher N — the wave halts safely and
resumes. To pause: send ONE SIGINT/SIGTERM to the runner (`pkill -INT -f
run_physical_stitch_wave` is fine — a duplicate signal within 2s is debounced);
it finishes in-flight packs, flushes, prints a pause message, and exits 130. A
deliberate second signal after 2s force-aborts, but the process still waits for
in-flight provider invocations (worker threads blocked in `subprocess.run`, up
to `--timeout`, default 600s) before exiting — do NOT escalate to SIGKILL while
it waits: that reparents the provider subprocesses and they keep spending quota
on discarded ballots (observed 2026-07-16). If you do SIGKILL, clean up with
`pkill -f 'opencode run|codex exec|claude -p|codex-linux'`.

Note (2026-07-16, #440): the manifest contract lives in
`agent_labeling/wave_manifest.py`. The production manifest file is untouched;
`--validate-only` warns that the legacy manifest predates the embedded integrity
digest — that warning is expected. New manifests get an embedded
`manifest_sha256` automatically.

**Sequencing guard (still active):** the separate Codex consistency diagnostic
was runtime-bound: its sealed holdout had to be analyzed with the current
`stitch_diagnostic.py`/`stitch_runner.py` bytes after the fix hypothesis was
frozen, but before #446 or the diagnostic-analyzer fix landed. **DONE 2026-07-17**
— pre-registered + unsealed + scored (see resume-checklist step 3 and
`research/codex_deep_v1_{fix_hypothesis,holdout_result}_2026-07-17.md`). See also
the tracked design [`research/codex_stitch_diagnostic.md`](research/codex_stitch_diagnostic.md)
and the local, non-exportable readouts at
`data/agents/stitching/diagnostics/codex_deep_v1/{development,holdout}/`. **PR
#446 is now fully ungated** — next step is rebase + green CI + squash-merge.

### FOLLOW-UPS: from the 2026-07-16 external wave audit (separate agent)

Triage of an independent agent's findings on the v7 wave + same-day changes.
The audit also confirmed the v7 archive clean: 26/26 human labels, 78
effective ballots, no roster/edge-set/option/evidence-link/consensus
mismatches.

**Status 2026-07-17: all 8 findings implemented.** #441-#445 are merged.
#447 (isolated Codex consistency diagnostic) is also merged and its 260/260
calls are complete. #446 remains open, unreviewed; with the v7 wave now
complete it is gated only on the diagnostic holdout read described above.

**Resume checklist, in order:**

1. ~~Resume the v7 wave and consolidate it without changing the current
   rubric/runtime.~~ **DONE 2026-07-17** — 65/65 voted; provenance archived to
   `labels/votes/`; the 6 auto-accept label mints are deferred (see the "OTHER
   PIECE" note in the wave section above).
2. ~~High-effort wave analysis over ballots, feedback, and the five 2x2
   controls.~~ **DONE 2026-07-17** — see the "Analysis DONE" block in the wave
   section (`research/v7_wave_analysis_{fable,codex}_2026-07-17.md`).
3. ~~Freeze the narrow diagnostic fix hypothesis, create the marker, and analyze
   the sealed holdout using the current #447 runtime.~~ **DONE 2026-07-17.**
   Pre-registered `research/codex_deep_v1_fix_hypothesis_2026-07-17.md` (fix_id
   `codex_deep_v1__cycleway_gate_and_none_accounting__2026-07-17`), created the
   marker, unsealed + scored: **7/7 blind predictions confirmed; both fixes
   corroborated out-of-sample** — result in
   `research/codex_deep_v1_holdout_result_2026-07-17.md`. P7 (`18ef284e`
   factorial invariance, informed) falsified — an anchor-exactness phenomenon in
   the out-of-scope exact-pair track, touches neither fix. **This lifts #446's
   last gate.**
4. **NEXT — #446:** adversarially reviewed 2026-07-17 (no correctness issues; see
   the review note in the audit-followups status). Rebase onto current `main`
   (6 behind; auto-merges clean with the wave auto-export change — `git
   merge-tree` shows 0 conflict markers), confirm green CI, squash-merge. Starts
   provenance era `2026-07-16.1`.
5. In a new era, implement the two corroborated diagnostic fixes as SEPARATE PRs
   (per the diagnostic design's attributability rule):
   - **Fix B** — analyzer `NONE`-expressibility accounting (deterministic
     reporting bug; verify the dev 3/3→0/3 flip, all else unchanged).
   - **Fix A** — truth-backed cycleway/separated-infrastructure uncertainty gate
     (rubric = new provenance era); add `7175635e` as a non-regression fixture
     (must NOT flip the correct unanimous cycleway merge to NONE).
   Then regenerate packs so #443's exact-pair overlays are present, and run the
   fresh Claude/Codex/Muse panel. NOTE the holdout's draw-level near-misses
   (`9f56d71d`, `35329743`, `92c0997f`) suggest a FUTURE, separately-registered
   extension of Fix A to layer/level identity — not part of this fix.
6. Once v7 is validated by the analysis, mint the 6 deferred auto-accept
   stitching labels (the "OTHER PIECE" command above).
7. Clean up merged agent worktrees/branches under `.claude/worktrees/`.

Note: the holdout's out-of-scope P9 (audit menu-gap 14/21 = 66.7%, double dev)
independently strengthens the exact-pair option-generation plan (see "PLAN:
Exact-pair option generation" below) with fresh out-of-sample support.

Finding-by-finding disposition:

- **Merged in PR #441**: `--group-workers` originally regrouped the
  schedule by batch dir, collapsing the factorial counterbalancing rotation.
  Reworked to an order-preserving dispatcher (workers claim the
  lowest-index row whose dir is free); rotation preserved within a ~3-row
  window.
- **Merged in PR #442**: monitoring counted NONE as abstention (excluded
  19/78 valid, decisive v7 ballots from dissent/confidence stats) — NONE is
  now a first-class reject-all verdict, `n_ballots == n_valid + n_none +
  n_abstain`. Same PR: overlapping vertical-level ranges double-counted
  duration in the experimental physical PROFILES path — now unioned via
  `interval_union_length` (evidence-summary path was already fixed in #436).
  Both were analysis-blocking for the post-wave 2x2 read; done before resume.
- **Merged in PR #443** (`feat/option-image-pair-rendering`): option images now
  draw green tie lines per exact pair edge (endpoints at aligned-span
  midpoints), so options differing only in pair structure render
  differently; prompt documents the semantics; legacy edge-less options fall
  back to member-only rendering. Wave packs are immutable, so the running
  wave is unaffected either way.
- **Merged in PR #444** (`fix/wave-manifest-semantic-checks`):
  `WaveManifest.load_validated` now cross-checks each schedule row's
  `dataset_id`/`variant`/group roster against the batch dir's `batch.json`
  (disagreement fatal; missing batch-side fields warn once per dir; legacy
  manifests still load).
- **Merged in PR #445** (`fix/resolver-audit-metadata-concat`): per-dataset
  `build_audit`/`build_stats` now survive multi-dataset `pd.concat`
  (dataset-keyed reattachment), persist to parquet via pyarrow schema
  metadata, and a compact `training_data_audit` summary lands in the model
  artifact. No `RESOLVER_FEATURE_VERSION` bump (metadata only).
- **Merged in PR #447** (`agent/codex-stitch-diagnostic`): isolated,
  non-exportable Codex-only repeated draws over all 65 immutable packs (three
  canonical draws plus one audit each). All 260 calls completed without error.
  Development: 27/44 unanimous, 15/44 strict majority, 2/44 split; only three
  human labels were available, with one unanimous truth-backed miss. The audit
  hypothesized `no_exact_option` on 14/44 packs. These are diagnostic findings,
  never panel consensus. The 21-pack / 15-group holdout remains sealed.
- **Open PR #446, DO-NOT-MERGE gated** (`era/seat-retries-provenance-binding`,
  next provenance era): seat-level retries (failed seats re-run individually,
  ballot-of-record = first valid draw, `attempt` counter in provenance;
  panel-era changes still nuke the whole group via signature mismatch) +
  invocation signature now binds `opencode.json` content hash,
  `PACK_FEEDBACK_INSTRUCTION` hash, per-provider effort→CLI translation, and
  injected `OPENCODE_CONFIG_CONTENT` hash. Export fail-closed gates needed no
  change for mixed-attempt groups (regression test added).

### NEXT: Panel and agent-instruction improvements (post-wave sequencing)

Decisions locked 2026-07-16: keep the lean Claude/Codex/Muse trio (no Gemini
or Kimi seat — revisit only when a materially better model generation ships);
keep unanimous+gates as the only label-minting rule (v7 human eval: majority
was exact on 3/12 vs unanimous 8/11, auto-accept 4/4).

Sequencing, in order:

1. Resume and complete the paused v7 physical wave (above) on the current
   rubric — do NOT edit `MATCHING_RUBRIC_VERSION` content before it runs, or
   the immutable packs become mixed-era.
2. High-effort (fable) analysis pass over the wave's ballots + `pack_feedback`
   + the 2x2 contrasts: physical-feature go/no-go, and draft rubric-v2 wording
   (frontage/service-road identity and vertically-layered roads are the
   dominant ambiguity themes MI-1..6 doesn't address explicitly). Rubric v2 =
   new provenance era.
3. Freeze the Codex diagnostic fix hypothesis and analyze its sealed holdout
   under the current #447 runtime before #446 changes `stitch_runner.py`.
4. Review/merge #446, then implement the analyzer and truth-backed uncertainty
   fixes and regenerate future packs before the fresh heterogeneous panel.
5. Buildable afterward (rubric-independent, from
   `research/v7_reasoning_analysis_2026-07-15.md` follow-ups): structured
   `none_reason` enum (`all_edges_no_match` / `no_exact_option` /
   `insufficient_evidence`) in the vote schema; "dense assignment minus small
   bad subset" option generation (biggest expressibility lever — menus had
   member coverage 1.000 but boundary precision 0.967); precomputed
   coverage/interval-partition + topology table in evidence packs; optional
   exact-pair adjudication mode after membership review in the UI.

### PLAN: Exact-pair option generation (the v7 wave's #1 expressibility lever)

Both v7 analysts (`research/v7_wave_analysis_{fable,codex}_2026-07-17.md`)
independently found the **binding constraint is the option menu, not evidence or
panel quality**: ~76–82% of the 74 NONE ballots are "no exact option offered",
often with all three seats naming the *same* missing set (e.g. `1b90f03b` = "all
edges except e1") that just isn't on the menu. Fix the generator, not the rubric,
first.

**Current generator** (source of truth for the fix):
`matching/alternatives.py::generate_top_k_alternatives` (per-target confidence-
ranked whole-group assignments; top-K) + two always-appended seeds in
`_seed_alternatives` (full candidate set + optimizer-selected set) →
`matching/stitch_options.py::build_stitch_options` builds the lettered menu. The
gap: "optimizer set minus exactly edge eᵢ" is never emitted deliberately — it
only appears if it happens to rank in the top-K.

**Offline validation harness (no panel quota):**
`crosswalk agent stitch-expressibility <dataset> [-k N]`
(`agent_labeling/stitch_expressibility.py::measure_expressibility`) measures the
fraction of settled `labels/stitching/` labels whose exact edge set some
generated option can express. Run it before/after every generator change;
`1b90f03b` / `e085519d` are the regression fixtures.

**Baseline measured 2026-07-17** (before any generator change): pair
expressibility us_boston 97.3% (2 misses), us_seattle 93.3% (1 miss),
de_berlin/fi_helsinki 100%; SET-label expressibility us_seattle 90.9% (boundary
precision 0.980), de_berlin 50%. **Key nuance:** pair-expressibility is
near-ceiling because settled labels are SURVIVORSHIP-BIASED — a human could only
ever pick a set the menu offered, so the labels we have are the expressible ones.
Every residual miss is the "correct set = a subset of what's offered" pattern
(e.g. seattle set-label `d35b4619`: boundary precision 0.643, coverage 1.000 —
every option over-includes members), which is exactly Phase 1's target. But the
v7 wave's real 56–61-ballot gap lives in PANEL-DESIRED sets that were never
recorded as labels (they became NONE), which this harness cannot see. **So the
real before/after test for Phase 1 is the v7 panel's STATED sets** — build it by
mining the wave's NONE ballots' stated edge sets (a slice of Phase 3) FIRST, then
measure minus-flagged-edge seed coverage against that set. The existing harness
is the regression guard (must not drop); the mined-set probe is the real lever.

Phases, in order:

- **Phase 0 — confirm/fix the generator bugs.** Both analysts flagged
  `750ae089` and `e085519d`: conf-0.99, high-coverage edges absent from *every*
  option (also `3b876df0`: every trimmed option keeps a cross-layer edge). If
  enumeration/truncation is pruning strong edges, that's a straight bug and the
  cheapest win. Reproduce first, then fix.
- **Phase 1 — optimizer-minus-flagged-edge seeds (the MVP).** Extend
  `_seed_alternatives` (same bounded, deduped append site as the existing +2
  seeds) to also emit "optimizer set minus edge eᵢ" for a *targeted* set of
  flagged edges — NOT a blind power set (menu bloat + worsens the pixel-identical
  image problem). Flag signals are exactly what the wave showed matter: low
  per-edge confidence, layer/bridge/tunnel mismatch at the aligned subspan,
  class/role mismatch, sliver overlaps. Keep to single-edge (maybe 2-edge)
  removals so added-option count stays bounded like the current +2. Validate with
  `stitch-expressibility` across the labeled datasets; expect the rate to climb.
  Ship with diff-highlighted option images (see #443 pair rendering) or two
  options differing by one edge render identically.
- **Phase 2 — free-form / structured edge-set ballot.** Add an edge_set field to
  the vote schema for a "propose" verdict when no option is exact. Bigger change
  (schema + `panel_invocation`/provenance + export gates); do after Phase 1
  proves the cheap seeds close most of the gap. Keep NONE as decisive reject-all.
- **Phase 3 — seat-stated-set mining.** Parse the *prior wave's* `reasoning` for
  explicitly-stated edge sets and seed the confirmation wave's menu with them —
  converts unanimous-NONE-lost-to-menu-shape groups (`1b90f03b`, `e085519d`,
  `c8da4c08`, `17053a69`) into decisive letters.

Couple with the `none_reason` enum (item 5 above) — the two together are what a
future "does context help" claim must measure against human truth, not NONE rate.

### FOLLOW-UPS: from the 2026-07-16 architecture review batch

Merged that day: #434 (experiment-boundary guard + `physical_flag_domains`),
#435 (backfill-skip warnings), #436 (LR coverage union + evidence-gate fixes),
#437 (`fetch/physical_tags.py` consolidation), #438 (resolver
feature/table versioning), #439 (drift-mapper + `selected_edges` parser
dedup + crosswalk↔mbench divergence tests), #440 (`wave_manifest.py`).
Remaining items, roughly in value order:

- **Group lineage persistence** (the root fix for drift-mapping): emit
  `predecessor_group_ids` in the groups sidecar at write time (diff current
  membership against the prior sidecar for the same dataset), turning the
  overlap-inference mappers into lookups with inference as legacy fallback.
  #424's pruned-ownership snapshot proved the pattern. OPEN PRODUCT QUESTION:
  where the pipeline finds the predecessor sidecar (previous `data/output`
  artifact? explicit `--prior-sidecar`? store a lineage ledger next to
  labels?). The crosswalk↔mbench semantic divergences are now pinned by
  `tests/unit/test_mbench_drift_parity.py` in the meantime.
- **`stitch_runner` schedule API**: add `forget_groups(batch_dir, group_ids)`
  and a cross-batch schedule entry point so `run_physical_stitch_wave.py`
  stops duplicating the timeout-breaker constant and doing surgery on
  `votes.partial.csv`/`consensus.partial.csv` files the library owns.
- **Module splits when next touched** (not standalone churn): `stitch_runner.py`
  (~2.6k lines; transports vs orchestration), `stitch_export.py` (~2k; move
  panel-era registries to a `panel_routing.py`), `stitch_evidence.py` (rendering
  vs prompt authoring; `build_prompt` accretes a conditional block per evidence
  type).
- Smaller review findings: provider→(mode, transport) table encoded twice
  (`stitch_runner._delivery_mode_transport` vs
  `stitch_provenance.DELIVERY_TRANSPORTS_BY_MODE` — derive one from the other);
  Overture raw `road_flags`-vs-legacy-`road`-struct branch triplicated inside
  `fetch/overture.py` (one `_raw_road_flags()` accessor); first-row staleness
  heuristics in `backfill_overture_physical_lr` → replace with a version stamp;
  `backfill_physical_lr_from_source_tags` docstring overclaims ("unrelated LR
  attributes untouched" — configured flags rebuild the whole column); name the
  `is_bridge`/`is_tunnel`/`is_covered`/`is_indoor` flag vocabulary once;
  shared groups-sidecar schema module (producer `pipeline/runner.py`, consumer
  `resolver/extract.py`, validator `mbench/adapters/crosswalk.py` each encode
  it independently); eventual retirement note for the legacy `is_bridge`
  sidecar key.

### DONE: Stitch Ground Truth Store and Review Flow

**Priority:** HIGH
**Status:** Implemented

Stitching assignment truth is separate from pairwise identity labels and lives
under `labels/stitching/dataset={id}/data.csv`. The implemented workflow now
includes current group/candidate generation, evidence packs, multi-agent panel
voting and consensus, safe automatic export, drift-aware human-review queues,
reject-all labels, and stitch evaluation in core/mbench. Panel disagreement,
low confidence, unsafe class combinations, and size-gated cases route to human
review rather than silently becoming labels.

PR #426 made new judgments provenance-complete and fail-closed: the exact
upstream candidate universe, displayed option menu, pack inputs, batch source,
ballots, invocation, and consensus policy must agree before a panel label can
be exported. Ambiguous historical stitching labels remain quarantined in
[`research/stitching_deferred_audit.md`](research/stitching_deferred_audit.md).

**Location:** `src/crosswalk/labeling/stitching_store.py`,
`src/crosswalk/agent_labeling/`, `src/crosswalk/web/routes/stitching.py`,
`mbench/src/mbench/eval/stitch_metrics.py`

### HIGH: Expand Candidate-Backed Stitch Ground Truth

**Priority:** HIGH
**Status:** Ready to execute with the existing panel and human fallback; this
is a data-production/voting task, not new review infrastructure.

**Immediate goal:** Grow the adjudication-clean corpus to 200–300 diverse
groups, including at least 20 reliably mapped reject-all groups whose exact
current candidate universes are preserved. Do not use legacy empty labels or
the quarantined split/collision cases as clean negative truth.

**Work order:**

1. Regenerate fresh uncapped group sidecars and typed candidate parquets for
   Bogotá roads, Helsinki, and Singapore footpaths; generate Tunis. Profile and
   run Nairobi in bounded chunks rather than attempting another unbounded local
   pass.
2. Build stratified voting batches emphasizing M:N interchanges, 1:N/N:1
   corridors, short connectors, complementary spans, low-margin alternatives,
   parallel siblings, cross-mode/reject-all cases, bridge-backbone edges,
   datasets without useful names/classes, and optimizer/panel disagreements.
3. Run the existing flow: `crosswalk agent stitch-batch`, `crosswalk agent
   stitch-run`, and `crosswalk agent stitch-export`. Refresh the drift-aware
   stitching queue and adjudicate every panel disagreement, NONE verdict, and
   other `human_review` route with a human; panel votes are acquisition
   evidence, not the untouched test set.
4. Record wave manifests, candidate/evidence hashes, per-stratum counts,
   consensus/routing outcomes, and human overrides. Stop and regenerate rather
   than accepting stale or provenance-incomplete packs.
5. Before model tuning, freeze group-level train/tune/confirmation partitions
   and at least one dataset-held-out confirmation slice. Keep the deferred
   collision/split cases as a separate sensitivity analysis.
6. Compare production optimizer-only, learned resolver-only, and a hybrid
   learned rank/prune + deterministic-constraint approach. Report edge F1,
   group exactness, reject-all accuracy, REVIEW coverage, grouped OOF, LODO,
   and bootstrap intervals; do not promote unless the untouched confirmation
   set improves without a material candidate-recall regression.

**Supporting detail:**
[`research/resolver_candidate_persistence_2026-07-11.md`](research/resolver_candidate_persistence_2026-07-11.md),
[`research/learned_stitcher_round3.md`](research/learned_stitcher_round3.md), and
[`research/resolver_benchmark.md`](research/resolver_benchmark.md).

### Medium: Precision/Recall Tuning for Balanced Matching

**Status:** Analysis complete, no changes needed yet

The matcher is intentionally tuned for high recall (stitch recall=0.99, precision=0.75 on Boston) to support full topology construction and avoid accidentally omitting edges. Hootenanny is more balanced (P=0.82, R=0.85). For use cases requiring higher precision ("don't add wrong stuff"), these levers are available:

**Easy (config values in `config.py`, no code changes):**

| Lever | Current | Effect |
|-------|---------|--------|
| `bridge_min_confidence` | 0.5 | Floor for edges entering bipartite graph. Raise to drop weakest candidates before grouping. Highest single-lever impact. |
| `optimizer_review_threshold` | 0.5 | Groups with avg confidence below this → REVIEW. Raise to 0.6-0.7 to auto-match fewer borderline groups. |
| `scoring_match_threshold` | 0.5 | ML confidence cutoff for MATCH vs REVIEW. Blunt but effective. |

**Structural (geometry/topology params):**

| Lever | Current | Effect |
|-------|---------|--------|
| `MAX_ALIGNMENT_OVERLAP_M` | 5m | Competing targets on same ref: overlaps above this demote the weaker. Tighten to 3m. |
| `contiguity_tolerance` | 5m | Max endpoint gap for grouping. Tighten to 3m to split loose groups. |
| `DIVERGENCE_DISTANCE_MULTIPLIER` | 3.0 | Controls alignment truncation aggressiveness when roads diverge. |

**Architectural (would need design work):**

| Lever | Notes |
|-------|-------|
| Per-edge confidence within groups | Currently groups are accepted/rejected as a whole (avg confidence). Per-edge minimum would prune weakest edges while keeping strong ones — most targeted lever for stitch precision specifically. |
| Model selection override | Force geometry-only model (more conservative) vs full model (78 features, higher recall). Currently auto-selected based on name coverage. |
| CLI-exposed profiles | Ship a "balanced" profile with alternative thresholds (higher min_confidence, tighter overlap tolerance) so users can select use-case without code changes. |

**Measured baseline (us_boston_streets, 13 labeled groups):**

| Tool | Stitch P | Stitch R | Stitch F1 | Extra edges |
|------|----------|----------|-----------|-------------|
| Matcher | 0.7541 | 0.9923 | 0.8570 | 40 |
| Hootenanny | 0.8185 | 0.8457 | 0.8319 | 25 |

**Location:** `src/crosswalk/config.py` (thresholds), `src/crosswalk/matching/optimizer.py` (group resolution), `src/crosswalk/matching/ml.py` (model selection)

---

## Pipeline Architecture: Stitch → Merge

### Stitch Pipeline — Graph-Level Resolution (Planned)

**Priority:** Medium
**Status:** Core pipeline implemented (`crosswalk stitch`); graph-level resolution planned (see [docs/MATCHING_MERGING_RULES.md](docs/MATCHING_MERGING_RULES.md) Section 2)

`crosswalk stitch` currently runs: candidate generation → feature computation → ML scoring → M:N optimization. The following graph-level resolution features would run after scoring and before optimization:

- Junction zone detection (degree≠2 node proximity)
- Match role assignment (STRONG_EDGE, JUNCTION_ANCHOR, PARALLEL_COMPANION, AMBIGUOUS)
- Neighborhood consistency enforcement
- Conflict resolution for competing matches
- Confidence promotion/demotion based on graph context

**Location:** New module `src/crosswalk/stitching/` or extend `src/crosswalk/matching/`

### Implement Merging Stage

**Priority:** Medium
**Status:** Design outlined (see [docs/MATCHING_MERGING_RULES.md](docs/MATCHING_MERGING_RULES.md) Section 3)

Formalize the merge step as a distinct stage with explicit policy:
- Geometry integration policy (replace/average/keep)
- Attribute transfer rules and thresholds
- Net-new gating with junction zone awareness

Currently partially implemented in `src/crosswalk/integration/`.

### CLI: Separate Stitch and Merge Commands

**Priority:** Medium

- `crosswalk stitch` — Pair matching + M:N optimization (done)
- `matcher merge` — Network integration (currently `crosswalk analyze integrate`, rename/restructure)

### mbench: Merge Evaluation Mode

**Priority:** Low

Add a MERGE evaluation mode to mbench for assessing integration quality (geometry replacement, attribute transfer). The current STITCH mode already handles pair-level F1. Graph consistency metrics (junction coverage, gap rate, false net-new rate) can be added to the existing STITCH evaluation when graph-level resolution is implemented.

**Location:** `mbench/src/mbench/eval/`

### mbench: Add README

**Priority:** Low

mbench has no README. Add one documenting:
- What mbench does (benchmarking conflation tools)
- Available adapters (matcher, hootenanny)
- Current evaluation (pair-level F1) and planned MERGE mode
- CLI usage and examples

### Audit Existing Labels for Rule Change Impact

**Priority:** HIGH (blocks retraining with new philosophy)

Relaxing the ≥10m intersection rule means some existing `no_match` labels are now `match` under the new criteria. Need to identify and relabel affected pairs.

**Criteria for audit candidates** (pairs likely mislabeled under old rules):
- `no_match` pairs where aligned overlap is >0 but <15m
- `no_match` pairs where both endpoints are near intersection nodes (degree≠2)
- `no_match` pairs with high geometric similarity (buffer_iou, low hausdorff) but short overlap
- `no_match` pairs where the rejection reason was likely "intersection-only overlap"

**Approach:**
1. Query labeled pairs matching the above criteria
2. Surface them in the labeling UI for re-review
3. Relabel under the new philosophy: if same traveled way → match regardless of length
4. Retrain after relabeling

**Location:** Could extend `crosswalk agent batch` or add a `matcher labels audit` command

---

## Feature Ideas

### Medium: Pre-compute Context to Eliminate Dataset Requirements for Backfill

**Priority:** Medium

The sampled connector pattern (PR #192) can be extended so backfill eventually needs ZERO external datasets:

| Feature group | Currently requires | Pre-compute + store pattern |
|---|---|---|
| **Target topology** | ~~Full target dataset~~ | Sampled connectors (done in PR #192) |
| **Ref topology** | Full ref dataset (Overture) | Already has explicit connectors; could sample and store |
| **Crossing angles** | `ref_sibling_context_full` | Pre-compute per-pair crossing features and store |
| **Sibling detection** | `ref_sibling_context_full` + `target_sibling_context_full` | Same approach |

The pattern: during candidate generation (when full datasets are loaded), pre-compute all network-context-dependent features, store the results per-pair. Backfill then only needs stored pair data.

**Location:** `src/crosswalk/features/pipeline.py`, `src/crosswalk/labeling/data_store.py`

### Low: Spatially-Grounded Topology Features — Remaining Work

**Priority:** Low (core idea implemented, remaining items are incremental)

**Implemented (PR #195):** Interior connector features now provide spatially-grounded topology comparison via `interior_connector_jaccard`, `interior_junction_count_ref/target/delta`, and `interior_junction_position_sim` in `spatial_context.py`. These compare Overture connector sets along the aligned portion — the core "Connector set IOU" idea.

**Remaining problems:**
- Endpoint features (`min/max_endpoint_proximity_m`, `shared_endpoint_count`) still only measure target-to-target connectivity. For dense networks, these are uniformly high and non-discriminative.
- `degree_match_score` at `spatial_context.py:921` still compares degree numbers without spatial co-location.

**Still open:** Cross-network endpoint proximity — measure distance from target aligned endpoints to nearest **reference** connector (not target connector). Currently `min_endpoint_proximity_m` measures target-to-target only.

**Location:** `src/crosswalk/features/spatial_context.py`, `src/crosswalk/features/pipeline.py`

### Dual Carriageway / Centerline Handling

**Priority:** Medium
**Status:** Partially addressed via parallel sibling features

Parallel sibling features (`has_parallel_sibling_ref`, `parallel_fraction_ref`, `offset_vs_half_corridor_ratio`, `offset_over_expected_halfwidth`, `likely_representation_mismatch`) partially address dual carriageway detection. Remaining work:
- Detect split carriageway start/end points (Y-junction patterns)
- Pre-filter dual carriageway cases with specialized logic

---

## Integration

### Conflict Detector
- Detect duplicate matches in integration output (deferred)

---

## Agent Labeling

### Manually Curate Few-Shot Examples
- Current few-shot selection is automatic (random balanced sample from ground truth in other batches)
- Manually curate a set of high-quality examples covering key edge cases: split carriageways, parallel sidewalks, bike lanes, short overlaps, name mismatches
- Store in a dedicated directory (e.g. `data/agents/few_shot/`) so they're reused across batches

---

## Label Data Management

### Label Archive & History
- ~~Archive orphaned labels to `labels/archived/` instead of losing them~~ Done (`labels/archived/` exists with data)
- Provide recovery tooling to re-link archived labels

### Data Lineage
- Store data versions in model metadata
- Add `matcher model-info` command to show training data provenance

---

## Stitching Review UI

### Low: Show Per-Edge Detail in Stitching Review

**Problem:** The stitching review card only shows per-segment info (R1, R2, T1, T2) but not per-edge info. In an M:N group, some edges may be tiny junction slivers (2-3% overlap) while others are full matches. Without per-edge visibility the user can't see which ref↔target pairings exist, their individual confidences, or overlap fractions.

**Solution:** Add a per-edge section to the expandable details showing each ref↔target pairing with confidence and aligned fraction. E.g., "R2↔T1: 12% conf, 2% overlap" vs "R2↔T2: 99% conf, 99% overlap".

**Location:** `src/crosswalk/web/templates/stitching/group.html`, `src/crosswalk/web/routes/stitching.py`

---

## Other Ideas

### Adaptive Buffer Distance
- Pipeline default is 75m with relaxed heading (90°) and length ratio (20.0) filters
- Could auto-detect optimal buffer per dataset via alignment statistics on sample

### Active Learning
- Use model uncertainty to prioritize labeling candidates

### Bike/Sidewalk Networks
- May need separate model or geometry-only approach
- Bike lane vs cycleway classification issue (PR #111)

---

## References

- **Ruiz-Lendinez et al. (2021)** - "Road Network Conflation Using Semantics and Geometry"
- **Juhasz et al. (2012)** - "Road Network Conflation Based on Iterative Hausdorff Distance Calculation"
- **Volz et al. (2011)** - "Map Conflation Using MRFs"
- **Hootenanny** (open-source conflation tool) - Junction angle distribution algorithms
