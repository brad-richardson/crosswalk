# Astra / Fable / Muse stitching trial — September 9, 2026

**The panel is operational and useful, but this round does not validate unconditional two-of-three labeling.** All 30 retained ballots were valid. Eight of ten packs had unanimous choices; two had a 2–1 letter split. On one split, however, the two `NONE` responses expressed different judgments. Replaying the existing routing policy sends only four packs to automatic acceptance and six to human review. Improving the supervision contract and gates remains necessary even with these newer models.

The full repository assessment is [here](../repo_deep_dive_2026-09-09.md). The trial's [selection and input hashes](selection.json), [descriptive statistics](summary.json), [existing-policy replay](existing_policy_replay.json), individual ballot JSONs, and raw final responses are retained alongside this note.

**Protocol and limits.** The user requested Fable 5.1, Astra 6, and Muse Spark 1.3 Contributor. Requested IDs were `claude-fable-5-1`, `gpt-6-astra`, and `meta/muse-spark-1.3-contributor`, all configured for high reasoning effort. These are the requested provider model IDs, not separately verified immutable weight revisions.

Ten existing packs from six datasets were selected before viewing the new ballots: two human-reviewed Boston matches; three stored human rejections from Sydney, Berlin, and Hong Kong; three known difficult Helsinki/Geneva cases; and two previously accepted children of failed decomposed parents. Five packs have stored human exact-edge rows. Sydney's row is disputed by a later investigation, so only four are used as the uncontested human comparison. The two Boston positives have optimizer-visible historical prompts and were not collected in the newer deanchored review session. This is a deliberately selected diagnostic, not a random accuracy sample or an independent new gold set.

Original prompts and images were reused: 134,475 prompt characters and 77 images across the ten packs, each delivered through the existing provider adapter. Claude accesses images through `Read`; Astra and Muse receive native CLI attachments. All ballots passed the runner's evidence preflight, and all three share identical evidence and image-set hashes within each group. Delivery preflight confirms the available/attached assets, not that every model actually inspected every image. Claude and Codex ran from neutral directories; Muse used the tool-less `vote` agent. Human expected answers and other voters' answers were not passed to any model. Eight packs have three historical ballots with matching evidence-pack hashes; the two Boston packs have no matching rows in the committed vote archive.

**Observed answers.** Letters refer to the exact archived menus, not to a common answer across cases.

| Pack | Fable 5.1 | Astra 6 | Muse 1.3 Contributor | Interpretation |
|---|---|---|---|---|
| Boston `72063362` | A | A | A | Both consecutive reference pieces match one target; all match the human answer |
| Boston `f69a827e` | A | A | A | Another clean N:1 split; all match the human answer |
| Sydney `66e22055` | B | B | B | Keep surface-road edge, reject tunnel edge; stored human rejection is disputed |
| Berlin `d4d2e782` | NONE | NONE | NONE | All explicitly reject underground-footway/surface-road pairs, matching the human answer |
| Hong Kong `4eed5e80` | NONE | NONE | NONE | All explicitly reject tunnel/surface-road pairs, matching the human answer |
| Helsinki `92c0997f` | NONE: missing exact option | NONE: insufficient evidence | A: all 18 edges | Three substantively different judgments despite a 2–1 `NONE` tally |
| Helsinki `7175635e` | A: three edges | D: two edges | A: three edges | Disagreement concerns one bridge-level conflict |
| Geneva `a451bf05` | P | P | P | Keep 14 road/route edges, reject three adjacent-footway edges; no fresh human adjudication |
| Sydney child `3fd8483c__p07ce5199df` | A | A | A | Repeats historical agreement on a child whose parent failed export |
| Helsinki child `c8b85bbc__p07741daa80` | A | A | A | Repeats historical agreement on a 40-edge corridor child whose parent failed export |

Each model matches all four uncontested stored human answers. Each disagrees with the Sydney stored row. Reporting this as “80% accuracy” would conceal the disputed reference answer and the purposive selection; reporting the four remaining matches as a reliable “100%” estimate would be equally unjustified. Neither split case received a fresh independent human judgment, so the quality of the newly admitted 2–1 tier remains unmeasured.

**What the difficult cases establish.**

In Helsinki `92c0997f`, Fable returns a proposed 13-edge set outside the menu, Astra declines to specify an exact set, and Muse keeps all 18 edges, treating layer differences as representation differences. A majority `NONE` is neither an empty selected set nor agreement on Fable's proposed set. The production policy already keeps `NONE` in review; retain that protection while recovering properly scoped partial observations. Fable's new proposed set is useful evidence to inspect, not newly established truth.

In Helsinki `7175635e`, Astra rejects the middle edge because the reference enters a layer-1 bridge while the target is explicitly layer 0. Fable and Muse accept it based on corridor continuity and coverage partitioning. The historical panel also split A/A/D on the identical pack, but the D voter was Opus 4.8; the new dissenter is Astra. The unresolved evidence survived model upgrades, and fixed reliability weights keyed only to `claude` or `codex` would hide the model change.

Sydney `66e22055` exposes a label-maintenance problem. The human row says the cycle facility is separated from the road. The later [July 18 investigation](../v9_rerun_adjudication.md), performed by agents using geometry and external map evidence, instead argues that it is an on-road cycle lane and supports B at 0.75 confidence. The new panel and historical panel both choose B. This does not settle the physical fact. It shows why a disputed label needs an explicit adjudication state instead of silently remaining authoritative or being overwritten by consensus.

Geneva improves from the historical O/P/P to P/P/P on identical evidence. The difference is three short corridor-continuation edges. Fable still reports only 0.5 confidence and explicitly questions whether those pieces should be treated as endpoint clips. New unanimity is a useful observation; independent truth and calibrated confidence are still absent.

**A model upgrade alone leaves much of the manual queue intact.** The current `compute_consensus` policy produces four `auto_accept` decisions: the two Boston cases and the two child cases. Berlin and Hong Kong remain in review because unanimous `NONE` never auto-exports. Both Helsinki splits remain in review. Sydney and Geneva are demoted by the class-mismatch gate; each also has a minimum self-reported confidence below the current 0.75 floor. The two child decisions still do not establish that their entire parents can be exported.

The class gate compares road/cycleway/pedestrian categories without receiving source-kind or same-pavement evidence. Geneva is explicitly configured as `target_kind: route_network`, and the canonical matching rubric allows a route that follows a road to match that road. Nevertheless, the gate routes the unanimously selected road-to-cycle-route edges to review. This is a deliberate older conservative policy, not an accidental failure to execute. It should be reconciled with the newer rubric through source-aware validation and audits; do not simply remove protection against genuinely separate facilities.

**Observed latency also favors smaller tasks.** Median elapsed time per retained ballot, including CLI overhead, was 8.64 seconds for Astra, 24.88 seconds for Fable, and 41.29 seconds for Muse. The largest Geneva menu had 16 options and 17 images: Astra took 20.42 seconds, Muse 351.04 seconds, and Fable 426.55 seconds. These are single-run harness timings, not model-only benchmarks; the image-delivery mechanisms differ. A compact exact-edge question and consistently native image delivery are useful experiments before scaling this workflow. Actual token usage and billed cost were not captured.

**Recommended next experiment.** Keep this panel available, but change the unit of supervision to explicit edge decisions with unknowns, reason codes, parent lineage, and source/evidence versions. Recover already accepted children and valid desired-edge proposals first. Then run a broader, independently audited comparison of gold-only training, unanimous weak labels, and weighted majority weak labels. Keep the existing optimizer as the comparator. The next human work should adjudicate a small set of consequential facts and randomly audit accepted cases, rather than reconstruct entire groups.

**Reproduction and execution notes.** The [trial harness](../frontier_stitch_trial_2026-09-09.py) prepares the pinned selection and runs the probe or remaining cases through the existing invokers. It never exports labels or trains a production model. Three initial CLI calls completed but their responses were lost to a local capture bug; their error records and original manifest are retained in `bootstrap_harness_error`. The corrected run retained 30 ballots from 33 total provider invocations. The initial network approval rejection was resolved after inspecting the public-road payload; no new user approval was needed.

To recompute the statistics and existing-policy replay without any model calls, run from the repository root:

```bash
UV_CACHE_DIR=/tmp/crosswalk-uv-cache uv run --no-sync python research/frontier_stitch_trial_2026-09-09/summarize.py
```
