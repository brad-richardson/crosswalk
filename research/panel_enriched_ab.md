# Enriched-pack A/B re-vote (enriched_ab1) — does #302 junction-sliver evidence move the panel?

**Date:** 2026-07-05
**Design:** strict A/B against `research/panel_diag_wave1.md` (PR #301). The SAME 32 groups
across 7 datasets were re-voted by the SAME panel v2 (claude `claude-opus-4-8` medium /
codex `gpt-5.5` low / agy `Gemini 3.5 Flash (Medium)`), from the SAME sidecars (the wave-1
worktree's copies, symlinked byte-identical), with the SAME option menus (k=8 + seeds,
verified: identifying (ref,target) pair-sets identical per option letter on spot-checked
groups), and the SAME prompts **except the #302 enrichment** (per-edge `overlap~Xm`,
BORDERLINE tags, #267 `deg/bridge/corr` structural pass-through, group structural summary,
junction zoom crops). `--pack-feedback` (#297) was ON both sides. **NO LABEL EXPORTS** —
all verdicts quarantined exactly like diag_wave1. Batch dirs: `enriched_ab1_*`
(session-worktree scratch).

The A-side (pre-enrichment) is diag_wave1's vote-level data, which survived in the wave-1
agent worktree (`data/agents/stitching/batches/diag_wave1_*`) — full vote-level A/B, no
granularity loss.

## Run incidents (affect interpretation; disclosed up front)

1. **First B-side run lost at group 3/32** — the host worktree was auto-cleaned mid-run.
   Re-run from a fresh worktree pinned at merge `9b3d362`. Silver lining: the two groups that
   completed twice reproduced **identically** at consensus level (00b5ea63 unanimous-A;
   2b99c180 majority-A with the same claude:F/codex:A/agy:A split) — an incidental B-side
   stability replication.
2. **agy capacity outage:** after 17 groups, agy began returning empty output for any
   non-trivial prompt (trivial probes still succeeded; even 8 KB text-only prompts failed).
   The last 15 groups (bogota, missoula, flathead) initially recorded agy ABSTAIN
   (`parse/validation: no JSON object found`). Per the resumable-driver rule, claude/codex
   continued; agy recovered after ~50 min and all 15 votes were backfilled with the same
   packs + prompts (same-session, ~40–70 min after their claude/codex votes). Consensus was
   recomputed from the completed panels.
3. **CONFIRMED pack-delivery bug found by the wave (fixed in this PR):** `_image_paths()` in
   `src/matcher/agent_labeling/stitch_runner.py` attaches only `overview.png` +
   `option_*.png` to codex's `-i` list — the #302 `zoom_*.png` junction crops were referenced
   in the prompt but **never delivered to codex** (codex reads no files itself; claude/agy
   read the paths). codex even self-reported it: "junction zooms were referenced but not
   shown inline" (9c1cd4f7). **All B-side codex votes are therefore text-enrichment-only**
   (overlap meters, tags, structural fields — no crops). The fix (+ regression test) is
   included here; it changes future runs, not this wave's data.

## Per-group A/B

Aconf/Bconf = consensus mean confidence. `*` = provider changed its vote.

| ds | group | A-cons | B-cons | A-ch | B-ch | Aconf | Bconf | claude | codex | agy |
|---|---|---|---|---|---|---|---|---|---|---|
| boston | 37a546e3 | unanimous | unanimous | A | A | 0.947 | 0.96 | A→A | A→A | A→A |
| bogota | 1ea5e608 | unanimous | unanimous | A | A | 0.94 | 0.947 | A→A | A→A | A→A |
| bogota | 3f112b8a | majority | majority | E | E | 0.865 | 0.81 | NONE→E* | E→E | E→B* |
| bogota | 5a03938b | unanimous | unanimous | A | A | 0.927 | 0.947 | A→A | A→A | A→A |
| bogota | aca7b5a7 | majority | unanimous | A | A | 0.96 | 0.93 | B→A* | A→A | A→A |
| bogota | aea47280 | majority | unanimous | B | B | 0.725 | 0.743 | B→B | NONE→B* | B→B |
| berlin | 1dfc9b52 | majority | none | F | G | 0.685 | 0.52 | F→G* | F→F | G→NONE* |
| berlin | 2b98a74f | unanimous | unanimous | A | A | 0.83 | 0.897 | A→A | A→A | A→A |
| berlin | 5503832a | unanimous | unanimous | B | B | 0.803 | 0.73 | B→B | B→B | B→B |
| berlin | 9c1cd4f7 | unanimous | unanimous | F | F | 0.803 | 0.74 | F→F | F→F | F→F |
| berlin | e1f4821a | majority | majority | A | B | 0.9 | 0.935 | A→A | B→B | A→B* |
| seattle | 00b5ea63 | unanimous | unanimous | A | A | 0.943 | 0.943 | A→A | A→A | A→A |
| seattle | 2b99c180 | unanimous | majority | G | A | 0.723 | 0.86 | G→F* | G→A* | G→A* |
| seattle | 46e57794 | none | none | G | C | 0.62 | 0.5 | G→C* | C→A* | J→NONE* |
| seattle | 670e939f | none | majority | D | J | 0.72 | 0.885 | D→A* | A→J* | J→J |
| seattle | 72699b72 | unanimous | unanimous | A | A | 0.92 | 0.87 | A→A | A→A | A→A |
| seattle | e919f4ab | unanimous | unanimous | A | A | 0.857 | 0.833 | A→A | A→A | A→A |
| tunis | 36f5f174 | none | **unanimous** | B | A | 0.5 | 0.803 | B→A* | A→A | I→A* |
| tunis | 506f7247 | unanimous | majority | D | D | 0.823 | 0.775 | D→D | D→G* | D→D |
| tunis | d54a5a8b | unanimous | unanimous | A | A | 0.947 | 0.953 | A→A | A→A | A→A |
| tunis | e6e0f483 | none | majority | A | B | 0.5 | 0.61 | A→B* | B→B | E→A* |
| tunis | ebb793ed | unanimous | unanimous | A | A | 0.947 | 0.943 | A→A | A→A | A→A |
| missoula | 0acf5ecd | unanimous | majority | E | E | 0.783 | 0.785 | E→E | E→B* | E→E |
| missoula | 2641fa35 | majority | unanimous | E | E | 0.685 | 0.73 | E→E | E→E | I→E* |
| missoula | 66a35845 | unanimous | unanimous | A | A | 0.95 | 0.957 | A→A | A→A | A→A |
| missoula | 786272f4 | majority | unanimous | G | E | 0.73 | 0.763 | G→E* | G→E* | E→E |
| missoula | b4c50f8c | unanimous | majority | A | A | 0.77 | 0.73 | A→A | A→A | A→C* |
| flathead | 51a42675 | unanimous | unanimous | A | A | 0.91 | 0.857 | A→A | A→A | A→A |
| flathead | 64f2d8d4 | majority | unanimous | B | A | 0.64 | 0.85 | B→A* | B→A* | A→A |
| flathead | bdb2b55e | unanimous | unanimous | A | A | 0.94 | 0.947 | A→A | A→A | A→A |
| flathead | f130d7ec | unanimous | unanimous | A | A | 0.893 | 0.883 | A→A | A→A | A→A |
| flathead | f3d0de6f | unanimous | unanimous | A | A | 0.873 | 0.86 | A→A | A→A | A→A |

## Aggregate A/B

| Metric | A (pre-#302) | B (enriched) | Delta |
|---|---|---|---|
| Unanimous | 20/32 (62%) | 22/32 (69%) | **+2** |
| Majority | 8 | 8 | 0 |
| No consensus | 4 | 2 | **−2** |
| Auto-accept routing | 20 | 22 | +2 |
| NONE votes | 2 (claude, codex — both bogota) | 2 (agy — 1dfc9b52, 46e57794) | 0, but different class |
| Provider vote flips | — | 27/96 (claude 10 / codex 8 / agy 9) | — |
| Median latency claude/codex/agy | 35.5 / 14.6 / 24.5 s | 36.9 / 15.9 / 31.9 s | ~flat (+30% agy) |

**Flip direction (the headline behavioral effect):** of the 23 flips between two lettered
options, **16 moved to a LARGER edge set vs 3 smaller** (4 same-size swaps; 4 more involved
NONE/prior-NONE). 10 flips landed ON the optimizer option vs 4 moving off it. The overlap
meters legitimize mid-size overlaps that providers previously discounted as possible
slivers — quantified evidence pulls the panel toward inclusion-heavier, optimizer-shaped
selections. Where wave-1 dissent was *pruning* dissent (Berlin M:N), nothing flipped; where
it was *coverage* hesitancy (sidewalk corridors, no-name roads), votes consolidated upward.

## The three named groups

### 46e57794 (Seattle 3-way split; disputed R3→T8/T4 now BORDERLINE + zoom crops)
Still `none` → human review (fail-safe holds), but the *axis of disagreement moved exactly
where the enrichment pointed*. A: claude:G(=settled exact)/codex:C/agy:J. B:
claude:C/codex:A/agy:NONE. All three now explicitly agree the tagged edges are junction
artifacts — claude: "the zooms show R3 simply ending where the red segment begins…
artifacts to exclude"; codex: "its only real weakness is the R3→T8 borderline
junction-kiss"; agy refuses A/J *because* they "incorrectly include borderline
junction-kisses… ~9m of overlap". The residual dispute is now the untagged mid-fraction
straddle edges (R1→T7/R1→T15) — for which claude and codex both request NEW zooms. The
enrichment settled the tagged question and exposed the next one. Note claude moved OFF the
exact settled set G to C (settled+1 extra, IoU 0.94): tags resolved its doubt on the kisses
but the crops don't cover the straddle edges it now includes.

### 670e939f (junction-sliver control: correctly ZERO-tagged, no crops)
The over-steering control shows **no tag-chasing** (there were no tags to chase) but a real
**number-anchoring** effect: A was a 3-way split (D/A/J); B is majority-J (A/J/J,
conf 0.885) — the whole-group 6-edge superset. codex and agy both cite the same new number
("the small R2→T1 overlap", 11.6 m) as evidence the edge is a *real split-overlap*, while
claude cites it as a *junction-kiss to exclude*. The bare meters, without a tag or crop,
consolidated 2/3 providers toward completeness. vs settled (B, 5 edges): IoU 0.80 (A-side
consensus D) → 0.83 (B-side J) — marginal, still wrong, still routed to review. Verdict:
no over-steering from tags; mild over-inclusion pull from untagged quantification.

### 37a546e3 (Boston unanimous-vs-settled one-edge divergence)
**The panel doubles down, now with the number in hand.** Unanimous A (optimizer 10-edge set)
on both sides; confidence rose 0.947 → 0.96. All three B-side rationales cite the
enrichment's measurement for the settled label's 11th edge: "a spurious R9→T1 cross-road
sliver (9.3m, East Cottage→Norfolk)" (claude), "tiny R9→T1 junction overlap… corner/sliver
artifact" (codex), "correctly excludes the R9→T1 intersection sliver" (agy). This is now a
5-run-stable, quantitatively argued unanimous disagreement with the settled label on one
dual-target boundary edge — the strongest evidence yet that either the settled label's 11th
edge is wrong, or the rubric must explicitly say dual-target boundary refs keep both edges.
Recommend a human re-adjudication of this one settled label rather than more panel runs.

## Berlin — and the DEBUT VERDICT

Per-group: the three M:N pruning verdicts that were wave-1's headline (consensus picks
pruning 4–6 optimizer edges) reproduced **exactly** — 2b98a74f unanimous A, 5503832a
unanimous B, 9c1cd4f7 unanimous F, same letters, no optimizer-pull (choices are unchanged
non-optimizer sets). The two review-tier groups stayed review-tier: e1f4821a majority
flipped choice A→B (agy joined codex; claude now minority — a 1-edge-smaller reading);
1dfc9b52 degraded majority-F → none (claude F→G same-size swap of the contested 8th edge,
agy → NONE citing the R8 corner split across T5/T7). Both were and remain human-review.

**BERLIN DEBUT VERDICT: the enrichment is NEUTRAL-to-POSITIVE on Berlin — precondition 3
of the diag_wave1 go/no-go is shipped and does not hurt.** The optimizer-over-selection
signal (the thing the debut is for) is *unchanged and now quantitatively argued* — B-side
rationales on the unanimous groups cite overlap meters for the pruned edges. No new
failure mode appeared on Berlin: zero tag-chasing, no optimizer-pull, the only cost is one
majority→none (which routes the same). The debut may proceed on the wave-1 conditions:
regenerate the (stale) Berlin sidecar on current main first — the fresh remote-box sidecar
was deliberately NOT used here to keep the A/B clean — and size the review queue for the
high alt-choice rate. Enriched packs should be used for the debut (see recommendation).

## Tunis (directionality gap #2 — expected "no change"; expectation REFUTED)

Both NONE-consensus groups **resolved**: 36f5f174 none → **unanimous A** (conf 0.803;
claude 0.5→0.75, and both dissenters converged on codex's A = the optimizer 15-edge set);
e6e0f483 none → majority B (claude A→B joined codex; agy still apart). The pre-registered
expectation was no change because the enrichment "does not address directionality" — but
the #267 corridor fields + overlap meters were passed through by #302, and they *partially
substitute* for directionality: claude's B-side rationales lean on corr/overlap to assign
lanes ("single continuous corridor (corr R0/T0)" reasoning appears across its flips).
Gap #2 is therefore **narrower than mapped but not closed**: codex still asks for
"per-segment direction or endpoint ordering" on e6e0f483, and the new 36f5f174 unanimity is
an auto-accept-shaped verdict on a geometry-only dataset — exactly the class the wave-1
quarantine exists for. Treat heading arrows as still-open, lower priority.

## Bogota (2 NONE-vote groups) + the cycleway policy gap

Both wave-1 NONE votes converted to substantive choices: claude NONE→E on 3f112b8a, codex
NONE→B on aea47280. Bogota went from 2/5 to **4/5 unanimous auto-accept-routed** groups.
This makes wave-1's gap #3 (cycleway↔vehicular policy is under-determined; the class gate
treats cycleway as NEUTRAL so these unanimities auto-accept) **more urgent, not less** —
the enrichment removed the hesitation that was previously keeping some of these matches in
review, without any policy decision having been made. Quarantine covers this wave; the
policy question needs an answer before any bike-network dataset is production-labeled.

## Complaint-class delta (the 383-vote historical sliver/junction complaint class)

Keyword incidence is nearly flat — sliver/junction-ambiguity language in 56/96 → 54/96
votes; zoom/crop requests 47/96 → 44/96; total feedback items 168 → 179 — but the
*content* changed category:

- Wave-1's direct complaints are GONE: no B-side vote says tags are missing ("SLIVER tags
  were referenced in guidance but not annotated" does not recur), and no vote asks for
  overlap magnitude — they now cite it.
- The surviving zoom requests RE-TARGET: providers ask for crops on specific untagged
  mid-fraction edges (46e57794 R1→T7/T15; 670e939f R4/T1, R2/T1; 36f5f174 R6→T8, R2→T5)
  — i.e. the crop *selector* (SLIVER/BORDERLINE only, cap 6) is too narrow, not the idea.
- codex's "zooms referenced but not shown inline" class disappears with the `_image_paths`
  fix in this PR.

So the enrichment converted "I can't see/measure it" complaints into "adjudicate the next
edge class" requests. Follow-up worth piloting: extend zoom-crop eligibility to edges with
`min(aln_frac) < ~0.35` (the straddle/boundary-crossing class), keeping the 6-crop cap.

## New failure mode watch

One candidate found, and it is NOT tag-chasing: **overlap-meter legitimization / coverage
pull.** Untagged, quantified overlaps read as endorsements; 16/23 letter-to-letter flips
grew the edge set and 10 landed on the optimizer option. Concretely: 2b99c180 lost its
(settled-closest, IoU 0.60) unanimous G to a majority on optimizer A (IoU 0.50); the
control 670e939f consolidated toward superset J. Both route to human review, so no label
risk this wave — but on *sidewalk-type corridor groups* the enrichment measurably trades
settled-agreement for optimizer-agreement. Seattle settled-IoU table:

| group | A consensus (IoU vs settled) | B consensus (IoU) |
|---|---|---|
| 00b5ea63 | unanimous A (1.00) | unanimous A (1.00) |
| e919f4ab | unanimous A (1.00) | unanimous A (1.00) |
| 72699b72 | unanimous A (0.67†) | unanimous A (0.67†) |
| 2b99c180 | unanimous G (0.60) | majority A (0.50) — review |
| 670e939f | none/D (0.80) | majority J (0.83) — review |
| 46e57794 | none/G (1.00 anchor) | none/C (0.94 anchor) — review |

† settled has 1 edge outside the group (wave-1 note). Every settled-exact unanimous stayed
settled-exact; every degradation is confined to review-routed groups.

## Recommendation: make enriched packs the panel default? **YES, with three riders.**

Evidence for: unanimity up (20→22) with NONE-consensus halved (4→2); zero
abstentions/parse failures attributable to the enrichment; complaint class converted from
missing-evidence to next-edge-adjudication; Berlin pruning signal intact and now
quantitatively argued; the two watch-groups behaved fail-safe; latency cost ~flat (+30%
agy). Evidence against: coverage pull on sidewalk corridor groups (all review-routed);
bogota unanimity now front-runs an unresolved modality policy.

Riders:
1. **Provenance bump required.** This is a panel-input composition change: if enriched
   packs become the default, the export labeler must bump (e.g. `panel_unanimous_v2` →
   `panel_unanimous_v3`) exactly as the v1→v2 panel bump did. **No default was flipped in
   this PR** — the runner/generator behavior is unchanged apart from the codex crop-delivery
   fix; flipping the default is a separate, deliberate change for the maintainer.
2. Ship the `_image_paths` codex fix (this PR) before any default flip, else the "same
   evidence" claim is false for one panel member.
3. Resolve the cycleway policy (wave-1 gap #3) before the first bike-network export under
   enriched packs, since the enrichment demonstrably increases cross-mode unanimity.

## Quota consumed

Successful substantive votes: **102** (B-wave 96 = 32/provider, + 6 from the killed first
seattle run). Overhead: ~60 near-instant empty agy attempts during its outage (in-run
retries + first backfill round) and ~25 trivial agy probes while polling for reset; claude
and codex never capped. Wall time: ~36 min for the 32-group main run + ~50 min agy outage
wait + 6 min backfill.

## Artifacts

- B-side batches + votes + consensus + pack_feedback: `data/agents/stitching/batches/enriched_ab1_*`
  (session-worktree scratch, mirrors diag_wave1 handling)
- A-side preserved at the wave-1 worktree `diag_wave1_*` batches
- Code: `_image_paths` zoom-crop delivery fix + regression test (this PR)
- No writes to `labels/`, `data/cache/stitch/`, or any live `data/output` sidecar
  (sidecars consumed via read-only symlinks to the wave-1 copies)
