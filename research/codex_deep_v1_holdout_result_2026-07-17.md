# Holdout result — scoring the frozen codex_deep_v1 fix hypothesis

> **Scored 2026-07-17, after unsealing.** This document scores the pre-registered
> hypothesis `research/codex_deep_v1_fix_hypothesis_2026-07-17.md`
> (`fix_id: codex_deep_v1__cycleway_gate_and_none_accounting__2026-07-17`)
> against the unsealed 21-pack / 15-group holdout
> (`data/agents/stitching/diagnostics/codex_deep_v1/holdout/`). All numbers
> computed from `holdout/summary.json` (84/84 results present, 21/21 valid
> audits); dev comparisons from `development_summary.json` (44 packs). The
> pre-registration, marker, and diagnostic outputs were not modified. Scored
> honestly against the frozen falsifiers: falsified predictions are reported as
> findings, not explained away.

## Scoring table

Per the frozen scoring decision, **primary scoring is on the blind predictions
P1–P3, P5, P8–P10**; P4/P6/P7 are informed (one public v7 wave ballot per
holdout pack existed pre-freeze) and scored with that disclosure. P11 was a
low-confidence side-bet outside both lists.

| P | Scope | Predicted | Actual | Verdict |
|---|---|---|---|---|
| P1 stability replicates | blind (H) | 11–16 unanimous of 21; ≥19 strict-majority; ≤2 splits | 12 unanimous (57.1%); 19/21 ≥ majority; 2 splits | **CONFIRMED** |
| P2 confidence ≠ stability proxy | blind (H) | mean conf ≥0.90; unan-vs-non gap <0.05 | mean 0.971; unan 0.983 vs non-unan 0.956, gap 0.027 | **CONFIRMED** |
| P3 Fix-A theme generalizes | blind (H) | ≥2 of 15 groups, ≥2 datasets | ≥9 groups across 7 datasets (6 groups cycling/pedestrian-vs-roadway alone); all 5 named candidates hit | **CONFIRMED (strongly)** |
| P4 `dd106a0f` → NONE | informed (M-H) | modal NONE, not confident lettered retention | NONE 3/3 (0.98–0.99), audit NONE; tunnel-vs-surface named in every draw | **CONFIRMED** |
| P5 full `66e22055` signature rare | blind (M) | 0–2 packs with full signature | **0** at modal level (three near-misses at draw level, see below) | **CONFIRMED** |
| P6 `92c0997f` diverges | informed (H) | enriched NONE (T3/T7 conflation); other 3 variants A | enriched NONE 3/3 exactly as predicted; no_coincidence + minimal A 3/3; **no_physical went NONE 2/3 (insufficient_evidence), not A** | **CONFIRMED** (falsifier untriggered; one variant sub-prediction missed) |
| P7 `18ef284e` invariant NONE | informed (H) | modal NONE all 4 variants, ≥3/4 unanimous | **DIVERGENT**: enriched H 3/3 (0.97–0.99), no_coincidence H 2/3; no_physical + minimal NONE 2/3 | **FALSIFIED** |
| P8 Fix-B mechanics | blind (det.) | (a) 0 strict holdout labels; (b) empty-set labels report "absent" pre-fix; (c) corrected analyzer flips dev 3/3→0/3 only | (a) `human_labels_available: 0` ✓; (b) dev shows the bug live: 3/3 inexpressible incl. two packs where modal NONE == human label exactly; (c) pending implementation, deterministic | **CONFIRMED** (a, b; c is the post-fix check) |
| P9 menu-gap rate (monitor) | blind (M) | 7–14 of 21 | **14/21 (66.7%)**, top of band (dev 14/44 = 31.8%) | **CONFIRMED** — strengthens deferred exact-pair track (≥10 threshold met) |
| P10 audit-modal agreement | blind (M) | 65–90% of unique-modal packs | 16/19 = 84.2% (dev 32/42 = 76.2%) | **CONFIRMED** |
| P11 split location side-bet | side-bet (L) | if a split occurs, most likely `35329743` | splits at `729f879b` (London) and `9f56d71d` (Amsterdam); `35329743` was 2/3-NONE majority | **FALSIFIED** (missed side-bet) |

**Tally: 9 confirmed, 2 falsified (P7 informed, P11 low-confidence side-bet), 0 partial.
All 7 primary blind predictions confirmed.**

## Per-prediction detail

### P1 — CONFIRMED (blind)

`stability_counts: {stable: 12, majority: 7, split: 2}`. 12/21 = 57.1%
unanimous, inside the predicted 52–76% band (dev 61.4%); 19/21 at strict
majority or better (threshold exactly met); 2 three-way splits (≤2). Both
splits are genuine three-way (`distinct_edge_sets: 3`): `729f879b` drew
C/NONE/G, `9f56d71d` drew A/NONE/F.

### P2 — CONFIRMED (blind)

Mean canonical confidence across all 63 draws: **0.971** (≥0.90 ✓). Unanimous
packs 0.983 vs non-unanimous 0.956: **gap 0.027** (<0.05 ✓; falsifier at ≥0.10
nowhere near). Replicates dev (0.968 overall, gap 0.024). Confidence again
carries almost no information about draw stability — the two split packs
averaged 0.96–0.97.

### P3 — CONFIRMED, strongly (blind)

Predicted ≥2 of 15 groups across ≥2 datasets where the audit names
cycling/pedestrian-vs-roadway identity or physical/vertical separation as
decisive missing evidence. Actual, counting conservatively:

**Cycling/pedestrian-vs-roadway identity (6 groups, 5 datasets):**

- `e085519d` (fi_helsinki): "R7 is a separate parallel pedestrian facility, so
  R7->T4, R7->T7, and R7->T2 fail the same-physical-way role test… therefore no
  option is exact." Missing: "target access and travel-mode attributes;
  cross-section or close-up showing lateral separation."
- `b8b5da4a` (hk_hongkong): "Every edge from footway R4 or R6 must be excluded
  because coincident geometry and the shared street name do not overcom[e]…"
  Missing on all ten disputed edges: "Explicit target travel mode and access
  classification."
- `9f56d71d` (nl_amsterdam): "All R1 edges are excluded because R1 is a distinct
  footway representation, not the target cycleway." Missing: "authoritative
  facility separation and shared-use/access attributes."
- `a451bf05` (ch_grand_geneva): R10 "is the coincident footway representation";
  missing "facility-separation and bicycle-access attributes for R10",
  "cross-section or facility-separation evidence on the bridge."
- `17053a69` (us_philadelphia_sidewalks): "R1 is Waverly Street, a residential
  road coincident with R3's footway… Every offered option includes e2, so none
  is exact." Missing: "Explicit confirmation of whether R1 is
  pedestrian-accessible shared pavement."
- `dd106a0f` (ch_grand_geneva): tunnel vs surface cycle-lane (also vertical);
  missing "target elevation and tunnel rules along T7."

**Physical/vertical separation (3 more groups, +2 datasets):**

- `35329743` (au_sydney): layer-0 roadway vs stacked bridge; missing "Explicit
  roadway-level connectivity or elevation data confirming whether R1 and R5 are
  stacked facilities."
- `92c0997f` (fi_helsinki): "T7 is a layer-0 service road coincident with the
  layer-−1 tunneled tertiary T3"; missing "Target-side connectivity and
  road-role provenance", "Exact linear location of R3's tunnel portal."
- `4148382c` (de_berlin): "The aligned R6 span is layer −1 and tunnel while T2
  is layer 0, so coincident geometry represents a different physical way."

≥9 of 15 groups across 7 of 8 datasets, versus a ≥2-groups/≥2-datasets bar. All
five pre-named candidates (`a451bf05`, `dd106a0f`, `e085519d`, `17053a69`,
`b8b5da4a`) hit. The Fix-A ambiguity class is emphatically not Sydney-local —
it is the dominant audit theme in the holdout.

### P4 — CONFIRMED (informed)

`dd106a0f` went NONE 3/3 at 0.98–0.99, audit NONE at 0.98, and every draw named
the Fix-A distinction: "R2 is the grade-separated vehicular Tranchée couverte de
Vésenaz tunnel, whereas the targets are the surface Pénétrante Corsier cycleway;
their planimetric coincidence does not establish the same physical traveled
way." No confident lettered retention — the outcome the human empty-set label
is consistent with. Honest caveat: agreement is choice-level, not intent-level —
the audit's desired set retains 15 surface edges (`none_reason:
no_exact_option`, closest D), i.e. the seat wanted an off-menu nonempty set
while the human (deanchored, `set` semantics) selected nothing. The prediction
as frozen ("modal diagnostic result NONE, not a confident lettered retention")
is met; the pre-committed asymmetric branch was not needed.

### P5 — CONFIRMED at 0 (blind), with three near-misses worth recording

Full signature = audit names a decisive missing separation/identity fact AND
canonical draws go ≥2/3 for one lettered option at conf ≥0.90. Strict count:
**0 of 21**. The eight lettered-modal packs break down as: five where the audit
agrees (`91570f54`, `3f53c7e7`, `61a926e3`, `92c0997f` no_coincidence/minimal),
two where the audit's complaint is omitted junction anchors, not
separation/identity (`18ef284e` enriched/no_coincidence), and one
(`a451bf05`, modal B 2/3 at 0.88/0.99) where the audit names missing
facility-separation evidence but **B excludes the disputed footway edges** —
draws and audit agree on the identity call; the dispute is omitted valid road
edges (a menu gap, not an unsafe retention).

Per the frozen reading, P5=0 means Fix A's *full* confident-wrong signature
rests on dev's `66e22055` alone (lower urgency). But three draw-level
near-misses show the hazard is live out-of-sample, just not modal:

- `9f56d71d` (split): draws 1 and 3 chose lettered A (0.98) and F (0.97), both
  retaining the R1 footway→cycleway edges the audit excludes on
  facility-identity grounds; only draw 2 went NONE for exactly the Fix-A reason
  ("the R1–R4 coincidence do[es] not establish identity").
- `35329743` (majority): draw 1 chose A at 0.97 retaining e1/e2, the layer-0-vs-
  bridge edges draws 2–3 and the audit reject as "vertically distinct false
  match[es]."
- `92c0997f` no_coincidence + minimal: A 3/3 at 0.93–0.99 retaining the R3→T7
  service-road edges that the enriched audit calls false ("different role and
  vertical context") — confident identity assumption exactly when the evidence
  resolving separation was removed (same-pack audits rationalized along).

### P6 — CONFIRMED with one variant deviation (informed)

Factorial `92c0997f` diverged as predicted, and the enriched mechanism was
nailed: NONE 3/3, all three draws citing the frozen sentence's exact conflation
— "Every option includes e15 (R3->T7), but T7 is a layer-0 service road
coincident with the layer-−1 tunneled tertiary T3." no_coincidence (A 3/3) and
minimal (A 3/3) matched the prediction. **Deviation:** no_physical was
predicted A ≥2/3 but went NONE 2/3 — and produced the holdout's only
`insufficient_evidence` NONE ("With physical and layer evidence unknown,
geometry cannot distinguish a legitimate split-carriageway representation from
two distinct facilities"). Neither falsifier fired (variants did not share a
modal; divergence did not run the other way), so the verdict stands, but the
miss is instructive: it is the *coincidence overlay*, not the physical
attributes, that alerts the voter to the two-facility conflict — with
coincidence flagged but physical evidence absent, the voter correctly abstains
(the behavior Fix A mandates); with *both* removed (minimal), it confidently
assumes identity (the behavior Fix A exists to stop).

### P7 — FALSIFIED (informed): the mechanism

Predicted: modal NONE in all four `18ef284e` variants at conf ≥0.85, ≥3/4
unanimous, expressibility-flavored because "option H omits boundary anchors
e3/e15/e18." Actual: **divergent** —

| Variant | Draws | Modal |
|---|---|---|
| enriched | H 0.97, H 0.99, H 0.98 | **H, stable 3/3** |
| no_coincidence | H 0.98, H 0.97, NONE 0.99 | **H, 2/3** |
| no_physical | NONE 0.93, NONE 0.97, H 0.98 | NONE, 2/3 |
| minimal | NONE 0.90, NONE 0.99, H 0.94 | NONE, 2/3 |

Any lettered modal falsifies the prediction; two variants went lettered, one of
them unanimous.

What the prediction got right: the fault line. Every seat, every variant,
agrees on H's nine-edge backbone, agrees R8→T6 beats the "covered layer-0
service segment" R7→T6, and the entire dispute is whether the short boundary
anchors — exactly the predicted e3 (R6→T6), e15 (R11→T4), e18 (R9→T2) — are
required same-way segmentation anchors (⇒ H inexact ⇒ NONE) or "divergent
continuations rather than same-way anchors" (⇒ H exact). All four audit draws
sided with requiring the anchors and chose NONE (`no_exact_option`), at the
holdout's lowest audit confidences (0.72–0.88) — the audit content matched the
prediction.

What it got wrong: the claim that this dispute resolves invariantly to NONE.
Resolution tracks the physical-evidence axis: with physical/layer evidence
present (enriched, no_coincidence), canonical draws read the junction zooms as
showing the anchors to be "divergent continuations" and declare H exact at
0.97–0.99; with physical evidence removed (no_physical, minimal), draws demand
the anchors and go NONE. So `18ef284e` is not an easy invariant case but a
genuinely evidence-sensitive anchor dispute — the same *menu/anchor* mechanism
as dev's `7bac1f1d`, which the prediction explicitly said it would differ from.
Note also the within-variant instability (draw 3 defected in three of four
variants, at 0.94–0.99) and that both sides of the fault held conf ≥0.90 —
another P2-style datum that confidence does not signal contestedness.

Scope of the damage: P7 was an informed prediction about one group's factorial
behavior. Its falsification impugns neither frozen fix — it concerns
exactness-strictness on short junction anchors, which belongs to the explicitly
out-of-scope option-menu/exact-pair track (independently strengthened by P9).
It does show the analyst's model of *when* Codex tolerates anchor omission was
wrong, and it flags "H-style confident exactness under rich evidence, NONE
under poor evidence" as a pattern the future exact-pair work must handle.

### P8 — CONFIRMED on (a) and (b); (c) is the post-implementation check

(a) `human_labels_available: 0` in the holdout — `dd106a0f` is `set`-semantics
and excluded from the strict pair metric, as pre-verified. (b) The dev summary
shows the bug exactly as diagnosed: `human_menu_inexpressible: 3` of 3, while
`4eed5e80` and `d4d2e782` simultaneously have `modal_exact_human: true` with
modal NONE — the analyzer marks the human empty set "absent from menu" even
where the modal draw expressed it via NONE. (c) — corrected analyzer flips dev
to 0/3 with nothing else changing — is deterministic and falls due when Fix B
lands. **Retraction condition (a nonempty labeled set marked expressible via
NONE) cannot trigger on current data:** all three dev strict labels are
reject-all empty sets, and the fix's rule (`NONE` expresses only ∅) excludes it
by construction; verify on dev when implemented.

### P9 — CONFIRMED at the top of the band (blind, monitor-only)

14/21 packs (66.7%) have audit `none_reason: no_exact_option` with a
reconstructed `desired_edges` set absent from the menu — the very top of the
predicted 7–14, and **double** dev's 31.8% (14/44). The ≥10/21 "strengthens"
threshold is met: the deferred exact-pair/menu track now has out-of-sample
support at a higher rate than dev. Per the freeze, no menu change ships under
this fix_id regardless.

### P10 — CONFIRMED (blind)

16 of 19 unique-modal packs had audit agreement: **84.2%**, inside 65–90% (dev
76.2%). The three disagreements: `18ef284e` enriched + no_coincidence (the P7
anchor dispute) and `a451bf05` (audit wants B plus e1/e3/e15).

### P11 — FALSIFIED (low-confidence side-bet)

The two splits landed on `729f879b` (gb_london: C/NONE/G) and `9f56d71d`
(nl_amsterdam: A/NONE/F), not `35329743`, which resolved 2/3 NONE on
vertical-separation grounds with one lettered dissent. A missed L-confidence
side-bet, priced as such.

## Primary vs informed verdict

- **Primary (blind) scoring — P1, P2, P3, P5, P8, P9, P10: 7/7 CONFIRMED.**
  Every genuinely out-of-sample, pre-committed quantitative band was hit: draw
  stability (57.1% vs 52–76%), the confidence/stability decoupling (gap 0.027 vs
  <0.05), the cross-dataset generality of the separation/identity theme (≥9
  groups vs ≥2), the rarity of the full confident-wrong signature (0 vs 0–2),
  the Fix-B accounting mechanics, the menu-gap rate (14/21 vs 7–14), and
  audit-modal agreement (84.2% vs 65–90%).
- **Informed — P4, P6, P7: 2/3 confirmed, P7 falsified.** As disclosed, these
  carried wave contamination and were kept out of primary scoring; the one
  falsification is a real miss on factorial invariance, characterized above.
- P11 (side-bet, L): missed.

## Retraction-rule evaluation (verbatim conditions)

- **"Retract/rescope Fix A if P3 fails AND P5=0":** P5=0, but P3 passed at
  ~4.5× its threshold with all five named candidate groups hitting. Conjunction
  not met → **no retraction.** The P5=0 branch's frozen reading applies: the
  full confident-wrong *modal* signature rests on dev's truth-backed `66e22055`
  alone, so implementation urgency is moderate rather than critical — tempered
  by the three draw-level near-misses (`9f56d71d`, `35329743`, `92c0997f`
  minimal/no_coincidence) showing confident cross-mode/cross-level retention
  does recur out-of-sample at the individual-draw level.
- **"Broaden-before-implement Fix A if P5 ≥3":** P5=0 → **not triggered.** The
  frozen cycling-centric scope stands; no silent widening. (The vertical-
  separation near-misses suggest a *future*, separately-registered extension to
  layer/level identity, informed by `35329743` and `92c0997f` — not part of
  this fix.)
- **"Fix B: only retraction is P8(c) failing":** not triggered; no nonempty
  label was (or, on current data, could be) marked expressible via NONE. **No
  retraction.**

## Bottom line

**The out-of-sample holdout corroborates both frozen diagnoses. Fix A
(cycleway/separated-infrastructure uncertainty gate) and Fix B (analyzer
NONE-expressibility accounting) are clear to proceed to implementation,
unblocking PR #446.**

- **Fix A** is corroborated on the diagnosis level: the same-pavement-vs-
  physically-separated identity question is the dominant decisive-missing-
  evidence theme in 7 of 8 holdout datasets (P3), the informed `dd106a0f` and
  `92c0997f`-enriched behaviors matched, and the factorial evidence-ablations
  show precisely the failure mode the gate targets — voters confidently assume
  facility identity when the resolving evidence is absent (`92c0997f` minimal,
  `9f56d71d` draws 1/3, `35329743` draw 1) and abstain when the conflict is
  surfaced. Implement as frozen, including the `7175635e` non-regression
  fixture; post-fix efficacy still requires the fresh-panel test named in the
  hypothesis.
- **Fix B** is a deterministic reporting bug whose signature is visible in the
  dev summary itself (`modal_exact_human: true` alongside "label absent from
  menu"); its retraction condition cannot fire. Implement, then verify the
  3/3→0/3 flip with all other dev numbers unchanged.
- **The P7 falsification does not touch either fix.** It falsifies a factorial-
  invariance claim about `18ef284e` and reveals evidence-dependent strictness
  about short junction-anchor omission — a menu/exactness phenomenon squarely in
  the out-of-scope exact-pair track, which this holdout independently
  strengthened (P9 at 66.7%, double dev). What it does change: treat
  "stable lettered at 0.97+ under rich evidence" as compatible with a live
  anchor dispute (the audit dissented at 0.72–0.88 all four times), and carry
  that into the exact-pair phase design.
