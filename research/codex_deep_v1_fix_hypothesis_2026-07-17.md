# Frozen fix hypothesis — codex_deep_v1 holdout (pre-registration)

> **FROZEN 2026-07-17, before any holdout unsealing.** Nothing under
> `data/agents/stitching/diagnostics/codex_deep_v1/holdout/` was read, no
> `analyze --include-holdout` was run, and no marker existed when this document
> was committed. This is the pre-registered hypothesis the sealed 21-pack /
> 15-group holdout tests. `fix_id`:
> `codex_deep_v1__cycleway_gate_and_none_accounting__2026-07-17`.
>
> Drafted by an independent Fable analyst from the development readout + the two
> v7 wave analyses; reviewed and approved by Brad before the freeze.

## Provenance and disclosure (what was consulted to write this)

- `research/codex_stitch_diagnostic.md` (design doc), `development_readout.md`,
  `development_report.md`, both v7 wave analyses
  (`research/v7_wave_analysis_{codex,fable}_2026-07-17.md`).
- `cohort_assignment.json` (holdout **membership** only — group ids, not
  results) — so predictions can name specific groups.
- `labels/stitching/` human labels (public truth, not sealed Codex output). Dev's
  three strict pair labels are `66e22055`, `d4d2e782`, `4eed5e80` (all reject-all,
  `pair` semantics). The only holdout group with any stitching label is
  `dd106a0f` (Geneva): `selected_edges=[]` but `label_semantics=set` (deanchored)
  — **confirmed excluded** from the analyzer's strict pair metric
  (`_strict_human_labels` skips non-`pair` rows), so the holdout has 0 strict
  pair labels.
- **Contamination caveat (material):** the 195 public v7 wave ballots include one
  `gpt-5.6-sol` ballot per holdout pack under the identical prompt — effectively
  "draw #0". Predictions about *choice direction* on named holdout groups (P4,
  P6, P7) are therefore **informed, not blind**. The genuinely out-of-sample
  content is draw-level *stability* (wave has 1 draw; holdout has 3 + audit),
  *confidence behavior across draws*, *audit content/rates*, and whether dev
  patterns replicate at their dev rates.
- **Scoring decision (Brad, 2026-07-17):** keep P4/P6/P7 with the disclosure, but
  **score the freeze primarily on the blind predictions P1–P3, P5, P8–P10.**

Holdout composition (15 groups / 21 packs): singletons `35329743`, `750ae089`
(Sydney); `a451bf05`, `dd106a0f` (Geneva); `3f53c7e7`, `4148382c` (Berlin);
`e085519d` (Helsinki); `729f879b`, `91570f54` (London); `b8b5da4a` (Hong Kong);
`61a926e3`, `9f56d71d` (Amsterdam); `17053a69` (Philadelphia); factorial ×4:
`92c0997f` (Helsinki), `18ef284e` (Hong Kong).

## 1. The frozen fixes (deliberately narrow — exactly two)

### Fix A — cycleway/separated-infrastructure uncertainty gate (rubric/model behavior)

**Qualification:** human-confirmed (`66e22055` strict pair label) AND an unsafe
*stable* result (3/3 draws + audit chose option B at 0.94 while the audit itself
named the decisive missing fact). Thematically corroborated by both wave
analyses (cycleway/service-vs-road identity + vertical layering are the dominant
ambiguity themes).

**The rule (to be adapted into the rubric when the fix lands):**

> When the identity decision between a cycling/pedestrian facility (cycleway,
> shared path, footway, sidewalk) and a roadway candidate hinges on whether the
> facility is **same-pavement** (on-road lane / shared carriageway) versus
> **physically separated** (own alignment, verge/barrier-separated, or
> vertically separated), and the pack does not contain evidence resolving that
> distinction (physical attributes, coincidence/offset context, layer data, or
> imagery at the decisive location), the voter MUST NOT assume same-pavement
> identity. It must select `NONE` (insufficient evidence) — unless other pack
> evidence (naming continuity, endpoint topology, coverage partition)
> independently establishes the exact edge set.

**Testable claim (post-fix, fresh panel on rebuilt packs):** on `66e22055`-class
groups, no seat stably retains a separated-infrastructure edge at high
confidence; `66e22055` itself no longer produces a stable lettered retention of
R1→T1; **zero new false auto-accepts** on cycling-identity groups. **Non-
regression fixture (Brad):** the gate must NOT flip dev's `7175635e` (unanimous,
correct cycleway-corridor A) to NONE — make `7175635e` an explicit fixture when
the fix lands. (See `[[cycleway-class-unreliable-cross-mode-signal]]`: cycle
lanes legitimately share road pavement, which is exactly why the gate is
conditioned on the identity call *hinging* on the separation question.)

### Fix B — analyzer `NONE`-expressibility accounting (diagnostic-analyzer bug)

**Qualification:** makes human truth appear inexpressible (a reporting defect);
does not touch ballots, stability, audit feedback, or the 2/3 exact-human result.

**The rule:**

```
expressible(label) :=
    (label edge set == some lettered option's edge set)
    OR (label edge set == ∅ AND NONE is selectable)
```

`NONE` expresses **only** the empty set (never a nonempty label).

**Testable claim:** re-running the corrected analyzer on development flips "human
labels absent from menu" from 3/3 to 0/3 while every other number in
`development_report.md` is unchanged.

### Explicitly OUT OF SCOPE for this freeze

- **Option-menu generation changes** (exact-pair options, edge-level overrides).
  The 14/44 audit finding and the wave's 76–82% expressibility-NONE finding are
  strong but same-model / not human-confirmed; menu changes wait for human or
  fresh-panel corroboration — tracked as the separate exact-pair phases. P9 below
  *monitors* but does not act on this.
- Pack/evidence content changes, `none_reason` enum, seat weighting, confidence
  recalibration, PR #443 overlay regeneration (fresh-panel hygiene, not a fix).

## 2. Holdout predictions (the falsifiable core)

All are about the sealed 21-pack / 15-group holdout as-run (3 canonical draws + 1
audit per pack, **pre-fix** rubric). The holdout does not test the fixes'
efficacy — it tests whether the *diagnoses* generalize out-of-sample. **Blind
predictions (primary scoring): P1–P3, P5, P8–P10. Informed (kept w/ disclosure):
P4, P6, P7.** Confidence tags H/M/L.

- **P1 — Stability replicates (H, blind).** Unanimous 3/3 on 11–16 of 21 (52–76%;
  dev 61.4%); ≥ strict majority on ≥19/21; three-way splits ≤2. *Falsified if*
  unanimity <40% or splits ≥4.
- **P2 — Confidence again fails as a stability proxy (H, blind).** Mean canonical
  confidence ≥0.90; |conf(unanimous) − conf(non-unanimous)| <0.05. *Falsified if*
  non-unanimous packs run ≥0.10 lower.
- **P3 — Fix-A theme generalizes across datasets (H, blind).** Audit names
  cycling/pedestrian-vs-roadway identity or physical/vertical separation as
  decisive missing evidence on **≥2 of 15 groups across ≥2 datasets** (candidates:
  `a451bf05`, `dd106a0f`, `e085519d`, `17053a69`, `b8b5da4a`). *Falsified if* ≤1
  group or only 1 dataset → Fix A looks Sydney-local; narrow or hold it.
- **P4 — `dd106a0f` consistent with the human empty set (M-H, informed).** Modal
  diagnostic result `NONE`, not a confident lettered retention. **Pre-committed
  asymmetric reading:** if instead the draws stably choose a lettered option
  retaining a cycle-lane/motorway edge, that falsifies this *stability*
  prediction but *strengthens* Fix A (a second `66e22055`-shaped confident-wrong
  case). Either outcome is informative; this sentence is the pre-commitment.
- **P5 — Full `66e22055` signature is rare but real (M, blind).** 0–2 packs
  (point 1) show the complete signature (audit names a decisive missing
  separation/identity fact AND draws still ≥2/3 lettered at conf ≥0.90). *If ≥3:*
  gate scope (cycling only) too small — reconsider before implementation. *If 0:*
  Fix A rests on dev evidence alone (still frozen, lower urgency).
- **P6 — Factorial `92c0997f` DIVERGES (H, informed).** Enriched modal `NONE`
  (expressibility-flavored: mandatory `e15` conflates tunnel T3 with surface
  service T7); no-physical / no-coincidence / minimal each modal option A, ≥2/3,
  conf ≥0.90. *Falsified if* all four variants share a modal choice, or
  divergence runs the other way.
- **P7 — Factorial `18ef284e` INVARIANT (H, informed).** Modal `NONE` in all four
  variants, expressibility-flavored (option H omits boundary anchors
  `e3`/`e15`/`e18`), conf ≥0.85; ≥3/4 variants unanimous. Mechanism differs from
  dev's `7bac1f1d` (a menu/anchor dispute, not an easy case). *Falsified if* any
  variant's modal is a lettered option.
- **P8 — Fix B mechanics (deterministic, blind).** (a) Holdout has **0** strict
  pair-semantics labels (`dd106a0f` is `set`-semantics, excluded — confirmed).
  (b) Wherever the analyzer surfaces a human empty-set label with `NONE`
  selectable, pre-fix reports "absent", corrected reports expressible. (c) On dev,
  the corrected analyzer changes only the expressibility line (3/3→0/3).
  *Falsified if* any **nonempty** labeled set is marked expressible via `NONE`.
- **P9 — Menu-gap rate (M, blind; OUT-OF-SCOPE track, monitored only).** Audit
  reconstructs an exact set absent from the menu on **7–14 of 21 packs** (point
  10, ~48%; dev 31.8%). <5/21 weakens, ≥10/21 strengthens, the deferred
  exact-pair track. No menu change ships under this freeze regardless.
- **P10 — Audit-modal agreement (M, blind).** Audit agrees with the unique
  canonical modal on 65–90% of holdout packs that have one (dev 76%).
- **P11 — Split-location side-bet (L).** If a three-way split occurs, `35329743`
  (Sydney elevated-vs-ground) is the most likely pack.

### Retraction conditions

- **Retract/rescope Fix A** if P3 fails AND P5=0 (one-case patch, better handled
  by evidence enrichment than a global rubric rule).
- **Broaden-before-implement Fix A** if P5 ≥3 (frozen wording aimed too small —
  record as a scoping miss, not a silent widening).
- **Fix B** is a deterministic bug fix; only retraction is P8(c) failing.

## 3. Marker JSON (committed to the diagnostic tree)

```json
{
  "fix_id": "codex_deep_v1__cycleway_gate_and_none_accounting__2026-07-17",
  "frozen_at": "2026-07-17",
  "frozen_by": "brad",
  "source_readout": "data/agents/stitching/diagnostics/codex_deep_v1/development_readout.md",
  "hypothesis_doc": "research/codex_deep_v1_fix_hypothesis_2026-07-17.md",
  "in_scope_fixes": [
    {"id": "A", "kind": "rubric", "title": "cycleway/separated-infrastructure uncertainty gate", "claim": "When road/cycleway identity hinges on same-pavement vs physically-separated and the pack lacks that fact, the voter must choose NONE (insufficient evidence) rather than assume same-pavement identity, unless other evidence establishes the exact set. Grounded in truth-backed stable miss 66e22055."},
    {"id": "B", "kind": "diagnostic-analyzer", "title": "NONE-expressibility accounting", "claim": "Human-truth expressibility counts the selectable empty set: expressible iff label set equals a lettered option OR (label set empty AND NONE selectable). Reporting-only; ballots untouched. Corrects dev 3/3-absent to 0/3-absent."}
  ],
  "out_of_scope": [
    "option-menu generation / exact-pair options (separate track, awaits human or fresh-panel corroboration)",
    "pack/evidence content changes; none_reason enum; seat weighting; confidence recalibration",
    "PR #443 overlay regeneration (fresh-panel hygiene, not a fix)"
  ],
  "primary_scoring_predictions": ["P1", "P2", "P3", "P5", "P8", "P9", "P10"],
  "informed_predictions_disclosed": ["P4", "P6", "P7"],
  "disclosure": "Predictions on named groups partially informed by public v7 wave ballots (one same-model draw per pack) and by cohort membership + human labels; sealed holdout results were not read before this freeze."
}
```
