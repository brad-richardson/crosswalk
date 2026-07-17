# v9 Fix Hypothesis — MI-4 soften + access/mode channel (pre-registered 2026-07-17)

**fix_id:** `v9_mi4_soften_access_channel__2026-07-17`
**Pre-registered before running the targeted rerun.** Blind per-group predictions
below are frozen; the rerun scores against them.

## What changed (the bundle under test)
- **MI-4 soften** — rubric era `2026-07-17+d1ba3b9a025a` (#453, `32f1eda`).
  Polarity inverted: structural + access evidence lead; unresolved (NONE) now
  requires a POSITIVE separation signal; absence of separation evidence no longer
  forces NONE. Access clauses gated behind "where present."
- **Access/mode evidence channel** — `access_lr` rendered into packs (#455,
  `3bb9b48`): per-edge `access='mv:.. bike:.. foot:..'` (`°`=class-default,
  bare=tagged, `?`=unknown), plus the `route_network` header for Geneva
  (`target_kind: route_network`). Never infers a denial except motorway.

## Hypothesis
The bundle recovers the cycleway merges the Fix-A gate wrongly pushed to NONE
(decidable by coverage partition and/or access affirming shared use / route
overlay) **without** losing the genuine-separation NONEs. The Fix-A guard
`7175635e` must return to a MERGE.

## Panel (unchanged from v7/v8)
`claude-opus-4-8`/high + `gpt-5.6-sol`/high + `meta/muse-spark-1.1`/high.

## Blind per-group predictions (the scoring checklist)

| group | dataset | prediction | primary basis | needs |
|-------|---------|-----------|---------------|-------|
| `7175635e` | fi_helsinki | **MERGE** | coverage partition ≈1.005 satisfies the affirming clause | Part A alone |
| `a451bf05` | ch_grand_geneva_cycle_schema | **MERGE** | route overlay (itinerary target) + `mv:yes` on the road; `route_network` header | Part B |
| `b33a27f5` | ch_grand_geneva_cycle_schema | **MERGE** | same route-overlay + access affirmation | Part B |
| `66e22055` | au_sydney | **NONE** | genuine separation: observed lateral offset + vertical stack + 15m fragment | — |
| `5faa0b72` | fi_helsinki | **NONE** | competing same-centerline representations + underpass conflict | — |
| `92c0997f` | fi_helsinki | **NONE** | vertical tunnel/surface conflict (also partly expressibility) | — |

## Scoring
- **Full pass:** all 6 match (3 MERGE, 3 NONE).
- **Partial:** `7175635e`/`5faa0b72`/`92c0997f`/`66e22055` are Part-A-testable even
  if the access channel under-delivers; the two Geneva MERGEs specifically validate
  Part B. Record each group's actual consensus + per-seat ballots + whether the
  pack rendered a non-empty `access=` line (so a Geneva miss can be attributed to
  access-data-absent vs rubric).
- A `66e22055`/`5faa0b72`/`92c0997f` that flips to MERGE is a **regression** (the
  soften over-corrected into a false merge) — the most important thing to catch.

## Scoring refinements (added pre-vote, from the holistic review)
- **Expressibility miss ≠ rubric miss.** A consensus of NONE with
  `none_reason=no_exact_option` and a *non-empty* desired merge set (plausible on
  `b33a27f5`, where a seat wants A-minus-e4 = {e1,e2,e3} that may not be on the
  menu) is a **menu/expressibility** failure, not a rubric failure — score it
  separately; the bundle is not blamed for a menu gap. (Consensus-desired-edges
  seeding, #457, is the fix for that class, not the rubric.)
- **Instrument every ballot:** record per-seat `none_reason` and whether any NONE
  cites the facility's OWN `designated` tags as the separation signal (that would
  be the finding-3 latent-wording risk firing — a distinct-mode designation is not
  physical-separation evidence).

## Access-verification gate (before voting — specific, not just "access= renders")
1. Geneva target segments render `bike:designated°` AND the `route_network` NOTE
   header is present.
2. Geneva ref segments render `mv:yes°`.
3. **Post-parser-fix guard:** `92c0997f`'s ref `fefac05e` does NOT render a tagged
   `mv:no`/`bike:no`/`foot:no` (the fabricated `when.vehicle` denial — must be gone).
4. `5faa0b72`'s cycleway still renders its genuine tagged `mv:denied` (real signal
   kept).
5. Pack `batch.json` rubric era is `2026-07-17+d1ba3b9a025a`, not `9463c80a0f77`.

## Method
Targeted rerun of exactly these 6 groups (+ any guard duplicates), packs rebuilt
under rubric `d1ba3b9a025a` WITH the (parser-fixed) access channel (re-extract →
sidecar rebuild → pack build; pass the gate above before voting). Do NOT re-run the
full 65 (quota). Score, then decide bless / iterate.

**Prerequisite:** the access-parser scoping fix (`when.using`/`when.vehicle`) must
merge and the references be re-extracted with it BEFORE the sidecar rebuild — else
the poisoned channel is baked in.
