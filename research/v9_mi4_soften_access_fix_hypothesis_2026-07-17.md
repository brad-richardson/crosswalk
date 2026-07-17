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

## Method
Targeted rerun of exactly these 6 groups (+ any guard duplicates), packs rebuilt
under rubric `d1ba3b9a025a` WITH the access channel (re-extract → sidecar rebuild →
pack build; verify `access=` renders before voting). Do NOT re-run the full 65
(quota). Score, then decide bless / iterate.
