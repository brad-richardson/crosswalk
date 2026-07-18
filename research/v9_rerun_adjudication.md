# v9 targeted-rerun adjudication (2026-07-18)

Ground-truth adjudication of the two pre-registered "expect NONE" groups that
flipped to merge in the `physical_context_v9_rerun_20260717` targeted vote
(pre-reg: 6e6fd4f, provenance: ed0adc2). Both adjudications were performed by
independent deep-read agents against stored geometry, candidate features, pack
images, and external OSM ground truth.

## Corrected scorecard

| group | pre-reg expected | panel voted | ground truth | panel correct? |
|-------|-----------------|-------------|--------------|----------------|
| 7175635e (fi) | MERGE | majority A | — | ✅ (as expected) |
| b33a27f5 (ch) | MERGE | unanimous A | — | ✅ (as expected) |
| 92c0997f (fi) | NONE | majority NONE (muse dissent A) | — | ✅ (as expected) |
| 66e22055 (au) | NONE | unanimous B (merge) | **SHARED, 0.75** | ✅ pre-reg was wrong |
| 5faa0b72 (fi) | NONE | majority H (merge, muse NONE) | **SHARED, 0.80** | ✅ pre-reg was wrong |
| a451bf05 (ch) | MERGE | unanimous NONE / no_exact_option | — | ⚠️ expressibility gap (#457) |

Panel ground-truth score: **5/6**, with the sixth blocked by option-menu
expressibility, not judgment. The feared over-correction regression did not
occur — the two "flips" were the MI-4 soften correctly overriding
mis-calibrated pre-reg expectations.

## 66e22055 (au_sydney_roads) — SHARED, confidence 0.75

R1 (`d910f21d…`, OSM way 881957767: `highway=cycleway, oneway, asphalt`, no
segregation tags) is the kerbside **exclusive on-road cycle lane** of William
Street: lateral offset 1.0–2.4 m from every William St representation
(`lateral_offset_m=1.70`, `offset_over_expected_halfwidth=0.44`), and William
St's own OSM way carries `cycleway:left=lane` + `cycleway:left:lane=exclusive`.
The genuinely separated shared-path network in the corridor sits 5.8–12.5 m
away and R1 is not part of it. A conservative "separately-mapped cycleway never
matches the road" reading is defensible (hence 0.75, not higher), but the
physical evidence favors merge.

## 5faa0b72 (fi_helsinki_roads) — SHARED, confidence 0.80

R1 sits at ~0.2–2.2 m median offset inside the targets' carriageway footprint
(all edges below the ~3 m separate-path signature; `has_parallel_sibling_ref=0`).
A 20 m corridor scan shows **no parallel carriageway exists** — the six targets
are the six closest features; the nearest real motor roads are 14 m+ away and
carry `bicycle:use_sidepath` pointing *at* this path. The targets' `mv:yes` is
`class_default`, never tagged; the corridor is a foot/cycle path feeding the
Lehtolan alikulkukäytävä underpass, imported into the Helsinki registry as
generic residential lines. Muse's NONE used the ref's tagged `mv:no` as
standalone separation proof and treated class-default access as tagged — both
explicitly against MI-4 as written.

## Decisions

1. **Finding 3 (tagged-denial tightening) is dead.** Either variant would have
   flipped Sydney's *correct* merge to wrong. Do not implement.
2. **No v10 rubric era.** MI-4 as frozen (2026-07-17+d1ba3b9a025a) produced the
   right answer wherever the menu could express it.
3. **The real gap is evidence plumbing:** `lateral_offset_m` /
   `lateral_offset_p95` / `offset_over_expected_halfwidth` are computed in
   candidate features but not rendered in packs — all seats flagged the
   omission and defaulted. Surfacing them is the minimal change that would
   likely have made both contested groups 3-0. (PR in flight.)
4. **a451bf05** re-vote via the #457 consensus-desired-edges seed, after the
   offset pack change lands (one pack-format change per re-vote wave).
5. **Muse seat calibration note:** muse went 0-for-2 on dissents this vote
   (A on 92c0997f's stacked tunnel corridor; NONE on 5faa0b72 via misread
   access evidence). Feed into the next voter-accuracy pass; no seat change now.
