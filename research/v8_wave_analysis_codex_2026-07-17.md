# Crosswalk v8 physical/coincidence stitch-wave analysis — Codex `gpt-5.6-sol`

Date: 2026-07-17

Analyst seat: Codex `gpt-5.6-sol`, high reasoning

Scope: archived v7/v8 ballots plus v8 batch working directories only; no other analyst report or v7 analyst report was read.

## Executive summary

- **Do not bless v8 as run.** The disposition-critical Fix-A guard `7175635e` regressed from v7 unanimous merge A (3/3) to v8 majority NONE: Claude and Muse returned `insufficient_evidence`; only Codex retained A. This is the exact non-regression the wave was required to pass.
- **#450 materially reduced, but did not close, the expressibility gap.** My documented hand classification puts v7 at 58 expressibility NONEs out of 74 NONE ballots (78.4%); v8 has 38 validated `no_exact_option` ballots out of 85 NONEs (44.7%), a **−33.7 percentage-point** change among NONEs (−10.3 points over all 195 ballots). Twenty-five of the 58 v7 gap ballots chose an option in v8, but only 14 chose an option absent from that group-instance's v7 menu; 11 resolved through changed judgment despite the selected set already being offered in v7.
- **Every current v8 expressibility claim is real under the required R#/T# test.** All 38 `no_exact_option` desired sets were nonempty, JSON-valid, fully mappable through `batch.json`, inside the displayed candidate universe, and different from every offered option. There were **0 mapping failures, 0 duplicate-pair degeneracies, 0 out-of-universe pairs, and 0 expressible NONE misfires**. The 47 other NONEs correctly left `desired_edges` empty.
- **The residual menu gap is often small and actionable.** The 38 desired sets span 2–27 edges (median 11); the nearest menu option differs by a median of 3 edges. Thirteen ballots are one edge away, including 11 that only need one offered edge removed, across `35329743`, `dc818edc`, `4dc33ddd`, `7680bb19`, `b8b5da4a`, `5e31936e`, and `17053a69`.
- **Fix A appears to over-trigger beyond the guard.** Among the 15 enriched groups containing a `cycleway` class (45 paired ballots), NONE rises from 15/45 in v7 to 31/45 in v8 (33.3%→68.9%, +35.6 points). There are 20 option→NONE seat transitions; 14 are now `insufficient_evidence`. Seven group consensuses move from an option in v7 to NONE in v8: `66e22055`, `a451bf05`, `b33a27f5`, `5faa0b72`, `7175635e`, `5e31936e`, and `9f56d71d`.
- **The 2×2 does not identify a dominant factor.** Physical evidence changes 7/30 paired ballots across its two contrasts; coincidence changes 9/30. Coincidence-on has *more* NONEs than coincidence-off (10/30 versus 7/30), and physical-on only slightly fewer than physical-off (8/30 versus 9/30). `minimal` has fewer NONEs and higher confidence than enriched, but this aggregate result hides the unsafe `1b90f03b` A-versus-J reversal.
- **Ten v8 group-instances route `auto_accept`, but two should be withheld.** Withhold `1b90f03b/minimal` because every context-bearing v8 cell unanimously chooses a different edge set, and withhold `6775ade1/enriched` pending a human cross-wave check because v7 was three-way split and the now-unanimous v8 option already existed in v7. The other eight auto-accept rows (six unique edge sets after deduplicating `7bac1f1d` variants) are well corroborated.
- **No on minting the deferred v7+v8 auto-accepts as a tranche.** In addition to the two unsafe v8 rows, v7 auto-accept `8f152b92` is routed to human review in v8 for low confidence. The wave-level guard failure means automatic production minting should wait for an MI-4 revision and a targeted rerun, even though several individual labels look sound.

## Scope, reconstruction, and counting rules

I loaded `votes.csv`, `consensus.csv`, and `evidence.csv` under all `labels/votes/dataset=*` directories, filtering v8 by `source_batch` containing `physical_context_v8_20260717` and v7 by `physical_context_v7_20260715`. The resulting canonical panels are complete:

| Wave | Ballots | Group-instances | Seats per group | NONE ballots |
|---|---:|---:|---:|---:|
| v7 | 195 | 65 | 3 | 74 (37.9%) |
| v8 | 195 | 65 | 3 | 85 (43.6%) |

All 195 v8 archive rows are successful ballots: `error` and `abstain_reason` are empty throughout. `attempt=2` occurs on three final successful rows, but the archive contains no duplicate failed attempts, so no row was double-counted. The rows are Claude on `35329743/enriched` and `fb8f359f/no_coincidence`, and Codex on `b049e0de/enriched`.

For every v8 group I reconstructed R#/T# labels positionally from `batch.json` (`ref_ids`→R1…Rn; `target_ids`→T1…Tn), reconstructed every offered source-id edge set from the group menu, and compared unordered edge sets. As a cross-check, all 65 positional maps exactly matched the labels and ids rendered in the group metadata.

For v7, where `none_reason` is absent, I used this mutually exclusive manual rule:

1. `all_edges_no_match` when the reasoning says the accepted set is empty;
2. `insufficient_evidence` when it says no exact set or option can be established from the evidence;
3. `no_exact_option` when it establishes a nonempty accepted-set principle and proves every offered option adds a rejected edge or omits an accepted edge, even if secondary uncertainty remains.

Representative anchors are v7 Codex on `bdbdf792`, “No offered option contains exactly those 12 edges” (`no_exact_option`); v7 Codex on `4eed5e80`, “No offered option represents the required empty edge set” (`all_edges_no_match`); and v7 Codex on `35329743`, “The evidence cannot establish an exact final set” (`insufficient_evidence`). The five v7 ballots placed in the last category are `35329743/codex`, `5faa0b72/codex`, `92c0997f/no_coincidence/claude`, `ca8d1f92/claude`, and `e8a39e6d/claude`.

## Goal 1 — Did exact-pair seeds close the expressibility gap?

### Headline comparison

| NONE category | v7 count | v7 share of NONE | v8 count | v8 share of NONE | Change |
|---|---:|---:|---:|---:|---:|
| No exact option offered | 58 | 78.4% | 38 | 44.7% | **−33.7 pp** |
| All edges are no-match | 11 | 14.9% | 10 | 11.8% | −3.1 pp |
| Insufficient evidence | 5 | 6.8% | 37 | 43.5% | **+36.8 pp** |
| Total NONE | 74 | 100% | 85 | 100% | +11 ballots |

As a fraction of all 195 ballots, expressibility NONEs fall from 29.7% to 19.5% (−10.3 points). Genuine reject-all is nearly flat, 5.6%→5.1%, while insufficient evidence rises sharply, 2.6%→19.0%. Thus v8's total NONE rate is higher despite a smaller menu gap; the new uncertainty behavior, especially around cycleways, more than offsets the menu improvement.

The paired fate of the 58 v7 expressibility ballots is:

| v8 outcome | Count | Share of 58 |
|---|---:|---:|
| Selected a concrete option | 25 | 43.1% |
| Still `no_exact_option` | 22 | 37.9% |
| Became `insufficient_evidence` | 11 | 19.0% |

Of the 25 resolved ballots, **14 selected a v8 option-id that was absent from that group-instance's v7 evidence menu**; 11 selected an option already present in v7. The 14 direct new-option selections occur in six group-instances: `53bb11d7/enriched` (Codex, Muse), `e085519d/enriched` (Codex), `1b90f03b/enriched` (all seats), `1b90f03b/no_physical` (Claude, Codex), `1b90f03b/no_coincidence` (all seats), and `c8da4c08/enriched` (all seats). This is the clearest positive evidence that #450 put previously wanted sets on the menu. It accounts for 24.1% of the 58 v7 expressibility ballots and 56.0% of the 25 resolved ballots.

Menu capacity did materially expand: the mean menu grew from 8.43 options in v7 to 12.26 in v8 (+45.4%); 52/65 group-instance menus changed and collectively contain 249 option-ids not present in their paired v7 menus. But **all 38 current v8 desired sets are still absent**, across 25 group-instances and 21 unique `group_id`s. #450 therefore helped but did not close the gap.

The residual gaps are not all combinatorial explosions. Desired sets contain 2–27 edges (mean 10.53, median 11), and their nearest offered option has a median symmetric difference of 3 edges:

| Nearest-menu symmetric difference | v8 `no_exact_option` ballots |
|---:|---:|
| 1 edge | 13 |
| 2 edges | 4 |
| 3 edges | 8 |
| 4 edges | 5 |
| 5+ edges | 8 |

Eleven of the 13 one-edge cases want an offered option minus one edge; two want an offered option plus one. The one-deletion cases span seven group-instances: `35329743` (Codex wants 9 edges vs J's 10), `dc818edc` (Claude 17 vs D's 18), `4dc33ddd` (Codex 11 vs A's 12), `7680bb19` (Muse 4 vs L's 5), `b8b5da4a` (three seats, each one deletion from C or E), `5e31936e` (Claude 2 vs G's 3), and `17053a69` (all seats, one deletion from E or H). Several disputed extras are semantic role/coincidence conflicts rather than geometric slivers, suggesting that “flagged edge” seeding does not yet cover the panel's full rejection vocabulary.

**Answer:** #450 put a wanted set on the menu in a meaningful minority of paired cases and reduced the expressibility-NONE share, but it did not close the gap. The remaining 38 claims survive the required mapping and option-set validation.

## Goal 2 — Full v8 `none_reason` breakdown and desired-edge validation

All 85 NONE ballots have a valid enum. Overall: 38 `no_exact_option` (44.7%), 37 `insufficient_evidence` (43.5%), and 10 `all_edges_no_match` (11.8%).

### By seat

| Seat | NONE / 65 | `no_exact_option` | `all_edges_no_match` | `insufficient_evidence` |
|---|---:|---:|---:|---:|
| Claude | 36 (55.4%) | 12 | 3 | 21 |
| Codex | 30 (46.2%) | 16 | 4 | 10 |
| Muse | 19 (29.2%) | 10 | 3 | 6 |
| **Total** | **85 (43.6%)** | **38** | **10** | **37** |

Claude is the uncertainty-gate driver: 21/36 of its NONEs are `insufficient_evidence`. Codex is the most expressibility-sensitive: 16/30 of its NONEs are `no_exact_option`. Muse has the lowest NONE rate but still contributes 10 validated menu gaps.

### By dataset and variant

Columns are `no_exact_option / all_edges_no_match / insufficient_evidence`.

| Dataset | Variant | NONE total | No exact | All no-match | Insufficient |
|---|---|---:|---:|---:|---:|
| au_sydney_roads | enriched | 8 | 4 | 0 | 4 |
| au_sydney_roads | no_physical | 2 | 1 | 0 | 1 |
| au_sydney_roads | no_coincidence | 1 | 0 | 0 | 1 |
| au_sydney_roads | minimal | 2 | 1 | 0 | 1 |
| ch_grand_geneva_cycle_schema | enriched | 12 | 1 | 1 | 10 |
| de_berlin_roads | enriched | 12 | 6 | 6 | 0 |
| de_berlin_roads | no_physical | 0 | 0 | 0 | 0 |
| de_berlin_roads | no_coincidence | 0 | 0 | 0 | 0 |
| de_berlin_roads | minimal | 0 | 0 | 0 | 0 |
| fi_helsinki_roads | enriched | 12 | 2 | 0 | 10 |
| fi_helsinki_roads | no_physical | 2 | 0 | 0 | 2 |
| fi_helsinki_roads | no_coincidence | 2 | 1 | 0 | 1 |
| fi_helsinki_roads | minimal | 1 | 0 | 0 | 1 |
| gb_london_roads | enriched | 4 | 2 | 0 | 2 |
| gb_london_roads | no_physical | 0 | 0 | 0 | 0 |
| gb_london_roads | no_coincidence | 0 | 0 | 0 | 0 |
| gb_london_roads | minimal | 0 | 0 | 0 | 0 |
| hk_hongkong_roads | enriched | 14 | 9 | 3 | 2 |
| hk_hongkong_roads | no_physical | 1 | 1 | 0 | 0 |
| hk_hongkong_roads | no_coincidence | 0 | 0 | 0 | 0 |
| hk_hongkong_roads | minimal | 1 | 1 | 0 | 0 |
| nl_amsterdam_roads | enriched | 7 | 6 | 0 | 1 |
| us_philadelphia_sidewalks | enriched | 4 | 3 | 0 | 1 |

Berlin's NONEs are decisive: half expressibility, half reject-all, zero insufficient. Geneva is the opposite: 10/12 NONEs are insufficient evidence. Hong Kong and Amsterdam contain the densest remaining menu gaps (9 and 6 enriched ballots respectively).

### R#/T# validation audit

For the 38 ballots where the contract requires `desired_edges`:

- 38/38 are nonempty and valid JSON;
- 38/38 map completely from positional R#/T# labels in `batch.json`;
- 0 contain unmapped labels;
- 0 collapse through duplicate edge pairs;
- 0 contain an edge outside that group's displayed candidate universe;
- 0 equal any offered option edge set.

For the other 47 NONEs (`all_edges_no_match` or `insufficient_evidence`), `desired_edges` is empty as required. No `none_reason` value needs to be reclassified as an expressible-option misfire. The mapping-failure and degenerate rate is therefore **0/38 among required desired sets** (and 0 malformed records among all 85 NONEs).

## Goal 3 — `7175635e` Fix-A non-regression and cycleway over-trigger scan

### The guard failed

The option menu and candidate universe are unchanged for `7175635e` between v7 and v8. The v7 panel unanimously merged all three edges with A:

| Wave | Claude | Codex | Muse | Consensus |
|---|---|---|---|---|
| v7 | A / 0.72 | A / 0.93 | A / 0.92 | unanimous A, mean 0.857 |
| v8 | NONE / 0.68 / insufficient | A / 0.90 | NONE / 0.62 / insufficient | majority NONE, mean 0.650; minority `codex=A` |

Claude and Muse both explicitly invoke the new MI-4 “painted/shared pavement versus raised/curbed track” uncertainty gate. Claude says the pack has “no physical separation attributes,” “no lateral-offset or geometric-coincidence measure,” and no junction zoom; Muse likewise says coverage partition alone does not distinguish a same-pavement lane from a separated track. Codex retains the correct merge because T1/T2/T3 continuously partition R1 and the cycleway/service class mismatch is non-dispositive under the strong corridor evidence.

This is not a subtle pass: **two seats went NONE and the group consensus flipped to NONE.** The required non-regression condition is false.

### Broader cycleway signal

I defined the scan from v8 batch classes, not a reasoning keyword: 15 enriched group-instances contain at least one `cycleway` class, yielding 45 exactly paired seat ballots.

| Metric | v7 | v8 | Change |
|---|---:|---:|---:|
| NONE ballots | 15/45 (33.3%) | 31/45 (68.9%) | +16, +35.6 pp |
| Option→NONE paired transitions | — | 20 | 14 insufficient, 6 no-exact |
| NONE→option paired transitions | — | 4 | — |

Seven of the 15 group consensuses flip from an option to NONE:

- `7175635e`: unanimous A → majority NONE; confirmed false non-regression.
- `a451bf05`: unanimous B → unanimous NONE; Claude and Codex are insufficient, Muse has a validated no-exact set.
- `b33a27f5`: unanimous A → majority NONE; Claude and Codex insufficient, Muse retains A.
- `5faa0b72`: majority H → unanimous NONE; all three v8 seats insufficient.
- `66e22055`: majority B → unanimous NONE; all three v8 seats insufficient. This case is genuinely harder because the candidates pair a cycleway with a secondary road and a motorway tunnel, and v7 Codex already rejected all edges.
- `5e31936e`: unanimous G → majority NONE; Claude wants a validated two-edge set, Codex is insufficient, Muse retains G.
- `9f56d71d`: majority H → majority NONE; Claude and Codex now want different validated non-menu sets, Muse selects A. This is more an expressibility/judgment shift than a pure uncertainty-gate failure.

Additional one-seat option→insufficient transitions occur at `f4f3387b` (Claude C→NONE), `dd106a0f` (Muse C→NONE), and `e4746a04` (Muse A→NONE). Not every new NONE is necessarily wrong, but the cohort-level doubling plus the known false guard makes a chance-only explanation implausible. The gate is treating missing separation evidence as a reason to suspend otherwise strong M:N continuity more often than v7 did.

There are counterexamples showing the rubric can still merge legitimate continuous infrastructure: `53bb11d7` moves from majority NONE to unanimous I, and `ca8d1f92` remains majority A. The result is therefore **over-triggering, not a universal reject rule**—but it is broad enough to block blessing.

## Goal 4 — 2×2 physical × coincidence contrasts

All four variants of each control have the same `option_menu_sha256`, so option letters are directly comparable within a group. Cells below are `choice / confidence / NONE reason`; `—` means a concrete option.

| Group | Variant (physical, coincidence) | Claude | Codex | Muse | Consensus / routing |
|---|---|---|---|---|---|
| `fb8f359f` | enriched (on,on) | NONE/.52/no-exact | NONE/.87/no-exact | A/.68/— | majority NONE / human |
|  | no_physical (off,on) | NONE/.72/insufficient | NONE/.87/no-exact | A/.62/— | majority NONE / human |
|  | no_coincidence (on,off) | NONE/.50/insufficient | A/.82/— | A/.74/— | majority A / human |
|  | minimal (off,off) | NONE/.40/insufficient | A/.87/— | NONE/.84/no-exact | majority NONE / human |
| `7bac1f1d` | enriched (on,on) | I/.82 | I/.97 | I/.92 | unanimous I / auto |
|  | no_physical (off,on) | I/.74 | I/.97 | I/.84 | unanimous I / human (low confidence) |
|  | no_coincidence (on,off) | I/.82 | I/.98 | I/.89 | unanimous I / auto |
|  | minimal (off,off) | I/.76 | I/.96 | I/.88 | unanimous I / auto |
| `92c0997f` | enriched (on,on) | NONE/.55/insufficient | NONE/.91/insufficient | J/.78/— | majority NONE / human |
|  | no_physical (off,on) | NONE/.52/insufficient | NONE/.96/insufficient | A/.88/— | majority NONE / human |
|  | no_coincidence (on,off) | NONE/.62/insufficient | NONE/.87/no-exact | A/.86/— | majority NONE / human |
|  | minimal (off,off) | NONE/.70/insufficient | A/.96/— | A/.93/— | majority A / human |
| `1b90f03b` | enriched (on,on) | J/.74 | J/.97 | J/.86 | unanimous J / human |
|  | no_physical (off,on) | J/.60 | J/.97 | J/.86 | unanimous J / human |
|  | no_coincidence (on,off) | J/.62 | J/.97 | J/.84 | unanimous J / human |
|  | minimal (off,off) | A/.86 | A/.96 | A/.86 | unanimous A / **auto** |
| `18ef284e` | enriched (on,on) | H/.78 | NONE/.83/no-exact | H/.82 | majority H / human |
|  | no_physical (off,on) | H/.72 | NONE/.86/no-exact | H/.81 | majority H / human |
|  | no_coincidence (on,off) | H/.78 | H/.89 | H/.72 | unanimous H / human |
|  | minimal (off,off) | H/.72 | NONE/.93/no-exact | H/.94 | majority H / human |

### Aggregate cell results

| Cell | NONE ballots / 15 | Mean confidence | Unanimous groups / 5 | Auto-accept / 5 |
|---|---:|---:|---:|---:|
| enriched | 5 | 0.801 | 2 | 1 |
| no_physical | 5 | 0.796 | 2 | 0 |
| no_coincidence | 3 | 0.795 | 3 | 1 |
| minimal | 4 | 0.838 | 2 | 2 |

Factor main effects on the binary NONE outcome are small and not in the hypothesized direction for coincidence:

- physical on: 8/30 NONE versus physical off: 9/30 (−3.3 points);
- coincidence on: 10/30 NONE versus coincidence off: 7/30 (+10.0 points);
- NONE difference-in-differences: `(enriched − no_physical) − (no_coincidence − minimal)` = +6.7 points.

On paired categorical choices, physical changes 1/15 ballots when coincidence is on and 6/15 when it is off (7/30 total contrasts). Coincidence changes 3/15 when physical is on and 6/15 when it is off (9/30). Coincidence is therefore slightly more influential by count, but neither factor is dominant or monotonic.

The group-level patterns are more informative:

- `7bac1f1d` is invariant: every seat chooses I in all four cells. Names/classes already distinguish the underground Mäusetunnel footway/steps from the surface road.
- `1b90f03b` exhibits redundancy/interaction: either physical **or** coincidence context is sufficient for unanimous J; only when both are absent does every seat choose all-edge A. This is a meaningful semantic degradation in `minimal` even though aggregate minimal metrics look good.
- `92c0997f` becomes majority A only in `minimal`; either evidence family alone is enough to keep at least two seats from agreeing on A.
- `fb8f359f` and Codex on `18ef284e` are non-monotonic, so they do not support a clean causal ranking of the factors.

**Answer:** no factor earns a robust dominance claim. `minimal` is not numerically worse than enriched—7/15 ballots change, it has one fewer NONE and +0.037 confidence—but its `1b90f03b` auto-accept demonstrates that aggregate confidence and unanimity can conceal an over-merge.

## Goal 5 — Blessing and the v8 auto-accept set

V8 has 10 auto-accept group-instances, representing eight unique `group_id`s and eight unique production edge sets after collapsing the three identical `7bac1f1d` variant rows.

| Dataset | Variant | Group | Choice | Edges | Mean conf. | Assessment |
|---|---|---|---|---:|---:|---|
| de_berlin | enriched | `33a36ca5` | A | 3 | .913 | Trustworthy: v7 and v8 both unanimous A; clean same-name 1:N partition. |
| de_berlin | enriched | `3f53c7e7` | A | 4 | .903 | Trustworthy: v7 and v8 both unanimous A; complementary junction fractions. |
| de_berlin | enriched | `7bac1f1d` | I | 6 | .903 | Trustworthy edge set; all four v8 variants and all v7 variants choose I. |
| de_berlin | no_coincidence | `7bac1f1d` | I | 6 | .897 | Same trustworthy edge set; deduplicate at mint time. |
| de_berlin | minimal | `7bac1f1d` | I | 6 | .867 | Same trustworthy edge set; deduplicate at mint time. |
| gb_london | enriched | `91570f54` | A | 2 | .900 | Trustworthy: stable v7/v8 unanimous N:1 segmentation. |
| gb_london | minimal | `1b90f03b` | A | 14 | .893 | **Withhold:** every context-bearing v8 cell unanimously chooses J, not A; v7 minimal had a Codex NONE dissent. |
| hk_hongkong | enriched | `e0099fb8` | A | 3 | .953 | Trustworthy: stable v7/v8 unanimous same-name 1:N partition. |
| nl_amsterdam | enriched | `6775ade1` | E | 11 | .867 | **Withhold pending human check:** v7 was Claude NONE / Codex E / Muse A; E already existed in v7, so v8 unanimity is cross-wave judgment drift, not direct menu repair. |
| us_philadelphia | enriched | `ee358f5a` | B | 3 | .883 | Trustworthy: v7 and v8 both unanimous B; excludes a perpendicular low-confidence branch. |

The six v7 auto-accept rows are `e0099fb8/enriched`, `8f152b92/enriched`, `7bac1f1d/no_coincidence`, `33a36ca5/enriched`, `3f53c7e7/enriched`, and `91570f54/enriched`. Five retain the same v8 auto-accept edge set. `8f152b92` remains unanimous B in v8 but is now routed to human review for low confidence (mean .867), so the latest policy signal says not to mint it automatically.

**Disposition:** **iterate the rubric**, rather than bless or treat expressibility as the only blocker. First revise MI-4 so strong contiguous partition/corridor evidence can clear the uncertainty gate without requiring unavailable cross-section evidence, then rerun at least the guard and the affected cycle cohort. In parallel, extend option seeds for semantic role/coincidence deletions, especially the seven one-edge group-instances above.

**Minting:** **No** on minting the deferred v7+v8 auto-accepts as a combined automatic tranche now. Even though eight v8 rows look individually sound, `1b90f03b/minimal`, `6775ade1/enriched`, and v7 `8f152b92/enriched` require withholding, and the wave itself failed its explicit non-regression gate.

## Goal 6 — Per-seat behavior

Agreement is measured only on the 61 group-instances with unanimous or majority consensus, excluding the four no-majority cases.

| Seat | Choice distribution | Mean conf. | NONE rate | NONE reason mix (exact/all/insuff.) | Agreement with decided consensus | Successful retry rate |
|---|---|---:|---:|---|---:|---:|
| Claude | NONE 36; A 8; H 5; I 5; J 4; B 3; C/D/E/K 1 each | .689 | 55.4% | 12 / 3 / 21 | 52/61 (85.2%) | 2/65 (3.1%) |
| Codex | NONE 30; A 15; I 6; B 4; J 3; K 2; C/D/E/H/N 1 each | .938 | 46.2% | 16 / 4 / 10 | 52/61 (85.2%) | 1/65 (1.5%) |
| Muse | NONE 19; A 19; J 6; B/H/I 5 each; K 2; C/E/G/N 1 each | .815 | 29.2% | 10 / 3 / 6 | 50/61 (82.0%) | 0/65 |

Characteristic behavior:

- **Claude** is the most conservative and lowest-confidence seat. It supplies 21/37 panel `insufficient_evidence` ballots and repeatedly treats missing junction/separation evidence as dispositive under MI-4. Its mean confidence drops from .738 on options to .649 on NONE.
- **Codex** is extremely high-confidence on both sides (.940 options, .935 NONE) and most likely to articulate an exact alternative set: 16 validated no-exact ballots. It is the persistent exact-set dissenter on `18ef284e` in enriched, no_physical, and minimal, while the other seats choose H.
- **Muse** is most selection-prone (46/65 concrete options, including A on 19) and has the lowest consensus agreement. It is the minority seat in 11 of the 29 majority groups versus 9 each for Claude and Codex. Its reasoning commonly uses explicit coverage arithmetic and is willing to accept an optimizer/full-M:N interpretation where Claude abstains.

No seat is a universal outlier: majority dissent is 11 Muse / 9 Claude / 9 Codex, and Claude and Codex tie on decided-consensus agreement. The systematic axes are instead **Claude uncertainty**, **Codex exact-menu maximalism**, and **Muse acceptance/full-coverage preference**.

Operationally, invocation provenance looks healthy: 3/195 final ballots required a second attempt, and 0/195 final ballots are abstains or errors.

## Goal 7 — Frontage-road and vertically layered ambiguity

This wave is saturated with physical ambiguity: a keyword audit finds bridge/tunnel/layer/vertical/grade-separation language in 160/195 reasonings across 62/65 group-instances. “Frontage” appears in 10 ballots across nine group-instances; broader service-road language appears in 58 ballots across 25.

### Correct non-merges and restraint

- `4eed5e80` is unanimous `all_edges_no_match`: Kai Tak Tunnel at layer −1 is not the layer-0 Kowloon City Road surface alignment.
- `d4d2e782` and `422d5d7b` are each unanimous `all_edges_no_match`: an indoor tunneled footway at layer −1 is not a surface tertiary/secondary road. These account for six more decisive reject ballots.
- `7bac1f1d` chooses I in all four factorial cells, consistently excluding the underground Mäusetunnel footway/steps from Friedrichstraße roadway targets. It does so even in `minimal`, showing that explicit names and roles can be enough without the two experimental factors.
- `3b876df0` is unanimous NONE with three validated desired sets because all offered choices retain surface→Tunnel Tiergarten conflicts; the panel does not blindly merge vertical coincidence.

### Evidence that the panel does not blindly over-split roads

- `33a36ca5` is stable unanimous A and auto-accepted even though bridge/layer attributes differ; all seats treat the same-name, near-exact coverage partition as a representation difference.
- `3f53c7e7` is stable unanimous A and preserves two short boundary anchors despite a partial bridge/layer discrepancy.
- `91570f54` is stable unanimous A; the short Blue Lion Place junction anchor survives unknown tunnel metadata.
- `6775ade1` is v8 unanimous E despite systematically offset layer conventions on the IJ-tunnel corridor. This is one reason the cross-wave shift merits review, but it also shows that the panel can retain a continuous layered road.

### What the factorial says about frontage/coincidence

`1b90f03b` is the clearest positive context result. In enriched, no_physical, and no_coincidence, all three seats choose J, excluding the distinct Harrier Avenue overlap while preserving legitimate Eastern Avenue boundary anchors. With neither context, all three choose A and the row auto-accepts. Thus either evidence family prevents a likely over-merge, and their information is redundant for this case.

`92c0997f` is also context-sensitive: majority A appears only in minimal; every cell with at least one factor remains majority NONE. But the uncertainty is not cleanly resolved—the seats disagree about deep-tunnel versus surface-service representation—so this is caution, not a validated correct answer.

`18ef284e` and `fb8f359f` are non-monotonic. Codex accepts H only in `18ef284e/no_coincidence` but wants a non-menu set in the other three cells. `fb8f359f` produces majority NONE in three cells and majority A in no_coincidence. These do not establish a general causal advantage.

**Assessment:** physical/coincidence evidence helps avoid some spatially coincident but identity-distinct merges, most clearly `1b90f03b`, and the panel still merges several legitimate continuous road cases. The main over-splitting problem is specifically the revised cycleway uncertainty gate, not a general inability to merge layered roads.

## Goal 8 — Disagreement map and pack-feedback synthesis

### Where the panel splits

V8 has 32 unanimous, 29 majority, and 4 no-majority group-instances. Non-unanimity is **not concentrated in ablations**: 25/50 enriched instances (50.0%) and 8/15 ablated instances (53.3%) split. All four no-majority cases are enriched.

Enriched disagreement is most concentrated in Helsinki (6/7 groups non-unanimous), followed by Amsterdam (3/5), Sydney (4/7), Hong Kong (4/8), Geneva/London/Philadelphia (2/5 each), and Berlin (2/8).

Among the 29 majority splits, 25 (86.2%) are NONE-versus-option fault lines: 14 have two NONEs against one option and 11 have two options against one NONE. Only four are option-letter disputes. The complete map is:

- **Two NONE vs one option (14):** `9f56d71d` (Muse A), `5e31936e` (Muse G), `cd320a3c` (Muse N), `b049e0de` (Codex B), `fb8f359f/enriched` (Muse A), `fb8f359f/no_physical` (Muse A), `fb8f359f/minimal` (Codex A), `e085519d` (Codex N), `7175635e` (Codex A), `92c0997f/enriched` (Muse J), `92c0997f/no_coincidence` (Muse A), `92c0997f/no_physical` (Muse A), `b33a27f5` (Muse A), and `4148382c` (Muse B).
- **Two options vs one NONE (11):** `18ef284e/enriched`, `/no_physical`, and `/minimal` (Codex NONE in all three); `35329743` (Codex NONE); `fb8f359f/no_coincidence` (Claude NONE); `f4f3387b` (Claude NONE); `4dc33ddd` (Codex NONE); `92c0997f/minimal` (Claude NONE); `ca8d1f92` (Claude NONE); `729f879b` (Claude NONE); and `b7f57035` (Claude NONE).
- **Option vs option (4):** `61a926e3` (Claude B vs Codex/Muse A), `8582dd97` (Claude A vs Codex/Muse K), `ea25e0bd` (Muse H vs Claude/Codex D), and `8dae3675` (Claude H vs Codex/Muse A).
- **No majority, all three different (4):** `750ae089` = Claude NONE / Codex A / Muse B; `1de025b8` = Claude C / Codex NONE / Muse B; `7680bb19` = Claude A / Codex I / Muse NONE; `dc818edc` = Claude NONE / Codex B / Muse J.

Ablated splits are limited to three variants of `fb8f359f`, three variants of `92c0997f`, and no_physical/minimal for `18ef284e`; all Berlin and London ablated controls are unanimous. The splits are therefore driven by a few intrinsically ambiguous controls rather than ablation globally destabilizing the panel.

### Recurring pack-feedback themes

I keyword-coded the JSON `pack_feedback` text; categories overlap and count ballots, not independent defects.

| Feedback theme | Ballots | Group-instances |
|---|---:|---:|
| Local zoom / higher-resolution / aerial or street-level imagery | 160 | 65 |
| Vertical or physical metadata (layer/bridge/tunnel/grade) | 166 | 64 |
| Coincidence, lateral separation, same-pavement, or carriageway detail | 157 | 62 |
| Role/class/name/access semantics | 156 | 64 |
| Topology, endpoints, direction, or connectivity | 103 | 56 |
| Short-edge/sliver/junction-anchor interpretation | 69 | 38 |
| Option/menu rendering or coverage | 26 | 22 |

The actionable synthesis is:

1. **Junction-scale visual evidence is the universal missing input.** Every group-instance has at least one seat asking for a zoom or higher-resolution view. This is especially important for determining ALONG boundary anchors versus endpoint clips.
2. **MI-4 needs an affirmative-evidence escape hatch.** `7175635e/claude` asks for separation attributes and a lateral-offset measure that the pack does not provide. If such data cannot routinely be supplied, the rubric must say when exact coverage partition and topology are sufficient to infer same-way identity.
3. **Vertical conventions need normalization or explicit uncertainty semantics.** `92c0997f`, `3b876df0`, and the Geneva cycle groups repeatedly cite incompatible layer scales and unknown target physical flags. Missing flags are correctly treated as unknown, but systematic source-specific layer offsets are still confusing the identity test.
4. **The menu generator needs semantic subtraction seeds.** Current residual gaps frequently require removing a high-confidence, non-sliver edge because it is a role/coincidence conflict. The 11 one-deletion ballots are direct test cases.
5. **Option rendering still has isolated failures.** Claude on `fb8f359f/minimal` reports “option images identical; tie-lines unreadable at this scale.” Twenty-six ballots across 22 group-instances mention menu/rendering/coverage shortcomings even after the exact-pair overlays.

## Prioritized human-review list

1. **`7175635e` first.** Confirm A from the v7 evidence and diagnose why the MI-4 wording overrode exact 1:N coverage for Claude and Muse. This is the release-blocking guard.
2. **The strongest adjacent cycle regressions:** `a451bf05`, `b33a27f5`, and `5faa0b72`, followed by `66e22055`, `5e31936e`, `9f56d71d`, `f4f3387b`, `dd106a0f`, and `e4746a04`. Determine which are valid uncertainty escalations and which are false holds.
3. **Withheld auto-accept `1b90f03b/minimal`.** Compare A's 14 edges to J in all three context-bearing cells; inspect the Harrier Avenue/Eastern Avenue overlap before any label is minted.
4. **Withheld auto-accept `6775ade1/enriched`.** Reconcile v7 Claude NONE / Codex E / Muse A with v8 unanimous E. Because E already existed in v7, this is rubric/judgment reproducibility, not menu repair.
5. **The one-deletion expressibility cases:** `35329743`, `dc818edc`, `4dc33ddd`, `7680bb19`, `b8b5da4a`, `5e31936e`, and `17053a69`. `b8b5da4a` and `17053a69` deserve priority because all three seats independently want a one-edge-trimmed set.
6. **All four no-majority groups:** `750ae089`, `1de025b8`, `7680bb19`, and `dc818edc`. Each has three distinct verdicts, including one NONE, and therefore exposes a different unresolved fault line rather than a simple outlier.
7. **The non-monotonic controls:** `fb8f359f`, `92c0997f`, and `18ef284e`. Use them to refine what physical and coincidence evidence is supposed to establish; do not infer a factor winner from current aggregate counts.
8. **v7 auto-accept `8f152b92`.** The edge choice remains unanimous B, but v8's current policy routes it to human review for low confidence; resolve that policy discrepancy before minting the v7 provenance row.

## Final disposition

**Recommendation: iterate the rubric, then run a targeted v8 successor/non-regression panel. Do not bless this v8 invocation.** Expressibility improved enough that I would not make it the sole blocker, but the residual 38 validated gaps still justify seed work. The decisive blocker is Fix A: it fails `7175635e` and coincides with a +35.6-point NONE increase across the paired cycleway cohort.

**Mint deferred v7+v8 auto-accepts now? No.** Do not mint them as an automatic production tranche until the MI-4 guard is corrected and rerun, `1b90f03b/minimal` and `6775ade1/enriched` are reviewed, and the current human-review routing of v7 auto-accept `8f152b92` is reconciled. The stable subset can be retained as candidates, but not promoted under a wave that failed its explicit non-regression condition.
