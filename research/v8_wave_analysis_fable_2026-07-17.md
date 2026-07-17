# v8 Physical/Coincidence Stitch Wave — Fable Analyst Report (2026-07-17)

Analyst: Claude Fable seat (independent; codex analyst report not read).
Data: `labels/votes/dataset=*/{votes,consensus,evidence}.csv` (v8 = `source_batch` containing
`physical_context_v8_20260717`, v7 rows in the same files as baseline) plus the git-ignored
v8/v7 batch working dirs under `data/agents/stitching/batches/` for R#/T# label maps and
option menus. All numbers below computed from those sources; nothing was modified.

Wave totals: **195 v8 ballots** (65 groups x 3 seats: 50 enriched + 5 controls x 4 variants
x 3 seats), **0 errors/abstains** in the archive, 3 seat-level retries that succeeded on
`attempt=2` (35329743/claude, fb8f359f/claude, b049e0de/codex). v8 consensus: 32 unanimous,
29 majority, 4 none; routing 10 auto_accept / 55 human_review. v7 baseline: 237 ballots,
231 valid, 6 timeout abstains, 85 decisive NONEs.

A structural fact that sharpens every comparison below: **all 50 v8 enriched groups are
exact repeats of v7 groups** (50/50 group_id overlap; 4 v7-only groups dropped). The wave is
fully paired, so v7->v8 deltas are ballot-to-ballot on identical geometry.

---

## Executive summary

- **The exact-pair seeds (#450) partially closed the expressibility gap, and the structured
  contract (#451) worked flawlessly — but the gap did not close overall.** Expressibility-gap
  NONEs fell from ~72% of v7 NONEs (61/85 hand-classified, +4 mixed) to 44.7% of v8 NONEs
  (38/85 via `none_reason`), and per-ballot from ~26-28% to 19.5%. Menus grew from a mean of
  8.3 to 12.0 options on the 50 paired groups, and 5 of the 22 v8 option-consensus picks were
  options that **did not exist on the v7 menu** — including three groups that were NONE
  consensus in v7 (`1b90f03b`->J, `53bb11d7`->I, `c8da4c08`->K).
- **But total NONE rate went UP (36.8% -> 43.6% of ballots), and paired consensus flips ran
  3 NONE->option vs 10 option->NONE.** The seeds fixed old gaps while the new MI-4 gate
  opened a bigger uncertainty channel (below).
- **The `none_reason`/`desired_edges` contract had a 100% clean capture rate**: every one of
  the 85 NONEs carries a valid enum; all 38 `no_exact_option` ballots supplied non-empty
  R#/T# `desired_edges` (400 edge entries, zero raw ids), **all mapped via batch.json with 0
  failures, 0 degenerate sets** (none equals any offered option), and all desired edges lie
  inside the displayed candidate universe. Zero mis-fired NONEs. 13/38 desired sets are only
  **one edge away** from a menu option — the residual gap is combinatorial (large groups,
  menus of 10-19), not conceptual.
- **The Fix-A non-regression check FAILED.** Guard group `7175635e` (fi_helsinki cycleway/
  service merge; v7 unanimous A at conf 0.72/0.93/0.92) flipped in v8 to **majority NONE**:
  claude NONE `insufficient_evidence` 0.68, muse NONE `insufficient_evidence` 0.62, only
  codex held A (0.9). Both NONE seats explicitly cite the new MI-4 painted-vs-separated
  dichotomy and say the pack cannot resolve it.
- **MI-4 over-trigger is systemic, not isolated**: 34/37 `insufficient_evidence` NONEs use
  MI-4/cycleway language and 31/37 cite the painted-vs-separated test verbatim. At least 6 of
  the 10 option->NONE consensus flips are MI-4-uncertainty driven (`7175635e`, `66e22055`,
  `a451bf05`, `b33a27f5`, `5faa0b72`, `92c0997f`) — all were decided (mostly unanimous)
  merges in v7. `insufficient_evidence` NONEs went from ~3 (v7, hand-classified) to 37.
- **The 2x2 says context matters and `minimal` is dangerous**: on `1b90f03b` the panel picked
  the corrected seeded option J (drop the false Harrier Avenue edge e1) in enriched,
  no_physical, AND no_coincidence — but in `minimal` it **unanimously auto-accepted optimizer
  A at mean conf 0.893**, the exact set every v7 seat rejected. Either context factor alone
  suffices (they are substitutes, not additive); with neither, the panel confidently accepts
  wrong optimizer output. `92c0997f` shows the same minimal-flip (majority A at 0.945).
- **Physical metadata mostly moves confidence, coincidence moves decisions**: 7bac1f1d chose
  I in all four variants but no_physical dropped below the auto-accept confidence bar;
  fb8f359f codex flips A<->NONE strictly with coincidence visibility.
- **Per-seat**: claude is the gate-responder (NONE rate 39.5%->55.4%, 21/37 insufficient
  ballots, lowest conf 0.689); codex is flat on NONE rate (46%) but is the most frequent
  dissenter (13 minority stints) at the highest confidence (0.938 mean — poorly calibrated);
  muse is the accept-pole (NONE 29.2%, picks A 19/65, `dissent:muse=A` x6).
- **Disposition (detail in final section): iterate — do not bless the composition yet — but
  mint a curated subset of the deferred auto-accepts now** (9 of 10 v8 + all 6 v7, which
  cross-confirm; withhold `1b90f03b` minimal-A, and adopt the rule that ablation-variant
  ballots never mint).

---

## Goal 1 — Did exact-pair option seeds close the expressibility gap?

### v7 baseline (hand-classified from free text)

85 decisive v7 NONEs (from 231 valid ballots, 36.8%). I read all 85 `reasoning` texts and
classified them:

| v7 NONE class | n | share of NONEs | share of valid ballots |
|---|---|---|---|
| expressibility gap ("no option is exact / wanted set not offered") | 61 | 71.8% | 26.4% |
| mixed gap + insufficient (SA-5 both cited) | 4 | 4.7% | 1.7% |
| genuine reject-all (correct set is empty) | 17 | 20.0% | 7.4% |
| insufficient evidence | 3 | 3.5% | 1.3% |

Representative gap citations: `750ae089`/codex "No option is exact. The long, collinear
corridor-continuity matches e22 (R13->T6) and e5 (R9->T4) should be retained, but B-I omit
them"; `1b90f03b`/codex "The accepted set is all edges except e1... No option contains
exactly those 13 edges"; `b8b5da4a`/muse "Final valid set is 5 edges {e1,e2,e3,e8,e12}. No
option lists exactly that set". Reject-all examples: `d4d2e782`, `422d5d7b`, `4eed5e80`,
`e4746a04` (vertical/role rejects). Insufficient: `35329743`/codex, `5faa0b72`/codex,
`92c0997f`/claude. Mixed: `dd106a0f`/claude, `ca8d1f92`/claude, `3b876df0`/claude,
`e8a39e6d`/claude.

### v8 (structured)

85 NONEs from 195 ballots (43.6%): `no_exact_option` 38 (44.7% of NONEs, 19.5% of ballots),
`insufficient_evidence` 37 (43.5%), `all_edges_no_match` 10 (11.8%).

**v7->v8 delta in expressibility-gap NONE share: -27.1 pts (71.8% -> 44.7%)** (or -31.8 pts
counting the 4 mixed as gap). Per-ballot gap-NONE rate: 26.4% -> 19.5% (-6.9 pts).

### Did #450 actually put the wanted set on the menu?

Direct positive evidence:

- Menus on the 50 paired enriched groups grew from mean 8.3 to 12.0 options.
- 5 of 22 v8 option-consensus choices were **new-in-v8 options absent from the v7 menu**:
  `1b90f03b`->J (gb_london; v7 unanimous NONE wanting "all edges except e1" — J is exactly
  that fix), `53bb11d7`->I (au_sydney; v7 majority NONE), `c8da4c08`->K (us_philadelphia; v7
  unanimous NONE wanting the footway-only set), `35329743`->J (au_sydney; v7 no-consensus),
  `8582dd97`->K (hk_hongkong; v7 unanimous A, refined in v8 to drop the layer-conflicted
  R3->T5 Texaco Road edge).
- Paired transitions confirm three v7 NONE-consensus groups resolved to options, and 4 of the
  6 v7 no-consensus groups reached a consensus in v8.

Countervailing evidence (why this is only a partial success):

- Paired consensus transitions overall: NONE->option 3, no-consensus->option 4, but
  **option->NONE 10** and NONE->NONE 14. Group-level NONE consensus went up, not down.
- The 38 remaining `no_exact_option` ballots concentrate in big groups (desired sets of 2-27
  edges; menus already at 10-19 options, i.e. near the max-menu-20 cap). Min symmetric
  difference between the desired set and the closest menu option: **1 edge for 13/38
  ballots**, 2-4 edges for 17, >=5 for 8. The seeds get close but the cap-4 exact-pair
  seeding cannot cover the combinatorics of 14-19-segment groups.
- Cross-seat convergence inside the residual gap is real and exploitable: on `bdbdf792`
  (nl_amsterdam) **all three seats emitted the identical 12-edge desired set**; on
  `00e8e9fd` (hk) codex+muse identical 7-edge sets; on `fb8f359f` (au_sydney) claude+codex
  identical 8-edge sets. These are effectively unanimous verdicts the option menu could not
  express — a "consensus desired_edges" minting/seeding path would have resolved them.

**Verdict**: #450 demonstrably put previously-missing correct sets on the menu and resolved
marquee v7 gap groups (`1b90f03b` above all). The expressibility-gap share of NONEs dropped
by ~27 points. But the headline hypothesis "the gap is closed" is **not supported**: 38 gap
NONEs remain, the total NONE rate rose, and the residual gap lives in large groups where
option enumeration cannot keep up — the fix for those is consuming `desired_edges`
(as consensus evidence and/or as next-wave seeds), not more menu options.

---

## Goal 2 — Full `none_reason` breakdown (with the R#/T# mapping caveat applied)

All 85 v8 NONE ballots (no abstains exist in the v8 archive; the 3 `attempt=2` rows are the
final successful attempts and are the only rows archived for those seat-groups).

### Distribution

| none_reason | total | claude | codex | muse |
|---|---|---|---|---|
| no_exact_option | 38 | 12 | 16 | 10 |
| insufficient_evidence | 37 | 21 | 10 | 6 |
| all_edges_no_match | 10 | 3 | 4 | 3 |
| **total NONE** | **85** | **36** | **30** | **19** |
| NONE rate (of 65) | 43.6% | 55.4% | 46.2% | 29.2% |

By dataset (enriched unless noted): hk_hongkong 14 (9 neo/2 ins/3 anm) + 2 in variants;
fi_helsinki 12 (2/10/0) + 4 in variants; de_berlin 12 (6/0/6); ch_geneva 12 (1/10/1);
au_sydney 8 (4/4/0) + 5 in variants; nl_amsterdam 7 (6/1/0); gb_london 4 (2/2/0);
us_philadelphia 4 (3/1/0). The `insufficient_evidence` mass sits squarely on the
cycle-heavy datasets (fi_helsinki, ch_geneva) — see Goal 3. The `no_exact_option` mass sits
on the big-junction datasets (hk_hongkong, de_berlin).

### Mapping caveat applied (this is the diagnostic re-run the production capture skipped)

Using each group's `batch.json` `ref_ids`/`target_ids` order (verified against
`stitch_evidence.py::_seg_labels` — R#/T# is a 1-based enumeration of those lists) and the
per-group `evidence.json` option menus:

- **38/38 `no_exact_option` ballots have non-empty `desired_edges`; 400/400 edge entries use
  well-formed R#/T# labels (zero raw source ids); 0 mapping failures** (every label resolves
  to a source id in its group).
- **0 degenerate sets**: no mapped desired set equals any offered option's edge set. So all
  38 are *bona fide* expressibility-gap evidence.
- **All 38 desired sets are subsets of the displayed candidate universe** — no seat invented
  edges outside the shown candidates.
- All 47 `insufficient_evidence`/`all_edges_no_match` ballots correctly left
  `desired_edges` empty. **No `none_reason` value was mis-fired** (no NONE whose wanted set
  was actually on the menu).

The contract compliance rate is therefore 100% across all three seats — the #451 prompt
contract can be trusted as a machine-analyzable channel without hand-reading.

### Character of each bucket

- `all_edges_no_match` (10) is concentrated and stable: unanimous 3-seat reject-alls on
  `d4d2e782` and `422d5d7b` (Berlin layer -1 indoor footway underpasses vs surface roads)
  and `4eed5e80` (Kai Tak Tunnel layer -1 vs surface Kowloon City Road), plus
  `e4746a04`/codex (Geneva parallel cycleway). All four were v7 reject-alls too —
  correct behavior, unchanged.
- `no_exact_option` (38): residual combinatorial gap, see Goal 1.
- `insufficient_evidence` (37): the new, v8-specific phenomenon — see Goal 3.

---

## Goal 3 — Fix-A (`7175635e`) non-regression check: **FAILED**, and the MI-4 gate
over-triggers broadly

### The guard group

`7175635e` (fi_helsinki, R1 unnamed cycleway vs T1-T3 unnamed service segments, coverage
partition summing to ~1.0):

| wave | claude | codex | muse | consensus |
|---|---|---|---|---|
| v7 | A (0.72) | A (0.93) | A (0.92) | **unanimous A** (human_review: class-mismatch) |
| v8 | **NONE / insufficient_evidence (0.68)** | A (0.90) | **NONE / insufficient_evidence (0.62)** | **majority NONE** (dissent:codex=A) |

The correct unanimous cycleway merge did **not** hold: two of three seats flipped to NONE and
the consensus flipped with them. Both flips cite the Fix-A dichotomy verbatim — claude: "If
R1 is a painted/sharrow/flexpost bike lane on the same service-road pavement, it matches the
road... if R1 is a raised or curbed cycle track... it is a separate feature... The pack gives
me nothin[g]"; muse: "Identity hinges on same-pavement vs physically separated. Pack provides
no names, no junction zooms, no lateral-offset/coincidence measurement, no aerial." Only
codex kept the v7 logic ("The cycleway/service class mismatch is non-dispositive given the
strong alignment and coverage partition").

Note the evidence pack did not lose information between waves — the same seats affirmed the
merge on the same geometry in v7. The rubric change alone (era `2026-07-17+9463c80a0f77`,
MI-4 uncertainty gate) caused the flip. Mechanically the gate did what it says (demand
separation evidence, else be uncertain); against its own guard definition, this is an
**over-trigger regression**.

### Wider over-trigger scan

- 34/37 `insufficient_evidence` ballots use MI-4/cycleway/bike language; **31/37 cite the
  painted-vs-separated test explicitly**. Claude accounts for 21 of the 37.
- Paired consensus flips option->NONE where the v8 NONEs are MI-4-uncertainty ballots:
  - `66e22055` (au_sydney): v7 majority B (keep cycleway->WILLIAM painted-lane edge) -> v8
    **unanimous NONE, all three `insufficient_evidence`** (0.62/0.95/0.72).
  - `a451bf05` (ch_geneva): v7 unanimous B -> v8 unanimous NONE (2x insufficient + 1x
    no_exact). claude: "This is exactly the MI-4 bike/pedestrian-facility identity question...
    The pack does not resolve that distinction — T1-T4 carry physical='unknown'".
  - `b33a27f5` (ch_geneva): v7 unanimous A -> v8 majority NONE (claude 0.62 + codex 0.99
    insufficient; muse held A 0.88). claude explicitly lays out both branches and says the
    pack cannot pick one.
  - `5faa0b72` (fi_helsinki): v7 majority H -> v8 unanimous NONE, 3x insufficient.
  - `92c0997f` (fi_helsinki control): v7 majority C -> v8 majority NONE (claude+codex
    insufficient; muse J).
  - `7175635e` as above.
- The other option->NONE flips (`4148382c`, `9f56d71d`, `b049e0de` — no_exact_option pairs;
  `5e31936e` mixed) are expressibility-gap, not MI-4, flips.

**Conclusion**: Fix A did not merely add a guard rail for genuinely ambiguous cycleways; on
this panel it converted at least 6 previously-decided groups (5 of them v7 unanimous or
majority merges) into `insufficient_evidence` NONEs and broke its own named guard. The
uncertainty the gate asks about (painted lane vs curbed track) is **unanswerable from the
current evidence pack** — the pack_feedback channel says so directly (44 items request
separation attributes, 31 request aerial/street imagery). As shipped, the gate converts
decidable-by-convention cases into permanent human-review load unless the pack gains
separation evidence (or the rubric gets a default convention for `physical='unknown'`).

---

## Goal 4 — 2x2 factorial contrasts on the 5 control groups

Per-seat ballots (choice / conf / none_reason abbreviated: ins=insufficient_evidence,
neo=no_exact_option). Consensus in brackets.

### `fb8f359f` (au_sydney — Western Motorway Onramp vs Homebush Bay Drive)

| variant | claude | codex | muse | consensus |
|---|---|---|---|---|
| enriched | NONE/neo 0.52 | NONE/neo 0.87 | A 0.68 | majority NONE (dissent:muse=A) |
| no_physical | NONE/ins 0.72 | NONE/neo 0.87 | A 0.62 | majority NONE (dissent:muse=A) |
| no_coincidence | NONE/ins 0.50 | **A 0.82** | A 0.74 | **majority A** (dissent:claude=NONE) |
| minimal | NONE/ins 0.40 | **A 0.87** | NONE/neo 0.84 | majority NONE (dissent:codex=A) |

Coincidence visibility is decision-relevant for codex: with the same-side coincidence table
present (enriched, no_physical) codex rejects optimizer A per role/coincidence arguments;
with it hidden (no_coincidence, minimal) codex accepts A at high confidence. claude+codex's
enriched desired sets are **identical 8-edge sets** — the v7-era complaint (drop the R1
onramp block) persists and is still not on the 14-option menu (min symdiff 6). muse's
accept-pole A in 3/4 variants but NONE in minimal shows muse is unstable without context.

### `7bac1f1d` (de_berlin)

Unanimous **I** in all four variants. Confidence is the only moving part: claude 0.82/0.74/
0.82/0.76 and muse 0.92/0.84/0.89/0.88 (enriched/no_physical/no_coincidence/minimal); codex
flat 0.96-0.98. Consequence: enriched, no_coincidence, and minimal auto-accepted; the
no_physical cell alone dropped to human_review on `low_confidence` (mean 0.85). Physical
metadata's effect here is purely a confidence lift — but that lift is what clears the
auto-accept bar.

### `92c0997f` (fi_helsinki — Itatuulenkuja; T3 tunnel vs T7 surface service coincidence)

| variant | claude | codex | muse | consensus |
|---|---|---|---|---|
| enriched | NONE/ins 0.55 | NONE/ins 0.91 | J 0.78 | majority NONE (dissent:muse=J) |
| no_physical | NONE/ins 0.52 | NONE/ins 0.96 | A 0.88 | majority NONE (dissent:muse=A) |
| no_coincidence | NONE/ins 0.62 | NONE/neo 0.87 | A 0.86 | majority NONE (dissent:muse=A) |
| minimal | NONE/ins 0.70 | **A 0.96** | **A 0.93** | **majority A** (dissent:claude=NONE, mean 0.945) |

Same pattern as fb8f359f: strip both context channels and two seats confidently accept the
optimizer set that the informed panel considers unresolved (codex's own v7/v8 reasoning:
"every offered set contains e15", conflating the tunnel with the surface service road).

### `1b90f03b` (gb_london — Eastern Avenue; false e1 to coincident Harrier Avenue)

| variant | claude | codex | muse | consensus |
|---|---|---|---|---|
| enriched | J 0.74 | J 0.97 | J 0.86 | unanimous J (human_review: low_confidence) |
| no_physical | J 0.60 | J 0.97 | J 0.86 | unanimous J (low_confidence) |
| no_coincidence | J 0.62 | J 0.97 | J 0.84 | unanimous J (low_confidence) |
| minimal | **A 0.86** | **A 0.96** | **A 0.86** | **unanimous A -> AUTO_ACCEPT (mean 0.893)** |

The single most important factorial cell in the wave. J is the new #450-seeded option
implementing exactly the v7 panel's unanimous desired fix (drop e1). All three seats find it
with either context channel present — physical and coincidence are **substitutes** here (the
e1 reject can be reached via the layer/tunnel route or via the T2-coincident-with-T4 route).
With both removed, all three seats confidently accept optimizer A including the false edge,
and the pipeline **auto-accepted it**. Ironically the correct-choice cells routed to human
review on low_confidence while the wrong-choice cell sailed through.

### `18ef284e` (hk_hongkong)

H majority in all four variants; codex NONE/neo in enriched (0.83), no_physical (0.86), and
minimal (0.93) with 12-14-edge desired sets (symdiff 3-5 from menu), but H in no_coincidence
(0.89). This is codex's characteristic exact-anchor-set perfectionism (see Goal 6), only
weakly modulated by context. claude/muse stable H across all cells.

### Factorial synthesis

- **Is either factor dominant?** Neither factor has a uniform main effect on choices. The
  informative effects are: (a) coincidence flips codex's accept/reject on fb8f359f; (b)
  physical lifts claude/muse confidence on 7bac1f1d (auto-accept-relevant); (c) on 1b90f03b
  the two factors are interchangeable — a pure interaction (only the both-off cell differs).
- **Is `minimal` meaningfully worse than `enriched`? Yes, and in the worst possible way**:
  in 3 of 15 control-cell comparisons (1b90f03b all seats, 92c0997f codex+muse, fb8f359f
  codex) removing context flipped seats from a defensible reject/curated choice to confident
  acceptance of a set the informed panel rejects — including one unanimous auto-accept.
  Context does not merely add confidence; it prevents confidently-wrong merges of
  coincident-but-distinct facilities.

---

## Goal 5 — Does v8 earn blessing, and can the deferred v7+v8 auto-accepts be minted?

### The v8 auto-accepts (10 rows, 8 distinct group-labels)

| dataset | group | variant | choice | mean conf | assessment |
|---|---|---|---|---|---|
| de_berlin | 33a36ca5 | enriched | A | 0.913 | **Mint.** Also v7 unanimous auto-accept A (0.89) — 6 concurring seat-votes across two rubric eras. |
| de_berlin | 3f53c7e7 | enriched | A | 0.903 | **Mint.** Same double-wave unanimity (v7 A 0.907). |
| de_berlin | 7bac1f1d | enriched | I | 0.903 | **Mint.** Unanimous I in all 4 v8 variants + v7 (incl. v7 no_coincidence auto-accept 0.903). Most corroborated label in the wave. |
| de_berlin | 7bac1f1d | no_coincidence | I | 0.897 | Redundant with enriched; fine in content, but should mint as one label from the enriched row. |
| de_berlin | 7bac1f1d | minimal | I | 0.867 | Same. |
| gb_london | 91570f54 | enriched | A | 0.900 | **Mint.** v7 unanimous auto-accept A (0.90); tiny 3-option menu, stable. |
| hk_hongkong | e0099fb8 | enriched | A | 0.953 | **Mint.** v7 unanimous auto-accept A (0.947). |
| nl_amsterdam | 6775ade1 | enriched | E | 0.867 | **Mint with a human spot-check.** v7 was no-consensus (claude NONE wanted a 10-edge set; v8 E has 11 edges and was on the v7 menu unpicked). v8 unanimity is genuine convergence — claude's v8 ballot (E 0.83) resolves its v7 objection via coverage-partition arithmetic — but this is the only auto-accept with a contradicting prior-wave ballot. |
| us_philadelphia | ee358f5a | enriched | B | 0.883 | **Mint.** v7 unanimous B too (routed low_confidence then); two-wave unanimity. |
| gb_london | 1b90f03b | minimal | A | 0.893 | **WITHHOLD.** Contradicts the same panel's unanimous J in enriched/no_physical/no_coincidence and v7's unanimous reject-of-A. A context-blinded artifact of the ablation arm. Includes the false e1 (Harrier Avenue) edge. |

Additional v7 deferred auto-accepts not re-listed above: `8f152b92` (au_sydney, unanimous B
0.877; v8 concurs unanimous B, routed low_confidence) — **mint**; plus v7's `7bac1f1d`
no_coincidence duplicate (covered above).

Net: **mint 9 distinct group labels** (33a36ca5, 3f53c7e7, 7bac1f1d, 91570f54, e0099fb8,
ee358f5a, 8f152b92, 6775ade1 [spot-check], and nothing else), **withhold 1b90f03b(minimal)**,
and adopt the policy that **ablation-variant ballots are experiment data, not labels**.

### Blessing assessment

Reasons v8 does not yet earn a wholesale bless of the candidate panel + rubric era:

1. The Fix-A guard `7175635e` flipped (Goal 3) — the wave's own non-regression criterion.
2. The MI-4 gate converted >=6 previously-decided groups to `insufficient_evidence` NONEs;
   total NONE rate rose 36.8% -> 43.6%; unanimous-NONE human_review routing is now the single
   largest route bucket (14 rows).
3. The `minimal` auto-accept `1b90f03b` shows the auto-accept rule (unanimity + confidence)
   is not variant-aware and can confidently mint context-blind errors.

Reasons v8's *mechanisms* deserve blessing:

1. The #451 contract performed perfectly (100% enum + mappable desired_edges, 0 degenerate).
2. #450 seeds resolved three marquee v7 NONE-consensus groups and supplied the winning
   option in 5 consensus picks.
3. Retry/provenance (#446) eliminated abstains (v7: 6 timeouts; v8: 0, with 3 clean retries).
4. Reject-all behavior on vertical-separation groups is stable and correct (Goal 7).

Recommendation is at the end of the report.

---

## Goal 6 — Per-seat behavior (carried over)

| metric (v8, n=65 each) | claude | codex | muse |
|---|---|---|---|
| NONE rate | 55.4% (36) | 46.2% (30) | 29.2% (19) |
| v7 NONE rate | 39.5% | 46.1% | 25.3% |
| mean confidence | 0.689 | 0.938 | 0.815 |
| v7 mean confidence | 0.616 | 0.933 | 0.800 |
| none_reason mix (ins/neo/anm) | 21/12/3 | 10/16/4 | 6/10/3 |
| agreement with consensus (where one exists) | 52/61 | 52/61 | 50/61 |
| minority stints (route_reason dissents) | 9 | 13 | 11 |
| optimizer-A picks | 8 | 15 | 19 |
| retries (attempt=2) | 2 | 1 | 0 |

- **claude** is the rubric-maximalist and the main MI-4 gate responder: +15.9 pts NONE rate
  v7->v8, 21 of the 37 insufficient_evidence ballots, lowest and most spread confidence.
  Style: long coverage-arithmetic derivations, explicit MI-x citations, frequently lays out
  both branches of the painted-vs-separated dichotomy and refuses to pick one. Its dissents
  are mostly lone NONE-while-others-accept (`ca8d1f92`, `729f879b`, `b7f57035`, `f4f3387b`,
  `92c0997f`/minimal, `fb8f359f`/no_coincidence).
- **codex** is unchanged in NONE rate but is the most frequent dissenter (13) at the highest
  stated confidence — 0.87-0.99 even on NONE ballots — i.e. the least calibrated. Signature
  behavior: exact-set perfectionism over M:N junction anchors (dissenting NONE/neo on
  `18ef284e` in 3 of 4 variants with near-menu desired sets, `35329743`, `4dc33ddd`). It was
  also the only seat to hold the correct(-per-guard) merge on `7175635e`.
- **muse** is the accept-pole: fewest NONEs, most optimizer-A picks, and the
  `dissent:muse=A` pattern (6 rows: fb8f359f x2, 92c0997f x2, b33a27f5, 9f56d71d) where it
  accepts sets the other two reject on role/coincidence grounds. Not a wrecker — 50/61
  consensus agreement — but it systematically under-weights the coincidence/role gates.
- Panel health: zero abstains in v8 vs 6 timeout abstains in v7; the #446 seat-level retry
  path (3 uses) worked every time.

---

## Goal 7 — Frontage-road & vertically-layered ambiguity (carried over)

59 group-variant ballot clusters mention vertical/frontage terms. The physical channel is
doing exactly what it was added for:

- **Coincident-but-unconnected correctly rejected** (stable across both waves, now with
  structured `all_edges_no_match`): `d4d2e782` and `422d5d7b` (Berlin layer -1 indoor
  footway underpasses vs surface Alt-Tempelhof / Karl-Marx-Strasse; unanimous 3x NONE/anm,
  conf up to 0.99), `4eed5e80` (Kai Tak Tunnel layer -1 vs surface Kowloon City Road,
  unanimous NONE/anm). codex on `4eed5e80`: "each aligned R1 portion is the layer -1 Kai Tak
  Tunnel, while T1, T2, and T3 are layer 0 surface segments".
- **Layer evidence driving fine-grained edge exclusions, not blanket splits**: `8582dd97`
  flipped from v7 unanimous A to v8 majority K purely to drop R3->T5 ("T5 is Texaco Road at
  layer 0 while the aligned R3 portion is the layer-1 bridge" — codex 0.93, muse concurs;
  claude dissented A 0.72 arguing layer disagreements are representational). One edge
  removed, 14 kept — no over-splitting.
- **No over-splitting of legitimately continuous layered corridors**: `6775ade1` (IJ-tunnel,
  systematically offset layer scheme between datasets) was **unanimously accepted** (E,
  auto-accept) with claude explicitly reasoning that the offset layer scheme is a
  representation difference (MI-2/MI-5), not identity evidence; `4dc33ddd` accepted A
  including the R7->T5 tunnel edge ("matches layer -1"). The panel distinguishes
  same-facility layer transitions from stacked distinct facilities.
- **Frontage/service ambiguity persists in the no-consensus knot groups**: `750ae089`
  (consensus none; claude feedback: "whether the second parallel blue line through the
  junction is a split carriageway, a ramp, or a frontage/service road"), `1de025b8` (Pacific
  Motorway ramp/mainline/tunnel mix, consensus none), `4dc33ddd`/`92c0997f` service-road vs
  cycleway/tunnel coincidences. These need junction zooms or carriageway attributes, not
  more rubric.

## Goal 8 — Disagreement map + pack_feedback synthesis (carried over)

33 of 65 group-variant panels were non-unanimous (29 majority + 4 no-consensus:
`750ae089`, `1de025b8` [C vs NONE vs B], `dc818edc` [NONE vs B vs J], `7680bb19`
[A vs I vs NONE] — all enriched, all big-junction or duplicated-corridor groups).

Fault lines (from `route_reason`):

- **muse-vs-rest accept axis** (muse picks an option, others NONE): 9 rows — the 6
  `dissent:muse=A` plus muse=J (`92c0997f`), muse=B (`4148382c`), muse=G (`5e31936e`).
- **codex-vs-rest exactness axis** (codex NONE/neo while others pick an option): 5 rows
  (`18ef284e` x3, `35329743`, `4dc33ddd`) plus codex=A holdout on `7175635e`.
- **claude-vs-rest caution axis** (claude NONE while others pick): 6 rows (`ca8d1f92`,
  `729f879b`, `b7f57035`, `f4f3387b`, `fb8f359f`/no_coincidence, `92c0997f`/minimal).
- Splits are **not** concentrated in ablated variants (8 of 33 split rows are in the 15
  ablation cells, proportional to their share); the ablation-specific pathology is
  confident *wrong unanimity* (Goal 4), which is worse.

`pack_feedback` is fully structured in v8 (missing_info/ambiguities/confidence_basis on all
195 ballots; 925 items). Recurring themes, ranked:

1. **Missing separation/physical attributes and names** (229 items) — e.g. `66e22055`/claude
   "no physical separation attribute for R1 (painted lane vs raised/kerbed cycle track)".
2. **Vertical-layer clarity** (194) — e.g. `66e22055`/muse "T1 secondary and T2 motorway
   tunnel share XY for 142m (vertical stack)".
3. **Junction zooms** (148) — the most actionable render gap: `750ae089`/claude "no junction
   zoom for the dense ... bridge cluster; labels overlap illegibly".
4. **Lateral-offset / coincidence metrics for ref-vs-target pairs** (125) — the coincidence
   table currently covers same-side pairs within one dataset; seats keep asking for the
   ref-target lateral offset (`66e22055`/claude: "coincidence context is only given T1 vs
   T2").
5. **One-way/carriageway/direction attributes** (108).
6. **Painted-vs-separated cycle facility type** (44) and **aerial/street imagery** (31) —
   the direct MI-4 unblocking asks.
7. **Menu/option gaps** (29) — e.g. `750ae089`/muse: "R1 20% + R11 66% would fully cover T5
   but no 8-edge option includes R11".

---

## Prioritized human-review list

1. **`7175635e` (fi_helsinki)** — adjudicate the Fix-A guard: either re-affirm the merge and
   soften the MI-4 gate (default convention when `physical='unknown'` + strong coverage
   partition), or formally retire the guard. Everything about the gate's disposition follows
   from this call.
2. **`1b90f03b` (gb_london)** — confirm J (all edges except e1) as the production label and
   formally withhold the minimal-variant auto-accept of A; use it as the canonical example
   for variant-aware minting policy.
3. **The MI-4 flip set: `66e22055`, `a451bf05`, `b33a27f5`, `5faa0b72`, `92c0997f`** — each
   had a v7 option consensus and now sits in NONE/insufficient. Human labels here double as
   calibration data for whichever MI-4 revision ships.
4. **`bdbdf792` (nl_amsterdam)** — all three seats emitted the identical 12-edge desired
   set; also `00e8e9fd` (codex+muse identical) and `fb8f359f` (claude+codex identical).
   Cheap, high-confidence labels waiting to be harvested; also the test cases for a
   desired-edges consensus/seeding path.
5. **`6775ade1` (nl_amsterdam)** — spot-check option E before minting (only auto-accept with
   a contradicting prior-wave ballot).
6. **`8582dd97` (hk_hongkong)** — verify the K-vs-A layer question (Texaco Road R3->T5).
7. **The 4 no-consensus knots: `750ae089`, `1de025b8`, `dc818edc`, `7680bb19`** — genuinely
   hard junction clusters; also the strongest justification for junction-zoom renders.

## Disposition recommendation

**Iterate the rubric (targeted), do not bless the v8 rubric era wholesale — but bless the
v8 mechanisms and mint the curated auto-accepts.**

- **Bless as-is**: the #451 none_reason/desired_edges contract (100% clean), #450 exact-pair
  seeds (kept every v7 win, added 5 winning options), #446 retries (0 abstains), and the
  physical/coincidence enrichment itself (the 2x2 shows it prevents confident wrong accepts;
  run all future production waves enriched-only).
- **Iterate before the next wave**: (a) revise the MI-4 uncertainty gate — it flipped its own
  non-regression guard and 5 more previously-decided groups into `insufficient_evidence`;
  either supply separation evidence in the pack (attributes/imagery/lateral-offset metric)
  or give the gate a default convention for `physical='unknown'` with strong coverage
  partition; (b) make auto-accept variant-aware (enriched-only minting) so a context-blinded
  unanimity like `1b90f03b`/minimal can never mint; (c) add a consensus-desired-edges path
  (route agreeing desired sets to a confirm ballot or human one-click) to attack the 38
  residual expressibility NONEs that menu enumeration cannot reach.
- **Not "fix expressibility" as the headline**: the expressibility fix largely worked
  (-27 pts share of NONEs); the dominant new problem is rubric-induced uncertainty, not the
  menu.

**Minting the deferred v7+v8 auto-accepts: YES, with two exclusions.** Mint the 9 distinct
group labels — `33a36ca5`(A), `3f53c7e7`(A), `7bac1f1d`(I), `91570f54`(A), `e0099fb8`(A),
`ee358f5a`(B), `8f152b92`(B, v7), `6775ade1`(E, after a quick spot-check), with the
variant-duplicate `7bac1f1d` rows collapsed into the enriched label. Every one of these
except `6775ade1` is unanimous in **both** waves under two different rubric eras — that is
stronger evidence than a single calibrated panel would provide. **Withhold `1b90f03b`
minimal-A** (contradicted by the same panel's informed cells and by v7), and record the
policy that ablation-variant ballots are experiment data, never labels.
