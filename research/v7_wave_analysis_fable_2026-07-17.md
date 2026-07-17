# v7 Physical/Frontage Stitch Wave — Findings (Claude Fable analyst, 2026-07-17)

Analyst: Claude Fable seat (independent; Codex analyst's output not consulted).
Data: `data/agents/stitching/batches/*physical_context_v7_20260715*/{votes,consensus}.csv`
(23 batch dirs), cross-checked against `labels/votes/dataset=*/votes.csv`
(`source_batch` filter). All numbers below were recomputed from raw CSVs.

**Data-scope note:** the task brief said "282 ballots"; both the batch dirs and
the archived `labels/votes` copies contain exactly **195 ballots** (65 schedule
rows × 3 seats; `votes.partial.csv` files are byte-identical snapshots, no extra
rows). The 5 control groups are a subset of the 50 unique groups (50 enriched
rows + 5×3 ablation rows = 65). There are **zero errors and zero abstains** —
every seat returned a decisive ballot on every pack. All 74 `NONE` ballots are
decisive verdicts, none has an `abstain_reason`.

---

## Executive summary

1. **The enrichment works, in the hypothesized direction.** On the 5 control
   groups (15 ballots per cell), NONE/reject rates by 2×2 cell: **enriched 8/15
   (53%), no_coincidence 8/15 (53%), no_physical 5/15 (33%), minimal 2/15
   (13%)**. Physical evidence is the dominant factor; every one of the 14
   ballot-level flips between cells moved toward *accepting more merges* as
   context was removed — never the reverse.
2. **`minimal` is not just noisier — it confidently endorses over-merges.**
   `fb8f359f` (Sydney, Homebush Bay onramp/arterial) flipped from *unanimous
   NONE* (enriched) to *unanimous Option A* (minimal, mean conf 0.763, saved
   from auto-accept only by the low-confidence gate). `1b90f03b` (London,
   Harrier Ave coincident with the Eastern Ave tunnel tile) flipped from
   unanimous NONE to majority A. The minimal ballots themselves admit blindness
   ("physical=unknown for all segments" — claude/minimal/fb8f359f pack_feedback).
3. **But the headline problem is not evidence — it's the option menu.** 61/74
   NONE ballots (82%) are wholly or partly "**no offered option is exact**"
   (expressibility gap); only 11 (15%) are genuine reject-alls and 2 (3%) pure
   insufficient-evidence. In the flagship case (`1b90f03b`) all three seats
   agree the correct set is "all edges except e1" across nearly all variants —
   and that set is simply not on the menu.
4. **Enriched panel outcomes are review-heavy:** on the 50 enriched groups, the
   panel-level choice is NONE in **22/50 (44%)**; only **5/50 (10%)**
   auto-accepted. The optimizer's Option A survives as panel choice in only
   11/44 majority-bearing enriched groups (25%).
5. **Seats have sharply different NONE thresholds:** codex 32/65 NONE (49%),
   claude 27/65 (42%), muse 15/65 (23%). Codex is an "exact-set purist" (28/32
   of its NONEs are expressibility), muse is a "menu pragmatist" that picks the
   best available letter (A-rate 38.5% vs claude 18.5%) and repeatedly
   rationalizes away layer conflicts when target physical attrs are unknown.
6. **Codex confidence is uninformative for routing:** mean 0.928, and 0.927
   even on NONE ballots (claude: 0.625/0.578). Any mean-confidence gate is
   dominated by codex's near-constant 0.9+.
7. **Unanimous NONE ≠ agreement.** On `fb8f359f`/enriched the three seats'
   NONEs encode three *different* correct sets (claude: A minus e10; codex:
   drop the whole R1-onramp block; muse: unresolved duplicates). A `NONE`
   consensus can mask real disagreement about ground truth.
8. **Vertical-layer cases are where enrichment shines brightest:** the two
   Berlin pedestrian-underpass groups (`d4d2e782`, `422d5d7b`) got unanimous,
   high-confidence (0.90–0.92) genuine reject-alls citing `layer -1; tunnel;
   indoor` explicitly. Legit continuous corridors still auto-accepted
   (`e0099fb8`, `3f53c7e7`, `33a36ca5`, `91570f54`, `8f152b92`) — no evidence of
   systematic over-splitting, with one flagged exception (`92c0997f`).
9. **One control is a caution, not a win:** `92c0997f` (Helsinki) is the only
   control where enrichment changed the panel's *letter* (A→C, deduping R1/R2);
   if R1/R2 are genuinely dual carriageways, enrichment caused an over-split.
   Needs human adjudication before citing it as a success.
10. **Recommendation (detail in final section): do not bless v7 for label
    minting yet; keep the enriched pack format, fix option-menu expressibility
    first, then a targeted rubric iteration.** The 2×2 validates the evidence
    change; the NONE forensics show the binding constraint is the menu.

---

## Goal 1 — Per-seat behavior

195 ballots, 65 per seat. No abstains, no errors.

| Seat | model | NONE rate | mean conf | conf on NONE | A-rate | agree w/ majority* | lone-dissenter count | mean latency | reasoning length (med chars) |
|---|---|---|---|---|---|---|---|---|---|
| claude | claude-opus-4-8 | 27/65 (41.5%) | 0.625 (min 0.40, max 0.90) | 0.578 | 18.5% | 50/59 (84.7%) | 9 | 134.7 s | 778 |
| codex | gpt-5.6-sol | 32/65 (49.2%) | 0.928 (min 0.72, max 0.99) | 0.927 | 23.1% | 47/59 (79.7%) | 12 | 153.6 s | 415 |
| muse | meta/muse-spark-1.1 | 15/65 (23.1%) | 0.801 | 0.791 | 38.5% | 47/59 (79.7%) | 12 | 109.0 s | 944 |

\* over the 59 rows where a majority exists.

Pairwise choice agreement: claude–codex 58.5%, claude–muse 58.5%, codex–muse
53.8%.

Characteristic styles (consistent across dozens of ballots):

- **claude** — longest deliberation per token of output, systematic MI-x/SA-x
  rule citation, explicit alternative-set construction ("the correct set is
  {…}, no option offers it"), *lowest and best-spread confidence* (only seat
  whose confidence visibly tracks difficulty; its 0.42 on `e8a39e6d` is the
  wave minimum, on the genuinely hardest group). Dissents split between
  letters (6) and NONE (3).
- **codex** — tersest, states an exact accepted edge list almost every time,
  near-constant 0.9+ confidence *including on NONE* — it treats confidence as
  "confidence my analysis is right", not "confidence the panel will agree".
  8 of its 12 lone dissents are NONE: it is the systematic exactness dissenter
  (e.g. `18ef284e` in all four variants, `66e22055`, `92c0997f`, `dc818edc`,
  `5faa0b72`, `00e8e9fd`…).
- **muse** — longest text, heaviest use of coverage arithmetic
  (tgt_aln sums to 1.0 etc.), most menu-pragmatic: 11 of its 12 lone dissents
  are *letters*, usually A/B against a claude+codex NONE pair (`e4746a04`,
  `4eed5e80`, `00e8e9fd`, `bdbdf792`, `750ae089`…). It repeatedly resolves
  MI-2-vs-MI-4 tension in favor of merging ("class difference … is evidence
  only per MI-5, not a role override" — `e4746a04`), and discounts physical
  mismatch when the target side lacks attributes ("missing tunnel flag is
  unknown, not proof of surface" — `4eed5e80`, where it voted A@0.65 against
  two layer-based reject-alls).

No seat is a pathological outlier by majority-agreement (80–85% for all
three), but the *direction* of dissent is seat-specific: codex dissents toward
NONE, muse toward merge.

## Goal 2 — enriched vs no_physical (physical-evidence effect)

Ballot-to-ballot on the 5 controls (15 pairs). **4/15 choices changed**, all
toward merging; mean confidence delta −0.023 (negligible).

| group | seat | enriched → no_physical |
|---|---|---|
| `fb8f359f` | muse | NONE → A |
| `1b90f03b` | muse | NONE → G |
| `92c0997f` | codex | NONE → A |
| `92c0997f` | muse | C → A |

Reading the flipped ballots confirms causality: codex's enriched NONE on
`92c0997f` hinges entirely on physical evidence — "T3 is the tertiary *tunnel*
and T7 is a *ground-level* service road … accepting e15 would conflate the
tunnel route with the separate surface service feature"; with physical off it
writes "class differences do not outweigh the continuous same-name geometry"
and endorses A@0.94. Muse's `fb8f359f` enriched NONE cites the e7/e10
layer-0-vs-bridge mismatches; without them it reconstructs the merge case.
Claude and codex on `fb8f359f`/`1b90f03b` held NONE without physical because
the coincidence context alone still exposed the duplicates (see Goal 4).

Panel-level choice changed on **1/5 groups** (`92c0997f` C→A); `fb8f359f` and
`1b90f03b` stayed NONE (though `fb8f359f` degraded unanimous→majority).

## Goal 3 — enriched vs no_coincidence (coincidence effect)

**3/15 choices changed**, mean confidence delta −0.001. All three changes are
on the same group, `92c0997f`:

| group | seat | enriched → no_coincidence |
|---|---|---|
| `92c0997f` | claude | C → NONE ("all 9 option images look identical … evidence insufficient") |
| `92c0997f` | codex | NONE → A |
| `92c0997f` | muse | C → A |

On `fb8f359f`, `1b90f03b`, `18ef284e`, `7bac1f1d`: zero ballot changes —
with physical evidence still present, removing the same-side-coincidence tables
changed nothing. Coincidence context matters mainly when it is the *only*
duplicate-detection signal (see the no_physical column, where it kept
claude+codex at NONE on `fb8f359f`/`1b90f03b`).

## Goal 4 — Full 2×2 interaction

NONE ballots per cell (of 15), controls only:

| | coincidence ON | coincidence OFF |
|---|---|---|
| **physical ON** | enriched: **8** | no_coincidence: **8** |
| **physical OFF** | no_physical: **5** | minimal: **2** |

- **Physical is dominant** (+20 to +40 pts of reject rate); coincidence adds
  nothing when physical is on (8→8) and +20 pts when physical is off (2→5).
  Sub-additive, "OR-like" redundancy on the ramp/duplicate groups: on
  `fb8f359f` and `1b90f03b`, *either* factor alone keeps the panel choice at
  NONE; only removing both flips the panel (to A on both).
- **`92c0997f` is "AND-like"**: removing *either* factor flips the panel from
  C to A — the dedup verdict needed both the T3/T7 coincidence table and the
  tunnel/ground contrast.
- **Two controls are invariant**: `7bac1f1d` (unanimous I in all four cells,
  conf 0.85–0.90) and `18ef284e` (claude/muse H, codex NONE, in all four).
  Important caveat on `7bac1f1d`: it looks like an ablation-robust
  vertical-layer win, but the tunnel identity **leaks through the street
  name** — even the minimal ballots exclude R5/R6 because they are named
  "**Mäusetunnel**" (muse/minimal: "R5 footway (Mäusetunnel)…"). It is *not*
  evidence that panels resolve layering without physical attributes.
- **Choice changes vs enriched**: no_physical 4/15, no_coincidence 3/15,
  minimal 7/15 ballots; panel-level choice changes 1/5, 1/5, 3/5 groups.
  `minimal` is meaningfully worse than `enriched`: it produced confident
  consensus merges (`fb8f359f` unanimous A@0.763; `1b90f03b` majority A) on
  the two groups whose enriched ballots identify concrete physical/coincidence
  disqualifiers.
- **Confidence is flat across cells** (panel mean 0.776–0.807; per-seat deltas
  ≤0.036). Context changes *choices*, not stated confidence — confidence
  cannot be used to detect an under-informed pack.
- Within-panel coherence: splits on controls = enriched 2/5, no_coincidence
  2/5, minimal 3/5, no_physical 4/5 — enrichment also modestly increases
  agreement.

**Honest caveat:** we have no adjudicated ground truth for the 5 controls. The
claim "enriched is better" rests on (i) the flips being monotone toward merge
as evidence is hidden, (ii) enriched ballots citing concrete, checkable
disqualifiers (layer conflicts, 88–100% same-side overlaps) while minimal
ballots explicitly note the missing attributes, and (iii) the experiment's
prior that over-merging is the dominant failure mode. `92c0997f` is the one
control where the enriched outcome (C, deduping R1/R2) could itself be the
error; several ballots on both sides flag the dual-carriageway reading as
plausible (claude/minimal: "R1/R2 cross rather than stay parallel, unusual for
carriageways"). Human adjudication of the 5 controls would convert this from
"consistent with hypothesis" to "confirmed".

## Goal 5 — Frontage/service-road & vertically-layered ambiguity

Prevalence (keyword tagging of `reasoning`+`pack_feedback`): vertical-layer
vocabulary (bridge/tunnel/overpass/underpass/layer/elevated/flyover…) appears
in ballots of **41/50 groups** (125/195 ballots); frontage/service/ramp
vocabulary in 20/50 groups; same-side-coincidence/duplicate vocabulary in
37/50. The wave was well-targeted at its two themes.

**Layered roads — enrichment steers away from false merges:**

- `422d5d7b`, `d4d2e782` (Berlin): pedestrian underpass (`layer -1; tunnel;
  indoor`) vs surface roads. All six ballots are genuine reject-alls citing
  the physical attrs verbatim; unanimous NONE at conf 0.897/0.920 — the
  highest-confidence NONEs in the wave. (codex, `422d5d7b`@0.99: "Both edges
  fail physical identity … a vertically separated overlap, not the same
  traveled way.")
- `4eed5e80` (HK): Kai Tak Tunnel (layer −1) under Kowloon City Road (layer 0).
  claude+codex reject-all on layer grounds (codex@0.99); **muse votes A@0.65**,
  arguing target physical "unknown, not proof of surface" — majority NONE. The
  correct anti-merge outcome survives, but only 2:1.
- `66e22055` (Sydney): surface cycleway vs Cross City Tunnel — majority B
  (keep only the surface match, drop the tunnel edge; codex reject-all
  dissent). Anti-merge with the legitimate edge retained.
- `3b876df0` (Berlin, Tunnel Tiergarten stack): claude+codex NONE because
  *every* trimmed option still keeps one cross-layer edge (e4 surface→tunnel)
  — a menu problem on a layering group.
- `35329743` (Sydney bridge layers): 3-way split (E/NONE/A) — codex explicitly
  cannot resolve whether T2 is "an abstract representation of both roads or
  corresponds only to the elevated R5 facility". Layering ambiguity that
  enrichment surfaced but could not resolve.

**Frontage/service/ramp — same direction:**

- `fb8f359f` (control; onramp vs arterial): enriched unanimous NONE vs minimal
  unanimous A — the cleanest demonstration (Goals 2–4).
- `92c0997f` (T7 service road vs T3 tunnel), `e085519d` (Baana cycleway vs
  pedestrian street), `b8b5da4a` (footways stacked on Cleverly St roadway),
  `c8da4c08`/`17053a69` (Philadelphia: residential road vs sidewalk footways):
  in each, at least two seats used role+coincidence to exclude the
  coincident-but-distinct facility; panel choices were NONE (expressibility)
  rather than the optimizer's merge-heavy A.
- `e4746a04` (Geneva, primary road vs consistently offset parallel cycleway):
  claude+codex genuine reject-all (codex@0.99: "neighboring facilities, not
  representation differences"); muse A@0.73. Majority NONE — anti-merge holds.

**Over-splitting check (the other half of the hypothesis):** legitimately
continuous corridors still merged fine — 5 enriched auto-accepts (`8f152b92` B,
`33a36ca5` A, `3f53c7e7` A@0.907, `91570f54` A, `e0099fb8` A@0.947, the last
explicitly waving through "partial bridge/layer evidence" as segmentation
detail), plus unanimous non-A merges (`7175635e`, `b33a27f5`, `8582dd97` A).
I found **no case** where a seat used physical/coincidence context to reject a
plainly continuous same-facility corridor. The one candidate over-split is
`92c0997f` (panel C dedups R1/R2, which may be genuine dual carriageways) —
flagged for human review rather than counted either way.

**Verdict on the hypothesis: supported.** Physical/coincidence context moves
the panel away from merging spatially-coincident-but-unconnected features, is
causally traceable in flipped ballots, and does not induce visible
over-splitting — with the `92c0997f` caveat.

## Goal 6 — NONE forensics

74 NONE ballots (claude 27, codex 32, muse 15). Flavor classification
(regex triage + hand-reading of every non-obvious ballot):

| Flavor | count | share | per seat (cl/cx/mu) |
|---|---|---|---|
| (b) **no exact option offered** (expressibility gap) | 51 | 69% | 15 / 25 / 11 |
| (b+c) expressibility with an insufficiency component | 10 | 14% | 8 / 0 / 2 |
| (a) genuine reject-all (correct set is empty) | 11 | 15% | 4 / 5 / 2 |
| (c) pure insufficient evidence | 2 | 3% | 0 / 2 / 0 |

- The 11 genuine reject-alls concentrate on 4 groups: `d4d2e782` ×3,
  `422d5d7b` ×3 (Berlin underpasses), `4eed5e80` ×2, `e4746a04` ×2,
  `66e22055` ×1. These are *correct uses* of NONE-as-verdict; note the seats
  still phrase them as "no option represents the empty set", i.e. even
  reject-all is expressed through the menu lens.
- The 2 pure insufficiency NONEs are both codex (`5faa0b72`@0.86 "the evidence
  is insufficient to determine an exact final set"; `35329743`@0.72), plus
  claude's `92c0997f`/no_coincidence ("evidence insufficient … NONE for human
  review") inside the mixed bucket. **These are semantically abstentions
  wearing a NONE costume** — with a `none_reason` enum they would be separable
  from decisive rejects.
- **82% of NONEs (61/74) are menu-driven.** Recurring exact patterns:
  - *"drop exactly one bad edge" not offered*: `1b90f03b` (all seats, most
    variants: correct set = all-except-e1; option G drops e1 but also the
    valid anchors e6/e8). `dc818edc` (e24-vs-e13 bundling).
  - *strong edges missing from every option*: `750ae089` (claude: e22 conf
    0.99 covering 81% of T6 "appear[s] in NO offered option"); `e085519d` (all
    three seats: options B–I keep only the weakest of six valid Baana→T7 edges);
    `b049e0de` (every clean option drops e6/e7, conf 0.989); `cd320a3c` (only
    over-inclusive A contains the required e28); `b7f57035` (every clean
    option omits e10, conf 0.99).
  - *anchor-granularity disputes*: `18ef284e` — codex NONE in all four
    variants because H omits boundary anchors e3/e15/e18 that its
    complementary-fraction analysis requires.
- **Direct implication for the planned `none_reason` enum**: the three flavors
  are reliably inferable from free text today, so an enum will be
  well-populated; and exact-pair option generation (or a "submit your edge
  set" channel) would convert the majority of current NONEs into decisive,
  consensus-bearing letters. In at least 4 groups (`1b90f03b`, `e085519d`,
  `c8da4c08`, `17053a69`) all three seats state essentially the *same* correct
  set — those are unanimous labels lost to menu shape.

## Goal 7 — Disagreement map

65 rows: 26 unanimous (16 letter + 10 unanimous-NONE), 33 majority (2-1),
6 no-majority. Routing: 6 auto_accept, 59 human_review (of which 7
low-confidence unanimous, 3 class-mismatch unanimous, 10 unanimous-NONE).

**Three-way splits (all 6 in enriched — none in ablated variants):**

| group | dataset | ballots |
|---|---|---|
| `35329743` | au_sydney | claude E@0.5 / codex NONE@0.72 / muse A@0.62 |
| `dc818edc` | de_berlin | claude D@0.52 / codex NONE@0.78 / muse B@0.68 |
| `4dc33ddd` | fi_helsinki | claude NONE@0.53 / codex F@0.76 / muse A@0.68 |
| `7680bb19` | fi_helsinki | claude A@0.6 / codex I@0.95 / muse NONE@0.84 |
| `6775ade1` | nl_amsterdam | claude NONE@0.5 / codex E@0.9 / muse A@0.8 |
| `b7f57035` | us_philadelphia | claude NONE@0.55 / codex A@0.97 / muse D@0.78 |

Common shape: everyone agrees on a large core edge set and splits on 1–3
sliver/anchor/duplicate edges the menu forces into bundles (`7680bb19` is
literally a 3-way split over the single 2.2m edge e9). These are
menu-resolution failures more than judgment failures.

**2-1 fault lines** (33 majority rows; dissenter: muse 12, codex 12, claude 9):

- codex dissents = 8 NONE + 4 letters → exactness purism (see `18ef284e`,
  `66e22055`, `00e8e9fd`, `92c0997f`).
- muse dissents = 11 letters + 1 NONE, characteristically A against a
  claude+codex NONE on coincidence/layer groups (`e4746a04`, `4eed5e80`,
  `bdbdf792`, `00e8e9fd`) → merge-side bias.
- claude dissents = mixed (3 NONE, 6 letters), usually on genuinely ambiguous
  anchor choices; its dissenting confidences are the lowest (0.42–0.7).

**Splits vs variant:** splits are *not* concentrated in ablated variants in
absolute terms (24 of 33 majority rows are enriched — but enriched is 50/65
rows). On the controls themselves, non-unanimous outcomes: enriched 2/5,
no_coincidence 2/5, minimal 3/5, no_physical 4/5 — ablation mildly *increases*
splitting. All 6 total-breakdowns are enriched, but on the wave's hardest
unique groups, which have no ablated counterparts to compare against.

## Goal 8 — pack_feedback synthesis

All 195 ballots carry structured JSON pack_feedback (0 parse failures):
388 `missing_info` items, 382 `ambiguities` items. Theme counts:

| Theme | missing_info | ambiguities | notes |
|---|---|---|---|
| Junction zooms (more/better/targeted) | **154 (40%)** | 14 | claude 81, muse 47, codex 26. By far the top ask; specific spots named (merge points, portals, boundary anchors). |
| Layer/physical attrs unknown or incomplete | 93 | 87 | esp. target side: `us_philadelphia`/`au_sydney` targets carry only is_bridge/is_tunnel, no `level` (per batch.json `target_physical_capabilities`), which is exactly the hole muse's `4eed5e80` A-vote drives through. |
| Direction/heading/oneway data | 46 | — | wanted for carriageway and ramp-divergence calls. |
| Carriageway-separation evidence | 43 | 28 | "dual carriageway vs duplicate digitization" is the single most repeated ambiguity (`92c0997f`, `7680bb19`, `9f56d71d`…). |
| Name/class semantics | 25 | 67 | name-change-at-junction vs different-road (`1b90f03b` "Harrier Avenue"), class-tag reliability. |
| Endpoint connectivity/topology | 34 | 36 | requests for node-level connectivity to distinguish touch vs continuation. |
| **Option images identical/indistinguishable** | — | 18 | 26 ballots across 17 groups complain the option PNGs are visually identical because diffs are 1–3 small edges (codex, `92c0997f`/minimal: "Many option images are pixel-identical because they visualize active segments rather than individual accepted edges"). |
| Option-menu gaps stated outright in feedback | 5 | 2 | usually phrased as "an option that drops only e1 is not offered" — undercounts the 61 menu-driven NONEs, which express it in `reasoning` instead. |

Rubric ambiguities that recur verbatim across seats and should be addressed in
the next revision:

1. **MI-2 vs MI-4 precedence for same-side coincidence** — when two
   representations share a centerline, is the instruction "pick one" (claude/
   codex reading) or "abstract centerline may match each constituent" (muse
   reading)? This single ambiguity explains most muse merge-side dissents.
2. **Junction-anchor vs endpoint-clip rule** — complementary tgt_aln fractions
   (0.835+0.165=1.0) are treated as proof of a legitimate straddle by codex,
   as droppable clips by muse in some variants, case-by-case by claude
   (`18ef284e` e3/e15/e18; `1b90f03b` e6/e8; `7680bb19` e9). Needs a numeric
   or explicit rule.
3. **Unknown physical attributes default** — is `physical=unknown` on one side
   evidence-absent (muse: can't reject) or does the attributed side dominate
   (claude/codex)? Decide and write it down.
4. **NONE semantics under menu failure** — seats currently overload NONE for
   "menu is wrong"; the rubric should split verdict-NONE from
   menu-gap-NONE once `none_reason` exists.

---

## What a human should review first (prioritized)

1. **Adjudicate the 5 controls, especially `92c0997f`** (fi_helsinki; the only
   control where enrichment changed the panel letter, A→C — dual-carriageway
   vs duplicate R1/R2 decides whether enrichment prevented an over-merge or
   caused an over-split) and **`fb8f359f`** (au_sydney; unanimous-A-under-
   minimal vs unanimous-NONE-under-enriched is the wave's central causal
   claim — confirming the enriched verdict certifies the whole experiment).
2. **`1b90f03b`** (gb_london): flagship expressibility case — all seats say
   "all except e1"; verify and use it as the acceptance test for exact-pair
   option generation.
3. **The 6 no-majority groups** (`35329743`, `dc818edc`, `4dc33ddd`,
   `7680bb19`, `6775ade1`, `b7f57035`) — all enriched, all human_review, all
   hinge on 1–3 sliver/duplicate edges; cheap to adjudicate and they seed the
   anchor-rule rubric fix.
4. **Menu-construction bugs**: `750ae089` and `e085519d` — option sets that
   exclude conf-0.99, high-coverage edges from *every* option while retaining
   weaker alternatives. If the option generator prunes by edge count or graph
   cut, these two groups are the repro cases. Also `3b876df0` (every trimmed
   option keeps a cross-layer edge).
5. **Muse's merge-side dissents on layered/coincident groups** (`4eed5e80`,
   `e4746a04`, `bdbdf792`, `00e8e9fd`): decide whether these are seat error or
   defensible MI-2 readings; feeds both the rubric fix and any seat-weighting
   decision.
6. **`18ef284e`** (hk): codex's variant-invariant NONE over boundary anchors —
   adjudicating e3/e15/e18 settles the anchor-granularity rule.
7. **`e8a39e6d`** (gb_london, Westway/Harrow Road flyover): the wave's hardest
   pack (claude 0.42, wave minimum), 38-edge multi-level interchange with an
   unresolved T6/T9 coincidence — a stress-test candidate for the next pack
   format.

## Recommendation on v7 disposition

**Fix option-menu expressibility first; keep the enriched pack format while
doing so; a focused rubric iteration rides along; do not bless for production
label minting yet.**

Reasoning:

- **The evidence change should be kept.** The 2×2 is unambiguous about
  direction: hiding physical evidence monotonically converts rejections of
  coincident/ramp/layered merges into confident endorsements (53%→13% reject
  rate enriched→minimal; every one of 14 ballot flips toward merge), and
  minimal came within one routing gate of consensus-accepting `fb8f359f`
  Option A. Coincidence context is a cheaper partial substitute (holds 2 of 3
  seats when physical is off) but not sufficient alone on `92c0997f`. Nothing
  in the data argues for shipping `no_physical`, `no_coincidence`, or
  `minimal`.
- **But blessing v7 as-is would mint very little and review-flood the rest:**
  10% auto-accept, 44% panel-NONE on enriched groups. The binding constraint
  is not evidence quality or seat quality — it is that **82% of NONE ballots
  are the menu failing to offer the set the seat (often all three seats)
  wants**. Exact-pair options (at minimum: "optimizer set minus each
  single flagged edge", plus the seats' stated sets when parseable) or a
  free-form edge-set ballot would convert several current unanimous-NONE
  groups (`1b90f03b`, `e085519d`, `c8da4c08`, `17053a69`) into unanimous
  letters immediately.
- **Concurrent, cheap fixes** surfaced by this wave: (i) implement
  `none_reason` enum (a/reject-all, b/no-exact-option, c/insufficient) — the
  flavors are already reliably inferable from free text, and flavor (c) NONEs
  from codex are abstentions in disguise; (ii) rubric clarifications #1–#3
  above (coincidence precedence, anchor rule, unknown-physical default);
  (iii) per-option diff-highlighted images (17 groups drew "options look
  identical" complaints); (iv) fill the target-side `level` capability gap
  where the source data has it; (v) revisit whether codex's near-constant
  0.93 confidence should be recalibrated or down-weighted in
  mean-confidence routing.
- **Panel composition** itself looks serviceable: three genuinely diverse
  failure modes (exact-set purist / calibrated hedger / merge-lean
  pragmatist), 80–85% majority agreement each, no seat dominates dissent. I
  see no data-driven case for a seat change; the muse merge bias is better
  attacked through rubric clarification #1 and #3 than through replacement.
- **Before promotion**, re-run a small confirmation wave after the menu +
  `none_reason` fixes, including the 5 controls (enriched only) plus human
  adjudication of the controls — if the converted-NONE groups then produce
  unanimous letters that match human judgment, v7 (enriched packs + this
  panel) is ready to bless.
