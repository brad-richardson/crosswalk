# v8 Physical/Coincidence Stitch Wave — Analyst Context & Goals (2026-07-17)

**You are one of two independent analysts** (a Claude Fable seat and a Codex
`gpt-5.6-sol` seat) reviewing the just-completed **v8** stitching panel wave.
Work independently — do **not** read the other analyst's output, and do not read
the v7 analyst outputs (`research/v7_wave_analysis_{codex,fable}_*.md`); you may
read the v7 *archived ballots* for baseline comparison (see below). Write your
findings to your own doc (path given in your task prompt). A separate
orchestrator will read both and synthesize. Your job is **analysis, not
consensus**: surface patterns, quantify them, flag what a human should look at.

## What this wave is and why it exists

Crosswalk conflates local road datasets to Overture GERS ids. The hard cases are
**M:N "stitch" groups** — clusters of reference (Overture) and target (local)
segments where the optimizer must pick which edges belong together. An AI panel
votes on the correct edge selection per group; unanimous/quorum accepts become
human-equivalent labels, everything else routes to human review.

v8 is a **single "generation jump"** that folds every corroborated improvement
from the v7 analysis into one wave (quota is the bottleneck, so we mint one v8,
not incremental v8/v9). Same experimental frame as v7 (50 enriched groups + a
2×2 factorial on 5 controls, same 8 datasets). What changed relative to v7:

- **Rubric era `2026-07-17+9463c80a0f77`** — the **MI-4 cycleway/separated-infra
  uncertainty gate** ("Fix A", holdout-corroborated in #448 / `3f9b488`).
- **`none_reason` + `desired_edges` prompt contract** (#451) — v8 NONE ballots
  are **machine-analyzable**: the panel emits a structured `none_reason` and the
  edge set it *wanted*. (In v7 this had to be hand-classified from free text.)
- **Exact-pair minus-flagged-edge option seeds** (#450, cap 4, max menu 20) —
  the intended fix for the dominant v7 "no exact option offered" expressibility
  gap. The correct edge set should now be **on the menu** far more often.
- **#443 exact-pair image overlays** + **#446 invocation era `2026-07-16.1`**
  (seat-level retries + full invocation-provenance binding).

The panel is **unchanged** from v7 and still a **candidate ("nonstandard")
composition, not calibration-promoted** — which is *why this analysis exists*:
it decides whether v8 can be **blessed** and whether the **deferred v7+v8
auto-accept stitching labels** can finally be minted (nothing has been minted
yet; vote provenance only).

## Panel composition (all three seats, high reasoning)

| Seat | Model | Effort |
|------|-------|--------|
| claude | `claude-opus-4-8` | high |
| codex | `gpt-5.6-sol` | high |
| muse | `meta/muse-spark-1.1` | high |

## Experimental design (the 2×2)

- **65 groups** = **50 enriched** unique groups + a **2×2 ablation** on **5
  control groups**, each run under all four variants.
- Variants (encoded in each batch dir name and in `batch.json → experiment.variant`):
  - `enriched` — physical **on**, coincidence **on** (the full pack; also the 50 singletons)
  - `no_physical` — physical **off**, coincidence **on**
  - `no_coincidence` — physical **on**, coincidence **off**
  - `minimal` — physical **off**, coincidence **off** (the 2×2 origin corner)
- **5 control groups** (appear in all 4 variants — the interaction cell), one per
  factorial dataset:
  `fb8f359f` (au_sydney), `7bac1f1d` (de_berlin), `92c0997f` (fi_helsinki),
  `1b90f03b` (gb_london), `18ef284e` (hk_hongkong).
- 8 datasets: `au_sydney_roads`, `de_berlin_roads`, `fi_helsinki_roads`,
  `gb_london_roads`, `hk_hongkong_roads` (each has all 4 variant batch dirs),
  plus `ch_grand_geneva_cycle_schema`, `nl_amsterdam_roads`,
  `us_philadelphia_sidewalks` (enriched-only).

## Where the data is (read-only — do NOT modify anything)

- **Archived provenance** (tracked, canonical): `labels/votes/dataset=*/{votes,consensus,evidence}.csv`.
  - `votes.csv` — one row **per seat per group**, `source_batch` distinguishes
    `physical_context_v8_20260717[_variant]` from the v7 rows in the same files.
    v8-only columns: **`attempt`** (seat retry index), **`none_reason`** (enum,
    populated on every v8 NONE), **`desired_edges`** (the edge set the seat
    wanted — see the R#/T# caveat below).
  - `consensus.csv` — one row per group: `consensus` (unanimous/majority/none),
    `choice`, `routing` (auto_accept/human_review), `n_valid`, `minority`,
    `mean_confidence`, `route_reason`.
  - `evidence.csv` — evidence pack hashes + displayed candidate count per group.
- **Batch working dirs** (richest source, git-ignored but present on disk):
  `data/agents/stitching/batches/<dataset>_physical_context_v8_20260717[_variant]/`
  - `batch.json` — `experiment.variant`, `experiment.physical_metadata`, dataset
    id, **and the per-group R#/T# → source-id label maps** (needed for the caveat).
  - Per-group hash subdirs — option images + evidence packs + the offered option menu.

## Vote semantics you must respect

- **`choice` is an option letter** (A, B, C, …) selecting an offered edge set, or
  **`NONE`**. `NONE` is a **first-class, decisive reject-all** verdict — NOT an
  abstention. In v8, disambiguate via the structured `none_reason` enum (plus
  `desired_edges`), not hand-reading, wherever possible.
- A true **abstain** has an `abstain_reason` (CLI timeout/error) / non-empty
  `error` and is not a decisive ballot. Separate these from NONE in all stats.
  Also mind `attempt` — a retried seat may have multiple rows; count the final
  successful attempt, not the timed-out ones.

### The R#/T# `desired_edges` label-mapping caveat (READ CAREFULLY)

`desired_edges` are stored as **raw R#/T# display labels** (what the seat saw in
the option menu), **NOT source ids** like `edge_set`/`chosen_option_id` use. To
judge whether a NONE reflects a genuine expressibility gap you MUST:

1. Map each R#/T# label → source id via that group's `batch.json` label map.
2. Validate `desired_edges` is **non-empty** and **differs from every offered
   option's edge set** — only then is it evidence that the correct set wasn't on
   the menu. (The v8 diagnostic path does this mapping/validation; the production
   capture that wrote these rows does **not** — so do it yourself and report how
   many desired_edges failed to map or were degenerate.)

## Analysis goals — the five the disposition hinges on

Produce **quantified** findings, each with specific `group_id`s, seats, and
numbers (not impressions). Where you infer from free text, quote the snippet and
cite `group_id`/`provider`.

1. **Did exact-pair option seeds close the expressibility gap?** This is the
   headline question. Using v8 `none_reason` (+ mapped/validated `desired_edges`),
   quantify what fraction of v8 NONE ballots are "no exact option offered"
   (expressibility gap) vs genuine reject-all vs insufficient-evidence. Compare
   against the v7 baseline (compute it yourself from the v7 rows in the same
   `votes.csv` files — v7 `none_reason` is empty, so classify v7 NONEs from their
   free-text `reasoning`/`pack_feedback`). Report the **v7→v8 delta** in
   expressibility-gap NONE rate. Did #450 actually put the wanted set on the menu?
2. **`none_reason` breakdown** — full distribution of the v8 `none_reason` enum
   across all 85 NONE ballots, sliced by seat and by dataset/variant, **with the
   R#/T# mapping caveat applied** (report mapping failure/degenerate rate). Flag
   any `none_reason` value whose `desired_edges` *was* actually expressible among
   the offered options (i.e. the seat mis-fired NONE).
3. **The `7175635e` Fix-A non-regression check.** Group `7175635e` is a
   fresh-panel non-regression guard for the MI-4 cycleway gate: it must **not**
   flip its correct unanimous cycleway *merge* to NONE. Locate its v8 ballots,
   confirm the merge held (consensus + per-seat), and report any seat that went
   NONE/dissented. Also scan for other cycleway/separated-infra groups where the
   new MI-4 gate plausibly **over-triggers** (correct merges pushed to NONE).
4. **2×2 factorial contrasts** on the 5 controls (`fb8f359f`, `7bac1f1d`,
   `92c0997f`, `1b90f03b`, `18ef284e`). Build the physical×coincidence table per
   group per seat (choice, confidence, none_reason). enriched vs no_physical
   (physical effect), enriched vs no_coincidence (coincidence effect), and the
   full interaction. Is either factor dominant? Is `minimal` meaningfully worse
   than `enriched`? Same paired group_id across variants — compare ballot-to-ballot.
5. **Does v8 earn blessing, and can the deferred v7+v8 auto-accepts be minted?**
   Give a clear recommendation (bless / iterate rubric / fix expressibility) with
   reasoning. Enumerate the specific v8 auto-accept groups (`routing=auto_accept`)
   and assess whether each is trustworthy enough to mint as a production label.
   Call out any auto-accept you would withhold and why.

### Also cover (carried from v7, still informative)

6. **Per-seat behavior**: choice distribution, mean confidence, NONE rate,
   abstain/retry rate, agreement with panel majority, characteristic reasoning
   style. Note any systematic dissenter.
7. **Frontage-road & vertically-layered ambiguity**: groups whose reasoning
   mentions frontage/service roads, overpass/underpass, bridge/tunnel, vertical
   layering. Does physical/coincidence context steer the panel toward *not*
   merging spatially-coincident-but-unconnected roads without over-splitting
   legitimately continuous roads?
8. **Disagreement map + pack_feedback synthesis**: which groups split the panel
   and along what fault line (which seat vs which); are splits concentrated in
   ablated variants? Aggregate free-text `pack_feedback` into recurring themes
   (evidence gaps, confusing menus, rubric ambiguities) for the next rubric revision.

## Deliverable

A thorough markdown report at your assigned path. Lead with a **5–10 bullet
executive summary** of the highest-signal findings, then a section per goal with
specific `group_id`s/seats/numbers as evidence. End with a **prioritized list of
what a human should look at first** and a concrete v8 disposition recommendation
(bless / iterate / fix) plus your explicit yes/no on minting the deferred v7+v8
auto-accepts, with reasoning.

Be rigorous and skeptical. Quantify everything you can. **If the data contradicts
the hypotheses** (exact-pair seeds closed the gap; physical/coincidence context
helps; Fix A didn't over-trigger), **say so plainly** — a null or negative result
is a valid and valuable finding.
