# v7 Physical/Frontage Stitch Wave — Analyst Context & Goals (2026-07-17)

**You are one of two independent analysts** (a Claude Fable seat and a Codex
`gpt-5.6-sol` seat) reviewing the just-completed v7 stitching panel wave. Work
independently — do not read the other analyst's output. Write your findings to
your own doc (path given in your task prompt). A separate orchestrator will read
both and synthesize. Your job is **analysis, not consensus**: surface patterns,
quantify them, and flag what a human reviewer should look at.

## What this wave is and why it exists

Crosswalk conflates local road datasets to Overture GERS ids. The hard cases are
**M:N "stitch" groups** — clusters of reference (Overture) and target (local)
segments where the optimizer must pick which edges belong together. An AI panel
votes on the correct edge selection for each group; unanimous/quorum accepts
become human-equivalent labels, everything else routes to human review.

This wave is a **controlled experiment**: does adding **physical evidence**
(bridge/tunnel/vertical-layer/elevation attributes) and **spatial coincidence
context** to a group's evidence pack change panel behavior — and does it help on
the two dominant ambiguity themes: **frontage/service-road identity** and
**vertically-layered roads** (overpasses/underpasses that are spatially
coincident but not connected)? The concern is that continuity-based reasoning
over-merges these cases; physical/coincidence context is the hypothesized fix.

## Panel composition (all three seats, high reasoning)

| Seat | Model | Effort |
|------|-------|--------|
| claude | `claude-opus-4-8` | high |
| codex | `gpt-5.6-sol` | high |
| muse | `meta/muse-spark-1.1` | high |

This is a candidate ("nonstandard") composition, **not yet calibration-promoted**
— which is *why this analysis exists*: it helps decide whether v7 can be blessed.
No production stitching labels have been minted from it; vote provenance only.

## Experimental design (the 2×2)

- **65 schedule rows** = **50 enriched** unique groups + a **2×2 ablation** on
  **5 control groups**, each run under all four variants.
- Variants (encoded in each batch dir name and in `batch.json → experiment.variant`):
  - `enriched` — physical **on**, coincidence **on** (the full pack; also the 50 singletons)
  - `no_physical` — physical **off**, coincidence **on**
  - `no_coincidence` — physical **on**, coincidence **off**
  - `minimal` — physical **off**, coincidence **off** (the 2×2 origin corner)
- **5 control groups** (appear in all 4 variants — this is the interaction cell):
  `fb8f359f`, `7bac1f1d`, `92c0997f`, `1b90f03b`, `18ef284e`.
- 8 datasets: `au_sydney_roads`, `de_berlin_roads`, `fi_helsinki_roads`,
  `gb_london_roads`, `hk_hongkong_roads` (each has all 4 variant batch dirs),
  plus `ch_grand_geneva_cycle_schema`, `nl_amsterdam_roads`,
  `us_philadelphia_sidewalks` (enriched-only).

## Where the data is (read-only — do NOT modify)

- **Batch working dirs** (richest source): `data/agents/stitching/batches/<dataset>_physical_context_v7_20260715[_variant]/`
  - `votes.csv` — one row **per seat per group**. Key columns:
    `group_id, provider, model, choice, confidence, reasoning, edge_set,
    abstain_reason, pack_feedback, chosen_option_id, evidence_id`.
    `reasoning` and `pack_feedback` are free-text — this is your primary
    qualitative signal.
  - `consensus.csv` — one row **per group**: `consensus` (unanimous/majority/none),
    `choice`, `routing` (auto_accept/human_review), `n_valid`, `minority`,
    `mean_confidence`, `route_reason`.
  - `batch.json` — `experiment.variant`, `experiment.physical_metadata`, dataset id.
  - Per-group subdirs (hash-named) — option images + evidence packs.
- **Archived provenance** (same ballots, tracked): `labels/votes/dataset=*/{votes,consensus,evidence}.csv`,
  with a `source_batch` column for cross-batch traceability.

## Vote semantics you must respect

- **`choice` is an option letter** (A, B, C, …) selecting an offered edge set, or
  **`NONE`**. `NONE` is a **first-class, decisive reject-all** verdict — NOT an
  abstention. Distinguish three flavors of NONE where the reasoning/pack_feedback
  reveals it: (a) genuine reject-all (no offered option is correct), (b) **no
  exact option offered** (the correct edge set wasn't on the menu — an
  expressibility gap, not a judgment), (c) insufficient evidence.
- A true **abstain** has an `abstain_reason` (CLI timeout/error) and is not a
  decisive ballot. Separate these from NONE in all statistics.

## Analysis goals (produce quantified findings on each)

1. **Per-seat behavior**: choice distribution, mean confidence, NONE rate,
   abstain rate, agreement with the panel majority, and characteristic reasoning
   style. Note any seat that is systematically the dissenter.
2. **enriched vs no_physical** (physical evidence effect): on the 5 control
   groups, did removing physical evidence change any seat's choice, confidence,
   or NONE-flavor? Same group_id across the two variants — compare ballot-to-ballot.
3. **enriched vs no_coincidence** (coincidence context effect): same paired
   comparison.
4. **Full 2×2 interaction** across the 5 controls: build the
   physical×coincidence table per group per seat. Is there an interaction
   (physical only helps when coincidence is present, or vice versa)? Is either
   factor dominant? Is `minimal` meaningfully worse than `enriched`?
5. **Frontage-road & vertically-layered ambiguity**: find groups whose
   reasoning mentions frontage/service roads, overpass/underpass, bridge/tunnel,
   or vertical layering. Does physical/coincidence context steer the panel toward
   *not* merging spatially-coincident-but-unconnected roads — without
   over-splitting legitimately continuous roads?
6. **NONE forensics**: for every NONE ballot, classify the flavor from its
   reasoning/pack_feedback. Quantify how many are "no exact option offered"
   (expressibility gap) vs genuine reject-all. This directly informs the planned
   `none_reason` enum and exact-pair option generation.
7. **Disagreement map**: which groups split the panel, and along what fault line
   (which seat vs which)? Cross-reference with variant — are splits concentrated
   in ablated variants?
8. **pack_feedback synthesis**: aggregate the free-text feedback into recurring
   themes (evidence gaps, confusing option menus, rubric ambiguities). These feed
   the next rubric revision.

## Deliverable

A thorough markdown findings report at your assigned path. Lead with an
executive summary (5–10 bullets of the highest-signal findings), then a section
per goal above with **specific group_ids, seats, and numbers** as evidence — not
vague impressions. Where you infer intent from free-text, quote the snippet and
cite `group_id`/`provider`. End with a prioritized list of what a human reviewer
should look at first and any concrete recommendations for v7's disposition
(bless / iterate rubric / fix expressibility) with your reasoning.

Be rigorous and skeptical. Quantify everything you can. If the data contradicts
the experiment's hypothesis (physical/coincidence context helps), say so plainly.
