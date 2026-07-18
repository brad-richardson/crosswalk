# Bulk stitching-vote wave plan (drafted 2026-07-18)

Status: **waiting on quota** (codex low). Preconditions tracked below.
Decision owner: Brad. Prepared at the close of the v9 arc
(`research/v9_rerun_adjudication.md`).

## Why now

- Rubric validated: v9 targeted rerun scored 5/6 against ground truth; the
  sixth (a451bf05) was an option-menu expressibility miss, since fixed (#459)
  and re-voted to a majority verdict. Brad's 2026-07-18 review session ratified
  the panel's merges at membership level (b14c342).
- Every vote now does double duty: failures feed the human review queue with
  genuine ambiguity only (queue is gated to panel failures); unanimous
  successes mint stitching labels that grow the learned-resolver base toward
  its documented flip condition (~94 → 200+ clean groups; #410 NO-GO).
- Evidence quality just improved: per-edge lateral offsets now render in packs
  (#460) — the exact signal whose absence caused v9's defaulting — and the
  consensus-desired-edges seed path (#457/#459) closes expressibility misses.
- The known noise source is held out: `ch_grand_geneva_cycle_schema` is under
  `panel_hold` (route-overlay targets, not infrastructure — PR #461).

## Panel eras vs rubric arcs (how v7-candidate interacts with v9)

Two orthogonal version axes; do not conflate:

- **Panel era (v1/v3/v4/v5/v7…)** versions the *seat composition*
  (provider+model+effort) for label provenance — `labeler=panel_unanimous_vN`.
  `stitch_export` only mints from *blessed* compositions.
- **Rubric/evidence arc (v8, v9…)** versions the *pack content* the seats see
  (MI-4 text, access channel, offset evidence, pre-registrations). The v9 arc's
  votes were cast BY the v7-candidate trio (claude opus-4-8 / codex gpt-5.6-sol
  / muse spark-1.1, all high).

Blessing v7-candidate is **forward-looking only**:

- Bulk-wave unanimous verdicts mint `panel_unanimous_v7` cleanly (no
  `--allow-nonstandard-panel` overrides, which don't scale and blur provenance).
- No retroactive minting: the v9-arc groups all routed `human_review`
  (dissents / class-mismatch gate), and Brad's human labels on them now take
  precedence via the export's human-precedence gate regardless.
- Era stamps stay meaningful downstream: v7-labeled rows are distinguishable
  from v1's pre-access-channel era for any future era-tiered consumption
  (resolver training weights, promotion gating).
- Unchanged by blessing: the deterministic cross-mode gate still demotes
  road↔cycleway unanimous merges to human review. Relaxing it (for unanimous +
  offset-confirmed merges) is a separate future decision, pending offset
  re-validation in the wave.

## Wave shape

- **Datasets (breadth pass, Geneva excluded via panel_hold):**
  `au_sydney_roads`, `fi_helsinki_roads`, `us_boston_streets`,
  `us_seattle_sidewalks`, `co_bogota_bike_network`.
  Bogotá bike is deliberate: its votes generate the cross-mode labels needed to
  measure (and eventually clear) its own publish `quality_hold` and to arm a
  cross-mode stitch-gate floor in `mbench/datasets.toml`.
- **Depth:** default tier sampling, ~15 groups/dataset, `--decompose` on so
  monster-group verdicts stay exportable. Seed files where prior
  `no_exact_option` ballots exist.
- **Panel:** v7-candidate (blessed), high effort, 3 seats.
- **Cost:** ~5 × 15 × 3 ≈ 225 high-reasoning invocations. First tranche can
  trim to 10/dataset (~150) if quota is tight.
- **Freeze rule:** all pack/config/rubric changes land BEFORE launch; nothing
  merges mid-wave (see stitch_runner wave-freeze precedent).

## Preconditions

| item | status |
|---|---|
| #460 offset evidence merged | ✅ d6dadf9 |
| #459 seed wiring merged | ✅ 82a9d93 |
| #461 panel_hold (Geneva) merged | ⏳ review chain running |
| Bless v7-candidate composition | ⏳ PR in flight (this decision: 2026-07-18) |
| Resolver ingestion audit (does minted influx count toward flip condition?) | ⏳ re-running |
| Codex quota recovered | ⏳ Brad's call |

## Post-wave sequence

1. `stitch-export` per dataset → mint `panel_unanimous_v7` labels; archive vote
   provenance; commit both.
2. Rebuild human queues (`data stitch-batch` per dataset + `stitch-batch-all`)
   → Brad reviews the failures.
3. Re-run `scripts/benchmark_resolver.py` at the grown label base; compare to
   #410 (edge-F1 0.899 prod / 0.875 learned; group-exact 0.730/0.748).
4. Revisit: cross-mode gate relaxation; access/mode pair features
   (Bogotá unblock); resolver GO/NO-GO.
