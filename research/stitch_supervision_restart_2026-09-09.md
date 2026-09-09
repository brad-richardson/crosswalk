# Stitch supervision restart

Four dependent implementation batches, based on `d13a93d`. No new human labels
or user adjudication are required. Existing human records remain unchanged;
ambiguous evidence is quarantined or retained as unknown. Production inference,
publication gates, and bundled models are outside these four batches.

| Batch | Deliverable | Status |
|---|---|---|
| 1 | Recover scoped observations from archived votes; preserve unknowns, evidence versions, overlap conflicts, and parent lineage | Complete |
| 2 | Freeze and audit the usable existing human evaluation set; exclude related weak observations | Complete |
| 3 | Support direct edge decisions and uncertainty; bind ballots to evidence and model configuration | Complete |
| 4 | Run weighted human/unanimous/majority supervision comparisons against a fixed evaluation set and optimizer baseline | Pending batch 3 |

Each batch receives a separate commit and dependent branch. Validation combines
focused semantic regression tests, real archived-data replay, and the repository's
required final suite checks. Research outputs are isolated from production labels.

## Batch 1: scoped archive recovery

`scripts/recover_stitch_observations.py` recovered 8,911 displayed edge rows from
836 evidence/invocation groups. Reconciliation leaves 7,592 rows, of which 7,157
have a usable opinion; 414 conflicting overlaps are masked. The archive contains
317 child observations across nine parents. These counts describe observations,
not independent human truths or fully labeled parent groups.

The reconciled tiers are 3,666 unanimous, 312 three-seat majority, 3,179 below
quorum, and 435 unknown. Below-quorum rows are retained for audit, not admitted to
the proposed consensus experiment. Sixty groups lacked compatible evidence and
were skipped. Source and output hashes are recorded in
`stitch_supervision_batch1_2026-09-09.json`; parquet outputs live under
`data/experiments/stitch-supervision-20260909/batch1`.

Features are computed over the full eligible candidate group before unknown
edges are masked. Focused resolver extraction, votes, and training regression
tests: **113 passed**. Formatting and lint passed for the changed files.

## Batch 2: existing human evaluation

The automatic audit retains **14 groups / 106 edges / 92 positives**, including
five deanchored groups, across seven datasets. Of 119 existing human records,
81 are membership labels rather than exact-edge truth; 13 exact labels have
split/missing edges, nine changed group without recorded original scope, one
is unmapped, and the disputed Sydney label is quarantined. No human record was
changed or relabeled. Other available Boston/Amsterdam factory snapshots were
checked for better scope recovery; none retained more complete exact groups.

Five deterministic folds hold connected road/parent components together. All
2,327 weak observations connected to **any** existing human scope are excluded
in every fold, including unknowns and related quarantined cases. The remaining
archive has 2,802 unanimous and 229 three-seat majority rows before source
artifact verification and parent weight caps. Repeated observations cannot
increase a parent's influence. New waves must pass the frozen component
exclusion, including transitive connections they introduce.

The immutable evaluation outputs live under
`data/experiments/stitch-supervision-20260909/batch2`; their hashes, source hashes,
audit counts, and preregistered experiment settings are committed in
`stitch_supervision_batch2_2026-09-09.json`. The loader verifies all frozen outputs
before an experiment. Focused audit/extraction tests: **56 passed**. The tests
also uncovered and fixed the extractor's empty-table handling when every group
is quarantined for contradictory human labels.

This is a small, mostly optimizer-anchored benchmark with fixed existing pair
scores. It supports a paired resolver experiment, not end-to-end accuracy or
model promotion claims. Historical human sessions did not hash their candidate
universe; unchanged group identity or explicit original membership coverage is
the strongest available scope check. No new labels are required to continue.

## Batch 3: direct edge panel

`scripts/run_stitch_edge_panel.py` prepares verified, menu-free evidence and runs
one blind draw per pinned Fable 5.1, Astra 6, and Muse Spark 1.3 Contributor seat.
Each displayed edge requires separate identity and keep/drop/unknown decisions;
omissions, incompatible evidence IDs, and contradictory NONE reasons invalidate
the ballot. A resolution drop never becomes a negative pair-identity label.
The new panel uses equal seats with no inherited provider reliability weights.
Schema, rubric, prompt, model/configuration, transport code, and image hashes
identify each wave; existing ballots cannot be replaced by retrying for agreement.

A live two-pack test produced **six valid ballots** and 19 edge observations:
13 unanimous, one three-seat majority, and five below quorum. On Helsinki
`92c0997f`, Astra kept nine edges, dropped four, and marked five unknown; Fable
kept 13 and dropped five. This preserves useful decisions from the earlier
whole-group insufficient-evidence response. Counts are protocol evidence, not
measured truth or calibrated accuracy. The live outputs are excluded from the
preregistered archive experiment and do not update production labels or routing.

Artifacts live under `data/experiments/stitch-supervision-20260909/batch3`;
the small report is committed as `stitch_supervision_batch3_2026-09-09.json`.
Focused panel, delivery, export, and observation tests: **410 passed**.

CI initially failed during collection because installing unlocked web extras
then partially syncing locked core dependencies mixed AnyIO and typing-extensions
versions. A separate follow-up commit on batch 1 now installs locked extras and
uses `uv run --no-sync` for subsequent commands. Both Spark architectures passed
after the repair; the repair is carried forward through the chain.
