# Stitch supervision restart

Four dependent implementation batches, based on `d13a93d`. No new human labels
or user adjudication are required. Existing human records remain unchanged;
ambiguous evidence is quarantined or retained as unknown. Production inference,
publication gates, and bundled models are outside these four batches.

| Batch | Deliverable | Status |
|---|---|---|
| 1 | Recover scoped observations from archived votes; preserve unknowns, evidence versions, overlap conflicts, and parent lineage | Complete |
| 2 | Freeze and audit the usable existing human evaluation set; exclude related weak observations | Complete |
| 3 | Support direct edge decisions and uncertainty; bind ballots to evidence and model configuration | Pending batch 2 |
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
