# Stitch supervision restart

Four dependent implementation batches, based on `d13a93d`. No new human labels
or user adjudication are required. Existing human records remain unchanged;
ambiguous evidence is quarantined or retained as unknown. Production inference,
publication gates, and bundled models are outside these four batches.

| Batch | Deliverable | Status |
|---|---|---|
| 1 | Recover scoped observations from archived votes; preserve unknowns, evidence versions, overlap conflicts, and parent lineage | Complete |
| 2 | Freeze and audit the usable existing human evaluation set; exclude related weak observations | Pending batch 1 |
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
