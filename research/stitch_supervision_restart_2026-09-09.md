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
| 4 | Run weighted human/unanimous/majority supervision comparisons against a fixed evaluation set and optimizer baseline | Complete |

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

## Batch 4: weighted supervision experiment

`scripts/run_stitch_supervision_experiment.py` verifies the frozen evaluation
outputs and original weak-evidence source artifacts before training. It excludes
270 rows whose source bytes changed. Features and leakage scope are computed
over the complete eligible parent context, including unvoted edges, before any
supervision mask. That extended scope excludes 2,493 of 7,322 verified rows.
The final training pool contains **2,589 unanimous + 216 majority rows**, spanning
306 connected components. No unknown or below-quorum target reaches the loss.

The three learned policies share the frozen five folds, 33 features, three
seeds, logistic loss, and existing expected-F1 selector. Unanimous observations
receive weight 0.35; majority observations receive 0.15 and retain their 1/3 or
2/3 target. Source versions and repeated observations share a physical pair's
budget, and each connected parent component has a maximum total weight of four.
The trainer now accepts sample weights and fails visibly if fractional training
cannot run; it never silently hardens a fractional target after a model error.

| Policy | Edge F1 | Exact groups | Weak rows | Weak weight mass |
|---|---:|---:|---:|---:|
| Existing optimizer | 0.8508 | 7/14 | 0 | 0 |
| Human only | 0.9436 | 9/14 | 0 | 0 |
| Human + unanimous archive | 0.9215 | 7/14 | 2,589 | 447.30 |
| Human + unanimous + majority archive | 0.9062 | 6/14 | 2,805 | 469.55 |

Adding majority archive supervision reduced F1 by 0.0373 versus human-only
(paired component bootstrap 95% interval **[-0.1042, -0.0040]**). Its incremental
effect versus unanimous-only was -0.0152, with interval **[-0.0606, +0.0090]**.
The unanimous-versus-human-only interval also crosses zero. These conditional,
small-sample intervals do not establish population accuracy or explain whether
the loss comes from label noise, source distribution, or the selected weights.

The human-only result is not grounds for promotion either. There are only 14
groups, most positive edges come from a few large groups, and all five surviving
deanchored groups are reject-all cases. Human-only and unanimous training each
reject one of those five groups correctly; majority rejects none. As a post-hoc
sanity check, simply keeping all 106 candidates would already score F1 0.9293
and 8/14 exact groups. The fixed benchmark exposes these limits instead of
turning more model agreement into claimed ground truth.

**Decision:** retain the recovered observations and direct-edge automation, but
leave weak-supervision training opt-in and do not replace the production model.
This experiment evaluates the **older archived panel**; the six new frontier
draws validate the direct-edge protocol and are too few to validate its accuracy.
Future model comparison can run on this same frozen set without waiting on new
human labels. Source/rubric-specific calibration and constrained routing remain
separate follow-up work.

The committed `stitch_supervision_batch4_2026-09-09.json` records full metrics,
paired intervals, slices, source checks, code/library versions, and output
hashes. Predictions, verified weak training rows, and three isolated research
models live under `data/experiments/stitch-supervision-20260909/batch4`.
Focused weighted-training, context, and evaluation tests: **52 passed**.

## Running the batches

The four PRs are chained: [#483](https://github.com/brad-richardson/crosswalk/pull/483)
→ [#484](https://github.com/brad-richardson/crosswalk/pull/484)
→ [#485](https://github.com/brad-richardson/crosswalk/pull/485)
→ [#486](https://github.com/brad-richardson/crosswalk/pull/486).

For a fresh replay, choose new output directories:

```bash
uv run --no-sync python scripts/recover_stitch_observations.py \
  --out data/experiments/stitch-supervision-replay/batch1
uv run --no-sync python scripts/freeze_stitch_evaluation.py \
  --observations data/experiments/stitch-supervision-replay/batch1/reconciled.parquet \
  --out data/experiments/stitch-supervision-replay/batch2
uv run --no-sync python scripts/run_stitch_edge_panel.py \
  --packs research/stitch_supervision_example_packs.json \
  --out data/experiments/stitch-supervision-replay/batch3
uv run --no-sync python scripts/run_stitch_supervision_experiment.py \
  --evaluation data/experiments/stitch-supervision-20260909/batch2 \
  --out data/experiments/stitch-supervision-replay/batch4
```

Batch 3 prepares evidence without model calls; add `--run` to request the six
new draws. Its live output stays separate from batch 4. Subsequent comparisons
should reuse the original `20260909/batch2` frozen evaluation, as the fourth
command does, instead of choosing a new benchmark after seeing results. Use
these scripts for the audited, weighted experiment; they enforce the source,
scope, and loss contracts together. No command writes human labels, publishes
bridges, or installs a bundled model.

## Final validation

The complete implementation passed [CI on both x86 and ARM](https://github.com/brad-richardson/crosswalk/actions/runs/34375733893):
**4,148 tests passed**, 44 skipped, one expected failure per architecture;
**42 serial performance tests passed** per architecture; **256 mbench tests
passed** (one skip); and **two Spark tests passed** per architecture. The actual
training quality regression ran and passed in both main CI jobs.

Local checks also passed the 42 serial performance tests, the actual training
regression plus full export suite (111 tests), and the 42 panel-monitor tests.
All 430 tracked/new Python files passed formatting and lint. The full local run
needed execution outside the sandbox because its FastAPI TestClient stalled
inside it; an isolated comparison confirmed the same test passed outside in
0.77 seconds. One local parallel rerun completed 4,118 tests but lost a worker
to a native pandas CSV-parser segmentation fault; the affected export suite
subsequently passed serially. No application workaround was introduced for
either environment issue. A separate console-width test fixture was made
explicit so terminal overrides do not hide the voter names it asserts.

Frozen evaluation hashes, experiment code/output hashes, and zero shared
reference/target segments between admitted weak rows and human evaluation were
verified. Production labels, inference defaults, publication gates, and bundled
models are unchanged. No new human labeling or adjudication blocks these batches.
