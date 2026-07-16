# Codex repeated-draw stitching diagnostic

**Status:** infrastructure only; diagnostic findings are not panel findings or labels.

## Purpose

This experiment measures whether the v7 Codex voter is self-consistent on immutable
stitching evidence packs and asks a separate structured audit pass what evidence,
option-menu shape, or rubric language made a decision difficult. It is designed to
identify candidate fixes before rebuilding packs and running a completely fresh
Claude/Codex/Muse panel.

The experiment deliberately does **not**:

- create a production panel with repeated fake Codex seats;
- compute consensus or routing;
- write `votes.csv` or `consensus.csv`;
- export or mint labels;
- change the current v7 packs, manifest, partial ballots, rubric, or `stitch_runner.py`;
- treat model feedback or increased unanimity as ground truth.

Raw diagnostic results live under
`data/agents/stitching/diagnostics/<run-id>/`, outside the production `batches/`
tree. The directory carries `.no-export`; development and holdout results are kept
in separate subtrees. `data/` is ignored, so raw reasoning remains a local research
artifact unless intentionally archived elsewhere.

## Design

The first run uses
`data/agents/stitching/batches/physical_context_v7_20260715_manifest.json`:

- 65 immutable packs / 50 unique groups / 8 datasets;
- three fresh canonical `gpt-5.6-sol` high-effort draws per pack;
- one fresh audit draw per pack with structured `pack_feedback`;
- 260 total Codex calls;
- 35 development groups and 15 sealed holdout groups;
- two of the five factorial groups assigned wholly to holdout.

All variants of one unique group share a cohort. Calls run pass-by-pass in the
source manifest's counterbalanced order. The audit choice is excluded from the
three-draw self-consistency classification because its extra instruction is a
prompt perturbation.

Self-consistency is based on the exact selected edge set:

- `stable`: 3/3;
- `majority`: 2/3;
- `split`: three different edge sets;
- `incomplete`: a missing or abstaining canonical draw.

Where an exact current pair-semantics human label exists, the analyzer also reports
modal exact match, Jaccard, and whether human truth was expressible in the option
menu. Set-semantics and drifted labels are not coerced into pair truth.

## Isolation and provenance

Planning and every resume revalidate:

- source wave-manifest bytes;
- each source `batch.json` and its evidence-to-batch semantic binding;
- every managed evidence-pack file and evidence identity;
- prompt, option-menu, and displayed-candidate-universe hashes;
- the exact Codex model/effort, timeout, and invocation signature;
- the diagnostic instruction and diagnostic runtime hashes;
- the unchanged `stitch_runner.py` runtime hash.

One atomic JSON file is written per call. An existing result must match its call,
plan, runtime, prompt, and evidence bindings byte-for-byte or resume fails. Results
are never overwritten. An OS file lock prevents two diagnostic processes from
writing the same run concurrently.

Holdout aggregation is disabled unless the operator supplies a JSON fix-frozen
marker with a nonblank `fix_id`. This is procedural sealing, not encryption; its
purpose is to prevent accidental tuning and accidental inclusion in ordinary
reports. Runtime progress also redacts holdout choices as `SEALED`, and the smoke
pair is selected from development packs whenever both pack shapes are available.

## Runbook

Validate the intended slate without writing anything:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_codex_stitch_diagnostic.py plan \
  data/agents/stitching/batches/physical_context_v7_20260715_manifest.json \
  --output data/agents/stitching/diagnostics/codex_deep_v1 \
  --run-id codex_deep_v1 \
  --canonical-draws 3 \
  --holdout-groups 15 \
  --holdout-factorial-groups 2 \
  --validate-only
```

Write the bound plan by removing `--validate-only`, then run the ordinary plus
factorial smoke pair:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_codex_stitch_diagnostic.py run \
  data/agents/stitching/diagnostics/codex_deep_v1/diagnostic_manifest.json \
  --workers 5 --smoke
```

If all eight smoke calls have valid ballots/audit payloads, resume the full run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_codex_stitch_diagnostic.py run \
  data/agents/stitching/diagnostics/codex_deep_v1/diagnostic_manifest.json \
  --workers 5
```

Five workers is the default initial ceiling: it is enough to reduce wall time
without immediately multiplying provider retries and local image/process load by
ten. The runner permits at most ten workers. Raise from five only after the smoke
and an initial full-run tranche show healthy latency, no provider-wide failures,
and no local resource pressure. A failure stops new claims, lets in-flight calls
finish, and preserves every atomic result for resume.

Inspect status and produce a development-only report:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_codex_stitch_diagnostic.py status \
  data/agents/stitching/diagnostics/codex_deep_v1/diagnostic_manifest.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_codex_stitch_diagnostic.py analyze \
  data/agents/stitching/diagnostics/codex_deep_v1/diagnostic_manifest.json
```

The JSON summary retains each canonical draw's choice, confidence, edge set,
reasoning, and error state plus the audit draw's normalized feedback. It also
groups multi-variant packs into paired factorial contrasts. These are diagnostic
records under `.no-export`, not panel ballots.

## From findings to fixes

A diagnostic pattern becomes a fix candidate only when it is human-confirmed,
appears across at least three groups and two datasets, makes human truth
inexpressible, creates an unsafe stable result, or is isolated by the factorial
controls. Codex-only style or confidence differences do not justify changing the
global rubric.

Evidence, option-generation, rendering, and rubric changes should land as separate
PRs so their effects stay attributable. After fixes are frozen, rebuild the same
65 packs under a new evidence/rubric provenance era and run 195 fresh heterogeneous
calls (Claude/Codex/Muse). Only then unseal the 15-group holdout. The primary gates
are human correctness, option expressibility, and zero new false auto-accepts—not
raw convergence.
