# Scaling Stitching-Group Ground Truth with Agents

Status: Tiers 0–2 shipped (#229, #230); option generator fixed for multi-ref
M:N shapes (#231); Phase-2 production run (60 groups, 180 votes) complete —
30 auto-accept candidates + 30 human-queue groups, gating artifacts in
[research/agent_stitching_panel_eval.md](../../research/agent_stitching_panel_eval.md)
(Phase 2 section) and
[research/panel_phase2_audit_sheet.md](../../research/panel_phase2_audit_sheet.md).
**Phase-2 audit result: 7/10 held / 3 edited — BELOW the 9/10 export gate, so
export stays OFF.** A deterministic pedestrian↔vehicular class-consistency gate
was added (`stitch_runner.compute_consensus`, `route_reason="class-mismatch"`)
plus a rubric strengthening, but the 3 audited failures are same-mode vehicular
reject-alls — the gate catches 0/3 and demotes 0/30 candidates on this batch
(see the "Phase 2 audit outcome" section). Export decision deferred to a
mini-audit of the multi-edge M:N survivors. Companion to
[EVAL_ROADMAP.md](../EVAL_ROADMAP.md) step 1 and
[2026-02-21-stitch-eval-design.md](2026-02-21-stitch-eval-design.md).

## Problem

Stitching labels are the ground truth for the optimizer's assignment decisions
— the least-measured, most-heuristic stage of the pipeline. Only 33 (Boston) +
2 (Seattle) exist; the learned group resolver and stitch-level CI gate both
need 100+. Hand-labeling is taxing because the UI makes the human *construct*
each answer: every group opens with all segment pills active, and the reviewer
solves the M:N assignment by elimination, with decisions scaling with group
size.

## Key observations (from code exploration, 2026-07-03)

1. `optimizer_assignment` — the optimizer's own proposed edge set — is in every
   `.groups.json` sidecar group (`pipeline/runner.py:389`) but is never
   pre-seeded into the review UI pills.
2. `generate_top_k_alternatives` (`matching/alternatives.py:22`) already
   produces confidence-ranked candidate edge-sets per group, but `stitch-batch`
   strips them before the UI (`cli/data.py:2262` — "only needed for scoring,
   not UI"). The pick-an-option workflow is ~90% built and unused.
3. The agent pair-labeling harness (`agent_labeling/`, `cli/agent.py`: Claude
   CLI batch runner, image renderer, few-shot infra, output validation) is
   directly retargetable from pair verdicts to group option-picks. No agent
   stitching experiment has been attempted.
4. Per-edge signals are already serialized per sidecar edge: confidence, four
   alignment fractions, contiguity (derivable), match_type.
5. The pair-labeling sweep (`research/agent_eval_full_sweep.md`) showed
   agreement is the reliability signal: 89% accuracy when two variants agree,
   coin-flip on disagreement. The analogous signal here is agent choice vs.
   optimizer assignment.

## Design principle

**Verify, don't construct.** Choosing among enumerated options is faster and
more reliable than constructing an answer — for humans (seconds vs. minutes)
and for LLMs (multiple-choice with trivial validation vs. fragile free-form
edge-set generation).

## Plan

### Tier 0 — UI: pre-seed + option picking (no agents; do first)
- Stop stripping `alternatives` in `stitch-batch`; include top-K in the batch.
- Pre-seed pills from `optimizer_assignment` instead of all-active.
- Render options as one-click choices: "Option A (optimizer) / B / C / none of
  these"; picking an option sets the pills, which remain editable.
- Clean groups → one click; hard groups → comparison instead of construction.

### Tier 1 — agent option-picker
- New `matcher agent stitch-batch` path reusing the pair harness: render the
  group + spatial context (existing image renderer), metadata with per-edge
  features and the top-K edge-set options, prompt the agent to PICK an option
  (or "none") with confidence + reasoning. Option selection only — no
  free-form edge sets. Validation: answer must be one of the enumerated
  options (a runner must validate `choice` and retry once on garbage — a
  malformed CLI flag can silently become the prompt and still burn a call).

#### Evidence pack design (decided 2026-07-03)
Pre-prepped images + metadata on disk, NOT an agent rendering tool:
consensus requires identical evidence across providers (else disagreement
conflates evidence with judgment); provenance requires storing exactly what
the agent saw; render-once is cheaper than per-call tool rendering; and
files-on-disk is the only provider-agnostic interface. Per group (K ≤ 5):
1 context overview image + 1 image PER OPTION with that option's edges
highlighted ("which picture looks right") + metadata table (per-edge
confidence, alignment fractions, contiguity, match_type). Escape hatch: a
bounded "zoom to junction" call at fixed resolutions may be added later, but
ONLY if the Boston-33 eval shows errors concentrated in can't-see-detail
failure modes.

#### Provider panel (measured 2026-07-03, synthetic group test)
Empirical CLI probe results (all panel members 2/2 correct, valid JSON):

| CLI | Headless form | Image input | Latency |
|---|---|---|---|
| `claude -p --model haiku --allowedTools Read --json-schema '...'` (prompt via stdin) | Read tool on path | ~11s |
| `codex exec --skip-git-repo-check -s read-only --ephemeral -i img.png -o out.json "<prompt>" </dev/null` | native `-i` (repeatable — only CLI with native multi-image) | ~6s |
| `agy --print-timeout=2m --model="Gemini 3.5 Flash (Low)" -p "<prompt>"` (`=`-form flags REQUIRED) | reads image path via its view_file tool | ~5-16s |

- **Panel v1 (labeler `panel_unanimous_v1`): claude (sonnet) + codex (gpt-5.4,
  low) + agy (Gemini 3.5 Flash Low)** — the original composition.
- **Panel v2 (labeler `panel_unanimous_v2`, default as of 2026-07-05): claude
  (Opus 4.8, `--effort medium`) + codex (gpt-5.5, low) + agy (Gemini 3.5 Flash
  Medium)** — three heterogeneous model families. Composition change → labeler
  tag bump (v1 rows stay untouched; `stitch_export` excludes any `panel_*`
  labeler from human precedence). Validated against settled v1 groups before
  shipping (see `research/panel_v2_validation.md`).
- `gemini` CLI is DEAD for individual tiers (IneligibleTierError → Antigravity
  is the Google path). Do not use agy's Claude/GPT-OSS models in the panel
  (collapses diversity; GPT-OSS is text-only anyway). `opencode` not
  installed; open-weights multimodal is a later cost play if volume demands —
  subscription quotas comfortably cover hundreds of groups.
- Runner gotchas: claude — prompt via stdin (variadic flags swallow trailing
  positionals), never `--bare` (breaks Max OAuth), run from neutral cwd to
  avoid inheriting CLAUDE.md; codex — stdout is a transcript, always `-o`;
  agy — accumulates session transcripts under
  `~/.gemini/antigravity-cli/brain/`, prune periodically.

### Tier 2 — agreement routing
- Agent choice == optimizer assignment → auto-accept as an agent-tier
  stitching label (distinct `labeler` value for provenance weighting).
- Disagreement or low confidence → human queue, presented as a pre-highlighted
  A/B comparison.
- Clear-winner tier (top option dominates) → auto-accept with sampled audit.

### Validation gate (before trusting Tier 1/2)
- Run the agent against the 33 existing Boston labels (same pattern as the
  pair `test-batch`): measure option-pick accuracy; that number sets the
  auto-accept policy. Do not export agent stitching labels before this.

## Schema decisions
- Keep labels as edge sets only for now; defer alignment-fraction capture
  (matches the stitch-eval design doc's deferral).
- The UI's segment-level selection cannot express "keep R1→T1, drop R1→T2";
  option-based flow sidesteps this because options are true edge sets. Full
  per-edge UI detail remains a separate TODO.

## Sequence
1. Tier 0 UI PR (small; independent value)
2. Agent stitch-batch pipeline (moderate; mostly reuse)
3. Eval vs. 33 Boston labels → set thresholds (small)
4. Agreement routing + export with provenance tier (small)

Target: 13 → 100+ labeled groups within days of labeling effort, unblocking
the stitch-level CI gate and the learned group resolver.
