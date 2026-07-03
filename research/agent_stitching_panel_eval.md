# Agent Stitching Panel Eval — Boston 33 (2026-07-03)

First live run of the Tier 1+2 agent stitching-label pipeline
(`matcher agent stitch-batch` / `stitch-run` / `stitch-eval`) against the 33
existing human stitching labels for `us_boston_streets`.

**Framing (per the validation-gate design): disagreement with the old human
labels is NOT agent failure.** The labels are old (2026-02) and of varying
quality, and the group universe shifted under them (#227 changed model params,
re-forming components). Panel-vs-human contradictions below are label-quality
review candidates; the option-coverage analysis is the more decision-relevant
output.

## Setup

- Fresh `matcher stitch` run on Boston (1,993 groups in sidecar).
- Panel: `claude` (sonnet, stdin prompt + Read tool + `--json-schema`),
  `codex` (**gpt-5.4, `model_reasoning_effort=low`** — the default gpt-5.5 was
  deliberately avoided; gpt-5.1-codex/-mini/5.2-codex are rejected with a
  ChatGPT-account error, gpt-5.4 verified working with one cheap probe),
  `agy` (Gemini 3.5 Flash (Low), `=`-form flags).
- 3 providers x 11 groups run in parallel per group; per-provider timeout 180 s;
  one retry on invalid output (timeouts abstain immediately); every raw vote in
  `votes.csv` under the batch dir (audit data, separate from `labels/`).
- Nothing written to `labels/stitching/` — export policy is gated on this eval.

## Group-ID recovery (33 labeled → fresh sidecar)

Group IDs are hashes of the exact ref/target ID sets, so component drift breaks
exact recovery:

| Recovery mode | Count |
|---|---|
| Exact group_id present in fresh sidecar | **6 / 33** |
| Edge-overlap recovery: all human edges inside one current group ("clean") | 9 |
| Human edges split across current groups | 3 |
| Human label has empty edge set (reject-all decisions — no segment ids stored) | 20 |
| Non-empty label whose edges no longer exist as candidate edges | 1 |

The 9 clean + 3 split labels map to **11 distinct current groups**; the eval
batch was built from those via `stitch-batch --recover-labeled`. 10 of the 11
mapped back to a human label at eval time (the 52x32 mega-group absorbed two
old labels; one mapping wins).

Takeaway: the 20 empty-edge-set labels are unusable for option-pick evaluation
(they record *that* everything was rejected but not *which segments* the group
contained). Any future labeling should persist the group's candidate edges
alongside the selection — the agent pipeline's `votes.csv` already does.

## Results

### Consensus routing (11 groups)

| Consensus | N | Routing |
|---|---|---|
| unanimous | 4 | auto-accept candidate |
| majority | 4 | human review |
| none | 3 | human review |

One abstention total (claude timeout on the 52x32-segment mega-group).

### Agreement with human labels (10 mapped groups)

| Scope | Exact edge-set match | Mean edge F1 |
|---|---|---|
| Panel consensus | 20% | 0.540 |
| — unanimous tier (n=3) | **67%** | **0.933** |
| — majority tier (n=4) | 0% | 0.412 |
| — none tier (n=3) | 0% | 0.317 |
| claude (n=10) | 20% | 0.568 |
| codex (n=10) | 20% | 0.414 |
| agy (n=10) | 20% | 0.490 |

The consensus tiers behave exactly as the pair-labeling sweep predicted:
**agreement is the reliability signal.** Unanimous groups are near-perfect
(the one non-exact unanimous case is a group-drift artifact, see below);
majority/none tiers are where the real ambiguity lives, and they route to
humans.

### Option-coverage gap: 8/10 (80%) — the dominant effect

For 8 of 10 groups the human edge set matches **no current option**, making
exact agreement impossible by construction. Two distinct causes:

1. **Group-composition drift (post-#227).** Current groups are supersets or
   re-cuts of the old labeled components. E.g. the human 2-edge label
   `efb92447` now lives inside the 52x32 mega-group `5d489b4e`; human 48-edge
   label `a89e4b84` maps to a current 21-edge group. The panel answered a
   different (larger) question than the human did.
2. **A real generator limitation.** `generate_top_k_alternatives` assigns each
   target to at most one ref in 1:N/M:N enumeration, so an M:N option where one
   target spans TWO refs (e.g. "T4 covers both R3 and R5") is inexpressible.
   Two panelists independently spotted this and voted NONE with explicit
   reasoning (below) — the NONE escape hatch works as designed and localizes
   the gap.

### Where the human edge set WAS option-covered (n=2)

Both groups (`2170ab83`, `72063362` — also the only two whose group_id survived
verbatim) → unanimous, exact match, F1 = 1.00. Small sample, but perfectly
consistent with the unanimous-tier signal.

### Per-group detail

| Panel group | Human label | Type | Consensus | Choice | Exact | F1 | Option-covered |
|---|---|---|---|---|---|---|---|
| 2170ab83 | 2170ab83 | N:1 | unanimous | A | yes | 1.00 | yes |
| 72063362 | 72063362 | N:1 | unanimous | A | yes | 1.00 | yes |
| f6e71865 | 3ac36248 | M:N | unanimous | A | no | 0.80 | no |
| 9ac35fb7 | f452e052 | M:N | majority | A | no | 0.89 | no |
| 874eccdf | bb93702d | M:N | majority | A | no | 0.50 | no |
| 68b9b487 | a89e4b84 | M:N | majority | C | no | 0.26 | no |
| dde5ebf9 | ed93d8b9 | M:N | majority | NONE | no | 0.00 | no |
| 8ed3be51 | 21b67ef2 | M:N | none | A | no | 0.50 | no |
| 95e09c60 | cc0c30a0 | M:N | none | D | no | 0.33 | no |
| 5d489b4e | efb92447 | M:N | none | D | no | 0.12 | no |

## Disagreement highlights (label-quality review candidates)

- **`dde5ebf9`** (human `ed93d8b9`, 2-edge N:1 label): majority NONE
  (codex 0.82, agy 0.95) vs claude A. Both NONE voters argued the true
  assignment needs `R3->T4` AND `R5->T4` (Custom House Street split across two
  refs) plus the other correspondences — a shape no option can express (cause 2
  above). The 2-edge human label is likely under-selected for the current group.
- **`8ed3be51`** (human `21b67ef2`): none (claude A 0.55 / codex B 0.95 /
  agy NONE 1.0). agy: "T1 (Water Street) matches both R1 and R3, T2
  (Batterymarch) matches both R2 and R4 — a correct match requires all 4 edges;
  no option has them." Same generator limitation; the human 2-edge label covers
  only half the correspondence.
- **`f6e71865`** (human `3ac36248`): unanimous A (mean conf 0.92) with F1 0.80 —
  the panel includes one extra edge (`7c5b4138->2186`) because the current
  group merged an additional ref segment the human never saw. Drift, not error.
- **`9ac35fb7`** (human `f452e052`, F1 0.89): majority A; codex voted NONE
  arguing T3 is genuinely split across R1+R4 (again the inexpressible shape).
  The human label and panel differ by exactly that junction.
- **`5d489b4e`** (52x32 mega-group): claude timed out at 180 s, codex D 0.69 /
  agy A 0.95 → none. Groups this large are not sensibly answerable as one
  multiple-choice question; they should be pre-filtered to human review (or
  decomposed) rather than spent on the panel.

## Latency / cost

| Provider | Model | Mean | Median | Max |
|---|---|---|---|---|
| agy | Gemini 3.5 Flash (Low) | 11.4 s | 11.7 s | 22.7 s |
| codex | gpt-5.4 (low effort) | 25.0 s | 25.7 s | 46.0 s |
| claude | sonnet | 67.3 s | 35.0 s | 170.5 s |

Panel wall time is claude-bound; the full 11-group run took ~25 min including
the mega-group timeout. All three CLIs ran on existing subscriptions. One
operational finding: agy writes derived crop images next to the images it
inspects, so the runner gives each invocation an isolated scratch copy of the
pack (canonical evidence dirs stay pristine).

## Recommended auto-accept policy

1. **Auto-accept only unanimous (3/3 valid, non-NONE) verdicts**, exported with
   `labeler=panel_unanimous_v1` so stitch-eval can filter/weight by provenance.
   Evidence: unanimous tier = 0.933 F1 overall and 2/2 exact on option-covered
   groups; majority tier = 0% exact — 2/3 agreement is NOT good enough to
   auto-accept for this task, unlike pair labeling.
2. **Gate the panel on group size** — skip groups with more than ~20 edges
   (route straight to human review). The mega-group produced one timeout, one
   coin-flip disagreement, and consumed the most tokens of the whole run.
3. **Fix the option generator before scaling up**: allow per-target multi-ref
   edges in M:N enumeration (the "T4 spans R3+R5" shape). Two of three NONE
   verdicts trace to exactly this gap; fixing it should convert several
   majority/none groups into clean unanimous picks.
4. **Do not export anything yet** (and this run didn't): after (3), re-run the
   panel on a fresh 30-50 group batch, hand-review the unanimous tier once, and
   if the sampled audit holds (>90% human agreement on option-covered groups),
   turn on unanimous auto-accept.
5. The 20 old human labels with empty edge sets cannot participate in this
   eval; treat the panel + human-review queue as the path to replacing them.
