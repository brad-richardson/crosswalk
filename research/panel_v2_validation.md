# Agent Stitching Panel v2 — Validation

**Date:** 2026-07-05
**Goal:** Upgrade the 3-provider agent-stitching panel to new default models and
verify the new panel (v2) still agrees with settled `panel_unanimous_v1` rounds
before shipping it as the default.

## Panel composition

| Provider | v1 (labeler `panel_unanimous_v1`) | v2 (labeler `panel_unanimous_v2`) |
|---|---|---|
| claude | `sonnet` | `claude-opus-4-8`, `--effort medium` |
| codex  | `gpt-5.4`, reasoning `low` | `gpt-5.5`, reasoning `low` |
| agy    | `Gemini 3.5 Flash (Low)` | `Gemini 3.5 Flash (Medium)` |

### Model-availability findings (probed on this machine)
- **claude**: `claude-opus-4-8` (alias `opus`) + `--effort {low,medium,high,xhigh,max}`
  both work headless. Pinned the explicit `claude-opus-4-8` id (not the drifting
  `opus` alias) so the v2 labeler tag stays tied to a fixed model.
- **codex**: `gpt-5.5` works via `codex exec ... -m gpt-5.5 -c model_reasoning_effort=low`.
  No usage cap encountered during this run.
- **agy**: `agy models` lists `Gemini 3.5 Flash (Medium)`; `=`-form flag required
  (`--model=...`). The `gemini` CLI remains dead for individual tiers.

All three returned valid JSON on a trivial probe; no abstentions or errors across
the 36 validation votes.

## Validation method
- Picked 12 SETTLED groups (labeler `panel_unanimous_v1`) whose `group_id` still
  exists verbatim in the current sidecars (`data/output/{us_boston_streets,
  us_seattle_sidewalks}_groups.json`, regenerated today) with the settled edge
  set still expressible as a subset of the current group's candidate edges:
  **8 Boston + 4 Seattle**, mixed sizes (1:N/N:1 up to 15-edge M:N).
- Generated NEW evidence batches (`*_panelv2check`) — new dirs only; no existing
  batch/cache/output/label file touched.
- Ran the v2 panel with a resumable per-group driver (persists after each group;
  no cap protection needed this run).
- Compared votes against the settled labels: per-group exact edge-set agreement,
  sliver-filtered edge F1 (`matcher.matching.sliver` / `matcher.config.is_sliver_edge`),
  new-trio unanimity, per-provider dissent, latency. **No labels exported.**

## Results

### Per-group (settled v1 label vs v2 panel)

| Dataset | group | match | v2 consensus | choice | exact | F1 | votes (dissent) |
|---|---|---|---|---|---|---|---|
| boston | 07632e1f | M:N | unanimous | A | ✅ | 1.00 | A/A/A |
| boston | 0b3a4f7d | M:N | unanimous | A | ✅ | 1.00 | A/A/A |
| boston | 0e3e10ad | M:N | majority | A | ✅ | 1.00 | claude A, agy A, **codex B** |
| boston | 166ce59a | N:1 | unanimous | A | ✅ | 1.00 | A/A/A |
| boston | 37a546e3 | M:N | unanimous | A | ❌ | 0.95 | A/A/A (option-coverage gap) |
| boston | 461ebf00 | 1:N | unanimous | A | ✅ | 1.00 | A/A/A |
| boston | 4bcea059 | M:N | unanimous | A | ✅ | 1.00 | A/A/A |
| boston | c3a963e9 | M:N | unanimous | A | ✅ | 1.00 | A/A/A |
| seattle | 2b99c180 | M:N | majority | C | ❌ | 0.71 | claude C, agy C, **codex B** |
| seattle | 46e57794 | M:N | majority | E | ❌ | 0.93 | claude E, codex E, **agy A** |
| seattle | 670e939f | M:N | majority | D | ❌ | 0.89 | claude D, codex D, **agy A** |
| seattle | e919f4ab | 1:N | unanimous | A | ✅ | 1.00 | A/A/A |

### Aggregate

| Metric | Boston (n=8) | Seattle (n=4) | Combined (n=12) |
|---|---|---|---|
| exact agreement (raw = sliver-filtered) | 87.5% (7/8) | 25% (1/4) | 67% (8/12) |
| mean edge F1 | 0.994 | 0.882 | 0.957 |
| new-trio unanimity | 87.5% (7/8) | 25% (1/4) | 67% (8/12) |
| option-coverage gap (settled set not offered) | 1/8 | 2/4 | 3/12 |

**Export-relevant number:** only *unanimous auto-accept* groups ever become
labels. Of the 8 unanimous groups, **7/8 (88%) exactly match the settled
label** — the one miss is an option-coverage artifact, not a disagreement.

### Per-provider

| Provider | exact vs settled | mean F1 | mean confidence | median latency |
|---|---|---|---|---|
| claude (opus 4.8) | 67% (8/12) | 0.957 | 0.79 | 29.9s |
| codex (gpt-5.5) | 58% (7/12) | 0.940 | 0.92 | 10.0s |
| agy (Gemini 3.5 Flash Med) | 67% (8/12) | 0.947 | 0.98 | 16.3s |

Latency range: codex 6.5–22s, agy 6.6–88s, claude 19–50s. No caps, no
abstentions. Wall time per group is set by the slowest provider (parallel).

### Dissent pattern (baseline: codex lone holdout on roads; agy on sidewalks)
- **Roads (Boston):** the single split (0e3e10ad) had **codex** as the lone
  holdout — matches the v1 baseline. Majority choice still matched settled exactly.
- **Sidewalks (Seattle):** on 2/3 splits (46e57794, 670e939f) **agy** was the lone
  holdout — matches the v1 baseline. On 2b99c180 codex was the odd vote.
- Notable: on 46e57794 agy's "dissent" (option A, 18 edges) had **full recall of
  the settled set** (all 15 settled edges present, +3 extra → precision 0.83,
  F1 0.91), while the claude+codex majority (E, 15 edges) was 14/15 (F1 0.93).
  Neither is exact because the settled 15-edge set is not an offered option; agy's
  dissent is a reasonable superset, not an error.

## Divergence investigation

- **boston 37a546e3** (unanimous, F1 0.95): the settled 11-edge set is **not
  expressible by any current top-K option** (`option_covered=False`). The panel
  unanimously chose the best available option A (10/11 edges), dropping one edge
  that no option offered. Root cause is top-K alternatives coverage, not the
  panel. Benign.
- **seattle 2b99c180 / 670e939f / 46e57794** (majority → human_review, NOT
  exported): genuinely ambiguous sidewalk M:N groups with 5–18 near-equivalent
  candidate options differing by 1–2 edges. For 2/3 the settled label isn't even
  the best-matching option (`option_covered=False`). The v2 panel appropriately
  fails to reach unanimity and routes to human review — safer, not a regression.
  None of these produce a v2 label that could conflict with the v1 label.

## Ship decision: **SHIP**

- On the only groups that become labels (unanimous auto-accept), v2 agrees with
  the settled labels 7/8 (88%); the sole miss is an option-coverage artifact.
- No systematic regression: there is no case where v2 is unanimously wrong. Where
  v2 diverges it is either (a) an option-coverage gap the panel cannot express, or
  (b) genuine ambiguity where v2 is *more conservative* (breaks v1 unanimity →
  human review), which is the safe direction.
- Composition change ⇒ new labeler tag `panel_unanimous_v2` (v1 rows untouched;
  `stitch_export` excludes any `panel_*` labeler from human precedence).

## Follow-ups
- First real v2 labeling wave (proposed debut: the **Berlin roads** batch —
  `data/output/de_berlin_roads_groups.json`).
- Improve top-K alternatives coverage: 3/12 groups (esp. Seattle sidewalks) have
  a settled edge set that no offered option can express — raising K or adding a
  full-candidate option would close the artifact gaps seen here.
- Consider recording per-provider effort in `votes.csv` for future audits (model
  string is recorded; effort currently is not).
