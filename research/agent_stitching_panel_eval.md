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

Eval-time mapping methodology: a human label maps to a panel group only if at
least one of its selected edges exists among the group's candidate edges (from
the batch file), with ties broken by max edge overlap; segment-ID membership is
only a fallback for packs without a batch file. This prevents a label whose
specific edges no longer exist from mis-mapping via shared segments and skewing
the coverage/agreement numbers. (Both mapping rules produce identical results
on this batch, since the eval groups were selected by edge-overlap recovery in
the first place.)

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

---

# Phase 2 — Production-Sized Run with the Fixed Option Generator (2026-07-03)

Re-run of the panel at production scale after #231 (multi-ref M:N option
enumeration) landed, gating the enablement of unanimous auto-accept export.
Batch: `data/agents/stitching/batches/us_boston_streets_phase2` (60 groups,
180 provider calls, same panel: claude sonnet / codex gpt-5.4 low effort /
agy Gemini 3.5 Flash (Low), 240 s timeout).

## Run-set composition (60 groups)

| Stratum | N | Selection |
|---|---|---|
| (a) eval continuity | 11 | `--recover-labeled` edge-overlap recovery of the 13 non-empty human labels (9 clean + 3 split → 11 distinct current groups) |
| (b) reject-all continuity | 4 | empty-edge human labels whose `group_id` survives verbatim in the fresh sidecar (new `recover_empty_reject_all` / `--recover-empty`) |
| (c) fresh fill | 45 | standard tier selection: 13 large / 13 borderline / 9 low-confidence / 10 clear-winner |

**16 of the 20 reject-all labels are unrecoverable**: they stored no segment
IDs (only the group-id hash of the original ref/target sets), and post-#227
component drift dissolved those exact sets. Only exact-hash survival can
recover them; 4 survived. This permanently confirms the Phase-1 takeaway —
labels must persist candidate edges.

The fresh sidecar was regenerated on #231 code (1,993 groups; `matcher stitch`
+ `matcher agent stitch-batch --group-ids-file`). 43/60 packs now contain at
least one multi-ref option (127/209 options total) — the shape that was
inexpressible in round 1.

## Results

### Consensus and routing (60 groups, 180 votes)

| Consensus | N |
|---|---|
| unanimous | 40 |
| majority | 14 |
| none | 6 |

Applying the recommended policy (auto-accept = unanimous non-NONE AND ≤20
candidate edges):

| Final routing | N | Why |
|---|---|---|
| **auto-accept candidate** | **30** | unanimous non-NONE, ≤20 edges (mean panel confidence 0.916; group sizes 1–18 edges) |
| human review | 30 | majority 14, size-gated unanimous 9 (39–92 edges), no consensus 6, unanimous NONE 1 |

Abstentions: 3 (all claude 240 s timeouts, all on 13+-edge groups, all of
which routed to human review anyway). No auth/quota failures; agy and codex
returned 60/60 valid votes each.

31 of the 57 non-NONE consensus picks chose a multi-ref option — the fixed
generator's new options are not decorative; the panel actively selects them.

### Eval refresh: labeled groups, round 1 vs Phase 2

Same 10 mapped human labels (11 recovered groups, mega-group absorbs two):

| Metric | Round 1 (pre-#231) | Phase 2 (post-#231) |
|---|---|---|
| Unanimity on eval groups | 4/11 (36%) | **9/10 (90%)** |
| NONE consensus verdicts | 1 (+2 NONE minority votes citing the inexpressible multi-ref shape) | 0 |
| Panel exact edge-set match | 20% | 20% |
| Panel mean edge F1 | 0.540 | 0.590 |
| Exact on option-covered groups | 2/2 | 2/2 (same two, F1 = 1.00) |
| Option-coverage gap | 8/10 | 8/10 |

Reading this correctly: **the generator fix converted disagreement into
agreement exactly where predicted.** `dde5ebf9` — round 1's poster child,
where two panelists voted NONE because "T4 needs R3 AND R5" was inexpressible
— is now a unanimous pick of an option containing exactly R3→T4 + R5→T4.
`8ed3be51` (none → unanimous) and `9ac35fb7` (majority w/ NONE dissent →
unanimous) follow the same pattern.

Agreement with the *old human labels* did not move (20% exact), because it is
capped by the unchanged option-coverage gap: 8/10 human edge sets answer a
different (smaller, pre-#227) group than the current sidecar poses. The two
groups where the question is unchanged remain perfect. The old round-1
"unanimous tier = 67% exact / 0.933 F1" was computed on n=3; the Phase-2
unanimous tier (n=9) now contains the drift-affected groups that round 1
routed to majority/none, so the tier numbers are not comparable — the
option-covered subset (2/2 exact both rounds) is the like-for-like signal.

Remaining NONE/dissent reasoning is now genuinely judgmental (parallel roads,
end-to-end junctions, wrong geometry) rather than "no option can express the
true assignment". Residual coverage complaints concentrate in large many-ref
corridors (e.g. "I Street needs R19+R5+R9+R21") — beyond the per-target chain
cap, and all in size-gated or no-consensus groups already routed to humans.

### The 20 reject-all human labels: the panel contradicts them

The user's suspicion was that the old empty-selection (reject-all) labels are
low-quality. The 4 recoverable ones — where the group is *identical* to what
the human saw — say yes loudly:

| Group | Human said | Panel says |
|---|---|---|
| `99bc755f` | reject all | **unanimous B** (claude/codex/agy all B) |
| `b2876328` | reject all | **unanimous A** |
| `461ebf00` | reject all | **unanimous A** |
| `5e8dad61` | reject all | majority A (codex C — dissent over *which* assignment, not whether) |

0/4 panel verdicts lean NONE. All three providers independently propose a
concrete assignment on every one of these groups. Combined with the round-1
finding that the reject-all labels stored no segments, these labels should be
treated as superseded: route their groups through the panel + human-review
path rather than weighting them in stitch-eval. (Two of the three unanimous
ones are in the Phase-2 audit sample — `b2876328` directly — so the human
audit will double-check this conclusion.)

### Latency

| Provider | Mean | Median | Max |
|---|---|---|---|
| agy | 11.5 s | 9.7 s | 38.8 s |
| codex | 21.4 s | 15.9 s | 87.6 s |
| claude | 56.5 s | 26.1 s | 220.4 s (3 timeouts at 240 s) |

Full 60-group run: ~65 min wall, claude-bound. All on existing subscriptions.

## Gating artifacts

- `data/agents/stitching/batches/us_boston_streets_phase2/consensus.csv` —
  per-group routing incl. `final_routing`/`route_reason` (size-gated policy).
- `data/agents/stitching/batches/us_boston_streets_phase2/votes.csv` — all
  180 raw votes with reasoning (audit trail).
- [research/panel_phase2_audit_sheet.md](panel_phase2_audit_sheet.md) — the
  10-group seeded audit sample (seed 20260703) with per-provider reasoning and
  stitching-review UI links. **Export stays OFF until this audit passes.**
- Human-review queue: 30 groups (14 majority, 9 size-gated, 6 no-consensus,
  1 unanimous-NONE) — viewable in the stitching-review UI (the Phase-2 batch
  was mirrored to `data/cache/stitch/us_boston_streets_batch.json`).

## Recommendation

1. **Audit the 10-group sample.** If ≥9/10 hold up, enable unanimous
   auto-accept export (`labeler=panel_unanimous_v1`, ≤20-edge gate) and export
   the 30 candidates — that alone takes Boston from 13 usable labels to 43+.
2. Keep the ≤20-edge gate permanently: every timeout and all 6 no-consensus
   verdicts came from 13+-edge groups (4/6 from 36+). Unanimity above the gate
   (9 groups, 39–92 edges) is partly option-count collapse — 3 of the 9 packs
   offered only a single expressible option, making unanimity trivial — so
   those stay human.
3. Treat the 20 reject-all labels as superseded (see above) — do not use them
   as eval truth.
4. Re-run `stitch-eval` after the human queue is worked to grow the
   option-covered eval set beyond n=2.

---

# Phase 2 audit outcome (2026-07-03) — audit FAILED the 9/10 gate

The human audit of the 10-group auto-accept sample came back **7/10 held /
3 edited** — below the 9/10 export gate, so **export stays OFF**. The user
labeled exactly the 10 audit-sample groups (fresh human labels in
`labels/stitching/dataset=us_boston_streets/data.csv`, `labeler=brad`,
`labeled_at` 2026-07-03) and did not label beyond them — the "group 15/60" in
the UI was just the next group the UI advanced to, not a submitted label.

## What the human actually changed (label diff vs. panel consensus)

| Group | Panel pick | Human action | Held? |
|---|---|---|---|
| 63bf7e48, 701d491e, 72063362, b2876328, f69a827e | A / A / A / A / A | **exact match** (5) | held |
| 16947985 | B (2 edges) | added 1 edge (panel under-selected; no wrong edge) | held |
| 9ac35fb7 | E (6 edges) | added 2 edges (panel under-selected; no wrong edge) | held |
| **04fc93e5** | B (5 edges) | **reject-all** (removed all 5) | **edited** |
| **79711407** | A (4 edges) | **reject-all** (removed all 4) | **edited** |
| **f170979a** | C (6 edges) | **reject-all** (removed all 6) | **edited** |

7 held = 5 exact + 2 where the panel's chosen edges were a correct *subset*
(human only *added* edges, never removed one). 3 edited = the reject-alls,
where the human dropped every panel edge.

## Failure-mode table — the reported pedestrian pattern is REFUTED

The reported failure mode was "false-positive pedestrian-class reference
(footway/sidewalk) paired with road-class targets." The removed-edge class
pairs across the 3 edited groups do **not** show that pattern — every removed
edge is vehicular↔vehicular:

| Group | Removed edges | ref_class → target_class | Cross-mode? |
|---|---|---|---|
| 04fc93e5 | 5 | residential→residential (4), service→residential (1) | no |
| 79711407 | 4 | secondary→primary (4) | no |
| f170979a | 6 | residential→residential (6) | no |

Crucially, `f170979a` **does** contain footway reference segments (3 of them,
"Bragdon Street" footway) and the batch offered `footway→residential` candidate
edges — but the panel's chosen option C **already excluded every footway edge**
and matched only residential↔residential. The human then rejected those
residential edges too. So the panel did not commit a pedestrian-vs-road false
positive on any audited group; the 3 failures are same-mode M:N / reject-all
disagreements (option-coverage or wrong sub-segment assignment), a different
failure mode than the one reported.

Batch-wide confirmation: **0 of 60 groups** have a cross-mode edge in their
*chosen* edge set, and **0 of 30 auto-accept candidates** do. Three auto-accept
candidates (`36726195`, `f170979a`, `f6e71865`) merely *contain* a footway
group segment; in all three the panel excluded the footway candidate edges.

## Deterministic class-consistency gate (shipped this PR)

Added to the routing step (`stitch_runner.compute_consensus`, gated on
`edge_classes`): an auto-accept candidate whose *chosen* edge set contains a
cross-mode edge — an unambiguously pedestrian class on one side and an
unambiguously vehicular class on the other — is demoted to human review with
`route_reason="class-mismatch"`. Mode sets are module constants
(`PEDESTRIAN_CLASSES` = footway/sidewalk/path/pedestrian/steps/crossing;
`VEHICULAR_CLASSES` = motorway/trunk/primary/secondary/tertiary/residential/
service/unclassified/living_street/driveway/road). cycleway, track, alley,
unknown and missing classes are NEUTRAL and never trigger the gate (no
over-gating on ambiguous/absent data; `alley` never appears, cycleway/track are
genuinely ambiguous). The prompt rubric was also strengthened to forbid
matching a footway/sidewalk/path to a road class merely for being parallel.

## Gate validation (recomputed over the recorded votes, no new LLM calls)

Reran `compute_consensus` over the existing `votes.csv` with the gate active
(size ≤20 gate reapplied identically):

- Sanity: reproduces the original **30/30** auto-accept candidates without the
  gate — the recompute is faithful.
- Class gate demotes **0 of 30** candidates → **30 survive**.
- **The gate catches 0 of the 3 user-corrected groups.** Because those
  corrections are same-mode vehicular reject-alls, no deterministic
  pedestrian↔vehicular gate can catch them. **This is the validation that
  matters, and it fails:** the class gate is sound and safe but does NOT explain
  or mitigate the 7/10 audit result on this batch. It is retained as
  forward-looking insurance for future batches where the panel might pick a
  cross-mode edge, and it correctly does **not** over-gate the two survivors
  that merely contain footway segments.

## Surviving candidates and export status

All 10 audit-sample groups (including the 3 edited) now carry fresh human
labels and are **superseded** — excluded from any panel export. That leaves
**20 surviving auto-accept candidates** (unanimous non-NONE, ≤20 edges,
class-consistent, not human-labeled) as the *proposed* export set:

`0d8c40ca, 166ce59a, 172050db, 2170ab83, 2802e4db, 36726195, 461ebf00,
5702414b, 6dde01fe, 6e5f877a, 747a7f1a, 7e218abf, 874eccdf, 8ed3be51,
99bc755f, be597061, bf0760b1, dde5ebf9, f4984915, f6e71865`

Of these, 11 are trivial 2-edge N:1/clear-winner picks (low risk); the 9
multi-edge M:N survivors (`0d8c40ca` 8/18, `874eccdf` 11/11, `6dde01fe` 6/10,
`747a7f1a` 6/10, `dde5ebf9` 6/8, `2802e4db` 4/6, `8ed3be51` 4/5, `f6e71865`
3/5, `99bc755f` 3/4) are where the 7/10 failure mode (reject-all on
plausible-looking M:N corridors) would recur.

## Recommendation

Because the audit missed the 9/10 gate **and** the deterministic gate does not
catch the observed failure mode, **do not enable blanket unanimous auto-accept
export yet.** The 3/10 reject-alls indicate ~30% of unanimous M:N auto-accepts
may be reject-worthy for reasons no class gate can detect. Recommended next
step: a **5-8 group mini-audit of the 9 multi-edge M:N survivors** above; if
that holds ≥90%, export the survivors (still `labeler=panel_unanimous_v1`), and
consider auto-accepting only the trivial ≤2-edge N:1 tier without further audit.
The class gate and rubric change ship regardless as defense-in-depth.
