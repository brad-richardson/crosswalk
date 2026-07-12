# Learned group resolver — design (candidate-graph training + constrained decode)

**Date:** 2026-07-06 · **Status:** design (M3 of `docs/SCALING_ROADMAP.md`), no code
**Prior rounds:** `research/learned_group_resolver_prototype.md` (round 1, PR #272) ·
`research/learned_stitcher_round2.md` (round 2, 2026-07-05)

This is the design for the third attempt at replacing the heuristic M:N stitch
edge selection with a learned group resolver. Rounds 1 and 2 both ended NO-GO,
and both times the reasons were quantified: ~40 clean labeled groups (round 1),
under-selection unlearnable because the sidecar persisted only optimizer-selected
edges (round 1, fixed by #282/#284), a one-parameter confidence prune that
absorbed most of the available headroom (shipped as `apply_confidence_drop_prune`,
now the production baseline), and label anchoring — reviewers grade proposals the
optimizer wrote (round 2, unfixed). This document designs the round-3 system so
that when the flip conditions from the roadmap hold (full candidate graph
persisted with features; 150–300 labeled groups), the model, the data plumbing,
the integration point, and the promotion bar are already agreed.

The honest prior: **production is currently winning.** Round 2's group bootstrap
on the clean slice put production (optimizer + tuned prune) ahead of the best
learned config by ΔF1 +0.026, 95% CI [−0.000, +0.055], P(learned ≥ production)
≈ 0.028. Nothing below assumes that gap closes; everything below is about making
the comparison fair (de-anchored labels, full candidate features), giving the
model the data it was starved of, and defining exactly what "beat it" means.

---

## 1. Problem statement & scope

### What the heuristic pipeline does today

`optimize_matches_with_grouping` (`src/crosswalk/matching/optimizer.py`, ~2 s of
a 7-min Berlin run) resolves scored candidates into 1:1 matches and 1:N / N:1 /
M:N groups:

1. Connected components over candidates above `min_confidence`
   (`find_match_components`), with the corridor-validated glue prune at
   calibrated 0.575 (#267; ARCHITECTURE.md "Glue-prune calibration equivalence").
2. `_classify_and_resolve_component` (optimizer.py:766–966) classifies each
   component. Case 4 (M:N) runs corridor-aware contiguity
   (`_find_contiguous_id_groups` with `require_collinear` + same-name rescue) on
   both sides and decomposes non-contiguous monsters into per-(ref-corridor ×
   target-corridor) subgroups.
3. `_create_group_results` (optimizer.py:487) tags every surviving edge of a
   subgroup with `group_id` — **within a decomposed group, selection is
   keep-all**. The only within-group edge decision made anywhere is:
4. `apply_confidence_drop_prune` (optimizer.py:1251), the post-optimizer
   one-parameter absolute-confidence prune, applied per-dataset via the
   `resolver_prune_overrides` allowlist (`pipeline/runner.py::
   _effective_prune_threshold`, step 4.5 at runner.py:1109–1133). Shipped
   floors: Boston 0.96 (filtered edge-F1 0.8671 → 0.8790, group exact 0.5093 →
   0.5833), Seattle 0.90 (0.8665 → 0.8913, 0.40 → 0.50) (#282/#284).

Downstream, `compute_group_structure` (optimizer.py:1337) derives the structural
layer (degree, bridges, biconnected blocks, corridor ids) that is persisted in
the sidecar, and `group_is_structurally_simple` (optimizer.py:1302) gates what
the panel/export path may auto-handle; monster groups (>20 edges, ~0.1–0.5% of
groups) route to humans.

### What the learned resolver replaces — and what it does not

**Replaces: the within-group edge-subset decision only** — i.e. the keep-all
default of `_create_group_results` plus the confidence-drop prune. Given a
group's *full candidate edge set* (selected + rejected + pruned), the resolver
outputs the kept subset. The empty set is a valid output (required for the
cross-mode defect, §4.3).

**Stays heuristic** (deliberately, all validated independently):

| Stage | Why it stays |
|---|---|
| Candidate generation + `min_confidence` floor | Recall source; round 2 showed the learned models' weakness is recall, not precision |
| Glue prune + corridor-aware Case-4 decomposition | #267 validated; oversized groups ≤ 0.07% everywhere measured (#274). Group *boundaries* are a different, much harder structured problem with ~0 labels for "this decomposition was wrong" |
| Greedy 1:1 + `_expand_greedy_matches` + coverage-conflict demotion | 1:1 matches were never in the resolver eval's scope and are the bulk of output |
| Sliver detection (`matching/sliver.py`) | Feeds the metric filter and the structural layer; orthogonal |
| Monster routing to humans (`group_is_structurally_simple`) | Unchanged in phase 1; see §7 |

This scoping matches where the measured errors live: the panel's early-signal
sweep (`research/early_signal_panel_2026_07.md`) found the optimizer's M:N
selection "measurably too greedy" — every non-optimizer panel pick was a
*strictly smaller* edge set (Tunis 67→35, Missoula 67→34, plus four 2→1 prunes)
— and found whole groups that should be empty (cross-mode, §4.3). Both are
within-group subset errors, not decomposition errors.

### Relationship to the shipped prune

The resolver is the **general form of the prune**: the prune is a
one-feature, one-parameter, monotone special case. It follows that the resolver
must strictly beat the *tuned* prune on each dataset before it can replace it
there, and that the rollout mechanism should be the same per-dataset allowlist
pattern (§5, §6). Where the resolver is not enabled, the prune (or keep-all for
never-tuned datasets) remains exactly as today.

---

## 2. Training data

### 2.1 Sources

Three label sources, in decreasing trust order:

1. **Curated stitching labels** — `labels/stitching/dataset=*/data.csv`
   (`group_id, dataset_id, selected_edges (JSON list of {ref_id, target_id}),
   match_type, num_refs, num_targets, labeler, labeled_at, session_id,
   label_semantics, ref_ids, target_ids`). On file today: 183 rows across 9
   datasets — Boston 117 (35 `brad`, 82 `panel_unanimous_v1`; 112 pair / 5 set),
   Seattle 46, plus 2–5 each on Bogotá bike, Tunis, Missoula, Singapore roads /
   footpaths, Flathead, Berlin (the #291-era `panel_unanimous_v3` waves). Of
   these, ~140 map to current sidecar groups (113 Boston / 27 Seattle per the
   roadmap; mapping is by edge overlap via
   `agent_labeling/stitch_eval.recover_labeled_groups`, robust to `group_id`
   churn). Only `label_semantics == "pair"` rows produce per-edge keep/drop
   truth; `set` rows (membership-only, #295/#298) contribute membership
   constraints but no edge labels and are excluded from the edge table, as the
   gate already does.
2. **Panel-unanimous weak labels** — exported unanimous verdicts already land in
   (1) with a `panel_unanimous_*` labeler stamp. Round 1 measured their value
   directly: +0.05 clean OOF F1 from adding 62 panel-soft groups to training,
   the single biggest lever observed. Keep them in training; slice them out of
   the headline eval (report per-labeler, as both rounds did).
3. **Raw panel votes** — `labels/votes/dataset=*/votes.csv` + `consensus.csv`
   (#333; schema = `stitch_runner.VOTES_COLUMNS` / `CONSENSUS_COLUMNS`:
   per-provider `choice`, `edge_set`, `confidence`, plus per-group `consensus`,
   `minority`, `n_valid`). These enable **soft per-edge targets and sample
   weights for non-unanimous groups** that never became labels: per-edge soft
   keep = reliability-weighted provider vote share (the round-1
   `resolver/votes.py` recipe: codex down-weighted 0.5 on roads, agy on
   sidewalks, per the dissent analysis in the roadmap M1).

### 2.2 Row derivation

One training row per (group, candidate edge), built by extending the existing
`resolver/extract.py` + `round2.py` table builder:

- Map each pair-semantics label to its current sidecar group by edge overlap;
  keep the `clean` / `split` / `lost` provenance tags. Only `clean` rows enter
  the headline eval; `split` rows may enter training down-weighted (round 2
  precedent).
- Candidate universe per group = the #344 `candidate_edges` list (uncapped;
  supersedes `edges` ∪ `rejected_edges` as the universe) — after §3 stage 2,
  the same universe with feature columns, as parquet.
- `y = 1` iff the edge ∈ the label's `selected_edges`; rows from unanimous-NONE
  groups (§2.4) are all `y = 0`.
- Sample weight = labeler trust × vote margin: human 1.0; panel-unanimous 1.0
  scaled by `mean_confidence`; majority-consensus soft rows weighted by vote
  share (e.g. 2/3 → 0.67) and only used in training, never eval. This is a
  design choice to validate by ablation, not an assumption: round 2 only tested
  the unanimous-soft variant.

### 2.3 Expected row counts

Round 2 measured 895 candidate-edge rows from 126 mapped groups (7.1 edges/group
with rejected edges included; 94 clean; 71 under-selection positives = 25% of
~284 rejected-edge rows). Extrapolating at the same mix:

| labeled groups (mapped) | edge rows | rejected-edge rows | under-selection positives |
|---:|---:|---:|---:|
| 126 (today, round 2) | 895 | ~284 | 71 |
| 150 | ~1,100 | ~340 | ~85 |
| 300 | ~2,100 | ~680 | ~170 |

Plus soft rows: every *voted* group (unanimous or not) contributes soft-labeled
edges — the #277-era Boston/Seattle waves plus the 35-group early-signal sweep
already cover ~200 voted groups, and M1's scheduled waves grow this faster than
curated labels. At 300 curated groups a ~26-feature model has ~80 rows/feature
on hard labels; a 100+-feature model (§4.2) leans on the soft rows and needs the
ablation to prove they carry the load. Honest uncertainty: 300 groups may still
be too few for the full feature set — the eval harness (§6) decides, not this
document.

### 2.4 Two label-side gaps this design requires closing

**(a) Empty-set labels don't exist.** The panel's unanimous-REJECT verdicts
(choice = NONE — 4 in the early-signal sweep alone, all cross-mode) route to
`human_review` and are never exported to `labels/stitching/`. So the training
table contains no group whose correct answer is "select nothing", which is
exactly the cross-mode defect's shape. Required change: `stitch_export` gains an
export path for unanimous-NONE verdicts as pair-semantics labels with
`selected_edges = []` (after human confirmation, or auto with an audit sample
once the pattern is validated). The eF1 decoder already handles k = 0
(`round2.select_expected_f1` scores the empty set as ∏(1−pᵢ)); the *data* is
what is missing.

**(b) Anchoring bias (round 2's confounder).** Every curated label was elicited
from a reviewer looking at the optimizer's proposal, so production is graded on
answers it helped write; the 218 keep=0 rejected-edge rows are systematically
easy for production. Mitigation, per round 2's flip condition #1: a de-anchored
review mode — sessions that present the full candidate set with no optimizer
pre-selection — for a **30–50-group unbiased eval slice**, stratified across
datasets. This slice is for *evaluation only*; training can keep anchored labels.
(`scripts/render_review_diffs.py` already flags cross-product artifacts in
de-anchored labels; reuse its plumbing.)

### 2.5 Leakage risks

- **Boston-heavy base** (113/140 mapped): a pooled CV number is mostly a Boston
  number. Required protocol: GroupKFold on `group_id` for the headline (both
  rounds' protocol), **plus leave-one-dataset-out** (train Boston → eval
  Seattle and vice versa) reported separately. Per-dataset allowlisting (§5)
  means the deployment question is per-dataset anyway.
- **Confidence-distribution shift**: calibrated confidences differ by dataset
  type (Seattle's prune floor is 0.90 vs Boston's 0.96 for this reason).
  Group-relative confidence features (`conf_rel_max`, `conf_rank_frac`) are
  shift-robust; absolute `confidence` is the leaky one. Keep both, check LODO.
- **Vote/label circularity**: panel-unanimous labels and vote-derived soft rows
  come from the same ballots. Never let a group appear in eval when any of its
  ballots contributed to training (group-level splits already guarantee this).
- **Dead feature families** (early-signal issue 3): Bogotá has empty names and
  constant class; Tunis constant `unknown` class. Per-dataset constant columns
  must be tolerated by the model (XGBoost handles them) but they make
  *cross-dataset* name/class importances misleading — another reason LODO is
  mandatory.

---

## 3. Candidate-graph persistence prerequisite (schema contract)

A separate implementation task is in flight for this; the schema below is the
contract this design needs it to satisfy. The guiding split: **groups.json stays
the human/UI artifact (capped, readable, back-compat); a parquet sidecar becomes
the resolver's canonical substrate (uncapped, feature-complete, joinable).**

### 3.1 What exists (post-#282/#284/#344) and what is missing

The sidecar already persists, per group: selected `edges` and `rejected_edges`
(sibling key, byte-invariant for `edges` consumers), each with `confidence`,
alignment fracs, the structural layer from `compute_group_structure`
(`degree_ref/tgt`, `is_bridge`, `biconnected_block`, `corridor_ref/tgt`,
`is_sliver`), `selected` and `pruned` flags; per group `n_edges`, `n_corridors`,
`n_assignment_components`, `largest_biconnected_block`, `oversized_group`,
`n_pruned`, `n_rejected_total`, `rejected_truncated`
(`config.py::stitch_persist_rejected_edges`, cap 64/group). Since #344 (stage 1
of this prerequisite) each group also carries **`candidate_edges`**: every
floor-passing candidate pair in the group's component, **uncapped**, attributed
to exactly one group, with `confidence`, `selected`, and `selected_elsewhere`
(`pipeline/runner.py::_compute_candidate_graph_by_group`) — a deliberately
minimal uniform schema that closes the "under-selection unlearnable" gap at the
pair level.

Still missing, and blocking round 3 (flip condition #3 in round 2):

1. **The 83 pair features for candidate edges.** They exist for only ~5% of
   group edges via `labels/features/` (pair-labeled pairs only), and #344's
   `candidate_edges` carries no feature columns. Yet the full vectors are
   computed for *every scored candidate* at match time — the factory
   `scored_cache.py` already serializes `MatchResult.features` per candidate
   (`features_json`). The data exists in memory in `optimize_and_export`; it is
   simply not persisted keyed to groups outside the factory path.
2. **The structural layer on candidate edges** (degree, bridge, corridor ids) —
   present on `edges`/`rejected_edges`, absent from the minimal
   `candidate_edges` schema.
3. **Rank/margin context** at selection time (cheap to derive, but persisting it
   pins the exact decode inputs).

### 3.2 Proposed schema: `<dataset>_candidates.parquet`

Written by `_export_groups_sidecar`'s caller alongside `<dataset>_groups.json`
(factory: `dataset=<name>/candidates.parquet`), one row per candidate edge of
every persisted group (selected + rejected + pruned, **no cap**):

| column group | columns |
|---|---|
| keys | `dataset_id`, `group_id`, `ref_id`, `target_id`, `ref_idx`, `target_idx` |
| status | `selected` (bool), `pruned` (bool), `is_sliver` (bool), `decision` |
| score | `confidence` (calibrated), alignment fracs ×4 |
| structural layer | `degree_ref`, `degree_tgt`, `is_bridge`, `biconnected_block`, `corridor_ref`, `corridor_tgt` |
| group context (denormalized) | `match_type`, `n_edges`, `n_corridors`, `n_assignment_components`, `largest_biconnected_block`, `oversized_group` |
| pair features | the 83 current `FEATURE_COLUMNS` as native float64 columns (NaN-preserving, not JSON — columnar reads matter for training) |
| UDF-derivability extras (§5.5) | `ref_class`, `target_class`, `ref_length_m`, `target_length_m`, `lateral_offset_signed_m` |
| provenance | `feature_version`, `model_hash`, `schema_version` (file-level metadata is fine) |

Geometries are **not** duplicated here — groups.json already carries WGS84
geometries per group and the parquet joins to it on `group_id`. The five
"extras" columns exist so that every group-level aggregate in §4.2 tier 3 and
every decode constraint in §4.1 is computable from parquet columns alone,
without repo-side geometry — the Spark-portability constraint (§5.5) demands
it, and it is what makes this parquet the training *and* the Spark scoring
substrate rather than a training-only export.

**Size estimate.** ~90 float columns ≈ 720 B/row raw. Seattle (21.7k selected +
25.6k rejected pre-cap ≈ 50k rows) ≈ 36 MB raw → ~10–15 MB snappy parquet, next
to its 65 MB groups.json. Tunis-scale (145 MB JSON sidecar) extrapolates to
roughly 40–60 MB parquet. Acceptable; no cap needed. If it ever isn't, drop
rejected rows below the candidate floor for non-group 1:1 components — group
rows are the only ones the resolver reads.

**Backward compat.** Purely additive sibling file; absence is tolerated
everywhere (resolver falls back per §5.4; review UI keeps reading JSON). The
JSON `rejected_edges` list and its cap stay as-is for the UI. Flag:
`stitch_persist_candidates` default True, mirroring
`stitch_persist_rejected_edges`.

**Convergence note (stage 2, following #344's stage 1):** the three
non-negotiables from this design's perspective are (a) *uncapped* group
candidate rows (already true of #344's `candidate_edges`), (b) the 83 features
as *typed columns keyed by (group_id, ref_id, target_id)*, (c) the §5.5
UDF-derivability rule — no resolver input may require repo-side state.
Everything else (exact column naming, whether group context is denormalized) is
negotiable.

---

## 4. Model family & features

### 4.1 Per-edge scorer + constrained set decode (the validated shape)

Round 2 settled the architecture question at this data scale: a small XGBoost
per-edge keep-probability model plus a **group-level expected-F1 subset decode**
(`resolver/round2.py::select_expected_f1`) was the best learned config in every
slice, and the set-level objective is the only thing that ever moved group
exact-match (+3.2 pts clean with soft labels) — the metric round 1's independent
per-edge thresholding could not budge. A GNN/CRF remains unjustified below
several hundred groups (round 1 §5 phase 3; unchanged). So:

- **Stage 1 — per-edge P(keep):** XGBoost, shallow (round-1 hyperparameters as
  the starting point), monotone constraint on `confidence` (the model must never
  *prefer* a lower-confidence edge ceteris paribus — keeps it an auditable
  superset of the prune).
- **Stage 2 — constrained decode per group:** expected-F1 subset selection over
  the group's candidate edges, extended with structural constraints the plain
  eF1 selector lacks:
  - never emit a selection violating the coverage-overlap rule
    (`MAX_ALIGNMENT_OVERLAP_M`, same as `_validate_assignment_coverage`);
  - within a corridor-pair, prefer contiguous alignment spans (penalize
    selections that leave a mid-corridor gap while keeping both flanks);
  - empty set allowed (cross-mode); slivers never selected alone.

  Groups are ≤ ~64 edges in the non-monster regime, so exact/beam decode per
  group is trivial computationally.

Why not end-to-end structured prediction: (i) label scale (§2.3); (ii) the
per-edge + decode factorization lets pair-level signal (5.4k pair labels) and
group-level signal (hundreds) contribute at their natural granularity; (iii)
determinism and auditability — a per-edge score column in the sidecar is
reviewable in the existing UI, a structured model's joint decision is not.

### 4.2 Features

Three tiers, all available per-row from the §3 parquet:

1. **The 26 round-1/round-2 sidecar features** (`resolver/features.py::
   FEATURE_COLUMNS` + `round2.EXTENDED_FEATURE_COLUMNS`): confidence +
   group-relative confidence (`conf_rel_max` dominated round-1 importances),
   spans, degree/bridge/sliver, competition (`n_share_*`, `conf_margin_*`,
   `is_best_for_*`, span-overlap-with-higher), group counts.
2. **The 83 pair features**, newly available for every candidate edge. The ones
   with a designed role:
   - *Cross-mode geometric* (§4.3): the Parallel Sibling family
     (`has_parallel_sibling_ref`, `parallel_fraction_ref`,
     `offset_vs_half_corridor_ratio`, `offset_over_expected_halfwidth`,
     `likely_representation_mismatch`), `lateral_offset_m` / `_iqr_m` / `_p95_m`,
     `class_similarity` (weak alone — see below), crossing-angle features.
   - *Coverage asymmetry*: `ref_coverage` / `target_coverage` /
     `max_coverage` — per CLAUDE.md, `max(ref, target)` is the fair filter under
     asymmetric segmentation; the model gets both sides and learns the
     asymmetry the prune cannot express.
   - *Name/class*: full similarity families, dead on some datasets (§2.5) but
     decisive on others.
3. **New group-level aggregates** (computed at decode time from persisted
   per-edge columns only — the §5.5 derivability rule):
   - corridor-geometry: per-corridor-pair aggregate lateral offset mean/IQR and
     *offset sign consistency* along the corridor (a separated cycleway sits
     consistently ~2–8 m on one side; a lane-on-road sits near 0 with mixed
     sign). Flag: the shipped `lateral_offset_m` is an unsigned magnitude
     (`compute_perpendicular_offset` uses `shapely.distance`), so sign
     consistency needs a new per-edge `lateral_offset_signed_m` column computed
     at match time and persisted in the §3.2 parquet — it cannot be derived
     downstream;
   - *exclusive-lane evidence*: does the ref segment have a competing same-class
     candidate elsewhere in the group with better geometry (if the road
     centerline already has a road-class target, the cycleway edge is
     redundant); share-of-ref/target span already claimed by higher-scored
     edges;
   - contiguity-gap features of the tentative selection (feeds the decode
     penalty, also exposed to stage 1 as "would keeping this edge bridge a
     gap").

### 4.3 The cross-mode defect is a named design target

The known failure (early-signal issue 1, BLOCKER for `co_bogota_bike_network`
and `sg_singapore_footpaths`): dedicated cycleways/footpaths matched to the
parallel road centerline of a different class at confidence 0.82–0.95, panel
rejecting all four (three unanimous, one majority). A hard class gate is the wrong fix — cycle *lanes*
legitimately share road geometry, so `class_similarity` is a weak cross-mode
signal on its own (per the enriched-A/B policy-gap analysis and the memory note
on cycleway class). The learned resolver is expected to fix it **geometrically**:
cross-class pairing × consistent non-zero lateral offset × exclusive-lane
evidence is separable where class alone is not. This requires (a) empty-set
labels (§2.4a), (b) the pair-feature persistence (§3), and (c) the acceptance
test in §6.3. Uncertainty stated plainly: today there are exactly 4 labeled
cross-mode reject groups; the design bets that targeted panel waves on Bogotá /
Singapore footpaths can grow this to ~20–30 cheaply because the panel detects
them reliably (4/4 in the sweep).

---

## 5. Inference integration

### 5.1 Where it slots in

Exactly the prune's seam — `pipeline/runner.py::optimize_and_export`, step 4.5
(runner.py:1109–1133), **replacing** `apply_confidence_drop_prune` for
allowlisted datasets:

```
optimized = optimize_matches_with_grouping(...)          # unchanged
per-group candidate graphs = selected ∪ rejected (in memory, pre-sidecar)
if dataset in learned_resolver_overrides:                # new
    optimized, resolver_scores = apply_learned_resolver(...)
elif dataset in resolver_prune_overrides:                # unchanged
    optimized, pruned = apply_confidence_drop_prune(...)
_export_groups_sidecar(...)  # records resolver_score + resolver_dropped per edge
```

Rationale for post-optimizer rather than inside
`_classify_and_resolve_component`: the optimizer remains the candidate/group
generator (its recall is the thing round 2 proved hard to beat); the resolver is
a pure subset function per group, testable in isolation against sidecar fixtures;
and the sidecar can record both the optimizer's and the resolver's opinion per
edge (`selected`, `resolver_score`, `resolver_dropped`), which keeps the gate,
the review UI, and shadow mode (§7) all working from one artifact. Under-selection
repair (promoting a rejected edge) is in scope for the resolver at this seam —
the rejected candidates are in memory here; the guarantee "never emit an edge the
optimizer never generated as a candidate" is preserved. A further argument for
this seam: score-then-decode over persisted columns is exactly the shape that
ports to Spark unchanged (§5.5).

### 5.2 Determinism

Same discipline as the optimizer's canonical-ordering fix
(optimizer.py:794–797): candidate edges sorted by `(str(ref_id),
str(target_id))` before scoring/decode; XGBoost inference is deterministic for a
fixed model artifact; the model hash goes into the factory `score_key`-adjacent
manifest metadata (a resolver change must invalidate `optimize_key`, *not*
`score_key` — re-decode is the 2-s `factory reoptimize` path, which becomes the
resolver-iteration loop). `PYTHONHASHSEED` must not matter; add a determinism
test mirroring the existing stitch tests.

### 5.3 Latency budget

Optimizer ≈ 2 s today. Resolver cost = one XGBoost `predict` over all group
candidate edges (Seattle ≈ 50k rows → well under 1 s single-threaded) + per-group
decode (≤ 64 edges/group, trivial). Budget: **≤ 1 s added, i.e. ≤ 50% of
optimizer time, ≈ 0.2% of pipeline wall time** (scoring is 84%, #274). Feature
vectors are already computed — no geometry recompute at this seam.

### 5.4 Fallback & flags

Mirrors the prune's machinery one-for-one: `learned_resolver_enabled` master
switch (default True), `learned_resolver_overrides` per-dataset allowlist
(dataset → model operating point / decode config), datasets absent → prune path
→ keep-all path. Model artifact missing/unloadable, calibration inactive
(resolver features were trained on calibrated confidence — same guard as
`_effective_prune_threshold`), or feature-version mismatch → log one line, fall
back to the prune path. No global default, ever — the #284 lesson (a
Boston-tuned setting silently over-pruned never-tuned datasets) is baked into
the rollout design.

### 5.5 Spark portability (a first-class design constraint)

Spark is Overture's big-data framework of choice, and the repo already ships a
Spark-portable *pair* model consumed by the tf-data-platform sister project:
XGBoost-native JSON booster + feature manifest bundled in the wheel under
`crosswalk/_model/`, exposed via the import-light accessors in
`src/crosswalk/spark.py` (stdlib-only at module import; numpy lazy inside
`apply_calibration`), kept in `FEATURE_VERSION` lockstep by
`tests/unit/test_shipped_spark_model.py`. The resolver must be integratable the
same way — this is a **design constraint, not a port to be attempted later**,
and it shapes three decisions already made above.

**(a) Scoring artifact contract.** The per-edge resolver model ships exactly
like the pair model: an XGBoost-native JSON booster at
`crosswalk/_model/spark_resolver_model.json` plus
`spark_resolver_manifest.json` carrying the ordered resolver feature list
(column order is the contract, as with the pair manifest's `features`), the
`feature_version` it was trained against, a note that inputs are already
**calibrated** confidences (the resolver consumes the pair model's calibrated
`P(match)`; a Spark job chains `apply_calibration` from the existing manifest
before resolver scoring), and the **decode hyperparameters** (eF1 variant,
overlap threshold, corridor-gap penalty weights) so decode behavior is pinned
by the artifact, not by repo code drift. Accessors `spark_resolver_model_json()`
/ `spark_resolver_manifest()` join `crosswalk.spark`, preserving its
stdlib-only-at-import guarantee; a `tests/unit/test_shipped_spark_resolver_model.py`
mirrors the existing lockstep test pattern (feature-version equality, feature
list vs config, decode params present).

**(b) The decode must be Spark-mappable.** The constrained expected-F1 decode
is implemented as a **pure, deterministic function over numpy arrays** — edge
probabilities plus the group's structure columns (alignment fracs, corridor
ids, sliver flags, segment lengths) in, a boolean keep-mask out. No shapely, no
pandas, no repo state, no I/O. That makes it runnable inside
`groupBy("group_id").applyInPandas(...)`: groups are small (p99 ≤ 10 edges;
monsters route to humans and never reach the decode), so per-group decode is
embarrassingly parallel with **no cross-group state** — the same property that
makes it unit-testable against sidecar fixtures locally. This constraint is an
additional, independent reason the per-edge XGBoost + local decode architecture
beats the alternatives from §4.1: a GNN would put torch on every executor for
sub-second-total work, and global/ILP-style structured prediction would drag in
solver dependencies and cross-partition coordination that `applyInPandas`
cannot express. The §4.1 constraints are therefore specified in *column* terms
— span overlap from the persisted alignment fracs × `ref_length_m` /
`target_length_m` (the frac-space equivalent of `MAX_ALIGNMENT_OVERLAP_M`),
corridor gaps from fracs grouped by `corridor_ref`/`corridor_tgt` — never by
re-measuring geometry.

**(c) Feature computability rule.** Every input the resolver consumes must be
either (i) a persisted candidate-graph column (#344's `candidate_edges` for
ids/confidence/selected; the stage-2 `candidates.parquet` of §3.2 for features
and structure) or (ii) derivable *inside the UDF* from those columns with
group-local numpy — never from repo-side state (GeoDataFrames, spatial
indexes, settings). Audit of §4.2 against this rule:

| feature tier | verdict |
|---|---|
| 26 sidecar features (tier 1) | Derivable in-UDF from `confidence` + structure + span columns (group-relative aggregates are group-local by construction) |
| 83 pair features (tier 2) | Persisted columns by design. The context-heavy ones (Parallel Sibling, crossing-angle, graphlet, topology) need full-dataset spatial context to *compute* — so they are computed once at match time repo-side and **persisted**; the UDF never recomputes them. This is the load-bearing reason tier 2 must live in the parquet at all |
| group aggregates (tier 3) | Derivable in-UDF **except** three inputs that must be added as persisted columns (now in §3.2): `lateral_offset_signed_m` (shipped offset is unsigned; sign consistency is not derivable downstream), `ref_class`/`target_class` (exclusive-lane evidence needs per-edge classes; groups.json holds them group-level only), `ref_length_m`/`target_length_m` (frac→meter conversion for overlap/gap constraints) |

No proposed feature had to be dropped; three required persistence instead of
derivation, and §3.2 was amended accordingly.

**Typical Spark use** (mirroring the `crosswalk/spark.py` docstring style):

```python
from crosswalk.spark import (
    spark_manifest, spark_model_json, apply_calibration,
    spark_resolver_manifest, spark_resolver_model_json, decode_group,
)

r_manifest = spark_resolver_manifest()
r_features = r_manifest["features"]        # broadcast; column order matters
booster.load_model(bytearray(spark_resolver_model_json().encode()))
# 1. pandas_udf: score candidate edges -> P(keep) per row
# 2. per-group decode, no cross-group state:
df.groupBy("group_id").applyInPandas(
    lambda pdf: pdf.assign(
        keep=decode_group(
            pdf["p_keep"].to_numpy(),
            pdf[STRUCTURE_COLS].to_numpy(),
            r_manifest["decode"],
        )
    ),
    schema=out_schema,
)
```

`decode_group` ships in `crosswalk.spark` under the same lazy-numpy discipline
as `apply_calibration`, and a parity test asserts it is bit-identical to the
decode the local pipeline runs — the train/serve-skew defense of §7, extended
to Spark/local skew.

---

## 6. Eval plan

### 6.1 Metrics & harness

The round-2 harness (`resolver/round2.py`, `scripts/run_stitcher_round2.py`) is
the eval instrument, extended with LODO splits (§2.5) and the de-anchored slice
(§2.4b). Headline metrics, unchanged from both rounds and from the mbench gate:
**sliver-filtered edge-F1** and **group exact-match**, clean slice, pooled OOF,
group bootstrap (2000 resamples) for the production comparison. The external
check is `mbench run crosswalk <ds> --gate` against `mbench/datasets.toml`
floors (Boston armed: filtered F1 ≥ 0.83, exact ≥ 0.50; baseline 0.8858 /
0.5946 at 111 mapped pair groups post-#295/#298; Seattle arms at 30 mapped).

### 6.2 The bar (promotion criterion)

Per dataset, the resolver ships to the allowlist only if, on that dataset's
clean slice under the #271 gate protocol:

1. **Beats the tuned confidence prune** — not keep-all. The #272 result is the
   floor of the argument (the one-parameter threshold beat the round-1 learned
   model 0.872 vs 0.852 clean F1); the shipped bar is higher: Boston
   optimizer+prune 0.8790 filtered F1 / 0.5833 exact, Seattle 0.8913 / 0.50.
   Round 2's bootstrap must invert: **P(learned ≥ production) ≥ 0.9** on ΔF1,
   not the current 0.028.
2. **Moves group exact-match** — the round-1 lesson that edge-F1 gains without
   exact-match gains don't justify replacing a group decision.
3. **Gate PASS** on armed datasets, trivially.
4. **No regression on the de-anchored slice** — the only slice where production
   isn't structurally favored; a learned win that exists *only* on anchored
   labels is anchoring artifact, not signal.

### 6.3 Cross-mode acceptance criterion (named)

A held-out **cross-mode testset**: all panel-REJECT groups from
`co_bogota_bike_network` and `sg_singapore_footpaths` (4 today, grown to ≥ 20
via targeted waves before eval; kept out of training). Acceptance: the resolver
empties (or reduces to genuinely-shared-geometry edges, human-adjudicated)
**≥ 80% of cross-mode reject groups**, while same-mode recall on
Boston/Seattle clean slices does not regress (edge recall drop ≤ 0.01). This is
the publish-unblocker for both datasets and is reported as its own table, not
folded into pooled F1 (where 20 groups would vanish).

### 6.4 Flip mechanics

Per-dataset allowlist cutover, one dataset per PR, in this order: Boston
(largest labels, armed gate), Seattle (arms its gate at 30), then Bogotá
bike + Singapore footpaths (cross-mode, currently publish-blocked, highest
marginal value), then early-signal over-selectors (Tunis, Missoula) as their
label bases reach ~20–30 mapped groups. Each flip PR carries the before/after
comparison table (CLAUDE.md workflow) + gate output + the §6.2 bootstrap.
Datasets never flipped keep today's behavior indefinitely — that is a feature
of the design, not a compromise.

---

## 7. Risks & staging

| Risk | Read | Mitigation |
|---|---|---|
| Label scarcity persists (300 groups still too few) | Real; round 2 lost with 126. The soft-label lever (+0.05 F1) and full-candidate features are the two untested multipliers | Go/no-go stays evidence-gated (§6.2); the prune remains shipped and untouched; worst case round 3 produces another calibrated NO-GO doc and the label factory keeps running |
| Boston skew → model that only works on Boston | Likely for v1 | LODO reporting; per-dataset allowlist means partial success ships partially |
| Anchoring bias inflates *either* side | Cuts both ways — production graded on its own answers; the model trained on them | De-anchored eval slice is a hard gate (§6.2.4) |
| Cross-mode testset too small (n=4 today) | Real | Grow via targeted panel waves first (the panel caught 4/4 in the sweep); §6.3 requires ≥ 20 before eval |
| Monster groups (>20 edges, ~0.1–0.5%) | Resolver *scores* help triage but auto-resolution stays out of scope: still routed to humans via `group_is_structurally_simple` (unchanged). The resolver ships edge rankings into the review pack (panel + human see P(keep) ordering) — a UX win, not a behavior change. Full monster resolution is the round-1 Phase-3 structured model, gated on several hundred labels | Explicit non-goal for this design |
| Sidecar/parquet size on Tunis-scale datasets | ~40–60 MB parquet estimated | Acceptable; escape hatch in §3.2 |
| Train/serve skew between table builder and runtime featurizer | Classic failure mode | One featurizer module consumed by both (the backfill-parity pattern, `tests/unit/test_backfill_parity.py` precedent); parity test on a committed fixture |
| Determinism regressions | group membership already hash-sensitive historically | §5.2 ordering + determinism test |

Staging order is deliberately conservative: persistence → labels → offline round
3 → shadow → per-dataset flips. Every stage before "flip" is production-inert.

---

## 8. Milestone breakdown (PR-sized)

Dependencies flow downward; R1–R3 can interleave with L1–L2.

| # | PR | Depends on | Acceptance criteria |
|---|---|---|---|
| **P1** | Candidate parquet sidecar (`<dataset>_candidates.parquet`, §3.2) — stage 2 following #344's `candidate_edges` | — | Schema per §3.2 (uncapped group rows, 83 typed feature cols + §5.5 derivability extras, keys); groups.json byte-identical; stitch gate PASS unchanged; size within estimate on Seattle; flag default-on. **Implemented and Seattle-validated 2026-07-11: 33,246 rows / 18,416 groups / 13.67 MiB.** |
| **L1** | Empty-set label export (unanimous-NONE → `selected_edges=[]`, §2.4a) + eval support for empty truth sets | — | Round-trips through `labels/stitching/` + `stitch_eval` mapping; gate metrics well-defined for empty labels; the 4 known cross-mode groups exported after human confirm |
| **L2** | De-anchored review mode + 30–50-group unbiased slice (§2.4b) | P1 (UI reads full candidate set) | Slice committed with `deanchored` provenance; render_review_diffs cross-product check clean |
| **L3** | Targeted cross-mode panel waves (Bogotá bike, SG footpaths) to ≥ 20 reject groups | L1 | ≥ 20 labeled cross-mode groups, held out of training by construction |
| **R1** | Training-table builder v3: join parquet features, soft-vote rows from `labels/votes/`, sample weights (§2.2) | P1 | Row counts match §2.3 within noise; parity test vs runtime featurizer; deterministic under PYTHONHASHSEED |
| **R2** | Round-3 offline eval at 150+ groups: extended features, eF1 + constrained decode, LODO + bootstrap + de-anchored slice | R1, L2; label base ≥ 150 mapped | A `research/learned_stitcher_round3.md` with §6.2 tables; explicit GO/NO-GO per dataset. **This is the go/no-go gate for everything below** |
| **R3** | Cross-mode acceptance run (§6.3) | R2, L3 | ≥ 80% cross-mode groups emptied, same-mode recall regression ≤ 0.01 |
| **I1** | Runtime integration behind `learned_resolver_overrides` (empty allowlist) + shadow mode: `resolver_score`/`resolver_dropped` annotated per edge, zero behavior change | P1, R2 GO | Empty-allowlist run byte-identical; determinism test; latency ≤ 1 s added on Seattle; fallback paths unit-tested; decode implemented as the pure numpy function of §5.5(b) |
| **S1** | Spark-portable resolver export: `crosswalk/_model/spark_resolver_model.json` + manifest, `crosswalk.spark` accessors + `decode_group`, lockstep + local/Spark decode-parity tests (§5.5a) | I1 | `import crosswalk.spark` stays stdlib-only; lockstep test red on FEATURE_VERSION drift; decode parity bit-identical on the committed fixture |
| **I2..n** | Per-dataset allowlist flips, one PR each, Boston first (§6.4) | I1, R2/R3 GO for that dataset | §6.2 criteria met on that dataset; before/after table + gate output in PR; floors in `mbench/datasets.toml` re-baselined if raised |
| **D1** | Docs: ARCHITECTURE.md thresholds-table row, SCALING_ROADMAP M3 status, BENCHMARKING gate notes | I1 | Consistency with shipped flags |

Explicitly *not* in this plan: replacing decomposition/grouping, monster
auto-resolution, GNN/structured end-to-end models, and the Meili hybrid
candidate-generator experiment (roadmap M3's separate bullet — it feeds the same
resolver seam if it ever lands, which is a design compatibility argument for the
post-optimizer integration point, not a dependency).

---

## Appendix — file-level anchor index

- `src/crosswalk/matching/optimizer.py` — `_classify_and_resolve_component`
  :766 (Case 4 :888), `_create_group_results` :487, `apply_confidence_drop_prune`
  :1251, `group_is_structurally_simple` :1302, `compute_group_structure` :1337
- `src/crosswalk/pipeline/runner.py` — `_effective_prune_threshold` :128, prune
  seam :1109–1133, sidecar build (rejected/pruned recording) :700–900,
  `optimize_and_export` :997, `_compute_candidate_graph_by_group` (#344
  `candidate_edges`)
- `src/crosswalk/spark.py` — import-light Spark accessors (pair model precedent
  for §5.5); `tests/unit/test_shipped_spark_model.py` (lockstep test pattern)
- `src/crosswalk/config.py` — `stitch_persist_rejected_edges` :784 (cap 64),
  prune allowlist commentary :795–834
- `src/crosswalk/resolver/` — `features.py` (26 sidecar features), `round2.py`
  (`EXTENDED_FEATURE_COLUMNS`, `select_expected_f1` :113), `votes.py`
  (soft labels), `extract.py` (label→group mapping), `evaluate.py` (CV harness)
- `src/crosswalk/factory/scored_cache.py` — per-candidate `features_json`
  precedent for feature persistence
- `src/crosswalk/agent_labeling/` — `stitch_runner.py` `VOTES_COLUMNS` :988 /
  `CONSENSUS_COLUMNS` :1001, `stitch_export.py` `write_vote_provenance` :457
- `labels/stitching/dataset=*/data.csv`, `labels/votes/dataset=*/{votes,consensus}.csv`
- `mbench/datasets.toml` — `[gate.us_boston_streets]` floors; `docs/BENCHMARKING.md`
  "Stitch-level quality gate"
