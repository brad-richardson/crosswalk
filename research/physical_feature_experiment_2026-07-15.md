# Physical road evidence: coverage audit and pairwise ablation

Date: 2026-07-15

## Outcome

Keep bridge/tunnel/layer evidence in the ingestion, sidecar, prompt, and review
UI, but do **not** promote the experimental pairwise features into the production
matcher yet.

The experiment found a small apparent global gain for the full feature bundle,
but most of it comes from `physical_comparable_count`: a proxy for which target
datasets expose physical metadata. The actual bridge/tunnel/vertical values are
too sparse and provider-dependent. On the 46 reviewed pairs where a physical
state is positive or conflicting, the full bundle reduced F1 by 0.0253.

This work is preserved rather than removed:

- The Overture top-level `road_flags` ingestion bug is repaired and existing
  snapshots are backfilled at load time.
- Alignment-aware experimental feature code and unit tests remain available.
- The complete 3-seed/5-fold output is in
  `research/results/physical_feature_ablation_2026-07-15.json`.
- Six exact negative pairs, one diagnostic pair, and the Geneva group case are
  pinned in `tests/fixtures/physical_match_regressions.json`.
- A geometry-derived same-side coincidence feature is implemented separately
  from physical layer truth.

## Ingestion finding

Current Overture segment Parquets expose `road_flags`, `level_rules`, and
`road_surface` as top-level columns. The fetch parser still looked only inside a
legacy nested `road` struct. Consequently, retained raw flags were populated but
every derived `road_flags_lr` value was empty. This masked exactly the bridge and
tunnel evidence under investigation.

The repaired path now:

1. prefers top-level `road_flags` while retaining the legacy fallback;
2. rebuilds only `road_flags_lr` and `level_lr` when loading an existing
   Overture snapshot, preserving row order and cached positional indices;
3. keeps each flag's `between` range for edge-level clipping; and
4. renders `is_covered` and `is_indoor` in evidence, while keeping absent target
   domains unknown.

## Target-side coverage gate

Raw target analogs remain the limiting factor. The seven configured datasets
have useful retained source fields, but the reviewed training corpus is much
sparser than the raw networks.

| Target dataset | Comparable target domains | Reviewed pairs | Comparable | Positive/conflicting |
| --- | --- | ---: | ---: | ---: |
| Sydney roads | bridge, tunnel | 226 | 224 | 6 |
| Helsinki/Digiroad | bridge, tunnel, signed level | 84 | 78 | 3 |
| London/OS Open Roads | tunnel | 201 | 189 | 2 |
| Hong Kong roads | level | 208 | 189 | 20 |
| Berlin roads | level | 121 | 120 | 11 |
| Amsterdam/NWB | relative height | 199 | 8 | 4 |
| Utah/Salt Lake roads | vertical level | 200 | 1 | 0 |

The raw target snapshots contain substantially more positive rows:

| Dataset | Locally observed target rows |
| --- | ---: |
| Sydney | 2,356 bridge; 348 tunnel |
| Helsinki | 4,359 bridge; 85 tunnel; 7,774 nonzero signed levels |
| London | 119 tunnel |
| Hong Kong | 2,236 nonzero levels |
| Berlin | 355 nonzero levels |
| Amsterdam | 1,130 nonzero relative-height rows |
| Utah | 508 nonzero levels |

The feature go/no-go rules are therefore:

- **Bridge/tunnel:** keep ingesting and displaying. Do not ship a generic
  pairwise feature until each flag has enough active reviewed examples across at
  least three target providers. Bridge currently has only Sydney and Helsinki;
  tunnel has Sydney, Helsinki, and London, but only three reviewed conflicts.
- **Exact or neighbor-relative numeric level:** deferred. Provider scales are
  not demonstrably interchangeable, and a same-side geometric alternative is
  not evidence that one segment is physically above the other.
- **Vertical sign/non-ground state:** valid as an experiment, but not promoted.
  It behaves differently by provider and needs targeted review first.
- **Covered/indoor:** ingest and display from Overture, but no pairwise feature;
  none of the audited targets has a safe analog.
- **Surface/material:** deferred; only Sydney exposes an obvious target analog
  in the audited set, so a future global ablation would mostly measure missingness.
- **Same-side coincidence:** does not require a target metadata analog because it
  is geometry-derived on either provider side. Evaluate it as group context, not
  as a synthetic layer.

## Experimental features

All features are clipped to the reviewed pair's reference and target alignment
fractions. Missing target domains are `NaN`, not false.

- `bridge_fraction_delta` and `tunnel_fraction_delta`: absolute difference in
  the fraction of the aligned span carrying each flag.
- `physical_flag_positive_match`: common positive bridge/tunnel fraction.
- `vertical_nonzero_fraction_delta`: difference in the aligned non-ground
  fraction.
- `vertical_sign_delta`: sign difference on a 0–1 scale; it never compares exact
  layer integers.
- `vertical_positive_match`: common non-ground fraction when signs agree.
- `physical_structure_conflict`, `physical_positive_match`: compact composites.
- `physical_comparable_count`: number of mutually observed domains. This is kept
  for diagnosis but should be treated as a dataset-availability proxy.

## Ablation protocol and results

- 5,428 valid human-labeled pairs after the existing plausibility filter.
- 4,934 segment-connected components.
- Leakage-resistant 5-fold `GroupKFold` over those components.
- XGBoost seeds 42, 73, and 999; existing tuned parameters; threshold 0.5.
- Metrics are means over the three complete out-of-fold runs.
- `Comparable` contains 809 rows with at least one mutually observed domain.
- `Informative` contains 46 rows with a nonzero conflict or positive match.

| Additive variant | Global F1 | Δ | Comparable F1 | Δ | Informative F1 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.9280 | — | 0.8757 | — | 0.9102 | — |
| Flag-positive only | 0.9279 | -0.0001 | 0.8737 | -0.0020 | 0.9112 | +0.0010 |
| Vertical-positive only | 0.9263 | -0.0017 | 0.8689 | -0.0068 | 0.8969 | -0.0133 |
| Any positive only | 0.9283 | +0.0003 | 0.8748 | -0.0009 | 0.8900 | -0.0202 |
| Conflict only | 0.9281 | +0.0001 | 0.8761 | +0.0003 | 0.9070 | -0.0031 |
| Availability count only | 0.9303 | +0.0023 | 0.8847 | +0.0090 | 0.8956 | -0.0146 |
| Conflict + positive, no count | 0.9282 | +0.0002 | 0.8750 | -0.0007 | 0.9038 | -0.0064 |
| Flag primitives | 0.9292 | +0.0012 | 0.8759 | +0.0002 | 0.9034 | -0.0067 |
| Vertical primitives | 0.9272 | -0.0009 | 0.8735 | -0.0022 | 0.9405 | +0.0303 |
| All primitives | 0.9293 | +0.0013 | 0.8772 | +0.0014 | 0.8954 | -0.0148 |
| All values, no count | 0.9302 | +0.0021 | 0.8796 | +0.0039 | 0.9099 | -0.0003 |
| Full bundle | 0.9308 | +0.0028 | 0.8880 | +0.0123 | 0.8848 | -0.0253 |

The vertical-primitives result on the informative slice is worth revisiting
after more labels, but it is not sufficient to ship: global and comparable F1
decline, only 46 informative rows exist, and provider-level behavior conflicts.

## What the existing labels teach

The active evidence is sparse and not monotonic:

| Signal | Active rows | Match rate | Important qualification |
| --- | ---: | ---: | --- |
| Bridge fraction differs | 5 | 100% | A difference is not currently a safe negative |
| Tunnel fraction differs | 3 | 67% | Two of three are matches |
| Positive flag agrees | 4 | 75% | Corroboration, not identity |
| Vertical state differs | 33 | 48% | Strongly provider-dependent |
| Positive vertical sign agrees | 6 | 100% | Promising but far too small |
| Any physical conflict | 40 | 55% | More matches than non-matches |
| Any positive physical agreement | 9 | 89% | One same-tunnel non-match proves it cannot gate |

Vertical conflict is most negative in Hong Kong (11 no-match, 4 match) and
mostly positive in Berlin (8 match, 3 no-match). Helsinki's three vertical
conflicts are all matches. A generic “properties aligned” score would erase
these semantic differences.

## Known-failure regression seeds

| Dataset/group | Exact edge(s) | Truth/use |
| --- | --- | --- |
| Sydney `66e22055` | `d910f21d…` → `au_sydney_831180…`, `au_sydney_831394…` | Both known negatives; the latter is a target tunnel versus a surface cycleway |
| Hong Kong `4eed5e80` | `635d37a9…` → targets `103785`, `103984`, `144878` | Three known negatives; aligned reference is tunnel/layer -1 and targets are elevation 0 |
| Philadelphia `b7f57035` | `0d653fb0…` → `…673046…` | Explicit note-level negative; `GRADE_SEPARATED` is not safe bridge polarity |
| Philadelphia `b7f57035` | `ce6d1dfe…` → `…673046…` | Bridge/layer-1 evidence probe only; set membership does not establish exact positive pair truth |
| Geneva `dd106a0f` | group context | Same-side coincident alternatives; not a pairwise layer label |

These fixtures intentionally distinguish pair truth, note-derived truth, and
diagnostic-only evidence so a future training importer cannot silently turn a
set-semantics observation into an exact positive edge.

## Geneva: geometry-derived ambiguity

In `dd106a0f`, the long reference segment `3c772d2e…` (“Tranchée couverte de
Vésenaz”) overlaps two human-retained “Route de Thonon” reference segments. At a
3 m tolerance and a 20 m absolute-overlap floor:

- two role-conflicting alternatives qualify;
- both shorter Route de Thonon segments are 100% within the trench corridor;
- their union covers 254.2 m, or 33.1%, of the long trench segment; and
- the trench is excluded from the human set while both Route de Thonon segments
  are retained.

The experimental `compute_coincident_alternatives` feature captures this as
same-side non-identifiability. It also requires at least 20 m of coincident
length, preventing a tiny endpoint stub from being mislabeled as a layered
alternative. The intended resolver interpretation is: reduce confidence in
pairwise geometry as a discriminator and allow role-compatible corridor
continuity to carry more weight. It does **not** assert which segment is above or
below.

## Curated manual-review queue

The full queue is embedded in the result JSON. These are the most diverse/high
leverage examples to inspect first; probabilities are mean out-of-fold baseline
→ full-bundle values, not production scores.

| Dataset | Pair | Human label | OOF change | Why review |
| --- | --- | --- | ---: | --- |
| London | `95cbd72f…` → `…1346FB30…` | match | 0.773 → 0.497 | Near-complete positive tunnel agreement causes overreach at an LR boundary |
| Berlin | `329121a0…` → `…34355…` | match | 0.627 → 0.380 | Tiny level-fraction delta is overvalued |
| Helsinki | `60ccd4fa…` → `…32ecbf16…` | match | 0.362 → 0.602 | Strong partial physical agreement may be a real rescue |
| Berlin | `10a76665…` → `…410…` | match | 0.665 → 0.492 | Large nominal level conflict on a human match |
| Hong Kong | `a1b3e53e…` → `…118642…` | match | 0.799 → 0.668 | Full vertical conflict; validate `ELEVATION` semantics |
| Sydney | `47581368…` → `…1049668…` | no-match | 0.492 → 0.602 | Full physical conflict moves the wrong way through interactions |
| Sydney | `b637f286…` → `…1074308…` | no-match | 0.635 → 0.725 | Both are tunnels, but shared structure is not road identity |
| Hong Kong | `9db4de83…` → `…2069…` | no-match | 0.360 → 0.463 | Flyover/ground conflict; useful hard negative |
| Hong Kong | `153d4950…` → `…9324…` | no-match | 0.518 → 0.499 | One case where physical conflict corrects a borderline false positive |
| Geneva | group `dd106a0f` | set | n/a | Canonical same-side coincidence/continuity case |

## Recommended next experiment

1. Regenerate the stitching evidence packs after the Overture backfill so the
   known v7 groups visibly carry correct flags.
2. Manually review the queue above plus additional raw-network positives until
   each candidate domain has at least 20 active examples across three providers.
3. Confirm the provider semantics of Berlin `verkehrsebene`, Hong Kong
   `ELEVATION`, Helsinki `silta_alik`, and Amsterdam `rel_hoogte` against their
   data dictionaries before comparing signs.
4. Re-run two constrained experiments:
   - positive physical agreement as weak corroboration only; and
   - same-side coincidence as a group-resolver interaction with continuity.
5. Do not add these columns to `FEATURE_COLUMNS`, bump `FEATURE_VERSION`, or
   reship the model until the informative slice improves without relying on the
   availability-count proxy.

## Operational handoff: targeted v7 stitching wave

**Status (2026-07-15):** packs complete and validated; voting intentionally
paused before the first group completed. Resume on or after 2026-07-16 when the
Claude allowance has more headroom. Claude was not confirmed exhausted; the
pause is precautionary because it is close to the daily limit.

The experiment-integrity changes went through adversarial review, fixes, and a
fully green CI run in PR #433 before being squash-merged. The packs were then
regenerated from clean `main` commit
`70957a232a61450f0dfff32a7acfd1524fde272e`. Every generated batch records zero
tracked changes and zero untracked files in its source provenance.

The production manifest is:

`data/agents/stitching/batches/physical_context_v7_20260715_manifest.json`

Manifest SHA-256:
`a6fd6372df1dea362eb45816ed6098ad03656cbfe904231c9ef4f79637d1ed93`.

It contains:

- 50 unique enriched groups from Sydney, Helsinki, London, Hong Kong, Berlin,
  Amsterdam, Geneva, and Philadelphia;
- five dataset-diverse groups repeated across the other three cells of a 2x2
  physical-evidence / same-side-coincidence design;
- 15 paired control packs, for 65 scheduled packs in 23 batch directories;
- forced known regressions whose exact pairs survive into the displayed option
  menus;
- a counterbalanced/interleaved run schedule; and
- the exact high-effort panel: Claude Opus 4.8, Codex GPT-5.6 Sol, and Muse
  Spark 1.1.

This is a v7-only wave. Do not substitute v6 ballots or packs. Earlier dry
builds under `/tmp` were generated from dirty provenance and are diagnostics,
not vote inputs.

### Why the first launch stopped

The first launch could not find `opencode` because the noninteractive shell did
not inherit `~/.opencode/bin`. The installed executable is
`/home/brad/.opencode/bin/opencode` (1.17.18).

After restarting with that directory on `PATH`, Muse returned `invalid_api_key`
because `META_API_KEY` was absent from the execution environment. During the
manual stop, Claude also returned one empty output that the runner classified
as likely quota-capped. Because the daily allowance was already close to its
limit, the entire panel was stopped rather than retrying. This is not evidence
that Claude's quota was fully exhausted.

No `votes*.csv` or `consensus*.csv` file was persisted. The next run therefore
starts cleanly at schedule row 1, Sydney group `66e22055`; there are no partial
ballots to reconcile.

### Resume checklist

1. Confirm `META_API_KEY` is present in the same noninteractive environment
   that will execute the runner. Do not copy a credential into this repository
   or this note.
2. Put the existing OpenCode binary on `PATH` and smoke-test Muse authentication
   before starting the three-seat panel. Avoid spending Claude quota while a
   different seat is known-broken.
3. Revalidate the immutable manifest:

   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv run python \
     scripts/run_physical_stitch_wave.py \
     data/agents/stitching/batches/physical_context_v7_20260715_manifest.json \
     --validate-only
   ```

4. Resume the counterbalanced schedule:

   ```bash
   PATH=/home/brad/.opencode/bin:$PATH \
   UV_CACHE_DIR=/tmp/uv-cache uv run python \
     scripts/run_physical_stitch_wave.py \
     data/agents/stitching/batches/physical_context_v7_20260715_manifest.json
   ```

5. Treat the first Claude quota/rate-limit or timeout symptom as a wave-level
   stop. Preserve completed partial CSVs and rerun the same command later; the
   runner resumes completed groups. Use `--retry-timeouts` only when timeout
   ballots were actually persisted and need selective replacement.

### Analysis after voting

Before human evaluation, analyze vote choice, confidence, reasoning, and
`pack_feedback` per provider and per paired group. The primary contrasts are:

- enriched vs no physical evidence;
- enriched vs no coincidence context;
- the full 2x2 interaction across the five repeated groups;
- behavior on known bridge/tunnel/layer regression pairs;
- NONE as genuine reject-all vs a missing exact option; and
- whether physical/coincidence context helps with the dominant frontage-road
  and vertically layered-road ambiguities without overvaluing continuity.

Then generate a v7-only stitching-review pack containing all 50 unique groups
and their current evidence snapshots. Factorial repeats should be compared in
the agent analysis but deduplicated for Brad's manual truth review. Do not mix
in v6 cases unless the same group independently appears in this v7 manifest.
