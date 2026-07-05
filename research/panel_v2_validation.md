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

## Appendix: raw panel outputs (batch dirs are gitignored)

### `us_boston_streets_panelv2check/votes.csv`

```csv
group_id,provider,model,choice,confidence,latency_s
07632e1f,claude,claude-opus-4-8,A,0.85,29.5
07632e1f,codex,gpt-5.5,A,0.93,9.37
07632e1f,agy,Gemini 3.5 Flash (Medium),A,1.0,16.72
0b3a4f7d,claude,claude-opus-4-8,A,0.72,39.04
0b3a4f7d,codex,gpt-5.5,A,0.93,9.46
0b3a4f7d,agy,Gemini 3.5 Flash (Medium),A,1.0,42.22
0e3e10ad,claude,claude-opus-4-8,A,0.78,42.42
0e3e10ad,codex,gpt-5.5,B,0.89,12.88
0e3e10ad,agy,Gemini 3.5 Flash (Medium),A,1.0,15.87
166ce59a,claude,claude-opus-4-8,A,0.96,26.34
166ce59a,codex,gpt-5.5,A,0.98,7.21
166ce59a,agy,Gemini 3.5 Flash (Medium),A,1.0,8.82
37a546e3,claude,claude-opus-4-8,A,0.95,19.34
37a546e3,codex,gpt-5.5,A,0.96,8.42
37a546e3,agy,Gemini 3.5 Flash (Medium),A,1.0,6.55
461ebf00,claude,claude-opus-4-8,A,0.95,20.66
461ebf00,codex,gpt-5.5,A,0.97,6.47
461ebf00,agy,Gemini 3.5 Flash (Medium),A,0.95,8.67
4bcea059,claude,claude-opus-4-8,A,0.78,26.88
4bcea059,codex,gpt-5.5,A,0.94,22.33
4bcea059,agy,Gemini 3.5 Flash (Medium),A,0.95,13.65
c3a963e9,claude,claude-opus-4-8,A,0.82,30.28
c3a963e9,codex,gpt-5.5,A,0.93,9.8
c3a963e9,agy,Gemini 3.5 Flash (Medium),A,1.0,17.18
```

### `us_boston_streets_panelv2check/consensus.csv`

```csv
group_id,consensus,choice,edge_set,routing,n_votes,n_valid,minority,mean_confidence,route_reason
07632e1f,unanimous,A,"[[""128ec53e-1759-49d8-ac75-3a38643764f4"", ""us_boston_streets_6733_882a30663b""], [""128ec53e-1759-49d8-ac75-3a38643764f4"", ""us_boston_streets_769_882a30663b""], [""7f7d085e-0445-420b-aa22-4755d8e8c3c0"", ""us_boston_streets_6733_882a30663b""], [""7f7d085e-0445-420b-aa22-4755d8e8c3c0"", ""us_boston_streets_9537_882a30663b""]]",auto_accept,3,3,,0.927,
0b3a4f7d,unanimous,A,"[[""043fcb81-b9cf-45e0-b3dc-ead383ecff38"", ""us_boston_streets_8911_882a30646b""], [""15a3921b-30b7-4320-951e-558b77ad8218"", ""us_boston_streets_5934_882a30646b""], [""3835dc1f-309e-48d4-8b81-3798ebc3524c"", ""us_boston_streets_8911_882a30646b""], [""3abcd6d7-9136-413e-ba2c-1ad53edf7c82"", ""us_boston_streets_5934_882a30646b""], [""45a8369e-f2e4-4374-adbb-091489e80b90"", ""us_boston_streets_5934_882a30646b""], [""5608c3e8-97e4-4195-8757-29366968a0d2"", ""us_boston_streets_5934_882a30646b""], [""d4a070e0-3ea8-4548-a27c-73f99f1eae63"", ""us_boston_streets_8911_882a30646b""], [""f42aa536-6c74-4e47-961e-8062362cc52c"", ""us_boston_streets_5934_882a30646b""], [""f42aa536-6c74-4e47-961e-8062362cc52c"", ""us_boston_streets_8911_882a30646b""]]",auto_accept,3,3,,0.883,
0e3e10ad,majority,A,"[[""01bb3500-1d0a-4dda-b3d9-8308808ac925"", ""us_boston_streets_2069_882a30660b""], [""01bb3500-1d0a-4dda-b3d9-8308808ac925"", ""us_boston_streets_4581_882a30660b""], [""8f66a63b-cc18-4230-85fd-ae766c95add9"", ""us_boston_streets_2069_882a30660b""]]",human_review,3,3,codex=B,0.89,
166ce59a,unanimous,A,"[[""08538ac7-6af4-465b-bdc4-15793af199ac"", ""us_boston_streets_2460_882a306685""], [""23eaa6bc-1826-4a35-8bad-a3fb6f49f631"", ""us_boston_streets_2460_882a306685""]]",auto_accept,3,3,,0.98,
37a546e3,unanimous,A,"[[""22fddb24-f47a-4e55-a4fe-a89f71d86109"", ""us_boston_streets_5103_882a3066e7""], [""3359a76b-37bb-4139-84b2-3f5287794bf7"", ""us_boston_streets_5103_882a3066e7""], [""41a6300f-5712-4536-859f-6b10fb37630a"", ""us_boston_streets_5103_882a3066e7""], [""6adb5b10-abca-4184-b277-c011b1b4645f"", ""us_boston_streets_5103_882a3066e7""], [""8dce4768-ab4f-48a1-867f-c6c06e4b649f"", ""us_boston_streets_5103_882a3066e7""], [""ac8e1bc0-f6ff-4f11-bfd5-979f101516a4"", ""us_boston_streets_5103_882a3066e7""], [""b2ad422a-448d-46b6-83ea-bb12bb7c2df3"", ""us_boston_streets_5103_882a3066e7""], [""b69ef9b3-d9bf-4b07-b2be-630c86f082da"", ""us_boston_streets_5103_882a3066e7""], [""bb246ff8-90d2-42a0-8d60-91ca28cd7f38"", ""us_boston_streets_7620_882a3066e3""], [""f5d0c3d3-0e85-4bca-ab50-102f37548d3a"", ""us_boston_streets_5103_882a3066e7""]]",auto_accept,3,3,,0.97,
461ebf00,unanimous,A,"[[""8ef981cb-fc24-4506-ad9b-f5d5d4196d79"", ""us_boston_streets_1936_882a30660d""], [""8ef981cb-fc24-4506-ad9b-f5d5d4196d79"", ""us_boston_streets_7735_882a30660d""]]",auto_accept,3,3,,0.957,
4bcea059,unanimous,A,"[[""07630b8d-4c0b-400a-a00f-a2e66dacb8b2"", ""us_boston_streets_3794_882a3064b5""], [""07630b8d-4c0b-400a-a00f-a2e66dacb8b2"", ""us_boston_streets_4557_882a339a49""], [""a2dfc6db-5ac1-4a34-9e70-69cb058e8e7e"", ""us_boston_streets_6818_882a3064b5""], [""a4d90467-8467-4182-aec6-5028a2fd2662"", ""us_boston_streets_3324_882a3064b5""], [""a4d90467-8467-4182-aec6-5028a2fd2662"", ""us_boston_streets_3794_882a3064b5""], [""a4d90467-8467-4182-aec6-5028a2fd2662"", ""us_boston_streets_9091_882a3064b5""], [""bc9e42cf-e9c9-4fdb-aa12-fd14dfdb8f92"", ""us_boston_streets_10036_882a339a4b""], [""bc9e42cf-e9c9-4fdb-aa12-fd14dfdb8f92"", ""us_boston_streets_3324_882a3064b5""]]",auto_accept,3,3,,0.89,
c3a963e9,unanimous,A,"[[""0e76ac7f-2ae8-4144-9449-f114f797ddcd"", ""us_boston_streets_1013_882a3066ab""], [""0e76ac7f-2ae8-4144-9449-f114f797ddcd"", ""us_boston_streets_2562_882a3066ab""], [""296938c6-db35-4ac8-b2dd-e43c8c7fff4b"", ""us_boston_streets_2562_882a3066ab""], [""c2a1ad97-1797-4ff0-8f8d-a01524e644cc"", ""us_boston_streets_2562_882a3066ab""], [""ca6210b6-9649-4e8a-b656-bf9531a3d529"", ""us_boston_streets_2562_882a3066ab""]]",auto_accept,3,3,,0.917,
```

### `us_seattle_sidewalks_panelv2check/votes.csv`

```csv
group_id,provider,model,choice,confidence,latency_s
2b99c180,claude,claude-opus-4-8,C,0.4,44.82
2b99c180,codex,gpt-5.5,B,0.78,21.47
2b99c180,agy,Gemini 3.5 Flash (Medium),C,0.95,52.23
46e57794,claude,claude-opus-4-8,E,0.62,38.88
46e57794,codex,gpt-5.5,E,0.86,10.34
46e57794,agy,Gemini 3.5 Flash (Medium),A,0.95,87.47
670e939f,claude,claude-opus-4-8,D,0.72,50.07
670e939f,codex,gpt-5.5,D,0.88,15.16
670e939f,agy,Gemini 3.5 Flash (Medium),A,0.95,43.82
e919f4ab,claude,claude-opus-4-8,A,0.9,21.67
e919f4ab,codex,gpt-5.5,A,0.93,10.19
e919f4ab,agy,Gemini 3.5 Flash (Medium),A,1.0,14.1
```

### `us_seattle_sidewalks_panelv2check/consensus.csv`

```csv
group_id,consensus,choice,edge_set,routing,n_votes,n_valid,minority,mean_confidence,route_reason
2b99c180,majority,C,"[[""44e23abe-d2b2-4ccb-9479-56c0fdbf81f7"", ""sea_sidewalk_27160066_8828d542cb""], [""58cd2f6a-eadd-4e0c-b214-e85226c41f68"", ""sea_sidewalk_27161003_8828d542cb""], [""58cd2f6a-eadd-4e0c-b214-e85226c41f68"", ""sea_sidewalk_27161005_8828d542cb""], [""77aa46c9-9488-45dd-adba-0c692d91d287"", ""sea_sidewalk_27150607_8828d542cb""], [""ad5c228f-1191-4711-ae0f-629c73af1777"", ""sea_sidewalk_27150023_8828d542cb""], [""ad5c228f-1191-4711-ae0f-629c73af1777"", ""sea_sidewalk_27150607_8828d542cb""], [""bca8b1ed-761d-4535-8f94-f81768215275"", ""sea_sidewalk_27150607_8828d542cb""], [""d9352aab-f9e6-4945-b373-c7b4d78982ab"", ""sea_sidewalk_27149904_8828d542cb""], [""d9352aab-f9e6-4945-b373-c7b4d78982ab"", ""sea_sidewalk_27161007_8828d542cb""]]",human_review,3,3,codex=B,0.675,
46e57794,majority,E,"[[""4fd1049a-0fe5-498c-891c-3cb0e38ded76"", ""sea_sidewalk_27160036_8828d542cb""], [""4fd1049a-0fe5-498c-891c-3cb0e38ded76"", ""sea_sidewalk_27160057_8828d55535""], [""4fd1049a-0fe5-498c-891c-3cb0e38ded76"", ""sea_sidewalk_27194369_8828d55535""], [""4fd1049a-0fe5-498c-891c-3cb0e38ded76"", ""sea_sidewalk_27194405_8828d55535""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27149785_8828d55523""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27149962_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27150518_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27150773_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27160275_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27160276_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27160277_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27160278_8828d542c9""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27160279_8828d55523""], [""99f74ea8-3f62-4594-b5bb-2b56635d82a9"", ""sea_sidewalk_27194371_8828d542c9""], [""e06aa740-40a6-400f-989b-75764f868f69"", ""sea_sidewalk_27160249_8828d55535""]]",human_review,3,3,agy=A,0.74,
670e939f,majority,D,"[[""007a7358-ce55-4643-bb4f-4d46d70bc40f"", ""sea_sidewalk_27159882_8828d542dd""], [""0fada7b9-a294-4f48-bf31-fcb0aa8be06a"", ""sea_sidewalk_27159882_8828d542dd""], [""e385d178-4004-4655-bf3c-7be0ec69028b"", ""sea_sidewalk_27159882_8828d542dd""], [""e7ee91c5-baa3-43c3-a78b-9a7d74f0cbb3"", ""sea_sidewalk_27159880_8828d542dd""]]",human_review,3,3,agy=A,0.8,
e919f4ab,unanimous,A,"[[""44d2b95c-a45b-4f08-818f-119c872afae2"", ""sea_sidewalk_27151163_8828d54295""], [""44d2b95c-a45b-4f08-818f-119c872afae2"", ""sea_sidewalk_27169203_8828d54295""]]",auto_accept,3,3,,0.943,
```
