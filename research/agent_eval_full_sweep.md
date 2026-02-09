# Agent Evaluation: Full Sweep - Claude Opus vs Human Labels

**Date**: 2026-02-08
**Batch**: `sweep_2026-02-08_175328`
**Model**: Claude Opus (via Claude Code CLI)

## Experiment Design

Two image variants evaluated on the same candidate pool:
- **subline_carto_positron**: Aligned sublines on CartoDB Positron basemap tiles
- **subline_road_context**: Aligned sublines with gray dashed context roads (no basemap)

Each candidate includes metadata (names, classes, 67 ML features from FeatureStore) and one image variant per run. The agent reads metadata.yaml + views the image, then outputs match/no_match/unsure with confidence and reasoning.

### Candidate Pool

2,425 candidates across 13 datasets (all human-labeled match/no_match pairs with >=5 of each class):

| Dataset | Match | No Match | Total |
|---------|-------|----------|-------|
| us_boston_streets | 275 | 351 | 626 |
| us_boston_sidewalks | 77 | 203 | 280 |
| us_fort_collins_sidewalks | 214 | 24 | 238 |
| au_sydney_roads | 139 | 87 | 226 |
| us_montana_helena | 68 | 124 | 192 |
| us_frisco_trails | 93 | 84 | 177 |
| us_philadelphia_sidewalks | 95 | 50 | 145 |
| co_bogota_roads | 97 | 37 | 134 |
| us_usfs_lolo | 97 | 9 | 106 |
| au_melbourne_roads | 76 | 12 | 88 |
| us_boston_bike_network | 34 | 52 | 86 |
| de_berlin_roads | 40 | 34 | 74 |
| us_boston_streets_osm | 8 | 45 | 53 |
| **Total** | **1,313** | **1,112** | **2,425** |

### Excluded Datasets (<5 of either class)

gb_london_roads, us_austin_sidewalks, us_fort_collins_streets, ke_kisumu_roads, ca_toronto_roads, hk_hongkong_roads

## Overall Results

| Variant | Accuracy | Precision | Recall | F1 | N (excl. unsure) |
|---------|----------|-----------|--------|-----|-------------------|
| **subline_carto_positron** | **86.0%** | 88.8% | 84.8% | **0.868** | 2,358 |
| **subline_road_context** | 85.6% | **91.9%** | 80.5% | 0.858 | 2,327 |

- Carto positron has slightly higher accuracy and F1 (+1 pt F1)
- Road context has significantly higher precision (+3.1 pts) but lower recall (-4.3 pts) -- more conservative
- Both produce ~67-73 "unsure" labels (~2.8% of candidates)

## Per-Dataset Accuracy

| Dataset | Carto Acc | Road Acc | Best | N (carto) |
|---------|-----------|----------|------|-----------|
| us_montana_helena | **95.8%** | 93.5% | carto | 192 |
| us_boston_sidewalks | 93.5% | **94.5%** | road | 278 |
| us_frisco_trails | 93.8% | **95.2%** | road | 176 |
| us_boston_streets_osm | 90.6% | **94.3%** | road | 53 |
| us_boston_bike_network | 90.7% | **93.6%** | road | 75 |
| us_fort_collins_sidewalks | **90.2%** | 86.0% | carto | 234 |
| us_philadelphia_sidewalks | 89.5% | **90.2%** | road | 143 |
| us_boston_streets | 85.9% | **87.9%** | road | 612 |
| us_usfs_lolo | **83.3%** | 80.6% | carto | 102 |
| de_berlin_roads | 76.4% | **79.2%** | road | 72 |
| co_bogota_roads | **74.6%** | 74.2% | ~tie | 126 |
| au_melbourne_roads | **71.4%** | 67.9% | carto | 84 |
| au_sydney_roads | **68.7%** | 62.8% | carto | 211 |

US datasets generally perform well (83-96%). International datasets (Australia, Colombia, Germany) are harder (63-79%).

## Error Analysis

### Error Type Breakdown

| | False Positives | False Negatives | Total Errors |
|---|---|---|---|
| **Carto positron** | 136 | 193 | 329 (14.0%) |
| **Road context** | 89 | 245 | 334 (14.4%) |

Road context's conservatism shows: fewer FPs (89 vs 136) but many more FNs (245 vs 193).

### Per-Dataset Error Breakdown

| Dataset | Carto FP | Carto FN | Carto Err | Road FP | Road FN | Road Err |
|---------|----------|----------|-----------|---------|---------|----------|
| au_melbourne_roads | 4 | 20 | 24 | 5 | 22 | 27 |
| au_sydney_roads | 23 | 43 | 66 | 19 | 58 | 77 |
| co_bogota_roads | 11 | 21 | 32 | 8 | 24 | 32 |
| de_berlin_roads | 5 | 12 | 17 | 3 | 12 | 15 |
| us_boston_bike_network | 4 | 3 | 7 | 2 | 3 | 5 |
| us_boston_sidewalks | 6 | 12 | 18 | 5 | 10 | 15 |
| us_boston_streets | 60 | 26 | 86 | 32 | 42 | 74 |
| us_boston_streets_osm | 5 | 0 | 5 | 3 | 0 | 3 |
| us_fort_collins_sidewalks | 0 | 23 | 23 | 0 | 32 | 32 |
| us_frisco_trails | 5 | 6 | 11 | 2 | 6 | 8 |
| us_montana_helena | 3 | 5 | 8 | 2 | 10 | 12 |
| us_philadelphia_sidewalks | 7 | 8 | 15 | 5 | 9 | 14 |
| us_usfs_lolo | 3 | 14 | 17 | 3 | 17 | 20 |

Notable: us_boston_streets has the most FPs (60 carto, 32 road) -- likely due to dense urban grid with many parallel/adjacent streets. au_sydney_roads has high FN counts in both variants.

### Reasoning Theme Analysis

Themes extracted via keyword matching from the `reasoning` field of incorrect predictions:

#### False Positives (225 combined across variants)

| Theme | Count | % |
|-------|-------|---|
| Short segment | 194 | 86% |
| Subsegment/coverage | 115 | 51% |
| Parallel/offset | 89 | 40% |
| Same name match | 85 | 38% |
| Different names | 46 | 20% |
| Low hausdorff | 45 | 20% |
| High buffer_iou | 40 | 18% |
| Different class | 22 | 10% |

**Pattern**: Claude over-matches on short segments that happen to sit near a longer line, especially when names agree. It sees low hausdorff / high buffer_iou on the overlapping portion and calls it a match, even when the human labeled it no_match (e.g., adjacent sidewalk, parallel service road).

#### False Negatives (438 combined across variants)

| Theme | Count | % |
|-------|-------|---|
| Short segment | 339 | 77% |
| Parallel/offset | 254 | 58% |
| Different names | 167 | 38% |
| Subsegment/coverage | 139 | 32% |
| Angle/crossing | 56 | 13% |
| Same name match | 44 | 10% |
| Different class | 32 | 7% |
| Endpoint issue | 20 | 5% |

**Pattern**: Claude rejects valid matches when it sees lateral offset (parallel lines ~3-5m apart) or different names. It interprets offset as "separate features" when the human considered them the same road. Also rejects when coverage is low or segments are at slight angles.

### Core Confusion: Subsegment Matching with Parallel/Offset Geometry

The dominant error pattern across both FP and FN is the **ambiguity of short segments near longer ones with some lateral offset**. Claude cannot reliably distinguish:
- A sidewalk running parallel to a road (no_match) from a road with slight GPS offset (match)
- A valid subsegment match with low coverage from an incidental overlap of unrelated features
- Name agreement + geometric proximity from an actual match when offset is present

### Confidence Calibration

| Variant | FP Mean Conf | FN Mean Conf | High-Conf Errors (>=0.85) |
|---------|-------------|-------------|---------------------------|
| Carto positron | 0.71 | 0.74 | 51 (16%) |
| Road context | 0.71 | 0.75 | 64 (19%) |

- Error confidence (0.71-0.75) is lower than overall mean (~0.84), indicating partial calibration
- However, 16-19% of errors are high-confidence -- the model is overconfident on ~50-64 candidates per variant

## Cross-Variant Analysis

### Error Overlap

| | Count |
|---|---|
| Errors in BOTH variants | 229 |
| Errors in carto_positron ONLY | 100 |
| Errors in road_context ONLY | 105 |

70% of errors are shared across variants -- these are fundamentally hard cases regardless of image type.

### Ensemble Signal

| Condition | N | Accuracy |
|-----------|---|----------|
| Both variants **agree** | 2,109 | **89.1%** |
| Variants **disagree** (both match/no_match) | 170 | coin flip (52/48) |

When both variants agree, accuracy jumps +3 pts to 89.1%. When they disagree, neither is reliably better. This suggests an ensemble strategy: auto-accept agreements and flag disagreements for human review.

## Runtime and Cost

### Execution Details

| Metric | Carto Positron | Road Context |
|--------|----------------|--------------|
| Total wall time | 5.6 hr | 5.4 hr |
| Chunks completed | 97 (84 timed) | 97 (82 timed) |
| Avg chunk time | 239s (4.0 min) | 239s (4.0 min) |
| Min chunk time | 201s (3.4 min) | 191s (3.2 min) |
| Max chunk time | 436s (7.3 min) | 305s (5.1 min) |
| Per-candidate avg | 9.5s | 9.6s |
| Chunk failures | 0 | 0 |
| Missing candidates | 0/2425 | 25/2425 (1 chunk) |

Both variants ran in parallel on the same machine. Periodic backups saved every 5 chunks (runner.py feature added in this session).

### Quota Usage

- **Plan**: Max 20x (Anthropic API)
- **Model**: Claude Opus (claude-opus-4-6 via Claude Code CLI)
- **Invocations per variant**: 97 chunks = 97 Claude Code CLI subprocess calls
- **Total invocations**: 194 across both variants
- **Each invocation processes 25 candidates**: reads 25 metadata.yaml files + 25 images, outputs 25 labels
- **Estimated tokens per chunk**: ~50-100K input (images + metadata + prompt + few-shot examples), ~5-10K output (25 labels with reasoning)
- **Total estimated token usage**: ~10-20M input + ~1-2M output per variant, ~20-40M input total across both variants
- **Wall clock**: ~5.5 hours with both running simultaneously (would be ~11 hours sequential)
- **Quota consumed**: Approximately 5 hours of sustained max-rate Opus usage across two parallel streams. This was near the daily quota limit for a 20x plan.

### Reproducibility Notes

- Sweep generation uses `--seed 42` for deterministic candidate sampling
- Batch ID: `sweep_2026-02-08_175328`
- The `agent run` command is resumable: if interrupted, re-running the same command skips already-completed candidates
- Few-shot examples are selected automatically (2 match, 2 no_match) from the batch's ground truth
- Road context variant ended with 2400/2425 labels (25 missing from one chunk -- likely a resume boundary edge case)

### Tips for Future Re-evaluation

1. **Budget ~6 hours per variant** at 25 candidates/chunk on Opus. Can run 2 variants in parallel.
2. **Chunk size 25 is a good balance** -- smaller chunks waste more tokens on repeated prompts/few-shot, larger chunks risk timeout.
3. **600s timeout per chunk is sufficient** -- max observed was 436s. Could reduce to 480s.
4. **Monitor early chunks for quality** -- first 2-3 chunks (~75 labels) are enough to detect garbage. Accuracy stabilized within ~5% of final after 100 labels.
5. **The runner auto-resumes** -- safe to kill and restart if quota runs out. Progress is saved per-chunk.
6. **Periodic backups** (every 5 chunks) protect against data.csv corruption during long runs.
7. **If quota is tight**, run one variant first (recommend carto_positron for higher F1), evaluate, then decide whether the second variant adds enough value.

## Conclusions and Recommendations

1. **Both variants achieve ~86% accuracy** across 2,425 candidates -- strong baseline for automated labeling
2. **Road context is more conservative** (higher precision, lower recall) -- better for avoiding false matches
3. **Carto positron is more balanced** (higher F1) -- better overall discriminator
4. **Ensemble agreement boosts accuracy to 89%** -- practical strategy for production use
5. **Hardest cases involve subsegment/offset ambiguity** -- may need explicit rules or better subline visualization to address
6. **International datasets are hardest** (63-79% accuracy) -- may benefit from region-specific few-shot examples
7. **Name agreement is a double-edged sword** -- it correctly identifies many matches but also drives false positives when geometry is offset

### Potential Improvements

- **Better subline visualization**: Make lateral offset more visually obvious (e.g., color-coded distance)
- **Explicit subsegment rules in prompt**: Define minimum coverage thresholds for valid subsegment matches
- **Region-specific few-shot examples**: Include Australian/international examples in training set
- **Confidence-based filtering**: Flag predictions with confidence < 0.75 for human review (~40% of errors have confidence < 0.70)
- **Ensemble voting**: Use both variants and only auto-accept agreements

## Files Modified for This Evaluation

| File | Change |
|------|--------|
| `src/matcher/cli/agent.py` | Fixed `target_geometry_wkb` -> `target_geometry`; load FeatureStore per dataset; use `FEATURE_COLUMNS` from config; add fallback name/class extraction |
| `src/matcher/agent_labeling/context_generator.py` | Use `FEATURE_CATEGORIES` from config for metadata feature organization |
| `src/matcher/agent_labeling/runner.py` | Add periodic backup of data.csv every 5 chunks |

Committed as `bb3772f` on main.
