# Matcher TODO

Actionable backlog for the road network matcher.

- For tried-and-removed features, see [docs/RESEARCH_GRAVEYARD.md](docs/RESEARCH_GRAVEYARD.md).
- For exploratory research ideas, see [docs/RESEARCH_IDEAS.md](docs/RESEARCH_IDEAS.md).

---

## Known Issues & Technical Debt

### HIGH: Scalability - Large Dataset Support

- **Problem**: `runner.py` uses `geopandas.read_parquet` which loads entire dataset into memory
- **Impact**: Will fail on state-sized or larger datasets
- **Location**: `src/matcher/pipeline/runner.py`
- **Solution**: Migrate to Spark/Sedona/GraphFrames (see [docs/RESEARCH_IDEAS.md](docs/RESEARCH_IDEAS.md#spark-migration-research-jan-2026))

### HIGH: Robust Feature Backfill Validation

**Problem**: Backfill now uses stored geometries from `labels/data/` (stable), but fallback to raw data lookup can still produce wrong features if data has been re-fetched with different filtering or extent.

**Needed**:
1. Candidate validation after resolving geometries (centroids within buffer distance)
2. Fallback rejection when stored geometry isn't available
3. Audit trail logging when backfill uses fallback lookup

**Location**: `src/matcher/cli/labels.py:backfill_features()`

### Medium: Divergence Detection Fails on Winding Roads (PR #81 follow-up)

**Problem**: `_detect_divergence_endpoints` doesn't truncate the ref subline when the reference loops far away from the target and back. The 1D offset model extends the ref subline through the loop because:

1. **No target parameter bounds check**: When `t - offset` falls outside `[0, target_length]`, `_interpolate_along_line` silently clamps to the target endpoint instead of flagging the sample as out-of-bounds. So the ref can extend arbitrarily beyond the target's actual extent.
2. **Coarse inter-sample direction vectors**: Direction parallelness is computed between consecutive samples, not from local line tangents. With 32 samples over a long comparison region, a winding loop within a single sample gap averages out and can pass the dot product check.
3. **20m distance threshold too generous**: `DIVERGENCE_MIN_DISTANCE_M = 20.0` lets winding roads stay "close enough" when they loop back near the target.

**Fix options** (in order of simplicity):
1. **Target parameter bounds check** (simplest): At line 256, if `t - offset < 0` or `t - offset > target_length`, mark the sample as diverged. This directly truncates where the ref extends beyond the target's extent.
2. **Monotonicity constraint**: Track the projected target parameter and truncate where it stops increasing (the ref is going somewhere the target doesn't).
3. **Local tangent-based direction**: Compute bearing from each line's own vertex segments instead of inter-sample differences.

**Reproduction pair**:
- GERS: `e720aba9-61d0-410a-b5f2-29eba4ae3048`
- Target: `us_montana_helena_223490_8827927a87`
- Dataset: `us_montana_helena`

<details>
<summary>WKT geometries</summary>

REF:
```
LINESTRING (-112.232657 46.720897, -112.2324275 46.7219862, -112.232585 46.723373, -112.232711 46.724186, -112.232697 46.725128, -112.232742 46.725635, -112.232753 46.726053, -112.232799 46.726247, -112.233405 46.726642, -112.234104 46.726964, -112.234617 46.727295, -112.23492 46.727625, -112.234937 46.727649, -112.235094 46.727867, -112.234941 46.728349, -112.234753 46.728993, -112.234063 46.729484, -112.233782 46.729701, -112.233963 46.729785, -112.235694 46.730586, -112.235963 46.730595, -112.236395 46.730385, -112.236548 46.730087, -112.236968 46.729644, -112.237354 46.729427, -112.238007 46.729564, -112.238556 46.729661, -112.2395 46.729725, -112.2402 46.729701, -112.241156 46.729733, -112.241892 46.72949, -112.242203 46.729526, -112.242523 46.729563, -112.242626 46.730232, -112.242765 46.730539, -112.242717 46.731038, -112.242564 46.731537, -112.242319 46.73222, -112.242166 46.732543, -112.242189 46.732817, -112.242446 46.732793, -112.242622 46.732423, -112.242868 46.732101, -112.24316 46.731828, -112.243394 46.731595, -112.243698 46.731419, -112.243827 46.731113, -112.24377 46.730791, -112.243735 46.730486, -112.243934 46.73026, -112.244075 46.729971, -112.24417 46.729616, -112.244136 46.729303, -112.244114 46.728925, -112.24401 46.72853, -112.24373 46.728264, -112.24331 46.728176, -112.242773 46.728199, -112.242108 46.728087, -112.241571 46.72807, -112.240917 46.728118, -112.240393 46.727964, -112.24009 46.727715, -112.2398836 46.7272639, -112.2402382 46.7274812, -112.2406133 46.7276011, -112.241087 46.7276168, -112.2415235 46.7275282, -112.24377 46.726916, -112.244143 46.72694, -112.244575 46.727093, -112.244878 46.727238, -112.245228 46.727239, -112.245625 46.727135, -112.246104 46.727119, -112.246349 46.72716, -112.246547 46.72733, -112.246721 46.727708, -112.24685 46.727917, -112.247222 46.728425, -112.247875 46.728634, -112.248318 46.729111, -112.2491526 46.7294596)
```

TGT:
```
LINESTRING (-112.23242434877292 46.720523234706526, -112.23247237989453 46.72059746576035, -112.23254796324423 46.720698505678456, -112.23261906939054 46.72080163250172, -112.23266980983105 46.72090344923081, -112.23270514326612 46.72099068387334, -112.23276180450267 46.721092987991256, -112.23279964064412 46.721188267652536, -112.23286240593302 46.721287002613686, -112.23293005715877 46.721378776587564, -112.23293260657752 46.7213799762383, -112.23293429900353 46.721382367533764, -112.2329348335011 46.72138388434046, -112.23293531948968 46.721386212820356, -112.23293550633926 46.72138680094473, -112.23296738485375 46.7214227325603, -112.23303019416011 46.72148877299183, -112.23309338435214 46.721546877690606, -112.23316508338654 46.72161774919121, -112.23323851527113 46.72169181985412, -112.23331373120985 46.7217700817726, -112.23340131695007 46.72185723310831, -112.23348036689842 46.72194909171271, -112.23353880859587 46.72205036188311, -112.23358417351771 46.72216743771035, -112.23363341556833 46.722283518723216, -112.23370323263222 46.722402437825906, -112.23377782873341 46.72251746096316, -112.23384974156684 46.722626063286114, -112.2339367416055 46.72274033710756, -112.2340223061363 46.72283924350553, -112.23411222749624 46.72293624622463, -112.23419907212633 46.723044848936205, -112.23427727855665 46.72314582515257, -112.2343341571855 46.723244609491424, -112.23441453933543 46.723347153813826, -112.23450059883797 46.72345316497114, -112.23458493177853 46.72353805109306, -112.23468447050189 46.72361806970704, -112.23476658101043 46.723679863282996, -112.23489458015865 46.72377099582335, -112.23500722709866 46.72384862721647, -112.23512775406031 46.72392670003219, -112.23524594270731 46.72400257676661, -112.23537322140666 46.724089762675966, -112.23548985417156 46.72415306751405, -112.2356483592085 46.724225904934336, -112.23575852500167 46.72426756841183, -112.2358990852925 46.72430165870298, -112.23605434112305 46.72432602789971, -112.23621906160344 46.72435620410935, -112.23655901195472 46.72444809479117, -112.23672085962284 46.724491308685465, -112.23687690237757 46.72454654913935, -112.23701131819186 46.72461437957305, -112.2371288286108 46.72467162740742, -112.23724765595998 46.72473938132479, -112.23737461845232 46.724828014399726, -112.23748355355356 46.72488663892968, -112.23758985479438 46.724944966582086, -112.23769120362307 46.7250037726447, -112.2379732404863 46.725113084900045, -112.23822578386213 46.72524246747956, -112.2383698511758 46.72542012347371, -112.23855073773805 46.72565732564286, -112.23873771128466 46.725792698522355, -112.23892433089505 46.726098752285175, -112.2390897951805 46.72642918056333, -112.23927186572227 46.726689501323925, -112.23940937084281 46.726871906331276, -112.2395720692155 46.72701659773178, -112.2397829397453 46.72717877417796, -112.23993988890015 46.72728430674189, -112.24006963347495 46.72737154830854, -112.24041584688041 46.72753965167486, -112.24064989035126 46.72762687558708, -112.24088578075835 46.72761660399675, -112.24118142170823 46.72758625042294, -112.24157080185805 46.72753967753705, -112.24172227757788 46.7274664110906, -112.24195976968112 46.727353875127704, -112.24223465864964 46.727314741767174, -112.24274903487964 46.72720481204493, -112.24310100289279 46.727104949352096)
```

</details>

**Location**: `src/matcher/features/alignment.py:_detect_divergence_endpoints()` (lines 185-325), config thresholds in `src/matcher/config.py:47-49`
**Related PR**: #81 (original divergence detection), #169 (multi-seed offset fix)

### Medium: Robustness Issues

- **Overly broad exception handling** in `blocking/spatial_index.py` — `except Exception: return None` silently swallows errors
- **Race condition in model selection** in `ml.py` — checks file existence but doesn't validate model is loadable
- **CRS validation gap** in `pipeline/runner.py` — no check for null/invalid geometries after reprojection

### Low: Datasets with Polygon Geometries

Some target datasets have Polygon geometries instead of LineStrings (files deleted, need re-fetch):
- `ca_toronto_roads`, `co_bogota_bike_network`, `co_bogota_sidewalks`

---

## Ablation Study

### Add Permutation Importance to Ablation Script

**Problem:** Single-feature ablation systematically underestimates feature importance with tree ensembles due to redundancy masking — XGBoost routes around any one missing feature via correlated alternatives. Feb 2026 ablation classified 64/74 features as "noise", but bulk-removing them causes -2.9% F1. Permutation importance (shuffling feature values) avoids this by breaking correlations without removing the feature entirely.

**Solution:** Add `--permutation` mode (or include alongside existing modes) that:
1. Trains a single model on the train split
2. For each feature, shuffles its test-set values N times (e.g., 5 repeats) and measures F1 drop
3. Reports mean/std importance per feature
4. Cross-references with ablation classification to flag false negatives (features ablation called "noise" but permutation shows are actively used)

**Validation:** Feb 2026 permutation analysis found 14 false negatives, including `buffer_iou_15m` (2nd most important feature by permutation, classified as noise by ablation).

---

## Feature Ideas

### HIGH: Target-Side Aligned Topology (Degree at Subline Endpoints)

**Priority:** High
**Problem:** Target topology uses full segment endpoints for degree computation, but target datasets often don't split segments at intersections (unlike Overture which has explicit connectors). A target road passing *through* an intersection gets degree 1 at its distant endpoints, even though the alignment boundary is at a high-degree intersection node. This makes target-side topology features unreliable for partial matches.

**Solution:** At alignment boundary points, query the target spatial index for nearby target segment endpoints to compute degree at that position (analogous to `compute_aligned_topology_at_position()` which already does this for the ref side using Overture connectors).

**Impact:** Would improve `from_degree_target`, `to_degree_target`, `degree_match_score`, `degree_signature_similarity`, and related features for partial-match cases.

**Location:** `src/matcher/features/spatial_context.py:compute_all_topology()` (geometry-based fallback, lines 1177+)

**Reproduction pair** (target passes through two 4-way intersections but gets degree 1/1):
- Dataset: `au_melbourne_roads`
- GERS: `5001b7be-3b4f-41dc-b395-df221ba66741`
- Target: `au_melbourne_8043_88be630a1b`

<details>
<summary>WKT geometries</summary>

REF:
```
LINESTRING (144.7344467 -37.8673933, 144.7345845 -37.867333)
```

TGT:
```
LINESTRING (144.734621881638 -37.8674845946421, 144.73467978024 -37.8674574316707, 144.734604739062 -37.8673078641339, 144.734537848093 -37.8673383138165)
```

</details>

Features: `from_degree_ref=4, to_degree_ref=4` (correct), `from_degree_target=1, to_degree_target=1` (incorrect — should reflect intersection degree at alignment boundary).

### Dual Carriageway / Centerline Handling

**Priority:** Medium
**Status:** Partially addressed via parallel sibling features

Parallel sibling features (`has_parallel_sibling_ref`, `parallel_fraction_ref`, `offset_vs_half_corridor_ratio`, `offset_over_expected_halfwidth`, `likely_representation_mismatch`) partially address dual carriageway detection. Remaining work:
- Detect split carriageway start/end points (Y-junction patterns)
- Pre-filter dual carriageway cases with specialized logic

---

## Integration

### Conflict Detector
- Detect duplicate matches in integration output (deferred)

---

## Agent Labeling

### Manually Curate Few-Shot Examples
- Current few-shot selection is automatic (random balanced sample from ground truth in other batches)
- Manually curate a set of high-quality examples covering key edge cases: split carriageways, parallel sidewalks, bike lanes, short overlaps, name mismatches
- Store in a dedicated directory (e.g. `data/agents/few_shot/`) so they're reused across batches

---

## Label Data Management

### Label Archive & History
- Archive orphaned labels to `labels/archived/` instead of losing them
- Provide recovery tooling to re-link archived labels

### Data Lineage
- Store data versions in model metadata
- Add `matcher model-info` command to show training data provenance

---

## Other Ideas

### Adaptive Buffer Distance
- Pipeline default is 75m with relaxed heading (90°) and length ratio (20.0) filters
- Could auto-detect optimal buffer per dataset via alignment statistics on sample

### Active Learning
- Use model uncertainty to prioritize labeling candidates

### Bike/Sidewalk Networks
- May need separate model or geometry-only approach
- Bike lane vs cycleway classification issue (PR #111)

---

## References

- **Ruiz-Lendinez et al. (2021)** - "Road Network Conflation Using Semantics and Geometry"
- **Juhasz et al. (2012)** - "Road Network Conflation Based on Iterative Hausdorff Distance Calculation"
- **Volz et al. (2011)** - "Map Conflation Using MRFs"
- **Hootenanny** (open-source conflation tool) - Junction angle distribution algorithms
