# Ablation Study & Dataset Coverage Analysis

February 2026

## Summary

A full ablation study (74 features, 3,580 labels) revealed that the standard single-feature ablation methodology systematically underestimates feature importance when used with tree ensembles like XGBoost. Permutation importance analysis and bulk removal testing confirmed that most features classified as "noise" are actively used by the model. Separately, an audit of dataset coverage identified critical gaps in geographic, structural, and linguistic diversity that limit the model's ability to generalize beyond US-style road networks.

---

## Part 1: Ablation Study Findings

### Baseline Performance (Feb 13, 2026)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Accuracy | 89.3% | 90% | Below threshold |
| F1 | 89.9% | — | — |
| CV F1 Mean | 91.5% +/- 0.6% | 90% | Passing |
| Features | 74 | — | — |

Improvement from prior run (Feb 3): accuracy 85.4% -> 89.3%, CV F1 88.0% -> 91.5%. The improvement is due to 6 new features added between runs (74 vs 68).

### Single-Feature Ablation Results

The standard ablation (remove one feature, measure F1 delta) classified:

| Classification | Count | Threshold |
|----------------|-------|-----------|
| Noise | 64 | F1 delta >= 0 |
| Redundant | 8 | F1 delta > -0.005 |
| Useful | 2 | F1 delta > -0.01 |
| Important | 0 | F1 delta <= -0.01 |

Only two features classified as "useful": `class_similarity` (-0.90% F1) and `ref_coverage` (-0.70% F1). Zero features classified as "important."

### Why Single-Feature Ablation Is Misleading

**The redundancy masking problem:** XGBoost routes around any single missing feature by using correlated alternatives. When you remove `buffer_iou_15m`, the model shifts weight to `buffer_iou_5m`, `hausdorff_distance_m`, and other geometric features. So every individual feature looks dispensable, even though the model relies on many of them.

**Bulk removal test:** Removing all 64 "noise" features at once (keeping only 10 features) caused a **-2.9% F1 drop** (89.9% -> 87.0%) and **-2.9% CV F1 drop** (91.5% -> 88.6%). This proves the features collectively carry significant information that the single-feature ablation cannot detect.

### Permutation Importance Analysis

Permutation importance (shuffling feature values in the test set, 5 repeats) avoids the redundancy masking problem because it breaks correlations without removing the feature. This revealed **14 false negatives** — features the ablation called "noise" that the model actively relies on:

| Feature | Ablation Classification | Permutation F1 Drop | Ablation F1 Delta |
|---------|------------------------|--------------------|--------------------|
| buffer_iou_15m | noise | +2.50% | 0.00% |
| heading_delta | noise | +0.90% | +0.09% |
| aligned_length_m | noise | +0.55% | +0.37% |
| min_coverage | noise | +0.53% | +0.19% |
| name_metaphone | noise | +0.44% | +0.27% |
| name_jaro_winkler | noise | +0.42% | +0.02% |
| sinuosity_delta | noise | +0.35% | +0.19% |
| sinuosity_target | noise | +0.24% | +0.08% |
| min_length_m | noise | +0.23% | 0.00% |
| from_degree_ref | noise | +0.17% | +0.19% |
| post_node_continuation_m | noise | +0.15% | +0.45% |
| endpoint_heading_divergence | noise | +0.15% | +0.18% |
| heading_consistency_ref | noise | +0.13% | +0.27% |
| clustering_coef_ref | noise | +0.11% | +0.11% |

`buffer_iou_15m` is the **second most important feature** by permutation importance but was classified as zero-impact noise by ablation.

### Category Importance (Ablation)

Category-level ablation (removing all features in a category) is somewhat more reliable because it removes the redundancy within a category:

| Category | F1 Delta | Classification |
|----------|----------|----------------|
| alignment_coverage | -1.72% | Important |
| class | -0.90% | Useful |
| name_similarity | -0.88% | Useful |
| crossing_angle | -0.28% | Redundant |
| geometric | -0.25% | Redundant |
| length | 0.00% | Noise |
| topology | +0.04% | Noise |
| intersection_overlap | +0.11% | Noise |
| endpoint_connectivity | +0.16% | Noise |
| sinuosity | +0.16% | Noise |
| shape_complexity | +0.19% | Noise |
| heading_consistency | +0.25% | Noise |
| lateral_offset | +0.32% | Noise |
| graphlet | +0.37% | Noise |
| parallel_sibling | +0.38% | Noise |
| clustering | +0.51% | Noise |
| vertex_density | +0.53% | Noise |

Note: even category ablation still suffers from cross-category redundancy. For example, topology features at +0.04% doesn't mean topology is useless — it means the model can compensate using geometry and name features when topology is removed. With more diverse training data (where topology signals diverge from geometric ones), topology features would likely show higher importance.

### Recommendations

1. **Do not remove features based on the current single-feature ablation.** The methodology has a known blind spot with tree ensembles.
2. **Permutation importance is the more trustworthy signal** for understanding which features the model uses. Both methods should be run together (TODO added to add permutation importance to the ablation script).
3. **Even features at zero permutation importance should be retained.** With ~3,500 labels, statistical power is limited. Features that appear useless on current data may become discriminative with more diverse training data (see Part 2).
4. **Revisit pruning at 10,000+ labels** across diverse geographies. At that scale, importance signals stabilize and you can make confident pruning decisions.

---

## Part 2: Dataset Coverage Analysis

### Current Label Distribution

**3,595 human labels + 1,051 agent labels across 21 datasets.**

> **Editor's note (2026-07-02):** The figures above reflect the label store as of February 2026, when this analysis was written. Current counts: **5,487 human labels across 34 datasets** (3,291 match / 2,166 no_match / 30 unsure) and **1,051 agent labels across 15 datasets**. The analysis below has not been re-run against the updated label store.

#### US Datasets (19 datasets, ~3,100 labels)

| Dataset | Labels | Match:No Match | Segments |
|---------|--------|---------------|----------|
| us_boston_streets | 639 | 274:352 | 10,844 |
| us_philadelphia_sidewalks | 345 | 203:142 | 204,760 |
| us_boston_sidewalks | 281 | 76:204 | 110,031 |
| us_fort_collins_sidewalks | 238 | 211:27 | 38,714 |
| us_fort_collins_streets | 202 | 116:86 | 10,994 |
| us_utah_slc_roads | 200 | 130:70 | 74,452 |
| us_seattle_sidewalks | 200 | 85:115 | 46,145 |
| us_montana_helena | 192 | 68:124 | 8,174 |
| us_frisco_trails | 177 | 92:85 | 529 |
| us_usfs_lolo | 107 | 97:9 | 783 |
| us_boston_bike_network | 86 | 34:52 | 3,477 |
| us_austin_sidewalks | 44 | 44:0 | 11,945 |

#### International Datasets (7 datasets, ~500 labels)

| Dataset | Labels | Match:No Match | Segments |
|---------|--------|---------------|----------|
| au_sydney_roads | 226 | 125:101 | 178,227 |
| hk_hongkong_roads | 208 | 77:131 | 36,107 |
| co_bogota_roads | 134 | 93:41 | 138,516 |
| ke_kisumu_roads | 107 | 104:3 | 722 |
| au_melbourne_roads | 91 | 76:15 | 2,927 |
| de_berlin_roads | 74 | 40:34 | 43,369 |
| ca_toronto_roads | 33 | 30:3 | 62,411 |
| in_mumbai_streets | 10 | 5:5 | 56,797 |
| gb_london_roads | 1 | 1:0 | 264,756 |

### Coverage Gaps

The current training data answers: "Can we match US-style professionally digitized roads?" It does not yet answer: "Can we generalize across digitization regimes, naming systems, and network topologies?"

#### Gap 1: European National Mapping Agency (NMA) Data

European NMAs produce the world's best professionally digitized road data, but with fundamentally different conventions from US municipal datasets:
- Segments split only at intersections (not microsegmented)
- Roundabouts modeled as arcs, not point-to-point chords
- Dual carriageways with explicit median modeling
- Slip roads and channelized intersections
- Numbered route naming (A6, D123) vs street names

No European NMA data is currently represented with meaningful label volume (Berlin has 74, London has 1).

#### Gap 2: Non-Latin Naming Systems

85%+ of current labels are from English-language datasets. Features like `name_metaphone` and `name_soundex` are English-phonetic algorithms — they produce meaningless output on Japanese, Korean, or Arabic text. The model has never been forced to match without name features, so we don't know whether it can. This is the most significant generalization risk: if the model leans on name features that silently fail on non-Latin scripts, international performance will degrade.

#### Gap 3: Structured Non-Grid Networks

Most US road networks are grid-based. The model has limited exposure to structured but non-grid networks (concentric canals, radial boulevards, organic medieval cores). These have different intersection angles, segment lengths, and topology patterns that stress geometry and topology features differently.

#### Gap 4: Sparse Rural Professional Networks

Current rural coverage is limited to US Forest Service roads and Montana. Scandinavian national mapping agencies produce excellent rural road data with very different characteristics: long segments with few intersections (low-degree topology), sparse naming, large geometric offsets between sources. This regime tests whether topology and geometry features work when the network is sparse.

#### Gap 5: ML-Extracted Road Networks

No ML-extracted road data is currently in training. While the matcher is focused on professionally digitized data today, production-grade ML-extracted road networks from major providers (Google, Microsoft, etc.) are an increasingly likely target dataset. These have distinct characteristics: no names, simplified geometry, topology gaps, spatially correlated positional offset from orthorectification. Ensuring readiness for this regime requires at least evaluation coverage.

### Label Balance Issues

Several international datasets have class imbalances that limit their training value:

| Dataset | Balance | Issue |
|---------|---------|-------|
| ke_kisumu_roads | 104:3 | Almost no negative examples |
| ca_toronto_roads | 30:3 | Too few negatives |
| us_austin_sidewalks | 44:0 | Zero negatives |
| us_usfs_lolo | 97:9 | Extreme match bias |

These datasets contribute almost nothing to learning the no-match decision boundary.

---

## Part 3: Dataset Expansion Priorities

### Priority 1: UK Ordnance Survey (OS Open Roads)

**Data:** Freely available via OS OpenData. Already loaded (264k segments, 1 label).

**Why highest priority:**
- English names isolate geometry/topology effects from linguistic effects. This lets us distinguish "does the model fail because of geometry differences?" from "does the model fail because names don't work?"
- European NMA segmentation: segments split at intersections only, roundabouts as arcs, dual carriageways with explicit median modeling
- Slip roads, channelized intersections, motorway junctions — intersection complexity the model hasn't seen
- Lowest cost to add: data already loaded, just needs labeling

**Target:** 200 balanced labels (100 match, 100 no-match). Focus on a mix of: urban streets, motorway junctions, roundabouts, dual carriageways, rural roads.

**What it tests:** European segmentation regime, roundabout geometry, dual carriageway matching. Isolates geometry/topology gaps from naming gaps.

### Priority 2: Netherlands NWB (Nationaal Wegenbestand)

**Data:** NWB (Nationaal Wegenbestand) is open data from Kadaster/Rijkswaterstaat. Excellent geometric precision.

**Why high priority:**
- Turbo roundabouts — unique intersection geometry not found anywhere else. Complex arc modeling that stresses geometric features
- Separated bike lane infrastructure everywhere — bike lanes as independent centerlines adjacent to roads. Tests parallel sibling features and near-match disambiguation
- Structured non-grid network (Amsterdam canal rings, radial boulevards) — different topology patterns from both US grids and organic European old towns
- Non-English Latin script — Dutch names test name features beyond English but without the total failure mode of non-Latin scripts

**Target:** 200 balanced labels. Mix of: urban Amsterdam, turbo roundabouts, bike lanes alongside roads, rural polders.

**What it tests:** Complex intersection geometry, parallel infrastructure disambiguation, structured non-grid topology, non-English Latin naming.

### Priority 3: Japan (MLIT Road Data)

**Data:** Ministry of Land, Infrastructure, Transport and Tourism (MLIT) publishes road data. Also available through Overture/OSM imports.

**Why critical:**
- Non-Latin script stress test. `name_metaphone`, `name_soundex`, `name_token_sort` produce meaningless output on Japanese characters. This forces the model to match entirely on geometry, topology, and structural features — revealing whether it actually can
- Dense structured network with unique characteristics: very short segments, complex rail/road grade separations, underground pedestrian networks near stations
- Different classification conventions: road classes based on administrative hierarchy (national, prefectural, municipal) rather than physical characteristics

**Target:** 200 balanced labels across Tokyo urban core + suburban areas.

**What it tests:** Model performance when name features are effectively disabled. Dense structured network matching. Non-Latin script robustness.

### Priority 4: Scandinavia (Sweden Lantmateriet or Norway Kartverket)

**Data:** Both agencies publish excellent open road data. Lantmateriet (Sweden) and Kartverket (Norway) have well-documented APIs and download portals.

**Why valuable:**
- Sparse professional rural networks: long segments (1-10km), low intersection density (degree 1-2 dominates), minimal naming. This is the opposite topology regime from everything in current training
- Tests whether topology features add value in low-degree networks (they may behave very differently when most nodes are degree 2)
- Latin script but non-English — Swedish/Norwegian names test name features on unfamiliar Latin-script languages
- High geometric precision with potentially large offsets from Overture in rural areas (fewer constraints to anchor alignment)

**Target:** 200 balanced labels. Mix of: rural roads, small town networks, highway interchanges.

**What it tests:** Sparse topology regime, model behavior with long segments and few intersections, name features on non-English Latin script.

### Priority 5: ML-Extracted Roads (SpaceNet or Similar)

**Data:** SpaceNet road extraction challenge datasets, or future production ML road products from major providers.

**Why include:**
- Future-proofing. Production-grade ML-extracted road centerlines from satellite imagery are an increasingly likely target dataset for the matcher
- Distinct data characteristics: no names, simplified geometry (fewer vertices), topology gaps (disconnected at intersections), spatially correlated positional offset from orthorectification
- Forces the model to work without name features and with degraded topology — similar to the Japan stress test but for different reasons
- Understanding which feature categories carry the weight when names and topology are degraded informs feature development priorities

**Target:** 200 balanced labels from a SpaceNet urban extraction, initially for evaluation only (not training). Assess whether the model generalizes or catastrophically fails before deciding whether to include in training.

**What it tests:** Model robustness to missing names, simplified geometry, broken topology. Readiness for ML-extracted road products.

### Summary: 1,000 Labels Across 5 Orthogonal Stress Dimensions

| Priority | Dataset | Unique Stress Axes | Labels | Cost |
|----------|---------|-------------------|--------|------|
| 1 | UK OS Open Roads | European NMA segmentation, roundabouts, dual carriageways | 200 | Free data, labeling only |
| 2 | Netherlands NWB | Turbo roundabouts, bike infra, structured non-grid | 200 | Free data, labeling only |
| 3 | Japan MLIT | Non-Latin naming, dense structured, rail/road | 200 | Free/open data |
| 4 | Sweden/Norway | Sparse rural, long segments, low-degree topology | 200 | Free data |
| 5 | ML-extracted (SpaceNet) | No names, simplified geometry, broken topology | 200 | Free data, eval-first |

**Expected outcome:** These 1,000 labels across 5 datasets would transform the training set from "US-centric with some international" to "tested across digitization regimes, naming systems, density regimes, topology structures, and data quality levels." Feature importance signals would stabilize, revealing which features genuinely generalize and which are overfit to US conventions.
