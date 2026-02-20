# Matching & Merging Rules — Canonical Ruleset

Defines the two-stage pipeline for road network conflation: **stitch** (pair matching + graph-level resolution + M:N optimization) and **merge** (network integration). The README has a concise summary; this document is the full reference.

### Pipeline Contract

- **Pair Matching** answers: "Are these the same physical traveled way?"
- **Graph-Level Resolution** answers: "Which of these matches can coexist in a consistent network mapping?"
- **Merging** answers: "How do we modify the base network given accepted matches?"

---

## Section 1: Pair Matching Rules (Pure Identity)

Pair matching determines whether two segments represent the same physical traveled way. Pair matching is intentionally recall-biased. Borderline same-role overlaps should be labeled as matches; graph-level resolution (Section 2) enforces consistency and resolves conflicts. Pair matching does **not** enforce graph consistency.

### Core Principle

**A match requires that the sublines represent the same network role, not just overlapping geometry.**

Two segments that overlap spatially are not necessarily a match. They must represent the same physical traveled way — the same road in the real world — even if the datasets differ in segmentation, naming, or classification.

### Definitions

#### Match

The aligned overlapping portions of the GERS segment and the local segment represent the same physical traveled way with the same network role. Segmentation points, names, and classification may differ.

#### No Match

Not the correct correspondence. Either:
- A different physical feature (sidewalk vs road, crosswalk vs mainline)
- The correct corresponding feature is a different candidate

#### New (Conceptual)

"New" is a property of a **segment**, not a pair — it emerges from aggregation. If a local segment receives `no_match` against every candidate, then that segment is "new" (no plausible correspondence exists in Overture). Individual pairs are always labeled `match` or `no_match`; "new" is never a pair-level label. The distinction matters for downstream "add to Overture" logic, not for the ML classifier.

### Network Roles

Every segment in a transportation network serves one of three roles. Match compatibility is constrained by role. Roles are a conceptual tool for labeling and reasoning — they may be inferred implicitly from geometry, topology, and tags rather than stored explicitly.

#### ALONG — Longitudinal / Corridor Movement

Movement along a facility. The most common role. This includes short intersection-internal slices that some datasets produce to represent continuity inside a junction — these are treated as ALONG segments.

Examples: road mainline, bike lane along a road, sidewalk along a street, rail track segment, canal segment, intersection-internal centerline slices.

**Match rule:** Matches with other ALONG segments representing the same facility. Parallel but laterally separated segments (e.g., opposite carriageways, frontage roads, parallel service roads) are not matches unless they represent the same traveled way.

#### ACROSS — Crossing / Transverse Movement

Movement across another facility.

Examples: crosswalk, bike crossing, rail crossing, road crossing over tracks, pedestrian refuge crossing.

**Match rule:** Never matches with ALONG. Never matches with TURN. Only matches with other ACROSS segments at the same crossing.

#### TURN / CONNECTOR — Hierarchy or Facility Transition

Movement that transitions between network levels or facility types — not a simple turn at an intersection (which is still ALONG).

Examples: highway off-ramps, slip roads, bike turn pockets at facility transitions, sidewalk curb ramps, rail switches. A regular left turn at a signalized intersection is still ALONG.

**Match rule:** Matches only with same role and same intent (e.g., an off-ramp matches another off-ramp, not a through-lane).

#### Role Compatibility Matrix

| | ALONG | ACROSS | TURN |
|---|:---:|:---:|:---:|
| **ALONG** | Yes | Never | Never |
| **ACROSS** | Never | Yes | Never |
| **TURN** | Never | Never | Same intent |

### Edge Cases

| Scenario | Result | Why |
|----------|--------|-----|
| Different segmentation points | Match | Same road, split differently |
| Split carriageways vs single centerline | M:N Match | Carriageway modeling and segmentation may differ between datasets |
| Road vs parallel sidewalk | No Match | Different physical features |
| Same road, different names | Match | Names are a signal, not a requirement |
| Opposite carriageways of divided road | No Match | Different physical traveled ways, even if part of the same road |
| Road vs crosswalk at intersection | No Match | Different roles: ALONG vs ACROSS |
| Short overlap at intersection | Match | Same traveled way; graph-level resolution evaluates consistency |
| Short colinear overlap near node | Match | Same traveled way for that subsegment; graph-level resolution resolves |
| Road mainline vs slip road/ramp | No Match | Different roles: ALONG vs TURN |
| Bike lane on same pavement as road | Match (to road) | Same physical surface, ALONG + ALONG |
| Separated cycle track (raised/curbed) | No Match (to road) | Different physical feature |

### Intersection Rules

#### Never Match on Overlap Alone (Different Roles)

Geometric overlap at an intersection is not sufficient when roles differ. Many different features (crosswalks, turn lanes, through-lanes, bike crossings) converge at intersections and overlap spatially without being the same feature. A crosswalk overlapping a road at an intersection is still No Match.

#### Same-Role Overlaps Near Intersections

If any subsegment represents the same physical traveled way, it is a match regardless of length. Length alone is not a disqualifier in pair matching; short overlaps are resolved at the graph level. Small gaps from GPS noise, digitization offset, or simplification do not disqualify.

For same-role overlaps near intersection nodes:
1. If the segments share a subsegment along the same direction, they are a match
2. Graph-level resolution decides whether to keep, demote, or reject based on neighborhood context

### Pair Matching Scope

The ML classifier operates as a pair-level identity matcher:

- Primarily 1:1 correspondences (with M:N for split carriageways / different segmentation)
- Recall-biased: borderline same-role overlaps should be matches; graph-level resolution resolves conflicts
- Pair matching does not enforce graph consistency
- The model is trained on binary match/no_match labels — the role concept guides labeling decisions, not the classifier features directly

### Heuristic: The Replacement Test

A helpful mental model: if replacing one aligned subline with the other would change the intent of the network (e.g., through movement becomes a turn, along becomes across), it is not a match.

---

## Section 2: Graph-Level Resolution (Planned)

> **Status: Planned** — Not yet implemented in code. This stage runs within the stitch pipeline after pair scoring and before M:N optimization.

### 2.1 Purpose

Graph-level resolution resolves pairwise matches into a coherent network mapping. It may promote, demote, or reject matches based on graph context. The goal is to enforce topological consistency across the match set without losing identity signal from pair matching.

### 2.2 Junction Zones

A **junction zone** is defined as:
- A segment endpoint within R meters of a degree≠2 node, OR
- A segment with length < L_short near such a node

In junction zones:
- Short overlaps are provisional anchors (not automatically rejected)
- Multiple candidates may temporarily exist
- Final acceptance requires neighborhood consistency (Section 2.4)
- Junction anchors preserve topological continuity but do not automatically authorize attribute transfer

### 2.3 Resolution Status

Graph-level resolution assigns each match a status reflecting its graph-level outcome:

| Status | Description |
|------|-------------|
| **STRONG_EDGE** | High-confidence corridor match with neighbor agreement |
| **JUNCTION_ANCHOR** | Short overlap near intersection, preserved for topology continuity |
| **PARALLEL_COMPANION** | Parallel match (dual carriageway / sibling detection) |
| **AMBIGUOUS** | Competing candidates, needs resolution |

### 2.4 Neighborhood Consistency

A match survives graph-level resolution if:
- It is supported by at least one adjacent segment whose accepted match shares a node or corridor with the candidate reference segment, OR
- It is the only geometrically plausible mapping within a defined corridor

Matches without neighborhood support are demoted to AMBIGUOUS and may be rejected if competing candidates exist.

### 2.5 Conflict Resolution

When multiple matches compete for the same segment:
1. Prefer longer overlap
2. Prefer better role compatibility
3. Prefer neighborhood-supported candidates
4. Prefer corridor-continuous candidates over isolated short matches

Cap max matches per segment except for defined M:N split cases (e.g., dual carriageway).

### 2.6 Output

Graph-level resolution produces:
- **Final accepted matches** with upgraded/downgraded confidence
- **Unmatched segments** classified as:
  - Likely covered (match exists nearby but didn't meet threshold)
  - Likely net new (no plausible correspondence in reference)
  - Ambiguous (insufficient evidence either way)

---

## Section 3: Merging Rules (Network Integration)

> **Status: Planned** — Not yet implemented as a formal stage. Defines the target architecture for integrating accepted matches into the base network.

### 3.1 Geometry Policy

For each accepted match, determine how to handle geometry:
- **Replace**: Use reference geometry (default for STRONG_EDGE matches)
- **Average**: Blend reference and target geometry (for moderate confidence)
- **Keep**: Retain target geometry as-is (when reference geometry is lower quality)

### 3.2 Attribute Transfer

Transfer attributes from target to reference based on:
- Confidence threshold for each attribute type
- Priority rules when multiple targets match the same reference
- Schema mapping between source and target attribute schemas

### 3.3 Net-New Gating

Unmatched target segments are candidates for addition to the reference network. Gating criteria:
- Must pass screening (not water, not building, not landcover)
- Must pass minimum length and connectivity checks
- **Key rule**: Unmatched segments inside junction zones require explicit absence confirmation before being classified as net new — they must pass additional absence checks to confirm the reference network genuinely lacks this feature

### 3.4 Provenance

All merge operations record:
- Source match confidence
- Match type (1:1, M:N, net new)
- Original source dataset and segment ID
