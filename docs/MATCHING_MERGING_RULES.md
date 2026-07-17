# Matching & Merging Rules — Canonical Ruleset

Defines the two-stage pipeline for road network conflation: **stitch** (pair matching + graph-level resolution + M:N optimization) and **merge** (network integration). The README has a concise summary; this document is the full reference.

### Pipeline Contract

- **Pair Matching** answers: "Are these the same physical traveled way?"
- **Graph-Level Resolution** answers: "Which of these matches can coexist in a consistent network mapping?"
- **Merging** answers: "How do we modify the base network given accepted matches?"

<!-- BEGIN VERSIONED_MATCHING_CONTRACT -->

### Labeling Contract

The pipeline has two distinct labeling surfaces and they must not silently
substitute for one another:

- A **pair label** records pure physical identity and network role. It is
  intentionally recall-biased and does not enforce graph consistency.
- A **pair-semantics stitching label** records the exact final edge set for an
  optimizer group. It starts from pair identity, then applies graph-level
  coexistence and conflict rules. It is not an approximate “best-looking
  option” and it is not obtained by maximizing the number or summed confidence
  of selected edges.
- A **set-semantics stitching label** records group membership only. It does not
  assert which individual candidate pairs are correct and must not be trained
  or scored as pair truth. Agent option picks and explicit human option
  ratifications produce pair semantics; non-empty manual pill-based human
  review (including de-anchored review) deliberately produces set semantics.
  A human-confirmed empty pill selection remains the unambiguous pair-semantics
  reject-all encoding.

For an enumerated stitching menu, an option is correct only when it contains
**all and only** the accepted edges in the displayed candidate universe. If
every edge is a no-match, no offered option is exact, or the evidence cannot
support an exact set, the agent selects `NONE` and explains which condition
applies. Because `NONE` intentionally covers these different conditions, it
always requires human confirmation before it can become a durable label.

The automated graph-resolution stage described in Section 2 remains planned,
but its rules are already normative for human and agent stitching labels used
to evaluate or train that stage. Graph resolution may demote or reject a
pair-identity match because it conflicts with a better-supported mapping; it
must not turn different physical features or network roles into a match.

### Versioned Agent Rubrics

The following compact blocks are the prompt-ready form of this canonical
document. Their stable rule IDs link compact instructions to the expanded
sections below; the expanded sections explain the rules but do not define a
second policy. The blocks are copied into the installed package because
repository docs are not guaranteed to ship with it. CI compares the copies
exactly, and both pair-labeling and stitching-labeling workflows import those
shared copies.

CI also hashes this entire marked matching contract, including Sections 1 and
2, after normalizing the embedded version text. The hash suffix is part of the
rubric version. Any edit inside the contract therefore requires an intentional
version update and review of the prompt-ready blocks, even when the edit is in
expanded prose rather than a copied block. This detects textual drift; review
is still responsible for judging whether two English statements agree.

<!-- BEGIN MATCH_IDENTITY_RUBRIC -->
CANONICAL MATCH-IDENTITY RUBRIC (version 2026-07-17+9463c80a0f77)
Apply these rules to each candidate pair before considering group-level conflicts:
MI-1. Identity and role: a match requires the aligned portions to represent the same physical traveled way with the same network role. ALONG matches ALONG on the same facility; ACROSS matches only the same dedicated transverse crossing; TURN/CONNECTOR matches only the same facility or hierarchy transition. A regular turn through an ordinary intersection remains ALONG, as does a through facility that merely passes over or under another facility. Treat role compatibility as the default identity gate; document any real-world exception explicitly in this canonical contract rather than inferring one from geometry alone.
MI-2. Representation differences: segmentation points, segment lengths, names, and class tags may differ. One abstract centerline may match each constituent split carriageway as M:N, but the physically separate opposite carriageways do not match each other. Do not force a 1:1 mapping.
MI-3. Intersections and short overlaps: length alone never decides identity. A short same-direction, same-role subline on the same traveled way is a match and may become a junction anchor. A mere endpoint touch, perpendicular crossing, or different-role overlap is not a match.
MI-4. Parallel features: laterally separate carriageways, frontage/service roads, sidewalks, separated cycle tracks, and other neighboring facilities are not matches merely because they are parallel or share a name. A painted, sharrow, or flexpost-separated bike lane on the same pavement matches the road rather than a separately mapped cycleway; a raised or curbed cycle track is a separate feature and matches the corresponding cycleway rather than the road. When the identity decision between such a bike or pedestrian facility and a roadway candidate hinges on whether it is same-pavement or physically separated, and the available evidence does not resolve that distinction — because the pack lacks the physical attributes, lateral-offset or coincidence context, layer data, or close-up imagery that would decide it — do not default to same-pavement identity: treat identity as unresolved (unsure per PL-3 at the pair level, or insufficient-evidence NONE per SA-5 at the stitch level) unless other evidence such as naming continuity, endpoint topology, or coverage partition independently establishes the exact set.
MI-5. Evidence, not verdicts: geometry, direction, topology, names, classes, model scores, overlap lengths, and SLIVER/BORDERLINE tags are evidence. No single tag, score, name, class mismatch, or overlap threshold overrides physical identity and network role. Small offsets from GPS or digitization are acceptable when the paths represent the same way.
MI-6. Replacement test: if replacing one aligned subline with the other would change movement intent (for example ALONG becomes ACROSS or a mainline becomes a ramp), the pair is not a match.
<!-- END MATCH_IDENTITY_RUBRIC -->

<!-- BEGIN PAIR_LABEL_RUBRIC -->
PAIR-LABEL OUTPUT CONTRACT (version 2026-07-17+9463c80a0f77)
PL-1. match: the pair satisfies the canonical match-identity rubric.
PL-2. no_match: the pair represents different physical features or roles, or has no plausible aligned subline after ordinary data noise.
PL-3. unsure: the available evidence cannot determine physical identity or network role reliably. Use uncertainty instead of guessing.
PL-4. Pair labeling is recall-biased: retain plausible same-role identity edges for graph-level resolution rather than rejecting them solely because they are short or participate in another candidate match.
<!-- END PAIR_LABEL_RUBRIC -->

<!-- BEGIN STITCH_ASSIGNMENT_RUBRIC -->
STITCH-ASSIGNMENT OUTPUT CONTRACT (version 2026-07-17+9463c80a0f77)
SA-1. Judge identity first: apply the canonical match-identity rubric independently to every displayed candidate edge. Graph resolution may decide which identity-compatible edges coexist; it must not redefine different features or roles as matches.
SA-2. Preserve legitimate M:N structure: keep all mutually consistent identity edges created by different segmentation or centerline/carriageway representation. Do not prefer a smaller set merely because it is simpler.
SA-3. Resolve actual conflicts only after role-incompatible and different-feature candidates are removed. Among the remaining mutually exclusive identity matches, consider neighborhood support, corridor continuity, and aligned coverage together. No single signal is a universal ordering: longer overlap must not by itself override stronger structural evidence, and a supported short same-way edge may remain as a junction anchor. When the evidence does not establish an exact final set, choose NONE for human review.
SA-4. Exact option semantics: choose an option only when it contains all and only the final accepted edges in the displayed candidate universe. Do not choose the closest option, trade a false positive against a false negative, maximize edge count, or defer to total/mean confidence.
SA-5. NONE semantics: choose NONE when every displayed edge is a no-match, when no offered option exactly represents the final accepted edge set, or when the evidence is insufficient to determine an exact set. State which reason applies so human review can distinguish them.
SA-6. Option metadata is non-normative: optimizer status, option order, edge count, and aggregate confidence are context only and never make an option correct.
<!-- END STITCH_ASSIGNMENT_RUBRIC -->

---

## Section 1: Pair Matching Rules (Pure Identity)

Pair matching determines whether two segments represent the same physical traveled way. Pair matching is intentionally recall-biased. Borderline same-role overlaps should be labeled as matches; graph-level resolution (Section 2) enforces consistency and resolves conflicts. Pair matching does **not** enforce graph consistency.

### Core Principle

**A match requires that the sublines represent the same network role, not just overlapping geometry.**

Two segments that overlap spatially are not necessarily a match. They must represent the same physical traveled way — the same road in the real world — even if the datasets differ in segmentation, naming, or classification.

### Definitions

#### Match (MI-1, PL-1)

The aligned overlapping portions of the GERS segment and the local segment represent the same physical traveled way with the same network role. Segmentation points, names, and classification may differ.

#### No Match (MI-1, PL-2)

The aligned portions are not the same physical traveled way with the same
network role. For example, they are different physical features (sidewalk vs
road), different roles (crosswalk vs mainline), or have no plausible aligned
subline after ordinary GPS/digitization noise. Another candidate being better
does not by itself make this pair a no-match; pair identity is judged
independently.

#### Unsure — Workflow State (PL-3)

`unsure` means the available evidence cannot determine identity or role
reliably. It routes the pair to review; it is not a third physical truth class
and must not be used as a binary classifier training label.

#### New (Conceptual)

"New" is a property of a **segment**, not a pair — it emerges from aggregation. If a local segment receives `no_match` against every candidate, then that segment is "new" (no plausible correspondence exists in Overture). Pair truth is binary (`match` or `no_match`), while the labeling workflow may temporarily output `unsure` to request review; "new" is never a pair-level label. The distinction matters for downstream "add to Overture" logic, not for the ML classifier.

### Network Roles (MI-1)

Every segment in a transportation network serves one of three roles. Match compatibility is constrained by role. Roles are a conceptual tool for labeling and reasoning — they may be inferred implicitly from geometry, topology, and tags rather than stored explicitly.

#### ALONG — Longitudinal / Corridor Movement

Movement along a facility. The most common role. This includes short intersection-internal slices that some datasets produce to represent continuity inside a junction — these are treated as ALONG segments.

Examples: road mainline, bike lane along a road, sidewalk along a street, rail track segment, canal segment, intersection-internal centerline slices.

**Match rule:** Matches with other ALONG segments representing the same facility. Parallel but laterally separated segments (e.g., opposite carriageways, frontage roads, parallel service roads) are not matches unless they represent the same traveled way.

#### ACROSS — Crossing / Transverse Movement

Movement across another facility.

Examples: crosswalk, bike crossing, pedestrian refuge crossing, or another
dedicated transverse connector. A through road or rail line that merely passes
over, under, or across another facility remains ALONG its own corridor.

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

### Edge Cases (MI-2 through MI-5)

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

### Pair Matching Scope (PL-4)

The ML classifier operates as a pair-level identity matcher:

- Primarily 1:1 correspondences (with M:N for split carriageways / different segmentation)
- Recall-biased: borderline same-role overlaps should be matches; graph-level resolution resolves conflicts
- Pair matching does not enforce graph consistency
- The model is trained on binary match/no_match labels — the role concept guides labeling decisions, not the classifier features directly

### Heuristic: The Replacement Test (MI-6)

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
| **PARALLEL_COMPANION** | Alternate-representation correspondence, such as one abstract centerline to a constituent split carriageway. Sibling/opposite carriageways do not match each other. |
| **AMBIGUOUS** | Competing candidates, needs resolution |

### 2.4 Neighborhood Consistency

A match survives graph-level resolution if:
- It is supported by at least one adjacent segment whose accepted match shares a node or corridor with the candidate reference segment, OR
- It is the only geometrically plausible mapping within a defined corridor

Matches without neighborhood support are demoted to AMBIGUOUS and may be rejected if competing candidates exist.

### 2.5 Conflict Resolution (SA-3)

Role compatibility is the default hard identity gate, not a tie-break. Any
real-world exception belongs in the canonical match-identity rules rather than
being inferred from longer overlap or another geometric signal. After removing
role-incompatible or different-feature candidates and preserving legitimate
M:N structure, weigh the following evidence together for genuinely competing
identity matches:

- Neighborhood support
- Corridor continuity versus an isolated clip
- Aligned coverage and overlap length

There is no universal lexicographic order among these signals. Longer overlap
must not by itself override stronger structural evidence, while a supported
short same-way edge may remain as a junction anchor. If the evidence does not
establish an exact final set, leave the conflict AMBIGUOUS and route it to human
review rather than forcing a winner.

Cap max matches per segment except for defined M:N split cases (e.g., dual carriageway).

### 2.6 Output

Graph-level resolution produces:
- **Final accepted matches** with upgraded/downgraded confidence
- **Unmatched segments** classified as:
  - Likely covered (match exists nearby but didn't meet threshold)
  - Likely net new (no plausible correspondence in reference)
  - Ambiguous (insufficient evidence either way)

<!-- END VERSIONED_MATCHING_CONTRACT -->

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
