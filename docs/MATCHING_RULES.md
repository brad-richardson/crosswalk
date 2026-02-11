# Matching Rules — Canonical Ruleset

Defines what constitutes a match in the matcher conflation pipeline. The README has a concise summary; this document is the full reference.

## Core Principle

**A match requires that the sublines represent the same network role, not just overlapping geometry.**

Two segments that overlap spatially are not necessarily a match. They must represent the same physical traveled way — the same road in the real world — even if the datasets differ in segmentation, naming, or classification.

## Definitions

### Match

The aligned overlapping portions of the GERS segment and the local segment represent the same physical feature with the same network role. Segmentation points, names, and classification may differ.

### No Match

Not the correct correspondence. Either:
- A different physical feature (sidewalk vs road, crosswalk vs mainline)
- The correct corresponding feature is a different candidate

### New (Conceptual)

No plausible correspondence exists in Overture after considering nearby candidates and continuity. A segment is "new" only when no candidate represents the same real-world feature. Still labeled `no_match` in practice — the distinction is conceptual and matters for downstream "add to Overture" logic, not for the ML classifier.

## Network Roles

Every segment in a transportation network serves one of four roles. Match compatibility is constrained by role.

### ALONG — Longitudinal / Corridor Movement

Movement along a facility. The most common role.

Examples: road mainline, bike lane along a road, sidewalk along a street, rail track segment, canal segment.

**Match rule:** Matches only with other ALONG segments representing the same facility.

### ACROSS — Crossing / Transverse Movement

Movement across another facility.

Examples: crosswalk, bike crossing, rail crossing, road crossing over tracks, pedestrian refuge crossing.

**Match rule:** Never matches with ALONG. Never matches with TURN. Only matches with other ACROSS segments at the same crossing.

### TURN / CONNECTOR — Direction-Changing Movement

Movement that connects two ALONG segments with intent to change direction.

Examples: turn lanes, slip roads / ramps, bike turn pockets, sidewalk curb ramps, rail switches.

**Match rule:** Matches only with same role and same intent (e.g., a left-turn pocket matches another left-turn pocket, not a through-lane).

### INTERNAL — Intersection-Scoped Slices

Geometry that exists only to represent continuity inside a junction. Not all datasets produce these.

Examples: short centerline slices inside intersections, pedestrian "through" slices clipped at junctions.

**Match rule:**
- INTERNAL to INTERNAL: may match if they represent the same through-movement
- INTERNAL to ALONG: only if the INTERNAL segment is clearly a clipped ALONG (rare — be conservative)

### Role Compatibility Matrix

| | ALONG | ACROSS | TURN | INTERNAL |
|---|:---:|:---:|:---:|:---:|
| **ALONG** | Yes | Never | Never | Rare |
| **ACROSS** | Never | Yes | Never | Never |
| **TURN** | Never | Never | Same intent | Never |
| **INTERNAL** | Rare | Never | Never | Yes |

## Edge Cases

| Scenario | Result | Why |
|----------|--------|-----|
| Different segmentation points | Match | Same road, split differently |
| Split carriageways vs single centerline | 1:N Match | One Overture centerline to multiple local segments |
| Road vs parallel sidewalk | No Match | Different physical features |
| Same road, different names | Match | Names are a signal, not a requirement |
| Opposite carriageways of divided road | Match (each to its own) | Each carriageway matches independently |
| Road vs crosswalk at intersection | No Match | Different roles: ALONG vs ACROSS |
| Overlap only at/inside intersection | No Match | Spatial overlap alone is insufficient |
| Colinear overlap <10m near node then diverge | No Match | Must continue ≥10m along shared direction |
| Road mainline vs slip road/ramp | No Match | Different roles: ALONG vs TURN |
| Bike lane on same pavement as road | Match (to road) | Same physical surface, ALONG + ALONG |
| Separated cycle track (raised/curbed) | No Match (to road) | Different physical feature |

## Intersection Rules

### Never Match on Overlap Alone

Geometric overlap at an intersection is not sufficient. Many different features (crosswalks, turn lanes, through-lanes, bike crossings) converge at intersections and overlap spatially without being the same feature.

For a match near an intersection node, **all** of the following must hold:

1. The target continues past the reference node by **≥10m** along the shared direction
2. It stays aligned while doing so (small heading delta and lateral offset)

If the overlap is only inside the intersection and the segments diverge at the node, it is **No Match**.

### Intersection-Internal Exception

If **both** aligned subsegments are fully contained within the same intersection footprint, and they represent the same through-movement, they may be considered a Match even if the overlap is short. This is rare and should be applied conservatively.

## Layer 1: Identity Match

The ML classifier operates as a "Layer 1" identity matcher:

- Primarily 1:1 correspondences (with 1:N for split carriageways)
- Safe for replace/average geometry operations
- Intersection overlap cases are **No Match** at this layer
- The model is trained on binary match/no_match labels — the role concept guides labeling decisions, not the classifier features directly
