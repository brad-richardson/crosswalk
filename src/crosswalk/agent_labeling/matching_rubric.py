"""Versioned matching rubrics shared by agent-labeling workflows.

The normative source is ``docs/MATCHING_MERGING_RULES.md``.  The text below is
copied verbatim from that document's marked agent-rubric blocks because the
installed Python package cannot rely on repository docs being present at
runtime.  A unit test compares every block byte-for-byte (after stripping the
outer newline), so documentation and executable prompts fail CI if they drift.
"""

from __future__ import annotations

MATCHING_RUBRIC_VERSION = "2026-07-17+e64703cf01b4"
CANONICAL_RULES_DOC = "docs/MATCHING_MERGING_RULES.md"

MATCH_IDENTITY_RUBRIC = """CANONICAL MATCH-IDENTITY RUBRIC (version 2026-07-17+e64703cf01b4)
Apply these rules to each candidate pair before considering group-level conflicts:
MI-1. Identity and role: a match requires the aligned portions to represent the same physical traveled way with the same network role. ALONG matches ALONG on the same facility; ACROSS matches only the same dedicated transverse crossing; TURN/CONNECTOR matches only the same facility or hierarchy transition. A regular turn through an ordinary intersection remains ALONG, as does a through facility that merely passes over or under another facility. Treat role compatibility as the default identity gate; document any real-world exception explicitly in this canonical contract rather than inferring one from geometry alone.
MI-2. Representation differences: segmentation points, segment lengths, names, and class tags may differ. One abstract centerline may match each constituent split carriageway as M:N, but the physically separate opposite carriageways do not match each other. Do not force a 1:1 mapping.
MI-3. Intersections and short overlaps: length alone never decides identity. A short same-direction, same-role subline on the same traveled way is a match and may become a junction anchor. A mere endpoint touch, perpendicular crossing, or different-role overlap is not a match.
MI-4. Parallel features: laterally separate carriageways, frontage/service roads, sidewalks, separated cycle tracks, and other neighboring facilities are not matches merely because they are parallel or share a name. A painted, sharrow, or flexpost-separated bike lane on the same pavement matches the road rather than a separately mapped cycleway; a raised or curbed cycle track is a separate feature and matches the corresponding cycleway rather than the road. When the available evidence does not resolve whether such a bike or pedestrian facility is same-pavement or physically separated — because the pack lacks the physical, lateral-offset or coincidence, layer, or close-up evidence that would decide it — do not default to same-pavement identity: treat identity as unresolved (unsure per PL-3 at the pair level, or insufficient-evidence NONE per SA-5 at the stitch level) unless other evidence such as naming continuity, endpoint topology, or coverage partition independently establishes the exact set.
MI-5. Evidence, not verdicts: geometry, direction, topology, names, classes, model scores, overlap lengths, and SLIVER/BORDERLINE tags are evidence. No single tag, score, name, class mismatch, or overlap threshold overrides physical identity and network role. Small offsets from GPS or digitization are acceptable when the paths represent the same way.
MI-6. Replacement test: if replacing one aligned subline with the other would change movement intent (for example ALONG becomes ACROSS or a mainline becomes a ramp), the pair is not a match."""

PAIR_LABEL_RUBRIC = """PAIR-LABEL OUTPUT CONTRACT (version 2026-07-17+e64703cf01b4)
PL-1. match: the pair satisfies the canonical match-identity rubric.
PL-2. no_match: the pair represents different physical features or roles, or has no plausible aligned subline after ordinary data noise.
PL-3. unsure: the available evidence cannot determine physical identity or network role reliably. Use uncertainty instead of guessing.
PL-4. Pair labeling is recall-biased: retain plausible same-role identity edges for graph-level resolution rather than rejecting them solely because they are short or participate in another candidate match."""

STITCH_ASSIGNMENT_RUBRIC = """STITCH-ASSIGNMENT OUTPUT CONTRACT (version 2026-07-17+e64703cf01b4)
SA-1. Judge identity first: apply the canonical match-identity rubric independently to every displayed candidate edge. Graph resolution may decide which identity-compatible edges coexist; it must not redefine different features or roles as matches.
SA-2. Preserve legitimate M:N structure: keep all mutually consistent identity edges created by different segmentation or centerline/carriageway representation. Do not prefer a smaller set merely because it is simpler.
SA-3. Resolve actual conflicts only after role-incompatible and different-feature candidates are removed. Among the remaining mutually exclusive identity matches, consider neighborhood support, corridor continuity, and aligned coverage together. No single signal is a universal ordering: longer overlap must not by itself override stronger structural evidence, and a supported short same-way edge may remain as a junction anchor. When the evidence does not establish an exact final set, choose NONE for human review.
SA-4. Exact option semantics: choose an option only when it contains all and only the final accepted edges in the displayed candidate universe. Do not choose the closest option, trade a false positive against a false negative, maximize edge count, or defer to total/mean confidence.
SA-5. NONE semantics: choose NONE when every displayed edge is a no-match, when no offered option exactly represents the final accepted edge set, or when the evidence is insufficient to determine an exact set. State which reason applies so human review can distinguish them.
SA-6. Option metadata is non-normative: optimizer status, option order, edge count, and aggregate confidence are context only and never make an option correct."""
