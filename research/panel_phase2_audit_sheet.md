# Phase-2 Panel Audit Sheet — us_boston_streets

Manually audit these 10 randomly-sampled (seed 20260703) auto-accept CANDIDATE groups before enabling unanimous auto-accept export. Each was a unanimous non-NONE panel pick on a group of ≤20 candidate edges.

**To view a group:** run `uv run matcher ui`, then open the URL below (port 8505). The stitching-review UI pre-seeds the panel's chosen option; confirm the highlighted edge set is the correct M:N assignment.

Evidence packs (overview + per-option images the panel saw) are under `data/agents/stitching/batches/us_boston_streets_phase2/<group_id>/`.

Auto-accept candidates total: **30**. Sampled: 10.


## 1. Group `04fc93e5`  (choice B, 8 candidate edges in group)

- **Panel choice:** option B, mean confidence 0.76
- **Chosen edge set (5 edges):** 1fd2f686-e..→..790_882a306633, 68161cd5-c..→..082_882a306633, 8cf1ac96-a..→..068_882a306633, 92569b8b-7..→..790_882a306633, b68b5ac8-9..→..790_882a306633
- **Per-provider votes:**
    - `claude` → **B** (conf 0.6): Options C, D, E include a near-zero-alignment edge (R2->T3 conf=0.305, ref_aln=0, tgt_aln=0.001) or a spurious R4->T2 edge (conf=0.202), which are clearly not real matches, so they're worse than A/B. Between A and B, the only difference is R2->T2 (ref_aln=0.99, tgt_aln=0.184) vs R1->T2 (ref_aln=1.0, tgt_aln=0.571). R1 …
    - `codex` → **B** (conf 0.72): T1 clearly corresponds to R4 and T3 clearly corresponds to R3; the options sending T3 to R2 are geometrically wrong because R2 is the diagonal Rutland continuation, not the L-shaped Haven Street segment. For T2, the strongest same-road correspondence is with the long central diagonal Rutland segment R1, plus the short …
    - `agy` → **B** (conf 0.95): Option B is the best match. It matches R1, R5, and R6 to T2, covering the majority of the length of Rutland Street (T2), including the long R1 segment (tgt_aln=0.571). In contrast, Option A leaves R1 unmatched and instead matches R2 (tgt_aln=0.184) at the very end of the segment, leaving a large portion of the …
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=04fc93e5
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/04fc93e5/`

## 2. Group `16947985`  (choice B, 3 candidate edges in group)

- **Panel choice:** option B, mean confidence 0.92
- **Chosen edge set (2 edges):** 2f3dc626-4..→..009_882a306615, bdb507dd-0..→..263_882a306615
- **Per-provider votes:**
    - `claude` → **B** (conf 0.85): The overview shows a single continuous road split into a left half (R1/T1) and right half (R2/T2), with reference and target closely overlapping along the whole length. The correct correspondence is positional: R1↔T1 (left) and R2↔T2 (right). Option B contains exactly these two edges. Option A adds a spurious …
    - `codex` → **B** (conf 0.96): The blue and red geometries align as two adjacent segments along the same physical road: the left reference segment R1 overlaps the left target segment T1, and the right reference segment R2 overlaps the right target segment T2. The extra edge R1->T2 in options A, C, and E incorrectly links non-corresponding adjacent …
    - `agy` → **B** (conf 0.95): R1 aligns well with T1 (high alignment parameters) and R2 aligns well with T2. The connection R1->T2 has extremely low alignment (ref_aln=0.12, tgt_aln=0.084) and low confidence, representing a spurious overlap where the consecutive segments touch at a junction rather than sharing the same physical path. Therefore …
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=16947985
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/16947985/`

## 3. Group `63bf7e48`  (choice A, 2 candidate edges in group)

- **Panel choice:** option A, mean confidence 0.97
- **Chosen edge set (2 edges):** 192618d9-2..→..654_882a33936d, 60001ddb-a..→..654_882a33936d
- **Per-provider votes:**
    - `claude` → **A** (conf 0.93): R1 and R2 are two reference segments that together span the same physical road as the single target segment T1 (their tgt_aln values 0.478+0.497≈0.975 sum to nearly full coverage of T1, and both have ref_aln≈1.0 meaning each ref segment is fully covered by T1). The overview and option images show a continuous blue …
    - `codex` → **A** (conf 0.97): The blue reference geometry appears to be two contiguous segments of the same road (R1 and R2) that together follow the same continuous red target road T1. Option A best captures that N:1 relationship. Options B and C each omit one of the two blue segments even though both appear to lie on the same physical traveled …
    - `agy` → **A** (conf 1.0): R1 and R2 are two consecutive reference segments representing the same physical road ('Hill Top Street'), and T1 is a single target segment covering the extent of both R1 and R2. Therefore, both R1->T1 and R2->T1 are correct correspondences, representing Option A.
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=63bf7e48
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/63bf7e48/`

## 4. Group `701d491e`  (choice A, 16 candidate edges in group)

- **Panel choice:** option A, mean confidence 0.88
- **Chosen edge set (7 edges):** 01fa8038-4..→..852_882a3064c7, 0b9be5a8-0..→..971_882a3064c7, 59b73d04-a..→..888_882a3064c7, 726ce920-9..→..833_882a3064c7, 8ec35d3b-e..→..727_882a3064c7, 9203993e-d..→..722_882a3064c7, b6b8a25b-f..→..653_882a3064c7
- **Per-provider votes:**
    - `claude` → **A** (conf 0.75): Option A is the only option and its bright edges show the red target segments closely tracking the blue reference segments along both Walk Hill Street and American Legion Highway, with matching names/classes for each edge (R1-R3,R2 -> Walk Hill Street targets; R4,R6,R7,R8 -> American Legion Highway targets). All conf …
    - `codex` → **A** (conf 0.92): The highlighted matches in Option A consistently pair the blue and red segments that lie on the same two physical roads: the Walk Hill Street segments along the upper corridor and the American Legion Highway segments along the diagonal branch. The geometries largely overlap end-to-end with only small offsets, and the …
    - `agy` → **A** (conf 0.98): Option A correctly associates reference and target segments along Walk Hill Street (R1->T7, R2->T1, R3->T3) and along American Legion Highway (R4->T2, R6->T5, R7->T6, R8->T4) based on name matches and geometry alignments, without linking perpendicular crossings or opposite carriageways.
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=701d491e
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/701d491e/`

## 5. Group `72063362`  (choice A, 2 candidate edges in group)

- **Panel choice:** option A, mean confidence 0.98
- **Chosen edge set (2 edges):** a374826e-6..→..432_882a3066b9, f192a418-9..→..432_882a3066b9
- **Per-provider votes:**
    - `claude` → **A** (conf 0.95): Overview shows R1 and R2 are two contiguous reference segments (both named Everett Street) that together span the full length of T1, with matching alignment (ref_aln=1.0 for both). Option A includes both R1->T1 and R2->T1 edges, correctly capturing the N:1 relationship since neither R1 nor R2 alone covers the whole …
    - `codex` → **A** (conf 0.98): In the overview, T1 is one continuous red segment along Everett Street, while the blue reference geometry is split into two contiguous segments, R1 and R2, on the same physical road. Option A correctly matches both reference segments to the single target segment in this N:1 case. Options B and C each omit one portion …
    - `agy` → **A** (conf 1.0): Target segment T1 corresponds to the same physical road as the reference segments R1 and R2 combined. R1 and R2 are consecutive segments of Everett Street, while T1 is a single segment representing the same stretch of Everett Street. Mapping both (Option A) provides complete coverage.
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=72063362
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/72063362/`

## 6. Group `79711407`  (choice A, 7 candidate edges in group)

- **Panel choice:** option A, mean confidence 0.96
- **Chosen edge set (4 edges):** 362e5532-e..→..658_882a3064cb, 695c4324-5..→..653_882a3064cb, bc40b316-3..→..164_882a3064cb, f99d5678-f..→..150_882a3064cb
- **Per-provider votes:**
    - `claude` → **A** (conf 0.9): The overview shows a single continuous straight road (Talbot Avenue) with all 4 reference and 4 target segments overlapping along it. Option A's highlighted edges (R3-T1, R4-T2, R1-T3, R2-T4) all show bright/solid overlapping segments following the same physical road line, matching the reference and target geometry …
    - `codex` → **A** (conf 0.97): All four highlighted reference and target segments lie on the same continuous Talbot Avenue centerline and the segmentation order aligns cleanly from west to east with only minor offset. No highlighted edge appears to jump to a parallel or crossing road, so Option A is a good representation of the true correspondences.
    - `agy` → **A** (conf 1.0): The reference segments (R1-R4) and target segments (T1-T4) are collinear segments representing the same physical road (Talbot Avenue). Option A correctly maps the overlapping segments: R3 to T1, R4 to T2, R1 to T3, and R2 to T4 along this traveled way.
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=79711407
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/79711407/`

## 7. Group `9ac35fb7`  (choice E, 8 candidate edges in group)

- **Panel choice:** option E, mean confidence 0.82
- **Chosen edge set (6 edges):** 1b670d46-c..→.._34_882a30644b, 1b670d46-c..→..185_882a30644b, 25489c34-6..→..050_882a30644b, 25eec939-5..→.._34_882a30644b, b3c8bbdc-0..→..185_882a30644b, b88d0172-4..→..305_882a306637
- **Per-provider votes:**
    - `claude` → **E** (conf 0.62): This group is a single continuous straight corridor (Boylston Street) where 5 reference sub-segments and 4 target sub-segments break at slightly different points, so visually all options look nearly identical (all points lie on the same line). The real differentiator is the edge-level alignment stats. Five edges are …
    - `codex` → **E** (conf 0.88): The group is one continuous Boylston Street corridor, and Option E keeps the strong along-corridor overlaps while excluding the clearly weak cross-segment assignments. In particular, R1->T2, R5->T2, and R4->T4 have near-zero alignment on at least one side and look like boundary-touching or non-overlapping matches …
    - `agy` → **E** (conf 0.95): Comparing the options, we look at the segment geometry and matching segments along Boylston Street. The physical layout is a single continuous road. The segments R1-R5 are blue reference segments from east to west (or west to east, let's trace: R5, T1, R2, T2, R1, T3, R4, T4). Specifically, R5 is on the far right …
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=9ac35fb7
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/9ac35fb7/`

## 8. Group `b2876328`  (choice A, 2 candidate edges in group)

- **Panel choice:** option A, mean confidence 0.95
- **Chosen edge set (2 edges):** 2b5f9868-a..→..806_882a3074b7, a84ee23d-b..→..806_882a3074b7
- **Per-provider votes:**
    - `claude` → **A** (conf 0.9): Overview shows R1 covering the upper/left portion and R2 covering the lower/right portion of the same physical road, while T1 spans the entire length. This is a genuine N:1 case where both reference segments correspond to the single target segment. Option A includes both R1->T1 and R2->T1 edges, correctly representing …
    - `codex` → **A** (conf 0.96): In the overview, both reference segments R1 and R2 lie on the same diagonal physical road as the single target segment T1. They are contiguous pieces of the same roadway and both overlap/follow T1’s alignment, so the best representation is the 2-to-1 assignment including both R1->T1 and R2->T1.
    - `agy` → **A** (conf 1.0): R1 and R2 represent the two consecutive reference segments that cover the full length of the single target segment T1 along the same physical road (Day Square). Therefore, Option A, which includes both matches (R1->T1 and R2->T1), is the correct selection.
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=b2876328
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/b2876328/`

## 9. Group `f170979a`  (choice C, 12 candidate edges in group)

- **Panel choice:** option C, mean confidence 0.78
- **Chosen edge set (6 edges):** 0528cc12-4..→..373_882a306419, 27300bc3-b..→..839_882a306419, 991484dd-a..→..828_882a306419, cd06bc62-c..→..373_882a306419, dec3bb15-7..→..839_882a306419, f69479c7-2..→..839_882a306419
- **Per-provider votes:**
    - `claude` → **C** (conf 0.68): All 6 residential ref segments (R1,R3,R4,R6,R7,R9) need assignment; footways R2,R5,R8 correctly excluded everywhere. T2's tgt_aln values for R3/R6/R7/R9 (0.337/0.087/0.344/0.231) sum to ~1.0, meaning T2 is fully composed of these four ref fragments — but no option includes all four, so the best proxy is whichever trio …
    - `codex` → **C** (conf 0.72): The true matches appear to be along the main diagonal Bragdon Street alignment, plus the leftmost overlap at T1 and the rightmost overlap at T3. Option C keeps the plausible diagonal correspondences (R1->T1, R3->T2, R7->T2, R9->T2, R4->T3) and avoids the clearly implausible cross-group edge R1->T3. Between the R3->T2 …
    - `agy` → **C** (conf 0.95): Option C is the most accurate representation of the same-physical-road correspondences. It correctly matches R3 to T2, which shows clear geometrical overlap at the left end of the street segment. It also correctly excludes the spurious low-confidence edge R1->T3, which represents two far-apart segments with near-zero …
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=f170979a
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/f170979a/`

## 10. Group `f69a827e`  (choice A, 2 candidate edges in group)

- **Panel choice:** option A, mean confidence 0.96
- **Chosen edge set (2 edges):** 6fc832cc-1..→..483_882a3066b5, f94a0f4c-9..→..483_882a3066b5
- **Per-provider votes:**
    - `claude` → **A** (conf 0.93): R1 and R2 are two consecutive reference segments of Chamberlain Street that together span the full length of target T1, overlapping the same physical road end-to-end. Option A includes both R1->T1 and R2->T1, correctly covering the whole matched length, whereas B and C only capture half the road each.
    - `codex` → **A** (conf 0.95): In the overview, R2 and R1 are contiguous blue segments along the same physical Chamberlain Street, and T1 is a single red segment that follows that same continuous road across both reference pieces. Option A is the only option that captures the full same-road correspondence, with T1 overlapping both R2 and R1 rather …
    - `agy` → **A** (conf 1.0): T1 is a single target segment that spans the entire length of both reference segments R1 and R2 along Chamberlain Street. Matching both R1 and R2 to T1 (Option A) represents the true same-physical-road correspondence.
- **View in UI:** http://localhost:8505/stitching-review/group?dataset=us_boston_streets&group_id=f69a827e
- **Evidence pack:** `data/agents/stitching/batches/us_boston_streets_phase2/f69a827e/`
