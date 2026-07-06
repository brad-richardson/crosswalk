# Stitching edge-case observations (running log)

Human-reported edge cases from `/stitching-review` sessions that aren't yet
covered by a fix, a rubric rule, or a test. When one gets addressed, move it
to a "resolved" section with the PR number instead of deleting it.

## Open

### 2026-07-06 — intersection-crossing connector dropped (false negative)

- **Dataset:** `us_boston_streets`, batch group `60475763` (menu-k8 batch of
  2026-07-05 16:06, shown as "Group 8 of 100"), M:N 7 ref × 9 target, group
  confidence 99%.
- **Location:** Meridian Street approaching Porter Street, East Boston.
- **Observation (Brad, mobile review session):** in the de-anchored view, the
  connector edge carrying the corridor *through* the Porter St intersection
  renders as excluded (dashed), i.e. the optimizer treats the ref/target pair
  through the junction as no-match — but the corridor visibly continues
  straight through and the edge should be included.
- **Why it matters:** this is the "false negative through an intersection"
  shape — a group that is otherwise a clean parallel corridor gets a gap
  exactly at a junction, which splits coverage and can push the group into
  review/regrouping. Likely interacting suspects: junction-area alignment
  fractions dropping below the glue prune on the short through-junction
  segment, and/or endpoint-proximity features degrading where cross-streets
  join (cf. #257 endpoint de-degeneracy).
- **Ground truth:** Brad's curated selection for this group in
  `labels/stitching/dataset=us_boston_streets/data.csv` is the authoritative
  edge set; any fix should be validated against it (and the stitch-level gate).
- **Status:** noted only — no fix attempted. Candidate for the next
  optimizer/glue-prune investigation batch; check whether other junction-gap
  false negatives exist in the same batch before designing a fix.
