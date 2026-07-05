# Stitch-Level Evaluation in cbench

> _Harness renamed `cbench` → `mbench` (2026-07-05). This is a historical design doc; "cbench" throughout refers to what is now `mbench`._

## Context

The matcher's optimizer resolves M:N match groups using confidence-based greedy assignment, contiguity checks, and connected component detection. The main failure modes observed are:

1. **Group boundaries wrong** - connected components too big or incorrectly split
2. **Alignment mismatches at intersections** - small gaps (~2m) that Hootenanny handles but matcher doesn't

We now have **stitching labels** (`labels/stitching/dataset={name}/data.csv`) that capture the correct edge selection per M:N group. These are curated in the stitching review UI and currently have 13 labeled groups for `us_boston_streets`.

The existing cbench pair-level evaluation (precision/recall/F1 on individual `(ref_id, target_id)` pairs) doesn't measure optimizer quality at the group level. We need a complementary metric.

## Design

### What we're measuring

For each curated stitching label, compare the curated edge selection against what actually appears in the bridge file. This tells us whether the optimizer makes the right assignment decisions within M:N groups.

### Metrics

**Per labeled group:**
- **Edge recall**: What fraction of curated edges appear in the bridge?
  - `|curated_edges & bridge_edges| / |curated_edges|`
- **Edge precision**: Of bridge edges involving the group's segment IDs, how many are correct?
  - `|curated_edges & bridge_edges| / |bridge_edges_for_group_ids|`
- **Extra edges**: Count of bridge edges involving group IDs that weren't curated

**Aggregate (across all labeled groups):**
- Macro-average precision, recall, F1 across groups
- Total groups evaluated
- Per-group breakdown in detailed output

### Label format (existing)

Stitching labels at `labels/stitching/dataset={name}/data.csv`:

| Column | Type | Description |
|--------|------|-------------|
| group_id | string | 8-char hex hash of ref_ids + target_ids |
| dataset_id | string | Target dataset identifier |
| selected_edges | JSON string | Array of `{ref_id, target_id}` pairs |
| match_type | string | "1:N", "N:1", "M:N" |
| num_refs | int | Count of unique ref_ids |
| num_targets | int | Count of unique target_ids |
| labeler | string | Curator name |
| labeled_at | timestamp | ISO timestamp |
| session_id | string | Session identifier |

### Implementation

**New file: `cbench/src/cbench/eval/stitch_metrics.py`**

```python
def evaluate_stitch_groups(
    bridge: pd.DataFrame,
    stitch_labels: pd.DataFrame,
) -> StitchEvalResult:
    """Compare bridge output against curated stitching labels.

    For each labeled group, finds bridge edges involving the group's
    segment IDs and computes edge precision/recall.
    """
```

- Input: bridge DataFrame (ref_id, target_id columns) + stitch labels DataFrame
- Output: `StitchEvalResult` with per-group and aggregate metrics

**Modified: `cbench/src/cbench/eval/labels.py`**

Add `load_stitch_labels(labels_dir, dataset)` to load stitching labels.

**Modified: `cbench/src/cbench/runner.py`**

After existing pair-level evaluation, check if stitching labels exist for the dataset. If so, run `evaluate_stitch_groups()` and include stitch metrics in the result.

**Modified: `cbench/src/cbench/results/store.py`**

Extend `BenchmarkResult` to optionally include stitch metrics.

### Output

When stitching labels exist, cbench reports both levels:

```
Pair-level:   P=81.1%  R=98.9%  F1=89.1%  (639 labeled pairs)
Stitch-level: P=XX.X%  R=XX.X%  F1=XX.X%  (13 groups evaluated)
```

### Scope boundaries

- **In scope**: Group edge precision/recall against stitching labels
- **Out of scope (for now)**:
  - Alignment fraction accuracy (stitching labels don't capture fractions yet)
  - Group boundary Jaccard (internal matcher diagnostic, not generic)
  - Automatic label generation or conversion
  - ML model for group resolution (needs 100+ labeled groups)

### Future work

1. **Grow stitching labels**: Label groups across multiple datasets to get meaningful metrics
2. **Add alignment fraction editing**: Extend stitching UI to capture start/end fractions, then add alignment fraction error metric
3. **Fix optimizer issues**: Use stitch metrics to guide and validate optimizer improvements (group boundaries, intersection gap handling)
4. **ML group resolution**: Once 100+ labeled groups exist, consider a learned model for edge selection within M:N groups

### Verification

1. Run `cbench run --dataset us_boston_streets` and verify stitch metrics appear alongside pair metrics
2. Manually check a few groups: compare reported precision/recall against visual inspection in stitching UI
3. Run existing cbench tests to ensure no regressions
