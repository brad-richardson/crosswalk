# Stitch-Level Evaluation Implementation Plan

> _Harness renamed `cbench` → `mbench` (2026-07-05). This is a historical implementation plan; "cbench" throughout refers to what is now `mbench`._

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add group-level edge precision/recall evaluation to cbench using curated stitching labels, complementing existing pair-level F1.

**Architecture:** New `stitch_metrics.py` module in `cbench/src/cbench/eval/` with a `StitchEvalResult` dataclass and `evaluate_stitch_groups()` function. Stitching labels loaded via new `load_stitch_labels()` in `labels.py`. Runner calls stitch eval when labels exist; CLI prints both pair-level and stitch-level metrics.

**Tech Stack:** pandas, dataclasses, JSON parsing. Tests with pytest.

---

### Task 1: StitchEvalResult dataclass and evaluate_stitch_groups()

**Files:**
- Create: `cbench/src/cbench/eval/stitch_metrics.py`
- Test: `cbench/tests/test_stitch_metrics.py`

**Step 1: Write the failing tests**

```python
"""Tests for stitch-level evaluation metrics."""

import json

import pandas as pd
import pytest

from cbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups


@pytest.fixture
def bridge_df():
    """Bridge with edges for two groups."""
    return pd.DataFrame({
        "ref_id": ["r1", "r1", "r2", "r3", "r3", "r4"],
        "target_id": ["t1", "t2", "t1", "t3", "t4", "t5"],
        "confidence": [0.9, 0.8, 0.7, 0.95, 0.6, 0.85],
    })


@pytest.fixture
def stitch_labels():
    """Two labeled groups."""
    return pd.DataFrame({
        "group_id": ["aaa", "bbb"],
        "dataset_id": ["test", "test"],
        "selected_edges": [
            json.dumps([
                {"ref_id": "r1", "target_id": "t1"},
                {"ref_id": "r2", "target_id": "t1"},
            ]),
            json.dumps([
                {"ref_id": "r3", "target_id": "t3"},
            ]),
        ],
        "match_type": ["N:1", "1:1"],
        "num_refs": [2, 1],
        "num_targets": [1, 1],
    })


def test_evaluate_stitch_groups_basic(bridge_df, stitch_labels):
    """Curated edges found in bridge, extra edges detected."""
    result = evaluate_stitch_groups(bridge_df, stitch_labels)

    assert result.groups_evaluated == 2
    # Group aaa: curated {r1-t1, r2-t1}, bridge has {r1-t1, r1-t2, r2-t1}
    #   recall = 2/2 = 1.0, precision = 2/3 = 0.667
    # Group bbb: curated {r3-t3}, bridge has {r3-t3, r3-t4}
    #   recall = 1/1 = 1.0, precision = 1/2 = 0.5
    assert result.recall == pytest.approx(1.0)  # macro avg
    assert result.precision == pytest.approx((2 / 3 + 0.5) / 2, abs=0.01)
    assert result.total_curated_edges == 3
    assert result.total_extra_edges == 2  # r1-t2 and r3-t4


def test_evaluate_stitch_groups_missing_edges(bridge_df, stitch_labels):
    """When bridge is missing curated edges, recall drops."""
    # Remove r2-t1 from bridge
    bridge_missing = bridge_df[~((bridge_df["ref_id"] == "r2") & (bridge_df["target_id"] == "t1"))]
    result = evaluate_stitch_groups(bridge_missing, stitch_labels)

    # Group aaa: curated {r1-t1, r2-t1}, bridge has {r1-t1, r1-t2}
    #   recall = 1/2 = 0.5, precision = 1/2 = 0.5
    assert result.recall == pytest.approx(0.75)  # macro avg (0.5 + 1.0) / 2


def test_evaluate_stitch_groups_empty_labels():
    """No stitch labels -> zero metrics."""
    bridge = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(columns=["group_id", "dataset_id", "selected_edges",
                                    "match_type", "num_refs", "num_targets"])
    result = evaluate_stitch_groups(bridge, labels)
    assert result.groups_evaluated == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_stitch_eval_result_to_dict():
    r = StitchEvalResult(
        groups_evaluated=5,
        precision=0.8,
        recall=0.9,
        f1=0.847,
        total_curated_edges=10,
        total_extra_edges=2,
    )
    d = r.to_dict()
    assert d["groups_evaluated"] == 5
    assert d["stitch_f1"] == 0.847
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/brad/dev/matcher/cbench && uv run pytest tests/test_stitch_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbench.eval.stitch_metrics'`

**Step 3: Write minimal implementation**

```python
"""Stitch-level evaluation metrics.

Compares bridge output against curated stitching labels to measure
whether the optimizer selects the correct edges within M:N groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd


@dataclass
class StitchEvalResult:
    """Metrics from evaluating bridge edges against stitching labels."""

    groups_evaluated: int
    precision: float
    recall: float
    f1: float
    total_curated_edges: int
    total_extra_edges: int

    def to_dict(self) -> dict:
        return {
            "groups_evaluated": self.groups_evaluated,
            "stitch_precision": self.precision,
            "stitch_recall": self.recall,
            "stitch_f1": self.f1,
            "total_curated_edges": self.total_curated_edges,
            "total_extra_edges": self.total_extra_edges,
        }


def evaluate_stitch_groups(
    bridge: pd.DataFrame,
    stitch_labels: pd.DataFrame,
) -> StitchEvalResult:
    """Compare bridge output against curated stitching labels.

    For each labeled group:
    1. Extract curated edges from selected_edges JSON
    2. Find all bridge edges involving the group's segment IDs
    3. Compute edge precision and recall

    Args:
        bridge: DataFrame with columns [ref_id, target_id, confidence].
        stitch_labels: DataFrame with columns [group_id, selected_edges, ...].

    Returns:
        StitchEvalResult with aggregate and per-group metrics.
    """
    if stitch_labels.empty:
        return StitchEvalResult(
            groups_evaluated=0, precision=0.0, recall=0.0, f1=0.0,
            total_curated_edges=0, total_extra_edges=0,
        )

    bridge_edges = set(
        zip(bridge["ref_id"].astype(str), bridge["target_id"].astype(str))
    )

    precisions = []
    recalls = []
    total_curated = 0
    total_extra = 0

    for _, row in stitch_labels.iterrows():
        selected = json.loads(row["selected_edges"])
        curated = {(str(e["ref_id"]), str(e["target_id"])) for e in selected}

        if not curated:
            continue

        # Collect all segment IDs in this group
        group_ref_ids = {r for r, _ in curated}
        group_target_ids = {t for _, t in curated}

        # Find bridge edges involving any of these IDs
        bridge_for_group = {
            (r, t) for r, t in bridge_edges
            if r in group_ref_ids or t in group_target_ids
        }

        found = curated & bridge_for_group
        extra = bridge_for_group - curated

        recall = len(found) / len(curated) if curated else 0.0
        precision = len(found) / len(bridge_for_group) if bridge_for_group else 0.0

        recalls.append(recall)
        precisions.append(precision)
        total_curated += len(curated)
        total_extra += len(extra)

    n = len(precisions)
    if n == 0:
        return StitchEvalResult(
            groups_evaluated=0, precision=0.0, recall=0.0, f1=0.0,
            total_curated_edges=0, total_extra_edges=0,
        )

    avg_p = sum(precisions) / n
    avg_r = sum(recalls) / n
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    return StitchEvalResult(
        groups_evaluated=n,
        precision=avg_p,
        recall=avg_r,
        f1=f1,
        total_curated_edges=total_curated,
        total_extra_edges=total_extra,
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/brad/dev/matcher/cbench && uv run pytest tests/test_stitch_metrics.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add cbench/src/cbench/eval/stitch_metrics.py cbench/tests/test_stitch_metrics.py
git commit -m "feat(cbench): add stitch-level evaluation metrics"
```

---

### Task 2: Load stitching labels

**Files:**
- Modify: `cbench/src/cbench/eval/labels.py`
- Test: `cbench/tests/test_labels.py`

**Step 1: Write the failing test**

Add to `cbench/tests/test_labels.py`:

```python
def test_load_stitch_labels(tmp_path):
    """Load stitching labels from hive-partitioned CSV."""
    import json
    from cbench.eval.labels import load_stitch_labels

    dataset_dir = tmp_path / "dataset=test_city"
    dataset_dir.mkdir()
    labels = pd.DataFrame({
        "group_id": ["abc123"],
        "dataset_id": ["test_city"],
        "selected_edges": [json.dumps([{"ref_id": "r1", "target_id": "t1"}])],
        "match_type": ["1:1"],
        "num_refs": [1],
        "num_targets": [1],
        "labeler": ["test"],
        "labeled_at": ["2026-02-21T00:00:00Z"],
        "session_id": ["s1"],
    })
    labels.to_csv(dataset_dir / "data.csv", index=False)

    result = load_stitch_labels(tmp_path, "test_city")
    assert len(result) == 1
    assert "selected_edges" in result.columns


def test_load_stitch_labels_missing_returns_none(tmp_path):
    """Return None when no stitch labels exist."""
    from cbench.eval.labels import load_stitch_labels
    result = load_stitch_labels(tmp_path, "nonexistent")
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/brad/dev/matcher/cbench && uv run pytest tests/test_labels.py::test_load_stitch_labels -v`
Expected: FAIL with `ImportError: cannot import name 'load_stitch_labels'`

**Step 3: Add load_stitch_labels() to labels.py**

Add to `cbench/src/cbench/eval/labels.py`:

```python
def load_stitch_labels(
    labels_path: Path,
    dataset: str,
) -> pd.DataFrame | None:
    """Load stitching labels for a dataset, if they exist.

    Reads from {labels_path}/dataset={dataset}/data.csv.
    Returns None if no stitching labels exist for this dataset.

    Args:
        labels_path: Root stitching labels directory.
        dataset: Dataset name.

    Returns:
        DataFrame with stitching label columns, or None if not found.
    """
    csv_path = labels_path / f"dataset={dataset}" / "data.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} stitch labels from {csv_path}")

    required = {"group_id", "selected_edges"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"Stitch labels missing required columns: {missing}")
        return None

    return df
```

**Step 4: Run tests**

Run: `cd /home/brad/dev/matcher/cbench && uv run pytest tests/test_labels.py -v`
Expected: All tests PASS (existing + new)

**Step 5: Commit**

```bash
git add cbench/src/cbench/eval/labels.py cbench/tests/test_labels.py
git commit -m "feat(cbench): add load_stitch_labels() for stitching label loading"
```

---

### Task 3: Wire stitch eval into runner and CLI

**Files:**
- Modify: `cbench/src/cbench/runner.py`
- Modify: `cbench/src/cbench/cli.py`
- Modify: `cbench/datasets.toml`

**Step 1: Add stitch_labels_dir to datasets.toml defaults**

Add to `[defaults]` in `cbench/datasets.toml`:

```toml
stitch_labels_dir = "../labels/stitching"
```

**Step 2: Modify runner.py to run stitch eval**

In `cbench/src/cbench/runner.py`, add `stitch_labels_dir` parameter to `run_single()`. After existing pair-level eval, try loading stitch labels and running stitch eval:

```python
# At top, add import:
from cbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups

# Add stitch_labels_dir parameter to run_single() signature
# Add to RunResult dataclass:
stitch_result: StitchEvalResult | None = None

# After line 123 (eval_result = evaluate(...)), add:
stitch_result = None
if stitch_labels_dir is not None:
    from cbench.eval.labels import load_stitch_labels
    stitch_labels = load_stitch_labels(stitch_labels_dir, dataset)
    if stitch_labels is not None:
        stitch_result = evaluate_stitch_groups(tool_output.matches, stitch_labels)
        metrics.update(stitch_result.to_dict())
        logger.info(
            f"Stitch eval: P={stitch_result.precision:.3f} "
            f"R={stitch_result.recall:.3f} F1={stitch_result.f1:.3f} "
            f"({stitch_result.groups_evaluated} groups)"
        )
```

**Step 3: Modify cli.py to pass stitch_labels_dir and print results**

In `_print_eval_result()`, add stitch result printing:

```python
def _print_eval_result(tool: str, dataset: str, result) -> None:
    # ... existing pair-level output ...
    if result.stitch_result is not None:
        sr = result.stitch_result
        console.print(f"  [bold]Stitch-level ({sr.groups_evaluated} groups):[/bold]")
        console.print(f"    Precision: {sr.precision:.4f}")
        console.print(f"    Recall:    {sr.recall:.4f}")
        console.print(f"    F1:        [bold]{sr.f1:.4f}[/bold]")
        console.print(f"    Curated edges: {sr.total_curated_edges}  Extra: {sr.total_extra_edges}")
```

In `run()` and `run_batch()`, resolve `stitch_labels_dir` from config defaults and pass to `run_single()`.

**Step 4: Run existing tests to verify no regressions**

Run: `cd /home/brad/dev/matcher/cbench && uv run pytest tests/ -v`
Expected: All existing tests PASS

**Step 5: Run full pipeline manually to verify end-to-end**

Run: `cd /home/brad/dev/matcher/cbench && uv run cbench run matcher us_boston_streets -l ../labels/human -r ../data/raw/us_boston_streets_overture_segments_v1.0.parquet -t ../data/raw/us_boston_streets_v1.0.parquet`

Expected: Pair-level metrics print as before, plus stitch-level metrics if stitching labels exist.

**Step 6: Commit**

```bash
git add cbench/src/cbench/runner.py cbench/src/cbench/cli.py cbench/datasets.toml
git commit -m "feat(cbench): wire stitch eval into runner and CLI output"
```

---

### Task 4: Format, lint, and final verification

**Step 1: Format and lint**

Run: `cd /home/brad/dev/matcher/cbench && uv run ruff format src/ tests/ && uv run ruff check src/ tests/`

**Step 2: Run all tests**

Run: `cd /home/brad/dev/matcher/cbench && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 3: Run end-to-end on Boston dataset**

Run: `cd /home/brad/dev/matcher/cbench && uv run cbench run matcher us_boston_streets -l ../labels/human -r ../data/raw/us_boston_streets_overture_segments_v1.0.parquet -t ../data/raw/us_boston_streets_v1.0.parquet`

Verify:
- Pair-level metrics display as before
- Stitch-level metrics display with 33 groups evaluated
- No errors or warnings

**Step 4: Commit any final fixes**

```bash
git add -A && git commit -m "chore: format and lint stitch eval"
```
