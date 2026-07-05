"""Stitch-level quality gate for cbench.

Turns the (already-computed, non-blocking) stitch-level metric into an enforced
gate that fails a benchmark run when the optimizer's group edge selection
regresses below per-dataset floors.

Why this lives at *benchmark* time and not in unit CI: the gate needs the live
pipeline outputs (a bridge parquet + its ``*_groups.json`` sidecar) and the
curated stitching labels, none of which exist in GitHub Actions (``data/raw`` /
``data/output`` are untracked and large). So the gate is enforced by
``cbench run --gate`` / ``cbench run-batch --gate`` in the documented pre-merge
checklist for matching-logic PRs. The gate *machinery itself* (arming, floor
comparison, sliver filtering) is regression-tested in CI against a committed
miniature fixture (``cbench/tests/test_gate.py``) so the logic cannot silently
rot.

Auto-arming: a dataset's floor is only enforced once enough stitching labels map
to current groups (``min_mapped_groups``). Below that the gate reports
``skip_unarmed`` (non-blocking) — so as the label base grows past the arming
threshold the gate engages on the next benchmark run with no code change or
second PR. Floors are set baseline-minus-margin, mirroring the LOO-CV CI gate
(``tests/regression/test_loo_cv.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from cbench.eval.stitch_metrics import StitchEvalResult


@dataclass(frozen=True)
class GateFloors:
    """Per-dataset stitch gate configuration.

    All comparisons use the SLIVER-FILTERED metrics so a junction-artifact edge
    on either side cannot swing pass/fail.
    """

    min_mapped_groups: int
    f1_filtered_floor: float
    exact_filtered_floor: float


# Statuses. ``fail`` is the only blocking outcome.
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP_UNARMED = "skip_unarmed"
STATUS_NO_CONFIG = "no_config"


@dataclass
class GateOutcome:
    """Result of applying a floor to one dataset's stitch metrics."""

    dataset: str
    status: str
    message: str
    # Observed values (0.0 when not evaluated), for reporting.
    groups_evaluated: int = 0
    f1_filtered: float = 0.0
    exact_filtered: float = 0.0

    @property
    def blocking(self) -> bool:
        return self.status == STATUS_FAIL


def load_gate_config(config: dict) -> dict[str, GateFloors]:
    """Parse ``[gate.<dataset>]`` sections from a loaded ``datasets.toml`` dict.

    Returns ``{dataset: GateFloors}``. Missing/partial sections are skipped with
    no error (a dataset with no gate block is simply ungated).
    """
    raw = config.get("gate", {})
    floors: dict[str, GateFloors] = {}
    for dataset, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        try:
            floors[dataset] = GateFloors(
                min_mapped_groups=int(cfg["min_mapped_groups"]),
                f1_filtered_floor=float(cfg["f1_filtered_floor"]),
                exact_filtered_floor=float(cfg["exact_filtered_floor"]),
            )
        except (KeyError, TypeError, ValueError):
            # A malformed gate block should not silently pass — but parsing lives
            # in the CLI, which surfaces the skipped dataset. Skip here.
            continue
    return floors


def evaluate_gate(
    dataset: str,
    result: StitchEvalResult | None,
    floors: GateFloors | None,
) -> GateOutcome:
    """Apply a dataset's floor to its stitch metrics.

    - No floor configured -> ``no_config`` (non-blocking).
    - Fewer than ``min_mapped_groups`` groups mapped -> ``skip_unarmed``
      (non-blocking): not enough curated ground truth mapped to current groups
      to gate on yet. This is the auto-arming mechanism.
    - Otherwise compare sliver-filtered F1 and exact-match against the floors;
      ``pass`` iff BOTH hold, else ``fail`` (blocking).
    """
    if floors is None:
        return GateOutcome(dataset, STATUS_NO_CONFIG, "no gate floor configured")

    if result is None:
        return GateOutcome(
            dataset,
            STATUS_SKIP_UNARMED,
            "no stitch metrics available (no labels / no groups sidecar)",
        )

    n = result.groups_evaluated
    if n < floors.min_mapped_groups:
        return GateOutcome(
            dataset,
            STATUS_SKIP_UNARMED,
            f"unarmed: {n} groups mapped < {floors.min_mapped_groups} required",
            groups_evaluated=n,
            f1_filtered=result.f1_filtered,
            exact_filtered=result.exact_match_rate_filtered,
        )

    f1 = result.f1_filtered
    exact = result.exact_match_rate_filtered
    f1_ok = f1 >= floors.f1_filtered_floor
    exact_ok = exact >= floors.exact_filtered_floor

    if f1_ok and exact_ok:
        return GateOutcome(
            dataset,
            STATUS_PASS,
            (
                f"F1={f1:.4f}>={floors.f1_filtered_floor} "
                f"exact={exact:.4f}>={floors.exact_filtered_floor} ({n} groups)"
            ),
            groups_evaluated=n,
            f1_filtered=f1,
            exact_filtered=exact,
        )

    reasons = []
    if not f1_ok:
        reasons.append(f"filtered F1 {f1:.4f} < floor {floors.f1_filtered_floor}")
    if not exact_ok:
        reasons.append(f"filtered exact {exact:.4f} < floor {floors.exact_filtered_floor}")
    return GateOutcome(
        dataset,
        STATUS_FAIL,
        "; ".join(reasons) + f" ({n} groups)",
        groups_evaluated=n,
        f1_filtered=f1,
        exact_filtered=exact,
    )
