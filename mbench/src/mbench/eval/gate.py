"""Stitch-level quality gate for mbench.

Turns the (already-computed, non-blocking) stitch-level metric into an enforced
gate that fails a benchmark run when the optimizer's group edge selection
regresses below per-dataset floors.

Why this lives at *benchmark* time and not in unit CI: the gate needs the live
pipeline outputs (a bridge parquet + its ``*_groups.json`` sidecar) and the
curated stitching labels, none of which exist in GitHub Actions (``data/raw`` /
``data/output`` are untracked and large). So the gate is enforced by
``mbench run --gate`` / ``mbench run-batch --gate`` in the documented pre-merge
checklist for matching-logic PRs. The gate *machinery itself* (arming, floor
comparison, sliver filtering) is regression-tested in CI against a committed
miniature fixture (``mbench/tests/test_gate.py``) so the logic cannot silently
rot.

Auto-arming: a dataset's floor is only enforced once enough stitching labels map
to current groups (``min_mapped_groups``). Below that the gate reports
``skip_unarmed`` (non-blocking) — so as the label base grows past the arming
threshold the gate engages on the next benchmark run with no code change or
second PR. Once established, ``armed = true`` persists that state so mapping
loss cannot disable the gate; an optional ``min_mapping_rate`` also catches
large retention regressions above the absolute auto-arm count, while
``min_labels_total`` protects the curated population itself (pair + set labels,
so pair->set semantic conversions do not read as deletions). Floors are set
baseline-minus-margin, mirroring the LOO-CV CI gate
(``tests/regression/test_loo_cv.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from mbench.eval.stitch_metrics import StitchEvalResult


@dataclass(frozen=True)
class GateFloors:
    """Per-dataset stitch gate configuration.

    All comparisons use the SLIVER-FILTERED metrics so a junction-artifact edge
    on either side cannot swing pass/fail.
    """

    min_mapped_groups: int
    f1_filtered_floor: float
    exact_filtered_floor: float
    # Persisted once a dataset has established a baseline above the mapping
    # threshold. Prevents a regrouping regression from silently self-unarming.
    armed: bool = False
    # Optional retention floor for an already-armed label population. Unlike
    # min_mapped_groups, this detects a large regression that still leaves
    # enough easy survivors to clear the auto-arm count.
    min_mapping_rate: float | None = None
    # Optional persisted floor on the TOTAL curated label population (pair +
    # set semantics, so pair->set conversions don't read as deletions).
    # Protects against deleting labels while retaining a perfect mapping rate.
    min_labels_total: int | None = None


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


class GateConfigError(ValueError):
    """Raised when a requested quality gate cannot be enforced safely."""


_KNOWN_GATE_KEYS = frozenset(
    {
        "min_mapped_groups",
        "f1_filtered_floor",
        "exact_filtered_floor",
        "armed",
        "min_mapping_rate",
        "min_labels_total",
    }
)


def _gate_int(cfg: dict, key: str, *, optional: bool = False) -> int | None:
    """Read a TOML integer, rejecting the silent bool/float coercions of int().

    ``int(True) == 1`` and ``int(102.9) == 102`` both clear a ``>= 1`` range
    check, so a typo'd value would quietly weaken a floor instead of erroring.
    """
    value = cfg.get(key)
    if value is None:
        if optional:
            return None
        raise KeyError(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer (got {value!r})")
    return value


def _gate_float(cfg: dict, key: str, *, optional: bool = False) -> float | None:
    """Read a TOML number as float, rejecting booleans (bool is an int subclass)."""
    value = cfg.get(key)
    if value is None:
        if optional:
            return None
        raise KeyError(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number (got {value!r})")
    return float(value)


def load_gate_config(config: dict, *, strict: bool = False) -> dict[str, GateFloors]:
    """Parse ``[gate.<dataset>]`` sections from a loaded ``datasets.toml`` dict.

    Returns ``{dataset: GateFloors}``. With ``strict=True``, malformed gate
    configuration raises rather than silently disabling an explicitly requested
    gate. Datasets without a gate block remain ungated.
    """
    raw = config.get("gate", {})
    if not isinstance(raw, dict):
        message = f"[gate] config is not a table (got {type(raw).__name__})"
        if strict:
            raise GateConfigError(message)
        logger.warning(f"{message}; no floors loaded")
        return {}
    floors: dict[str, GateFloors] = {}
    errors: list[str] = []
    for dataset, cfg in raw.items():
        if not isinstance(cfg, dict):
            message = f"[gate.{dataset}] is not a table"
            errors.append(message)
            if not strict:
                logger.warning(f"{message}; skipping (no floor enforced)")
            continue
        try:
            # An unknown key is most likely a typo of a known one ("armd",
            # "min_mapping_ratee") that would silently weaken the gate.
            unknown_keys = sorted(set(cfg) - _KNOWN_GATE_KEYS)
            if unknown_keys:
                raise ValueError(f"unknown keys: {unknown_keys}")
            armed_raw = cfg.get("armed", False)
            if not isinstance(armed_raw, bool):
                raise TypeError("armed must be a boolean")
            floor = GateFloors(
                min_mapped_groups=_gate_int(cfg, "min_mapped_groups"),
                f1_filtered_floor=_gate_float(cfg, "f1_filtered_floor"),
                exact_filtered_floor=_gate_float(cfg, "exact_filtered_floor"),
                armed=armed_raw,
                min_mapping_rate=_gate_float(cfg, "min_mapping_rate", optional=True),
                min_labels_total=_gate_int(cfg, "min_labels_total", optional=True),
            )
            if floor.min_mapped_groups < 1:
                raise ValueError("min_mapped_groups must be >= 1")
            if not 0.0 <= floor.f1_filtered_floor <= 1.0:
                raise ValueError("f1_filtered_floor must be between 0 and 1")
            if not 0.0 <= floor.exact_filtered_floor <= 1.0:
                raise ValueError("exact_filtered_floor must be between 0 and 1")
            if floor.min_mapping_rate is not None and not 0.0 <= floor.min_mapping_rate <= 1.0:
                raise ValueError("min_mapping_rate must be between 0 and 1")
            if floor.min_labels_total is not None and floor.min_labels_total < 1:
                raise ValueError("min_labels_total must be >= 1")
            if not floor.armed and (
                floor.min_mapping_rate is not None or floor.min_labels_total is not None
            ):
                # Retention floors are only enforced on armed gates; accepting
                # them on an auto-arming gate would leave them silently dead.
                raise ValueError(
                    "min_mapping_rate/min_labels_total require armed = true "
                    "(they protect an established baseline)"
                )
            floors[dataset] = floor
        except (KeyError, TypeError, ValueError) as exc:
            # A malformed gate block must not silently disable enforcement — warn
            # loudly so a typo/missing key is visible rather than an accidental pass.
            message = f"[gate.{dataset}] malformed ({exc})"
            errors.append(message)
            if not strict:
                logger.warning(
                    f"{message}; skipping (no floor enforced). "
                    "Required keys: min_mapped_groups, f1_filtered_floor, exact_filtered_floor."
                )
            continue
    if strict and errors:
        raise GateConfigError("; ".join(errors))
    return floors


def evaluate_gate(
    dataset: str,
    result: StitchEvalResult | None,
    floors: GateFloors | None,
) -> GateOutcome:
    """Apply a dataset's floor to its stitch metrics.

    - No floor configured -> ``no_config`` (non-blocking).
    - Floor configured but NO stitch metrics available (``result is None``) ->
      ``fail`` (blocking): the dataset was explicitly configured to be gated but
      could not be evaluated (missing labels dir / missing groups sidecar / a
      swallowed stitch-eval error), so ``--gate`` must NOT silently pass it.
    - Mapping diagnostics unavailable -> ``fail`` for an armed gate. For an
      unarmed configured gate, legacy (no-sidecar) metrics can still FAIL the
      floors once ``min_mapped_groups`` labels scored — but they can never
      PASS: meeting the floors on legacy metrics reports ``skip_unarmed``.
    - An armed curated label population (pair + set) below ``min_labels_total``
      -> ``fail``.
    - Fewer than ``min_mapped_groups`` groups mapped -> ``skip_unarmed`` for a
      new floor, but ``fail`` when its persisted ``armed`` flag is true. This
      preserves auto-arming without allowing established gates to self-disable.
    - Otherwise compare sliver-filtered F1 and exact-match against the floors;
      ``pass`` iff BOTH hold, else ``fail`` (blocking).
    """
    if floors is None:
        return GateOutcome(dataset, STATUS_NO_CONFIG, "no gate floor configured")

    if result is None:
        return GateOutcome(
            dataset,
            STATUS_FAIL,
            "configured to gate but no stitch metrics available "
            "(missing labels/groups sidecar or stitch eval errored)",
        )

    if not result.mapping_diagnostics_available:
        if floors.armed:
            return GateOutcome(
                dataset,
                STATUS_FAIL,
                "armed gate requires sidecar mapping diagnostics, but they are "
                "unavailable (missing, malformed, or empty groups sidecar)",
                groups_evaluated=result.groups_evaluated,
                f1_filtered=result.f1_filtered,
                exact_filtered=result.exact_match_rate_filtered,
            )
        # Unarmed configured gate on legacy metrics: a floor violation still
        # BLOCKS (the operator configured floors to catch it), but meeting the
        # floors can never PASS — legacy segment-id mapping is not the metric
        # the floors were calibrated on, so arming requires diagnostics.
        n = result.groups_evaluated
        f1 = result.f1_filtered
        exact = result.exact_match_rate_filtered
        if n >= floors.min_mapped_groups and (
            f1 < floors.f1_filtered_floor or exact < floors.exact_filtered_floor
        ):
            return GateOutcome(
                dataset,
                STATUS_FAIL,
                f"legacy (no-sidecar) metrics below configured floors: "
                f"F1 {f1:.4f} vs {floors.f1_filtered_floor}, "
                f"exact {exact:.4f} vs {floors.exact_filtered_floor} ({n} groups); "
                "sidecar mapping diagnostics also unavailable",
                groups_evaluated=n,
                f1_filtered=f1,
                exact_filtered=exact,
            )
        return GateOutcome(
            dataset,
            STATUS_SKIP_UNARMED,
            "unarmed: requires sidecar mapping diagnostics, but they are "
            "unavailable (missing, malformed, or empty groups sidecar); "
            "legacy metrics cannot pass the gate",
            groups_evaluated=n,
            f1_filtered=f1,
            exact_filtered=exact,
        )

    curated_population = result.pair_labels_total + result.set_labels_total
    if (
        floors.armed
        and floors.min_labels_total is not None
        and curated_population < floors.min_labels_total
    ):
        return GateOutcome(
            dataset,
            STATUS_FAIL,
            f"curated label population {curated_population} "
            f"(pair {result.pair_labels_total} + set {result.set_labels_total}) "
            f"< floor {floors.min_labels_total}",
            groups_evaluated=result.groups_evaluated,
            f1_filtered=result.f1_filtered,
            exact_filtered=result.exact_match_rate_filtered,
        )

    n = result.groups_evaluated
    if n < floors.min_mapped_groups:
        if floors.armed:
            return GateOutcome(
                dataset,
                STATUS_FAIL,
                f"mapping regression: armed gate mapped {n} groups < "
                f"{floors.min_mapped_groups} required",
                groups_evaluated=n,
                f1_filtered=result.f1_filtered,
                exact_filtered=result.exact_match_rate_filtered,
            )
        return GateOutcome(
            dataset,
            STATUS_SKIP_UNARMED,
            f"unarmed: {n} groups mapped < {floors.min_mapped_groups} required",
            groups_evaluated=n,
            f1_filtered=result.f1_filtered,
            exact_filtered=result.exact_match_rate_filtered,
        )

    if (
        floors.armed
        and floors.min_mapping_rate is not None
        and result.pair_label_mapping_rate < floors.min_mapping_rate
    ):
        return GateOutcome(
            dataset,
            STATUS_FAIL,
            f"mapping retention {result.pair_label_mapping_rate:.4f} < floor "
            f"{floors.min_mapping_rate:.4f} ({result.pair_labels_mapped}/"
            f"{result.pair_labels_total} pair labels)",
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
