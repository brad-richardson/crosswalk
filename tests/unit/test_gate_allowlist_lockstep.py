"""Lockstep guard: every armed mbench stitch-gate dataset must be resolver-prune
allowlisted in crosswalk.

Regression context (#372/#378): the mbench stitch gate scored an UNPRUNED row
set for ``us_boston_streets`` because the crosswalk adapter didn't pass the
dataset name through to ``crosswalk stitch``, so ``runner.py::
_effective_prune_threshold`` couldn't resolve the dataset's identity and
silently fell back to 0.0 (no prune) — a different row set than the one the
gate's floors in ``mbench/datasets.toml`` were calibrated against. #378 fixed
the adapter to pass the dataset name.

That fix only re-engages the pruned path when the dataset's key is ALSO
present in crosswalk's ``resolver_prune_overrides`` allowlist
(``src/crosswalk/config.py``) — a dataset absent from the allowlist is never
pruned regardless of identity, by design (see the allowlist's docstring). So
half of the #372 invariant was never pinned by a test: if a future maintainer
arms a new ``[gate.<dataset>]`` block in ``mbench/datasets.toml`` (or a key's
spelling drifts between the two files), ``_effective_prune_threshold`` will
silently return 0.0 again and the gate goes back to measuring an unpruned row
set — the #372 failure mode, recurring silently with no signal until someone
notices the floors no longer match reality.

This test pins the invariant directly: every ``[gate.*]`` dataset key in
``mbench/datasets.toml`` must exist in crosswalk's ``resolver_prune_overrides``.

This lives on the crosswalk side (not ``mbench/tests/test_gate.py``) because
the ``mbench`` CI job installs mbench's own minimal dependency set (pandas,
pyarrow, typer, rich, loguru, geopandas, shapely, osmium, pytest — see
``.github/workflows/ci.yml``) and never installs the ``crosswalk`` package
itself, so a test importing ``crosswalk.config`` cannot run there. The
crosswalk ``test`` job installs the full package and runs from the repo root,
so it can read ``mbench/datasets.toml`` by path and import ``crosswalk.config``
in the same test.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from crosswalk.config import settings

_DATASETS_TOML = Path(__file__).resolve().parents[2] / "mbench" / "datasets.toml"


def _gate_dataset_keys() -> set[str]:
    cfg = tomllib.loads(_DATASETS_TOML.read_text())
    gate = cfg.get("gate", {})
    assert isinstance(gate, dict), f"[gate] table in {_DATASETS_TOML} did not parse as a table"
    return set(gate)


def test_gate_datasets_are_prune_allowlisted():
    gate_keys = _gate_dataset_keys()
    # If this ever becomes empty, the invariant this test guards is vacuous —
    # fail loudly rather than silently pass on nothing.
    assert gate_keys, (
        f"no [gate.*] blocks found in {_DATASETS_TOML}; expected at least "
        "us_boston_streets to be armed"
    )

    allowlist = set(settings.resolver_prune_overrides)
    missing = gate_keys - allowlist

    assert not missing, (
        f"dataset(s) {sorted(missing)} have an armed [gate.*] block in "
        f"{_DATASETS_TOML} but are absent from crosswalk's "
        "resolver_prune_overrides allowlist (src/crosswalk/config.py, "
        "Settings.resolver_prune_overrides). A gated dataset not in this "
        "allowlist is never pruned, so `_effective_prune_threshold` silently "
        "returns 0.0 and the stitch gate scores an UNPRUNED row set against "
        "floors calibrated on the PRUNED path — the exact #372 failure mode "
        "(fixed for the identity-passing half in #378). Add the dataset to "
        "resolver_prune_overrides (after validating a threshold per the #284 "
        "sweep recipe) or remove/rename its [gate.*] block to keep the two "
        "configs in lockstep."
    )
