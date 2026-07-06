"""Resolve the agent panel's per-group routing for a dataset.

The 3-provider stitching panel (see :mod:`stitch_runner`) votes on M:N groups
and writes a ``consensus.csv`` per batch dir with a ``routing`` column whose
values are ``auto_accept`` (unanimous, safe to promote) or ``human_review``
(non-unanimous vote, NONE consensus, cross-mode flags, oversize, ...). A group
that the panel could not auto-accept is exactly the kind of decision worth a
human's 1-2 minutes.

This module discovers a dataset's panel batch dirs (``{dataset}`` or
``{dataset}_*`` under ``data/agents/stitching/batches``), resolves the MOST
RECENT vote per group (a group may be voted across several waves), and exposes
the set of group ids the panel routed to ``human_review``. The human
``/stitching-review`` queue is gated to that set so it only ever contains
panel failures — never never-voted curiosity/calibration samples.

Recency is determined by each ``consensus.csv``'s mtime (older waves first, so
the newest wave's routing wins on conflict; the dir name breaks ties). That
mirrors the precedence-ordered merge used by the label exporter
(:func:`stitch_export._merge_consensus`) without requiring the caller to know
the wave order.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..filenames import PROJECT_ROOT

# Root under which each panel wave writes ``<batch_name>/consensus.csv``.
STITCH_BATCHES_DIR = PROJECT_ROOT / "data" / "agents" / "stitching" / "batches"

ROUTING_HUMAN_REVIEW = "human_review"
ROUTING_AUTO_ACCEPT = "auto_accept"


def _dataset_batch_dirs(dataset: str, batches_root: Path) -> list[Path]:
    """Batch dirs belonging to ``dataset``, oldest consensus.csv first.

    A dir belongs to the dataset when its name equals the dataset or begins with
    ``{dataset}_`` (e.g. ``us_seattle_sidewalks_phase2``). Only dirs that carry a
    ``consensus.csv`` are returned. Ordering is by consensus mtime ascending
    (name as tie-breaker) so a later wave supersedes an earlier one when both
    voted the same group.
    """
    if not dataset or not batches_root.exists():
        return []
    dirs = [
        d
        for d in batches_root.iterdir()
        if d.is_dir()
        and (d.name == dataset or d.name.startswith(dataset + "_"))
        and (d / "consensus.csv").is_file()
    ]
    dirs.sort(key=lambda d: ((d / "consensus.csv").stat().st_mtime, d.name))
    return dirs


def latest_panel_routing(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> dict[str, str]:
    """Map each voted ``group_id`` to its most-recent panel ``routing`` value.

    Later waves (by consensus mtime) overwrite earlier ones. Returns an empty
    dict when the dataset has no panel batches. Rows with a blank group_id are
    skipped; a blank routing is preserved as-is (it is simply not
    ``human_review``, so it will not gate anything in).
    """
    routing: dict[str, str] = {}
    for batch_dir in _dataset_batch_dirs(dataset, batches_root):
        with open(batch_dir / "consensus.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                gid = str(row.get("group_id", "") or "").strip()
                if not gid:
                    continue
                routing[gid] = str(row.get("routing", "") or "").strip()
    return routing


def panel_failed_group_ids(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> set[str]:
    """Group ids whose most-recent panel vote routed to ``human_review``.

    This is the allow-list for the human review queue: the groups the agent
    panel could not auto-accept. Groups never voted by the panel are absent (they
    do not enter the human queue by default).
    """
    return {
        gid
        for gid, routing in latest_panel_routing(dataset, batches_root).items()
        if routing == ROUTING_HUMAN_REVIEW
    }
