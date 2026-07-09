"""Assemble the ``targets/`` publication staging tree (target dataset snapshots).

Sibling to ``publish.py`` (which assembles the ``bridges/`` tree). Where bridge
publishing turns factory *output* (matched bridge tables) into a public artifact,
target publishing turns factory *input* — the raw local dataset snapshot each
bridge was matched against — into one. Both trees share the same license gate
(``datasets/licenses.toml`` via :class:`~crosswalk.factory.licenses.LicenseRegistry`)
and the same R2 sync layer (``publish_sync.py``); this module only handles
target-specific discovery, snapshot-date resolution, and staging-tree layout:

    <staging>/
      targets/
        index.json                                  # {generated_from, datasets: {...}}
        dataset=<name>/
          latest.json                                # {dataset, latest_snapshot, path}
          snapshot=<fetch-date>/
            data.parquet                              # copied verbatim from data/raw
            meta.yaml                                  # normalized provenance sidecar

``snapshot=<fetch-date>`` partitions are immutable once published — enforced at
sync time by ``publish_sync.sync_targets_local`` / ``sync_targets_r2``, not here;
this module only assembles the (always rebuildable) local staging tree.

This module has NO dependency on the matching pipeline — it only reads
``data/raw/<name>_v1.0.parquet`` files and dataset/license config, so it is cheap
and safe to run anywhere those are available.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..config import DATA_VERSION
from .licenses import LicenseRegistry
from .publish import _load_dataset_yaml

TARGETS_PREFIX = "targets"
TARGET_DATA_FILENAME = "data.parquet"
TARGET_META_FILENAME = "meta.yaml"
LATEST_JSON_FILENAME = "latest.json"
TARGETS_INDEX_FILENAME = "index.json"

# Source-file naming convention (see ``filenames.py``): a bare target parquet is
# ``<dataset>_<DATA_VERSION>.parquet``. Overture / OSM reference-side files share
# the same version suffix but end (once the version suffix is stripped) with one
# of these role suffixes, so they must never be mistaken for a target. Matching
# the *exact* role suffix — not a bare ``_osm_`` / ``_overture_`` substring —
# avoids silently excluding a hypothetical target whose own name happens to
# contain those tokens.
_EXCLUDED_NAME_SUFFIXES = (
    "_overture_segments",
    "_overture_connectors",
    "_osm_segments",
    "_osm_connectors",
)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def discover_target_files(raw_dir: Path) -> dict[str, Path]:
    """Local target parquet files under ``raw_dir``, keyed by dataset name.

    Matches the ``<dataset>_<DATA_VERSION>.parquet`` naming convention (see
    ``filenames.py``); excludes Overture/OSM reference-side files, identified by
    their role suffix (``_overture_segments`` / ``_overture_connectors`` /
    ``_osm_segments`` / ``_osm_connectors``) rather than a bare substring.

    The glob (``*_<DATA_VERSION>.parquet``) already restricts to ``.parquet``
    names, so ``.bak`` / ``.suspect`` / ``.meta.yaml`` sidecars (which end in
    those extensions, e.g. ``<name>_v1.0.parquet.bak``) never match and need no
    separate filter.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return {}
    suffix = f"_{DATA_VERSION}.parquet"
    out: dict[str, Path] = {}
    for path in sorted(raw_dir.glob(f"*{suffix}")):
        name = path.name[: -len(suffix)]
        if any(name.endswith(role) for role in _EXCLUDED_NAME_SUFFIXES):
            continue
        out[name] = path
    return out


# --------------------------------------------------------------------------
# Snapshot-date + provenance resolution
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SnapshotProvenance:
    """Resolved provenance for a target snapshot, ready to write into ``meta.yaml``."""

    snapshot: str  # ISO date, e.g. "2026-07-06"
    source: str | None
    source_url: str | None
    feature_count: int | None
    id_column: str | None
    bbox: tuple[float, float, float, float] | None
    provenance_from: str  # "sidecar" | "yaml+mtime"


def _row_count(path: Path) -> int | None:
    """Parquet row count from file metadata only (no full read); None on failure."""
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def resolve_snapshot_provenance(
    path: Path,
    dataset: str,
    *,
    datasets_dir: Path | None = None,
) -> SnapshotProvenance:
    """Resolve the snapshot date + provenance fields for a target file.

    Precedence: the fetch sidecar ``<path>.meta.yaml`` (written by ``crosswalk
    fetch target``) when present — its ``fetched_at`` (first 10 chars) becomes
    the snapshot date. Otherwise falls back to ``datasets/<dataset>.yaml``'s
    ``source:`` block for source/source_url/id_column/bbox, and the parquet
    file's mtime (**UTC** calendar date) for the snapshot date.

    Both paths resolve the date in UTC — the sidecar's ``fetched_at`` is a
    tz-aware UTC datetime, so the mtime fallback must interpret the epoch mtime
    in UTC too. Using local time would mint a different immutable
    ``snapshot=<date>/`` path for a file whose mtime straddles UTC midnight
    depending on the machine's timezone (e.g. a laptop vs. a UTC CI runner).
    """
    from ..fetch.metadata import load_metadata

    meta = load_metadata(path)
    if meta is not None:
        return SnapshotProvenance(
            snapshot=meta.fetched_at.date().isoformat(),
            source=meta.source,
            source_url=meta.source_url,
            feature_count=meta.feature_count or _row_count(path),
            id_column=meta.id_column,
            bbox=meta.bbox,
            provenance_from="sidecar",
        )

    d = _load_dataset_yaml(dataset, datasets_dir)
    src = d.get("source") or {}
    fetch = d.get("fetch") or {}
    last_fetch = d.get("last_fetch") or {}
    target_fetch = last_fetch.get("target") or {} if isinstance(last_fetch, dict) else {}

    bbox_raw = fetch.get("bbox")
    bbox = tuple(float(x) for x in bbox_raw) if bbox_raw else None
    feature_count = target_fetch.get("feature_count") if isinstance(target_fetch, dict) else None

    snapshot = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date().isoformat()
    return SnapshotProvenance(
        snapshot=snapshot,
        source=src.get("type"),
        source_url=src.get("url") or src.get("portal_url"),
        feature_count=feature_count or _row_count(path),
        id_column=fetch.get("id_column"),
        bbox=bbox,
        provenance_from="yaml+mtime",
    )


# --------------------------------------------------------------------------
# Report structures
# --------------------------------------------------------------------------
@dataclass
class TargetSnapshotPublication:
    dataset: str
    status: str  # "published" | "excluded"
    reason: str | None = None
    display_name: str | None = None
    license: dict[str, Any] | None = None
    snapshot: str | None = None
    size_bytes: int | None = None
    source: str | None = None
    source_url: str | None = None
    provenance_from: str | None = None

    @property
    def published(self) -> bool:
        return self.status == "published"


@dataclass
class TargetsPublishReport:
    staging_dir: Path
    raw_dir: Path
    datasets: list[TargetSnapshotPublication] = field(default_factory=list)
    generated_from: str = ""

    @property
    def published(self) -> list[TargetSnapshotPublication]:
        return [d for d in self.datasets if d.published]

    @property
    def n_published(self) -> int:
        return len(self.published)

    @property
    def n_excluded(self) -> int:
        return len(self.datasets) - self.n_published


# --------------------------------------------------------------------------
# Staging assembly
# --------------------------------------------------------------------------
def assemble_targets_staging(
    raw_dir: Path,
    staging_dir: Path,
    registry: LicenseRegistry,
    *,
    datasets: list[str] | None = None,
    datasets_dir: Path | None = None,
    clean: bool = True,
) -> TargetsPublishReport:
    """Build the ``targets/`` staging tree from local target dataset snapshots.

    Only datasets that are BOTH ``approved`` in the license registry AND have a
    local ``data/raw/<name>_<DATA_VERSION>.parquet`` file are published; every
    other discovered target is recorded as excluded (with its reason) but never
    copied. Deterministic given identical inputs (files are copied verbatim);
    only the resolved snapshot date can change run-to-run if the sidecar/mtime
    provenance changes.
    """
    raw_dir = Path(raw_dir)
    staging_dir = Path(staging_dir)

    if clean and staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    target_files = discover_target_files(raw_dir)
    if datasets is not None:
        wanted = set(datasets)
        target_files = {k: v for k, v in target_files.items() if k in wanted}

    pubs: list[TargetSnapshotPublication] = []
    index_datasets: dict[str, Any] = {}
    targets_root = staging_dir / TARGETS_PREFIX

    for name, path in sorted(target_files.items()):
        decision = registry.decision(name)
        if not decision.approved:
            pubs.append(
                TargetSnapshotPublication(dataset=name, status="excluded", reason=decision.reason)
            )
            continue

        prov = resolve_snapshot_provenance(path, name, datasets_dir=datasets_dir)
        display = registry.display_name(name) or name

        ds_dir = targets_root / f"dataset={name}"
        snap_dir = ds_dir / f"snapshot={prov.snapshot}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        out_data = snap_dir / TARGET_DATA_FILENAME
        shutil.copyfile(path, out_data)
        size_bytes = out_data.stat().st_size

        meta = {
            "fetched_at": prov.snapshot,
            "source": prov.source,
            "source_url": prov.source_url,
            "feature_count": prov.feature_count,
            "id_column": prov.id_column,
            "bbox": list(prov.bbox) if prov.bbox else None,
            "dataset": name,
            "snapshot": prov.snapshot,
            "provenance_from": prov.provenance_from,
            "license": decision.license,
            "attribution": decision.attribution,
        }
        (snap_dir / TARGET_META_FILENAME).write_text(
            yaml.dump(meta, default_flow_style=False, sort_keys=False, allow_unicode=True)
        )

        latest = {
            "dataset": name,
            "latest_snapshot": prov.snapshot,
            "path": f"{TARGETS_PREFIX}/dataset={name}/snapshot={prov.snapshot}/{TARGET_DATA_FILENAME}",
        }
        (ds_dir / LATEST_JSON_FILENAME).write_text(json.dumps(latest, indent=2))

        index_datasets[name] = {
            "latest_snapshot": prov.snapshot,
            "size_bytes": size_bytes,
            "display_name": display,
            "license": decision.license,
            "attribution": decision.attribution,
            "source_url": decision.source_url,
        }

        pubs.append(
            TargetSnapshotPublication(
                dataset=name,
                status="published",
                display_name=display,
                license=decision.to_dict(),
                snapshot=prov.snapshot,
                size_bytes=size_bytes,
                source=prov.source,
                source_url=prov.source_url,
                provenance_from=prov.provenance_from,
            )
        )

    generated_from = str(raw_dir)
    index = {"generated_from": generated_from, "datasets": index_datasets}
    targets_root.mkdir(parents=True, exist_ok=True)
    (targets_root / TARGETS_INDEX_FILENAME).write_text(json.dumps(index, indent=2, sort_keys=True))

    return TargetsPublishReport(
        staging_dir=staging_dir,
        raw_dir=raw_dir,
        datasets=pubs,
        generated_from=generated_from,
    )
