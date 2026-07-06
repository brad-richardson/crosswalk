"""Discover stitchable dataset triples from a raw-data directory.

A stitchable dataset is a local target parquet paired with Overture segments
(the matching reference) and, by convention, Overture connectors. Discovery uses
the shared :class:`DatasetLoader` file-resolution logic (naming convention +
version pinning) so it stays consistent with ``crosswalk stitch``.

The Overture release identifier is derived from the segments file's sidecar
``.meta.yaml`` (``release:`` field), falling back to a caller-supplied override.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from ..datasets.loader import DatasetLoader


@dataclass(frozen=True)
class DatasetPair:
    """A stitchable local↔Overture dataset triple resolved on disk."""

    name: str
    reference_path: Path  # Overture segments
    target_path: Path  # local dataset
    connectors_path: Path | None  # Overture connectors (may be absent)
    release: str | None  # Overture release from segments .meta.yaml, if present

    @property
    def has_connectors(self) -> bool:
        return self.connectors_path is not None and self.connectors_path.exists()


def _connectors_for(segments_path: Path) -> Path | None:
    """Derive the Overture connectors path from a segments path by convention."""
    name = segments_path.name.replace("overture_segments", "overture_connectors", 1)
    cand = segments_path.with_name(name)
    return cand if cand.exists() else None


def read_release_from_meta(segments_path: Path) -> str | None:
    """Read the Overture ``release`` from a segments file's ``.meta.yaml`` sidecar.

    Returns None when the sidecar is missing, unparseable, or has no non-null
    ``release`` field.
    """
    meta_path = segments_path.with_name(segments_path.name + ".meta.yaml")
    if not meta_path.exists():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not parse {meta_path}: {exc}")
        return None
    if not isinstance(meta, dict):
        return None
    release = meta.get("release")
    if release is None or (isinstance(release, str) and not release.strip()):
        return None
    return str(release)


def discover_pairs(
    raw_dir: Path | str | None = None,
    names: list[str] | None = None,
) -> list[DatasetPair]:
    """Discover stitchable dataset triples under ``raw_dir``.

    Args:
        raw_dir: Raw-data directory (default: project ``data/raw``).
        names: Restrict discovery to these dataset names; None = all discovered.

    Returns:
        Sorted list of :class:`DatasetPair` for datasets with both reference and
        target present. Requested names that are not stitchable are logged and
        skipped (never raise — one missing dataset must not abort a batch).
    """
    loader = DatasetLoader(data_dir=raw_dir)
    if names:
        candidate_names = list(dict.fromkeys(names))  # de-dupe, preserve order
    else:
        candidate_names = loader.list_available()

    pairs: list[DatasetPair] = []
    for name in candidate_names:
        ref = loader.find_reference_path(name)
        tgt = loader.find_target_path(name)
        if ref is None or tgt is None:
            missing = "reference" if ref is None else "target"
            logger.warning(f"Skipping '{name}': missing {missing} file under {loader.data_dir}")
            continue
        pairs.append(
            DatasetPair(
                name=name,
                reference_path=ref,
                target_path=tgt,
                connectors_path=_connectors_for(ref),
                release=read_release_from_meta(ref),
            )
        )
    return sorted(pairs, key=lambda p: p.name)


def resolve_release(pair: DatasetPair, override: str | None = None) -> str:
    """Resolve the release identifier for a pair.

    Precedence: explicit ``override`` > the pair's ``release`` (from meta.yaml).
    Raises ``ValueError`` when neither is available, so the factory never writes
    to an ambiguous ``release=`` partition.
    """
    if override:
        return override
    if pair.release:
        return pair.release
    raise ValueError(
        f"No Overture release for '{pair.name}': its segments .meta.yaml has no "
        "'release' field. Pass --release to set one explicitly."
    )


def segment_count(path: Path) -> int:
    """Return the parquet row count from file metadata (no full read)."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows
