"""The wave-manifest contract for physical/coincidence stitch voting waves.

A wave manifest is the immutable handoff between :mod:`build_physical_stitch_wave`
(which packs evidence and stamps the manifest) and :mod:`run_physical_stitch_wave`
(which validates the manifest and executes its counterbalanced schedule). Before
this module the field names were spelled as independent string literals in the
builder, the runner's validator, and the tests, the ``schema_version`` was
stamped but never checked, and there was no internal integrity digest — the only
tamper check was an out-of-band sha256 in the operator's handoff notes.

This module owns:

* the field-name constants (so builder and runner cannot drift),
* :func:`WaveManifest.write` — stamps ``schema_version`` and an internal
  ``manifest_sha256`` digest, and
* :func:`WaveManifest.load_validated` — re-derives the panel roster, checks the
  structural invariants, resolves ``batch_dir`` entries relative to the manifest
  file, and hard-fails on a schema mismatch or a tampered digest.

Backward compatibility: the JSON layout is unchanged apart from the trailing
``manifest_sha256`` field. A legacy manifest that predates the digest (it has no
``manifest_sha256``) validates with a warning rather than a hard failure; a
manifest whose digest is present but does not match its content is fatal.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stitch_runner import ProviderSpec, get_panel, panel_descriptor

# The manifest schema version. ``write`` stamps this; ``load_validated`` refuses
# a manifest stamped with any other version (a forward/backward incompatible
# layout must bump this and grow an explicit migration path).
SCHEMA_VERSION = 1

# Top-level manifest field names — the contract, spelled once.
FIELD_SCHEMA_VERSION = "schema_version"
FIELD_PANEL = "panel"
FIELD_REQUIRED_PANEL = "required_panel"
FIELD_TOTAL_PACK_COUNT = "total_pack_count"
FIELD_RUN_SCHEDULE = "run_schedule"
FIELD_BATCH_DIRS = "batch_dirs"
FIELD_DIGEST = "manifest_sha256"

# Per-schedule-row field names.
ROW_RUN_INDEX = "run_index"
ROW_BATCH_DIR = "batch_dir"
ROW_GROUP_ID = "group_id"
ROW_DATASET_ID = "dataset_id"
ROW_VARIANT = "variant"


def compute_digest(content: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of ``content`` EXCLUDING the digest field.

    The digest is taken over canonicalized JSON (sorted keys, tight separators)
    rather than the file bytes, so re-indenting or reordering the on-disk file
    never changes it — only a change to the manifest's *content* does.
    """
    payload = {key: value for key, value in content.items() if key != FIELD_DIGEST}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_batch_dir(raw: str, manifest_path: Path) -> Path:
    """Resolve a schedule row's ``batch_dir`` without depending on the cwd.

    Absolute paths pass through. Otherwise the candidates are tried in order and
    the first that exists wins:

    1. manifest-relative (``manifest_dir / raw``),
    2. sibling-by-basename (``manifest_dir / basename(raw)``) — the v7 layout,
       where the manifest and every batch dir are siblings but the stored path
       is the full repo-relative string, and
    3. cwd-relative (``raw`` as-is) — the legacy behavior, for a runner invoked
       from the repo root.

    If none exist, the sibling-by-basename candidate is returned so validation's
    ``FileNotFoundError`` points at the manifest's own directory rather than a
    cwd-dependent guess.
    """
    path = Path(raw)
    if path.is_absolute():
        return path
    parent = manifest_path.parent
    manifest_relative = parent / path
    sibling = parent / path.name
    cwd_relative = path
    for candidate in (manifest_relative, sibling, cwd_relative):
        if candidate.exists():
            return candidate
    return sibling


@dataclass
class WaveManifest:
    """A wave manifest and its resolved panel.

    ``content`` is the full manifest dict (structural fields plus the builder's
    descriptive blocks such as ``selections`` and ``factorial_controls``, which
    the runner ignores but which are preserved verbatim). ``panel`` is populated
    by :meth:`load_validated` with the resolved :class:`ProviderSpec` roster.
    """

    content: dict[str, Any]
    source_path: Path | None = None
    panel: list[ProviderSpec] = field(default_factory=list)

    @property
    def panel_name(self) -> str:
        return str(self.content.get(FIELD_PANEL, ""))

    @property
    def run_schedule(self) -> list[dict[str, Any]]:
        return self.content.get(FIELD_RUN_SCHEDULE) or []

    @property
    def total_pack_count(self) -> int:
        return int(self.content.get(FIELD_TOTAL_PACK_COUNT, -1))

    def write(self, path: Path) -> Path:
        """Serialize to ``path``, stamping ``schema_version`` and the digest.

        The digest is computed last, over everything else, and appended — so a
        manifest written here always round-trips through
        :meth:`load_validated` cleanly.
        """
        content = dict(self.content)
        content[FIELD_SCHEMA_VERSION] = SCHEMA_VERSION
        content.pop(FIELD_DIGEST, None)
        content[FIELD_DIGEST] = compute_digest(content)
        path.write_text(json.dumps(content, indent=2) + "\n")
        self.content = content
        self.source_path = path
        return path

    @classmethod
    def load_validated(cls, path: Path) -> WaveManifest:
        """Load, integrity-check, and structurally validate a wave manifest.

        Checks, in order: schema version, integrity digest (legacy-missing is a
        warning, mismatch is fatal), panel-roster drift, schedule/pack-count
        agreement, contiguous run indices, no duplicate packs, and the presence
        of every referenced ``batch.json`` and ``evidence.json``. ``batch_dir``
        entries are resolved relative to the manifest file and rewritten in the
        returned content so downstream code is cwd-independent.
        """
        path = Path(path)
        content = json.loads(path.read_text())

        _validate_schema_version(content, path)
        _validate_digest(content, path)

        panel_name = str(content.get(FIELD_PANEL, ""))
        panel = get_panel(panel_name)
        expected_panel = panel_descriptor(panel)
        if expected_panel != content.get(FIELD_REQUIRED_PANEL):
            raise ValueError(
                f"Panel drift for {panel_name!r}: expected {expected_panel}, "
                f"got {content.get(FIELD_REQUIRED_PANEL)}"
            )

        schedule = content.get(FIELD_RUN_SCHEDULE) or []
        if len(schedule) != int(content.get(FIELD_TOTAL_PACK_COUNT, -1)):
            raise ValueError("run_schedule length does not match total_pack_count")
        expected_indices = list(range(1, len(schedule) + 1))
        if [row.get(ROW_RUN_INDEX) for row in schedule] != expected_indices:
            raise ValueError("run_schedule indices are not contiguous and ordered")

        seen: set[tuple[str, str]] = set()
        for row in schedule:
            batch_dir = resolve_batch_dir(str(row[ROW_BATCH_DIR]), path)
            row[ROW_BATCH_DIR] = str(batch_dir)
            group_id = str(row[ROW_GROUP_ID])
            key = (str(batch_dir), group_id)
            if key in seen:
                raise ValueError(f"duplicate scheduled pack: {key}")
            seen.add(key)
            if not (batch_dir / "batch.json").is_file():
                raise FileNotFoundError(batch_dir / "batch.json")
            if not (batch_dir / group_id / "evidence.json").is_file():
                raise FileNotFoundError(batch_dir / group_id / "evidence.json")

        return cls(content=content, source_path=path, panel=panel)


def _validate_schema_version(content: dict[str, Any], path: Path) -> None:
    version = content.get(FIELD_SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported manifest schema_version {version!r} "
            f"(this runner understands version {SCHEMA_VERSION})"
        )


def _validate_digest(content: dict[str, Any], path: Path) -> None:
    stored = content.get(FIELD_DIGEST)
    if stored is None:
        warnings.warn(
            f"{path}: manifest has no {FIELD_DIGEST}; treating as a legacy "
            "pre-digest manifest (integrity is unverified)",
            stacklevel=2,
        )
        return
    actual = compute_digest(content)
    if stored != actual:
        raise ValueError(
            f"{path}: manifest integrity digest mismatch "
            f"(stored {stored}, computed {actual}) — the manifest has been "
            "modified since it was written; refusing to run"
        )
