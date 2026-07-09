"""Publication license registry (Milestone M5).

The published R2 bridge tables are derived works of both the local source dataset
and Overture. Every published artifact must therefore carry attribution, and a
dataset may only be published once its source license has been *verified* by a
human. This module loads ``datasets/licenses.toml`` and answers, per dataset,
whether it is cleared to publish and with what attribution.

Policy (the publisher never guesses a license):

* ``status = "approved"`` — publishable; requires ``license`` + ``attribution``.
* ``status = "pending_review"`` — EXCLUDED (excluded-pending-review) until a human
  verifies the source terms and flips it to ``approved``.
* no registry entry — treated as ``pending_review`` (excluded).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_FILENAME = "licenses.toml"


@dataclass(frozen=True)
class LicenseDecision:
    """Per-dataset publication decision derived from the registry."""

    dataset: str
    approved: bool
    license: str | None = None
    attribution: str | None = None
    source_url: str | None = None
    note: str | None = None
    reason: str | None = None  # why excluded, when not approved

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "license": self.license,
            "attribution": self.attribution,
            "source_url": self.source_url,
            "note": self.note,
            "reason": self.reason,
        }


class LicenseRegistry:
    """Loads and queries ``datasets/licenses.toml``."""

    def __init__(self, data: dict[str, Any]):
        self._overture: dict[str, Any] = data.get("overture", {}) or {}
        self._datasets: dict[str, Any] = data.get("datasets", {}) or {}

    @classmethod
    def load(cls, path: Path | None = None) -> LicenseRegistry:
        """Load the registry from ``path`` (default: ``datasets/licenses.toml``)."""
        if path is None:
            from ..datasets.schema import get_datasets_dir

            path = get_datasets_dir() / REGISTRY_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"License registry not found at {path}. Publishing requires a "
                "reviewed license registry — see docs/PUBLISHING.md."
            )
        with path.open("rb") as fh:
            return cls(tomllib.load(fh))

    @property
    def overture(self) -> dict[str, Any]:
        """The global Overture attribution block (applies to every published table)."""
        return dict(self._overture)

    def display_name(self, dataset: str) -> str | None:
        """Best-effort human display name for ``dataset`` from its registry entry.

        Distinct from ``factory.publish.dataset_display()`` (which reads the
        dataset's own ``datasets/<name>.yaml``): this reads the ``display_name``
        already carried alongside each dataset's license entry in
        ``licenses.toml``, so target-snapshot publishing (which has no factory
        output to attach a fuller display block to) can label datasets without
        a second config lookup.
        """
        entry = self._datasets.get(dataset) or {}
        name = entry.get("display_name")
        return str(name) if name else None

    def decision(self, dataset: str) -> LicenseDecision:
        """Return the publication decision for ``dataset``.

        A missing entry, a non-approved status, or an approved entry missing its
        ``license``/``attribution`` all resolve to *excluded* — the publisher will
        never emit a table without verified attribution.
        """
        entry = self._datasets.get(dataset)
        if entry is None:
            return LicenseDecision(
                dataset=dataset,
                approved=False,
                reason="no license registry entry (excluded-pending-review)",
            )
        status = str(entry.get("status", "pending_review")).lower()
        lic = entry.get("license")
        attribution = entry.get("attribution")
        source_url = entry.get("source_url")
        note = entry.get("note")
        if status != "approved":
            return LicenseDecision(
                dataset=dataset,
                approved=False,
                source_url=source_url,
                note=note,
                reason=f"license status '{status}' (excluded-pending-review)",
            )
        if not lic or not attribution:
            return LicenseDecision(
                dataset=dataset,
                approved=False,
                source_url=source_url,
                note=note,
                reason="approved but missing license/attribution (excluded)",
            )
        return LicenseDecision(
            dataset=dataset,
            approved=True,
            license=lic,
            attribution=attribution,
            source_url=source_url,
            note=note,
        )
