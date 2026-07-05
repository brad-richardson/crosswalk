"""GERS churn delta between two factory releases of one dataset.

Compares two bridge parquets at the local (target) id level and classifies each
local id's match into: ``same`` (identical GERS set), ``changed`` (matched in both
releases but to a different GERS set), ``lost`` (matched in ``from``, unmatched in
``to``), ``gained`` (unmatched in ``from``, matched in ``to``). This is the
consumer-facing release-notes artifact for a new Overture release.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CATEGORIES = ("same", "changed", "lost", "gained")


def _gers_by_local(bridge_path: Path) -> dict[str, set[str]]:
    """Map ``local_id`` -> set of matched ``gers_id`` from a bridge parquet."""
    df = pd.read_parquet(bridge_path, columns=["local_id", "gers_id"])
    out: dict[str, set[str]] = {}
    for local_id, gers_id in zip(df["local_id"].astype(str), df["gers_id"].astype(str)):
        out.setdefault(local_id, set()).add(gers_id)
    return out


@dataclass
class DeltaResult:
    """Result of a churn comparison between two releases."""

    dataset: str
    from_release: str
    to_release: str
    summary: dict[str, int]
    details: pd.DataFrame  # columns: local_id, category, from_gers, to_gers

    def to_markdown(self) -> str:
        s = self.summary
        total = sum(s[c] for c in CATEGORIES)
        churn = s["changed"] + s["lost"] + s["gained"]
        lines = [
            f"# GERS churn delta — {self.dataset}",
            "",
            f"- **From release:** `{self.from_release}`",
            f"- **To release:** `{self.to_release}`",
            f"- **Local ids compared (union):** {total}",
            f"- **Churn (changed + lost + gained):** {churn} ({100 * churn / total:.2f}% of union)"
            if total
            else "- **Churn:** 0",
            "",
            "| Category | Count | Meaning |",
            "|---|---:|---|",
            f"| same | {s['same']} | identical GERS set across releases |",
            f"| changed | {s['changed']} | matched in both, different GERS set |",
            f"| lost | {s['lost']} | matched in {self.from_release}, unmatched in {self.to_release} |",
            f"| gained | {s['gained']} | unmatched in {self.from_release}, matched in {self.to_release} |",
        ]
        return "\n".join(lines) + "\n"


def compute_delta(
    dataset: str,
    from_bridge: Path,
    to_bridge: Path,
    from_release: str,
    to_release: str,
) -> DeltaResult:
    """Compute the churn delta between two bridge parquets."""
    a = _gers_by_local(from_bridge)
    b = _gers_by_local(to_bridge)
    all_ids = sorted(set(a) | set(b))

    counts = {c: 0 for c in CATEGORIES}
    rows: list[dict[str, Any]] = []
    for lid in all_ids:
        ga = a.get(lid)
        gb = b.get(lid)
        if ga and gb:
            category = "same" if ga == gb else "changed"
        elif ga and not gb:
            category = "lost"
        else:
            category = "gained"
        counts[category] += 1
        if category != "same":
            rows.append(
                {
                    "local_id": lid,
                    "category": category,
                    "from_gers": ";".join(sorted(ga)) if ga else "",
                    "to_gers": ";".join(sorted(gb)) if gb else "",
                }
            )

    details = pd.DataFrame(rows, columns=["local_id", "category", "from_gers", "to_gers"])
    return DeltaResult(
        dataset=dataset,
        from_release=from_release,
        to_release=to_release,
        summary=counts,
        details=details,
    )
