"""Build the Phase-2 agent-stitching panel run set for us_boston_streets.

Composes ~50-60 sidecar groups from three strata and writes a group-ids file
that ``matcher agent stitch-batch --group-ids-file`` renders into one batch:

  (a) eval-continuity: sidecar groups recovered from the non-empty human
      stitching labels (edge-overlap recovery, same set as --recover-labeled);
  (b) reject-all continuity: sidecar groups for the empty-edge reject-all human
      labels that still exist verbatim (segment overlap is impossible — no
      segments were stored — so only exact group_id survival recovers them);
  (c) fresh fill: standard tier-selected unlabeled groups (large / borderline /
      low-confidence / clear-winner mix), excluding (a), (b) and reviewed ids.

Usage:
    uv run python scripts/build_panel_phase2_runset.py \
        --dataset us_boston_streets --fill 45 -o /tmp/phase2_runset.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matcher.agent_labeling.stitch_eval import (
    recover_empty_reject_all,
    recover_labeled_groups,
)
from matcher.filenames import PROJECT_ROOT, bridge_filename, groups_sidecar_path
from matcher.labeling.stitching_store import StitchingLabelStore
from matcher.matching.alternatives import generate_top_k_alternatives
from matcher.matching.batch_selection import select_stitching_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="us_boston_streets")
    ap.add_argument("--fill", type=int, default=45, help="fresh tier-selected group count")
    ap.add_argument("-k", "--alternatives", type=int, default=5)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    bridge = PROJECT_ROOT / "data" / "output" / bridge_filename(args.dataset)
    sidecar_path = groups_sidecar_path(bridge)
    if not sidecar_path.exists():
        raise SystemExit(
            f"Groups sidecar not found: {sidecar_path}\n"
            f"Run the pipeline first, e.g.:\n"
            f"  uv run matcher stitch <reference.parquet> <target.parquet> "
            f"-m xgboost -o {bridge}"
        )
    sidecar = json.loads(sidecar_path.read_text())
    groups = sidecar.get("groups", [])
    if not groups:
        raise SystemExit(f"Sidecar has no groups: {sidecar_path}")
    print(f"Loaded {len(groups)} sidecar groups")

    store = StitchingLabelStore(args.dataset)
    human_df = store.load(args.dataset)

    # (a) eval-continuity from non-empty human labels.
    rec = recover_labeled_groups(groups, human_df)
    labeled = list(rec["target_group_ids"])
    print(
        f"(a) recover-labeled: {len(rec['clean'])} clean + {len(rec['split'])} split "
        f"-> {len(labeled)} groups"
    )

    # (b) reject-all continuity.
    emp = recover_empty_reject_all(groups, human_df)
    empty_new = [g for g in emp["recovered"] if g not in labeled]
    print(
        f"(b) reject-all: {len(emp['recovered'])} recoverable (+{len(empty_new)} new), "
        f"{len(emp['unrecoverable'])} unrecoverable"
    )

    # (c) fresh tier-selected fill, excluding (a)+(b)+reviewed.
    for g in groups:
        g["alternatives"] = generate_top_k_alternatives(
            g.get("edges", []),
            ref_geoms=g.get("ref_geometries", {}),
            target_geoms=g.get("target_geometries", {}),
            k=args.alternatives,
        )
    exclude = set(labeled) | set(empty_new) | store.get_reviewed_group_ids(args.dataset)
    fill = select_stitching_batch(groups, exclude, k=args.fill)
    fill_ids = [g["group_id"] for g in fill]
    tiers = {}
    for g in fill:
        tiers[g.get("review_tier", "?")] = tiers.get(g.get("review_tier", "?"), 0) + 1
    print(f"(c) fresh fill: {len(fill_ids)} groups, tiers={tiers}")

    combined: list[str] = []
    for gid in labeled + empty_new + fill_ids:
        if gid not in combined:
            combined.append(gid)
    args.output.write_text("\n".join(combined) + "\n")
    print(f"\nTotal run set: {len(combined)} groups -> {args.output}")

    # Emit a machine-readable composition sidecar for the analysis step.
    comp = {
        "labeled": labeled,
        "empty_recovered": empty_new,
        "empty_unrecoverable": emp["unrecoverable"],
        "fresh_fill": fill_ids,
        "fresh_tiers": tiers,
        "all": combined,
    }
    args.output.with_suffix(".composition.json").write_text(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
