#!/usr/bin/env python
"""Prepare/run a bounded direct-edge panel. Outputs weak observations, never labels.

--packs is a JSON list of {dataset_id, path}, pointing at existing verified packs.
Preparation makes no model calls. --run draws each pinned seat once and resumes
completed draws without retrying malformed answers until a majority appears.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from crosswalk.agent_labeling.stitch_edge_ballots import (
    FRONTIER_PANEL,
    invoke_direct_seat,
    panel_configuration,
    parse_edge_ballot,
    prepare_direct_pack,
)
from crosswalk.agent_labeling.stitch_provenance import (
    load_evidence_manifest,
    sha256_file,
    sha256_json,
)
from crosswalk.resolver.observations import build_vote_observations, reconcile_observations


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opencode-config", type=Path, default=Path("opencode.json"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    config = json.loads(args.opencode_config.read_text())
    models = config["provider"]["meta"]["models"]
    models["muse-spark-1.3-contributor"] = {
        **models["muse-spark-1.1"],
        "name": "Muse Spark 1.3 Contributor",
    }
    configuration = panel_configuration(config)
    configuration["driver_sha256"] = sha256_file(__file__)
    signature = sha256_json(configuration)
    requested = json.loads(args.packs.read_text())
    if not requested:
        raise ValueError("At least one explicitly selected pack is required")
    selection_path = args.out / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["panel_invocation_sha256"] != signature or selection["requested"] != requested:
            raise ValueError("Selection or panel contract changed; use a new wave directory")
        for case in selection["cases"]:
            original = load_evidence_manifest(Path(case["path"]), allow_legacy=False)
            if (
                original["evidence_pack_sha256"]
                != case["manifest"]["evidence"]["source_evidence_pack_sha256"]
            ):
                raise ValueError("Source pack changed since wave preparation")
            for name, digest in case["manifest"]["files"].items():
                if sha256_file(args.out / case["pack"] / name) != digest:
                    raise ValueError("Prepared direct evidence changed")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        cases = []
        identities = set()
        for item in requested:
            source = Path(item["path"])
            dataset = item["dataset_id"]
            if any(c in dataset for c in ["/", "\\"]) or dataset in {"", ".", ".."}:
                raise ValueError("Invalid dataset identity")
            identity = (dataset, source.name)
            if identity in identities:
                raise ValueError("Duplicate selected group")
            identities.add(identity)
            pack = Path(f"dataset={dataset}") / source.name
            manifest = prepare_direct_pack(source, args.out / pack)
            cases.append({**item, "group_id": source.name, "pack": str(pack), "manifest": manifest})
        selection = {
            "configuration": configuration,
            "panel_invocation_sha256": signature,
            "requested": requested,
            "cases": cases,
        }
        write_json(selection_path, selection)
    if not args.run:
        print(f"Prepared {len(selection['cases'])} direct-edge packs; no model calls")
        return

    def draw(case, provider):
        pack = args.out / case["pack"]
        destination = pack / f"{provider.name}.ballot.json"
        if destination.exists():
            record = json.loads(destination.read_text())
            if record["panel_invocation_sha256"] != signature:
                raise ValueError("Stored ballot invocation mismatch")
            return record
        evidence = case["manifest"]["evidence"]
        start = time.monotonic()
        record = {
            "dataset_id": case["dataset_id"],
            "group_id": case["group_id"],
            "source_batch": args.out.name,
            "provider": provider.name,
            "model": provider.model,
            "protocol_version": evidence["protocol_version"],
            "evidence_id": evidence["evidence_id"],
            "evidence_pack_sha256": case["manifest"]["evidence_pack_sha256"],
            "panel_invocation_sha256": signature,
            "timestamp": datetime.now(UTC).isoformat(),
            "choice": "EDGES",
            "edge_set": "[]",
            "error": "",
            "none_reason": "",
            "edge_decisions": "[]",
            "attempt": 1,
            "evidence_delivery": {
                "images": {
                    k: v for k, v in case["manifest"]["files"].items() if k.endswith(".png")
                },
                "transport": {"fable": "claude:Read", "astra": "codex:-i", "muse": "opencode:-f"}[
                    provider.name
                ],
            },
        }
        try:
            raw = invoke_direct_seat(provider, pack, case["manifest"], config)
            (pack / f"{provider.name}.raw.txt").write_text(raw)
            ballot = parse_edge_ballot(raw, evidence)
            record.update(
                none_reason=ballot["none_reason"],
                reasoning=ballot["reasoning"],
                edge_decisions=json.dumps(ballot["edge_decisions"], sort_keys=True),
            )
            record["edge_set"] = json.dumps(
                [
                    [e["ref_id"], e["target_id"]]
                    for e in ballot["edge_decisions"]
                    if e["resolution"] == "keep"
                ]
            )
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            message = f"{type(exc).__name__}: {exc}"
            for key, value in os.environ.items():
                if any(part in key for part in ["KEY", "TOKEN", "SECRET"]) and len(value) > 8:
                    message = message.replace(value, "[REDACTED]")
            record.update(choice="ABSTAIN", error=message[:2000])
        record["latency_s"] = round(time.monotonic() - start, 2)
        write_json(destination, record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in ["group_id", "provider", "none_reason", "latency_s", "error"]
                }
            ),
            flush=True,
        )
        return record

    # Models are independently blind to peers; at most one three-seat panel runs
    # concurrently. Each result is durably recorded before proceeding.
    votes = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for case in selection["cases"]:
            votes.extend(
                executor.map(lambda provider, case=case: draw(case, provider), FRONTIER_PANEL)
            )
    evidence_rows = [
        {
            "dataset_id": c["dataset_id"],
            "group_id": c["group_id"],
            "evidence_id": c["manifest"]["evidence"]["evidence_id"],
            "evidence_pack_sha256": c["manifest"]["evidence_pack_sha256"],
            "evidence": json.dumps(c["manifest"]["evidence"], sort_keys=True),
        }
        for c in selection["cases"]
    ]
    frame, evidence = pd.DataFrame(votes), pd.DataFrame(evidence_rows)
    for dataset, subset in frame.groupby("dataset_id"):
        directory = args.out / f"dataset={dataset}"
        subset.to_csv(directory / "votes.csv", index=False)
        evidence[evidence["dataset_id"] == dataset].to_csv(directory / "evidence.csv", index=False)
    observations = build_vote_observations(frame, evidence)
    reconciled = reconcile_observations(observations)
    reconciled.to_parquet(args.out / "reconciled.parquet", index=False)
    report = {
        "panel_invocation_sha256": signature,
        "cases": len(selection["cases"]),
        "draws": len(frame),
        "valid_draws": int((frame["error"] == "").sum()),
        "none_reasons": frame["none_reason"].value_counts().to_dict(),
        "audit": reconciled.attrs["observation_audit"],
        "tiers": reconciled["vote_tier"].value_counts().to_dict(),
        "selection_sha256": sha256_file(selection_path),
        "observations_sha256": sha256_file(args.out / "reconciled.parquet"),
        "scope": "Purposive protocol smoke test, not a population accuracy estimate; not admitted to the frozen archive experiment.",
    }
    write_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
