"""Bounded, advisory-only panel replay; never exports labels or changes models.

Run from the repo root with uv run --no-sync python <this file> --probe, then
--remaining. Requires the existing authenticated CLIs and META_API_KEY in the
environment or .env. Original pack bytes and selection are recorded before calls.
"""

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from crosswalk.agent_labeling import stitch_runner as runner
from crosswalk.agent_labeling.stitch_provenance import load_evidence_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/frontier_stitch_trial_2026-09-09"
PACKS = ROOT / "data/agents/stitching/batches"
CASES = [
    ("us_boston_streets", "us_boston_streets_quadcal0710", "72063362", "human_exact_positive"),
    ("us_boston_streets", "us_boston_streets_quadcal0710", "f69a827e", "human_exact_positive"),
    (
        "au_sydney_roads",
        "au_sydney_roads_physical_context_v9_rerun_20260717",
        "66e22055",
        "human_exact_rejection",
    ),
    (
        "de_berlin_roads",
        "de_berlin_roads_physical_context_v8_20260717",
        "d4d2e782",
        "human_exact_rejection",
    ),
    (
        "hk_hongkong_roads",
        "hk_hongkong_roads_physical_context_v8_20260717",
        "4eed5e80",
        "human_exact_rejection",
    ),
    (
        "fi_helsinki_roads",
        "fi_helsinki_roads_physical_context_v9_rerun_20260717",
        "92c0997f",
        "targeted_diagnostic",
    ),
    (
        "fi_helsinki_roads",
        "fi_helsinki_roads_physical_context_v9_rerun_20260717",
        "7175635e",
        "targeted_diagnostic",
    ),
    (
        "ch_grand_geneva_cycle_schema",
        "ch_grand_geneva_cycle_schema_physical_context_v9_seeded_20260718",
        "a451bf05",
        "targeted_diagnostic",
    ),
    (
        "au_sydney_roads",
        "au_sydney_roads_bulk_v7_20260718",
        "3fd8483c__p07ce5199df",
        "accepted_child_failed_parent",
    ),
    (
        "fi_helsinki_roads",
        "fi_helsinki_roads_bulk_v7_20260718",
        "c8b85bbc__p07741daa80",
        "accepted_child_failed_parent",
    ),
]
PANEL = [
    runner.ProviderSpec("claude", "claude-fable-5-1", "high", timeout=480),
    runner.ProviderSpec("codex", "gpt-6-astra", "high", timeout=480),
    runner.ProviderSpec(
        "muse", "meta/muse-spark-1.3-contributor", "high", timeout=480, opencode_agent="vote"
    ),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "opencode.json").read_text())
    muse = config["provider"]["meta"]["models"].pop("muse-spark-1.1")
    muse["name"] = "Muse Spark 1.3 Contributor"
    config["provider"]["meta"]["models"]["muse-spark-1.3-contributor"] = muse
    manifest = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "purpose": "Purposive ten-pack smoke/diagnostic comparison, not a population accuracy estimate.",
        "panel": [dataclasses.asdict(p) for p in PANEL],
        "harness_sha256": digest(__file__),
        "runner_sha256": digest(runner.__file__),
        "opencode_override": config,
        "transport_note": "Existing invokers; Codex runs from a neutral temporary directory. Muse receives an explicit config override. Claude uses Read; the others receive native images.",
        "cases": [],
    }
    for dataset, batch, group, stratum in CASES:
        source = PACKS / batch / group
        letters, options, metadata = runner._load_group_context(source)
        item = {
            "dataset": dataset,
            "source_batch": batch,
            "group_id": group,
            "stratum": stratum,
            "source_path": str(source.relative_to(ROOT)),
            "files": {p.name: digest(p) for p in sorted(source.iterdir()) if p.is_file()},
            "n_reference": len(metadata["segments"]["reference"]),
            "n_target": len(metadata["segments"]["target"]),
            "n_images": len(list(source.glob("*.png"))),
            "prompt_chars": len((source / "prompt.txt").read_text()),
            "options": options,
            "human_expected": None,
        }
        if stratum.startswith("human_exact"):
            path = ROOT / "labels/stitching" / f"dataset={dataset}" / "data.csv"
            rows = [
                r
                for r in csv.DictReader(path.open())
                if r["group_id"] == group
                and r["labeler"] == "brad"
                and r["label_semantics"] == "pair"
            ]
            assert len(rows) == 1
            row = rows[0]
            truth = sorted((e["ref_id"], e["target_id"]) for e in json.loads(row["selected_edges"]))
            choices = [letter for letter, edges in options.items() if sorted(edges) == truth]
            assert choices or not truth, (group, "human answer missing from menu")
            item["human_expected"] = {
                "edges": truth,
                "choices": choices or ["NONE"],
                "session_id": row["session_id"],
                "notes": row["notes"],
                "source_sha256": digest(path),
            }
        manifest["cases"].append(item)
    # Normalize tuples to JSON arrays before comparing a resumed selection.
    manifest = json.loads(json.dumps(manifest))
    path = OUTPUT / "selection.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old == manifest, "Selection/provenance changed; use a new trial directory."
    else:
        write_json(path, manifest)
    return manifest, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--remaining", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    manifest, config = prepare()
    if not (args.probe or args.remaining):
        print(f"Prepared {len(CASES)} packs; no model calls.")
        return

    original_claude = runner.invoke_claude
    original_opencode = runner.invoke_opencode
    active = threading.local()

    def capture(provider, group_dir, raw):
        # The runner's scratch directory contains images, not metadata.
        group = active.group_id
        (OUTPUT / f"{group}.{provider}.raw.txt").write_text(raw)
        return raw

    def claude(prompt, group_dir, letters, model, timeout, effort):
        return capture(
            "claude", group_dir, original_claude(prompt, group_dir, letters, model, timeout, effort)
        )

    def muse(prompt, group_dir, letters, model, timeout, effort):
        return capture(
            "muse",
            group_dir,
            original_opencode(
                prompt,
                group_dir,
                letters,
                model,
                timeout,
                effort,
                agent="vote",
                config_content=config,
            ),
        )

    def codex(prompt, group_dir, letters, model, timeout, effort):
        with tempfile.TemporaryDirectory(prefix="crosswalk_trial_codex_") as tmp:
            output = Path(tmp) / "answer.json"
            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "--ephemeral",
                "-m",
                model,
                "-c",
                f"model_reasoning_effort={effort}",
            ]
            for img in runner._image_paths(group_dir, letters):
                cmd += ["-i", img]
            cmd += ["-o", str(output), "-"]
            result = subprocess.run(
                cmd, input=prompt, text=True, capture_output=True, timeout=timeout, cwd=tmp
            )
            runner._check_exit("codex", result)
            raw = output.read_text() if output.exists() else result.stdout
            return capture("codex", group_dir, raw)

    runner._INVOKERS.update(claude=claude, codex=codex, muse=muse)
    selected = manifest["cases"][:1] if args.probe else manifest["cases"][1:]

    def run_one(item, provider):
        group = item["group_id"]
        active.group_id = group
        destination = OUTPUT / f"{group}.{provider.name}.json"
        if destination.exists():
            return json.loads(destination.read_text())
        started = time.monotonic()
        record = {
            "group_id": group,
            "dataset": item["dataset"],
            "provider": provider.name,
            "model": provider.model,
        }
        try:
            source = ROOT / item["source_path"]
            assert all(digest(source / name) == sha for name, sha in item["files"].items())
            with tempfile.TemporaryDirectory(prefix="crosswalk_trial_pack_") as tmp:
                copied = Path(tmp) / group
                shutil.copytree(source, copied)
                evidence = load_evidence_manifest(copied)
                letters, options, _ = runner._load_group_context(copied)
                vote = runner.run_provider_on_group(
                    provider,
                    group,
                    copied,
                    (copied / "prompt.txt").read_text(),
                    letters,
                    options,
                    retries=0,
                    invocation_budget_s=0,
                    evidence_manifest=evidence,
                )
                ballot = dataclasses.asdict(vote)
                ballot["edge_set"] = sorted(vote.edge_set)
                record.update(
                    status="completed",
                    ballot=ballot,
                    evidence_id=evidence["evidence_id"],
                    evidence_pack_sha256=evidence["evidence_pack_sha256"],
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for key, value in os.environ.items():
                if ("KEY" in key or "TOKEN" in key or "SECRET" in key) and len(value) > 8:
                    message = message.replace(value, "[REDACTED]")
            record.update(status="error", error=message[:3500])
        record["wall_seconds"] = round(time.monotonic() - started, 2)
        write_json(destination, record)
        print(
            json.dumps(
                {k: record[k] for k in ["group_id", "provider", "status", "wall_seconds"]}
                | {
                    "choice": record.get("ballot", {}).get("choice"),
                    "error": record.get("error", "")[:500],
                }
            ),
            flush=True,
        )
        return record

    # At most two concurrent three-model packs; each ballot is blind to peers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(run_one, item, provider) for item in selected for provider in PANEL
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
