"""Recompute this trial's descriptive statistics without invoking any model."""

import collections
import csv
import json
import statistics
import sys
from pathlib import Path

from crosswalk.agent_labeling import stitch_runner as runner

csv.field_size_limit(sys.maxsize)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
selection = json.loads((HERE / "selection.json").read_text())
cases = []
policy_rows = []
times = collections.defaultdict(list)
stored_scores = collections.Counter()
undisputed_scores = collections.Counter()
historical = {}
for dataset in {c["dataset"] for c in selection["cases"]}:
    path = ROOT / "labels/votes" / f"dataset={dataset}" / "votes.csv"
    historical[dataset] = list(csv.DictReader(path.open())) if path.exists() else []

for case in selection["cases"]:
    gid = case["group_id"]
    ballots = []
    for provider in selection["panel"]:
        record = json.loads((HERE / f"{gid}.{provider['name']}.json").read_text())
        assert record["status"] == "completed", record
        ballot = record["ballot"]
        ballots.append(record)
        times[provider["name"]].append(record["wall_seconds"])
    hashes = {r["evidence_pack_sha256"] for r in ballots}
    assert len(hashes) == 1, (gid, "mixed evidence")
    asset_hashes = {
        json.loads(r["ballot"]["evidence_delivery"])["asset_set_sha256"] for r in ballots
    }
    assert len(asset_hashes) == 1, (gid, "mixed images")
    valid_ballots = [r for r in ballots if r["ballot"]["choice"] != "ABSTAIN"]
    counts = collections.Counter(r["ballot"]["choice"] for r in valid_ballots)
    assert counts, (gid, "no valid ballots")
    choice, count = counts.most_common(1)[0]
    # NONE/insufficient_evidence does not assert that all displayed edges are false.
    semantics = [
        (r["ballot"]["choice"], r["ballot"]["none_reason"], r["ballot"]["desired_edges"])
        for r in valid_ballots
    ]
    semantic_counts = collections.Counter(json.dumps(s, sort_keys=True) for s in semantics)
    old = [
        r
        for r in historical[case["dataset"]]
        if r["source_batch"] == case["source_batch"] and r["group_id"] == gid
    ]
    item = {
        "group_id": gid,
        "dataset": case["dataset"],
        "stratum": case["stratum"],
        "choice_counts": dict(counts),
        "n_valid": len(valid_ballots),
        "majority_choice": choice if count >= 2 else None,
        "n_agree_choice": count,
        "n_agree_semantics": max(semantic_counts.values()),
        "current": [
            {
                k: r["ballot"][k]
                for k in [
                    "provider",
                    "model",
                    "choice",
                    "confidence",
                    "none_reason",
                    "desired_edges",
                    "reasoning",
                    "error",
                ]
            }
            for r in ballots
        ],
        "historical_same_batch": [
            {
                k: r.get(k, "")
                for k in [
                    "provider",
                    "model",
                    "choice",
                    "confidence",
                    "none_reason",
                    "evidence_pack_sha256",
                ]
            }
            for r in old
        ],
        "historical_hash_matches": sum(r.get("evidence_pack_sha256") in hashes for r in old),
        "human_expected": case["human_expected"],
        "disputed_human_label": gid == "66e22055",
    }
    expected = case["human_expected"]
    if expected:
        for record in ballots:
            b = record["ballot"]
            correct = b["choice"] != "ABSTAIN" and sorted(map(tuple, b["edge_set"])) == sorted(
                map(tuple, expected["edges"])
            )
            if not expected["edges"]:
                correct = b["choice"] == "NONE" and b["none_reason"] == "all_edges_no_match"
            stored_scores[b["provider"]] += int(correct)
            if not item["disputed_human_label"]:
                undisputed_scores[b["provider"]] += int(correct)
    cases.append(item)
    replay_votes = []
    for record in ballots:
        fields = dict(record["ballot"])
        fields["edge_set"] = frozenset(map(tuple, fields["edge_set"]))
        replay_votes.append(runner.Vote(**fields))
    metadata = runner._load_group_context(ROOT / case["source_path"])[2]
    ref_classes, target_classes = runner._segment_class_maps(metadata)
    base = runner.compute_consensus(replay_votes)
    decision = runner.compute_consensus(
        replay_votes,
        edge_classes=runner._edge_classes_for(base.edge_set, ref_classes, target_classes),
        n_candidate_edges=runner.candidate_edge_count(metadata),
        min_voter_confidence=runner.settings.stitch_min_voter_confidence,
    )
    policy_rows.append(
        {
            "group_id": gid,
            "choice": decision.choice,
            "routing": decision.routing,
            "route_reason": decision.route_reason,
            "n_candidate_edges": runner.candidate_edge_count(metadata),
            "min_confidence": min(v.confidence for v in replay_votes),
        }
    )

result = {
    "purpose": "Descriptive diagnostic; purposive small sample, shared historical prompts, no fresh human adjudication.",
    "models_requested": selection["panel"],
    "completed_valid_ballots": sum(c["n_valid"] for c in cases),
    "abstain_ballots": sum(len(selection["panel"]) - c["n_valid"] for c in cases),
    "packs": len(cases),
    "datasets": len({c["dataset"] for c in cases}),
    "unanimous_choice_packs": sum(c["n_agree_choice"] == 3 for c in cases),
    "unanimous_semantics_packs": sum(c["n_agree_semantics"] == 3 for c in cases),
    "two_one_choice_packs": sum(c["n_valid"] == 3 and c["n_agree_choice"] == 2 for c in cases),
    "packs_with_incomplete_quorum": sum(c["n_valid"] != 3 for c in cases),
    "existing_policy_routing": dict(collections.Counter(r["routing"] for r in policy_rows)),
    "against_five_stored_human_rows": dict(stored_scores),
    "against_four_after_excluding_disputed_sydney_row": dict(undisputed_scores),
    "latency_seconds_per_ballot_including_cli_overhead": {
        p: {
            "median": round(statistics.median(v), 2),
            "min": min(v),
            "max": max(v),
            "sum": round(sum(v), 2),
        }
        for p, v in times.items()
    },
    "bootstrap_note": "Three earlier CLI calls returned but their answers were lost to a local capture bug. Error records and their selection manifest are retained in bootstrap_harness_error; they are not scored. Total provider calls: 33, retained ballots: 30.",
    "cost": "Actual billed cost and token usage were not captured; no dollar estimate is asserted.",
    "cases": cases,
}
(HERE / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
(HERE / "existing_policy_replay.json").write_text(
    json.dumps(
        {"confidence_floor": runner.settings.stitch_min_voter_confidence, "rows": policy_rows},
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ["cases", "models_requested"]}, indent=2
    )
)
