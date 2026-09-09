#!/usr/bin/env python
"""Run the frozen human-only / unanimous / majority resolver comparison."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
from loguru import logger

from crosswalk.agent_labeling.stitch_provenance import sha256_file
from crosswalk.resolver.evaluation_set import load_frozen_evaluation
from crosswalk.resolver.supervision_experiment import (
    prepare_verified_weak_context,
    run_frozen_comparison,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "report.json").exists():
        raise ValueError("Comparison already recorded; use a new directory for another experiment")
    manifest, gold, weak, token_index = load_frozen_evaluation(args.evaluation)
    logger.disable("crosswalk.resolver.extract")
    audit = json.loads((args.evaluation / "audit.json").read_text())
    prepared, context_audit = prepare_verified_weak_context(
        weak, token_index, audit, root=args.root
    )
    print(json.dumps(context_audit["counts"], indent=2), flush=True)
    if not len(prepared):
        raise ValueError("No weak rows survive verification and human-scope exclusion")
    args.out.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.out / "weak_training.parquet", index=False)
    predictions, results, models = run_frozen_comparison(gold, prepared, manifest["experiment"])
    predictions.to_parquet(args.out / "predictions.parquet", index=False)
    manifest_sha = sha256_file(args.evaluation / "manifest.json")
    for policy, model in models.items():
        joblib.dump(
            {
                "model": model,
                "feature_columns": manifest["experiment"]["features"],
                "evaluation_manifest_sha256": manifest_sha,
                "policy": policy,
                "research_only": True,
                "promotion": False,
            },
            args.out / f"{policy}.joblib",
        )
    report = {
        "evaluation_manifest_sha256": manifest_sha,
        "experiment": manifest["experiment"],
        "context_audit": context_audit,
        "results": results,
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["xgboost", "pandas", "numpy", "scikit-learn"]},
        },
        "code": {
            str(p): sha256_file(p)
            for p in [
                Path(__file__),
                *[
                    Path("src/crosswalk/resolver") / name
                    for name in [
                        "supervision_experiment.py",
                        "train.py",
                        "extract.py",
                        "features.py",
                        "round2.py",
                        "evaluate.py",
                        "evaluation_set.py",
                    ]
                ],
            ]
        },
        "outputs": {p.name: sha256_file(p) for p in sorted(args.out.iterdir()) if p.is_file()},
        "promotion": False,
        "limitations": manifest["limitations"]
        + [
            "Three fixed model seeds share one frozen partition; seeds are not independent evaluation samples.",
            "Full candidate context may conservatively exclude more weak observations than the initial displayed-edge audit.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = [
        "# Frozen stitch supervision comparison",
        "",
        f"Evaluation manifest: `{manifest_sha}`. No new human labels; no model promotion.",
        "",
        "| Policy | Edge F1 | Exact groups | Sliver-filtered F1 | Weak edges | Weak weight |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, result in results.items():
        m = result["metrics"]
        rows.append(
            f"| {policy} | {m['f1']:.4f} | {m['group_exact_rate']:.4f} | {m['f1_filtered']:.4f} | {result.get('weak_rows', 0)} | {result.get('weak_weight_mass', 0):.2f} |"
        )
    rows += [
        "",
        "All learned policies use the same frozen folds, 33 features, three seeds, logistic objective, and expected-F1 selector. Group/component bootstrap intervals and all dataset/deanchored slices are in report.json. Weak targets remain fractional; duplicates share their pair budget and connected parents have a capped total weight.",
        "",
        "These are exploratory results on 14 existing human groups. The optimizer baseline is its fixed selection in each verified evaluation snapshot. The saved models are isolated research artifacts fitted after the held-out comparison; they are not promoted or installed in production.",
        "",
    ]
    (args.out / "report.md").write_text("\n".join(rows))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
