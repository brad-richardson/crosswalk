"""Controlled weak-supervision comparison on a frozen human resolver benchmark."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from crosswalk.agent_labeling.stitch_provenance import sha256_file
from crosswalk.resolver.evaluate import _eval_from_predictions, paired_group_bootstrap
from crosswalk.resolver.evaluation_set import connect_frozen_scopes, scope_tokens
from crosswalk.resolver.extract import (
    build_candidate_context,
    load_candidates_parquet,
    load_sidecar_groups,
)
from crosswalk.resolver.round2 import (
    TRAIN_LABEL_COLUMN,
    featurize_extended,
    select_group_predictions,
)
from crosswalk.resolver.train import predict_keep_probability, train_model


def prepare_verified_weak_context(
    weak: pd.DataFrame,
    token_index: dict[str, str],
    human_audit: list[dict],
    *,
    root: Path = Path("."),
) -> tuple[pd.DataFrame, dict]:
    """Join observations only to byte-identical originating source snapshots.

    Full eligible parent context is featurized before masking any edge or tier.
    Full context also extends leakage exclusion: a human segment seen only in an
    unvoted candidate must still block its connected weak observations.
    """
    frames, snapshots = [], []
    audit = Counter(input_rows=len(weak))
    hashes = {}
    for (dataset, snapshot), part in weak.groupby(["dataset_id", "source_snapshot_id"], sort=True):
        artifacts = json.loads(part.iloc[0]["source_artifacts"])
        verified = {}
        for kind, record in artifacts.items():
            if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                continue
            path = root / record["path"]
            if path not in hashes:
                hashes[path] = sha256_file(path) if path.is_file() else None
            verified[kind] = hashes[path] == record["sha256"]
        record = {
            "dataset_id": dataset,
            "source_snapshot_id": snapshot,
            "rows": len(part),
            "verified": verified,
        }
        snapshots.append(record)
        if (
            not verified.get("groups_sidecar")
            or not verified.get("candidates_parquet")
            or not all(verified.values())
        ):
            record["status"] = "missing_or_changed_source_artifact"
            audit[record["status"]] += len(part)
            continue
        groups = load_sidecar_groups(root / artifacts["groups_sidecar"]["path"])
        required = set(part["parent_group_id"])
        selected_groups = [
            g for g in groups if g["group_id"] in required and g.get("candidate_edges")
        ]
        del groups
        candidates = load_candidates_parquet(root / artifacts["candidates_parquet"]["path"])
        context = build_candidate_context(selected_groups, dataset, candidates_df=candidates)
        del candidates
        if context.empty:
            record["status"] = "missing_complete_parent_context"
            audit[record["status"]] += len(part)
            continue
        context = featurize_extended(context)
        tokens = {}
        for group in selected_groups:
            rows = context[context["group_id"] == group["group_id"]]
            tokens[group["group_id"]] = json.dumps(
                sorted(
                    scope_tokens(
                        dataset,
                        [group["group_id"]],
                        set(group.get("ref_ids", [])) | set(rows["ref_id"]),
                        set(group.get("target_ids", [])) | set(rows["target_id"]),
                    )
                )
            )
        observed = part.copy()
        observed["source_group_id"] = observed["group_id"]
        observed["group_id"] = observed["parent_group_id"]
        observed["context_scope_tokens"] = observed["group_id"].map(tokens)
        joined = observed.merge(
            context,
            on=["dataset_id", "group_id", "ref_id", "target_id"],
            how="inner",
            validate="many_to_one",
            suffixes=("_observation", ""),
        )
        record.update(status="verified", joined_rows=len(joined))
        audit["observations_outside_eligible_parent_context"] += len(part) - len(joined)
        frames.append(joined)
    if not frames:
        raise ValueError("No weak observations retain a verified source snapshot")
    frame = pd.concat(frames, ignore_index=True)
    blocked = {token_index[token] for record in human_audit for token in record["scope_tokens"]}
    expanded = connect_frozen_scopes(frame, token_index, blocked)
    frame["component_id"] = expanded["component_id"]
    frame["excluded_human_scope"] = expanded["excluded_human_scope"]
    audit["verified_context_rows"] = len(frame)
    audit["excluded_after_full_context"] = int(frame["excluded_human_scope"].sum())
    # Unknowns contributed to exclusion above; they never reach the loss.
    frame = frame[
        ~frame["excluded_human_scope"]
        & frame["known"]
        & frame["vote_tier"].isin(["unanimous", "majority"])
    ].copy()
    frame[TRAIN_LABEL_COLUMN] = frame["soft_keep"].astype(float)
    frame["keep"] = frame[TRAIN_LABEL_COLUMN]  # training only; never evaluation truth
    frame["labeler"] = "panel"
    frame["provenance"] = "scoped_weak_vote"
    audit["eligible_training_rows"] = len(frame)
    return frame, {
        "counts": dict(audit),
        "snapshots": snapshots,
        "tiers": frame["vote_tier"].value_counts().to_dict(),
        "components": int(frame["component_id"].nunique()),
    }


def weight_weak_rows(frame: pd.DataFrame, policy: str, settings: dict) -> pd.DataFrame:
    """Weight each physical pair once across versions; cap connected parents.

    Replicated child observations or source versions split their existing pair
    budget, rather than increasing it. The budget is the pair's strongest tier
    weight, so admitting a lower tier never lowers a pair that already had a
    unanimous version. A component cap is stronger than a parent cap when
    shared segments connect otherwise separate parents.
    """
    if policy not in {"human_only", "human_unanimous", "human_majority"}:
        raise ValueError(f"Unknown supervision policy {policy}")
    allowed = [] if policy == "human_only" else ["unanimous"]
    if policy == "human_majority":
        allowed.append("majority")
    out = frame[
        frame["vote_tier"].isin(allowed) & frame["known"] & ~frame["excluded_human_scope"]
    ].copy()
    weights = {"unanimous": settings["unanimous_weight"], "majority": settings["majority_weight"]}
    cap = settings["weak_component_weight_cap"]
    if any(not np.isfinite(v) or v <= 0 for v in [*weights.values(), cap]):
        raise ValueError("Weak weights and component cap must be finite and positive")
    if out.empty:
        out["sample_weight"] = pd.Series(dtype=float)
        return out
    out["sample_weight"] = out["vote_tier"].map(weights).astype(float)
    by_pair = out.groupby(["dataset_id", "ref_id", "target_id"])["sample_weight"]
    out["sample_weight"] = by_pair.transform("max") / by_pair.transform("size")
    mass = out.groupby("component_id")["sample_weight"].transform("sum")
    out["sample_weight"] *= np.minimum(1.0, cap / mass)
    return out


def run_frozen_comparison(
    gold: pd.DataFrame, weak: pd.DataFrame, settings: dict
) -> tuple[pd.DataFrame, dict, dict]:
    """Vary supervision only: same folds, features, seeds, objective and selector."""
    features = settings["features"]
    if (
        settings["objective"] != "reg:logistic"
        or settings["hyperparameter_search"]
        or settings["promotion"]
    ):
        raise ValueError("This harness only supports the preregistered non-promoting comparison")
    if gold.groupby("component_id")["fold"].nunique().max() != 1:
        raise ValueError("A human component crosses evaluation folds")
    if gold["labeler"].ne("brad").any() or not gold["keep"].isin([0, 1]).all():
        raise ValueError("Evaluation must contain only hard human truth")
    predictions = gold.copy().reset_index(drop=True)
    baseline = predictions["selected"].astype(int).to_numpy()
    predictions["optimizer_prediction"] = baseline
    results = {
        "optimizer": {"metrics": asdict(_eval_from_predictions("optimizer", predictions, baseline))}
    }
    final_models = {}
    for policy in settings["policies"]:
        panel = weight_weak_rows(weak, policy, settings)
        probabilities = np.full((len(settings["seeds"]), len(predictions)), np.nan)
        folds = []
        for seed_index, seed in enumerate(settings["seeds"]):
            for fold in sorted(predictions["fold"].unique()):
                test = predictions["fold"] == fold
                train = predictions.loc[~test].copy()
                train["sample_weight"] = 1.0
                model = train_model(
                    train, features, soft_extra=panel, seed=seed, force_soft_objective=True
                )
                probabilities[seed_index, test] = predict_keep_probability(
                    model, predictions.loc[test, features].to_numpy(dtype=float)
                )
                folds.append(
                    {
                        "seed": seed,
                        "fold": int(fold),
                        "human_train_edges": len(train),
                        "human_test_edges": int(test.sum()),
                        "weak_edges": len(panel),
                        "weak_weight_mass": float(panel["sample_weight"].sum()),
                    }
                )
        if not np.isfinite(probabilities).all():
            raise ValueError("Incomplete held-out predictions")
        mean = probabilities.mean(axis=0)
        pred = select_group_predictions(predictions, mean, selector=settings["selector"])
        predictions[f"{policy}_probability"] = mean
        predictions[f"{policy}_prediction"] = pred
        results[policy] = {
            "metrics": asdict(_eval_from_predictions(policy, predictions, pred)),
            "seed_metrics": [
                asdict(
                    _eval_from_predictions(
                        f"{policy}:{seed}",
                        predictions,
                        select_group_predictions(
                            predictions, probabilities[i], selector=settings["selector"]
                        ),
                    )
                )
                for i, seed in enumerate(settings["seeds"])
            ],
            "folds": folds,
            "weak_rows": len(panel),
            "fractional_targets": int((~panel[TRAIN_LABEL_COLUMN].isin([0, 1])).sum()),
            "weak_weight_mass": float(panel["sample_weight"].sum()),
            "max_component_weight": float(
                panel.groupby("component_id")["sample_weight"].sum().max()
            )
            if len(panel)
            else 0.0,
            "paired_vs_optimizer": paired_group_bootstrap(
                predictions, pred, baseline, resample_columns=["component_id"], seed=1729
            ),
            "slices": {},
        }
        for name, mask in {
            "deanchored": ~predictions["anchored"],
            **{ds: predictions["dataset_id"] == ds for ds in predictions["dataset_id"].unique()},
        }.items():
            if mask.any():
                results[policy]["slices"][name] = {
                    "candidate": asdict(
                        _eval_from_predictions(policy, predictions.loc[mask], pred[mask])
                    ),
                    "optimizer": asdict(
                        _eval_from_predictions("optimizer", predictions.loc[mask], baseline[mask])
                    ),
                }
        # Fit isolated artifacts AFTER evaluation. Their training predictions
        # are never used as benchmark predictions.
        all_human = gold.copy()
        all_human["sample_weight"] = 1.0
        final_models[policy] = train_model(
            all_human,
            features,
            soft_extra=panel,
            seed=settings["seeds"][0],
            force_soft_objective=True,
        )
    for candidate, baseline_policy in [
        ("human_unanimous", "human_only"),
        ("human_majority", "human_unanimous"),
        ("human_majority", "human_only"),
    ]:
        results[candidate][f"paired_vs_{baseline_policy}"] = paired_group_bootstrap(
            predictions,
            predictions[f"{candidate}_prediction"].to_numpy(),
            predictions[f"{baseline_policy}_prediction"].to_numpy(),
            resample_columns=["component_id"],
            seed=1729,
        )
    return predictions, results, final_models
