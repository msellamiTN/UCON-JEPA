"""G8.4.8.6 LOPO evaluation of joint action/state JEPA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import _metrics, _softmax
from joint_action_state_jepa import (
    JointActionHeadAdapter, _calibrated_joint_action_metrics, _fit_joint_model,
    _paired_joint_vs_posthoc_bootstrap,
)
from uncertain_action_dataset import AGENT_POLICIES, build_dataset, build_lopo_fold
from uncertain_action_jepa import (
    _action_features, _family_onehot, _transform, evaluate_horizon, fit_e1_e2_models,
)

EPS = 1e-12
HORIZONS = (1, 2, 4, 8)


def _calibrate(prob: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.where(prob > 0.0, np.log(np.maximum(prob, EPS)) / temperature, -np.inf)
    return _softmax(logits)


def _action_metrics_for_fold(
    fold: dict[str, Any],
    baseline_model,
    baseline_head,
    raw_head,
    scalers: dict[str, Any],
    joint_model,
    joint_temperature: float,
) -> dict[str, Any]:
    test = fold["test"]
    base_latent = _transform(test["context"], scalers["context_scaler"]) @ baseline_model.W_context
    base_features = _action_features(_family_onehot(test["context"]), base_latent)
    base_prob = _calibrate(baseline_head.predict_proba(base_features), scalers["temperature"])
    raw_prob = _calibrate(raw_head.predict_proba(test["context"]), scalers["raw_context_temperature"])
    joint_latent = _transform(test["context"], scalers["context_scaler"]) @ joint_model.W_context
    joint_prob = _calibrate(
        joint_model.predict_action_proba(joint_latent, _family_onehot(test["context"])),
        joint_temperature,
    )
    return {
        "posthoc_latent_head": _metrics(base_prob, test["action"]),
        "raw_context_diagnostic_only": _metrics(raw_prob, test["action"]),
        "joint_action_state_head": _metrics(joint_prob, test["action"]),
    }


def run_seed(
    seed: int,
    episodes_per_family_policy: int = 300,
    epochs: int = 180,
    action_weight: float = 0.5,
    mc_samples: int = 32,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    dataset = build_dataset(seed, episodes_per_family_policy, episode_length=8)
    folds: dict[str, Any] = {}
    for fold_index, policy in enumerate(AGENT_POLICIES):
        fold = build_lopo_fold(dataset, policy)
        for part in ("train", "validation", "calibration"):
            if policy in set(fold[part]["agent_policy"].tolist()):
                raise AssertionError(f"held-out policy {policy} leaked into {part}")
        if set(fold["test"]["agent_policy"].tolist()) != {policy}:
            raise AssertionError("LOPO test partition must contain exactly the held-out policy")

        fold_seed = seed + (fold_index + 1) * 10_000
        baseline_model, baseline_head, raw_head, scalers = fit_e1_e2_models(fold, fold_seed, epochs)
        joint_model, joint_temperature, joint_parameter_count = _fit_joint_model(
            fold, scalers, fold_seed, epochs, action_weight
        )
        joint_head = JointActionHeadAdapter(joint_model)
        joint_scalers = dict(scalers)
        joint_scalers["temperature"] = joint_temperature
        action_metrics = _action_metrics_for_fold(
            fold, baseline_model, baseline_head, raw_head, scalers,
            joint_model, joint_temperature,
        )

        horizons = {}
        for horizon in HORIZONS:
            baseline_eval = evaluate_horizon(
                baseline_model, baseline_head, fold["test"], horizon, scalers,
                mc_samples, fold_seed + horizon * 100, bootstrap_replicates,
            )
            joint_eval = evaluate_horizon(
                joint_model, joint_head, fold["test"], horizon, joint_scalers,
                mc_samples, fold_seed + horizon * 100 + 50_000, bootstrap_replicates,
            )
            horizons[str(horizon)] = {
                "posthoc_latent_head": {
                    "E1_known_action_privileged": baseline_eval["E1_known_action_privileged"],
                    "E2_predicted_action": baseline_eval["E2_predicted_action"],
                    "paired_episode_bootstrap_E2_minus_E1": baseline_eval["paired_episode_bootstrap_E2_minus_E1"],
                },
                "joint_action_state": {
                    "E1_known_action_privileged": joint_eval["E1_known_action_privileged"],
                    "E2_predicted_action": joint_eval["E2_predicted_action"],
                    "paired_episode_bootstrap_E2_minus_E1": joint_eval["paired_episode_bootstrap_E2_minus_E1"],
                },
                "paired_episode_bootstrap_joint_E2_minus_posthoc_E2": {
                    "risk_brier": _paired_joint_vs_posthoc_bootstrap(
                        joint_eval["episode_losses"]["E2_predicted_action"],
                        baseline_eval["episode_losses"]["E2_predicted_action"],
                        "risk_brier", fold_seed + horizon * 1000 + 1, bootstrap_replicates,
                    ),
                    "security_mse": _paired_joint_vs_posthoc_bootstrap(
                        joint_eval["episode_losses"]["E2_predicted_action"],
                        baseline_eval["episode_losses"]["E2_predicted_action"],
                        "security_mse", fold_seed + horizon * 1000 + 2, bootstrap_replicates,
                    ),
                },
            }
        folds[policy] = {
            "train_policies": sorted(set(fold["train"]["agent_policy"].tolist())),
            "validation_policies": sorted(set(fold["validation"]["agent_policy"].tolist())),
            "calibration_policies": sorted(set(fold["calibration"]["agent_policy"].tolist())),
            "test_policy": policy,
            "action_prediction": action_metrics,
            "models": {
                "posthoc_parameter_count": baseline_model.parameter_count(),
                "joint_parameter_count": joint_parameter_count,
                "action_weight": action_weight,
                "joint_action_temperature": joint_temperature,
                "joint_action_loss_final": joint_model.action_losses[-1],
                "joint_ema_target_distance": joint_model.ema_target_distance(),
            },
            "horizons": horizons,
        }
    return {"seed": seed, "dataset_manifest": dataset["manifest"], "lopo": folds}


def summarize_runs(runs: dict[str, Any]) -> dict[str, Any]:
    policies = list(AGENT_POLICIES)
    summary: dict[str, Any] = {}
    for policy in policies:
        summary[policy] = {"action_prediction": {}, "horizons": {}}
        for variant in ("posthoc_latent_head", "raw_context_diagnostic_only", "joint_action_state_head"):
            summary[policy]["action_prediction"][variant] = {
                metric: {
                    "mean": float(np.mean([r["lopo"][policy]["action_prediction"][variant][metric] for r in runs.values()])),
                    "sd_across_seeds": float(np.std([r["lopo"][policy]["action_prediction"][variant][metric] for r in runs.values()], ddof=1)),
                }
                for metric in ("nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin")
            }
        for horizon in HORIZONS:
            summary[policy]["horizons"][str(horizon)] = {}
            for variant in ("posthoc_latent_head", "joint_action_state"):
                rows = [r["lopo"][policy]["horizons"][str(horizon)][variant]["E2_predicted_action"] for r in runs.values()]
                summary[policy]["horizons"][str(horizon)][variant] = {
                    "violation": {
                        metric: {
                            "mean": float(np.mean([row["violation"][metric] for row in rows])),
                            "sd_across_seeds": float(np.std([row["violation"][metric] for row in rows], ddof=1)),
                        }
                        for metric in ("brier", "auroc", "auprc", "ece_10")
                        if all(row["violation"][metric] is not None for row in rows)
                    },
                    "security_mse": {
                        "mean": float(np.mean([row["security_mse"] for row in rows])),
                        "sd_across_seeds": float(np.std([row["security_mse"] for row in rows], ddof=1)),
                    },
                }
            diff = [
                r["lopo"][policy]["horizons"][str(horizon)]["paired_episode_bootstrap_joint_E2_minus_posthoc_E2"]["risk_brier"]["joint_minus_posthoc_mean"]
                for r in runs.values()
            ]
            summary[policy]["horizons"][str(horizon)]["joint_minus_posthoc_brier_difference"] = {
                "mean_across_seeds": float(np.mean(diff)),
                "sd_across_seeds": float(np.std(diff, ddof=1)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122")
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("joint-jepa-lopo-output"))
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    runs = {str(seed): run_seed(
        seed, args.episodes_per_family_policy, args.epochs,
        args.action_weight, args.mc_samples, args.bootstrap_replicates,
    ) for seed in seeds}
    result = {
        "benchmark": "G8.4.8.6 LOPO joint action/state JEPA",
        "seeds": seeds,
        "runs": runs,
        "summary_by_held_out_policy": summarize_runs(runs),
        "protocol": {
            "held_out_policies": list(AGENT_POLICIES),
            "training_validation_calibration": "held-out policy excluded from all three",
            "test": "held-out policy only",
            "action_weight": args.action_weight,
            "horizons": list(HORIZONS),
            "mc_samples": args.mc_samples,
            "episode_bootstrap_replicates": args.bootstrap_replicates,
        },
        "limitations": [
            "five synthetic behavior policies only",
            "LOPO folds have overlapping training populations and are not independent policy samples",
            "three seeds do not establish real-agent generalization",
            "Monte Carlo uncertainty covers sampled action sequences, not model epistemic uncertainty",
            "E1 remains a privileged known-action diagnostic",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "joint_jepa_lopo_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "policies": list(AGENT_POLICIES)}, indent=2))


if __name__ == "__main__":
    main()
