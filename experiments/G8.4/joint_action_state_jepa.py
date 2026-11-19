"""G8.4.8.5 joint action/state JEPA vs post-hoc latent action head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import _fit_temperature, _metrics, _softmax
from jepa_numpy import JEPAConfig
from jepa_joint_action import JointActionStateJEPA
from uncertain_action_dataset import build_dataset
from uncertain_action_jepa import (
    _family_onehot, _transform, evaluate_horizon, fit_e1_e2_models,
)

EPS = 1e-12
HORIZONS = (1, 2, 4, 8)


class JointActionHeadAdapter:
    """Adapt joint JEPA action logits to the E2 rollout's past-only policy API."""
    def __init__(self, model: JointActionStateJEPA):
        self.model = model

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        return self.model.predict_action_proba(x[:, 3:], x[:, :3])


def _fit_joint_model(
    dataset: dict[str, Any],
    scalers: dict[str, Any],
    seed: int,
    epochs: int,
    action_weight: float,
) -> tuple[JointActionStateJEPA, float, int]:
    train, validation, calibration = (dataset[k] for k in ("train", "validation", "calibration"))
    context_scaler = scalers["context_scaler"]
    future_scaler = scalers["future_scaler"]
    action_scaler = scalers["action_scaler"]
    xc = _transform(train["context"], context_scaler)
    xa = _transform(train["action"], action_scaler)
    yf = _transform(train["future"], future_scaler)
    config = JEPAConfig(
        latent_dim=12, learning_rate=0.008, epochs=epochs, batch_size=128,
        ema_tau=0.99, variance_floor=0.2, variance_weight=0.1,
        security_weight=0.5, seed=seed,
    )
    model = JointActionStateJEPA(
        xc.shape[1], xa.shape[1], yf.shape[1], train["security"].shape[1],
        config, action_weight=action_weight,
    ).fit(
        xc, xa, yf, train["security"], train["action"],
        np.argmax(train["context"][:, :3], axis=1),
    )
    cal_latent = _transform(calibration["context"], context_scaler) @ model.W_context
    cal_prob = model.predict_action_proba(cal_latent, _family_onehot(calibration["context"]))
    temperature = _fit_temperature(cal_prob, calibration["action"])
    return model, temperature, model.parameter_count()


def _calibrated_joint_action_metrics(
    model: JointActionStateJEPA,
    dataset: dict[str, Any],
    scalers: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    test = dataset["test"]
    latent = _transform(test["context"], scalers["context_scaler"]) @ model.W_context
    prob = model.predict_action_proba(latent, _family_onehot(test["context"]))
    logits = np.where(prob > 0.0, np.log(np.maximum(prob, EPS)) / temperature, -np.inf)
    return _metrics(_softmax(logits), test["action"])


def _summary(runs: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS:
        output[str(horizon)] = {}
        for variant in ("posthoc_latent_head", "joint_action_state"):
            output[str(horizon)][variant] = {}
            for method in ("E1_known_action_privileged", "E2_predicted_action"):
                rows = [r["horizons"][str(horizon)][variant][method] for r in runs.values()]
                output[str(horizon)][variant][method] = {
                    metric: {
                        "mean": float(np.mean([x[metric] for x in rows])),
                        "sd_across_seeds": float(np.std([x[metric] for x in rows], ddof=1)),
                    }
                    for metric in ("security_mse", "terminal_state_mse")
                }
                output[str(horizon)][variant][method]["violation"] = {
                    metric: {
                        "mean": float(np.mean([x["violation"][metric] for x in rows])),
                        "sd_across_seeds": float(np.std([x["violation"][metric] for x in rows], ddof=1)),
                    }
                    for metric in ("brier", "auroc", "auprc", "ece_10")
                    if all(x["violation"][metric] is not None for x in rows)
                }
    return output


def run_seed(
    seed: int,
    episodes_per_family_policy: int = 300,
    epochs: int = 180,
    action_weight: float = 0.5,
    mc_samples: int = 32,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    dataset = build_dataset(seed, episodes_per_family_policy, episode_length=8)
    baseline_model, baseline_head, raw_head, scalers = fit_e1_e2_models(dataset, seed, epochs)
    joint_model, joint_temperature, joint_parameter_count = _fit_joint_model(
        dataset, scalers, seed, epochs, action_weight
    )
    joint_head = JointActionHeadAdapter(joint_model)
    joint_scalers = dict(scalers)
    joint_scalers["temperature"] = joint_temperature

    test = dataset["test"]
    baseline_action_metrics = baseline_head.predict_proba(
        np.concatenate((
            _family_onehot(test["context"]),
            _transform(test["context"], scalers["context_scaler"]) @ baseline_model.W_context,
        ), axis=1)
    )
    baseline_action_metrics = _metrics(baseline_action_metrics, test["action"])
    raw_action_metrics = _metrics(raw_head.predict_proba(test["context"]), test["action"])
    joint_action_metrics = _calibrated_joint_action_metrics(
        joint_model, dataset, scalers, joint_temperature
    )

    horizons = {}
    for horizon in HORIZONS:
        baseline_eval = evaluate_horizon(
            baseline_model, baseline_head, test, horizon, scalers,
            mc_samples, seed + horizon * 100, bootstrap_replicates,
        )
        joint_eval = evaluate_horizon(
            joint_model, joint_head, test, horizon, joint_scalers,
            mc_samples, seed + horizon * 100 + 50_000, bootstrap_replicates,
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
        }
    return {
        "seed": seed,
        "dataset_manifest": dataset["manifest"],
        "action_prediction": {
            "posthoc_latent_head": baseline_action_metrics,
            "raw_context_diagnostic_only": raw_action_metrics,
            "joint_action_state_head": joint_action_metrics,
        },
        "models": {
            "posthoc_jepa_parameter_count": baseline_model.parameter_count(),
            "joint_jepa_parameter_count": joint_parameter_count,
            "action_weight": action_weight,
            "joint_action_head_temperature": joint_temperature,
            "joint_action_loss_final": joint_model.action_losses[-1],
            "joint_action_loss_min": float(np.min(joint_model.action_losses)),
            "joint_ema_target_distance": joint_model.ema_target_distance(),
            "joint_final_latent_loss": joint_model.history.latent_loss[-1],
            "joint_final_security_loss": joint_model.history.security_loss[-1],
        },
        "horizons": horizons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122")
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("joint-action-state-jepa-output"))
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    runs = {str(seed): run_seed(
        seed, args.episodes_per_family_policy, args.epochs,
        args.action_weight, args.mc_samples, args.bootstrap_replicates,
    ) for seed in seeds}
    result = {
        "benchmark": "G8.4.8.5 joint action/state JEPA",
        "seeds": seeds,
        "runs": runs,
        "summary": _summary(runs),
        "objective": {
            "latent_prediction": "JEPA predictor matches future observation EMA-target latents",
            "security": "security decoder predicts transition security labels",
            "action": "auxiliary cross-entropy predicts the current action from context latent",
            "total": "latent + variance + security + action_weight * action_cross_entropy",
        },
        "limitations": [
            "linear reference JEPA and hand-designed synthetic behavior policies",
            "action head is trained jointly but future policy rollout still evaluates the head on predicted target latents",
            "three seeds do not establish real-agent generalization",
            "Monte Carlo uncertainty covers sampled action trajectories, not model epistemic uncertainty",
            "E1 remains a privileged known-action diagnostic",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "joint_action_state_jepa_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "horizons": list(map(str, HORIZONS))}, indent=2))


if __name__ == "__main__":
    main()
