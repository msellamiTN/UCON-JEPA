"""G8.4.8.7.3 one-step JEPA action representation under controlled rho shift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import (
    FirstOrderMarkovPredictor, HistoricalFrequencyPredictor,
    MajorityActionPredictor, PastOnlySoftmaxPredictor,
    _fit_temperature, _metrics, _softmax,
)
from joint_action_state_jepa import _fit_joint_model
from policy_parameter_dataset import (
    POLICY_ARCHETYPES, build_dataset, write_dataset,
)
from policy_shift_action_baselines import previous_actions
from uncertain_action_jepa import (
    _action_features, _family_onehot, _transform, fit_e1_e2_models,
)

EPS = 1e-12
BASELINE_NAMES = (
    "majority", "historical_frequency", "first_order_markov", "past_only_softmax"
)
MODEL_NAMES = BASELINE_NAMES + (
    "raw_context_diagnostic",
    "posthoc_jepa_latent",
    "joint_action_state_jepa",
)


def _calibrate(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.where(
        probability > 0.0,
        np.log(np.maximum(probability, EPS)) / temperature,
        -np.inf,
    )
    return _softmax(logits)


def _episode_metric_losses(
    part: dict[str, np.ndarray], probability: np.ndarray
) -> dict[str, dict[str, float]]:
    truth = np.argmax(part["action"], axis=1)
    nll = -np.log(np.clip(probability[np.arange(len(truth)), truth], EPS, 1.0))
    onehot = np.zeros_like(probability)
    onehot[np.arange(len(truth)), truth] = 1.0
    brier = np.square(probability - onehot).sum(axis=1)
    groups: dict[str, list[int]] = {}
    for i, episode_id in enumerate(part["episode_id"].tolist()):
        groups.setdefault(str(episode_id), []).append(i)
    return {
        eid: {"nll": float(nll[idx].mean()), "brier": float(brier[idx].mean())}
        for eid, idx in groups.items()
    }


def _paired_bootstrap(
    model_losses: dict[str, dict[str, float]],
    reference_losses: dict[str, dict[str, float]],
    metric: str,
    seed: int,
    replicates: int,
    reference_name: str,
) -> dict[str, Any]:
    ids = sorted(set(model_losses) & set(reference_losses))
    if len(ids) != len(model_losses) or len(ids) != len(reference_losses):
        raise AssertionError("paired model comparison requires identical complete episode sets")
    differences = np.asarray([
        model_losses[eid][metric] - reference_losses[eid][metric] for eid in ids
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(replicates, len(ids)))
    draws = differences[indices].mean(axis=1)
    return {
        "unit": "whole episode",
        "episodes": len(ids),
        "replicates": replicates,
        "model_minus_reference_mean": float(differences.mean()),
        "reference_model": reference_name,
        "ci95_percentile": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "negative_favours_model": True,
    }


def _fit_baselines(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    return {
        "majority": MajorityActionPredictor().fit(train["context"], train["action"]),
        "historical_frequency": HistoricalFrequencyPredictor().fit(
            train["context"], train["action"]
        ),
        "first_order_markov": FirstOrderMarkovPredictor().fit(
            train["context"], train["action"], train["episode_id"], train["step"]
        ),
        "past_only_softmax": PastOnlySoftmaxPredictor(
            max_epochs=max_epochs, patience=patience
        ).fit(
            train["context"], train["action"],
            validation["context"], validation["action"],
        ),
    }


def _predict_baseline(
    name: str, model: Any, part: dict[str, np.ndarray]
) -> np.ndarray:
    if name == "first_order_markov":
        return model.predict_proba(part["context"], previous_actions(part))
    return model.predict_proba(part["context"])


def _group_metrics(
    part: dict[str, np.ndarray], probabilities: dict[str, np.ndarray]
) -> dict[str, Any]:
    grouped = {}
    for archetype in POLICY_ARCHETYPES:
        grouped[archetype] = {}
        for family in sorted(set(part["family"].tolist())):
            mask = (part["agent_policy"] == archetype) & (part["family"] == family)
            if np.any(mask):
                grouped[archetype][family] = {
                    name: _metrics(prob[mask], part["action"][mask])
                    for name, prob in probabilities.items()
                }
    return grouped


def run_seed(
    seed: int,
    episodes_per_cell: int = 300,
    jepa_epochs: int = 120,
    baseline_epochs: int = 150,
    baseline_patience: int = 15,
    action_weight: float = 0.5,
    bootstrap_replicates: int = 2000,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
    train, validation, calibration = (
        dataset[name] for name in ("train", "validation", "calibration")
    )
    baseline_models = _fit_baselines(
        train, validation, baseline_epochs, baseline_patience
    )
    posthoc_jepa, posthoc_action_head, raw_context_head, posthoc_scalers = (
        fit_e1_e2_models(dataset, seed=seed, epochs=jepa_epochs)
    )
    joint_jepa, joint_temperature, joint_parameter_count = _fit_joint_model(
        dataset, posthoc_scalers, seed, jepa_epochs, action_weight
    )

    # Fit each baseline temperature on nominal-support calibration only.
    temperatures = {}
    calibration_raw = {}
    for name, model in baseline_models.items():
        p = _predict_baseline(name, model, calibration)
        calibration_raw[name] = p
        temperatures[name] = _fit_temperature(p, calibration["action"])
    temperatures["posthoc_jepa_latent"] = posthoc_scalers["temperature"]
    temperatures["raw_context_diagnostic"] = posthoc_scalers["raw_context_temperature"]
    temperatures["joint_action_state_jepa"] = joint_temperature

    tests = {}
    for split_name in ("test_shifted", "test_reverse"):
        part = dataset[split_name]
        raw_probabilities = {
            name: _predict_baseline(name, model, part)
            for name, model in baseline_models.items()
        }
        context_scaled = _transform(part["context"], posthoc_scalers["context_scaler"])
        posthoc_latent = context_scaled @ posthoc_jepa.W_context
        posthoc_features = _action_features(_family_onehot(part["context"]), posthoc_latent)
        raw_probabilities["posthoc_jepa_latent"] = posthoc_action_head.predict_proba(posthoc_features)
        raw_probabilities["raw_context_diagnostic"] = raw_context_head.predict_proba(part["context"])
        joint_latent = context_scaled @ joint_jepa.W_context
        raw_probabilities["joint_action_state_jepa"] = joint_jepa.predict_action_proba(
            joint_latent, _family_onehot(part["context"])
        )
        calibrated = {
            name: _calibrate(raw_probabilities[name], temperatures[name])
            for name in MODEL_NAMES
        }
        metrics = {
            name: _metrics(calibrated[name], part["action"])
            for name in MODEL_NAMES
        }
        grouped = _group_metrics(part, calibrated)
        losses = {
            name: _episode_metric_losses(part, calibrated[name])
            for name in MODEL_NAMES
        }
        paired = {}
        references = ("historical_frequency", "past_only_softmax", "posthoc_jepa_latent")
        for name in MODEL_NAMES:
            if name in references:
                continue
            paired[name] = {}
            for reference in references:
                paired[name][reference] = {
                    metric: _paired_bootstrap(
                        losses[name], losses[reference], metric,
                        seed + (0 if split_name == "test_shifted" else 200_000)
                        + len(name) * 1000 + len(reference) * 10
                        + (0 if metric == "nll" else 1),
                        bootstrap_replicates, reference,
                    )
                    for metric in ("nll", "brier")
                }
        tests[split_name] = {
            "rho_values": sorted(set(float(x) for x in part["rho"].tolist())),
            "episodes": int(len(set(part["episode_id"].tolist()))),
            "transitions": int(len(part["action"])),
            "calibrated_metrics": metrics,
            "calibrated_by_archetype_family": grouped,
            "paired_episode_bootstrap": paired,
        }

    model_info = {
        "posthoc_jepa_parameter_count": posthoc_jepa.parameter_count(),
        "joint_jepa_parameter_count": joint_parameter_count,
        "action_weight": action_weight,
        "joint_action_loss_final": joint_jepa.action_losses[-1],
        "joint_ema_target_distance": joint_jepa.ema_target_distance(),
        "posthoc_action_head_selected_epoch": posthoc_scalers["action_head_selected_epoch"],
        "raw_context_head_selected_epoch": raw_context_head.best_epoch,
        "baseline_softmax_selected_epoch": baseline_models["past_only_softmax"].best_epoch,
        "temperatures_fitted_on": "nominal-support calibration partition only",
    }
    if out_dir is not None:
        write_dataset(dataset, out_dir / f"dataset-{seed}")
    return {
        "seed": seed,
        "dataset_manifest": dataset["manifest"],
        "models": model_info,
        "tests": tests,
    }


def summarize(runs: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for split_name in ("test_shifted", "test_reverse"):
        summary[split_name] = {}
        for model in MODEL_NAMES:
            summary[split_name][model] = {}
            for metric in ("nll", "top1_accuracy", "top2_accuracy",
                           "multiclass_brier", "top_label_ece_10bin"):
                values = [
                    run["tests"][split_name]["calibrated_metrics"][model][metric]
                    for run in runs.values()
                ]
                summary[split_name][model][metric] = {
                    "mean_across_seeds": float(np.mean(values)),
                    "sd_across_seeds": float(np.std(values, ddof=1)),
                }
        summary[split_name]["equal_archetype_macro"] = {}
        for model in MODEL_NAMES:
            summary[split_name]["equal_archetype_macro"][model] = {}
            for metric in ("nll", "top1_accuracy", "top2_accuracy",
                           "multiclass_brier", "top_label_ece_10bin"):
                per_seed = []
                for run in runs.values():
                    grouped = run["tests"][split_name]["calibrated_by_archetype_family"]
                    archetype_means = [
                        np.mean([grouped[policy][family][model][metric]
                                 for family in grouped[policy]])
                        for policy in POLICY_ARCHETYPES
                    ]
                    per_seed.append(float(np.mean(archetype_means)))
                summary[split_name]["equal_archetype_macro"][model][metric] = {
                    "mean_across_seeds": float(np.mean(per_seed)),
                    "sd_across_seeds": float(np.std(per_seed, ddof=1)),
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--jepa-epochs", type=int, default=120)
    parser.add_argument("--baseline-epochs", type=int, default=150)
    parser.add_argument("--baseline-patience", type=int, default=15)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("policy-shift-jepa-action-output"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {
        str(seed): run_seed(
            seed, args.episodes_per_cell, args.jepa_epochs,
            args.baseline_epochs, args.baseline_patience,
            args.action_weight, args.bootstrap_replicates, args.out,
        )
        for seed in seeds
    }
    result = {
        "benchmark": "G8.4.8.7.3 JEPA action representation under controlled rho shift",
        "seeds": seeds,
        "runs": runs,
        "summary": summarize(runs),
        "protocol": {
            "train_rho": [0.20, 0.40, 0.60],
            "test_shifted_rho": 0.85,
            "test_reverse_rho": 0.05,
            "metadata_not_features": ["agent_policy", "rho", "policy_config", "episode_id", "step", "partition"],
            "calibration": "all model temperatures fitted on nominal-support calibration only",
            "bootstrap_unit": "whole episode",
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "limitations": [
            "one-step action prediction only; recursive E2 security forecasting is a later stage",
            "synthetic behavior archetypes and parameter shifts only",
            "no real-agent generalization or authorization-safety claim",
        ],
    }
    (args.out / "jepa_action_representation_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out / "jepa_action_representation_results.json"), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
