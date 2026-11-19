"""G8.4.8.7.2 past-only action baselines under controlled rho shift."""
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
from policy_parameter_dataset import (
    POLICY_ARCHETYPES, build_dataset, write_dataset,
)

EPS = 1e-12
MODEL_NAMES = (
    "majority",
    "historical_frequency",
    "first_order_markov",
    "past_only_softmax",
)


def previous_actions(part: dict[str, np.ndarray]) -> np.ndarray:
    """Previous observed action at each decision point; BOS at episode start."""
    previous = np.full(len(part["action"]), FirstOrderMarkovPredictor.BOS, dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for index, episode_id in enumerate(part["episode_id"].tolist()):
        groups.setdefault(str(episode_id), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: int(part["step"][index]))
        for before, after in zip(indices, indices[1:]):
            previous[after] = int(np.argmax(part["action"][before]))
    return previous


def _predict(model_name: str, model: Any, part: dict[str, np.ndarray]) -> np.ndarray:
    if model_name == "first_order_markov":
        return model.predict_proba(part["context"], previous_actions(part))
    return model.predict_proba(part["context"])


def _calibrate(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.where(
        probability > 0.0,
        np.log(np.maximum(probability, EPS)) / temperature,
        -np.inf,
    )
    return _softmax(logits)


def _episode_losses(
    part: dict[str, np.ndarray], probability: np.ndarray
) -> dict[str, dict[str, float]]:
    truth = np.argmax(part["action"], axis=1)
    nll = -np.log(np.clip(probability[np.arange(len(truth)), truth], EPS, 1.0))
    onehot = np.zeros_like(probability)
    onehot[np.arange(len(truth)), truth] = 1.0
    brier = np.square(probability - onehot).sum(axis=1)
    grouped: dict[str, list[int]] = {}
    for i, episode_id in enumerate(part["episode_id"].tolist()):
        grouped.setdefault(str(episode_id), []).append(i)
    return {
        episode_id: {
            "nll": float(nll[indices].mean()),
            "brier": float(brier[indices].mean()),
        }
        for episode_id, indices in grouped.items()
    }


def _paired_episode_bootstrap(
    model_losses: dict[str, dict[str, float]],
    reference_losses: dict[str, dict[str, float]],
    metric: str,
    seed: int,
    replicates: int = 2000,
) -> dict[str, Any]:
    episode_ids = sorted(set(model_losses) & set(reference_losses))
    if len(episode_ids) != len(model_losses) or len(episode_ids) != len(reference_losses):
        raise AssertionError("paired models must be evaluated on identical complete episodes")
    differences = np.asarray([
        model_losses[episode_id][metric] - reference_losses[episode_id][metric]
        for episode_id in episode_ids
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(episode_ids), size=(replicates, len(episode_ids)))
    draws = differences[indices].mean(axis=1)
    return {
        "unit": "whole episode",
        "episodes": len(episode_ids),
        "replicates": replicates,
        "model_minus_historical_frequency_mean": float(differences.mean()),
        "ci95_percentile": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "negative_favours_model": True,
    }


def _group_metrics(
    part: dict[str, np.ndarray], probabilities: dict[str, np.ndarray]
) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for archetype in POLICY_ARCHETYPES:
        grouped[archetype] = {}
        for family in sorted(set(part["family"].tolist())):
            mask = (part["agent_policy"] == archetype) & (part["family"] == family)
            if not np.any(mask):
                continue
            grouped[archetype][family] = {
                name: _metrics(prob[mask], part["action"][mask])
                for name, prob in probabilities.items()
            }
    return grouped


def run_seed(
    seed: int,
    episodes_per_cell: int = 300,
    max_epochs: int = 150,
    patience: int = 15,
    bootstrap_replicates: int = 2000,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
    train, validation, calibration = (
        dataset[name] for name in ("train", "validation", "calibration")
    )
    models = {
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

    calibration_probabilities = {
        name: _predict(name, model, calibration)
        for name, model in models.items()
    }
    temperatures = {
        name: _fit_temperature(calibration_probabilities[name], calibration["action"])
        for name in MODEL_NAMES
    }

    tests = {}
    for split_name in ("test_shifted", "test_reverse"):
        part = dataset[split_name]
        raw_probabilities = {
            name: _predict(name, model, part) for name, model in models.items()
        }
        calibrated_probabilities = {
            name: _calibrate(raw_probabilities[name], temperatures[name])
            for name in MODEL_NAMES
        }
        raw_metrics = {
            name: _metrics(raw_probabilities[name], part["action"])
            for name in MODEL_NAMES
        }
        calibrated_metrics = {
            name: _metrics(calibrated_probabilities[name], part["action"])
            for name in MODEL_NAMES
        }
        grouped = _group_metrics(part, calibrated_probabilities)

        episode_losses = {
            name: _episode_losses(part, calibrated_probabilities[name])
            for name in MODEL_NAMES
        }
        paired = {}
        for name in MODEL_NAMES:
            if name == "historical_frequency":
                continue
            paired[name] = {
                metric: _paired_episode_bootstrap(
                    episode_losses[name], episode_losses["historical_frequency"],
                    metric, seed + (0 if split_name == "test_shifted" else 100_000)
                    + len(name) * 100 + (0 if metric == "nll" else 1),
                    bootstrap_replicates,
                )
                for metric in ("nll", "brier")
            }

        tests[split_name] = {
            "rho_values": sorted(set(float(x) for x in part["rho"].tolist())),
            "episodes": int(len(set(part["episode_id"].tolist()))),
            "transitions": int(len(part["action"])),
            "temperatures_fitted_on_nominal_calibration": temperatures,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "calibrated_by_archetype_family": grouped,
            "paired_episode_bootstrap_vs_historical_frequency": paired,
        }

    if out_dir is not None:
        write_dataset(dataset, out_dir / f"dataset-{seed}")
    return {
        "seed": seed,
        "dataset_manifest": dataset["manifest"],
        "model_selection": {
            "past_only_softmax_selected_epoch": models["past_only_softmax"].best_epoch,
            "temperatures_fit_on": "nominal-support calibration partition only",
            "feature_scaling_fit_on": "train partition only",
        },
        "tests": tests,
    }


def summarize(runs: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for split_name in ("test_shifted", "test_reverse"):
        summary[split_name] = {}
        for model_name in MODEL_NAMES:
            summary[split_name][model_name] = {}
            for metric in ("nll", "top1_accuracy", "top2_accuracy",
                           "multiclass_brier", "top_label_ece_10bin"):
                values = [
                    run["tests"][split_name]["calibrated_metrics"][model_name][metric]
                    for run in runs.values()
                ]
                summary[split_name][model_name][metric] = {
                    "mean_across_seeds": float(np.mean(values)),
                    "sd_across_seeds": float(np.std(values, ddof=1)),
                }
        summary[split_name]["equal_archetype_macro"] = {}
        for model_name in MODEL_NAMES:
            summary[split_name]["equal_archetype_macro"][model_name] = {}
            for metric in ("nll", "top1_accuracy", "top2_accuracy",
                           "multiclass_brier", "top_label_ece_10bin"):
                per_seed = []
                for run in runs.values():
                    part = run["tests"][split_name]["calibrated_by_archetype_family"]
                    archetype_means = []
                    for archetype in POLICY_ARCHETYPES:
                        family_values = [
                            part[archetype][family][model_name][metric]
                            for family in part[archetype]
                        ]
                        archetype_means.append(float(np.mean(family_values)))
                    per_seed.append(float(np.mean(archetype_means)))
                summary[split_name]["equal_archetype_macro"][model_name][metric] = {
                    "mean_across_seeds": float(np.mean(per_seed)),
                    "sd_across_seeds": float(np.std(per_seed, ddof=1)),
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("policy-shift-action-baseline-output"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("provide distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {
        str(seed): run_seed(
            seed, args.episodes_per_cell, args.max_epochs,
            args.patience, args.bootstrap_replicates, args.out,
        )
        for seed in seeds
    }
    result = {
        "benchmark": "G8.4.8.7.2 past-only action baselines under controlled rho shift",
        "seeds": seeds,
        "runs": runs,
        "summary": summarize(runs),
        "protocol": {
            "training_rho": [0.20, 0.40, 0.60],
            "test_shifted_rho": 0.85,
            "test_reverse_rho": 0.05,
            "metadata_not_features": ["agent_policy", "rho", "policy_config", "episode_id", "step", "partition"],
            "calibration": "one scalar temperature per model, fit on nominal-support calibration only",
            "bootstrap_unit": "whole episode",
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "limitations": [
            "synthetic policies and parameter shifts only",
            "one-step action metrics do not establish recursive rollout quality",
            "raw Monte Carlo or model epistemic uncertainty is not evaluated in this stage",
            "no real-agent generalization or authorization-safety claim",
        ],
    }
    (args.out / "action_baseline_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out / "action_baseline_results.json"), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
