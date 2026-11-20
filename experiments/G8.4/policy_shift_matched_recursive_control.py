"""G8.4.8.7.5 matched JEPA vs supervised recursive risk benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import PastOnlySoftmaxPredictor, _fit_temperature
from jepa_numpy import JEPAConfig
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from policy_parameter_dataset import build_dataset, write_dataset
from policy_shift_recursive_risk import (
    EPS, HORIZONS, _apply_platt, _fit_platt, _group_metrics,
    _paired_loss_bootstrap, _transform, _make_windows,
)
from uncertain_action_jepa import (
    _action_features, _family_onehot, fit_e1_e2_models,
    rollout_predicted_actions,
)


def _latent_features(model: Any, part: dict[str, np.ndarray],
                     context_scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    scaled = _transform(part["context"], context_scaler)
    latent = scaled @ model.W_context
    return _action_features(_family_onehot(part["context"]), latent)


def _fit_supervised_action_head(
    model: LinearSupervisedStateDynamics,
    dataset: dict[str, Any],
    context_scaler: tuple[np.ndarray, np.ndarray],
) -> tuple[PastOnlySoftmaxPredictor, float]:
    train, validation, calibration = (
        dataset["train"], dataset["validation"], dataset["calibration"]
    )
    train_x = _latent_features(model, train, context_scaler)
    validation_x = _latent_features(model, validation, context_scaler)
    calibration_x = _latent_features(model, calibration, context_scaler)
    action_head = PastOnlySoftmaxPredictor(max_epochs=250, patience=20).fit(
        train_x, train["action"], validation_x, validation["action"]
    )
    calibration_prob = action_head.predict_proba(calibration_x)
    temperature = _fit_temperature(calibration_prob, calibration["action"])
    return action_head, temperature


def _predict_e2(
    model: Any,
    action_head: PastOnlySoftmaxPredictor,
    part: dict[str, np.ndarray],
    horizon: int,
    context_scaler: tuple[np.ndarray, np.ndarray],
    action_scaler: tuple[np.ndarray, np.ndarray],
    temperature: float,
    mc_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Only initial context is passed into the model's recursive inference call."""
    windows = _make_windows(part, horizon)
    rollout = rollout_predicted_actions(
        model, action_head, windows["context"], context_scaler, action_scaler,
        temperature, horizon, mc_samples=mc_samples, seed=seed,
    )
    truth = windows["security"][:, :, 0].max(axis=1).astype(np.int64)
    return {
        "windows": windows,
        "truth": truth,
        "raw_risk": np.clip(rollout["risk_mean"], EPS, 1.0 - EPS),
        "risk_std": rollout["risk_std"],
        "risk_q05": rollout["risk_q05"],
        "risk_q95": rollout["risk_q95"],
    }


def _decision_time_mask(windows: dict[str, np.ndarray]) -> np.ndarray:
    """Select exactly one step-0 forecast window per complete episode."""
    mask = np.asarray(windows["start_step"]) == 0
    if not np.any(mask):
        raise ValueError("decision-time evaluation requires step-0 windows")
    selected_ids = windows["episode_id"][mask].tolist()
    all_ids = set(windows["episode_id"].tolist())
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != all_ids:
        raise ValueError("expected exactly one step-0 window for every episode")
    return mask


def _run_one_seed(
    seed: int,
    episodes_per_cell: int,
    epochs: int,
    mc_samples: int,
    bootstrap_replicates: int,
    out_dir: Path | None,
) -> dict[str, Any]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
    if out_dir is not None:
        write_dataset(dataset, out_dir / f"dataset-{seed}")

    # JEPA fit uses train for representation learning, validation for action
    # checkpoint selection, and calibration only for action temperature.
    jepa, jepa_action, _raw_action, scalers = fit_e1_e2_models(
        dataset, seed=seed, epochs=epochs
    )

    # Directly predict the next observable context, then re-encode that
    # prediction into the same latent space used by the past-only action head.
    # The smaller latent dimension keeps total capacity within 10% of JEPA.
    config = JEPAConfig(
        latent_dim=8, learning_rate=0.008, epochs=epochs, batch_size=128,
        ema_tau=0.99, variance_floor=0.2, variance_weight=0.1,
        security_weight=0.5, seed=seed,
    )
    train = dataset["train"]
    supervised = LinearSupervisedStateDynamics(
        train["context"].shape[1], train["action"].shape[1],
        train["security"].shape[1], config,
    ).fit(
        _transform(train["context"], scalers["context_scaler"]),
        _transform(train["action"], scalers["action_scaler"]),
        _transform(train["future"], scalers["context_scaler"]),
        train["security"],
    )
    supervised_action, supervised_temperature = _fit_supervised_action_head(
        supervised, dataset, scalers["context_scaler"]
    )

    models = {
        "JEPA_E2": (jepa, jepa_action, float(scalers["temperature"])),
        "supervised_E2": (supervised, supervised_action, float(supervised_temperature)),
    }
    calibration = {name: {} for name in models}
    for horizon in HORIZONS:
        for name, (model, action_head, temperature) in models.items():
            prediction = _predict_e2(
                model, action_head, dataset["calibration"], horizon,
                scalers["context_scaler"], scalers["action_scaler"],
                temperature, mc_samples, seed + horizon * 10_000,
            )
            independent = prediction["windows"]["start_step"] == 0
            calibration[name][horizon] = {
                "platt": _fit_platt(
                    prediction["raw_risk"][independent],
                    prediction["truth"][independent],
                ),
                "episodes": int(independent.sum()),
                "window_rule": "one step-0 window per calibration episode",
            }

    tests: dict[str, Any] = {}
    for split_name in ("test_shifted", "test_reverse"):
        tests[split_name] = {}
        for horizon in HORIZONS:
            split_predictions: dict[str, np.ndarray] = {}
            split_raw: dict[str, dict[str, Any]] = {}
            truth = None
            windows = None
            for name, (model, action_head, temperature) in models.items():
                prediction = _predict_e2(
                    model, action_head, dataset[split_name], horizon,
                    scalers["context_scaler"], scalers["action_scaler"],
                    temperature, mc_samples,
                    seed + horizon * 100_000
                    + (0 if split_name == "test_shifted" else 500_000),
                )
                truth = prediction["truth"]
                windows = prediction["windows"]
                calibrated = _apply_platt(
                    prediction["raw_risk"], calibration[name][horizon]["platt"]
                )
                split_predictions[name] = calibrated
                split_raw[name] = prediction

            train_windows = _make_windows(dataset["train"], horizon)
            all_window_prevalence = float(
                train_windows["security"][:, :, 0].max(axis=1).mean()
            )
            all_window_baseline = np.full(
                len(truth), all_window_prevalence, dtype=np.float64
            )
            all_predictions = {
                "train_prevalence": all_window_baseline,
                **split_predictions,
            }

            # Primary estimand: a forecast made at the episode's decision time.
            # Calibration also uses exactly one start_step=0 window per episode.
            decision_mask = _decision_time_mask(windows)
            train_decision_mask = _decision_time_mask(train_windows)
            decision_truth = truth[decision_mask]
            decision_episode_ids = windows["episode_id"][decision_mask]
            decision_windows = {
                key: value[decision_mask] for key, value in windows.items()
            }
            decision_train_prevalence = float(
                train_windows["security"][train_decision_mask, :, 0]
                .max(axis=1).mean()
            )
            decision_predictions = {
                "train_prevalence": np.full(
                    len(decision_truth), decision_train_prevalence, dtype=np.float64
                ),
                **{
                    name: prediction[decision_mask]
                    for name, prediction in split_predictions.items()
                },
            }
            jepa_decision_loss = (
                decision_predictions["JEPA_E2"] - decision_truth
            ) ** 2
            supervised_decision_loss = (
                decision_predictions["supervised_E2"] - decision_truth
            ) ** 2

            tests[split_name][str(horizon)] = {
                "target": "any formal UCON DENY across the next H transitions",
                "rho": sorted(set(float(x) for x in dataset[split_name]["rho"].tolist())),
                "decision_time_primary": {
                    "population": "exactly one start_step=0 forecast window per episode",
                    "calibration_population": "exactly one start_step=0 window per nominal-support calibration episode",
                    "windows": int(len(decision_truth)),
                    "episodes": int(len(set(decision_episode_ids.tolist()))),
                    "prevalence": float(decision_truth.mean()),
                    "train_horizon_prevalence": decision_train_prevalence,
                    "metrics": {
                        name: __import__("predictive_benchmark").metrics(
                            decision_truth, pred
                        )
                        for name, pred in decision_predictions.items()
                    },
                    "paired_episode_bootstrap_JEPA_minus_supervised_Brier": {
                        **_paired_loss_bootstrap(
                            jepa_decision_loss, supervised_decision_loss,
                            decision_episode_ids,
                            seed + horizon * 1000
                            + (0 if split_name == "test_shifted" else 900_000),
                            bootstrap_replicates,
                        ),
                        "negative_favours_JEPA": True,
                    },
                    "by_archetype_family": _group_metrics(
                        decision_windows, decision_truth, decision_predictions
                    ),
                },
                "secondary_all_windows": {
                    "population": "all overlapping sliding windows; descriptive secondary analysis only",
                    "windows": int(len(truth)),
                    "episodes": int(len(set(windows["episode_id"].tolist()))),
                    "prevalence": float(truth.mean()),
                    "train_horizon_prevalence": all_window_prevalence,
                    "metrics": {
                        name: __import__("predictive_benchmark").metrics(truth, pred)
                        for name, pred in all_predictions.items()
                    },
                    "paired_episode_bootstrap_JEPA_minus_supervised_Brier": {
                        **_paired_loss_bootstrap(
                            (split_predictions["JEPA_E2"] - truth) ** 2,
                            (split_predictions["supervised_E2"] - truth) ** 2,
                            windows["episode_id"],
                            seed + horizon * 1000 + 1_500_000
                            + (0 if split_name == "test_shifted" else 900_000),
                            bootstrap_replicates,
                        ),
                        "negative_favours_JEPA": True,
                    },
                    "by_archetype_family": _group_metrics(
                        windows, truth, all_predictions
                    ),
                },
                "uncertainty": {
                    name: {
                        "mean_mc_std": float(split_raw[name]["risk_std"][decision_mask].mean()),
                        "mean_raw_q05": float(split_raw[name]["risk_q05"][decision_mask].mean()),
                        "mean_raw_q95": float(split_raw[name]["risk_q95"][decision_mask].mean()),
                    }
                    for name in models
                },
            }

    return {
        "seed": seed,
        "dataset_manifest": dataset["manifest"],
        "models": {
            "JEPA_E2": {
                "trainable_parameter_count": jepa.parameter_count(),
                "stored_parameter_count_including_ema_target": (
                    jepa.parameter_count() + jepa.W_target.size
                ),
                "action_head_selected_epoch": jepa_action.best_epoch,
                "action_temperature": float(scalers["temperature"]),
                "latent_dim": jepa.config.latent_dim,
                "latent_loss_final": float(jepa.history.latent_loss[-1]),
                "security_loss_final": float(jepa.history.security_loss[-1]),
            },
            "supervised_E2": {
                "trainable_parameter_count": supervised.parameter_count(),
                "stored_parameter_count": supervised.parameter_count(),
                "action_head_selected_epoch": supervised_action.best_epoch,
                "action_temperature": supervised_temperature,
                "security_loss_final": float(supervised.history[-1]["security_mse"]),
                "future_context_mse_final": float(supervised.history[-1]["future_mse"]),
                "latent_dim": supervised.config.latent_dim,
            },
            "shared_training_budget": {
                "epochs": epochs, "batch_size": 128, "learning_rate": 0.008,
                "JEPA_latent_dim": 12, "supervised_latent_dim": 8,
                "same_seed": seed,
                "trainable_parameter_ratio_supervised_to_JEPA": (
                    supervised.parameter_count() / jepa.parameter_count()
                ),
            },
        },
        "risk_calibrators": {
            name: {
                str(h): {
                    **calibration[name][h]["platt"],
                    "calibration_episodes": calibration[name][h]["episodes"],
                    "window_rule": calibration[name][h]["window_rule"],
                }
                for h in HORIZONS
            }
            for name in models
        },
        "tests": tests,
        "information_boundary": {
            "both_models": "initial context only; future actions sampled autoregressively from model-specific past-only action head",
            "future_actions_observations_and_security_labels": "targets only; never inference inputs",
            "authorization": "formal UCON remains sole authority; predictions cannot authorize or override DENY",
        },
    }


def _summarize(runs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("test_shifted", "test_reverse"):
        summary[split] = {}
        for horizon in map(str, HORIZONS):
            summary[split][horizon] = {}
            for method in ("train_prevalence", "JEPA_E2", "supervised_E2"):
                values = [
                    run["tests"][split][horizon]["decision_time_primary"]["metrics"][method]
                    for run in runs.values()
                ]
                summary[split][horizon][method] = {
                    key: {
                        "mean": float(np.mean([v[key] for v in values])),
                        "sd_across_seeds": float(np.std([v[key] for v in values], ddof=1)),
                    }
                    for key in ("auroc", "auprc", "brier", "ece_10", "prevalence")
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("matched-recursive-control-output"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {
        str(seed): _run_one_seed(
            seed, args.episodes_per_cell, args.epochs, args.mc_samples,
            args.bootstrap_replicates, args.out,
        )
        for seed in seeds
    }
    result = {
        "benchmark": "G8.4.8.7.7 decision-time matched recursive JEPA vs supervised evaluation",
        "seeds": seeds,
        "runs": runs,
        "summary": _summarize(runs),
        "protocol": {
            "train_rho": [0.20, 0.40, 0.60],
            "test_shifted_rho": 0.85,
            "test_reverse_rho": 0.05,
            "horizons": list(HORIZONS),
            "episodes_per_cell": args.episodes_per_cell,
            "primary_population": "one start_step=0 window per test episode; same one-window-per-episode rule as risk calibration",
            "secondary_population": "all overlapping sliding windows; descriptive only",
            "primary_contrast": "decision-time JEPA_E2 minus supervised_E2 Brier; negative favours JEPA",
            "bootstrap_unit": "whole episode; exactly one primary window per episode",
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "limitations": [
            "five synthetic behavior archetypes and one scalar within-archetype mixture parameter",
            "supervised control directly predicts next observable context and security targets, then re-encodes predicted context for recursive rollout; it has no EMA target encoder",
            "E1 privileged known-action results are reported in G8.4.8.7.4 and are not a deployable comparator",
            "no real-agent transfer, intervention utility, or production authorization claim",
        ],
    }
    output = args.out / "matched_recursive_control_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": seeds, "horizons": HORIZONS}, indent=2))


if __name__ == "__main__":
    main()
