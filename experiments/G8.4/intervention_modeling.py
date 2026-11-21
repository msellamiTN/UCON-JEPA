"""Fit JEPA and supervised E2 risk predictors for G8.4.8.8.

All model fitting uses task-aware train/validation/calibration partitions.
Predictive thresholds are episode-maximum quantiles fitted on validation only.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from intervention_simulator import fit_episode_max_threshold
from policy_parameter_dataset import build_dataset
from policy_shift_matched_recursive_control import _fit_supervised_action_head
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from uncertain_action_jepa import (
    _make_windows,
    _transform,
    fit_e1_e2_models,
    rollout_predicted_actions,
)
from jepa_numpy import JEPAConfig


def _collect_arrays(prefix: str, value: Any, output: dict[str, np.ndarray]) -> None:
    if isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, dict):
        for name, item in value.items():
            _collect_arrays(f"{prefix}_{name}", item, output)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _collect_arrays(f"{prefix}_{index}", item, output)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for name, item in vars(value).items():
            if name in ("config", "history"):
                continue
            _collect_arrays(f"{prefix}_{name}", item, output)


def save_model_checkpoint(
    path: str | Path,
    models: dict[str, Any],
    scalers: dict[str, Any],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, (model, action_head, temperature) in models.items():
        _collect_arrays(f"{name}_dynamics", model, arrays)
        _collect_arrays(f"{name}_action_head", action_head, arrays)
        arrays[f"{name}_action_temperature"] = np.asarray([temperature], dtype=np.float64)
    _collect_arrays("scaler", scalers, arrays)
    if not arrays:
        raise RuntimeError("no model/scaler arrays found for checkpoint")
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trainable_parameter_count(model: Any) -> int:
    if hasattr(model, "parameter_count"):
        value = model.parameter_count
        return int(value() if callable(value) else value)
    total = 0
    for name in ("weights", "bias"):
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray):
            total += int(value.size)
    if total == 0:
        raise TypeError(f"cannot determine trainable parameter count for {type(model).__name__}")
    return total


def _episode_maxima(
    scores: np.ndarray,
    episode_ids: np.ndarray,
) -> list[tuple[str, float]]:
    maxima: dict[str, float] = {}
    for episode_id, score in zip(episode_ids.tolist(), scores.tolist()):
        key = str(episode_id)
        value = float(score)
        if not math.isfinite(value):
            raise RuntimeError("non-finite validation risk score")
        maxima[key] = max(maxima.get(key, -math.inf), value)
    return sorted(maxima.items())


def fit_predictors(
    seed: int,
    episodes_per_cell: int,
    epochs: int,
    mc_samples: int,
    *,
    alert_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, Any]:
    """Fit both predictors and freeze episode-max alert thresholds on validation."""
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    dataset = build_dataset(
        seed=seed, episodes_per_cell=episodes_per_cell,
        include_reverse=True, task_aware=True,
    )
    jepa, jepa_action, _raw_action, scalers = fit_e1_e2_models(
        dataset, seed=seed, epochs=epochs
    )
    train = dataset["train"]
    config = JEPAConfig(
        latent_dim=8, learning_rate=0.008, epochs=epochs, batch_size=128,
        ema_tau=0.99, variance_floor=0.2, variance_weight=0.1,
        security_weight=0.5, seed=seed,
    )
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
        "jepa": (jepa, jepa_action, float(scalers["temperature"])),
        "supervised": (supervised, supervised_action, float(supervised_temperature)),
    }

    score_functions = {}
    thresholds_by_fraction: dict[float, dict[str, dict[str, Any]]] = {
        float(rate): {} for rate in alert_fractions
    }
    threshold_manifest: dict[str, dict[str, Any]] = {}
    validation = dataset["validation"]
    windows = _make_windows(validation, horizon=1)
    for name, (model, action_head, temperature) in models.items():
        def score(
            context: np.ndarray,
            horizon: int,
            rollout_seed: int,
            _model=model,
            _head=action_head,
            _temperature=temperature,
        ) -> float:
            prediction = rollout_predicted_actions(
                _model, _head, np.asarray(context, dtype=np.float64)[None, :],
                scalers["context_scaler"], scalers["action_scaler"],
                _temperature, int(horizon), mc_samples=mc_samples, seed=int(rollout_seed),
            )
            return float(prediction["risk_mean"][0])

        score_functions[name] = score
        prediction = rollout_predicted_actions(
            model, action_head, windows["context"],
            scalers["context_scaler"], scalers["action_scaler"],
            temperature, horizon=1, mc_samples=mc_samples,
            seed=seed + 910_000 + (0 if name == "jepa" else 50_000),
        )
        episode_max_scores = _episode_maxima(
            prediction["risk_mean"], windows["episode_id"]
        )
        threshold_manifest[name] = {}
        for rate in alert_fractions:
            fitted = fit_episode_max_threshold(
                episode_max_scores,
                alert_fraction=float(rate),
                tie_seed=seed + 884_808 + int(round(1000 * rate)),
            )
            thresholds_by_fraction[float(rate)][name] = fitted
            threshold_manifest[name][str(rate)] = {
                **fitted,
                "support": "validation partition only",
                "score_source": "recursive E2 H=1 Monte Carlo risk mean",
                "episode_aggregation": "maximum over eligible decision points",
            }

    partition_ids = {
        key: set(str(value) for value in dataset[key]["episode_id"].tolist())
        for key in ("train", "validation", "calibration")
    }
    for left, right in (
        ("train", "validation"), ("train", "calibration"),
        ("validation", "calibration"),
    ):
        if partition_ids[left] & partition_ids[right]:
            raise AssertionError(f"{left}/{right} episode leakage")
    manifest = dict(dataset["manifest"])
    manifest["partition_counts"] = {
        key: value for key, value in manifest["partition_counts"].items()
        if key in ("train", "validation", "calibration")
    }
    manifest["denial_rates_by_partition_archetype_family_rho"] = {
        key: value for key, value in manifest[
            "denial_rates_by_partition_archetype_family_rho"
        ].items() if key in ("train", "validation", "calibration")
    }
    manifest["risk_threshold_support"] = "validation episode maxima only"
    manifest["risk_threshold_horizon"] = 1

    model_capacity = {}
    for name, (model, action_head, _temperature) in models.items():
        model_capacity[name] = {
            "dynamics_parameters": _trainable_parameter_count(model),
            "action_head_parameters": _trainable_parameter_count(action_head),
            "total_trainable_parameters": int(
                _trainable_parameter_count(model) + _trainable_parameter_count(action_head)
            ),
        }
    model_capacity["supervised_to_jepa_parameter_ratio"] = (
        model_capacity["supervised"]["total_trainable_parameters"]
        / model_capacity["jepa"]["total_trainable_parameters"]
    )
    model_capacity["strict_capacity_match"] = False

    result = {
        "models": models,
        "score_functions": score_functions,
        "thresholds_by_fraction": thresholds_by_fraction,
        "threshold_manifest": threshold_manifest,
        "model_capacity": model_capacity,
        "scalers": scalers,
        "dataset_manifest": manifest,
        "partition_ids": partition_ids,
        "train_episodes": len(partition_ids["train"]),
        "validation_episodes": len(partition_ids["validation"]),
        "calibration_episodes": len(partition_ids["calibration"]),
    }
    # Test partitions are never returned to the pilot or model-selection code.
    del dataset["test_shifted"]
    del dataset["test_reverse"]
    return result
