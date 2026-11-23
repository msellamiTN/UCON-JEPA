"""G8.4.8.7.8 crossed dataset/training-seed audit for initial-position direction.

The benchmark is diagnostic only. It varies dataset-generation and combined
model-training stochasticity independently, while holding Monte Carlo rollout
randomness fixed across the full matrix and both models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from jepa_numpy import JEPAConfig
from policy_parameter_dataset import build_dataset, write_dataset
from policy_shift_matched_recursive_control import (
    _fit_supervised_action_head,
    _predict_e2,
)
from policy_shift_recursive_risk import _apply_platt, _fit_platt
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from predictive_benchmark import metrics as binary_metrics
from uncertain_action_jepa import _transform, fit_e1_e2_models

DATASET_SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
MODEL_TRAINING_SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
HORIZON = 1
CALIBRATION_ROLLOUT_SEED = 20261119 + 10_000 * HORIZON
SHIFTED_ROLLOUT_SEED = 20261119 + 100_000 * HORIZON
REVERSE_ROLLOUT_SEED = SHIFTED_ROLLOUT_SEED + 500_000
EPS = 1e-8


def _score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    positive, negative = y == 1, y == 0
    if len(s) != len(y) or not len(s) or not positive.any() or not negative.any():
        raise ValueError("score summary requires aligned data with both classes")
    return {
        "n": int(len(s)),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "positive_mean": float(s[positive].mean()),
        "negative_mean": float(s[negative].mean()),
        "positive_minus_negative_mean": float(s[positive].mean() - s[negative].mean()),
        "quantiles": [float(v) for v in np.quantile(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
    }


def _position_zero(prediction: dict[str, Any]) -> dict[str, np.ndarray]:
    windows = prediction["windows"]
    mask = np.asarray(windows["start_step"]) == 0
    selected = {
        "episode_id": np.asarray(windows["episode_id"])[mask],
        "agent_policy": np.asarray(windows["agent_policy"])[mask],
        "family": np.asarray(windows["family"])[mask],
        "truth": np.asarray(prediction["truth"], dtype=np.int64)[mask],
        "raw": np.asarray(prediction["raw_risk"], dtype=np.float64)[mask],
    }
    if not len(selected["truth"]):
        raise ValueError("no position-0 windows")
    if len(set(selected["episode_id"].tolist())) != len(selected["episode_id"]):
        raise AssertionError("position 0 must contain exactly one window per episode")
    if not np.array_equal(
        np.unique(selected["truth"]), np.asarray([0, 1], dtype=np.int64)
    ):
        raise ValueError("position-0 support must contain both classes")
    return selected


def _assert_episode_disjoint(dataset: dict[str, Any]) -> None:
    partitions = ("train", "validation", "calibration", "test_shifted", "test_reverse")
    ids = {name: set(dataset[name]["episode_id"].tolist()) for name in partitions}
    for i, left in enumerate(partitions):
        for right in partitions[i + 1:]:
            overlap = ids[left] & ids[right]
            if overlap:
                raise AssertionError(
                    f"episode leakage between {left} and {right}: {len(overlap)} IDs"
                )


def _strata_indices(part: dict[str, np.ndarray]) -> list[np.ndarray]:
    episode_ids = np.asarray(part["episode_id"])
    policies = np.asarray(part["agent_policy"])
    families = np.asarray(part["family"])
    labels = np.asarray(part["truth"], dtype=np.int64)
    if len(set(episode_ids.tolist())) != len(episode_ids):
        raise AssertionError("bootstrap input must have one position-0 row per episode")
    groups: dict[tuple[str, str, int], list[int]] = {}
    for i, (policy, family, label) in enumerate(zip(policies, families, labels)):
        groups.setdefault((str(policy), str(family), int(label)), []).append(i)
    if not groups or not any(key[2] == 0 for key in groups) or not any(key[2] == 1 for key in groups):
        raise ValueError("bootstrap strata must retain both outcome classes")
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def _strata_episode_counts(part: dict[str, np.ndarray]) -> dict[str, int]:
    policies = np.asarray(part["agent_policy"])
    families = np.asarray(part["family"])
    labels = np.asarray(part["truth"], dtype=np.int64)
    return {
        f"{policy}/{family}/label={label}": int(
            np.sum((policies == policy) & (families == family) & (labels == label))
        )
        for policy in sorted(set(policies.tolist()))
        for family in sorted(set(families.tolist()))
        for label in (0, 1)
    }


def _sample_stratified_indices(
    groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    return np.concatenate([
        rng.choice(group, size=len(group), replace=True) for group in groups
    ])


def _fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann–Whitney AUROC with average ranks for ties."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def _percentile_ci(values: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr) or not np.isfinite(arr).all():
        raise FloatingPointError("bootstrap produced empty or non-finite values")
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def _bootstrap_cell(
    calibration: dict[str, dict[str, np.ndarray]],
    shifted: dict[str, dict[str, np.ndarray]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Paired, label-stratified episode bootstrap for calibration and shifted test."""
    models = ("JEPA_E2", "supervised_E2")
    cal_ref = calibration["JEPA_E2"]
    test_ref = shifted["JEPA_E2"]
    for model in models:
        if not np.array_equal(cal_ref["episode_id"], calibration[model]["episode_id"]):
            raise AssertionError("models do not share calibration episode order")
        if not np.array_equal(test_ref["episode_id"], shifted[model]["episode_id"]):
            raise AssertionError("models do not share shifted-test episode order")
        if not np.array_equal(cal_ref["truth"], calibration[model]["truth"]):
            raise AssertionError("models do not share calibration labels")
        if not np.array_equal(test_ref["truth"], shifted[model]["truth"]):
            raise AssertionError("models do not share shifted-test labels")

    cal_groups = _strata_indices(cal_ref)
    test_groups = _strata_indices(test_ref)
    rng_cal = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)

    draws: dict[str, dict[str, list[float]]] = {
        model: {
            "calibration_delta": [],
            "calibration_slope": [],
            "test_delta": [],
            "test_auroc": [],
            "test_brier": [],
            "test_calibrated_brier": [],
        }
        for model in models
    }
    paired = {"raw_brier": [], "calibrated_brier": []}

    for _ in range(replicates):
        cal_idx = _sample_stratified_indices(cal_groups, rng_cal)
        test_idx = _sample_stratified_indices(test_groups, rng_test)
        params_by_model: dict[str, dict[str, float]] = {}
        for model in models:
            cal = calibration[model]
            y_cal = cal["truth"][cal_idx]
            s_cal = cal["raw"][cal_idx]
            params = _fit_platt(s_cal, y_cal)
            params_by_model[model] = params
            draws[model]["calibration_delta"].append(
                float(s_cal[y_cal == 1].mean() - s_cal[y_cal == 0].mean())
            )
            draws[model]["calibration_slope"].append(float(params["slope"]))

        test_values: dict[str, dict[str, float]] = {}
        for model in models:
            test = shifted[model]
            y_test = test["truth"][test_idx]
            s_test = test["raw"][test_idx]
            calibrated = _apply_platt(s_test, params_by_model[model])
            values = {
                "delta": float(s_test[y_test == 1].mean() - s_test[y_test == 0].mean()),
                "auroc": _fast_auc(y_test, s_test),
                "brier": float(np.mean((s_test - y_test) ** 2)),
                "calibrated_brier": float(np.mean((calibrated - y_test) ** 2)),
            }
            test_values[model] = values
            draws[model]["test_delta"].append(values["delta"])
            draws[model]["test_auroc"].append(values["auroc"])
            draws[model]["test_brier"].append(values["brier"])
            draws[model]["test_calibrated_brier"].append(values["calibrated_brier"])

        paired["raw_brier"].append(
            test_values["JEPA_E2"]["brier"] - test_values["supervised_E2"]["brier"]
        )
        paired["calibrated_brier"].append(
            test_values["JEPA_E2"]["calibrated_brier"]
            - test_values["supervised_E2"]["calibrated_brier"]
        )

    return {
        "unit": "whole episode, resampled within policy × family × label strata",
        "replicates": replicates,
        "interval_interpretation": "conditional on observed class counts in each policy-family stratum",
        "calibration_stratum_episode_counts": _strata_episode_counts(cal_ref),
        "shifted_test_stratum_episode_counts": _strata_episode_counts(test_ref),
        "models": {
            model: {
                "calibration_delta_ci95": _percentile_ci(draws[model]["calibration_delta"]),
                "calibration_slope_ci95": _percentile_ci(draws[model]["calibration_slope"]),
                "shifted_test_delta_ci95": _percentile_ci(draws[model]["test_delta"]),
                "shifted_test_auroc_ci95": _percentile_ci(draws[model]["test_auroc"]),
                "shifted_test_brier_ci95": _percentile_ci(draws[model]["test_brier"]),
                "shifted_test_calibrated_brier_ci95": _percentile_ci(
                    draws[model]["test_calibrated_brier"]
                ),
            }
            for model in models
        },
        "paired_JEPA_minus_supervised": {
            "shifted_test_raw_brier_ci95": _percentile_ci(paired["raw_brier"]),
            "shifted_test_calibrated_brier_ci95": _percentile_ci(paired["calibrated_brier"]),
        },
    }


def _classify_direction(
    calibration_delta: float,
    calibration_slope: float,
    test_delta: float,
    test_auroc: float,
    intervals: dict[str, list[float]],
) -> str:
    signs = (
        np.sign(calibration_delta),
        np.sign(calibration_slope),
        np.sign(test_delta),
        np.sign(test_auroc - 0.5),
    )
    if any(value == 0 for value in signs):
        return "weak_or_inconclusive"
    if len(set(int(value) for value in signs)) != 1:
        return "calibration_test_or_component_discordance"
    direction = int(signs[0])
    intervals_to_check = (
        (intervals["calibration_delta_ci95"], 0.0),
        (intervals["calibration_slope_ci95"], 0.0),
        (intervals["shifted_test_delta_ci95"], 0.0),
        (intervals["shifted_test_auroc_ci95"], 0.5),
    )
    if direction > 0 and all(bounds[0] > threshold for bounds, threshold in intervals_to_check):
        return "supported_positive"
    if direction < 0 and all(bounds[1] < threshold for bounds, threshold in intervals_to_check):
        return "supported_negative"
    return "weak_or_inconclusive"


def _matrix_factor_summary(
    values: dict[str, float],
    dataset_seeds: tuple[int, ...],
    model_seeds: tuple[int, ...],
) -> dict[str, Any]:
    matrix = np.asarray([
        [values[f"{d}:{m}"] for m in model_seeds]
        for d in dataset_seeds
    ], dtype=np.float64)
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    row_effect = row_means - grand
    col_effect = col_means - grand
    fitted_additive = grand + row_effect[:, None] + col_effect[None, :]
    residual = matrix - fitted_additive
    ss_total = float(np.square(matrix - grand).sum())
    ss_dataset = float(len(model_seeds) * np.square(row_effect).sum())
    ss_training = float(len(dataset_seeds) * np.square(col_effect).sum())
    ss_residual = float(np.square(residual).sum())
    return {
        "additive_residuals_by_dataset_seed": {
            str(seed): [float(v) for v in residual[i]]
            for i, seed in enumerate(dataset_seeds)
        },
        "dataset_seed_rows": {
            str(seed): [float(v) for v in matrix[i]]
            for i, seed in enumerate(dataset_seeds)
        },
        "model_training_seed_columns": {
            str(seed): [float(matrix[i, j]) for i in range(len(dataset_seeds))]
            for j, seed in enumerate(model_seeds)
        },
        "grand_mean": grand,
        "dataset_seed_row_means": {
            str(seed): float(row_means[i]) for i, seed in enumerate(dataset_seeds)
        },
        "model_training_seed_column_means": {
            str(seed): float(col_means[j]) for j, seed in enumerate(model_seeds)
        },
        "additive_decomposition_sum_squares": {
            "total": ss_total,
            "dataset_seed": ss_dataset,
            "model_training_seed": ss_training,
            "residual_including_interaction": ss_residual,
            "interaction_not_separately_identifiable": True,
        },
    }


def _recurrence_summary(
    cells: dict[str, Any],
    dataset_seeds: tuple[int, ...],
    model_seeds: tuple[int, ...],
) -> dict[str, Any]:
    def counts_for_row(dataset_seed: int) -> dict[str, int]:
        selected = [cells[f"{dataset_seed}:{m}"] for m in model_seeds]
        return _recurrence_counts(selected)

    def counts_for_column(model_seed: int) -> dict[str, int]:
        selected = [cells[f"{d}:{model_seed}"] for d in dataset_seeds]
        return _recurrence_counts(selected)

    rows = {str(d): counts_for_row(d) for d in dataset_seeds}
    columns = {str(m): counts_for_column(m) for m in model_seeds}

    def qualifies(counts: dict[str, int]) -> bool:
        return (
            counts["JEPA_supported_negative"] >= 4
            and counts["supervised_supported_negative"] <= 1
            and counts["JEPA_minus_supervised_raw_AUROC_negative"] >= 4
            and counts["JEPA_minus_supervised_calibration_slope_negative"] >= 4
        )

    return {
        "dataset_seed_rows": rows,
        "model_training_seed_columns": columns,
        "dataset_seed_linked_recurrence": {
            key: value["JEPA_supported_negative"] >= 4
            for key, value in rows.items()
        },
        "training_seed_linked_recurrence": {
            key: value["JEPA_supported_negative"] >= 4
            for key, value in columns.items()
        },
        "JEPA_specific_dataset_seed_recurrence": {
            key: qualifies(value) for key, value in rows.items()
        },
        "JEPA_specific_training_seed_recurrence": {
            key: qualifies(value) for key, value in columns.items()
        },
    }


def _recurrence_counts(selected_cells: list[dict[str, Any]]) -> dict[str, int]:
    jepa_negative = supervised_negative = auroc_negative = slope_negative = 0
    for cell in selected_cells:
        models = cell["models"]
        jepa_negative += int(
            models["shifted"]["JEPA_E2"]["direction_classification"] == "supported_negative"
        )
        supervised_negative += int(
            models["shifted"]["supervised_E2"]["direction_classification"] == "supported_negative"
        )
        paired = models["paired_JEPA_minus_supervised"]
        auroc_negative += int(paired["JEPA_minus_supervised_raw_auroc"] < 0.0)
        slope_negative += int(paired["JEPA_minus_supervised_calibration_slope"] < 0.0)
    return {
        "JEPA_supported_negative": jepa_negative,
        "supervised_supported_negative": supervised_negative,
        "JEPA_minus_supervised_raw_AUROC_negative": auroc_negative,
        "JEPA_minus_supervised_calibration_slope_negative": slope_negative,
    }


def _legacy_rollout_seeds(dataset_seed: int) -> tuple[int, int]:
    return (
        dataset_seed + 10_000 * HORIZON,
        dataset_seed + 100_000 * HORIZON,
    )


def _check_diagonal_reproduction(
    dataset_seed: int,
    models: dict[str, tuple[Any, Any, float]],
    dataset: dict[str, Any],
    scalers: dict[str, Any],
    reference: dict[str, Any],
    mc_samples: int,
) -> dict[str, Any]:
    cal_seed, test_seed = _legacy_rollout_seeds(dataset_seed)
    output: dict[str, Any] = {}
    for model_name, (model, action_head, temperature) in models.items():
        cal_all = _predict_e2(
            model, action_head, dataset["calibration"], HORIZON,
            scalers["context_scaler"], scalers["action_scaler"],
            temperature, mc_samples, cal_seed,
        )
        cal = _position_zero(cal_all)
        params = _fit_platt(cal["raw"], cal["truth"])
        test_all = _predict_e2(
            model, action_head, dataset["test_shifted"], HORIZON,
            scalers["context_scaler"], scalers["action_scaler"],
            temperature, mc_samples, test_seed,
        )
        test = _position_zero(test_all)
        summary = _score_summary(test["raw"], test["truth"])
        metrics = binary_metrics(test["truth"], test["raw"])
        expected = reference["seeds"][str(dataset_seed)][model_name]
        observed = {
            "calibration_slope": float(params["slope"]),
            "test_auroc": float(metrics["auroc"]),
            "test_brier": float(metrics["brier"]),
            "test_delta": float(summary["positive_minus_negative_mean"]),
            "n": int(summary["n"]),
            "positive": int(summary["positive"]),
        }
        differences = {}
        for key in ("calibration_slope", "test_auroc", "test_brier", "test_delta"):
            differences[key] = float(observed[key] - expected[key])
            if abs(differences[key]) > 1e-12:
                raise AssertionError(
                    f"legacy diagonal reproduction failed for {dataset_seed}/{model_name}/{key}: "
                    f"observed={observed[key]:.17g}, expected={expected[key]:.17g}"
                )
        for key in ("n", "positive"):
            if observed[key] != expected[key]:
                raise AssertionError(
                    f"legacy diagonal count mismatch for {dataset_seed}/{model_name}/{key}"
                )
        output[model_name] = {
            "observed": observed,
            "reference": expected,
            "difference": differences,
            "passed": True,
        }
    return output


def _fit_models(
    dataset: dict[str, Any],
    model_seed: int,
    epochs: int,
) -> tuple[dict[str, tuple[Any, Any, float]], dict[str, Any]]:
    jepa, jepa_action, _raw_action, scalers = fit_e1_e2_models(
        dataset, seed=model_seed, epochs=epochs
    )
    train = dataset["train"]
    config = JEPAConfig(
        latent_dim=8, learning_rate=0.008, epochs=epochs, batch_size=128,
        ema_tau=0.99, variance_floor=0.2, variance_weight=0.1,
        security_weight=0.5, seed=model_seed,
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
    if jepa.parameter_count() != 852 or supervised.parameter_count() != 776:
        raise AssertionError(
            f"unexpected model capacity: JEPA={jepa.parameter_count()}, "
            f"supervised={supervised.parameter_count()}"
        )
    models = {
        "JEPA_E2": (jepa, jepa_action, float(scalers["temperature"])),
        "supervised_E2": (supervised, supervised_action, float(supervised_temperature)),
    }
    metadata = {
        "JEPA_E2": {
            "trainable_parameters": jepa.parameter_count(),
            "latent_dim": jepa.config.latent_dim,
            "final_latent_loss": float(jepa.history.latent_loss[-1]),
            "final_security_loss": float(jepa.history.security_loss[-1]),
            "ema_target_distance": float(jepa.ema_target_distance()),
            "action_head_selected_epoch": int(jepa_action.best_epoch),
            "action_temperature": float(scalers["temperature"]),
        },
        "supervised_E2": {
            "trainable_parameters": supervised.parameter_count(),
            "latent_dim": supervised.config.latent_dim,
            "final_future_context_mse": float(supervised.history[-1]["future_mse"]),
            "final_security_mse": float(supervised.history[-1]["security_mse"]),
            "action_head_selected_epoch": int(supervised_action.best_epoch),
            "action_temperature": float(supervised_temperature),
        },
        "supervised_to_JEPA_parameter_ratio": float(
            supervised.parameter_count() / jepa.parameter_count()
        ),
        "model_training_seed_controls": "initialization and minibatch permutation",
    }
    return models, {"scalers": scalers, "metadata": metadata}


def _run_matrix(
    dataset_seeds: tuple[int, ...],
    model_seeds: tuple[int, ...],
    episodes_per_cell: int,
    epochs: int,
    mc_samples: int,
    bootstrap_replicates: int,
    output_dir: Path,
    reference_path: Path,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    datasets: dict[int, dict[str, Any]] = {}
    manifests: dict[str, Any] = {}
    for dataset_seed in dataset_seeds:
        dataset = build_dataset(seed=dataset_seed, episodes_per_cell=episodes_per_cell)
        _assert_episode_disjoint(dataset)
        write_dataset(dataset, output_dir / f"dataset-{dataset_seed}")
        datasets[dataset_seed] = dataset
        manifests[str(dataset_seed)] = dataset["manifest"]

    cells: dict[str, Any] = {}
    diagonal: dict[str, Any] = {}
    bootstrap_index = 0
    for dataset_seed in dataset_seeds:
        dataset = datasets[dataset_seed]
        for model_seed in model_seeds:
            cell_key = f"{dataset_seed}:{model_seed}"
            models, fitted = _fit_models(dataset, model_seed, epochs)
            scalers = fitted["scalers"]
            predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {
                model_name: {} for model_name in models
            }
            calibration_data: dict[str, dict[str, np.ndarray]] = {}
            for model_name, (model, action_head, temperature) in models.items():
                cal_all = _predict_e2(
                    model, action_head, dataset["calibration"], HORIZON,
                    scalers["context_scaler"], scalers["action_scaler"],
                    temperature, mc_samples, CALIBRATION_ROLLOUT_SEED,
                )
                calibration_data[model_name] = _position_zero(cal_all)

            calibrators = {
                model_name: _fit_platt(data["raw"], data["truth"])
                for model_name, data in calibration_data.items()
            }
            cell_models: dict[str, Any] = {}
            for split_name, rollout_seed in (
                ("shifted", SHIFTED_ROLLOUT_SEED),
                ("reverse", REVERSE_ROLLOUT_SEED),
            ):
                partition = "test_shifted" if split_name == "shifted" else "test_reverse"
                for model_name, (model, action_head, temperature) in models.items():
                    test_all = _predict_e2(
                        model, action_head, dataset[partition], HORIZON,
                        scalers["context_scaler"], scalers["action_scaler"],
                        temperature, mc_samples, rollout_seed,
                    )
                    test_data = _position_zero(test_all)
                    calibrated = _apply_platt(test_data["raw"], calibrators[model_name])
                    predictions[model_name][split_name] = {
                        **test_data,
                        "calibrated": calibrated,
                    }
                    raw_metrics = binary_metrics(test_data["truth"], test_data["raw"])
                    calibrated_metrics = binary_metrics(test_data["truth"], calibrated)
                    if split_name not in cell_models:
                        cell_models[split_name] = {}
                    cell_models[split_name][model_name] = {
                        "windows": int(len(test_data["truth"])),
                        "episodes": int(len(set(test_data["episode_id"].tolist()))),
                        "positive": int(test_data["truth"].sum()),
                        "prevalence": float(test_data["truth"].mean()),
                        "raw_score": _score_summary(test_data["raw"], test_data["truth"]),
                        "raw_metrics": raw_metrics,
                        "calibrated_metrics": calibrated_metrics,
                        "calibration": calibrators[model_name],
                    }

            # Store episode-level position-0 outputs; no raw future data are used by inference.
            np.savez_compressed(
                output_dir / f"predictions-d{dataset_seed}-m{model_seed}.npz",
                calibration_episode_id=calibration_data["JEPA_E2"]["episode_id"],
                calibration_policy=calibration_data["JEPA_E2"]["agent_policy"],
                calibration_family=calibration_data["JEPA_E2"]["family"],
                calibration_truth=calibration_data["JEPA_E2"]["truth"],
                jepa_calibration_raw=calibration_data["JEPA_E2"]["raw"],
                supervised_calibration_raw=calibration_data["supervised_E2"]["raw"],
                shifted_episode_id=predictions["JEPA_E2"]["shifted"]["episode_id"],
                shifted_policy=predictions["JEPA_E2"]["shifted"]["agent_policy"],
                shifted_family=predictions["JEPA_E2"]["shifted"]["family"],
                shifted_truth=predictions["JEPA_E2"]["shifted"]["truth"],
                jepa_shifted_raw=predictions["JEPA_E2"]["shifted"]["raw"],
                supervised_shifted_raw=predictions["supervised_E2"]["shifted"]["raw"],
                jepa_shifted_calibrated=predictions["JEPA_E2"]["shifted"]["calibrated"],
                supervised_shifted_calibrated=predictions["supervised_E2"]["shifted"]["calibrated"],
                reverse_episode_id=predictions["JEPA_E2"]["reverse"]["episode_id"],
                reverse_policy=predictions["JEPA_E2"]["reverse"]["agent_policy"],
                reverse_family=predictions["JEPA_E2"]["reverse"]["family"],
                reverse_truth=predictions["JEPA_E2"]["reverse"]["truth"],
                jepa_reverse_raw=predictions["JEPA_E2"]["reverse"]["raw"],
                supervised_reverse_raw=predictions["supervised_E2"]["reverse"]["raw"],
                jepa_reverse_calibrated=predictions["JEPA_E2"]["reverse"]["calibrated"],
                supervised_reverse_calibrated=predictions["supervised_E2"]["reverse"]["calibrated"],
            )

            bootstrap_index += 1
            bootstrap = _bootstrap_cell(
                calibration_data,
                {
                    "JEPA_E2": predictions["JEPA_E2"]["shifted"],
                    "supervised_E2": predictions["supervised_E2"]["shifted"],
                },
                seed=30_000_000 + bootstrap_index,
                replicates=bootstrap_replicates,
            )
            for model_name in models:
                calibration_summary = _score_summary(
                    calibration_data[model_name]["raw"],
                    calibration_data[model_name]["truth"],
                )
                intervals = bootstrap["models"][model_name]
                cell_models["shifted"][model_name]["calibration_score"] = calibration_summary
                cell_models["shifted"][model_name]["direction_classification"] = _classify_direction(
                    calibration_summary["positive_minus_negative_mean"],
                    calibrators[model_name]["slope"],
                    cell_models["shifted"][model_name]["raw_score"]["positive_minus_negative_mean"],
                    cell_models["shifted"][model_name]["raw_metrics"]["auroc"],
                    intervals,
                )
            jepa = cell_models["shifted"]["JEPA_E2"]
            supervised = cell_models["shifted"]["supervised_E2"]
            paired = {
                "JEPA_minus_supervised_raw_auroc": float(
                    jepa["raw_metrics"]["auroc"] - supervised["raw_metrics"]["auroc"]
                ),
                "JEPA_minus_supervised_raw_brier": float(
                    jepa["raw_metrics"]["brier"] - supervised["raw_metrics"]["brier"]
                ),
                "JEPA_minus_supervised_calibration_slope": float(
                    jepa["calibration"]["slope"] - supervised["calibration"]["slope"]
                ),
                "JEPA_minus_supervised_calibration_delta": float(
                    jepa["calibration_score"]["positive_minus_negative_mean"]
                    - supervised["calibration_score"]["positive_minus_negative_mean"]
                ),
                "JEPA_minus_supervised_calibrated_brier": float(
                    jepa["calibrated_metrics"]["brier"] - supervised["calibrated_metrics"]["brier"]
                ),
                "bootstrap": bootstrap["paired_JEPA_minus_supervised"],
            }
            cell_models["paired_JEPA_minus_supervised"] = paired
            cells[cell_key] = {
                "dataset_seed": dataset_seed,
                "model_training_seed": model_seed,
                "models": cell_models,
                "model_metadata": fitted["metadata"],
                "bootstrap": bootstrap,
                "rollout_randomness": {
                    "calibration_seed": CALIBRATION_ROLLOUT_SEED,
                    "shifted_test_seed": SHIFTED_ROLLOUT_SEED,
                    "reverse_test_seed": REVERSE_ROLLOUT_SEED,
                    "held_constant_across_both_seed_factors_and_models": True,
                },
            }
            if dataset_seed == model_seed:
                diagonal[str(dataset_seed)] = _check_diagonal_reproduction(
                    dataset_seed, models, dataset, scalers, reference, mc_samples
                )

    factor_analysis: dict[str, Any] = {}
    for model_name in ("JEPA_E2", "supervised_E2"):
        factor_analysis[model_name] = {}
        metric_extractors = {
            "calibration_delta": lambda cell: cell["models"]["shifted"][model_name]["calibration_score"]["positive_minus_negative_mean"],
            "calibration_slope": lambda cell: cell["models"]["shifted"][model_name]["calibration"]["slope"],
            "test_raw_delta": lambda cell: cell["models"]["shifted"][model_name]["raw_score"]["positive_minus_negative_mean"],
            "test_raw_auroc": lambda cell: cell["models"]["shifted"][model_name]["raw_metrics"]["auroc"],
            "test_raw_brier": lambda cell: cell["models"]["shifted"][model_name]["raw_metrics"]["brier"],
            "test_calibrated_brier": lambda cell: cell["models"]["shifted"][model_name]["calibrated_metrics"]["brier"],
        }
        for metric_name, extractor in metric_extractors.items():
            values = {
                key: float(extractor(cell)) for key, cell in cells.items()
            }
            factor_analysis[model_name][metric_name] = _matrix_factor_summary(
                values, dataset_seeds, model_seeds
            )
        factor_analysis[model_name]["supported_negative_counts_by_dataset_seed"] = {
            str(d): sum(
                cells[f"{d}:{m}"]["models"]["shifted"][model_name]["direction_classification"]
                == "supported_negative"
                for m in model_seeds
            )
            for d in dataset_seeds
        }
        factor_analysis[model_name]["supported_negative_counts_by_model_training_seed"] = {
            str(m): sum(
                cells[f"{d}:{m}"]["models"]["shifted"][model_name]["direction_classification"]
                == "supported_negative"
                for d in dataset_seeds
            )
            for m in model_seeds
        }

    factor_analysis["recurrence_rules"] = _recurrence_summary(cells, dataset_seeds, model_seeds)

    result = {
        "benchmark": "G8.4.8.7.8 crossed-seed initial-position direction audit",
        "code_sha": os.environ.get("GITHUB_SHA"),
        "dataset_seeds": list(dataset_seeds),
        "model_training_seeds": list(model_seeds),
        "cells": cells,
        "diagonal_legacy_reproduction": diagonal,
        "factor_analysis": factor_analysis,
        "dataset_manifests": manifests,
        "protocol": {
            "design": "fully crossed 5x5 dataset-generation seed × combined model-training seed",
            "models": ["JEPA_E2", "supervised_E2"],
            "horizon": HORIZON,
            "primary_support": "start_step == 0 for calibration and test",
            "primary_test": "shifted rho=0.85",
            "diagnostic_test": "reverse rho=0.05",
            "mc_samples": mc_samples,
            "mc_randomness": "fixed across all 25 cells and both models; independent of both seed factors",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_unit": "whole episode, stratified by archetype × family × label",
            "bootstrap_interpretation": "conditional on observed class counts",
            "parameter_counts": {"JEPA_E2": 852, "supervised_E2": 776},
            "diagonal_reproduction": "legacy rollout RNG; compare retained summary metrics to reference JSON at abs_tol=1e-12, rtol=0",
        },
        "information_boundary": (
            "E2 inference receives initial decision-time context only; future actions are sampled "
            "autoregressively. Future actions, observations, security labels, policy metadata, "
            "episode IDs and start positions are evaluation-only and never enter inference."
        ),
        "authorization_invariant": (
            "Predictive outputs are advisory only. Formal UCON remains the sole authorization "
            "authority and a predictive score cannot grant access or override DENY."
        ),
        "limitations": [
            "synthetic five-archetype, three-family benchmark only",
            "model-training seed combines initialization and minibatch order",
            "one observation per crossed cell means dataset×training interaction is confounded with residual",
            "no real-agent transfer, intervention utility, or production security assurance",
        ],
    }
    (output_dir / "crossed_seed_initial_direction_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "diagonal_reference.json").write_text(
        reference_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    artifact_files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": int(path.stat().st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifact_files
        }
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-seeds", default=",".join(map(str, DATASET_SEEDS)))
    parser.add_argument("--model-seeds", default=",".join(map(str, MODEL_TRAINING_SEEDS)))
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--out", type=Path, default=Path("crossed-seed-initial-direction-output")
    )
    args = parser.parse_args()
    dataset_seeds = tuple(int(v.strip()) for v in args.dataset_seeds.split(",") if v.strip())
    model_seeds = tuple(int(v.strip()) for v in args.model_seeds.split(",") if v.strip())
    if dataset_seeds != DATASET_SEEDS or model_seeds != MODEL_TRAINING_SEEDS:
        raise ValueError("the preregistered benchmark requires the exact 5x5 seed grid")
    if args.episodes_per_cell != 300 or args.epochs != 180 or args.mc_samples != 32:
        raise ValueError("episodes, epochs, and MC samples are fixed by preregistration")
    if args.bootstrap_replicates != 2000:
        raise ValueError("bootstrap replicate count is fixed at 2000")
    args.out.mkdir(parents=True, exist_ok=True)
    reference_path = Path(__file__).parent / "references" / "g8_4_8_7_7_diagonal_position0_reference.json"
    result = _run_matrix(
        dataset_seeds, model_seeds, args.episodes_per_cell, args.epochs,
        args.mc_samples, args.bootstrap_replicates, args.out, reference_path,
    )
    print(json.dumps({
        "output": str(args.out / "crossed_seed_initial_direction_results.json"),
        "dataset_seeds": result["dataset_seeds"],
        "model_training_seeds": result["model_training_seeds"],
        "cells": len(result["cells"]),
        "diagonal_reproduction_passed": len(result["diagonal_legacy_reproduction"]) == 5,
    }, indent=2))


if __name__ == "__main__":
    main()
