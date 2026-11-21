"""End-to-end G8.4.8.8 randomized intervention benchmark.

All model fitting and threshold selection precede generation of confirmatory
intervention episodes. Pilot power uses nominal-rho synthetic episodes only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from jepa_numpy import JEPAConfig
from policy_parameter_dataset import (
    POLICY_ARCHETYPES, TRAIN_RHO, SHIFT_RHO, REVERSE_RHO, build_dataset,
)
from policy_shift_matched_recursive_control import (
    _fit_supervised_action_head, _transform,
)
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from randomized_intervention_simulator import (
    ARMS, assert_q0_negative_control, fit_alert_threshold,
    make_randomized_assignment, run_episode,
)
from uncertain_action_jepa import (
    _make_windows, fit_e1_e2_models, rollout_predicted_actions,
)
from stateful_trajectory_dataset import FAMILIES


SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
PRIMARY_MDE = 0.05
POWER_TARGET = 0.80
POWER_ALPHA = 0.025  # conservative Bonferroni bound for two superiority contrasts
MAX_EPISODES_PER_ARM_PER_STRATUM = 300
BOOTSTRAP_REPLICATES = 5000


def _fit_models_and_thresholds(
    seed: int, episodes_per_cell: int, epochs: int, mc_samples: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
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
    thresholds: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
        name: {} for name in models
    }
    calibration_manifest = {}
    for name, (model, action_head, temperature) in models.items():
        def score(context: np.ndarray, horizon: int, rollout_seed: int,
                  _model=model, _head=action_head, _temperature=temperature) -> float:
            prediction = rollout_predicted_actions(
                _model, _head, context[None, :],
                scalers["context_scaler"], scalers["action_scaler"],
                _temperature, horizon, mc_samples=mc_samples, seed=rollout_seed,
            )
            return float(prediction["risk_mean"][0])
        score_functions[name] = score
        for horizon in (1, 2, 4, 8):
            windows = _make_windows(dataset["calibration"], horizon)
            prediction = rollout_predicted_actions(
                model, action_head, windows["context"],
                scalers["context_scaler"], scalers["action_scaler"],
                temperature, horizon, mc_samples=mc_samples,
                seed=seed + horizon * 100_000 + (0 if name == "jepa" else 50_000),
            )
            thresholds[name][horizon] = {}
            calibration_manifest.setdefault(name, {})[str(horizon)] = {}
            for position in range(9 - horizon):
                mask = windows["start_step"] == position
                if int(mask.sum()) < 20:
                    raise RuntimeError(
                        f"insufficient calibration support: {name}/H={horizon}/p={position}"
                    )
                fitted = fit_alert_threshold(
                    prediction["risk_mean"][mask],
                    windows["episode_id"][mask],
                    windows["start_step"][mask],
                    alert_rate=0.10,
                )
                thresholds[name][horizon][position] = fitted
                calibration_manifest[name][str(horizon)][str(position)] = {
                    **fitted,
                    "unique_episodes": int(len(set(windows["episode_id"][mask].tolist()))),
                    "support": "calibration partition only",
                    "score_source": "recursive E2 Monte Carlo risk mean",
                }
    # Test partition arrays and their outcome-derived manifest summaries are
    # deliberately excluded from all model/threshold/pilot objects.
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
    partition_ids = {
        key: set(dataset[key]["episode_id"].astype(str).tolist())
        for key in ("train", "validation", "calibration")
    }
    if any(
        partition_ids[a] & partition_ids[b]
        for a, b in (("train", "validation"), ("train", "calibration"),
                     ("validation", "calibration"))
    ):
        raise AssertionError("train/validation/calibration episode leakage")
    del dataset["test_shifted"]
    del dataset["test_reverse"]
    return models, score_functions, {
        "dataset_manifest": manifest,
        "_partition_ids": partition_ids,
        "thresholds": calibration_manifest,
        "_threshold_objects": thresholds,
        "_model_objects": {"models": models, "scalers": scalers},
        "train_episodes": int(len(set(dataset["train"]["episode_id"].tolist()))),
        "validation_episodes": int(len(set(dataset["validation"]["episode_id"].tolist()))),
        "calibration_episodes": int(len(set(dataset["calibration"]["episode_id"].tolist()))),
    }


def _power_for_n(n_per_arm_per_stratum: int, sigma2: float,
                 n_strata: int = 15) -> float:
    if sigma2 <= 0.0:
        return 1.0
    se = math.sqrt(2.0 * sigma2 / (n_per_arm_per_stratum * n_strata))
    noncentrality = PRIMARY_MDE / se
    critical = NormalDist().inv_cdf(1.0 - POWER_ALPHA / 2.0)
    normal = NormalDist()
    return (
        normal.cdf(-critical - noncentrality)
        + 1.0 - normal.cdf(critical - noncentrality)
    )


def _simulate_power(
    n_per_arm_per_stratum: int,
    residual_pools: list[np.ndarray],
    sigma2: float,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 128,
) -> float:
    """Empirical-residual Monte Carlo power for the blocked standardized ITT."""
    n_strata = len(residual_pools)
    if n_per_arm_per_stratum < 1 or n_strata < 1 or replicates < 100:
        raise ValueError("invalid power simulation dimensions")
    if sigma2 <= 1e-12:
        raise RuntimeError("degenerate pilot residual variance; power is not estimable")
    se = math.sqrt(2.0 * sigma2 / (n_per_arm_per_stratum * n_strata))
    critical = NormalDist().inv_cdf(1.0 - POWER_ALPHA / 2.0)
    rng = np.random.default_rng(seed + 1_000_003 * n_per_arm_per_stratum)
    rejected = 0
    completed = 0
    while completed < replicates:
        batch = min(batch_size, replicates - completed)
        standardized_effect = np.zeros(batch, dtype=np.float64)
        for pool in residual_pools:
            control = rng.choice(pool, size=(batch, n_per_arm_per_stratum), replace=True).mean(axis=1)
            treated = rng.choice(pool, size=(batch, n_per_arm_per_stratum), replace=True).mean(axis=1)
            standardized_effect += treated - control
        # Inject the fixed alternative after standardizing the stratum means.
        z = (standardized_effect / n_strata - PRIMARY_MDE) / se
        rejected += int(np.sum(np.abs(z) >= critical))
        completed += batch
    return float(rejected / replicates)


def _power_gate(
    pilot: list[dict[str, Any]], max_per_arm: int,
    *, replicates: int = 5000, seed: int = 884_808,
) -> dict[str, Any]:
    # Residual pools are built within each nominal-rho × arm cell, then pooled
    # over nominal rho to represent each fixed seed × archetype × family stratum.
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in pilot:
        key = (
            int(row["seed"]), str(row["archetype"]), str(row["family"]),
            float(row["rho"]), str(row["arm"]),
        )
        grouped.setdefault(key, {}).setdefault("values", []).append(
            float(row["denied_rate"])
        )
    residual_by_primary_stratum: dict[tuple[int, str, str], list[float]] = {}
    for (seed_value, archetype, family, _rho, arm), entry in grouped.items():
        values = np.asarray(entry["values"], dtype=np.float64)
        if len(values) < 2:
            raise RuntimeError("each pilot stratum × arm requires at least two episodes")
        residuals = values - values.mean()
        residual_by_primary_stratum.setdefault(
            (seed_value, archetype, family), []
        ).extend(residuals.tolist())
    keys = sorted(residual_by_primary_stratum)
    residual_pools = [
        np.asarray(residual_by_primary_stratum[key], dtype=np.float64)
        for key in keys
    ]
    if not residual_pools or any(len(pool) < 20 for pool in residual_pools):
        raise RuntimeError("insufficient empirical residual support for power simulation")
    sigma2 = float(np.mean([np.var(pool, ddof=1) for pool in residual_pools]))
    if sigma2 <= 1e-12:
        raise RuntimeError("degenerate pilot residual variance; power is not estimable")
    n_strata = len(residual_pools)
    if n_strata != 15 * len({int(row["seed"]) for row in pilot}):
        raise AssertionError("pilot does not cover all 15 archetype × family cells per seed")

    analytic_candidate = next(
        (n for n in range(1, max_per_arm + 1)
         if _power_for_n(n, sigma2, n_strata=n_strata) >= POWER_TARGET),
        None,
    )
    power_at_maximum = _simulate_power(
        max_per_arm, residual_pools, sigma2,
        replicates=replicates, seed=seed + 17,
    )
    candidate = analytic_candidate if analytic_candidate is not None else max_per_arm
    candidate_power = _simulate_power(
        candidate, residual_pools, sigma2, replicates=replicates, seed=seed + 31,
    )
    while candidate_power < POWER_TARGET and candidate < max_per_arm:
        candidate += 1
        candidate_power = _simulate_power(
            candidate, residual_pools, sigma2, replicates=replicates, seed=seed + 31,
        )
    while candidate > 1:
        previous_power = _simulate_power(
            candidate - 1, residual_pools, sigma2,
            replicates=replicates, seed=seed + 31,
        )
        if previous_power < POWER_TARGET:
            break
        candidate -= 1
        candidate_power = previous_power

    required = candidate if candidate_power >= POWER_TARGET else None
    selected = required if required is not None else max_per_arm
    return {
        "pilot_source": "independent nominal-rho pilot episodes; no shifted/reverse test outcomes",
        "pilot_episodes": len(pilot),
        "pilot_strata": n_strata,
        "variance_estimator": "empirical residual bootstrap, centered within nominal-rho × arm cells and pooled over nominal rho",
        "pooled_within_stratum_arm_variance": sigma2,
        "minimum_detectable_effect": PRIMARY_MDE,
        "target_power": POWER_TARGET,
        "alpha_two_sided": POWER_ALPHA,
        "power_simulation_replicates": replicates,
        "power_simulation_seed": seed,
        "power_test": "blocked two-sided Wald z test at fixed MDE using empirical-residual Monte Carlo sampling distribution",
        "analytic_candidate_used_only_for_search_initialization": analytic_candidate,
        "simulated_power_at_selected_n": float(candidate_power),
        "simulated_power_at_maximum": float(power_at_maximum),
        "standardized_strata": n_strata,
        "required_episodes_per_arm_per_archetype_family_stratum": required,
        "selected_episodes_per_arm_per_stratum": selected,
        "maximum_episodes_per_arm_per_stratum": max_per_arm,
        "confirmatory_power_gate_passed": required is not None,
        "if_failed": "run at maximum only as explicitly underpowered; do not relax MDE or alpha",
    }


def _run_pilot(
    seed: int, score_functions: dict[str, Any], thresholds: dict[str, Any],
    episodes_per_stratum: int,
) -> list[dict[str, Any]]:
    assignments = make_randomized_assignment(
        seeds=[seed], archetypes=POLICY_ARCHETYPES, families=FAMILIES,
        rhos=TRAIN_RHO, qs=[0.5], horizons=[1],
        episodes_per_stratum=episodes_per_stratum,
        assignment_seed=seed + 884_800,
    )
    return [
        run_episode(spec, score_functions=score_functions, thresholds=thresholds)
        for spec in assignments
    ]


def _bootstrap_contrast(
    rows: list[dict[str, Any]], arm_a: str, arm_b: str,
    replicates: int, seed: int,
) -> dict[str, Any]:
    selected = [
        r for r in rows
        if r["rho"] == SHIFT_RHO and r["q"] == 0.5 and r["horizon"] == 1
        and r["arm"] in (arm_a, arm_b)
    ]
    groups: dict[tuple[int, str, str], dict[str, list[float]]] = {}
    for row in selected:
        key = (int(row["seed"]), str(row["archetype"]), str(row["family"]))
        groups.setdefault(key, {arm_a: [], arm_b: []})[row["arm"]].append(
            float(row["denied_rate"])
        )
    keys = sorted(groups)
    if len(keys) != len(set((s, a, f) for s, a, f in keys)) or not keys:
        raise RuntimeError("invalid primary-analysis strata")
    for key in keys:
        if not groups[key][arm_a] or not groups[key][arm_b]:
            raise RuntimeError(f"missing randomized arm in stratum {key}")
    observed = float(np.mean([
        np.mean(groups[k][arm_a]) - np.mean(groups[k][arm_b]) for k in keys
    ]))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        diffs = []
        for key in keys:
            a = np.asarray(groups[key][arm_a], dtype=np.float64)
            bb = np.asarray(groups[key][arm_b], dtype=np.float64)
            sa = rng.choice(a, size=len(a), replace=True).mean()
            sb = rng.choice(bb, size=len(bb), replace=True).mean()
            diffs.append(sa - sb)
        draws[b] = np.mean(diffs)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    # Centered bootstrap tail probability, reported as a bootstrap p-value
    # diagnostic; the randomization test below is the confirmatory p-value.
    centered = draws - observed
    p_boot = (1 + np.sum(np.abs(centered) >= abs(observed))) / (replicates + 1)
    return {
        "contrast": f"{arm_a} minus {arm_b}",
        "unit": "whole episode",
        "strata": len(keys),
        "episodes_arm_a": int(sum(len(groups[k][arm_a]) for k in keys)),
        "episodes_arm_b": int(sum(len(groups[k][arm_b]) for k in keys)),
        "estimate": observed,
        "bootstrap_ci95_percentile": [float(lo), float(hi)],
        "bootstrap_p_diagnostic": float(p_boot),
        "replicates": replicates,
        "negative_favours_arm_a": True,
    }


def _randomization_pvalue(
    rows: list[dict[str, Any]], arm_a: str, arm_b: str,
    replicates: int, seed: int,
) -> float:
    selected = [
        r for r in rows
        if r["rho"] == SHIFT_RHO and r["q"] == 0.5 and r["horizon"] == 1
        and r["arm"] in (arm_a, arm_b)
    ]
    groups: dict[tuple[int, str, str], tuple[list[float], list[float]]] = {}
    for row in selected:
        key = (int(row["seed"]), str(row["archetype"]), str(row["family"]))
        a, b = groups.setdefault(key, ([], []))
        (a if row["arm"] == arm_a else b).append(float(row["denied_rate"]))
    observed = np.mean([
        np.mean(groups[k][0]) - np.mean(groups[k][1]) for k in groups
    ])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(replicates):
        diffs = []
        for a_values, b_values in groups.values():
            combined = np.asarray(a_values + b_values, dtype=np.float64)
            order = rng.permutation(len(combined))
            n_a = len(a_values)
            diffs.append(combined[order[:n_a]].mean() - combined[order[n_a:]].mean())
        if abs(float(np.mean(diffs))) >= abs(float(observed)):
            extreme += 1
    return float((extreme + 1) / (replicates + 1))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _analyze(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    contrasts = {
        "JEPA_vs_UCON": ("jepa_advisory", "ucon_only"),
        "JEPA_vs_random": ("jepa_advisory", "random_advisory"),
        "JEPA_vs_supervised": ("jepa_advisory", "supervised_advisory"),
    }
    boot = {
        name: _bootstrap_contrast(rows, a, b, replicates, seed + i * 101)
        for i, (name, (a, b)) in enumerate(contrasts.items())
    }
    p_raw = {
        name: _randomization_pvalue(rows, *contrasts[name], replicates, seed + i * 1009)
        for i, name in enumerate(("JEPA_vs_UCON", "JEPA_vs_random"))
    }
    p_adjusted = _holm(p_raw)
    ni_upper = boot["JEPA_vs_supervised"]["bootstrap_ci95_percentile"][1]
    return {
        "primary_population": {"rho": SHIFT_RHO, "q": 0.5, "horizon": 1},
        "standardization": "equal weight over seed x behavior-archetype x resource-family strata",
        "superiority_contrasts": {
            name: {
                **boot[name], "randomization_p": p_raw[name],
                "holm_adjusted_p": p_adjusted[name],
            }
            for name in ("JEPA_vs_UCON", "JEPA_vs_random")
        },
        "noninferiority": {
            **boot["JEPA_vs_supervised"],
            "margin": 0.02,
            "one_sided_alpha": 0.025,
            "upper_97_5_percentile_bound": ni_upper,
            "noninferiority_supported": bool(ni_upper < 0.02),
        },
        "inference": {
            "bootstrap": "stratified whole-episode resampling",
            "randomization": "within-stratum arm-label permutation preserving arm counts",
            "replicates": replicates,
            "no_transition_or_window_level_pseudoreplication": True,
        },
    }


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


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


def _save_model_checkpoint(
    path: Path, models: dict[str, Any], scalers: dict[str, Any],
) -> str:
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


def _mean_defined(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _cohort_descriptives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["arm"], float(row["rho"]), float(row["q"]), int(row["horizon"]),
            int(row["seed"]), row["archetype"], row["family"],
        )
        groups.setdefault(key, []).append(row)
    output = {}
    for key, values in sorted(groups.items()):
        output["|".join(map(str, key))] = {
            "episodes": len(values),
            "mean_denied_rate": float(np.mean([x["denied_rate"] for x in values])),
            "any_deny_rate": float(np.mean([x["any_deny"] for x in values])),
            "task_completion_rate": float(np.mean([x["task_complete"] for x in values])),
            "mean_warning_rate": float(np.mean([x["warning_rate"] for x in values])),
            "mean_warning_precision": _mean_defined([x["warning_precision"] for x in values]),
            "mean_warning_recall": _mean_defined([x["warning_recall"] for x in values]),
            "mean_revision_count": float(np.mean([x["revision_count"] for x in values])),
            "mean_distinct_action_count": float(np.mean([x["distinct_action_count"] for x in values])),
            "mean_repeat_after_deny_count": float(np.mean([x["repeat_after_deny_count"] for x in values])),
            "state_violation_count": int(sum(x["state_violation_count"] for x in values)),
        }
    return output


def _write_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(_native(row), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    path.write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--pilot-episodes-per-stratum", type=int, default=40)
    parser.add_argument("--max-episodes-per-arm-per-stratum", type=int,
                        default=MAX_EPISODES_PER_ARM_PER_STRATUM)
    parser.add_argument("--power-simulation-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--out", type=Path, default=Path("randomized-intervention-output"))
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("provide distinct seeds")
    if args.max_episodes_per_arm_per_stratum > MAX_EPISODES_PER_ARM_PER_STRATUM:
        raise ValueError("sample-size cap exceeds preregistered maximum")
    args.out.mkdir(parents=True, exist_ok=True)

    per_seed: dict[int, dict[str, Any]] = {}
    thresholds_by_seed = {}
    checkpoint_hashes = {}
    for seed in seeds:
        models, score_functions, calibration = _fit_models_and_thresholds(
            seed, args.episodes_per_cell, args.epochs, args.mc_samples
        )
        # Retain unrounded threshold structures for scoring.
        thresholds_by_seed[seed] = calibration.pop("_threshold_objects")
        partition_ids = calibration.pop("_partition_ids")
        model_objects = calibration.pop("_model_objects")
        checkpoint_path = args.out / f"model_checkpoint_seed_{seed}.npz"
        checkpoint_hashes[str(seed)] = _save_model_checkpoint(
            checkpoint_path, model_objects["models"], model_objects["scalers"]
        )
        per_seed[seed] = {
            "model_objects": model_objects,
            "partition_ids": partition_ids,
            "models": {
                name: {
                    "parameter_count": int(model.parameter_count()),
                    "latent_dim": int(model.config.latent_dim),
                    "action_head_selected_epoch": int(getattr(_head, "best_epoch", 0)),
                    "action_temperature": float(_temp),
                }
                for name, (model, _head, _temp) in models.items()
            },
            "calibration": calibration,
            "score_functions": score_functions,
        }

    pilot_rows = []
    for seed in seeds:
        pilot_rows.extend(_run_pilot(
            seed, per_seed[seed]["score_functions"], thresholds_by_seed[seed],
            args.pilot_episodes_per_stratum,
        ))
    power = _power_gate(
        pilot_rows, args.max_episodes_per_arm_per_stratum,
        replicates=args.power_simulation_replicates, seed=884_808,
    )
    n_primary = int(power["selected_episodes_per_arm_per_stratum"])
    n_secondary = min(40, n_primary)

    assignments = []
    for seed in seeds:
        assignments.extend(make_randomized_assignment(
            seeds=[seed], archetypes=POLICY_ARCHETYPES, families=FAMILIES,
            rhos=[SHIFT_RHO], qs=[0.5], horizons=[1],
            episodes_per_stratum=4 * n_primary, assignment_seed=seed + 884_801,
        ))
        # Prespecified secondary cohorts: q=0 and q=1 at H=1; reverse-shift
        # q=.5 at H=1; H=2/4/8 at shifted q=.5. Each has smaller fixed N.
        secondary_specs = (
            ([SHIFT_RHO], [0.0, 1.0], [1]),
            ([REVERSE_RHO], [0.5], [1]),
            ([SHIFT_RHO], [0.5], [2, 4, 8]),
        )
        for rhos, qs, horizons in secondary_specs:
            assignments.extend(make_randomized_assignment(
                seeds=[seed], archetypes=POLICY_ARCHETYPES, families=FAMILIES,
                rhos=rhos, qs=qs, horizons=horizons,
                episodes_per_stratum=4 * n_secondary,
                assignment_seed=seed + 884_802 + len(assignments),
            ))

    # Assignment IDs must be unique and disjoint from all model-development
    # partitions before any confirmatory episode is simulated.
    assigned_ids = [str(spec["episode_id"]) for spec in assignments]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise AssertionError("an episode was assigned more than once")
    for seed in seeds:
        development_ids = set().union(*per_seed[seed]["partition_ids"].values())
        test_ids = {
            str(spec["episode_id"]) for spec in assignments
            if int(spec["seed"]) == seed
        }
        if development_ids & test_ids:
            raise AssertionError(f"development/test episode overlap for seed {seed}")
    all_results = []
    q0_checks = 0
    for spec in assignments:
        seed = int(spec["seed"])
        result = run_episode(
            spec, score_functions=per_seed[seed]["score_functions"],
            thresholds=thresholds_by_seed[seed],
        )
        all_results.append(result)
        if float(spec["q"]) == 0.0 and spec["arm"] in ("jepa_advisory", "supervised_advisory"):
            assert_q0_negative_control(
                spec, score_functions=per_seed[seed]["score_functions"],
                thresholds=thresholds_by_seed[seed],
            )
            q0_checks += 1

    primary_analysis = _analyze(
        all_results, args.bootstrap_replicates, seed=991_884_808
    )
    arm_summaries = {}
    for arm in ARMS:
        selected = [r for r in all_results if r["arm"] == arm]
        arm_summaries[arm] = {
            "episodes": len(selected),
            "mean_denied_rate": float(np.mean([r["denied_rate"] for r in selected])),
            "task_completion_rate": float(np.mean([r["task_complete"] for r in selected])),
            "warning_rate": float(np.mean([r["warning_rate"] for r in selected])),
            "state_violation_count": int(sum(r["state_violation_count"] for r in selected)),
        }
    episode_artifact = args.out / "episode_results.jsonl.gz"
    assignment_artifact = args.out / "randomization_manifest.jsonl.gz"
    pilot_artifact = args.out / "validation_pilot_results.jsonl.gz"
    episode_sha = _write_jsonl_gzip(episode_artifact, all_results)
    assignment_sha = _write_jsonl_gzip(assignment_artifact, assignments)
    pilot_sha = _write_jsonl_gzip(pilot_artifact, pilot_rows)
    result = {
        "benchmark": "G8.4.8.8 randomized predictive-intervention utility",
        "protocol_version": "g8.4.8.8-v1",
        "seeds": seeds,
        "power_gate": power,
        "primary_analysis": primary_analysis,
        "inferential_status": (
            "confirmatory_power_gate_passed"
            if power["confirmatory_power_gate_passed"]
            else "underpowered_exploratory_only"
        ),
        "all_arm_descriptives": arm_summaries,
        "stratified_cohort_descriptives": _cohort_descriptives(all_results),
        "q0_exact_negative_control_replays": q0_checks,
        "validation_pilot_descriptives": _cohort_descriptives(pilot_rows),
        "retained_pilot_artifact": {"file": pilot_artifact.name, "sha256": pilot_sha},
        "calibration": {
            str(seed): per_seed[seed]["calibration"] for seed in seeds
        },
        "model_capacity": {
            str(seed): per_seed[seed]["models"] for seed in seeds
        },
        "model_checkpoints": {
            str(seed): {
                "file": f"model_checkpoint_seed_{seed}.npz",
                "sha256": checkpoint_hashes[str(seed)],
            } for seed in seeds
        },
        "retained_episode_artifact": {"file": episode_artifact.name, "sha256": episode_sha},
        "retained_assignment_artifact": {"file": assignment_artifact.name, "sha256": assignment_sha},
        "limitations": [
            "synthetic action-substitution response model; not an LLM-agent behavior model",
            "five hand-designed policy archetypes and three resource families",
            "power is estimated from nominal-rho synthetic pilot variance and assumes comparable shifted-test variance",
            "JEPA has an EMA target encoder while the supervised comparator does not; report parameter counts and do not claim capacity equivalence",
            "formal UCON remains the sole authorization authority",
        ],
    }
    output = args.out / "randomized_intervention_results.json"
    output.write_text(json.dumps(_native(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "files": {
            output.name: digest,
            episode_artifact.name: episode_sha,
            assignment_artifact.name: assignment_sha,
            pilot_artifact.name: pilot_sha,
            **{
                f"model_checkpoint_seed_{seed}.npz": checkpoint_hashes[str(seed)]
                for seed in seeds
            },
        },
        "seeds": seeds, "power_gate": _native(power),
        "thresholds_frozen_before_confirmatory_episode_generation": True,
        "episode_count": len(all_results),
        "configuration": {
            "epochs": args.epochs, "episodes_per_cell": args.episodes_per_cell,
            "mc_samples": args.mc_samples,
            "pilot_episodes_per_stratum": args.pilot_episodes_per_stratum,
            "power_simulation_replicates": args.power_simulation_replicates,
            "bootstrap_replicates": args.bootstrap_replicates,
            "max_episodes_per_arm_per_stratum": args.max_episodes_per_arm_per_stratum,
        },
        "runtime": {
            "python": sys.version, "numpy": np.__version__,
            "platform": platform.platform(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "ucon_jepa_head_sha": os.environ.get("UCON_JEPA_HEAD_SHA"),
        },
        "episode_unit": "one complete eight-transition synthetic episode",
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out / "SHA256SUMS.txt").write_text(
        f"{digest}  {output.name}\n{episode_sha}  {episode_artifact.name}\n"
        f"{assignment_sha}  {assignment_artifact.name}\n"
        f"{pilot_sha}  {pilot_artifact.name}\n"
        + "".join(
            f"{checkpoint_hashes[str(seed)]}  model_checkpoint_seed_{seed}.npz\n"
            for seed in seeds
        ), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output), "sha256": digest,
        "power_gate_passed": power["confirmatory_power_gate_passed"],
        "episodes": len(all_results), "primary": primary_analysis,
    }, indent=2))


if __name__ == "__main__":
    main()
