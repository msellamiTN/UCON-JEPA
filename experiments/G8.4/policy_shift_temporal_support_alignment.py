"""G8.4.8.7.7 temporal-support alignment audit.

The script keeps JEPA and the matched supervised recursive control fixed, and
changes only the prespecified support used to fit downstream risk calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import PastOnlySoftmaxPredictor
from jepa_numpy import JEPAConfig
from policy_parameter_dataset import build_dataset, write_dataset
from policy_shift_matched_recursive_control import (
    _fit_supervised_action_head, _predict_e2,
)
from policy_shift_recursive_risk import (
    HORIZONS, _apply_platt, _fit_platt,
)
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from predictive_benchmark import metrics as binary_metrics
from uncertain_action_jepa import fit_e1_e2_models


SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
SPLITS = ("test_shifted", "test_reverse")


def _score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    positive, negative = y == 1, y == 0
    return {
        "n": int(len(s)),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "quantiles": [float(x) for x in np.quantile(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
        "positive_mean": float(s[positive].mean()) if positive.any() else None,
        "negative_mean": float(s[negative].mean()) if negative.any() else None,
        "positive_minus_negative_mean": (
            float(s[positive].mean() - s[negative].mean())
            if positive.any() and negative.any() else None
        ),
    }


def _safe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) != 2:
        return {
            "available": False,
            "reason": "only one class in this fixed evaluation stratum",
            "n": int(len(y)),
            "positive": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else None,
        }
    return {"available": True, **binary_metrics(y, scores)}


def _fit_support_calibrators(
    raw: np.ndarray, truth: np.ndarray, positions: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, Any]:
    """Fit C0, CP and CA using calibration arrays only."""
    valid_positions = sorted(int(x) for x in np.unique(positions))
    if not valid_positions or valid_positions[0] != 0:
        raise AssertionError("calibration windows must include start position 0")
    c0 = positions == 0
    if len(np.unique(truth[c0])) != 2:
        raise ValueError("C0 calibration position 0 lacks both classes")
    result: dict[str, Any] = {
        "C0": {
            "scope": "position_0",
            "params": _fit_platt(raw[c0], truth[c0]),
            "calibration_windows": int(c0.sum()),
            "calibration_episodes": int(len(set(episode_ids[c0].tolist()))),
        },
        "CA": {
            "scope": "all_valid_positions",
            "params": _fit_platt(raw, truth),
            "calibration_windows": int(len(raw)),
            "calibration_episodes": int(len(set(episode_ids.tolist()))),
        },
        "CP": {"scope": "separate_per_position", "params_by_position": {}},
    }
    for position in valid_positions:
        mask = positions == position
        if len(np.unique(truth[mask])) != 2:
            raise ValueError(
                f"CP calibration at start_step={position} lacks both classes"
            )
        result["CP"]["params_by_position"][str(position)] = {
            **_fit_platt(raw[mask], truth[mask]),
            "calibration_windows": int(mask.sum()),
            "calibration_episodes": int(len(set(episode_ids[mask].tolist()))),
        }
    return result


def _apply_support_calibrators(
    raw: np.ndarray, positions: np.ndarray, calibrators: dict[str, Any],
) -> dict[str, np.ndarray]:
    predictions = {
        "C0": _apply_platt(raw, calibrators["C0"]["params"]),
        "CA": _apply_platt(raw, calibrators["CA"]["params"]),
    }
    cp = np.empty_like(raw, dtype=np.float64)
    for position, params in calibrators["CP"]["params_by_position"].items():
        mask = positions == int(position)
        cp[mask] = _apply_platt(raw[mask], params)
    predictions["CP"] = cp
    return predictions


def _per_position_report(
    truth: np.ndarray, raw: np.ndarray, positions: np.ndarray,
    episode_ids: np.ndarray, calibrated: dict[str, np.ndarray],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for position in sorted(int(x) for x in np.unique(positions)):
        mask = positions == position
        report[str(position)] = {
            "windows": int(mask.sum()),
            "episodes": int(len(set(episode_ids[mask].tolist()))),
            "prevalence": float(truth[mask].mean()),
            "raw_score": _score_summary(raw[mask], truth[mask]),
            "raw_metrics": _safe_metrics(truth[mask], raw[mask]),
            "calibrated": {
                name: {
                    "score_summary": _score_summary(pred[mask], truth[mask]),
                    "metrics": _safe_metrics(truth[mask], pred[mask]),
                }
                for name, pred in calibrated.items()
            },
        }
    return report


def _group_metrics_safe(
    windows: dict[str, np.ndarray], truth: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy in sorted(set(windows["agent_policy"].tolist())):
        output[policy] = {}
        for family in sorted(set(windows["family"].tolist())):
            mask = (windows["agent_policy"] == policy) & (windows["family"] == family)
            if np.any(mask):
                output[policy][family] = {
                    name: _safe_metrics(truth[mask], pred[mask])
                    for name, pred in predictions.items()
                }
    return output


def _stratified_episode_bootstrap(
    loss_a: np.ndarray, loss_b: np.ndarray, windows: dict[str, np.ndarray],
    seed: int, replicates: int = 2000,
) -> dict[str, Any]:
    """Paired CA-minus-C0 bootstrap, resampling episodes within policy/family."""
    episode_ids = np.asarray(windows["episode_id"])
    policies = np.asarray(windows["agent_policy"])
    families = np.asarray(windows["family"])
    records: dict[str, dict[str, Any]] = {}
    for eid in sorted(set(episode_ids.tolist())):
        mask = episode_ids == eid
        key = f"{policies[np.flatnonzero(mask)[0]]}/{families[np.flatnonzero(mask)[0]]}"
        records[str(eid)] = {
            "stratum": key,
            "difference": float(np.mean(loss_a[mask]) - np.mean(loss_b[mask])),
        }
    strata: dict[str, list[float]] = {}
    for record in records.values():
        strata.setdefault(record["stratum"], []).append(record["difference"])
    if not strata or any(not values for values in strata.values()):
        raise ValueError("empty policy/family episode stratum in bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates, dtype=np.float64)
    for values in strata.values():
        arr = np.asarray(values, dtype=np.float64)
        sampled = rng.integers(0, len(arr), size=(replicates, len(arr)))
        draws += arr[sampled].mean(axis=1) * (len(arr) / len(records))
    observed = float(np.mean([x["difference"] for x in records.values()]))
    return {
        "unit": "whole episode, stratified by behavior archetype/resource family",
        "episodes": len(records),
        "strata": {key: len(value) for key, value in sorted(strata.items())},
        "replicates": replicates,
        "CA_minus_C0_brier": observed,
        "ci95_percentile": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "negative_favours_CA": True,
    }


def _run_seed(
    seed: int, episodes_per_cell: int, epochs: int, mc_samples: int,
    bootstrap_replicates: int, output_dir: Path | None,
) -> dict[str, Any]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
    if output_dir is not None:
        write_dataset(dataset, output_dir / f"dataset-{seed}")

    jepa, jepa_action, _raw_head, scalers = fit_e1_e2_models(
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
        (train["context"] - scalers["context_scaler"][0]) / scalers["context_scaler"][1],
        (train["action"] - scalers["action_scaler"][0]) / scalers["action_scaler"][1],
        (train["future"] - scalers["context_scaler"][0]) / scalers["context_scaler"][1],
        train["security"],
    )
    supervised_action, supervised_temperature = _fit_supervised_action_head(
        supervised, dataset, scalers["context_scaler"]
    )
    models = {
        "JEPA_E2": (jepa, jepa_action, float(scalers["temperature"])),
        "supervised_E2": (supervised, supervised_action, float(supervised_temperature)),
    }

    results: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_key = str(horizon)
        results[horizon_key] = {}
        cal_predictions = {}
        for model_name, (model, action_head, temperature) in models.items():
            cal_predictions[model_name] = _predict_e2(
                model, action_head, dataset["calibration"], horizon,
                scalers["context_scaler"], scalers["action_scaler"],
                temperature, mc_samples, seed + horizon * 10_000,
            )
        for split_name in SPLITS:
            model_reports = {}
            paired_windows = None
            paired_truth = None
            paired_raw = {}
            for model_name, (model, action_head, temperature) in models.items():
                cal = cal_predictions[model_name]
                cal_windows = cal["windows"]
                cal_params = _fit_support_calibrators(
                    cal["raw_risk"], cal["truth"], cal_windows["start_step"],
                    cal_windows["episode_id"],
                )
                test = _predict_e2(
                    model, action_head, dataset[split_name], horizon,
                    scalers["context_scaler"], scalers["action_scaler"],
                    temperature, mc_samples,
                    seed + horizon * 100_000
                    + (0 if split_name == "test_shifted" else 500_000),
                )
                windows, truth, raw = test["windows"], test["truth"], test["raw_risk"]
                calibrated = _apply_support_calibrators(
                    raw, windows["start_step"], cal_params
                )
                model_reports[model_name] = {
                    "calibration": {
                        name: {
                            "scope": value["scope"],
                            "calibration_windows": value["calibration_windows"],
                            "params": value.get("params"),
                            "params_by_position": value.get("params_by_position"),
                        }
                        for name, value in cal_params.items()
                    },
                    "test": {
                        "windows": int(len(truth)),
                        "episodes": int(len(set(windows["episode_id"].tolist()))),
                        "positions": sorted(int(x) for x in np.unique(windows["start_step"])),
                        "prevalence": float(truth.mean()),
                        "raw_score": _score_summary(raw, truth),
                        "raw_metrics": _safe_metrics(truth, raw),
                        "calibrated": {
                            name: {
                                "score_summary": _score_summary(pred, truth),
                                "metrics": _safe_metrics(truth, pred),
                            }
                            for name, pred in calibrated.items()
                        },
                        "per_position": _per_position_report(
                            truth, raw, windows["start_step"], windows["episode_id"], calibrated
                        ),
                        "by_archetype_family": {
                            condition: _group_metrics_safe(
                                windows, truth, {condition: calibrated[condition]}
                            )
                            for condition in ("C0", "CP", "CA")
                        },
                    },
                }
                if model_name == "JEPA_E2":
                    paired_windows, paired_truth = windows, truth
                paired_raw[model_name] = {
                    "raw": raw,
                    "calibrated": calibrated,
                    "windows": windows,
                }

            assert paired_windows is not None and paired_truth is not None
            primary = {}
            for model_name in models:
                ca = paired_raw[model_name]["calibrated"]["CA"]
                c0 = paired_raw[model_name]["calibrated"]["C0"]
                primary[model_name] = _stratified_episode_bootstrap(
                    (ca - paired_truth) ** 2, (c0 - paired_truth) ** 2,
                    paired_raw[model_name]["windows"],
                    seed + horizon * 1000 + (0 if split_name == "test_shifted" else 900_000),
                    bootstrap_replicates,
                )
            tests = {
                "JEPA_minus_supervised_Brier": {},
            }
            for condition in ("raw", "C0", "CP", "CA"):
                if condition == "raw":
                    a = paired_raw["JEPA_E2"]["raw"]
                    b = paired_raw["supervised_E2"]["raw"]
                else:
                    a = paired_raw["JEPA_E2"]["calibrated"][condition]
                    b = paired_raw["supervised_E2"]["calibrated"][condition]
                tests["JEPA_minus_supervised_Brier"][condition] = float(
                    np.mean((a - paired_truth) ** 2 - (b - paired_truth) ** 2)
                )
            results[horizon_key][split_name] = {
                "models": model_reports,
                "primary_CA_minus_C0_episode_bootstrap": primary,
                "JEPA_minus_supervised_Brier": tests["JEPA_minus_supervised_Brier"],
                "window_support": {
                    "positions": sorted(int(x) for x in np.unique(paired_windows["start_step"])),
                    "windows": int(len(paired_truth)),
                    "episodes": int(len(set(paired_windows["episode_id"].tolist()))),
                },
            }

    return {
        "seed": seed,
        "model_capacity": {
            "JEPA_trainable_parameters": jepa.parameter_count(),
            "supervised_trainable_parameters": supervised.parameter_count(),
            "supervised_to_JEPA_ratio": supervised.parameter_count() / jepa.parameter_count(),
        },
        "tests": results,
        "information_boundary": "future actions, observations, security labels, policy identity and start positions never enter E2 model inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("temporal-support-alignment-output"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {
        str(seed): _run_seed(
            seed, args.episodes_per_cell, args.epochs, args.mc_samples,
            args.bootstrap_replicates, args.out,
        )
        for seed in seeds
    }
    result = {
        "benchmark": "G8.4.8.7.7 temporal-support alignment audit",
        "seeds": seeds,
        "runs": runs,
        "protocol": {
            "horizons": list(HORIZONS),
            "splits": {"test_shifted": 0.85, "test_reverse": 0.05},
            "calibration_conditions": {
                "C0": "fit start_step=0, apply all test positions",
                "CP": "fit/apply separately at each matching start_step",
                "CA": "fit on all valid calibration positions, apply all test positions",
            },
            "calibration": "ridge-Platt with fixed score-logit standardization and slope L2 penalty 1.0",
            "bootstrap": "2000 paired whole-episode replicates stratified by behavior archetype/resource family",
            "models": ["JEPA_E2", "supervised_E2"],
        },
        "limitations": [
            "synthetic five-archetype, three-family, scalar-mixture benchmark only",
            "five seeds do not support broad population-level claims",
            "calibration is an offline diagnostic and never affects model inference or UCON authorization",
        ],
    }
    path = args.out / "temporal_support_alignment_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "horizons": HORIZONS}, indent=2))


if __name__ == "__main__":
    main()
