"""G8.4.8.7.4 recursive E1/E2 risk forecasting under controlled rho shift.

E1 is a privileged known-action diagnostic. E2 predicts its own future actions
and recursively rolls the JEPA latent state forward. Both remain advisory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from policy_parameter_dataset import build_dataset, write_dataset
from predictive_benchmark import metrics as binary_metrics
from uncertain_action_jepa import (
    _make_windows, _transform, fit_e1_e2_models,
    rollout_known_actions, rollout_predicted_actions,
)

EPS = 1e-8
HORIZONS = (1, 2, 4, 8)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Fit a two-parameter logistic calibrator on nominal calibration episodes."""
    x = _logit(scores)
    y = np.asarray(labels, dtype=np.float64)
    if len(x) != len(y) or len(np.unique(y)) != 2:
        raise ValueError("risk calibration requires aligned examples from both classes")
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.asarray([0.0, 1.0], dtype=np.float64)
    penalty = np.diag([1e-8, 1e-5])
    for _ in range(100):
        p = _sigmoid(design @ beta)
        weights = np.maximum(p * (1.0 - p), 1e-8)
        gradient = design.T @ (p - y) + penalty @ beta
        hessian = (design.T * weights) @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.linalg.norm(step) < 1e-9:
            break
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("non-finite Platt calibration parameters")
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def _apply_platt(scores: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return _sigmoid(params["intercept"] + params["slope"] * _logit(scores))


def _predict_risk_pair(
    model: Any, action_model: Any, part: dict[str, np.ndarray],
    horizon: int, scalers: dict[str, Any], mc_samples: int, seed: int,
) -> dict[str, Any]:
    windows = _make_windows(part, horizon)
    truth = windows["security"][:, :, 0].max(axis=1).astype(np.int64)
    e1_risk, e1_state = rollout_known_actions(
        model, windows["context"], windows["actions"],
        scalers["context_scaler"], scalers["action_scaler"],
    )
    e2 = rollout_predicted_actions(
        model, action_model, windows["context"], scalers["context_scaler"],
        scalers["action_scaler"], scalers["temperature"], horizon,
        mc_samples=mc_samples, seed=seed,
    )
    return {
        "windows": windows, "truth": truth,
        "e1_raw": np.clip(e1_risk, EPS, 1.0 - EPS),
        "e2_raw": np.clip(e2["risk_mean"], EPS, 1.0 - EPS),
        "e2_std": e2["risk_std"],
        "e2_q05": e2["risk_q05"], "e2_q95": e2["risk_q95"],
        "e1_terminal_state": e1_state,
        "e2_terminal_state": e2["terminal_state_mean"],
    }


def _paired_loss_bootstrap(
    e2_loss: np.ndarray, e1_loss: np.ndarray, episode_ids: np.ndarray,
    seed: int, replicates: int,
) -> dict[str, Any]:
    ids = sorted(set(episode_ids.tolist()))
    differences = np.asarray([
        float(e2_loss[episode_ids == episode_id].mean() - e1_loss[episode_ids == episode_id].mean())
        for episode_id in ids
    ])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    means = differences[draws].mean(axis=1)
    return {
        "unit": "whole episode", "episodes": len(ids), "replicates": replicates,
        "e2_minus_e1_mean": float(differences.mean()),
        "ci95_percentile": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
        "negative_favours_E2": True,
    }


def _group_metrics(windows: dict[str, np.ndarray], truth: np.ndarray,
                   predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    output = {}
    for policy in sorted(set(windows["agent_policy"].tolist())):
        output[policy] = {}
        for family in sorted(set(windows["family"].tolist())):
            mask = (windows["agent_policy"] == policy) & (windows["family"] == family)
            if np.any(mask):
                output[policy][family] = {
                    name: binary_metrics(truth[mask], pred[mask])
                    for name, pred in predictions.items()
                }
    return output


def run_seed(seed: int, episodes_per_cell: int = 300, epochs: int = 180,
             mc_samples: int = 32, bootstrap_replicates: int = 2000,
             out_dir: Path | None = None) -> dict[str, Any]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
    if out_dir is not None:
        write_dataset(dataset, out_dir / f"dataset-{seed}")
    model, action_model, _raw_action_model, scalers = fit_e1_e2_models(
        dataset, seed=seed, epochs=epochs
    )
    calibration_predictions = {}
    for horizon in HORIZONS:
        calibration_predictions[horizon] = _predict_risk_pair(
            model, action_model, dataset["calibration"], horizon, scalers,
            mc_samples, seed + horizon * 10_000,
        )
    calibrators = {}
    for horizon, pred in calibration_predictions.items():
        calibrators[horizon] = {
            "E1_known_action_privileged": _fit_platt(pred["e1_raw"], pred["truth"]),
            "E2_predicted_action": _fit_platt(pred["e2_raw"], pred["truth"]),
        }

    tests = {}
    for split_name in ("test_shifted", "test_reverse"):
        tests[split_name] = {}
        for horizon in HORIZONS:
            pred = _predict_risk_pair(
                model, action_model, dataset[split_name], horizon, scalers,
                mc_samples, seed + horizon * 100_000
                + (0 if split_name == "test_shifted" else 500_000),
            )
            truth, windows = pred["truth"], pred["windows"]
            e1 = _apply_platt(pred["e1_raw"], calibrators[horizon]["E1_known_action_privileged"])
            e2 = _apply_platt(pred["e2_raw"], calibrators[horizon]["E2_predicted_action"])
            train_windows = _make_windows(dataset["train"], horizon)
            train_horizon_prevalence = float(train_windows["security"][:, :, 0].max(axis=1).mean())
            baseline = np.full(len(truth), train_horizon_prevalence, dtype=np.float64)
            predictions = {"train_prevalence": baseline, "E1_known_action_privileged": e1,
                            "E2_predicted_action": e2}
            e1_loss, e2_loss = (e1 - truth) ** 2, (e2 - truth) ** 2
            tests[split_name][str(horizon)] = {
                "target": "any formal UCON DENY across the next H transitions",
                "rho": sorted(set(float(x) for x in dataset[split_name]["rho"].tolist())),
                "windows": int(len(truth)),
                "episodes": int(len(set(windows["episode_id"].tolist()))),
                "prevalence": float(truth.mean()),
                "train_horizon_prevalence": train_horizon_prevalence,
                "metrics": {name: binary_metrics(truth, p) for name, p in predictions.items()},
                "E2_uncertainty": {
                    "mean_mc_standard_deviation": float(pred["e2_std"].mean()),
                    "mean_raw_q05": float(pred["e2_q05"].mean()),
                    "mean_raw_q95": float(pred["e2_q95"].mean()),
                    "mc_samples": mc_samples,
                },
                "paired_episode_bootstrap_E2_minus_E1_Brier": _paired_loss_bootstrap(
                    e2_loss, e1_loss, windows["episode_id"],
                    seed + horizon * 1000 + (0 if split_name == "test_shifted" else 900_000),
                    bootstrap_replicates,
                ),
                "by_archetype_family": _group_metrics(windows, truth, predictions),
            }
    return {
        "seed": seed, "dataset_manifest": dataset["manifest"],
        "model": {
            "parameter_count": model.parameter_count(),
            "ema_target_distance": model.ema_target_distance(),
            "final_latent_loss": model.history.latent_loss[-1],
            "final_security_loss": model.history.security_loss[-1],
            "minimum_predicted_batch_std": min(model.history.predicted_std_min),
        },
        "risk_calibrators": {str(h): calibrators[h] for h in HORIZONS},
        "calibration_scope": "nominal-support calibration episodes only; separate calibrator per horizon and E1/E2 condition",
        "tests": tests,
        "information_boundary": {
            "E1": "receives realized future actions; privileged diagnostic only",
            "E2": "receives initial context only and samples its own actions recursively from the past-only JEPA action head",
            "future_actions_observations_and_security_labels": "never E2 inference inputs",
            "authorization": "formal UCON remains sole authority; predictive outputs cannot authorize or override DENY",
        },
    }


def summarize(runs: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for split in ("test_shifted", "test_reverse"):
        result[split] = {}
        for horizon in map(str, HORIZONS):
            result[split][horizon] = {}
            for method in ("train_prevalence", "E1_known_action_privileged", "E2_predicted_action"):
                values = [r["tests"][split][horizon]["metrics"][method] for r in runs.values()]
                result[split][horizon][method] = {
                    key: {"mean": float(np.mean([v[key] for v in values])),
                          "sd_across_seeds": float(np.std([v[key] for v in values], ddof=1))}
                    for key in ("auroc", "auprc", "brier", "ece_10", "prevalence")
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("policy-shift-recursive-risk-output"))
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {str(seed): run_seed(seed, args.episodes_per_cell, args.epochs,
                                args.mc_samples, args.bootstrap_replicates, args.out) for seed in seeds}
    result = {
        "benchmark": "G8.4.8.7.4 recursive E1/E2 future-DENY forecasting under controlled rho shift",
        "seeds": seeds, "runs": runs, "summary": summarize(runs),
        "protocol": {
            "train_rho": [0.20, 0.40, 0.60], "test_shifted_rho": 0.85,
            "test_reverse_rho": 0.05, "horizons": list(HORIZONS),
            "episodes_per_cell": args.episodes_per_cell,
            "calibration": "per-horizon Platt calibration fitted only on nominal-support calibration episodes",
            "bootstrap_unit": "whole episode",
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "limitations": [
            "synthetic behavior archetypes and one scalar within-archetype mixture parameter only",
            "E1 is privileged and not deployable; E2 recursively uses predicted actions and latent states",
            "risk decoder and transition model are linear reference components",
            "no real-agent transfer, intervention utility, or production authorization claim",
        ],
    }
    path = args.out / "recursive_risk_forecasting_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "horizons": HORIZONS}, indent=2))


if __name__ == "__main__":
    main()
