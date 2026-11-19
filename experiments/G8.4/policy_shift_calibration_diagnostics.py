"""G8.4.8.7.6 calibration-direction and recursive JEPA failure diagnosis.

All calibrators are fitted on nominal-support calibration episodes only.
Shifted/reverse test partitions are diagnostic-only and never tune models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import _metrics
from policy_parameter_dataset import build_dataset, write_dataset
from policy_shift_recursive_risk import (
    EPS, HORIZONS, _apply_platt, _fit_platt, _make_windows, _transform,
)
from predictive_benchmark import metrics as binary_metrics
from uncertain_action_jepa import (
    _action_features, _calibrated_action_probs, _family_onehot,
    fit_e1_e2_models, rollout_predicted_actions,
)


def _fit_platt_nonnegative(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calibration-only monotone Platt diagnostic with standardized score logits."""
    x = np.log(np.clip(scores, EPS, 1.0 - EPS) / (1.0 - np.clip(scores, EPS, 1.0 - EPS)))
    y = np.asarray(labels, dtype=np.float64)
    if len(x) != len(y) or len(np.unique(y)) != 2:
        raise ValueError("calibration requires aligned scores and both classes")
    center, scale = float(x.mean()), float(x.std())
    if scale < 1e-8:
        scale = 1.0
    z = (x - center) / scale
    design = np.column_stack((np.ones(len(z)), z))
    prevalence = float(np.clip(y.mean(), EPS, 1.0 - EPS))
    beta = np.asarray([np.log(prevalence / (1.0 - prevalence)), 0.0])
    penalty = np.diag([0.0, 1.0])

    def objective(b: np.ndarray) -> float:
        eta = design @ b
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * b @ penalty @ b)

    current = objective(beta)
    for iteration in range(1, 201):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35, 35)))
        w = np.maximum(p * (1.0 - p), 1e-10)
        grad = design.T @ (p - y) + penalty @ beta
        hess = (design.T * w) @ design + penalty
        step = np.linalg.solve(hess, grad)
        if not np.all(np.isfinite(step)):
            raise FloatingPointError("non-finite constrained Platt update")
        if np.linalg.norm(step) < 1e-8:
            break
        factor, accepted = 1.0, False
        for _ in range(40):
            candidate = beta - factor * step
            candidate[1] = max(0.0, candidate[1])
            value = objective(candidate)
            if np.isfinite(value) and value <= current + 1e-10:
                beta, current, accepted = candidate, value, True
                break
            factor *= 0.5
        if not accepted or np.linalg.norm(factor * step) < 1e-8:
            break
    if not np.all(np.isfinite(beta)) or beta[1] < -1e-12:
        raise FloatingPointError("invalid nonnegative-slope Platt solution")
    return {
        "intercept": float(beta[0]), "slope": float(max(0.0, beta[1])),
        "center": center, "scale": scale, "slope_l2_penalty": 1.0,
        "constraint": "slope >= 0", "iterations": int(iteration),
    }


def _score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return {
        "n": int(len(s)),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "quantiles": [float(v) for v in np.quantile(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
        "positive_mean": float(s[y == 1].mean()) if np.any(y == 1) else None,
        "negative_mean": float(s[y == 0].mean()) if np.any(y == 0) else None,
        "positive_minus_negative_mean": (
            float(s[y == 1].mean() - s[y == 0].mean())
            if np.any(y == 1) and np.any(y == 0) else None
        ),
    }


def _instrumented_rollout(
    model: Any, action_model: Any, context: np.ndarray,
    context_scaler: tuple[np.ndarray, np.ndarray],
    action_scaler: tuple[np.ndarray, np.ndarray],
    temperature: float, horizon: int, mc_samples: int, seed: int,
) -> dict[str, Any]:
    """Mirror E2 inference while recording per-step latent/action/risk diagnostics."""
    family = _family_onehot(context)
    initial = _transform(context, context_scaler) @ model.W_context
    latent = np.repeat(initial, mc_samples, axis=0)
    family_repeated = np.repeat(family, mc_samples, axis=0)
    rng = np.random.default_rng(seed)
    survival = np.ones(len(latent), dtype=np.float64)
    steps = []
    for step in range(horizon):
        probs = _calibrated_action_probs(
            action_model, family_repeated, latent, temperature
        )
        entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=1)
        uniforms = rng.random(len(probs))
        cumulative = np.cumsum(probs, axis=1)
        sampled = np.sum(uniforms[:, None] > cumulative, axis=1)
        sampled = np.minimum(sampled, probs.shape[1] - 1)
        actions = np.zeros_like(probs)
        actions[np.arange(len(sampled)), sampled] = 1.0
        action_latent = _transform(actions, action_scaler) @ model.W_action
        latent = model.predict_from_latents(latent, action_latent)
        decoded = np.clip(model.decode_security(latent), 0.0, 1.0)
        survival *= (1.0 - decoded[:, 0])
        steps.append({
            "step": step + 1,
            "mean_action_entropy": float(entropy.mean()),
            "mean_action_probability": probs.mean(axis=0).tolist(),
            "sampled_action_frequency": actions.mean(axis=0).tolist(),
            "mean_max_action_probability": float(probs.max(axis=1).mean()),
            "mean_latent_norm": float(np.linalg.norm(latent, axis=1).mean()),
            "latent_norm_sd": float(np.linalg.norm(latent, axis=1).std()),
            "mean_decoded_deny_probability": float(decoded[:, 0].mean()),
            "decoded_deny_probability_sd": float(decoded[:, 0].std()),
        })
    risk_samples = (1.0 - survival).reshape(len(context), mc_samples)
    return {
        "risk_mean": risk_samples.mean(axis=1),
        "risk_std": risk_samples.std(axis=1),
        "steps": steps,
    }


def _run_seed(seed: int, episodes_per_cell: int = 300, epochs: int = 180,
              mc_samples: int = 32, out_dir: Path | None = None) -> dict[str, Any]:
    dataset = build_dataset(seed=seed, episodes_per_cell=episodes_per_cell)
    if out_dir is not None:
        write_dataset(dataset, out_dir / f"dataset-{seed}")
    model, action_model, _raw_head, scalers = fit_e1_e2_models(
        dataset, seed=seed, epochs=epochs
    )

    # Observed-context action prediction is a separate diagnostic. It is not
    # substituted for the recursive predicted-latent action sequence.
    test_rows = dataset["test_shifted"]
    latent = _transform(test_rows["context"], scalers["context_scaler"]) @ model.W_context
    action_features = _action_features(_family_onehot(test_rows["context"]), latent)
    observed_probs = action_model.predict_proba(action_features)
    observed_action_metrics = _metrics(observed_probs, test_rows["action"])

    results: dict[str, Any] = {}
    for split_name in ("test_shifted", "test_reverse"):
        results[split_name] = {}
        for horizon in HORIZONS:
            cal = _make_windows(dataset["calibration"], horizon)
            cal_raw_rollout = rollout_predicted_actions(
                model, action_model, cal["context"], scalers["context_scaler"],
                scalers["action_scaler"], scalers["temperature"], horizon,
                mc_samples=mc_samples, seed=seed + horizon * 10_000,
            )
            cal_raw = np.clip(cal_raw_rollout["risk_mean"], EPS, 1.0 - EPS)
            cal_truth = cal["security"][:, :, 0].max(axis=1).astype(np.int64)
            cal_start0 = cal["start_step"] == 0
            cal_params = _fit_platt(cal_raw[cal_start0], cal_truth[cal_start0])
            mono_params = _fit_platt_nonnegative(cal_raw[cal_start0], cal_truth[cal_start0])

            test = _make_windows(dataset[split_name], horizon)
            test_seed = seed + horizon * 100_000 + (
                0 if split_name == "test_shifted" else 500_000
            )
            standard = rollout_predicted_actions(
                model, action_model, test["context"], scalers["context_scaler"],
                scalers["action_scaler"], scalers["temperature"], horizon,
                mc_samples=mc_samples, seed=test_seed,
            )
            instrumented = _instrumented_rollout(
                model, action_model, test["context"], scalers["context_scaler"],
                scalers["action_scaler"], scalers["temperature"], horizon,
                mc_samples, test_seed,
            )
            if not np.allclose(standard["risk_mean"], instrumented["risk_mean"], atol=1e-12):
                raise AssertionError("instrumented rollout diverges from the production E2 rollout")
            truth = test["security"][:, :, 0].max(axis=1).astype(np.int64)
            test_start0 = test["start_step"] == 0
            calibrated = _apply_platt(standard["risk_mean"], cal_params)
            monotone = _apply_platt(standard["risk_mean"], mono_params)

            results[split_name][str(horizon)] = {
                "rho": sorted(set(float(v) for v in dataset[split_name]["rho"].tolist())),
                "calibration": {
                    "episodes": int(cal_start0.sum()),
                    "positive_rate": float(cal_truth[cal_start0].mean()),
                    "raw_score": _score_summary(cal_raw[cal_start0], cal_truth[cal_start0]),
                    "unconstrained_platt": cal_params,
                    "nonnegative_platt": mono_params,
                },
                "test": {
                    "all_windows": {
                        "raw_score": _score_summary(standard["risk_mean"], truth),
                        "raw_metrics": binary_metrics(truth, standard["risk_mean"]),
                        "unconstrained_platt_metrics": binary_metrics(truth, calibrated),
                        "nonnegative_platt_metrics": binary_metrics(truth, monotone),
                        "monotone_calibration_mean_abs_change": float(
                            np.mean(np.abs(monotone - calibrated))
                        ),
                    },
                    "step0_only": {
                        "raw_score": _score_summary(
                            standard["risk_mean"][test_start0], truth[test_start0]
                        ),
                        "raw_metrics": binary_metrics(
                            truth[test_start0], standard["risk_mean"][test_start0]
                        ),
                        "unconstrained_platt_metrics": binary_metrics(
                            truth[test_start0], calibrated[test_start0]
                        ),
                        "nonnegative_platt_metrics": binary_metrics(
                            truth[test_start0], monotone[test_start0]
                        ),
                    },
                    "instrumented_rollout_steps": instrumented["steps"],
                },
            }

    return {
        "seed": seed,
        "model_diagnostics": {
            "parameter_count": model.parameter_count(),
            "latent_dim": model.config.latent_dim,
            "final_latent_loss": float(model.history.latent_loss[-1]),
            "final_security_loss": float(model.history.security_loss[-1]),
            "ema_target_distance": float(model.ema_target_distance()),
            "action_head_selected_epoch": int(action_model.best_epoch),
            "action_temperature": float(scalers["temperature"]),
            "observed_context_action_metrics": observed_action_metrics,
        },
        "tests": results,
        "information_boundary": "future actions, observations, security labels and policy metadata are never passed to E2 inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("policy-shift-calibration-diagnostics"))
    args = parser.parse_args()
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {
        str(seed): _run_seed(
            seed, args.episodes_per_cell, args.epochs, args.mc_samples, args.out
        ) for seed in seeds
    }
    result = {
        "benchmark": "G8.4.8.7.6 JEPA calibration-direction and recursive failure diagnosis",
        "seeds": seeds,
        "runs": runs,
        "protocol": {
            "calibration": "one start-step-0 window per nominal-support calibration episode",
            "test": "untouched rho=.85 and rho=.05; report all windows and step-0-only windows",
            "monotone_calibration": "nonnegative-slope ridge-Platt diagnostic fitted only on calibration",
            "diagnostic_rollout": "instrumented rollout is asserted numerically identical to production E2 inference",
        },
        "limitations": [
            "synthetic five-archetype controlled-shift setting only",
            "calibration diagnostics do not establish real-world calibration",
            "nonnegative-slope calibration is a prespecified diagnostic, not a test-selected replacement",
        ],
    }
    path = args.out / "calibration_direction_diagnostics.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "horizons": HORIZONS}, indent=2))


if __name__ == "__main__":
    main()

# Calibration is never selected against shifted or reverse test outcomes.
