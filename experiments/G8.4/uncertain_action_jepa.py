"""G8.4.8.4 E1/E2: privileged known-action vs predicted-action JEPA rollouts.

E1 is explicitly privileged: it receives the realized future action sequence.
E2 is the deployable information condition: it receives only the initial
decision-time context, then samples its own autoregressive action sequences
from a past-only policy head operating on the JEPA latent rollout. No future
action, future observation or security target is accepted by the E2 inference
function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import (
    PastOnlySoftmaxPredictor, _fit_temperature, _metrics, _normalize_rows, _softmax,
)
from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA
from predictive_benchmark import metrics as binary_metrics
from uncertain_action_dataset import ACTION_DIM, ACTION_VOCAB, PARTITIONS, build_dataset

EPS = 1e-12
FAMILY_ORDER = ("iam", "data", "sod")


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(x, dtype=np.float64).std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _transform(x: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - params[0]) / params[1]


def _family_onehot(context: np.ndarray) -> np.ndarray:
    x = np.asarray(context, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    family = x[:, :3]
    if np.any(np.max(family, axis=1) < 0.5):
        raise ValueError("invalid family one-hot in context")
    return family.copy()


def _action_features(family: np.ndarray, latent: np.ndarray) -> np.ndarray:
    return np.concatenate((family, latent), axis=1)


def _calibrated_action_probs(
    model: PastOnlySoftmaxPredictor,
    family: np.ndarray,
    latent: np.ndarray,
    temperature: float,
) -> np.ndarray:
    features = _action_features(family, latent)
    p = model.predict_proba(features)
    logits = np.where(p > 0.0, np.log(np.maximum(p, EPS)) / temperature, -np.inf)
    calibrated = _softmax(logits)
    allowed = np.zeros_like(calibrated, dtype=bool)
    family_id = np.argmax(family, axis=1)
    action_indices = {"iam": (0, 1, 2), "data": (3, 4, 5), "sod": (6, 7, 8)}
    for i, name in enumerate(FAMILY_ORDER):
        allowed[np.ix_(np.flatnonzero(family_id == i), action_indices[name])] = True
    calibrated *= allowed
    return _normalize_rows(calibrated)


def _make_windows(partition: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    """Create contiguous episode-local windows; target arrays remain evaluation-only."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    out: dict[str, list[Any]] = {
        "context": [], "actions": [], "security": [], "episode_id": [],
        "family": [], "agent_policy": [], "start_step": [],
    }
    for episode_id in sorted(set(partition["episode_id"].tolist())):
        indices = np.flatnonzero(partition["episode_id"] == episode_id)
        indices = indices[np.argsort(partition["step"][indices])]
        steps = partition["step"][indices]
        for start in range(len(indices) - horizon + 1):
            chosen = indices[start:start + horizon]
            if not np.array_equal(partition["step"][chosen], np.arange(steps[start], steps[start] + horizon)):
                continue
            out["context"].append(partition["context"][chosen[0]])
            out["actions"].append(partition["action"][chosen])
            out["security"].append(partition["security"][chosen])
            out["episode_id"].append(str(episode_id))
            out["family"].append(partition["family"][chosen[0]])
            out["agent_policy"].append(partition["agent_policy"][chosen[0]])
            out["start_step"].append(int(partition["step"][chosen[0]]))
    if not out["context"]:
        raise ValueError(f"no complete windows for horizon {horizon}")
    return {
        "context": np.asarray(out["context"], dtype=np.float64),
        "actions": np.asarray(out["actions"], dtype=np.float64),
        "security": np.asarray(out["security"], dtype=np.float64),
        "episode_id": np.asarray(out["episode_id"]),
        "family": np.asarray(out["family"]),
        "agent_policy": np.asarray(out["agent_policy"]),
        "start_step": np.asarray(out["start_step"], dtype=np.int64),
    }


def fit_e1_e2_models(
    dataset: dict[str, Any],
    seed: int,
    epochs: int = 180,
) -> tuple[LinearActionConditionedJEPA, PastOnlySoftmaxPredictor, dict[str, Any]]:
    """Fit JEPA on train; fit past-only action head on train latents; calibrate separately."""
    train, validation, calibration = (dataset[k] for k in ("train", "validation", "calibration"))
    context_scaler = _fit_scaler(train["context"])
    future_scaler = _fit_scaler(train["future"])
    action_scaler = _fit_scaler(train["action"])

    xc = _transform(train["context"], context_scaler)
    xa = _transform(train["action"], action_scaler)
    yf = _transform(train["future"], future_scaler)
    config = JEPAConfig(
        latent_dim=12, learning_rate=0.008, epochs=epochs, batch_size=128,
        ema_tau=0.99, variance_floor=0.2, variance_weight=0.1,
        security_weight=0.5, seed=seed,
    )
    jepa = LinearActionConditionedJEPA(
        xc.shape[1], xa.shape[1], yf.shape[1], train["security"].shape[1], config
    ).fit(xc, xa, yf, train["security"])

    def latent_features(part: dict[str, np.ndarray]) -> np.ndarray:
        scaled = _transform(part["context"], context_scaler)
        latent = scaled @ jepa.W_context
        return _action_features(_family_onehot(part["context"]), latent)

    action_head = PastOnlySoftmaxPredictor(max_epochs=250, patience=20).fit(
        latent_features(train), train["action"],
        latent_features(validation), validation["action"],
    )
    calibration_prob = action_head.predict_proba(latent_features(calibration))
    temperature = _fit_temperature(calibration_prob, calibration["action"])
    return jepa, action_head, {
        "context_scaler": context_scaler,
        "future_scaler": future_scaler,
        "action_scaler": action_scaler,
        "temperature": temperature,
        "action_head_selected_epoch": action_head.best_epoch,
    }


def rollout_known_actions(
    model: LinearActionConditionedJEPA,
    context: np.ndarray,
    action_sequence: np.ndarray,
    context_scaler: tuple[np.ndarray, np.ndarray],
    action_scaler: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """E1 privileged rollout: uses the supplied realized future action sequence."""
    x = _transform(context, context_scaler)
    actions = np.asarray(action_sequence, dtype=np.float64)
    if actions.ndim != 3 or len(x) != len(actions):
        raise ValueError("E1 expects aligned context and (batch,horizon,action_dim) actions")
    latent = x @ model.W_context
    survival = np.ones(len(x), dtype=np.float64)
    terminal_state = None
    for step in range(actions.shape[1]):
        scaled_action = _transform(actions[:, step, :], action_scaler)
        action_latent = scaled_action @ model.W_action
        latent = model.predict_from_latents(latent, action_latent)
        decoded = np.clip(model.decode_security(latent), 0.0, 1.0)
        survival *= (1.0 - decoded[:, 0])
        terminal_state = decoded[:, 1:]
    return 1.0 - survival, terminal_state


def rollout_predicted_actions(
    model: LinearActionConditionedJEPA,
    action_model: PastOnlySoftmaxPredictor,
    context: np.ndarray,
    context_scaler: tuple[np.ndarray, np.ndarray],
    action_scaler: tuple[np.ndarray, np.ndarray],
    temperature: float,
    horizon: int,
    mc_samples: int = 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """E2 autoregressive rollout; inference receives no future actions or labels."""
    if horizon < 1 or mc_samples < 1:
        raise ValueError("horizon and mc_samples must be positive")
    raw_context = np.asarray(context, dtype=np.float64)
    family = _family_onehot(raw_context)
    initial = _transform(raw_context, context_scaler) @ model.W_context
    n = len(initial)
    latent = np.repeat(initial, mc_samples, axis=0)
    family_repeated = np.repeat(family, mc_samples, axis=0)
    rng = np.random.default_rng(seed)
    survival = np.ones(n * mc_samples, dtype=np.float64)
    terminal_state = None

    for _ in range(horizon):
        probabilities = _calibrated_action_probs(
            action_model, family_repeated, latent, temperature
        )
        uniforms = rng.random(len(probabilities))
        cumulative = np.cumsum(probabilities, axis=1)
        sampled = np.sum(uniforms[:, None] > cumulative, axis=1)
        sampled = np.minimum(sampled, ACTION_DIM - 1)
        actions = np.zeros((len(sampled), ACTION_DIM), dtype=np.float64)
        actions[np.arange(len(sampled)), sampled] = 1.0
        scaled_action = _transform(actions, action_scaler)
        action_latent = scaled_action @ model.W_action
        latent = model.predict_from_latents(latent, action_latent)
        decoded = np.clip(model.decode_security(latent), 0.0, 1.0)
        survival *= (1.0 - decoded[:, 0])
        terminal_state = decoded[:, 1:]

    risk_samples = (1.0 - survival).reshape(n, mc_samples)
    state_samples = terminal_state.reshape(n, mc_samples, -1)
    return {
        "risk_mean": risk_samples.mean(axis=1),
        "risk_std": risk_samples.std(axis=1),
        "risk_q05": np.quantile(risk_samples, 0.05, axis=1),
        "risk_q50": np.quantile(risk_samples, 0.50, axis=1),
        "risk_q95": np.quantile(risk_samples, 0.95, axis=1),
        "terminal_state_mean": state_samples.mean(axis=1),
        "terminal_state_std": state_samples.std(axis=1),
    }


def _episode_bootstrap_difference(
    e2: np.ndarray,
    e1: np.ndarray,
    episode_ids: np.ndarray,
    seed: int,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Paired bootstrap of E2-minus-E1 window loss, resampling whole episodes."""
    ids = sorted(set(episode_ids.tolist()))
    grouped = []
    for eid in ids:
        mask = episode_ids == eid
        grouped.append(np.asarray([e2[mask].mean(), e1[mask].mean()]))
    diff = np.asarray(grouped)[:, 0] - np.asarray(grouped)[:, 1]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    means = diff[draws].mean(axis=1)
    return {
        "unit": "episode",
        "episodes": len(ids),
        "replicates": replicates,
        "e2_minus_e1_mean": float(diff.mean()),
        "ci95_percentile": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "interpretation": "negative favours predicted-action E2; positive favours privileged known-action E1",
    }


def evaluate_horizon(
    model: LinearActionConditionedJEPA,
    action_model: PastOnlySoftmaxPredictor,
    test: dict[str, np.ndarray],
    horizon: int,
    scalers: dict[str, Any],
    mc_samples: int,
    seed: int,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    windows = _make_windows(test, horizon)
    truth = np.column_stack((
        windows["security"][:, :, 0].max(axis=1),
        windows["security"][:, -1, 1:],
    ))
    e1_risk, e1_state = rollout_known_actions(
        model, windows["context"], windows["actions"],
        scalers["context_scaler"], scalers["action_scaler"],
    )
    e2 = rollout_predicted_actions(
        model, action_model, windows["context"], scalers["context_scaler"],
        scalers["action_scaler"], scalers["temperature"], horizon,
        mc_samples=mc_samples, seed=seed,
    )
    e1_pred = np.column_stack((e1_risk, e1_state))
    e2_pred = np.column_stack((e2["risk_mean"], e2["terminal_state_mean"]))
    e1_risk_loss = (e1_risk - truth[:, 0]) ** 2
    e2_risk_loss = (e2["risk_mean"] - truth[:, 0]) ** 2
    e1_security_loss = np.square(e1_pred - truth).mean(axis=1)
    e2_security_loss = np.square(e2_pred - truth).mean(axis=1)

    def summarize(pred: np.ndarray) -> dict[str, Any]:
        by_policy_family = {}
        for policy in sorted(set(windows["agent_policy"].tolist())):
            for family in sorted(set(windows["family"].tolist())):
                mask = (windows["agent_policy"] == policy) & (windows["family"] == family)
                if np.any(mask):
                    by_policy_family[f"{policy}/{family}"] = {
                        "violation": binary_metrics(truth[mask, 0], pred[mask, 0]),
                        "security_mse": float(np.square(pred[mask] - truth[mask]).mean()),
                        "windows": int(mask.sum()),
                        "episodes": int(len(set(windows["episode_id"][mask].tolist()))),
                    }
        return {
            "violation": binary_metrics(truth[:, 0], pred[:, 0]),
            "security_mse": float(np.square(pred - truth).mean()),
            "terminal_state_mse": float(np.square(pred[:, 1:] - truth[:, 1:]).mean()),
            "windows": int(len(pred)),
            "episodes": int(len(set(windows["episode_id"].tolist()))),
            "by_policy_family": by_policy_family,
        }

    return {
        "horizon": horizon,
        "target": "any DENY within horizon plus terminal workflow-state features",
        "E1_known_action_privileged": summarize(e1_pred),
        "E2_predicted_action": {
            **summarize(e2_pred),
            "risk_uncertainty": {
                "mean_mc_std": float(e2["risk_std"].mean()),
                "mean_q05": float(e2["risk_q05"].mean()),
                "mean_q50": float(e2["risk_q50"].mean()),
                "mean_q95": float(e2["risk_q95"].mean()),
                "mc_samples": mc_samples,
            },
        },
        "paired_episode_bootstrap_E2_minus_E1": {
            "risk_brier": _episode_bootstrap_difference(
                e2_risk_loss, e1_risk_loss, windows["episode_id"], seed + 1, bootstrap_replicates
            ),
            "security_mse": _episode_bootstrap_difference(
                e2_security_loss, e1_security_loss, windows["episode_id"], seed + 2, bootstrap_replicates
            ),
        },
    }


def run_seed(
    seed: int,
    episodes_per_family_policy: int = 300,
    epochs: int = 180,
    mc_samples: int = 32,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    dataset = build_dataset(seed, episodes_per_family_policy, episode_length=8)
    model, action_model, scalers = fit_e1_e2_models(dataset, seed, epochs)
    test = dataset["test"]
    action_features = _action_features(
        _family_onehot(test["context"]),
        _transform(test["context"], scalers["context_scaler"]) @ model.W_context,
    )
    action_prob = action_model.predict_proba(action_features)
    action_logits = np.where(action_prob > 0.0, np.log(np.maximum(action_prob, EPS)) / scalers["temperature"], -np.inf)
    action_prob_cal = _softmax(action_logits)
    action_metrics = _metrics(action_prob_cal, test["action"])
    by_horizon = {}
    for horizon in (1, 2, 4, 8):
        by_horizon[str(horizon)] = evaluate_horizon(
            model, action_model, test, horizon, scalers, mc_samples,
            seed + horizon * 100, bootstrap_replicates,
        )
    return {
        "seed": seed,
        "dataset_manifest": dataset["manifest"],
        "action_model": {
            "selected_epoch": action_model.best_epoch,
            "temperature": scalers["temperature"],
            "test_metrics": action_metrics,
        },
        "jepa": {
            "parameter_count": model.parameter_count(),
            "ema_target_distance": model.ema_target_distance(),
            "final_latent_loss": model.history.latent_loss[-1],
            "final_security_loss": model.history.security_loss[-1],
            "minimum_predicted_batch_std": min(model.history.predicted_std_min),
        },
        "horizons": by_horizon,
        "information_boundary": {
            "E1": "receives realized future action sequence; privileged diagnostic only",
            "E2": "receives initial context only; future actions sampled autoregressively from the past-only action head on predicted JEPA latents",
            "future_observations_and_security_labels": "training targets/evaluation only; never E2 inference inputs",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122")
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("predicted-action-jepa-output"))
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    runs = {str(seed): run_seed(
        seed, args.episodes_per_family_policy, args.epochs,
        args.mc_samples, args.bootstrap_replicates,
    ) for seed in seeds}
    result = {
        "benchmark": "G8.4.8.4 E1 privileged known-action vs E2 predicted-action JEPA",
        "seeds": seeds,
        "runs": runs,
        "limitations": [
            "synthetic hand-designed behavior policies only",
            "E1 is privileged and not deployable",
            "E2 samples from a calibrated past-only action model and rolls the JEPA latent state forward; no true future context is observed",
            "the action policy head is fitted on JEPA context latents, and recursive use assumes predicted target latents remain compatible with the context-latent action head",
            "Monte Carlo sampling propagates categorical action uncertainty but does not establish calibrated real-world uncertainty",
            "no intervention utility or formal runtime controller is evaluated",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "predicted_action_jepa_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "horizons": [1, 2, 4, 8]}, indent=2))


if __name__ == "__main__":
    main()
