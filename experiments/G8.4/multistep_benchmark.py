"""Multi-horizon action-conditioned JEPA benchmark.

Direct-horizon models receive the ordered candidate-action sequence and predict
the terminal latent state plus whether any violation occurs within the horizon.
Recursive rollout uses the one-step JEPA predictor autoregressively in latent
space; cumulative violation probability is 1 - product(1 - per-step risk).

All windows are constructed within an episode and contiguous in step index.
The train/validation/test episode assignments are inherited from the dataset.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA
from predictive_benchmark import metrics, ridge_fit, ridge_predict, scaler, transform
from stateful_trajectory_dataset import build_dataset, FAMILIES

HORIZONS = (1, 2, 4, 8)


def make_windows(partition, horizon):
    """Build aligned (context, ordered actions, terminal future, horizon labels)."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    output = {k: [] for k in ("context", "actions", "future", "security", "episode_id", "family", "start_step")}
    episode_ids = sorted(set(partition["episode_id"].tolist()))
    for eid in episode_ids:
        idx = np.flatnonzero(partition["episode_id"] == eid)
        idx = idx[np.argsort(partition["step"][idx])]
        steps = partition["step"][idx]
        for start in range(0, len(idx) - horizon + 1):
            chosen = idx[start:start + horizon]
            if not np.array_equal(partition["step"][chosen], np.arange(steps[start], steps[start] + horizon)):
                continue
            output["context"].append(partition["context"][chosen[0]])
            output["actions"].append(partition["action"][chosen].reshape(-1))
            output["future"].append(partition["future"][chosen[-1]])
            # First target is any violation in the forecast interval; remaining
            # targets are the terminal workflow state after the final action.
            sec = np.concatenate((
                np.asarray([partition["security"][chosen, 0].max()]),
                partition["security"][chosen[-1], 1:],
            ))
            output["security"].append(sec)
            output["episode_id"].append(eid)
            output["family"].append(partition["family"][chosen[0]])
            output["start_step"].append(int(partition["step"][chosen[0]]))
    if not output["context"]:
        raise ValueError(f"no complete horizon-{horizon} windows in partition")
    return {
        "context": np.asarray(output["context"], dtype=float),
        "actions": np.asarray(output["actions"], dtype=float),
        "future": np.asarray(output["future"], dtype=float),
        "security": np.asarray(output["security"], dtype=float),
        "episode_id": np.asarray(output["episode_id"]),
        "family": np.asarray(output["family"]),
        "start_step": np.asarray(output["start_step"], dtype=np.int64),
    }


def _predict_recursive(model, context, action_sequences):
    """Roll one-step predictor forward in latent space over supplied actions."""
    context = np.asarray(context, dtype=float)
    action_sequences = np.asarray(action_sequences, dtype=float)
    batch, horizon, action_dim = action_sequences.shape
    if action_dim != model.action_dim:
        raise ValueError("recursive rollout requires one action vector per step")
    latent = context @ model.W_context
    risk_survival = np.ones(batch, dtype=float)
    terminal_security = None
    for step in range(horizon):
        action_latent = action_sequences[:, step, :] @ model.W_action
        latent = model.predict_from_latents(latent, action_latent)
        sec = np.clip(model.decode_security(latent), 0.0, 1.0)
        risk_survival *= (1.0 - sec[:, 0])
        terminal_security = sec
    prediction = terminal_security.copy()
    prediction[:, 0] = 1.0 - risk_survival
    return prediction


def _fit_jepa(train, context_scaler, future_scaler, action_scaler, seed, epochs):
    xc = transform(train["context"], context_scaler)
    xa = transform(train["actions"], action_scaler)
    yf = transform(train["future"], future_scaler)
    cfg = JEPAConfig(latent_dim=12, learning_rate=.008, epochs=epochs, batch_size=64,
                     ema_tau=.99, variance_floor=.2, variance_weight=.1,
                     security_weight=.5, seed=seed)
    model = LinearActionConditionedJEPA(
        xc.shape[1], xa.shape[1], yf.shape[1], train["security"].shape[1], cfg
    )
    model.fit(xc, xa, yf, train["security"])
    return model


def _score(method_predictions, test, horizon):
    truth = test["security"]
    results = {}
    for name, pred in method_predictions.items():
        pred = np.clip(np.asarray(pred), 0.0, 1.0)
        results[name] = {
            "violation": metrics(truth[:, 0], pred[:, 0]),
            "security_mse": float(np.mean((truth - pred) ** 2)),
            "terminal_state_mse": float(np.mean((truth[:, 1:] - pred[:, 1:]) ** 2)),
        }
        per_episode = []
        for eid in sorted(set(test["episode_id"].tolist())):
            mask = test["episode_id"] == eid
            per_episode.append({
                "episode_id": str(eid),
                "brier": float(np.mean((truth[mask, 0] - pred[mask, 0]) ** 2)),
                "security_mse": float(np.mean((truth[mask] - pred[mask]) ** 2)),
            })
        results[name]["episode_errors"] = per_episode
    return results


def evaluate_seed(seed, episodes_per_family, epochs):
    dataset = build_dataset(seed=seed, episodes_per_family=episodes_per_family, episode_length=8)
    by_horizon = {}
    one_step_model = None
    one_step_context_scaler = None
    one_step_action_scaler = None
    for horizon in HORIZONS:
        train = make_windows(dataset["train"], horizon)
        validation = make_windows(dataset["validation"], horizon)
        test = make_windows(dataset["test"], horizon)
        # All preprocessing statistics come from this horizon's training windows.
        cp = scaler(train["context"])
        ap = scaler(train["actions"])
        fp = scaler(train["future"])
        xc = transform(train["context"], cp)
        xa = transform(train["actions"], ap)
        xt = transform(test["context"], cp)
        at = transform(test["actions"], ap)
        model = _fit_jepa(train, cp, fp, ap, seed + horizon, epochs)
        direct_jepa = np.clip(model.decode_security(model.predict_latent(xt, at)), 0, 1)
        direct_supervised = np.clip(
            ridge_predict(ridge_fit(np.column_stack((xc, xa)), train["security"]),
                          np.column_stack((xt, at))), 0, 1
        )
        predictions = {
            "direct_horizon_jepa": direct_jepa,
            "direct_horizon_supervised": direct_supervised,
        }
        if horizon == 1:
            one_step_model = model
            one_step_context_scaler = cp
            one_step_action_scaler = ap
        # Validation is kept separate and reported; it is not used to tune the
        # fixed 0.5 operating threshold or inspect test outcomes.
        xv = transform(validation["context"], cp)
        av = transform(validation["actions"], ap)
        validation_metrics = {
            "jepa_violation_auroc": metrics(
                validation["security"][:, 0],
                np.clip(model.decode_security(model.predict_latent(xv, av))[:, 0], 0, 1)
            )["auroc"],
            "rows": int(len(validation["context"])),
            "episodes": int(len(set(validation["episode_id"].tolist()))),
        }
        if horizon == 1:
            # Preserve the one-step model and its own preprocessing for rollout.
            one_step_model = model
            one_step_context_scaler = cp
            one_step_action_scaler = ap
        else:
            # Recursive rollout is evaluated on the same horizon windows, with
            # each action step supplied in its original order.
            raw_actions = test["actions"].reshape(len(test["actions"]), horizon, -1)
            rollout_actions = transform(
                raw_actions.reshape(-1, raw_actions.shape[-1]), one_step_action_scaler
            ).reshape(raw_actions.shape)
            rollout_context = transform(test["context"], one_step_context_scaler)
            recursive = _predict_recursive(one_step_model, rollout_context, rollout_actions)
            predictions["recursive_one_step_jepa"] = recursive
        by_horizon[str(horizon)] = {
            "train_windows": int(len(train["context"])),
            "validation_windows": int(len(validation["context"])),
            "test_windows": int(len(test["context"])),
            "test_episodes": int(len(set(test["episode_id"].tolist()))),
            "validation_diagnostics": validation_metrics,
            "methods": _score(predictions, test, horizon),
            "jepa_diagnostics": {
                "parameter_count": model.parameter_count(),
                "ema_target_distance": model.ema_target_distance(),
                "minimum_predicted_batch_std": min(model.history.predicted_std_min),
            },
        }
    return {"seed": seed, "horizons": by_horizon}


def aggregate_runs(runs, bootstrap_replicates=2000):
    rng = np.random.default_rng(8462026)
    output = {}
    for horizon in map(str, HORIZONS):
        names = list(runs[0]["horizons"][horizon]["methods"])
        output[horizon] = {}
        for name in names:
            fields = ("auroc", "auprc", "brier", "ece_10", "recall",
                      "false_positive_rate", "security_mse", "terminal_state_mse")
            output[horizon][name] = {}
            for field in fields:
                vals = []
                for run in runs:
                    m = run["horizons"][horizon]["methods"][name]
                    val = m["security_mse"] if field == "security_mse" else (
                        m["terminal_state_mse"] if field == "terminal_state_mse"
                        else m["violation"].get(field)
                    )
                    if val is not None:
                        vals.append(float(val))
                output[horizon][name][field] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std_across_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    "seed_count": len(vals),
                }
        # Paired cluster bootstrap: one mean loss per episode; preserve pairing
        # between JEPA and supervised predictions when resampling episodes.
        pooled = {}
        for method in ("direct_horizon_jepa", "direct_horizon_supervised", "recursive_one_step_jepa"):
            errors = {}
            for run in runs:
                entry = run["horizons"][horizon]["methods"].get(method)
                if entry is None:
                    continue
                for row in entry["episode_errors"]:
                    errors[(run["seed"], row["episode_id"])] = row
            pooled[method] = errors
        comparisons = {}
        for candidate in ("direct_horizon_jepa", "recursive_one_step_jepa"):
            if candidate not in pooled or not pooled[candidate] or not pooled["direct_horizon_supervised"]:
                continue
            common = sorted(set(pooled[candidate]) & set(pooled["direct_horizon_supervised"]))
            brier_delta = np.asarray([
                pooled[candidate][key]["brier"] - pooled["direct_horizon_supervised"][key]["brier"]
                for key in common
            ])
            mse_delta = np.asarray([
                pooled[candidate][key]["security_mse"] - pooled["direct_horizon_supervised"][key]["security_mse"]
                for key in common
            ])
            if len(common) < 2:
                continue
            def ci(values):
                samples = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
                means = values[samples].mean(axis=1)
                return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]
            comparisons[candidate + "_minus_supervised"] = {
                "episode_clusters": len(common),
                "brier_difference_mean": float(brier_delta.mean()),
                "brier_difference_95pct_cluster_bootstrap_ci": ci(brier_delta),
                "security_mse_difference_mean": float(mse_delta.mean()),
                "security_mse_difference_95pct_cluster_bootstrap_ci": ci(mse_delta),
                "interpretation": "positive difference favours supervised; negative favours candidate",
            }
        output[horizon]["paired_episode_bootstrap"] = comparisons
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--episodes-per-family", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("multistep-benchmark"))
    args = parser.parse_args()
    runs = [evaluate_seed(seed, args.episodes_per_family, args.epochs) for seed in args.seeds]
    result = {
        "benchmark": "G8.4.6 direct-horizon and recursive multi-step JEPA",
        "horizons": list(HORIZONS), "seeds": args.seeds,
        "episodes_per_family": args.episodes_per_family, "epochs": args.epochs,
        "fixed_risk_threshold": 0.5,
        "target_semantics": {
            "violation": "any policy-oracle violation within the next h transitions",
            "terminal_state": "approved, sensitive_read and role_granted after transition h",
            "direct": "one horizon-specific model predicts the terminal target",
            "recursive": "one-step JEPA predictor rolled forward in latent space; cumulative risk is 1-product(1-step risk)",
        },
        "split": "episode-disjoint train/validation/test inherited from stateful_trajectory_dataset",
        "per_seed": runs,
        "aggregate": aggregate_runs(runs, args.bootstrap_replicates),
        "limitations": [
            "synthetic policy semantics and action-generation distribution only",
            "episodes terminate at the first denied action, so longer windows are conditional on surviving that far",
            "recursive rollout compounds model error and assumes the supplied future action sequence is known",
            "linear JEPA is a transparent reference model, not a high-capacity JEPA architecture",
            "bootstrap intervals quantify episode sampling conditional on these generated seeds and this synthetic distribution only",
            "no real-world security, causal, deployment or formal assurance claim",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "multistep_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
