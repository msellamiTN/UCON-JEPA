"""G8.4.8.3 past-only action forecasting baselines.

The inference API accepts only the decision-time context (or, for the Markov
baseline, the immediately preceding observed action). It never accepts target
actions, future observations, security labels, policy IDs, episode IDs or split
labels. All fitting statistics are learned from train; temperature is selected
on calibration; test is evaluation-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_action_dataset import ACTION_DIM, ACTION_VOCAB, AGENT_POLICIES, PARTITIONS, build_dataset, build_lopo_fold

FAMILY_ACTIONS = {
    "iam": ("grant_role", "assume_admin", "iam_read"),
    "data": ("read_sensitive", "export_sensitive", "data_list"),
    "sod": ("request", "approve", "pay"),
}
FAMILY_ORDER = ("iam", "data", "sod")
ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_VOCAB)}
HISTORY_START = 15
HISTORY_END = HISTORY_START + ACTION_DIM
EPS = 1e-12


def _family_index(context: np.ndarray) -> np.ndarray:
    """Recover family only from its one-hot decision-time context features."""
    x = np.asarray(context)
    if x.ndim == 1:
        x = x[None, :]
    fam = np.argmax(x[:, :3], axis=1)
    if np.any(np.max(x[:, :3], axis=1) < 0.5):
        raise ValueError("context does not contain a valid family one-hot")
    return fam


def _allowed_mask(context: np.ndarray) -> np.ndarray:
    fam = _family_index(context)
    mask = np.zeros((len(fam), ACTION_DIM), dtype=bool)
    for i, name in enumerate(FAMILY_ORDER):
        rows = np.flatnonzero(fam == i)
        cols = [ACTION_TO_INDEX[a] for a in FAMILY_ACTIONS[name]]
        mask[np.ix_(rows, cols)] = True
    return mask


def _normalize_rows(prob: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(prob, dtype=np.float64), 0.0)
    totals = p.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("probability row has no mass")
    return p / totals


class MajorityActionPredictor:
    """Family-conditioned majority action, fitted on training labels only."""
    def fit(self, context: np.ndarray, action: np.ndarray) -> "MajorityActionPredictor":
        y = np.argmax(action, axis=1)
        fam = _family_index(context)
        self.counts = np.zeros((3, ACTION_DIM), dtype=np.float64)
        for f in range(3):
            for a in y[fam == f]:
                self.counts[f, a] += 1
        return self

    def predict_proba(self, context: np.ndarray) -> np.ndarray:
        fam = _family_index(context)
        p = self.counts[fam].copy()
        for i, name in enumerate(FAMILY_ORDER):
            allowed = [ACTION_TO_INDEX[a] for a in FAMILY_ACTIONS[name]]
            p[np.ix_(fam == i, np.arange(ACTION_DIM))] += 1e-3
            disallowed = sorted(set(range(ACTION_DIM)) - set(allowed))
            p[np.ix_(fam == i, disallowed)] = 0.0
        return _normalize_rows(p)


class HistoricalFrequencyPredictor:
    """Episode-local action-history frequencies with a train prior as smoothing."""
    def fit(self, context: np.ndarray, action: np.ndarray) -> "HistoricalFrequencyPredictor":
        self.prior = MajorityActionPredictor().fit(context, action).counts
        return self

    def predict_proba(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        fam = _family_index(x)
        history = np.clip(x[:, HISTORY_START:HISTORY_END], 0.0, None)
        p = history + 0.5 * self.prior[fam] / np.maximum(self.prior[fam].sum(axis=1, keepdims=True), 1.0)
        p *= _allowed_mask(x)
        return _normalize_rows(p)


class FirstOrderMarkovPredictor:
    """P(next action | previous observed action, family), estimated on train."""
    BOS = -1

    def fit(self, context: np.ndarray, action: np.ndarray, episode_id: np.ndarray, step: np.ndarray) -> "FirstOrderMarkovPredictor":
        self.prior = MajorityActionPredictor().fit(context, action).counts
        self.transitions = np.zeros((3, ACTION_DIM, ACTION_DIM), dtype=np.float64)
        groups: dict[str, list[int]] = {}
        for i, eid in enumerate(episode_id.tolist()):
            groups.setdefault(str(eid), []).append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: int(step[i]))
            for prev_i, cur_i in zip(indices, indices[1:]):
                f = int(_family_index(context[cur_i])[0])
                prev_a = int(np.argmax(action[prev_i]))
                cur_a = int(np.argmax(action[cur_i]))
                self.transitions[f, prev_a, cur_a] += 1
        return self

    def predict_proba(self, context: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        prev = np.asarray(previous_action, dtype=np.int64).reshape(-1)
        if len(prev) != len(x):
            raise ValueError("previous_action length must match context rows")
        fam = _family_index(x)
        out = np.zeros((len(x), ACTION_DIM), dtype=np.float64)
        for i, (f, a) in enumerate(zip(fam, prev)):
            if a == self.BOS:
                row = self.prior[f]
            else:
                row = self.transitions[f, a] + 0.5 * self.prior[f] / max(self.prior[f].sum(), 1.0)
            out[i] = row * _allowed_mask(x[i])[0]
        return _normalize_rows(out)


class PastOnlySoftmaxPredictor:
    """Regularized multinomial linear policy model over decision-time context."""
    def __init__(self, learning_rate: float = 0.03, max_epochs: int = 250, patience: int = 20, l2: float = 1e-4):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2

    def fit(self, train_x: np.ndarray, train_action: np.ndarray, val_x: np.ndarray, val_action: np.ndarray) -> "PastOnlySoftmaxPredictor":
        x = np.asarray(train_x, dtype=np.float64)
        xv = np.asarray(val_x, dtype=np.float64)
        y = np.argmax(train_action, axis=1)
        yv = np.argmax(val_action, axis=1)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = (x - self.mean) / self.scale
        zv = (xv - self.mean) / self.scale
        self.weights = np.zeros((z.shape[1], ACTION_DIM), dtype=np.float64)
        self.bias = np.zeros(ACTION_DIM, dtype=np.float64)
        m_w = np.zeros_like(self.weights); v_w = np.zeros_like(self.weights)
        m_b = np.zeros_like(self.bias); v_b = np.zeros_like(self.bias)
        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(1, self.max_epochs + 1):
            logits = z @ self.weights + self.bias
            logits[~_allowed_mask(x)] = -1e9
            probs = _softmax(logits)
            probs[np.arange(len(y)), y] -= 1.0
            grad_w = z.T @ probs / len(y) + self.l2 * self.weights
            grad_b = probs.mean(axis=0)
            m_w = 0.9*m_w + 0.1*grad_w; v_w = 0.999*v_w + 0.001*(grad_w*grad_w)
            m_b = 0.9*m_b + 0.1*grad_b; v_b = 0.999*v_b + 0.001*(grad_b*grad_b)
            self.weights -= self.learning_rate * (m_w/(1-0.9**epoch)) / (np.sqrt(v_w/(1-0.999**epoch)) + 1e-8)
            self.bias -= self.learning_rate * (m_b/(1-0.9**epoch)) / (np.sqrt(v_b/(1-0.999**epoch)) + 1e-8)
            val_logits = zv @ self.weights + self.bias
            val_logits[~_allowed_mask(xv)] = -1e9
            val_loss = _nll(_softmax(val_logits), yv)
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = (self.weights.copy(), self.bias.copy(), epoch)
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("supervised action model failed to select a validation checkpoint")
        self.weights, self.bias, self.best_epoch = best_state
        return self

    def predict_proba(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        z = (x - self.mean) / self.scale
        logits = z @ self.weights + self.bias
        logits[~_allowed_mask(x)] = -1e9
        return _softmax(logits)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -700, 0))
    return _normalize_rows(e)


def _nll(prob: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], EPS, 1.0)).mean())


def _metrics(prob: np.ndarray, action: np.ndarray, top_k: int = 2) -> dict[str, float]:
    y = np.argmax(action, axis=1)
    order = np.argsort(-prob, axis=1)
    conf = prob[np.arange(len(y)), order[:, 0]]
    correct = (order[:, 0] == y)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        selected = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if np.any(selected):
            ece += float(selected.mean() * abs(correct[selected].mean() - conf[selected].mean()))
    onehot = np.zeros_like(prob); onehot[np.arange(len(y)), y] = 1.0
    return {
        "nll": _nll(prob, y),
        "top1_accuracy": float(correct.mean()),
        "top2_accuracy": float(np.any(order[:, :top_k] == y[:, None], axis=1).mean()),
        "multiclass_brier": float(np.square(prob - onehot).sum(axis=1).mean()),
        "top_label_ece_10bin": ece,
        "examples": int(len(y)),
    }


def _fit_temperature(prob: np.ndarray, action: np.ndarray) -> float:
    """Select scalar temperature on calibration only; no test labels are read."""
    y = np.argmax(action, axis=1)
    logits = np.log(np.clip(prob, EPS, 1.0))
    best_t, best_loss = 1.0, float("inf")
    for temperature in np.linspace(0.5, 3.0, 51):
        p = _softmax(logits / temperature)
        loss = _nll(p, y)
        if loss < best_loss:
            best_t, best_loss = float(temperature), loss
    return best_t


def _previous_actions(dataset: dict[str, np.ndarray]) -> np.ndarray:
    prev = np.full(len(dataset["action"]), FirstOrderMarkovPredictor.BOS, dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for i, eid in enumerate(dataset["episode_id"].tolist()):
        groups.setdefault(str(eid), []).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: int(dataset["step"][i]))
        for before, after in zip(indices, indices[1:]):
            prev[after] = int(np.argmax(dataset["action"][before]))
    return prev


def _evaluate_split(models: dict[str, Any], split: dict[str, np.ndarray], temperature: float = 1.0) -> dict[str, Any]:
    x, y = split["context"], split["action"]
    prev = _previous_actions(split)
    result = {}
    for name, model in models.items():
        p = model.predict_proba(x, prev) if name == "first_order_markov" else model.predict_proba(x)
        result[name] = {"raw": _metrics(p, y), "by_policy_family": {}}
        for policy in sorted(set(split["agent_policy"].tolist())):
            for family in sorted(set(split["family"].tolist())):
                mask = (split["agent_policy"] == policy) & (split["family"] == family)
                if not np.any(mask):
                    continue
                key = f"{policy}/{family}"
                result[name]["by_policy_family"][key] = _metrics(p[mask], y[mask])
        if name == "past_only_softmax":
            calibrated = _softmax(np.log(np.clip(p, EPS, 1.0)) / temperature)
            result[name]["calibrated"] = _metrics(calibrated, y)
            result[name]["calibrated_by_policy_family"] = {}
            for key, mask in _stratum_masks(split).items():
                result[name]["calibrated_by_policy_family"][key] = _metrics(calibrated[mask], y[mask])
            result[name]["temperature"] = temperature
    return result


def _stratum_masks(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {}
    for policy in sorted(set(split["agent_policy"].tolist())):
        for family in sorted(set(split["family"].tolist())):
            mask = (split["agent_policy"] == policy) & (split["family"] == family)
            if np.any(mask):
                masks[f"{policy}/{family}"] = mask
    return masks


def run_experiment(dataset: dict[str, Any]) -> dict[str, Any]:
    tr, va, ca, te = (dataset[k] for k in PARTITIONS)
    majority = MajorityActionPredictor().fit(tr["context"], tr["action"])
    history = HistoricalFrequencyPredictor().fit(tr["context"], tr["action"])
    markov = FirstOrderMarkovPredictor().fit(tr["context"], tr["action"], tr["episode_id"], tr["step"])
    supervised = PastOnlySoftmaxPredictor().fit(tr["context"], tr["action"], va["context"], va["action"])
    models = {"majority": majority, "historical_frequency": history, "first_order_markov": markov, "past_only_softmax": supervised}
    calibration_p = supervised.predict_proba(ca["context"])
    temperature = _fit_temperature(calibration_p, ca["action"])
    return {
        "validation_selected_epoch": supervised.best_epoch,
        "calibration_temperature": temperature,
        "test": _evaluate_split(models, te, temperature),
        "validation": _evaluate_split(models, va, temperature),
        "calibration": _evaluate_split(models, ca, temperature),
    }


def run_lopo(dataset: dict[str, Any]) -> dict[str, Any]:
    folds = {}
    for policy in AGENT_POLICIES:
        fold = build_lopo_fold(dataset, policy)
        folds[policy] = run_experiment(fold)
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("action-prediction-output"))
    parser.add_argument("--skip-lopo", action="store_true")
    args = parser.parse_args()
    ds = build_dataset(args.seed, args.episodes_per_family_policy, args.episode_length)
    results = {
        "benchmark": "G8.4.8.3 past-only action forecasting",
        "seed": args.seed,
        "dataset_manifest": ds["manifest"],
        "in_policy": run_experiment(ds),
        "lopo": {} if args.skip_lopo else run_lopo(ds),
        "inference_contract": "context only; Markov baseline additionally receives only the immediately preceding observed action",
        "limitations": [
            "synthetic hand-designed behavior policies only",
            "single fixed seed in this implementation run; no population-level or real-agent generalization claim",
            "linear supervised model is a transparent baseline, not an architecture search",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out / "results.json"), "in_policy_test": results["in_policy"]["test"], "lopo_policies": list(results["lopo"])}, indent=2))


if __name__ == "__main__":
    main()
