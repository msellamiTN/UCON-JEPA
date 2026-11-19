"""Episode-disjoint compositional generalization benchmark.

The held-out test episodes contain action bigrams never admitted to training:
  IAM:  iam_read -> assume_admin
  Data: read_sensitive -> export_sensitive
  SoD:  request -> pay

Each component action appears in ordinary training episodes, but the ordered
composition is excluded. Test episodes are selected by observed trajectories,
not by labels or model predictions. All rows from an episode remain in one
partition. The test set therefore measures generalization to these specific
unseen ordered action pairs under the synthetic policy distribution.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

from predictive_benchmark import metrics, ridge_fit, ridge_predict, scaler, transform
from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA
from stateful_trajectory_dataset import build_dataset, ACTION_VOCAB, FAMILIES

HELDOUT_PAIRS = {
    "iam": ("iam_read", "assume_admin"),
    "data": ("read_sensitive", "export_sensitive"),
    "sod": ("request", "pay"),
}


def _all_rows(dataset):
    keys = ("context", "action", "future", "security", "episode_id", "family", "step")
    return {k: np.concatenate([dataset[p][k] for p in ("train", "validation", "test")], axis=0)
            for k in keys}


def _episode_actions(rows, episode_id):
    idx = np.flatnonzero(rows["episode_id"] == episode_id)
    idx = idx[np.argsort(rows["step"][idx])]
    return [ACTION_VOCAB[int(np.argmax(rows["action"][i]))] for i in idx]


def _contains_pair(sequence, pair):
    return any(tuple(sequence[i:i+2]) == tuple(pair) for i in range(len(sequence)-1))


def compositional_split(dataset):
    rows = _all_rows(dataset)
    episode_ids = sorted(set(rows["episode_id"].tolist()))
    test_ids, train_ids, validation_ids = set(), set(), set()
    coverage = {}
    for eid in episode_ids:
        family = str(rows["family"][np.flatnonzero(rows["episode_id"] == eid)[0]])
        actions = _episode_actions(rows, eid)
        pair = HELDOUT_PAIRS[family]
        if _contains_pair(actions, pair):
            test_ids.add(eid)
            coverage[family] = coverage.get(family, 0) + 1
        else:
            digest = int(hashlib.sha256(("g845:" + eid).encode()).hexdigest()[:8], 16) % 5
            (validation_ids if digest == 0 else train_ids).add(eid)

    missing = [f for f in FAMILIES if coverage.get(f, 0) == 0]
    if missing:
        raise ValueError(f"no test episodes contain held-out pairs for: {missing}")

    partitions = {}
    for name, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids)):
        mask = np.isin(rows["episode_id"], list(ids))
        partitions[name] = {k: v[mask] for k, v in rows.items()}
    # Assert pair absence in train/validation and presence in test by episode.
    for part in ("train", "validation"):
        for eid in set(partitions[part]["episode_id"].tolist()):
            fam = str(partitions[part]["family"][np.flatnonzero(partitions[part]["episode_id"] == eid)[0]])
            if _contains_pair(_episode_actions(partitions[part], eid), HELDOUT_PAIRS[fam]):
                raise AssertionError(f"held-out pair leaked into {part}: {eid}")
    observed = {}
    for fam in FAMILIES:
        fam_ids = set(partitions["test"]["episode_id"][partitions["test"]["family"] == fam].tolist())
        observed[fam] = len(fam_ids)
        if not fam_ids:
            raise AssertionError(f"no compositional test episodes for {fam}")
    manifest = {
        "split_type": "episode-disjoint held-out ordered action bigrams",
        "heldout_pairs": {k: list(v) for k, v in HELDOUT_PAIRS.items()},
        "test_episodes_by_family": observed,
        "episode_counts": {p: len(set(partitions[p]["episode_id"].tolist()))
                           for p in ("train", "validation", "test")},
        "row_counts": {p: len(partitions[p]["context"]) for p in ("train", "validation", "test")},
        "leakage_check": "held-out ordered pair absent from all train and validation episodes",
    }
    return partitions, manifest


def _evaluate_seed(seed, episodes_per_family, epochs):
    ds = build_dataset(seed=seed, episodes_per_family=episodes_per_family, episode_length=8)
    split, split_manifest = compositional_split(ds)
    tr, te = split["train"], split["test"]
    if len(tr["context"]) < 20 or len(te["context"]) < 10:
        raise ValueError("insufficient rows after compositional split; increase episodes_per_family")
    cp, ap, fp = scaler(tr["context"]), scaler(tr["action"]), scaler(tr["future"])
    xc, xa = transform(tr["context"], cp), transform(tr["action"], ap)
    yf = transform(tr["future"], fp)
    xt, at = transform(te["context"], cp), transform(te["action"], ap)
    cfg = JEPAConfig(latent_dim=12, learning_rate=.008, epochs=epochs, batch_size=64,
                     ema_tau=.99, variance_floor=.2, variance_weight=.1,
                     security_weight=.5, seed=seed)
    model = LinearActionConditionedJEPA(xc.shape[1], xa.shape[1], yf.shape[1],
                                        tr["security"].shape[1], cfg)
    model.fit(xc, xa, yf, tr["security"])
    predictions = {
        "action_conditioned_jepa": np.clip(model.decode_security(model.predict_latent(xt, at)), 0, 1),
        "supervised_context_plus_action": np.clip(ridge_predict(
            ridge_fit(np.column_stack((xc, xa)), tr["security"]), np.column_stack((xt, at))), 0, 1),
        "context_only": np.clip(ridge_predict(ridge_fit(xc, tr["security"]), xt), 0, 1),
        "action_only": np.clip(ridge_predict(ridge_fit(xa, tr["security"]), at), 0, 1),
    }
    y = te["security"]
    result = {}
    for method, pred in predictions.items():
        result[method] = {
            "overall": {"violation": metrics(y[:, 0], pred[:, 0]),
                        "security_feature_mse": float(np.mean((y-pred)**2))}
        }
        for family in FAMILIES:
            mask = te["family"] == family
            result[method][family] = {
                "n": int(mask.sum()),
                "positive": int(y[mask, 0].sum()),
                "violation": metrics(y[mask, 0], pred[mask, 0]) if mask.any() else None,
                "security_feature_mse": float(np.mean((y[mask]-pred[mask])**2)) if mask.any() else None,
            }
    return {"seed": seed, "split": split_manifest, "methods": result,
            "jepa_diagnostics": {"parameter_count": model.parameter_count(),
                                 "ema_target_distance": model.ema_target_distance()}}


def _aggregate(runs):
    methods = list(runs[0]["methods"])
    fields = ("auroc", "auprc", "brier", "ece_10", "recall", "false_positive_rate",
              "security_feature_mse")
    result = {}
    for method in methods:
        result[method] = {}
        for scope in ("overall",) + FAMILIES:
            result[method][scope] = {}
            for field in fields:
                values = []
                for run in runs:
                    item = run["methods"][method][scope]
                    value = (item.get("security_feature_mse") if field == "security_feature_mse"
                             else item["violation"].get(field) if item.get("violation") else None)
                    if value is not None:
                        values.append(float(value))
                result[method][scope][field] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "seed_count": len(values),
                }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--episodes-per-family", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("compositional-benchmark"))
    args = parser.parse_args()
    runs = [_evaluate_seed(s, args.episodes_per_family, args.epochs) for s in args.seeds]
    output = {
        "benchmark": "G8.4.5 held-out action-composition generalization",
        "seeds": args.seeds, "episodes_per_family": args.episodes_per_family,
        "epochs": args.epochs, "fixed_threshold": 0.5,
        "split_protocol": "test episodes contain family-specific ordered action bigram absent from train and validation; all rows grouped by episode",
        "per_seed": runs, "aggregate": _aggregate(runs),
        "limitations": [
            "synthetic policy/context/action distribution only",
            "only three predeclared ordered bigrams are held out",
            "test episode selection is conditional on the held-out sequence being observed",
            "metrics are episode-clustered in sampling but current aggregate SD across seeds is not a confidence interval",
            "no production security, causal, or general compositional reasoning claim",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compositional_benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
