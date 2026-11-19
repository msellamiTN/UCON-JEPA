"""Multi-seed and episode-cluster validation for G8.4.8.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_policy_prediction import run_experiment, run_lopo
from uncertain_action_dataset import build_dataset

METRICS = ("nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin")
MODELS = ("majority", "historical_frequency", "first_order_markov", "past_only_softmax")


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "seed_count": len(values),
    }


def summarize_runs(runs: dict[str, Any]) -> dict[str, Any]:
    seed_entries = list(runs.values())
    in_policy = {}
    for model in MODELS:
        in_policy[model] = {
            metric: _mean_sd([r["in_policy"]["test"][model]["raw"][metric] for r in seed_entries])
            for metric in METRICS
        }

    lopo_macro = {}
    lopo_by_policy = {}
    for model in MODELS:
        lopo_macro[model] = {}
        for metric in METRICS:
            per_seed = [
                float(np.mean([fold["test"][model]["raw"][metric] for fold in r["lopo"].values()]))
                for r in seed_entries
            ]
            lopo_macro[model][metric] = _mean_sd(per_seed)
    policies = sorted(seed_entries[0]["lopo"])
    for policy in policies:
        lopo_by_policy[policy] = {}
        for model in MODELS:
            lopo_by_policy[policy][model] = {
                metric: _mean_sd([r["lopo"][policy]["test"][model]["raw"][metric] for r in seed_entries])
                for metric in METRICS
            }
    return {
        "in_policy_across_seeds": in_policy,
        "lopo_macro_across_seeds": lopo_macro,
        "lopo_by_policy_across_seeds": lopo_by_policy,
        "aggregation_note": "LOPO macro metrics first average the five held-out-policy folds within each seed; reported SD is across independent dataset seeds.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("action-multiseed-output"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least three distinct predeclared seeds")
    if args.bootstrap_replicates < 100:
        raise ValueError("use at least 100 episode-bootstrap replicates")

    runs: dict[str, Any] = {}
    for seed in seeds:
        dataset = build_dataset(seed, args.episodes_per_family_policy, args.episode_length)
        runs[str(seed)] = {
            "dataset_manifest": dataset["manifest"],
            "in_policy": run_experiment(
                dataset, bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=seed + 700_000,
            ),
            "lopo": run_lopo(
                dataset, bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=seed + 900_000,
            ),
        }
    results = {
        "benchmark": "G8.4.8.3.1 multi-seed past-only action prediction validation",
        "seeds": seeds,
        "episodes_per_family_policy": args.episodes_per_family_policy,
        "episode_length": args.episode_length,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "whole episode; paired methods evaluated on identical test episodes",
        "runs": runs,
        "summary": summarize_runs(runs),
        "limitations": [
            "synthetic hand-designed behavior policies only",
            "five seeds quantify sensitivity to the declared synthetic generator, not real-agent population uncertainty",
            "episode bootstrap intervals are conditional on each generated test set and do not replace between-seed variability",
            "LOPO folds are highly dependent because their training populations overlap",
            "no JEPA future-state forecasting or operational intervention is evaluated",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "multiseed_results.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seeds": seeds, "summary": results["summary"]}, indent=2))


if __name__ == "__main__":
    main()
