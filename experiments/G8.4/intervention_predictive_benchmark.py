"""Held-out predictive benchmark for JEPA, supervised E2, and DTMC-style risk.

This is a prediction-only evaluation. It does not estimate intervention effects
and never uses held-out labels for model fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from intervention_modeling import fit_predictors, save_model_checkpoint


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_benchmark(
    *,
    seeds: tuple[int, ...],
    episodes_per_cell: int,
    epochs: int,
    mc_samples: int,
    output: Path,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    output.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in seeds:
        fitted = fit_predictors(
            int(seed), episodes_per_cell, epochs, mc_samples,
            evaluate_holdout=True,
        )
        checkpoint = output / "models" / f"seed_{seed}_neural_predictors.npz"
        checkpoint_sha = save_model_checkpoint(
            checkpoint, fitted["models"], fitted["scalers"]
        )
        dtmc_path = output / "models" / f"seed_{seed}_dtmc.json"
        dtmc_sha = _write_json(dtmc_path, fitted["dtmc_model"].to_dict())
        per_seed[str(seed)] = {
            "predictive_metrics": fitted["predictive_metrics"],
            "threshold_manifest": fitted["threshold_manifest"],
            "dataset_manifest": fitted["dataset_manifest"],
            "model_capacity": fitted["model_capacity"],
            "dtmc_metadata": fitted["dtmc_metadata"],
            "neural_checkpoint": str(checkpoint.relative_to(output)),
            "neural_checkpoint_sha256": checkpoint_sha,
            "dtmc_model": str(dtmc_path.relative_to(output)),
            "dtmc_model_sha256": dtmc_sha,
        }
    summary = {
        "experiment": "G8.4.8.8 held-out predictive risk benchmark",
        "status": "PREDICTION_ONLY_NOT_CAUSAL",
        "seeds": list(seeds),
        "models": ["jepa", "supervised", "dtmc"],
        "test_partitions": ["test_shifted", "test_reverse"],
        "metrics": ["auroc", "brier", "ece_10"],
        "threshold_support": "validation episode maxima only",
        "test_labels_used_for_fitting_or_thresholds": False,
        "intervention_effects_or_p_values": None,
        "per_seed": per_seed,
    }
    summary_sha = _write_json(output / "predictive_benchmark_summary.json", summary)
    _write_json(output / "file_manifest.json", {
        "predictive_benchmark_summary.json": summary_sha,
        "per_seed_artifacts": per_seed,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261120,20261121,20261122,20261123,20261124")
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--out", default="predictive-benchmark-output")
    args = parser.parse_args()
    summary = run_benchmark(
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        episodes_per_cell=args.episodes_per_cell,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
        output=Path(args.out),
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
