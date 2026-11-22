"""Arm-blinded engineering/variance pilot for G8.4.8.8.

This is not the confirmatory experiment. It uses separate seeds and nominal rho
support, retains episode-level evidence, and reports pooled dispersion only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from intervention_modeling import fit_predictors, save_model_checkpoint
from intervention_simulator import (
    ARMS,
    PILOT_ONLY_ARMS,
    make_blocked_episode_specs,
    run_episode,
)
from intervention_task_manifests import task_manifest_records, validate_task_manifests

PILOT_RHOS = (0.20, 0.40, 0.60)
DEFAULT_PILOT_SEEDS = (20261110, 20261111)
DEFAULT_CONFIRMATORY_SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return _sha256(path)


def _write_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    return _sha256(path)


def _pooled_within_block_dispersion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return pooled residual dispersion without exposing arm mean contrasts."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"denied_count": [], "task_complete": []}
    )
    for row in rows:
        key = (str(row["stratum"]), str(row["arm"]))
        grouped[key]["denied_count"].append(float(row["denied_count"]))
        grouped[key]["task_complete"].append(float(row["task_complete"]))
    residuals: dict[str, list[float]] = {
        "denied_count": [], "task_complete": []
    }
    cell_counts = []
    for (_stratum, _arm), values in grouped.items():
        cell_counts.append(len(values["denied_count"]))
        for endpoint in residuals:
            data = np.asarray(values[endpoint], dtype=np.float64)
            if len(data) > 1:
                residuals[endpoint].extend((data - data.mean()).tolist())
    return {
        "arm_level_means_suppressed": True,
        "within_block_arm_cells": len(grouped),
        "minimum_cell_size": int(min(cell_counts)) if cell_counts else 0,
        "maximum_cell_size": int(max(cell_counts)) if cell_counts else 0,
        "pooled_within_block_arm_variance": {
            endpoint: (
                float(np.var(values, ddof=1)) if len(values) > 1 else None
            )
            for endpoint, values in residuals.items()
        },
        "residual_support": {
            endpoint: len(values) for endpoint, values in residuals.items()
        },
        "interpretation": (
            "variance-planning input only; no treatment-effect contrasts or "
            "confirmatory p-values are computed in this pilot"
        ),
    }


def run_pilot(
    *,
    seeds: tuple[int, ...],
    confirmatory_seeds: tuple[int, ...],
    episodes_per_arm_per_stratum: int,
    episodes_per_cell: int,
    epochs: int,
    mc_samples: int,
    output: Path,
) -> dict[str, Any]:
    if not seeds or set(seeds) & set(confirmatory_seeds):
        raise ValueError("pilot seeds must be nonempty and disjoint from confirmatory seeds")
    if episodes_per_arm_per_stratum < 2:
        raise ValueError("pilot needs at least two episodes per arm per stratum")
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    seed_metadata = {}
    threshold_fraction = 0.10

    for seed in seeds:
        fitted = fit_predictors(
            int(seed), episodes_per_cell, epochs, mc_samples
        )
        checkpoint_path = output / "models" / f"seed_{seed}_predictors.npz"
        checkpoint_sha = save_model_checkpoint(
            checkpoint_path, fitted["models"], fitted["scalers"]
        )
        dtmc_path = output / "models" / f"seed_{seed}_dtmc.json"
        dtmc_sha = _write_json(dtmc_path, fitted["dtmc_model"].to_dict())
        thresholds = fitted["thresholds_by_fraction"][threshold_fraction]
        seed_rows = []
        seed_assignments = []
        for rho in PILOT_RHOS:
            assignments = make_blocked_episode_specs(
                [int(seed)],
                episodes_per_arm_per_stratum,
                assignment_seed=int(seed) + 884_800 + int(round(1000 * rho)),
                rho=float(rho),
                arms=ARMS + PILOT_ONLY_ARMS,
            )
            seed_assignments.extend(assignments)
            for spec in assignments:
                row = run_episode(
                    spec,
                    score_functions=fitted["score_functions"],
                    thresholds=thresholds,
                    horizon=1,
                    random_trigger_fraction=0.10,
                )
                seed_rows.append(row)
        all_rows.extend(seed_rows)
        assignment_path = output / "assignments" / f"seed_{seed}_assignments.jsonl.gz"
        assignment_sha = _write_jsonl_gzip(assignment_path, seed_assignments)
        seed_metadata[str(seed)] = {
            "checkpoint": str(checkpoint_path.relative_to(output)),
            "checkpoint_sha256": checkpoint_sha,
            "dtmc_model": str(dtmc_path.relative_to(output)),
            "dtmc_model_sha256": dtmc_sha,
            "dtmc_metadata": fitted["dtmc_metadata"],
            "predictive_metrics": fitted["predictive_metrics"],
            "assignment_manifest": str(assignment_path.relative_to(output)),
            "assignment_manifest_sha256": assignment_sha,
            "dataset_manifest": fitted["dataset_manifest"],
            "train_episodes": fitted["train_episodes"],
            "validation_episodes": fitted["validation_episodes"],
            "calibration_episodes": fitted["calibration_episodes"],
            "thresholds": fitted["threshold_manifest"],
            "model_capacity": fitted["model_capacity"],
            "pilot_episode_count": len(seed_rows),
        }

    episode_path = output / "episodes" / "pilot_episodes.jsonl.gz"
    episode_sha = _write_jsonl_gzip(episode_path, all_rows)
    summary = {
        "experiment": "G8.4.8.8 arm-blinded engineering and variance pilot",
        "status": "PILOT_ONLY_NOT_CONFIRMATORY",
        "seeds": list(seeds),
        "confirmatory_seeds_reserved": list(confirmatory_seeds),
        "nominal_rhos": list(PILOT_RHOS),
        "episodes_per_arm_per_stratum": episodes_per_arm_per_stratum,
        "confirmatory_arms": list(ARMS),
        "pilot_only_arms": list(PILOT_ONLY_ARMS),
        "random_trigger_negative_control_arm": "random_overlay_pilot",
        "dtmc_style_predictive_comparator_arm": "dtmc_overlay_pilot",
        "random_overlay_expected_trigger_fraction": 0.10,
        "horizon": 1,
        "episode_length": 8,
        "task_manifests": task_manifest_records(),
        "task_manifest_sha256": validate_task_manifests(),
        "seed_metadata": seed_metadata,
        "episode_records": str(episode_path.relative_to(output)),
        "episode_records_sha256": episode_sha,
        "episode_count": len(all_rows),
        "arm_blinded_dispersion": _pooled_within_block_dispersion(all_rows),
        "confirmatory_effects_or_p_values": None,
        "margins_approved": False,
        "next_gate": (
            "PI/domain owner must approve delta_U, delta_JS, delta_L and task "
            "manifest semantics before the max-t sample-size gate and any "
            "confirmatory test generation."
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_sha": os.environ.get("UCON_JEPA_HEAD_SHA", os.environ.get("GITHUB_SHA")),
        },
    }
    summary_path = output / "pilot_summary.json"
    summary_sha = _write_json(summary_path, summary)
    _write_json(output / "pilot_file_manifest.json", {
        "pilot_summary.json": summary_sha,
        "episode_records": episode_sha,
        "seed_artifacts": seed_metadata,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_PILOT_SEEDS)))
    parser.add_argument(
        "--confirmatory-seeds",
        default=",".join(map(str, DEFAULT_CONFIRMATORY_SEEDS)),
    )
    parser.add_argument("--episodes-per-arm-per-stratum", type=int, default=10)
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--out", default="intervention-pilot-output")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    confirmatory_seeds = tuple(
        int(value) for value in args.confirmatory_seeds.split(",") if value
    )
    summary = run_pilot(
        seeds=seeds,
        confirmatory_seeds=confirmatory_seeds,
        episodes_per_arm_per_stratum=args.episodes_per_arm_per_stratum,
        episodes_per_cell=args.episodes_per_cell,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
        output=Path(args.out),
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
