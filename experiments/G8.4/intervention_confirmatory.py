"""Confirmatory four-arm G8.4.8.8 episode trial driver.

This script refuses to generate confirmatory episodes unless a retained
max-t power report authorizes the exact task-manifest and seed design.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from intervention_modeling import fit_predictors, save_model_checkpoint
from intervention_simulator import ARMS, make_blocked_episode_specs, run_episode
from intervention_task_manifests import validate_task_manifests

DEFAULT_CONFIRMATORY_SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
PRIMARY_RHO = 0.85
PRIMARY_HORIZON = 1
PRIMARY_ALERT_FRACTION = 0.10
ASSIGNMENT_SEED = 884_808


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def _validate_power_report(
    report_path: Path,
    confirmatory_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], str]:
    if not report_path.is_file():
        raise FileNotFoundError(f"power report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha = _sha256(report_path)
    if report.get("status") != "POWERED" or not report.get("confirmatory_generation_authorized"):
        raise RuntimeError("confirmatory generation blocked: max-t power gate did not authorize it")
    if sorted(int(value) for value in report.get("confirmatory_seeds", [])) != sorted(confirmatory_seeds):
        raise ValueError("confirmatory seed list differs from the approved power report")
    if len(confirmatory_seeds) != 5 or len(set(confirmatory_seeds)) != 5:
        raise ValueError("exactly five unique confirmatory seeds are required")
    current_hashes = validate_task_manifests()
    if report.get("task_manifest_sha256") != current_hashes:
        raise ValueError("task-manifest hashes differ from the approved power report")
    if abs(float(report.get("target_power", 0.0)) - 0.90) > 1e-12:
        raise ValueError("power report target must be the preregistered 90%")
    if abs(float(report.get("alpha_one_sided", 0.0)) - 0.05) > 1e-12:
        raise ValueError("power report alpha must be the preregistered one-sided 0.05")
    if float(report.get("selected_joint_power", 0.0)) < 0.90:
        raise ValueError("selected power does not reach the preregistered target")
    n = int(report.get("episodes_per_arm_per_stratum", 0))
    if not 2 <= n <= 300:
        raise ValueError("power report sample size is outside the preregistered bounds")
    expected_blocks = len(confirmatory_seeds) * 5 * 3 * 2
    expected_total = expected_blocks * len(ARMS) * n
    if int(report.get("confirmatory_strata_count", -1)) != expected_blocks:
        raise ValueError("power report has an unexpected confirmatory stratum count")
    if int(report.get("total_confirmatory_episodes", -1)) != expected_total:
        raise ValueError("power report total episode count is inconsistent")
    return report, report_sha


def run_confirmatory(
    *,
    power_report_path: Path,
    confirmatory_seeds: tuple[int, ...],
    episodes_per_cell: int,
    epochs: int,
    mc_samples: int,
    approval_reference: str,
    output: Path,
) -> dict[str, Any]:
    if not approval_reference.strip():
        raise ValueError("an explicit PI/domain-owner approval reference is required")
    power_report, power_report_sha = _validate_power_report(
        power_report_path, confirmatory_seeds
    )
    output.mkdir(parents=True, exist_ok=True)
    n = int(power_report["episodes_per_arm_per_stratum"])
    freeze = {
        "experiment": "G8.4.8.8 confirmatory randomized capability-overlay trial",
        "status": "PRE_REGISTERED_CONFIRMATORY_RUN",
        "pi_domain_owner_approval_reference": approval_reference.strip(),
        "source_sha": os.environ.get("UCON_JEPA_HEAD_SHA", os.environ.get("GITHUB_SHA")),
        "power_report_sha256": power_report_sha,
        "task_manifest_sha256": validate_task_manifests(),
        "confirmatory_seeds": list(confirmatory_seeds),
        "rho": PRIMARY_RHO,
        "horizon": PRIMARY_HORIZON,
        "episodes_per_arm_per_stratum": n,
        "total_episodes": int(power_report["total_confirmatory_episodes"]),
        "arms": list(ARMS),
        "assignment_seed": ASSIGNMENT_SEED,
        "overlay_duration_decisions": 2,
        "alert_fraction": PRIMARY_ALERT_FRACTION,
        "model_fit": {
            "episodes_per_cell": int(episodes_per_cell),
            "epochs": int(epochs),
            "mc_samples": int(mc_samples),
            "task_aware": True,
        },
        "task_manifest_semantics": "current versioned manifest hashes; PI/domain-owner approval must be recorded externally before dispatch",
        "target_effect_assumptions": power_report["target_effect_assumptions"],
        "margins": power_report["margins"],
        "primary_endpoint": "episode count of submitted proposals with contemporaneous UCON DENY",
        "task_guardrail": "episode task completion; simultaneous non-inferiority against all three comparators",
        "randomization": "balanced within seed × archetype × family × task-difficulty block",
        "agent_response_model": (
            "synthetic capability-aware replan: if base candidate is unavailable, "
            "uniformly sample from the currently available capability set"
        ),
    }
    freeze_path = output / "preregistration_freeze.json"
    freeze_sha = _write_json(freeze_path, freeze)

    seed_metadata = {}
    total_episode_count = 0
    file_hashes = {"preregistration_freeze.json": freeze_sha}
    for seed in confirmatory_seeds:
        fitted = fit_predictors(
            int(seed), episodes_per_cell, epochs, mc_samples,
            evaluate_holdout=False,
        )
        assignment_rows = make_blocked_episode_specs(
            [int(seed)], n,
            assignment_seed=ASSIGNMENT_SEED,
            rho=PRIMARY_RHO,
            arms=ARMS,
        )
        assignment_ids = {str(row["episode_id"]) for row in assignment_rows}
        training_ids = set().union(*fitted["partition_ids"].values())
        if assignment_ids & training_ids:
            raise AssertionError("confirmatory episode IDs overlap model-fitting partitions")
        if len(assignment_ids) != 5 * 3 * 2 * len(ARMS) * n:
            raise AssertionError("unexpected per-seed assignment count")

        model_path = output / "models" / f"seed_{seed}_neural_predictors.npz"
        model_sha = save_model_checkpoint(
            model_path, fitted["models"], fitted["scalers"]
        )
        dtmc_path = output / "models" / f"seed_{seed}_dtmc.json"
        dtmc_sha = _write_json(dtmc_path, fitted["dtmc_model"].to_dict())
        assignment_path = output / "assignments" / f"seed_{seed}_assignments.jsonl.gz"
        assignment_sha = _write_jsonl_gzip(assignment_path, assignment_rows)

        episode_path = output / "episodes" / f"seed_{seed}_episodes.jsonl.gz"
        thresholds = fitted["thresholds_by_fraction"][PRIMARY_ALERT_FRACTION]
        predictor_failure_count = 0
        trigger_counts = {arm: 0 for arm in ARMS}
        episode_count = 0
        with _open_jsonl_gzip_writer(episode_path) as episode_handle:
            for spec in assignment_rows:
                episode_row = run_episode(
                    spec,
                    score_functions=fitted["score_functions"],
                    thresholds=thresholds,
                    horizon=PRIMARY_HORIZON,
                )
                episode_handle.write(
                    json.dumps(_jsonable(episode_row), sort_keys=True) + "\\n"
                )
                episode_count += 1
                predictor_failure_count += int(episode_row["predictor_failed"])
                trigger_counts[episode_row["arm"]] += int(episode_row["overlay_triggered"])
        episode_sha = _sha256(episode_path)
        total_episode_count += episode_count
        assignment_rel = str(assignment_path.relative_to(output))
        episode_rel = str(episode_path.relative_to(output))
        file_hashes[assignment_rel] = assignment_sha
        file_hashes[episode_rel] = episode_sha
        file_hashes[str(model_path.relative_to(output))] = model_sha
        file_hashes[str(dtmc_path.relative_to(output))] = dtmc_sha
        seed_metadata[str(seed)] = {
            "neural_model_checkpoint": str(model_path.relative_to(output)),
            "neural_model_checkpoint_sha256": model_sha,
            "dtmc_model": str(dtmc_path.relative_to(output)),
            "dtmc_model_sha256": dtmc_sha,
            "assignment_manifest": str(assignment_path.relative_to(output)),
            "assignment_manifest_sha256": assignment_sha,
            "episode_records": str(episode_path.relative_to(output)),
            "episode_records_sha256": episode_sha,
            "episode_count": episode_count,
            "overlay_trigger_counts_by_arm": trigger_counts,
            "train_episode_id_sha256": hashlib.sha256(
                "\n".join(sorted(fitted["partition_ids"]["train"])).encode()
            ).hexdigest(),
            "validation_episode_id_sha256": hashlib.sha256(
                "\n".join(sorted(fitted["partition_ids"]["validation"])).encode()
            ).hexdigest(),
            "calibration_episode_id_sha256": hashlib.sha256(
                "\n".join(sorted(fitted["partition_ids"]["calibration"])).encode()
            ).hexdigest(),
            "threshold_manifest": fitted["threshold_manifest"],
            "dataset_manifest": fitted["dataset_manifest"],
            "model_capacity": fitted["model_capacity"],
            "predictor_failure_episodes": int(predictor_failure_count),
        }

    expected_total = int(power_report["total_confirmatory_episodes"])
    if total_episode_count != expected_total:
        raise AssertionError("confirmatory episode total does not match the authorized power report")

    manifest = {
        **freeze,
        "preregistration_freeze_sha256": freeze_sha,
        "power_report": str(power_report_path),
        "power_report_sha256": power_report_sha,
        "episode_count": total_episode_count,
        "seed_metadata": seed_metadata,
        "artifact_hashes": file_hashes,
        "inference_performed": False,
        "treatment_contrasts_or_p_values": None,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "next_step": "Run the preregistered blocked episode-bootstrap and randomization analysis; do not tune or modify the frozen trial.",
    }
    manifest_path = output / "confirmatory_run_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    _write_json(output / "file_manifest.json", {
        "preregistration_freeze.json": freeze_sha,
        "confirmatory_run_manifest.json": manifest_sha,
        "seed_artifact_hashes": file_hashes,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-report", required=True)
    parser.add_argument("--confirmatory-seeds", default=",".join(map(str, DEFAULT_CONFIRMATORY_SEEDS)))
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--out", default="confirmatory-output")
    args = parser.parse_args()
    manifest = run_confirmatory(
        power_report_path=Path(args.power_report),
        confirmatory_seeds=tuple(
            int(value) for value in args.confirmatory_seeds.split(",") if value
        ),
        episodes_per_cell=args.episodes_per_cell,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
        approval_reference=args.approval_reference,
        output=Path(args.out),
    )
    print(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
