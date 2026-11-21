"""Preregistered blocked episode-bootstrap and randomization analysis."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from intervention_simulator import ARMS
from intervention_task_manifests import TASK_MANIFEST_HASHES, validate_task_manifests

CONTRAST_SPECS = (
    ("jepa_vs_ucon_denied", 0, 0, "upper", "delta_U"),
    ("jepa_vs_reactive_denied", 0, 1, "upper", "delta_U"),
    ("jepa_vs_supervised_denied", 0, 3, "upper", "delta_JS"),
    ("jepa_vs_ucon_task_completion", 1, 0, "lower", "delta_L"),
    ("jepa_vs_reactive_task_completion", 1, 1, "lower", "delta_L"),
    ("jepa_vs_supervised_task_completion", 1, 3, "lower", "delta_L"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_and_validate_trial(
    trial_dir: Path,
    power_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]:
    run_path = trial_dir / "confirmatory_run_manifest.json"
    freeze_path = trial_dir / "preregistration_freeze.json"
    file_manifest_path = trial_dir / "file_manifest.json"
    for path in (run_path, freeze_path, file_manifest_path, power_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    file_manifest = json.loads(file_manifest_path.read_text(encoding="utf-8"))
    power_report_sha = _sha256(power_report_path)
    if power_report_sha != run_manifest.get("power_report_sha256"):
        raise ValueError("power report checksum does not match confirmatory run manifest")
    power_report = json.loads(power_report_path.read_text(encoding="utf-8"))
    if power_report.get("status") != "POWERED" or not power_report.get("confirmatory_generation_authorized"):
        raise ValueError("confirmatory trial references a non-authorizing power report")
    if _sha256(freeze_path) != run_manifest.get("preregistration_freeze_sha256"):
        raise ValueError("preregistration freeze checksum mismatch")
    if freeze.get("power_report_sha256") != power_report_sha:
        raise ValueError("freeze and power report checksums differ")
    if freeze.get("task_manifest_sha256") != validate_task_manifests():
        raise ValueError("current task manifests differ from frozen trial")
    if power_report.get("task_manifest_sha256") != freeze.get("task_manifest_sha256"):
        raise ValueError("power report and frozen task manifests differ")
    if freeze.get("margins") != power_report.get("margins"):
        raise ValueError("frozen margins differ from power report")
    if freeze.get("target_effect_assumptions") != power_report.get("target_effect_assumptions"):
        raise ValueError("frozen target effects differ from power report")
    if tuple(freeze.get("arms", [])) != tuple(ARMS):
        raise ValueError("confirmatory arm order differs from preregistration")
    if float(freeze.get("rho", -1)) != 0.85 or int(freeze.get("horizon", -1)) != 1:
        raise ValueError("confirmatory rho/horizon differs from preregistration")
    if not freeze.get("pi_domain_owner_approval_reference"):
        raise ValueError("missing PI/domain-owner approval reference")
    if _sha256(run_path) != file_manifest.get("confirmatory_run_manifest.json"):
        raise ValueError("run manifest checksum mismatch")

    expected_seeds = [int(value) for value in freeze["confirmatory_seeds"]]
    if sorted(int(value) for value in run_manifest.get("confirmatory_seeds", [])) != sorted(expected_seeds):
        raise ValueError("run manifest seed list differs from freeze")
    n = int(freeze["episodes_per_arm_per_stratum"])
    if n != int(power_report["episodes_per_arm_per_stratum"]):
        raise ValueError("sample size differs from approved power report")

    rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        metadata = run_manifest["seed_metadata"][str(seed)]
        for key in ("episode_records", "assignment_manifest", "neural_model_checkpoint", "dtmc_model"):
            relative = Path(metadata[key])
            path = trial_dir / relative
            expected_hash = metadata[
                "episode_records_sha256" if key == "episode_records"
                else "assignment_manifest_sha256" if key == "assignment_manifest"
                else "neural_model_checkpoint_sha256" if key == "neural_model_checkpoint"
                else "dtmc_model_sha256"
            ]
            if _sha256(path) != expected_hash:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        assignment_rows = _read_jsonl_gzip(trial_dir / metadata["assignment_manifest"])
        episode_rows = _read_jsonl_gzip(trial_dir / metadata["episode_records"])
        assigned = {
            str(row["episode_id"]): str(row["arm"]) for row in assignment_rows
        }
        observed = {
            str(row["episode_id"]): str(row["arm"]) for row in episode_rows
        }
        if assigned != observed:
            raise ValueError(f"assignment/outcome mismatch for seed {seed}")
        rows.extend(episode_rows)

    expected_total = int(power_report["total_confirmatory_episodes"])
    if len(rows) != expected_total or len({str(row["episode_id"]) for row in rows}) != expected_total:
        raise ValueError("confirmatory episode count or episode-ID uniqueness check failed")
    if int(run_manifest.get("episode_count", -1)) != expected_total:
        raise ValueError("run manifest episode count mismatch")
    return run_manifest, freeze, rows, n


def _build_outcome_array(
    rows: list[dict[str, Any]],
    n_per_arm_per_block: int,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_ids = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode id: {episode_id}")
        seen_ids.add(episode_id)
        if row["arm"] not in ARMS:
            raise ValueError(f"unexpected confirmatory arm: {row['arm']}")
        if float(row["rho"]) != 0.85:
            raise ValueError("non-primary rho found in confirmatory outcomes")
        if row["task_manifest_sha256"] != TASK_MANIFEST_HASHES[str(row["family"])]:
            raise ValueError("episode task-manifest hash mismatch")
        if int(row["unauthorized_execution_count"]) != 0:
            raise ValueError("unauthorized execution detected")
        if int(row["state_violation_count"]) != 0:
            raise ValueError("UCON state-mutation invariant violation detected")
        if int(row["ucon_oracle_call_count"]) != 8 or len(row["records"]) != 8:
            raise ValueError("each proposal must be checked by UCON at all eight decision steps")
        grouped[(str(row["stratum"]), str(row["arm"]))].append(row)

    blocks = sorted({key[0] for key in grouped})
    expected_blocks = 150
    if len(blocks) != expected_blocks:
        raise ValueError(f"expected {expected_blocks} confirmatory blocks, observed {len(blocks)}")
    values = np.empty((len(blocks), len(ARMS), n_per_arm_per_block, 2), dtype=np.float64)
    block_summaries = []
    for block_index, block in enumerate(blocks):
        for arm_index, arm in enumerate(ARMS):
            cell = grouped.get((block, arm), [])
            if len(cell) != n_per_arm_per_block:
                raise ValueError(
                    f"unbalanced block {block}, arm {arm}: {len(cell)} != {n_per_arm_per_block}"
                )
            for episode_index, row in enumerate(cell):
                values[block_index, arm_index, episode_index, 0] = float(row["denied_count"])
                values[block_index, arm_index, episode_index, 1] = float(row["task_complete"])
        sample = grouped[(block, ARMS[0])][0]
        for arm in ARMS:
            cell = grouped[(block, arm)]
            block_summaries.append({
                "stratum": block,
                "seed": int(sample["seed"]),
                "archetype": sample["archetype"],
                "family": sample["family"],
                "task_difficulty": sample["task_difficulty"],
                "arm": arm,
                "n": len(cell),
                "mean_denied_count": float(np.mean([row["denied_count"] for row in cell])),
                "task_completion_rate": float(np.mean([row["task_complete"] for row in cell])),
                "safe_task_completion_rate": float(np.mean([row["safe_task_complete"] for row in cell])),
                "overlay_trigger_rate": float(np.mean([row["overlay_triggered"] for row in cell])),
                "predictor_failure_rate": float(np.mean([row["predictor_failed"] for row in cell])),
            })
    return values, blocks, block_summaries


def _contrast_estimates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    block_count, _arm_count, n, _outcomes = values.shape
    means = values.mean(axis=2)
    variances = values.var(axis=2, ddof=1)
    estimates = np.empty(6, dtype=np.float64)
    standard_errors = np.empty(6, dtype=np.float64)
    for index, (_name, outcome, comparator, _bound, _margin) in enumerate(CONTRAST_SPECS):
        difference = means[:, 2, outcome] - means[:, comparator, outcome]
        estimates[index] = float(np.mean(difference))
        variance = np.sum(
            variances[:, 2, outcome] / n + variances[:, comparator, outcome] / n
        ) / (block_count ** 2)
        standard_errors[index] = float(np.sqrt(max(variance, 1e-18)))
    return estimates, standard_errors


def _bootstrap_max_t(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 20,
) -> tuple[float, np.ndarray]:
    block_count, arm_count, n, _outcomes = values.shape
    rng = np.random.default_rng(seed)
    maxima = np.empty(replicates, dtype=np.float64)
    orientations = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    completed = 0
    while completed < replicates:
        chunk = min(chunk_size, replicates - completed)
        indices = rng.integers(0, n, size=(chunk, block_count, arm_count, n))
        block_idx = np.arange(block_count)[None, :, None, None]
        arm_idx = np.arange(arm_count)[None, None, :, None]
        samples = values[block_idx, arm_idx, indices, :]
        means = samples.mean(axis=3)
        variances = samples.var(axis=3, ddof=1)
        t_values = np.empty((chunk, 6), dtype=np.float64)
        for index, (_name, outcome, comparator, _bound, _margin) in enumerate(CONTRAST_SPECS):
            tau = np.mean(means[:, :, 2, outcome] - means[:, :, comparator, outcome], axis=1)
            variance = np.sum(
                variances[:, :, 2, outcome] / n
                + variances[:, :, comparator, outcome] / n,
                axis=1,
            ) / (block_count ** 2)
            se = np.sqrt(np.maximum(variance, 1e-18))
            t_values[:, index] = orientations[index] * (tau - observed[index]) / se
        maxima[completed:completed + chunk] = np.max(t_values, axis=1)
        completed += chunk
    critical = float(np.quantile(maxima, 0.95))
    return critical, maxima


def _randomization_max_t(
    values: np.ndarray,
    observed: np.ndarray,
    observed_se: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 20,
) -> tuple[dict[str, float], float]:
    block_count, arm_count, n, _outcomes = values.shape
    rng = np.random.default_rng(seed)
    pooled = values.reshape(block_count, arm_count * n, 2)
    max_statistics = np.empty(replicates, dtype=np.float64)
    orientations = np.asarray([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    completed = 0
    while completed < replicates:
        chunk = min(chunk_size, replicates - completed)
        keys = rng.random((chunk, block_count, arm_count * n))
        permutations = np.argsort(keys, axis=2)
        permuted = pooled[np.arange(block_count)[None, :, None], permutations, :]
        permuted = permuted.reshape(chunk, block_count, arm_count, n, 2)
        means = permuted.mean(axis=3)
        variances = permuted.var(axis=3, ddof=1)
        t_values = np.empty((chunk, 6), dtype=np.float64)
        for index, (_name, outcome, comparator, _bound, _margin) in enumerate(CONTRAST_SPECS):
            tau = np.mean(means[:, :, 2, outcome] - means[:, :, comparator, outcome], axis=1)
            variance = np.sum(
                variances[:, :, 2, outcome] / n
                + variances[:, :, comparator, outcome] / n,
                axis=1,
            ) / (block_count ** 2)
            se = np.sqrt(np.maximum(variance, 1e-18))
            t_values[:, index] = orientations[index] * tau / se
        max_statistics[completed:completed + chunk] = np.max(t_values, axis=1)
        completed += chunk
    observed_t = orientations * observed / np.maximum(observed_se, 1e-18)
    adjusted = {
        name: float((1 + np.sum(max_statistics >= observed_t[index])) / (replicates + 1))
        for index, (name, *_rest) in enumerate(CONTRAST_SPECS)
    }
    global_p = float(
        (1 + np.sum(max_statistics >= np.max(observed_t))) / (replicates + 1)
    )
    return adjusted, global_p


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_trial(
    *,
    trial_dir: Path,
    power_report_path: Path,
    output: Path,
    bootstrap_replicates: int = 10000,
    randomization_replicates: int = 100000,
    seed: int = 20261121,
) -> dict[str, Any]:
    run_manifest, freeze, rows, n = _load_and_validate_trial(
        trial_dir, power_report_path
    )
    values, blocks, block_summaries = _build_outcome_array(rows, n)
    estimates, standard_errors = _contrast_estimates(values)
    critical, bootstrap_maxima = _bootstrap_max_t(
        values, estimates,
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    margins = freeze["margins"]
    contrast_results = []
    for index, (name, outcome, comparator, bound_type, margin_key) in enumerate(CONTRAST_SPECS):
        estimate = float(estimates[index])
        se = float(standard_errors[index])
        if bound_type == "upper":
            bound = estimate + critical * se
            threshold = (
                -float(margins[margin_key])
                if margin_key == "delta_U"
                else float(margins[margin_key])
            )
            passed = bound < threshold
            bound_label = "simultaneous_upper_bound"
        else:
            bound = estimate - critical * se
            threshold = -float(margins[margin_key])
            passed = bound > threshold
            bound_label = "simultaneous_lower_bound"
        contrast_results.append({
            "contrast": name,
            "estimate_jepa_minus_comparator": estimate,
            "standard_error": se,
            bound_label: float(bound),
            "decision_threshold": float(threshold),
            "passes_preregistered_directional_claim": bool(passed),
        })

    adjusted_p, global_p = _randomization_max_t(
        values, estimates, standard_errors,
        replicates=randomization_replicates,
        seed=seed + 2,
    )
    for row in contrast_results:
        row["sharp_null_max_t_adjusted_p_sensitivity"] = adjusted_p[row["contrast"]]

    per_arm = {}
    means = values.mean(axis=2)
    for arm_index, arm in enumerate(ARMS):
        per_arm[arm] = {
            "mean_denied_count_equal_block_weight": float(np.mean(means[:, arm_index, 0])),
            "mean_denied_rate_equal_block_weight": float(np.mean(means[:, arm_index, 0]) / 8.0),
            "task_completion_equal_block_weight": float(np.mean(means[:, arm_index, 1])),
        }

    secondary = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        secondary[arm] = {
            "any_deny_rate": float(np.mean([row["any_deny"] for row in arm_rows])),
            "safe_task_completion_rate": float(np.mean([row["safe_task_complete"] for row in arm_rows])),
            "recovered_task_completion_rate": float(np.mean([row["recovered_task_complete"] for row in arm_rows])),
            "overlay_trigger_rate": float(np.mean([row["overlay_triggered"] for row in arm_rows])),
            "mean_overlay_suppressed_count": float(np.mean([row["overlay_suppressed_count"] for row in arm_rows])),
            "mean_replanned_count": float(np.mean([row["replanned_count"] for row in arm_rows])),
            "predictor_failure_rate": float(np.mean([row["predictor_failed"] for row in arm_rows])),
        }

    all_claims_pass = all(
        row["passes_preregistered_directional_claim"] for row in contrast_results
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "block_arm_summary.csv", block_summaries)
    report = {
        "experiment": "G8.4.8.8 confirmatory randomized capability-overlay analysis",
        "status": "ANALYZED",
        "trial_manifest_sha256": _sha256(trial_dir / "confirmatory_run_manifest.json"),
        "power_report_sha256": _sha256(power_report_path),
        "preregistration_freeze_sha256": _sha256(trial_dir / "preregistration_freeze.json"),
        "confirmatory_seed_count": len(freeze["confirmatory_seeds"]),
        "confirmatory_block_count": len(blocks),
        "episodes_per_arm_per_block": n,
        "total_episode_count": len(rows),
        "arms": list(ARMS),
        "primary_endpoint": "DENY-labelled proposals per episode under the synthetic UCON oracle",
        "task_guardrail": "task completion probability",
        "simultaneous_confidence_level": 0.95,
        "bootstrap": {
            "method": "stratified episode-level bootstrap within block × arm",
            "replicates": int(bootstrap_replicates),
            "one_sided_max_t_critical": critical,
            "max_t_95th_percentile_check": float(np.quantile(bootstrap_maxima, 0.95)),
        },
        "simultaneous_contrasts": contrast_results,
        "all_six_preregistered_claims_pass": bool(all_claims_pass),
        "randomization_inference_sensitivity": {
            "method": "blocked label permutation preserving per-block arm counts; sharp global null",
            "replicates": int(randomization_replicates),
            "global_max_t_p_value": global_p,
            "interpretation": "sensitivity test of the sharp no-effect null; not a substitute for margin-based simultaneous confidence bounds",
        },
        "equal_block_weight_arm_summaries": per_arm,
        "secondary_descriptive_outcomes": secondary,
        "invariants": {
            "unauthorized_execution_count": 0,
            "state_violation_count": 0,
            "ucon_gate_calls_per_episode": 8,
            "assignment_outcome_identity_verified": True,
            "all_task_manifest_hashes_verified": True,
        },
        "scientific_limitations": [
            "Synthetic task and capability-aware replanning environment; not a real LLM-agent evaluation.",
            "DENY-labelled proposal burden is not breach incidence or realized security loss.",
            "Randomization inference is a sharp-null sensitivity; the six margin-based claims are decided by simultaneous bootstrap bounds.",
            "No action-level counterfactual or triggered-episode causal effect is identified.",
        ],
    }
    report_path = output / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_sha = _sha256(report_path)
    (output / "analysis_file_manifest.json").write_text(
        json.dumps({
            "analysis_report.json": report_sha,
            "block_arm_summary.csv": _sha256(output / "block_arm_summary.csv"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument("--power-report", required=True)
    parser.add_argument("--out", default="intervention-analysis-output")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--randomization-replicates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    report = analyze_trial(
        trial_dir=Path(args.trial_dir),
        power_report_path=Path(args.power_report),
        output=Path(args.out),
        bootstrap_replicates=args.bootstrap_replicates,
        randomization_replicates=args.randomization_replicates,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
