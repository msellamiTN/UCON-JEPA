"""G8.4.8.7.9 three-factor seed decomposition benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from jepa_numpy import JEPAConfig
from policy_parameter_dataset import build_dataset, write_dataset
from policy_shift_crossed_seed_initial_direction import (
    CALIBRATION_ROLLOUT_SEED, SHIFTED_ROLLOUT_SEED, REVERSE_ROLLOUT_SEED,
    _assert_episode_disjoint, _position_zero, _score_summary,
    _bootstrap_cell, _classify_direction, _check_diagonal_reproduction,
    _fit_models as _fit_legacy_models, _predict_e2, _group_metrics_safe,
)
from policy_shift_matched_recursive_control import _fit_supervised_action_head
from policy_shift_recursive_risk import _apply_platt, _fit_platt
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from predictive_benchmark import metrics as binary_metrics
from uncertain_action_jepa import _transform, fit_e1_e2_models

SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
HORIZON = 1
LEGACY_SEED = 20261119


def fit_factor_models(dataset: dict[str, Any], init_seed: int, order_seed: int,
                      epochs: int):
    jepa, action_head, _, scalers = fit_e1_e2_models(
        dataset, seed=LEGACY_SEED, epochs=epochs,
        initialization_seed=init_seed, minibatch_order_seed=order_seed,
    )
    train = dataset["train"]
    config = JEPAConfig(
        latent_dim=8, learning_rate=0.008, epochs=epochs, batch_size=128,
        ema_tau=0.99, variance_floor=0.2, variance_weight=0.1,
        security_weight=0.5, seed=LEGACY_SEED,
        initialization_seed=init_seed, minibatch_order_seed=order_seed,
    )
    supervised = LinearSupervisedStateDynamics(
        train["context"].shape[1], train["action"].shape[1],
        train["security"].shape[1], config,
    ).fit(
        _transform(train["context"], scalers["context_scaler"]),
        _transform(train["action"], scalers["action_scaler"]),
        _transform(train["future"], scalers["context_scaler"]),
        train["security"],
    )
    supervised_action, supervised_temperature = _fit_supervised_action_head(
        supervised, dataset, scalers["context_scaler"]
    )
    if jepa.parameter_count() != 852 or supervised.parameter_count() != 776:
        raise AssertionError("model parameter count changed")
    models = {
        "JEPA_E2": (jepa, action_head, float(scalers["temperature"])),
        "supervised_E2": (supervised, supervised_action, float(supervised_temperature)),
    }
    metadata = {
        "JEPA_E2": {
            "parameters": jepa.parameter_count(),
            "final_latent_loss": float(jepa.history.latent_loss[-1]),
            "final_security_loss": float(jepa.history.security_loss[-1]),
            "ema_target_distance": float(jepa.ema_target_distance()),
        },
        "supervised_E2": {
            "parameters": supervised.parameter_count(),
            "final_future_context_mse": float(supervised.history[-1]["future_mse"]),
            "final_security_mse": float(supervised.history[-1]["security_mse"]),
        },
        "initialization_seed": init_seed,
        "minibatch_order_seed": order_seed,
        "rng_scheme": "NumPy SeedSequence with domain tags 0x1A17 and 0xB47C",
    }
    return models, scalers, metadata


def additive_three_factor(values: dict[str, float]) -> dict[str, Any]:
    cube = np.empty((5, 5, 5), dtype=np.float64)
    for i, d in enumerate(SEEDS):
        for j, init in enumerate(SEEDS):
            for k, order in enumerate(SEEDS):
                cube[i, j, k] = values[f"{d}:{init}:{order}"]
    grand = float(cube.mean())
    effects = [cube.mean(axis=tuple(a for a in range(3) if a != axis)) - grand
               for axis in range(3)]
    ss_total = float(((cube - grand) ** 2).sum())
    ss = [float(25 * np.square(effect).sum()) for effect in effects]
    residual = max(0.0, ss_total - sum(ss))
    names = ("dataset_seed", "initialization_seed", "minibatch_order_seed")
    return {
        "grand_mean": grand,
        "factor_means": {
            name: {str(seed): float(grand + effects[idx][level])
                   for level, seed in enumerate(SEEDS)}
            for idx, name in enumerate(names)
        },
        "sum_squares": {
            "total": ss_total,
            **{name: ss[idx] for idx, name in enumerate(names)},
            "residual_including_interactions": residual,
            "three_way_interaction_not_separately_identifiable": True,
        },
        "sum_squares_shares_percent": {
            name: (100.0 * ss[idx] / ss_total if ss_total else 0.0)
            for idx, name in enumerate(names)
        } | {"residual_including_interactions":
             (100.0 * residual / ss_total if ss_total else 0.0)},
        "raw_cube": cube.tolist(),
    }


def run(out: Path, epochs: int, episodes: int, mc_samples: int, bootstrap_replicates: int):
    out.mkdir(parents=True, exist_ok=True)
    datasets = {}
    manifests = {}
    for ds in SEEDS:
        data = build_dataset(seed=ds, episodes_per_cell=episodes)
        _assert_episode_disjoint(data)
        write_dataset(data, out / f"dataset-{ds}")
        datasets[ds] = data
        manifests[str(ds)] = data["manifest"]

    cells = {}
    diagonal_legacy = {}
    counter = 0
    for ds in SEEDS:
        data = datasets[ds]
        for init in SEEDS:
            for order in SEEDS:
                key = f"{ds}:{init}:{order}"
                models, scalers, metadata = fit_factor_models(data, init, order, epochs)
                calibration = {}
                calibrators = {}
                for name, (model, action_head, temperature) in models.items():
                    pred = _predict_e2(
                        model, action_head, data["calibration"], HORIZON,
                        scalers["context_scaler"], scalers["action_scaler"],
                        temperature, mc_samples, CALIBRATION_ROLLOUT_SEED,
                    )
                    calibration[name] = _position_zero(pred)
                    calibrators[name] = _fit_platt(
                        calibration[name]["raw"], calibration[name]["truth"]
                    )

                split_results = {}
                saved = {}
                for split, partition, seed in (
                    ("shifted", "test_shifted", SHIFTED_ROLLOUT_SEED),
                    ("reverse", "test_reverse", REVERSE_ROLLOUT_SEED),
                ):
                    split_results[split] = {}
                    for name, (model, action_head, temperature) in models.items():
                        pred = _predict_e2(
                            model, action_head, data[partition], HORIZON,
                            scalers["context_scaler"], scalers["action_scaler"],
                            temperature, mc_samples, seed,
                        )
                        part = _position_zero(pred)
                        calibrated = _apply_platt(part["raw"], calibrators[name])
                        split_results[split][name] = {
                            "n": len(part["truth"]),
                            "positive": int(part["truth"].sum()),
                            "raw_score": _score_summary(part["raw"], part["truth"]),
                            "raw_metrics": binary_metrics(part["truth"], part["raw"]),
                            "calibrated_metrics": binary_metrics(part["truth"], calibrated),
                            "by_archetype_family": _group_metrics_safe(part, calibrated),
                            "calibration": calibrators[name],
                        }
                        saved[(split, name)] = (part, calibrated)

                counter += 1
                bootstrap = _bootstrap_cell(
                    calibration,
                    {name: saved[("shifted", name)][0] for name in models},
                    seed=41_000_000 + counter,
                    replicates=bootstrap_replicates,
                )
                for name in models:
                    cal_summary = _score_summary(
                        calibration[name]["raw"], calibration[name]["truth"]
                    )
                    result = split_results["shifted"][name]
                    result["calibration_score"] = cal_summary
                    result["direction_classification"] = _classify_direction(
                        cal_summary["positive_minus_negative_mean"],
                        calibrators[name]["slope"],
                        result["raw_score"]["positive_minus_negative_mean"],
                        result["raw_metrics"]["auroc"],
                        bootstrap["models"][name],
                    )

                jepa = split_results["shifted"]["JEPA_E2"]
                sup = split_results["shifted"]["supervised_E2"]
                paired = bootstrap["paired_JEPA_minus_supervised"]
                np.savez_compressed(
                    out / f"predictions-d{ds}-i{init}-b{order}.npz",
                    calibration_episode_id=calibration["JEPA_E2"]["episode_id"],
                    calibration_truth=calibration["JEPA_E2"]["truth"],
                    calibration_policy=calibration["JEPA_E2"]["agent_policy"],
                    calibration_family=calibration["JEPA_E2"]["family"],
                    jepa_calibration_raw=calibration["JEPA_E2"]["raw"],
                    supervised_calibration_raw=calibration["supervised_E2"]["raw"],
                    shifted_episode_id=saved[("shifted", "JEPA_E2")][0]["episode_id"],
                    shifted_truth=saved[("shifted", "JEPA_E2")][0]["truth"],
                    shifted_policy=saved[("shifted", "JEPA_E2")][0]["agent_policy"],
                    shifted_family=saved[("shifted", "JEPA_E2")][0]["family"],
                    jepa_shifted_raw=saved[("shifted", "JEPA_E2")][0]["raw"],
                    supervised_shifted_raw=saved[("shifted", "supervised_E2")][0]["raw"],
                    jepa_shifted_calibrated=saved[("shifted", "JEPA_E2")][1],
                    supervised_shifted_calibrated=saved[("shifted", "supervised_E2")][1],
                )
                cells[key] = {
                    "dataset_seed": ds,
                    "initialization_seed": init,
                    "minibatch_order_seed": order,
                    "models": split_results,
                    "model_metadata": metadata,
                    "paired": {
                        "JEPA_minus_supervised_raw_auroc":
                            jepa["raw_metrics"]["auroc"] - sup["raw_metrics"]["auroc"],
                        "JEPA_minus_supervised_calibration_slope":
                            jepa["calibration"]["slope"] - sup["calibration"]["slope"],
                        "bootstrap": paired,
                    },
                    "bootstrap": bootstrap,
                    "rollout_seeds": {
                        "calibration": CALIBRATION_ROLLOUT_SEED,
                        "shifted": SHIFTED_ROLLOUT_SEED,
                        "reverse": REVERSE_ROLLOUT_SEED,
                    },
                }

    # Historical combined-seed path is independently reproduced and never mixed
    # with the domain-separated factorial cells.
    for ds in SEEDS:
        legacy_models, legacy_fitted = _fit_legacy_models(datasets[ds], ds, epochs)
        diagonal_legacy[str(ds)] = _check_diagonal_reproduction(
            ds, legacy_models, datasets[ds], legacy_fitted["scalers"],
            json.loads((Path(__file__).parent / "references" /
                        "g8_4_8_7_7_diagonal_position0_reference.json").read_text()),
            mc_samples,
        )

    factor_analysis = {}
    for model in ("JEPA_E2", "supervised_E2"):
        metrics = {
            "calibration_delta": lambda c: c["models"]["shifted"][model]["calibration_score"]["positive_minus_negative_mean"],
            "calibration_slope": lambda c: c["models"]["shifted"][model]["calibration"]["slope"],
            "shifted_raw_delta": lambda c: c["models"]["shifted"][model]["raw_score"]["positive_minus_negative_mean"],
            "shifted_raw_auroc": lambda c: c["models"]["shifted"][model]["raw_metrics"]["auroc"],
            "shifted_raw_brier": lambda c: c["models"]["shifted"][model]["raw_metrics"]["brier"],
            "shifted_calibrated_brier": lambda c: c["models"]["shifted"][model]["calibrated_metrics"]["brier"],
        }
        factor_analysis[model] = {
            metric: additive_three_factor({
                key: float(extractor(cell)) for key, cell in cells.items()
            }) for metric, extractor in metrics.items()
        }

    recurrence = {}
    for factor, field in (
        ("dataset_seed", "dataset_seed"),
        ("initialization_seed", "initialization_seed"),
        ("minibatch_order_seed", "minibatch_order_seed"),
    ):
        recurrence[factor] = {}
        for level in SEEDS:
            selected = [c for c in cells.values() if c[field] == level]
            negative = sum(
                c["models"]["shifted"]["JEPA_E2"]["direction_classification"] == "supported_negative"
                for c in selected
            )
            supervised_negative = sum(
                c["models"]["shifted"]["supervised_E2"]["direction_classification"] == "supported_negative"
                for c in selected
            )
            paired_auc = sum(
                c["paired"]["bootstrap"]["shifted_test_raw_auroc_ci95"][1] < 0
                for c in selected
            )
            paired_slope = sum(
                c["paired"]["bootstrap"]["calibration_slope_ci95"][1] < 0
                for c in selected
            )
            recurrence[factor][str(level)] = {
                "cells": len(selected),
                "JEPA_supported_negative": negative,
                "supervised_supported_negative": supervised_negative,
                "paired_raw_auroc_CI_below_zero": paired_auc,
                "paired_calibration_slope_CI_below_zero": paired_slope,
                "JEPA_specific_recurrence": (
                    negative >= 20 and supervised_negative <= 5
                    and paired_auc >= 20 and paired_slope >= 20
                ),
            }

    interaction = {}
    for init in SEEDS:
        interaction[str(init)] = {}
        for order in SEEDS:
            selected = [c for c in cells.values()
                        if c["initialization_seed"] == init
                        and c["minibatch_order_seed"] == order]
            interaction[str(init)][str(order)] = {
                "JEPA_supported_negative": sum(
                    c["models"]["shifted"]["JEPA_E2"]["direction_classification"] == "supported_negative"
                    for c in selected
                ),
                "supervised_supported_negative": sum(
                    c["models"]["shifted"]["supervised_E2"]["direction_classification"] == "supported_negative"
                    for c in selected
                ),
                "mean_JEPA_minus_supervised_AUROC": float(np.mean([
                    c["paired"]["JEPA_minus_supervised_raw_auroc"] for c in selected
                ])),
            }

    result = {
        "benchmark": "G8.4.8.7.9 initialization × minibatch-order × dataset seed decomposition",
        "code_sha": os.environ.get("UCON_JEPA_CODE_SHA"),
        "dataset_seeds": list(SEEDS),
        "initialization_seeds": list(SEEDS),
        "minibatch_order_seeds": list(SEEDS),
        "cell_count": len(cells),
        "cells": cells,
        "legacy_combined_seed_reproduction": diagonal_legacy,
        "factor_analysis": factor_analysis,
        "factor_recurrence": recurrence,
        "initialization_by_minibatch_interaction": interaction,
        "dataset_manifests": manifests,
        "protocol": {
            "design": "fully crossed 5x5x5 dataset × initialization × minibatch-order",
            "models": ["JEPA_E2", "supervised_E2"],
            "horizon": 1,
            "primary_support": "trajectory start position 0",
            "primary_test": "shifted rho=0.85",
            "reverse_test": "rho=0.05 diagnostic only",
            "mc_samples": mc_samples,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_unit": "whole episode stratified by policy × family × label",
            "rng_domain_tags": {"initialization": "0x1A17", "minibatch_order": "0xB47C"},
        },
        "limitations": [
            "synthetic benchmark only",
            "five levels per factor; recurrence thresholds are preregistered descriptive criteria",
            "episode bootstrap intervals are conditional on observed class counts",
            "three-way interaction is confounded with residual with one observation per cell",
            "training-seed decomposition does not establish real-agent generalization or production assurance",
        ],
        "authorization_invariant": "Predictions advisory only; formal UCON remains the sole authorization authority.",
    }
    (out / "init_minibatch_factorial_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    manifest = {"files": {str(p.relative_to(out)): {
        "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
    } for p in sorted(files)}}
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("init-minibatch-factorial-output"))
    args = parser.parse_args()
    if (args.episodes_per_cell, args.epochs, args.mc_samples, args.bootstrap_replicates) != (300, 180, 32, 2000):
        raise ValueError("all benchmark budgets are fixed by preregistration")
    result = run(args.out, args.epochs, args.episodes_per_cell,
                 args.mc_samples, args.bootstrap_replicates)
    print(json.dumps({
        "code_sha": result["code_sha"],
        "cells": result["cell_count"],
        "legacy_diagonal_reproductions": len(result["legacy_combined_seed_reproduction"]),
        "output": str(args.out / "init_minibatch_factorial_results.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
