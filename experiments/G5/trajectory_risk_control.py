#!/usr/bin/env python3
"""Finite-sample workflow trajectory selective-risk simulation.

One row is one complete workflow episode, not one tool call. The script
compares fixed-sequence Clopper-Pearson, simultaneous Bonferroni CP, and
naive empirical threshold selection on synthetic IID episode populations.

Requires: numpy, scipy.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import beta


def cp_upper(errors: int, selected: int, delta: float) -> float:
    """One-sided exact Clopper-Pearson upper confidence bound."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if selected < 0 or errors < 0 or errors > selected:
        raise ValueError("require 0 <= errors <= selected")
    if selected == 0 or errors == selected:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, selected - errors))


@dataclass(frozen=True)
class Decision:
    threshold: float | None
    selected_count: int
    error_count: int
    upper_bound: float | None


def fixed_sequence_cp(
    scores: np.ndarray,
    outcomes: np.ndarray,
    thresholds: np.ndarray,
    target_risk: float,
    delta: float,
) -> Decision:
    """Return last certified threshold; stop at first uncertified candidate."""
    last: Decision = Decision(None, 0, 0, None)
    for threshold in thresholds:
        mask = scores <= threshold
        n = int(mask.sum())
        k = int(outcomes[mask].sum())
        upper = cp_upper(k, n, delta)
        if upper <= target_risk:
            last = Decision(float(threshold), n, k, upper)
        else:
            break
    return last


def bonferroni_cp(
    scores: np.ndarray,
    outcomes: np.ndarray,
    thresholds: np.ndarray,
    target_risk: float,
    delta: float,
) -> Decision:
    """Simultaneous CP bounds; choose largest certified threshold."""
    last: Decision = Decision(None, 0, 0, None)
    local_delta = delta / len(thresholds)
    for threshold in thresholds:
        mask = scores <= threshold
        n = int(mask.sum())
        k = int(outcomes[mask].sum())
        upper = cp_upper(k, n, local_delta)
        if upper <= target_risk:
            last = Decision(float(threshold), n, k, upper)
    return last


def naive_empirical(
    scores: np.ndarray,
    outcomes: np.ndarray,
    thresholds: np.ndarray,
    target_risk: float,
) -> Decision:
    """Invalid negative baseline: no finite-sample correction."""
    last: Decision = Decision(None, 0, 0, None)
    for threshold in thresholds:
        mask = scores <= threshold
        n = int(mask.sum())
        k = int(outcomes[mask].sum())
        empirical = (k / n) if n else 1.0
        if empirical <= target_risk:
            last = Decision(float(threshold), n, k, empirical)
    return last


def risk_curve(scenario: str) -> Callable[[np.ndarray], np.ndarray]:
    if scenario == "monotone":
        return lambda u: np.clip(0.02 + 0.30 * u, 0.0, 1.0)
    if scenario == "nonmonotone":
        return lambda u: np.clip(
            0.03 + 0.18 * u + 0.10 * np.sin(8.0 * np.pi * u),
            0.001,
            0.5,
        )
    if scenario == "infeasible":
        return lambda u: np.full_like(u, 0.14, dtype=float)
    raise ValueError(f"unknown scenario: {scenario}")


def population_selective_risks(
    thresholds: np.ndarray, p_error: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Numerically integrate E[p(U) | U <= t], U ~ Uniform(0,1)."""
    grid = np.linspace(0.0, 1.0, 200_001)
    values = p_error(grid)
    dx = grid[1] - grid[0]
    cumulative = np.concatenate(
        ([0.0], np.cumsum((values[:-1] + values[1:]) * 0.5 * dx))
    )
    return np.interp(thresholds, grid, cumulative) / thresholds


def simulate(
    seed: int = 20261118,
    repetitions: int = 1000,
    sample_size: int = 1500,
    target_risk: float = 0.10,
    delta: float = 0.05,
    threshold_count: int = 19,
) -> list[dict[str, float | int | str]]:
    if repetitions <= 0 or sample_size <= 0 or threshold_count <= 0:
        raise ValueError("repetitions, sample_size and threshold_count must be positive")
    if not (0.0 < target_risk < 1.0 and 0.0 < delta < 1.0):
        raise ValueError("target_risk and delta must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    thresholds = np.linspace(0.10, 1.00, threshold_count)
    output: list[dict[str, float | int | str]] = []

    for scenario in ("monotone", "nonmonotone", "infeasible"):
        p_error = risk_curve(scenario)
        true_risk = population_selective_risks(thresholds, p_error)
        methods = ("fixed_sequence_cp", "bonferroni_cp", "naive_empirical")
        failures = {method: 0 for method in methods}
        nulls = {method: 0 for method in methods}
        coverages = {method: [] for method in methods}

        for _ in range(repetitions):
            scores = rng.uniform(0.0, 1.0, size=sample_size)
            outcomes = rng.binomial(1, p_error(scores)).astype(np.int64)

            decisions = {
                "fixed_sequence_cp": fixed_sequence_cp(
                    scores, outcomes, thresholds, target_risk, delta
                ),
                "bonferroni_cp": bonferroni_cp(
                    scores, outcomes, thresholds, target_risk, delta
                ),
                "naive_empirical": naive_empirical(
                    scores, outcomes, thresholds, target_risk
                ),
            }
            for method, decision in decisions.items():
                if decision.threshold is None:
                    nulls[method] += 1
                    continue
                idx = int(np.argmin(np.abs(thresholds - decision.threshold)))
                failures[method] += int(true_risk[idx] > target_risk)
                coverages[method].append(decision.threshold)

        for method in methods:
            output.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "repetitions": repetitions,
                    "sample_size": sample_size,
                    "target_risk": target_risk,
                    "delta": delta,
                    "unsafe_return_rate": failures[method] / repetitions,
                    "mean_acceptance_probability": (
                        float(np.mean(coverages[method])) if coverages[method] else 0.0
                    ),
                    "null_rate": nulls[method] / repetitions,
                    "mc_failure_upper95": (
                        1.0 if failures[method] == repetitions else
                        float(beta.ppf(0.95, failures[method] + 1, repetitions - failures[method]))
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--target-risk", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--threshold-count", type=int, default=19)
    parser.add_argument("--output", type=Path, default=Path("g5_simulation.csv"))
    args = parser.parse_args()

    rows = simulate(
        seed=args.seed,
        repetitions=args.repetitions,
        sample_size=args.sample_size,
        target_risk=args.target_risk,
        delta=args.delta,
        threshold_count=args.threshold_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            row["scenario"],
            row["method"],
            f"unsafe={row['unsafe_return_rate']:.4f}",
            f"mean_acceptance={row['mean_acceptance_probability']:.4f}",
            f"null={row['null_rate']:.4f}",
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
