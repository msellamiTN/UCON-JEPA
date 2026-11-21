"""G8.4.8.8 episode-randomized adaptive-warning simulator.

The simulator is deliberately separate from the frozen G8.4.8.7 generators.
Predictors emit advisory scores only; every proposed action is independently
checked by both the formal oracle and the online enforcement transition.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from policy_parameter_dataset import (
    FAMILY_ACTIONS, POLICY_ARCHETYPES, TRAIN_RHO, SHIFT_RHO, REVERSE_RHO,
    _parameterized_action, _policy_context,
)
from stateful_trajectory_dataset import (
    FAMILIES, _context, _initial_state, _online_gate_step, _policy_oracle,
)

ARMS = ("ucon_only", "jepa_advisory", "supervised_advisory", "random_advisory")
RESPONSE_PROPENSITIES = (0.0, 0.5, 1.0)
PRIMARY_HORIZON = 1
SECONDARY_HORIZONS = (2, 4)
EXPLORATORY_HORIZON = 8
EPISODE_LENGTH = 8
RANDOM_WARNING_RATE = 0.10


def _keyed_seed(seed: int, episode_id: str, step: int, stream: str) -> int:
    """Stable independent stream seed; independent of draw counts elsewhere."""
    payload = f"g8.4.8.8-v1|{int(seed)}|{episode_id}|{int(step)}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _keyed_rng(seed: int, episode_id: str, step: int, stream: str) -> random.Random:
    return random.Random(_keyed_seed(seed, episode_id, step, stream))


def _tie_uniform(seed: int, episode_id: str, step: int, stream: str) -> float:
    return _keyed_seed(seed, episode_id, step, stream) / float(2**64)


def fit_alert_threshold(
    scores: np.ndarray,
    episode_ids: np.ndarray,
    steps: np.ndarray,
    alert_rate: float = 0.10,
) -> dict[str, float | int | str]:
    """Fit a deterministic score/tie threshold using calibration scores only."""
    scores = np.asarray(scores, dtype=np.float64)
    episode_ids = np.asarray(episode_ids).astype(str)
    steps = np.asarray(steps, dtype=np.int64)
    if scores.ndim != 1 or not (len(scores) == len(episode_ids) == len(steps)):
        raise ValueError("scores, episode_ids and steps must be aligned vectors")
    if len(scores) < 20 or not np.isfinite(scores).all():
        raise ValueError("at least 20 finite calibration scores are required")
    if not 0.0 < alert_rate < 1.0:
        raise ValueError("alert_rate must be strictly between zero and one")
    n_alert = max(1, int(math.ceil(alert_rate * len(scores))))
    tie = np.asarray([
        _tie_uniform(0, episode_ids[i], int(steps[i]), "threshold-tie")
        for i in range(len(scores))
    ])
    order = np.lexsort((tie, -scores))
    cutoff_index = min(n_alert - 1, len(order) - 1)
    cutoff_score = float(scores[order[cutoff_index]])
    above = scores > cutoff_score
    equal = scores == cutoff_score
    remaining = max(0, n_alert - int(above.sum()))
    tie_cutoff = float(np.sort(tie[equal])[remaining - 1]) if remaining else -1.0
    return {
        "cutoff_score": cutoff_score,
        "tie_cutoff": tie_cutoff,
        "calibration_n": int(len(scores)),
        "target_alert_rate": float(alert_rate),
        "rule": "score>cutoff or (score==cutoff and keyed_tie<=tie_cutoff)",
    }


def threshold_fires(
    score: float, threshold: dict[str, Any], seed: int,
    episode_id: str, step: int,
) -> bool:
    if not math.isfinite(float(score)):
        raise ValueError("predictive risk score must be finite")
    cutoff = float(threshold["cutoff_score"])
    if score > cutoff:
        return True
    if score < cutoff:
        return False
    # Calibration and deployment use the same seed-independent tie key.
    tie = _tie_uniform(0, episode_id, step, "threshold-tie")
    return tie <= float(threshold["tie_cutoff"])


def make_randomized_assignment(
    *,
    seeds: tuple[int, ...] | list[int],
    archetypes: tuple[str, ...] | list[str] = POLICY_ARCHETYPES,
    families: tuple[str, ...] | list[str] = FAMILIES,
    rhos: tuple[float, ...] | list[float] = (SHIFT_RHO,),
    qs: tuple[float, ...] | list[float] = RESPONSE_PROPENSITIES,
    episodes_per_stratum: int = 40,
    assignment_seed: int = 884808,
    horizons: tuple[int, ...] | list[int] = (1, 2, 4, 8),
) -> list[dict[str, Any]]:
    """Balanced episode-level assignment, randomized independently per stratum."""
    if episodes_per_stratum < len(ARMS) or episodes_per_stratum % len(ARMS):
        raise ValueError("episodes_per_stratum must be >=4 and divisible by four")
    if not set(archetypes).issubset(POLICY_ARCHETYPES):
        raise ValueError("unknown behavior archetype")
    if not set(families).issubset(FAMILIES):
        raise ValueError("unknown resource family")
    if any(not 0.0 <= float(q) <= 1.0 for q in qs):
        raise ValueError("q must be in [0,1]")
    if any(int(h) not in (1, 2, 4, 8) for h in horizons):
        raise ValueError("supported horizons are 1, 2, 4, and 8")
    output: list[dict[str, Any]] = []
    strata = itertools.product(seeds, archetypes, families, rhos, qs, horizons)
    for seed, archetype, family, rho, q, horizon in strata:
        stratum = f"{seed}|{archetype}|{family}|{float(rho):.8f}|{float(q):.8f}|{int(horizon)}"
        ids = [f"g8848-{hashlib.sha256(f'{stratum}|{i}'.encode()).hexdigest()[:20]}"
               for i in range(episodes_per_stratum)]
        rng = random.Random(_keyed_seed(assignment_seed, stratum, 0, "arm-assignment"))
        shuffled_arms = [arm for arm in ARMS for _ in range(episodes_per_stratum // len(ARMS))]
        rng.shuffle(shuffled_arms)
        for episode_id, arm in zip(ids, shuffled_arms):
            output.append({
                "episode_id": episode_id, "seed": int(seed),
                "archetype": archetype, "family": family, "rho": float(rho),
                "q": float(q), "horizon": int(horizon), "arm": arm,
                "stratum": stratum,
            })
    return output


def _task_complete(family: str, state: dict[str, Any]) -> bool:
    # Conservative legitimate-workflow milestones, not an authorization label.
    if family == "iam":
        return bool(state["role_granted"])
    if family == "data":
        return bool(state["sensitive_read"])
    if family == "sod":
        return bool(state["approved"])
    raise ValueError(f"unknown family: {family}")


def run_episode(
    spec: dict[str, Any],
    *,
    score_functions: dict[str, Callable[[np.ndarray, int, int], float]] | None = None,
    thresholds: dict[str, dict[int, dict[int, dict[str, Any]]]] | None = None,
    random_warning_rate: float = RANDOM_WARNING_RATE,
) -> dict[str, Any]:
    """Execute one adaptive episode; score callback receives context, H, seed."""
    arm = str(spec["arm"])
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    horizon = int(spec["horizon"])
    if horizon not in (1, 2, 4, 8):
        raise ValueError("unsupported horizon")
    q = float(spec["q"])
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    seed, episode_id = int(spec["seed"]), str(spec["episode_id"])
    family, archetype, rho = str(spec["family"]), str(spec["archetype"]), float(spec["rho"])
    if family not in FAMILIES or archetype not in POLICY_ARCHETYPES:
        raise ValueError("unknown family or archetype")
    if not 0.0 <= random_warning_rate <= 1.0:
        raise ValueError("random_warning_rate must be in [0,1]")
    if arm in ("jepa_advisory", "supervised_advisory"):
        model_name = "jepa" if arm == "jepa_advisory" else "supervised"
        if score_functions is None or model_name not in score_functions:
            raise ValueError(f"missing score function for {model_name}")
        if thresholds is None or model_name not in thresholds:
            raise ValueError(f"missing thresholds for {model_name}")
    else:
        model_name = ""

    context_rng = _keyed_rng(seed, episode_id, 0, "initial-context")
    state = _initial_state(_context(context_rng, family))
    last_denied_action: str | None = None
    records: list[dict[str, Any]] = []
    for step in range(EPISODE_LENGTH):
        context = _policy_context(state, last_denied_action)
        policy_rng = _keyed_rng(seed, episode_id, step, "base-policy")
        original = _parameterized_action(archetype, rho, state, last_denied_action, step, policy_rng)

        warning = False
        risk_score: float | None = None
        eligible = step <= EPISODE_LENGTH - horizon
        if eligible and arm in ("jepa_advisory", "supervised_advisory"):
            risk_score = float(score_functions[model_name](
                context.copy(), horizon,
                _keyed_seed(seed, episode_id, step, f"{model_name}-rollout"),
            ))
            by_horizon = thresholds[model_name].get(horizon, {})
            if step not in by_horizon:
                raise ValueError(f"missing frozen threshold: {model_name}/H={horizon}/step={step}")
            warning = threshold_fires(risk_score, by_horizon[step], seed, episode_id, step)
        elif eligible and arm == "random_advisory":
            warning = _keyed_rng(seed, episode_id, step, "random-warning").random() < random_warning_rate

        proposed = original
        revised = False
        if warning:
            response_coin = _keyed_rng(seed, episode_id, step, "response-coin").random()
            if response_coin < q:
                alternatives = [a for a in FAMILY_ACTIONS[family] if a != original]
                proposed = _keyed_rng(seed, episode_id, step, "alternative-action").choice(alternatives)
                revised = True

        oracle_allowed, oracle_reason = _policy_oracle(state, proposed)
        if oracle_allowed:
            oracle_reason = None
        after, gate_allowed, gate_reason = _online_gate_step(state, proposed)
        if oracle_allowed != gate_allowed or oracle_reason != gate_reason:
            raise AssertionError(
                f"oracle/gate disagreement at {episode_id}/{step}: "
                f"{(oracle_allowed, oracle_reason)} != {(gate_allowed, gate_reason)}"
            )
        # Explicitly assert that the gate's protected-state mutation follows
        # the oracle verdict; a DENY may update attempt history only.
        if not gate_allowed:
            for key in ("role_granted", "sensitive_read", "request_seen", "approved"):
                if after[key] != state[key]:
                    raise AssertionError(f"DENY mutated protected state field {key}")
        records.append({
            "episode_id": episode_id, "step": step, "arm": arm,
            "original_action": original, "proposed_action": proposed,
            "warning": bool(warning), "revised": bool(revised),
            "risk_score": risk_score, "ucon_allowed": bool(gate_allowed),
            "denial_reason": gate_reason or "",
            "task_complete": _task_complete(family, after),
            "state_violation": False,  # DENY is not a state violation; gate/oracle disagreement raises above.
        })
        state = after
        last_denied_action = None if gate_allowed else proposed

    denied = sum(not row["ucon_allowed"] for row in records)
    warnings = sum(row["warning"] for row in records)
    revisions = sum(row["revised"] for row in records)
    return {
        "episode_id": episode_id, "seed": seed, "archetype": archetype,
        "family": family, "rho": rho, "q": q, "horizon": horizon, "arm": arm,
        "denied_count": int(denied), "denied_rate": denied / EPISODE_LENGTH,
        "any_deny": bool(denied > 0),
        "task_complete": bool(_task_complete(family, state)),
        "warning_count": int(warnings), "warning_rate": warnings / EPISODE_LENGTH,
        "revision_count": int(revisions), "state_violation_count": int(
            sum(row["state_violation"] for row in records)
        ),
        "records": records,
    }


def assert_q0_negative_control(
    spec: dict[str, Any],
    *,
    score_functions: dict[str, Callable[[np.ndarray, int, int], float]],
    thresholds: dict[str, dict[int, dict[int, dict[str, Any]]]],
) -> None:
    """Matched q=0 replay: advisory warnings must not alter proposed actions."""
    if float(spec["q"]) != 0.0:
        raise ValueError("negative-control assertion requires q=0")
    base = dict(spec, arm="ucon_only")
    for arm, model in (("jepa_advisory", "jepa"), ("supervised_advisory", "supervised")):
        warned = run_episode(
            dict(spec, arm=arm), score_functions=score_functions, thresholds=thresholds
        )
        control = run_episode(base)
        a = [r["proposed_action"] for r in warned["records"]]
        b = [r["proposed_action"] for r in control["records"]]
        if a != b:
            raise AssertionError(f"q=0 negative control failed for {model}")
        if warned["denied_count"] != control["denied_count"]:
            raise AssertionError(f"q=0 denied burden changed for {model}")


def summarize_episodes(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Episode-level descriptive summaries; no transition-level pseudo-replication."""
    if not results:
        raise ValueError("no episode results")
    groups: dict[tuple[str, float, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(row["arm"], float(row["rho"]), float(row["q"]), int(row["horizon"]))].append(row)
    out = {}
    for key, rows in sorted(groups.items()):
        out["|".join(map(str, key))] = {
            "episodes": len(rows),
            "mean_denied_rate": float(np.mean([r["denied_rate"] for r in rows])),
            "any_deny_rate": float(np.mean([r["any_deny"] for r in rows])),
            "task_completion_rate": float(np.mean([r["task_complete"] for r in rows])),
            "mean_warning_rate": float(np.mean([r["warning_rate"] for r in rows])),
            "mean_revision_count": float(np.mean([r["revision_count"] for r in rows])),
            "state_violation_count": int(sum(r["state_violation_count"] for r in rows)),
        }
    return out
