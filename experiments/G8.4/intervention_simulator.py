"""Episode-level randomized simulation for predictive capability overlays.

The simulator keeps UCON authoritative. Predictive models only trigger a
predeclared restrictive overlay; they never supply an authorization verdict.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Callable

from intervention_runtime import (
    CapabilityOverlay,
    ProposalOutcome,
    UCONVerdict,
)
from intervention_task_manifests import (
    TASK_CAPABILITY_MANIFESTS,
    TASK_DIFFICULTY_LEVELS,
    sample_task_context,
    task_difficulty,
    task_goal_satisfied,
    validate_task_manifests,
)
from policy_parameter_dataset import (
    POLICY_ARCHETYPES,
    _parameterized_action,
    _policy_context,
)
from stateful_trajectory_dataset import (
    FAMILIES,
    _context,
    _initial_state,
    _online_gate_step,
    _policy_oracle,
)

ARMS = ("ucon_only", "reactive_overlay", "jepa_overlay", "supervised_overlay")
PREDICTIVE_ARM_MODEL = {
    "jepa_overlay": "jepa",
    "supervised_overlay": "supervised",
}
TASK_MANIFEST_HASHES = validate_task_manifests()
EPISODE_LENGTH = 8
PRIMARY_RHO = 0.85
OVERLAY_DURATION_DECISIONS = 2
PROTECTED_STATE_FIELDS = ("role_granted", "sensitive_read", "request_seen", "approved")


def _keyed_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _hash_uniform(*parts: Any) -> float:
    payload = json.dumps(list(parts), sort_keys=True, separators=(",", ":"))
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value / float(2**64)


def _should_trigger(score: float, threshold: dict[str, Any], episode_id: str, model: str) -> bool:
    cutoff = float(threshold["threshold"])
    if score > cutoff:
        return True
    if score < cutoff:
        return False
    tie_fraction = float(threshold.get("tie_fraction", 1.0))
    return _hash_uniform(
        int(threshold.get("tie_seed", 0)), episode_id, model, "threshold-tie"
    ) < tie_fraction


def fit_episode_max_threshold(
    episode_max_scores: list[tuple[str, float]],
    *,
    alert_fraction: float = 0.10,
    tie_seed: int = 884_808,
) -> dict[str, Any]:
    """Fit an episode-level max-score quantile with deterministic tie handling."""
    if not 0.0 < alert_fraction < 1.0:
        raise ValueError("alert_fraction must be in (0, 1)")
    if len(episode_max_scores) < 20:
        raise ValueError("at least 20 validation episodes are required")
    if len({episode_id for episode_id, _ in episode_max_scores}) != len(episode_max_scores):
        raise ValueError("validation episode IDs must be unique")
    scores = [float(score) for _, score in episode_max_scores]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("validation scores must be finite")
    target = max(1, int(math.ceil(alert_fraction * len(scores))))
    ordered = sorted(scores)
    threshold = float(ordered[len(ordered) - target])
    above = sum(score > threshold for score in scores)
    tied = sum(score == threshold for score in scores)
    tie_needed = max(0, target - above)
    tie_fraction = min(1.0, tie_needed / tied) if tied else 0.0
    return {
        "threshold": threshold,
        "alert_fraction": float(alert_fraction),
        "validation_episode_count": len(scores),
        "target_validation_alerts": target,
        "above_threshold_count": above,
        "at_threshold_count": tied,
        "tie_fraction": float(tie_fraction),
        "tie_seed": int(tie_seed),
        "threshold_unit": "episode maximum across eligible decision points",
    }


def _task_difficulty_levels() -> dict[str, tuple[str, ...]]:
    return {family: tuple(levels) for family, levels in TASK_DIFFICULTY_LEVELS.items()}


def make_blocked_episode_specs(
    seeds: list[int],
    episodes_per_arm_per_stratum: int,
    *,
    assignment_seed: int,
    rho: float = PRIMARY_RHO,
) -> list[dict[str, Any]]:
    """Create feasible task episodes and balanced four-arm blocked assignments."""
    if episodes_per_arm_per_stratum < 1:
        raise ValueError("episodes_per_arm_per_stratum must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    if not math.isfinite(float(rho)) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be finite and in [0, 1]")
    specs: list[dict[str, Any]] = []
    for seed in sorted(set(int(value) for value in seeds)):
        for archetype in POLICY_ARCHETYPES:
            for family in FAMILIES:
                for difficulty in TASK_DIFFICULTY_LEVELS[family]:
                    block_specs = []
                    for index in range(4 * episodes_per_arm_per_stratum):
                        episode_id = (
                            f"g8-4-8-8-{seed}-{archetype}-{family}-"
                            f"{difficulty}-{index:05d}"
                        )
                        context_rng = random.Random(
                            _keyed_seed(seed, episode_id, "initial-context")
                        )
                        context = sample_task_context(
                            context_rng, family, _context, difficulty=difficulty
                        )
                        if task_difficulty(family, context) != difficulty:
                            raise AssertionError("sampled context escaped its task-difficulty block")
                        block_specs.append({
                            "episode_id": episode_id,
                            "seed": seed,
                            "archetype": archetype,
                            "family": family,
                            "rho": float(rho),
                            "task_difficulty": difficulty,
                            "task_manifest_sha256": TASK_MANIFEST_HASHES[family],
                            "initial_context": context,
                            "stratum": (
                                f"seed={seed}|archetype={archetype}|family={family}|"
                                f"difficulty={difficulty}"
                            ),
                        })
                    arms = list(ARMS) * episodes_per_arm_per_stratum
                    assignment_rng = random.Random(_keyed_seed(
                        assignment_seed, seed, archetype, family, difficulty, "arm-assignment"
                    ))
                    assignment_rng.shuffle(arms)
                    for spec, arm in zip(block_specs, arms):
                        specs.append({
                            **spec,
                            "arm": arm,
                            "assignment_seed": int(assignment_seed),
                        })
    _assert_balanced_assignment(specs, episodes_per_arm_per_stratum)
    return specs


def _assert_balanced_assignment(
    specs: list[dict[str, Any]], expected_per_arm: int,
) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ids = set()
    for spec in specs:
        if spec["episode_id"] in ids:
            raise AssertionError("duplicate randomized episode ID")
        ids.add(spec["episode_id"])
        counts[spec["stratum"]][spec["arm"]] += 1
    for stratum, arm_counts in counts.items():
        if set(arm_counts) != set(ARMS):
            raise AssertionError(f"missing randomized arm in {stratum}")
        if any(arm_counts[arm] != expected_per_arm for arm in ARMS):
            raise AssertionError(f"unbalanced arm allocation in {stratum}: {dict(arm_counts)}")


def run_episode(
    spec: dict[str, Any],
    *,
    score_functions: dict[str, Callable[[Any, int, int], float]] | None = None,
    thresholds: dict[str, dict[str, Any]] | None = None,
    horizon: int = 1,
) -> dict[str, Any]:
    """Run one dynamically evolving episode under one randomized strategy arm."""
    score_functions = score_functions or {}
    thresholds = thresholds or {}
    episode_id = str(spec["episode_id"])
    seed = int(spec["seed"])
    archetype = str(spec["archetype"])
    family = str(spec["family"])
    rho = float(spec["rho"])
    arm = str(spec["arm"])
    difficulty = str(spec["task_difficulty"])
    if arm not in ARMS:
        raise ValueError(f"unknown intervention arm: {arm}")
    for model_name, threshold in thresholds.items():
        if not math.isfinite(float(threshold["threshold"])):
            raise ValueError(f"non-finite threshold for {model_name}")
        tie_fraction = float(threshold.get("tie_fraction", 1.0))
        if not 0.0 <= tie_fraction <= 1.0:
            raise ValueError(f"invalid tie fraction for {model_name}")
    if archetype not in POLICY_ARCHETYPES or family not in FAMILIES:
        raise ValueError("unknown policy archetype or task family")
    if difficulty not in TASK_DIFFICULTY_LEVELS[family]:
        raise ValueError("unknown task difficulty")
    if task_difficulty(family, spec["initial_context"]) != difficulty:
        raise ValueError("initial context does not match its randomized difficulty stratum")

    manifest = TASK_CAPABILITY_MANIFESTS[family]
    overlay = CapabilityOverlay(manifest, duration_decisions=OVERLAY_DURATION_DECISIONS)
    state = _initial_state(dict(spec["initial_context"]))
    last_denied_action: str | None = None
    pending_reactive_trigger = False
    trigger_step: int | None = None
    trigger_reason: str | None = None
    predictor_failed = False
    predictor_failure_reason: str | None = None
    denied_count = 0
    suppressed_count = 0
    replanned_count = 0
    active_decisions = 0
    state_violation_count = 0
    unauthorized_execution_count = 0
    oracle_call_count = 0
    base_candidate_denied_shadow_count = 0
    records: list[dict[str, Any]] = []

    for step in range(EPISODE_LENGTH):
        risk_score: float | None = None
        model_name = PREDICTIVE_ARM_MODEL.get(arm)
        if arm == "reactive_overlay" and pending_reactive_trigger and not overlay.triggered:
            if overlay.trigger(f"reactive_after_deny_at_step_{step - 1}"):
                trigger_step = step
                trigger_reason = overlay.trigger_reason
            pending_reactive_trigger = False

        if model_name is not None and not overlay.triggered and not predictor_failed:
            if model_name not in score_functions or model_name not in thresholds:
                predictor_failed = True
                predictor_failure_reason = f"missing_{model_name}_score_or_threshold"
            else:
                current_context = _policy_context(state, last_denied_action)
                try:
                    risk_score = float(score_functions[model_name](
                        current_context, int(horizon),
                        _keyed_seed(seed, episode_id, step, model_name, "predictive-score"),
                    ))
                    if not math.isfinite(risk_score):
                        raise ValueError("non-finite predictive score")
                    if _should_trigger(risk_score, thresholds[model_name], episode_id, model_name):
                        if overlay.trigger(f"{model_name}_risk_threshold"):
                            trigger_step = step
                            trigger_reason = overlay.trigger_reason
                except Exception as exc:  # predictive failure must never weaken UCON
                    predictor_failed = True
                    predictor_failure_reason = f"{type(exc).__name__}: {exc}"
                    risk_score = None

        available = overlay.begin_decision()
        if not available:
            raise AssertionError("task manifest allowed an empty capability set")
        base_rng = random.Random(_keyed_seed(seed, episode_id, step, "base-policy"))
        base_candidate = _parameterized_action(
            archetype, rho, state, last_denied_action, step, base_rng
        )
        replanned = base_candidate not in available
        if replanned:
            fallback_rng = random.Random(_keyed_seed(seed, episode_id, step, "overlay-replan"))
            submitted_action = sorted(available)[fallback_rng.randrange(len(available))]
            replanned_count += 1
        else:
            submitted_action = base_candidate

        # These shadow labels are recorded only after action selection and never
        # influence the capability-aware replan or the UCON/overlay decision.
        base_candidate_allowed, base_candidate_reason = _policy_oracle(state, base_candidate)
        if base_candidate_allowed:
            base_candidate_reason = None
        else:
            base_candidate_denied_shadow_count += 1

        oracle_allowed, oracle_reason = _policy_oracle(state, submitted_action)
        if oracle_allowed:
            oracle_reason = None
        oracle_call_count += 1
        ucon_verdict = UCONVerdict.PERMIT if oracle_allowed else UCONVerdict.DENY
        decision = overlay.evaluate_proposal(
            submitted_action, lambda _proposal: ucon_verdict
        )
        if decision.ucon_verdict is not ucon_verdict:
            raise AssertionError("overlay controller changed the current UCON verdict")
        overlay_active = bool(decision.overlay_active)
        if overlay_active:
            active_decisions += 1

        before = {key: state[key] for key in PROTECTED_STATE_FIELDS}
        gate_reason = None
        if decision.outcome is ProposalOutcome.READY:
            after, gate_allowed, gate_reason = _online_gate_step(state, submitted_action)
            if not gate_allowed or gate_reason != (oracle_reason or None):
                raise AssertionError("online gate disagrees with UCON oracle on PERMIT")
            if decision.ucon_verdict is not UCONVerdict.PERMIT or not decision.may_execute:
                raise AssertionError("proposal executed without UCON PERMIT and overlay allow")
            if decision.ucon_verdict is UCONVerdict.DENY:
                unauthorized_execution_count += 1
            state = after
            last_denied_action = None
            pending_reactive_trigger = False
        elif decision.outcome is ProposalOutcome.DENIED_BY_UCON:
            after, gate_allowed, gate_reason = _online_gate_step(state, submitted_action)
            if gate_allowed or gate_reason != oracle_reason:
                raise AssertionError("online gate disagrees with UCON oracle on DENY")
            if any(after[key] != before[key] for key in PROTECTED_STATE_FIELDS):
                state_violation_count += 1
                raise AssertionError("DENY mutated protected workflow state")
            state = after
            last_denied_action = submitted_action
            denied_count += 1
            pending_reactive_trigger = arm == "reactive_overlay"
        elif decision.outcome is ProposalOutcome.SUPPRESSED_BY_OVERLAY:
            if decision.ucon_verdict is not UCONVerdict.PERMIT or decision.may_execute:
                raise AssertionError("overlay suppression must not alter UCON or permit execution")
            suppressed_count += 1
            last_denied_action = None
            pending_reactive_trigger = False
        else:
            raise AssertionError(f"unexpected fail-closed outcome: {decision.outcome}")

        overlay.end_decision()
        records.append({
            "step": step,
            "arm": arm,
            "risk_model": model_name,
            "risk_score": risk_score,
            "risk_threshold": (
                float(thresholds[model_name]["threshold"])
                if model_name is not None and model_name in thresholds else None
            ),
            "base_candidate": base_candidate,
            "base_candidate_ucon_denied_shadow": bool(not base_candidate_allowed),
            "base_candidate_denial_reason_shadow": base_candidate_reason or "",
            "submitted_action": submitted_action,
            "replanned_due_to_overlay": bool(replanned),
            "available_capabilities": sorted(available),
            "overlay_active": overlay_active,
            "overlay_remaining_after_decision": overlay.remaining_decisions,
            "ucon_verdict": decision.ucon_verdict.value if decision.ucon_verdict else None,
            "proposal_outcome": decision.outcome.value,
            "ucon_denial_reason": oracle_reason or "",
            "gate_denial_reason": gate_reason or "",
            "predictor_failed": predictor_failed,
            "triggered_at_step": trigger_step == step,
        })

    task_complete = task_goal_satisfied(family, state)
    overlay_remaining = overlay.remaining_decisions
    if unauthorized_execution_count != 0:
        raise AssertionError("UCON-DENY operation was executed")
    if oracle_call_count != EPISODE_LENGTH:
        raise AssertionError("every submitted proposal must be checked by UCON")
    return {
        "episode_id": episode_id,
        "seed": seed,
        "archetype": archetype,
        "family": family,
        "rho": rho,
        "task_difficulty": difficulty,
        "task_id": manifest.task_id,
        "task_manifest_sha256": TASK_MANIFEST_HASHES[family],
        "arm": arm,
        "assignment_seed": int(spec["assignment_seed"]),
        "stratum": str(spec["stratum"]),
        "denied_count": int(denied_count),
        "denied_rate": float(denied_count / EPISODE_LENGTH),
        "any_deny": bool(denied_count > 0),
        "task_complete": bool(task_complete),
        "safe_task_complete": bool(task_complete and denied_count == 0),
        "recovered_task_complete": bool(task_complete and denied_count > 0),
        "overlay_triggered": bool(overlay.triggered),
        "overlay_trigger_step": trigger_step,
        "overlay_trigger_reason": trigger_reason,
        "overlay_active_decisions": int(active_decisions),
        "overlay_remaining_at_episode_end": int(overlay_remaining),
        "overlay_suppressed_count": int(suppressed_count),
        "replanned_count": int(replanned_count),
        "base_candidate_denied_shadow_count": int(base_candidate_denied_shadow_count),
        "predictor_failed": bool(predictor_failed),
        "predictor_failure_reason": predictor_failure_reason,
        "ucon_oracle_call_count": int(oracle_call_count),
        "unauthorized_execution_count": int(unauthorized_execution_count),
        "state_violation_count": int(state_violation_count),
        "records": records,
    }
