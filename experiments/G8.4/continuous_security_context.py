"""Controlled continuous security-context dynamics for G8.4.8.9.

This is a separately versioned synthetic benchmark extension, not a claim that
the existing categorical UCON state is continuous or that these variables
match real enterprise distributions. It supplies interpretable continuous
readouts plus a deterministic threshold policy boundary.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from counterfactual_policy_world_model import keyed_exogenous_seed

FAMILIES = ("iam", "data", "sod")
ACTIONS = {
    "iam": ("grant_role", "assume_admin", "iam_read"),
    "data": ("read_sensitive", "export_sensitive", "data_list"),
    "sod": ("request", "approve", "pay"),
}
# Coordinates: cumulative exposure, resource pressure, anomaly intensity.
ACTION_EFFECTS = {
    "grant_role": np.array([0.12, 0.08, 0.02]),
    "assume_admin": np.array([0.55, 0.18, 0.42]),
    "iam_read": np.array([0.04, 0.07, 0.01]),
    "read_sensitive": np.array([0.18, 0.16, 0.05]),
    "export_sensitive": np.array([0.82, 0.30, 0.58]),
    "data_list": np.array([0.03, 0.04, 0.01]),
    "request": np.array([0.08, 0.09, 0.02]),
    "approve": np.array([0.16, 0.12, 0.03]),
    "pay": np.array([0.72, 0.38, 0.35]),
}
DISTURBANCE_SCALE = np.array([0.18, 0.22, 0.15])
HIGH_RISK_ACTIONS = frozenset({"assume_admin", "export_sensitive", "pay"})
DEFAULT_EXPOSURE_THRESHOLD = 1.25


def _vector(name: str, value: Any, size: int = 3) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result


def initial_context(seed: int) -> np.ndarray:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    return np.array([rng.uniform(0.15, 0.55), rng.uniform(0.1, 0.6), rng.uniform(0.05, 0.4)])


def exogenous_disturbance(
    root_seed: int, parent_episode_id: str, decision_step: int, draw_index: int = 0
) -> np.ndarray:
    seed = keyed_exogenous_seed(
        root_seed, parent_episode_id, "continuous-environment-disturbance", decision_step, draw_index
    )
    return np.random.default_rng(seed).standard_normal(3)


def transition(
    state: Any,
    action: str,
    disturbance: Any,
    *,
    step_size: float = 0.1,
    interaction: float = 0.0,
    second_order_interaction: float = 0.0,
) -> np.ndarray:
    """One-step numeric state transition.

    interaction=0 and second_order_interaction=0 is the exact additive
    separate-channel positive control. A nonzero second-order interaction
    creates an O(h^2) action/disturbance interaction. A nonzero first-order
    interaction creates a command-scaled disturbance negative control.
    """
    x = _vector("state", state)
    eps = _vector("disturbance", disturbance)
    if action not in ACTION_EFFECTS:
        raise ValueError(f"unsupported action: {action}")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if not np.isfinite(interaction) or not np.isfinite(second_order_interaction):
        raise ValueError("interaction coefficients must be finite")
    action_effect = ACTION_EFFECTS[action]
    disturbance_effect = DISTURBANCE_SCALE * eps
    first_order = x + step_size * (action_effect + disturbance_effect)
    first_order += step_size * float(interaction) * action_effect * eps[0]
    second_order = (
        0.5 * step_size**2 * float(second_order_interaction) * action_effect * eps[0]
    )
    return first_order + second_order


def policy_verdict(
    family: str,
    action: str,
    next_context: Any,
    *,
    exposure_threshold: float = DEFAULT_EXPOSURE_THRESHOLD,
    base_policy_allowed: bool = True,
) -> str:
    """Deterministic synthetic policy extension; verdict is never learned."""
    if family not in FAMILIES or action not in ACTIONS[family]:
        raise ValueError("action must belong to the declared task family")
    x = _vector("next_context", next_context)
    if not np.isfinite(exposure_threshold):
        raise ValueError("exposure_threshold must be finite")
    if not base_policy_allowed:
        return "DENY"
    if action in HIGH_RISK_ACTIONS and x[0] >= float(exposure_threshold):
        return "DENY"
    return "PERMIT"


def paired_action_branches(
    state: Any,
    family: str,
    actions: tuple[str, ...] | None = None,
    *,
    root_seed: int,
    parent_episode_id: str,
    decision_step: int,
    step_size: float = 0.1,
    interaction: float = 0.0,
    second_order_interaction: float = 0.0,
    exposure_threshold: float = DEFAULT_EXPOSURE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Evaluate alternative actions under one shared exogenous disturbance."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    selected = ACTIONS[family] if actions is None else tuple(actions)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("actions must be a non-empty unique tuple")
    if any(action not in ACTIONS[family] for action in selected):
        raise ValueError("all counterfactual actions must be supported by the task family")
    eps = exogenous_disturbance(root_seed, parent_episode_id, decision_step)
    branches = {}
    for action in selected:
        next_state = transition(
            state, action, eps, step_size=step_size,
            interaction=interaction,
            second_order_interaction=second_order_interaction,
        )
        branches[action] = {
            "next_context": next_state,
            "ucon_verdict": policy_verdict(
                family, action, x, exposure_threshold=exposure_threshold
            ),
            "exogenous_disturbance": eps.copy(),
        }
    return branches


def factual_transition(
    state: Any,
    family: str,
    action: str,
    *,
    root_seed: int,
    parent_episode_id: str,
    decision_step: int,
    step_size: float = 0.1,
    interaction: float = 0.0,
    second_order_interaction: float = 0.0,
    exposure_threshold: float = DEFAULT_EXPOSURE_THRESHOLD,
) -> dict[str, Any]:
    """One factual transition; returns the observed outcome, not alternative labels."""
    branches = paired_action_branches(
        state, family, (action,), root_seed=root_seed,
        parent_episode_id=parent_episode_id, decision_step=decision_step,
        step_size=step_size, interaction=interaction,
        second_order_interaction=second_order_interaction,
        exposure_threshold=exposure_threshold,
    )
    branch = branches[action]
    return {
        "next_context": branch["next_context"],
        "ucon_verdict": branch["ucon_verdict"],
    }
