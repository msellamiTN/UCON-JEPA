"""Factual-only training and paired counterfactual evaluation data for G8.4.8.9.

The two APIs deliberately return different records. Factual training records
never contain disturbance values or alternative-action outcomes. Paired
counterfactual truth is a simulator-only evaluation object and must be stored
separately from model inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from continuous_security_context import (
    ACTIONS,
    exogenous_disturbance,
    initial_context,
    paired_action_branches,
    policy_verdict,
    transition,
)
from counterfactual_policy_world_model import keyed_exogenous_seed


def _sample_factual_action(root_seed: int, episode_id: str, family: str) -> tuple[str, float]:
    seed = keyed_exogenous_seed(root_seed, episode_id, "factual-action-randomization", 0)
    rng = np.random.default_rng(seed)
    actions = ACTIONS[family]
    index = int(rng.integers(0, len(actions)))
    return actions[index], 1.0 / len(actions)


def generate_factual_dataset(
    root_seed: int,
    episode_count: int,
    family: str,
    *,
    step_size: float = 0.1,
    interaction: float = 0.0,
    second_order_interaction: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate one factual transition per independent parent episode only."""
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    if not isinstance(episode_count, int) or isinstance(episode_count, bool) or episode_count < 1:
        raise ValueError("episode_count must be a positive integer")
    if family not in ACTIONS:
        raise ValueError(f"unknown family: {family}")
    records = []
    for index in range(episode_count):
        episode_id = f"factual-{root_seed}-{family}-{index:08d}"
        context_seed = keyed_exogenous_seed(root_seed, episode_id, "initial-context", 0)
        context = initial_context(context_seed)
        action, propensity = _sample_factual_action(root_seed, episode_id, family)
        disturbance = exogenous_disturbance(root_seed, episode_id, 0)
        next_context = transition(
            context, action, disturbance, step_size=step_size,
            interaction=interaction,
            second_order_interaction=second_order_interaction,
        )
        records.append({
            "episode_id": episode_id,
            "family": family,
            "context": context,
            "action": action,
            "action_propensity": propensity,
            "next_context": next_context,
            "ucon_verdict": policy_verdict(family, action, context),
        })
    return records


def generate_paired_counterfactual_test(
    root_seed: int,
    episode_count: int,
    family: str,
    *,
    step_size: float = 0.1,
    interaction: float = 0.0,
    second_order_interaction: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (factual model-input records, paired simulator-only test truth).

    The factual records have the same schema as generate_factual_dataset.
    The second return value contains alternative-action outcomes and must not
    be passed to training, checkpoint selection, calibration, or inference.
    """
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    if not isinstance(episode_count, int) or isinstance(episode_count, bool) or episode_count < 1:
        raise ValueError("episode_count must be a positive integer")
    if family not in ACTIONS:
        raise ValueError(f"unknown family: {family}")
    factual_records, paired_truth = [], []
    for index in range(episode_count):
        episode_id = f"paired-test-{root_seed}-{family}-{index:08d}"
        context_seed = keyed_exogenous_seed(root_seed, episode_id, "initial-context", 0)
        context = initial_context(context_seed)
        factual_action, propensity = _sample_factual_action(root_seed, episode_id, family)
        branches = paired_action_branches(
            context, family, root_seed=root_seed, parent_episode_id=episode_id,
            decision_step=0, step_size=step_size, interaction=interaction,
            second_order_interaction=second_order_interaction,
        )
        factual_branch = branches[factual_action]
        factual_records.append({
            "episode_id": episode_id,
            "family": family,
            "context": context,
            "action": factual_action,
            "action_propensity": propensity,
            "next_context": factual_branch["next_context"].copy(),
            "ucon_verdict": factual_branch["ucon_verdict"],
        })
        paired_truth.append({
            "episode_id": episode_id,
            "family": family,
            "action_support": tuple(ACTIONS[family]),
            "disturbance_key": f"{episode_id}|continuous-environment-disturbance|step=0",
            "branches": {
                action: {
                    "next_context": branch["next_context"].copy(),
                    "ucon_verdict": branch["ucon_verdict"],
                }
                for action, branch in branches.items()
            },
        })
    return factual_records, paired_truth
