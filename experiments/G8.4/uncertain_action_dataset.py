"""G8.4.8.2 synthetic agent-policy trajectories and leakage-safe splits.

This module extends the fixed-length G8.4.7 trajectory semantics with five
explicit behavior policies and four episode-disjoint partitions. It is a
synthetic benchmark generator, not a model of real autonomous agents.

Only observable state and denial feedback are exposed to behavior policies.
Policy identity, episode identifiers, split labels, action names and oracle
reasons are metadata/targets and are never included in the context feature
matrix. The future action is stored as a target for action forecasting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from stateful_trajectory_dataset import (
    ACTION_VOCAB,
    FAMILIES,
    _context,
    _initial_state,
    _online_gate_step,
    _policy_oracle,
    _state_features,
)

POLICY_VERSION = "ucon-jeepa-synthetic-v3-g8.4.8.2"
AGENT_POLICIES = (
    "goal_directed",
    "uniform",
    "retrying_adversary",
    "prerequisite_blind_planner",
    "denial_adaptive_explorer",
)
PARTITIONS = ("train", "validation", "calibration", "test")
CONTEXT_DIM = len(_state_features(_initial_state({
    "family": "iam",
    "security_admin": False,
    "scope_match": False,
    "clearance": 0,
    "classification": 1,
    "export_approved": False,
    "recipient_owner": False,
    "approver_independent": False,
    "payer_independent": False,
}))) + len(ACTION_VOCAB)
ACTION_DIM = len(ACTION_VOCAB)
SECURITY_DIM = 4

FAMILY_ACTIONS = {
    "iam": ("grant_role", "assume_admin", "iam_read"),
    "data": ("read_sensitive", "export_sensitive", "data_list"),
    "sod": ("request", "approve", "pay"),
}


def _policy_context(state: dict, last_denied_action: str | None) -> np.ndarray:
    """Observable policy context: state features plus explicit denial feedback."""
    feedback = np.zeros(ACTION_DIM, dtype=np.float64)
    if last_denied_action is not None:
        feedback[ACTION_VOCAB.index(last_denied_action)] = 1.0
    return np.concatenate((_state_features(state), feedback))


def _goal_directed_action(state: dict) -> str:
    family, ctx = state["context"]["family"], state["context"]
    if family == "iam":
        if not state["role_granted"]:
            return "grant_role" if ctx["security_admin"] else "iam_read"
        return "assume_admin" if ctx["scope_match"] else "iam_read"
    if family == "data":
        if not state["sensitive_read"]:
            return "read_sensitive" if ctx["clearance"] >= ctx["classification"] else "data_list"
        return "export_sensitive" if ctx["export_approved"] and ctx["recipient_owner"] else "data_list"
    if family == "sod":
        if not state["request_seen"]:
            return "request"
        if not state["approved"]:
            return "approve" if ctx["approver_independent"] else "request"
        return "pay" if ctx["payer_independent"] else "request"
    raise ValueError(f"unknown family: {family}")


def _agent_action(
    policy: str,
    state: dict,
    last_denied_action: str | None,
    step: int,
    rng: random.Random,
) -> str:
    """Select an action using only the policy's declared observable inputs."""
    family = state["context"]["family"]
    actions = FAMILY_ACTIONS[family]

    if policy == "goal_directed":
        return _goal_directed_action(state)
    if policy == "uniform":
        return rng.choice(actions)
    if policy == "retrying_adversary":
        if last_denied_action is not None:
            return last_denied_action
        return {
            "iam": "assume_admin",
            "data": "export_sensitive",
            "sod": "pay",
        }[family]
    if policy == "prerequisite_blind_planner":
        # Fixed workflow order. It deliberately ignores governance state,
        # denial feedback, and whether a prerequisite has been satisfied.
        return actions[step % len(actions)]
    if policy == "denial_adaptive_explorer":
        # On denial, move one position forward from the denied action in the
        # family action cycle. Without denial feedback, choose uniformly.
        if last_denied_action is not None:
            return actions[(actions.index(last_denied_action) + 1) % len(actions)]
        return rng.choice(actions)
    raise ValueError(f"unknown agent policy: {policy}")


def _episode(seed: int, family: str, policy: str, length: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ctx = _context(rng, family)
    state = _initial_state(ctx)
    last_denied_action: str | None = None
    rows: list[dict[str, Any]] = []

    for step in range(length):
        # The behavior policy sees state/feedback only; no future target exists.
        action_name = _agent_action(policy, state, last_denied_action, step, rng)
        context = _policy_context(state, last_denied_action)
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[ACTION_VOCAB.index(action_name)] = 1.0

        oracle_allowed, oracle_reason = _policy_oracle(state, action_name)
        if oracle_allowed:
            oracle_reason = None
        after, gate_allowed, gate_reason = _online_gate_step(state, action_name)
        if oracle_allowed != gate_allowed or oracle_reason != gate_reason:
            raise AssertionError(
                f"oracle/gate disagreement family={family} policy={policy} "
                f"step={step} action={action_name}: "
                f"oracle={(oracle_allowed, oracle_reason)} "
                f"gate={(gate_allowed, gate_reason)}"
            )

        next_denied = None if oracle_allowed else action_name
        security = np.asarray(
            [
                float(not oracle_allowed),
                float(after["approved"]),
                float(after["sensitive_read"]),
                float(after["role_granted"]),
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "context": context,
                "action": action,
                "future": _policy_context(after, next_denied),
                "security": security,
                "episode_id": f"{policy}-{family}-{seed}",
                "step": step,
                "family": family,
                "agent_policy": policy,
                "action_name": action_name,
                "denial_reason": oracle_reason or "",
            }
        )

        # DENY never mutates protected workflow flags. The gate records the
        # attempt; the next policy decision receives the denial feedback.
        state = after
        last_denied_action = next_denied

    return rows


def _partition_for_episode(episode_id: str) -> str:
    """Stable 60/10/10/20 split, hashing episode identity only."""
    bucket = int(hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 6:
        return "train"
    if bucket == 6:
        return "validation"
    if bucket == 7:
        return "calibration"
    return "test"


def _pack_rows(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot pack an empty partition")
    return {
        "context": np.stack([r["context"] for r in rows]),
        "action": np.stack([r["action"] for r in rows]),
        "future": np.stack([r["future"] for r in rows]),
        "security": np.stack([r["security"] for r in rows]),
        "episode_id": np.asarray([r["episode_id"] for r in rows]),
        "family": np.asarray([r["family"] for r in rows]),
        "agent_policy": np.asarray([r["agent_policy"] for r in rows]),
        "action_name": np.asarray([r["action_name"] for r in rows]),
        "denial_reason": np.asarray([r["denial_reason"] for r in rows]),
        "step": np.asarray([r["step"] for r in rows], dtype=np.int64),
    }


def build_dataset(
    seed: int = 20261120,
    episodes_per_family_policy: int = 300,
    episode_length: int = 8,
) -> dict[str, Any]:
    if episodes_per_family_policy < 50:
        raise ValueError("episodes_per_family_policy must be >= 50")
    if episode_length < 2:
        raise ValueError("episode_length must be >= 2")

    rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(AGENT_POLICIES):
        for family_index, family in enumerate(FAMILIES):
            for episode_index in range(episodes_per_family_policy):
                episode_seed = (
                    seed
                    + policy_index * 10_000_000
                    + family_index * 1_000_000
                    + episode_index
                )
                rows.extend(_episode(episode_seed, family, policy, episode_length))

    episode_partition: dict[str, str] = {}
    for row in rows:
        partition = _partition_for_episode(row["episode_id"])
        previous = episode_partition.setdefault(row["episode_id"], partition)
        if previous != partition:
            raise AssertionError("an episode was assigned to multiple partitions")
        row["partition"] = partition

    output = {
        partition: _pack_rows([r for r in rows if r["partition"] == partition])
        for partition in PARTITIONS
    }

    # Verify every policy/family pair appears in every partition. This prevents
    # accidental empty strata in the in-policy evaluation protocol.
    coverage: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        part_rows = [r for r in rows if r["partition"] == partition]
        counts = {
            f"{policy}/{family}": sum(
                1 for r in part_rows
                if r["agent_policy"] == policy and r["family"] == family
            ) // episode_length
            for policy in AGENT_POLICIES
            for family in FAMILIES
        }
        if any(value == 0 for value in counts.values()):
            raise AssertionError(f"missing policy/family stratum in {partition}: {counts}")
        coverage[partition] = counts

    behavior_stats = {}
    for policy in AGENT_POLICIES:
        for family in FAMILIES:
            subset = [r for r in rows if r["agent_policy"] == policy and r["family"] == family]
            episode_map: dict[str, list[dict[str, Any]]] = {}
            for row in subset:
                episode_map.setdefault(row["episode_id"], []).append(row)
            repeated = changed = 0
            for episode_rows in episode_map.values():
                episode_rows.sort(key=lambda item: item["step"])
                for previous, current in zip(episode_rows, episode_rows[1:]):
                    if previous["security"][0] == 1.0:
                        if current["action_name"] == previous["action_name"]:
                            repeated += 1
                        else:
                            changed += 1
            denied = sum(int(r["security"][0]) for r in subset)
            behavior_stats[f"{policy}/{family}"] = {
                "episodes": len(episode_map),
                "transitions": len(subset),
                "denied_transitions": denied,
                "denial_rate": denied / len(subset),
                "repeated_same_action_after_deny": repeated,
                "changed_action_after_deny": changed,
                "mean_unique_actions_per_episode": float(
                    np.mean([
                        len({r["action_name"] for r in ep_rows})
                        for ep_rows in episode_map.values()
                    ])
                ),
            }

    episode_counts = {
        partition: len(set(output[partition]["episode_id"].tolist()))
        for partition in PARTITIONS
    }
    manifest = {
        "benchmark": "G8.4.8.2 uncertain-action policy-conditioned UCON trajectories",
        "seed": seed,
        "episodes_per_family_policy": episodes_per_family_policy,
        "episode_length": episode_length,
        "policy_version": POLICY_VERSION,
        "families": list(FAMILIES),
        "agent_policies": list(AGENT_POLICIES),
        "action_vocabulary": list(ACTION_VOCAB),
        "context_dim": int(output["train"]["context"].shape[1]),
        "action_dim": ACTION_DIM,
        "security_dim": SECURITY_DIM,
        "split_unit": "episode",
        "split_ratio_target": {"train": 0.6, "validation": 0.1, "calibration": 0.1, "test": 0.2},
        "episode_counts": episode_counts,
        "transition_counts": {
            partition: int(len(output[partition]["context"]))
            for partition in PARTITIONS
        },
        "episodes_by_policy_family": coverage,
        "policy_behavior_stats": behavior_stats,
        "fixed_length_check": "all episodes contain exactly episode_length transitions, including after DENY",
        "denial_semantics": "protected workflow state unchanged; attempt history and denial feedback observed; episode continues",
        "feature_boundary": "context contains observable state and denial feedback only; policy identity, episode ID, action name, split, future state, security labels and oracle reason are metadata/targets",
        "limitations": [
            "fully synthetic policy and behavior-policy distributions",
            "goal_directed policy observes synthetic governance preconditions",
            "uniform policy is memoryless over its family action set",
            "retrying_adversary repeats the last denied action",
            "prerequisite_blind_planner follows a fixed cyclic workflow without inspecting governance state",
            "denial_adaptive_explorer advances cyclically after denial and otherwise samples uniformly",
            "not a model of real LLM planning, tool selection or production security",
        ],
    }
    return {**output, "manifest": manifest}


def build_lopo_fold(dataset: dict[str, Any], held_out_policy: str) -> dict[str, Any]:
    """Create a strict LOPO fold: held-out policy appears only in test.

    Non-held-out policies retain the episode-hash train/validation/calibration
    assignments. All episodes of the held-out policy, irrespective of their
    original partition, are pooled into the LOPO test set.
    """
    if held_out_policy not in AGENT_POLICIES:
        raise ValueError(f"unknown held-out policy: {held_out_policy}")

    fold: dict[str, Any] = {}
    for partition in ("train", "validation", "calibration"):
        source = dataset[partition]
        mask = source["agent_policy"] != held_out_policy
        if not np.any(mask):
            raise ValueError(f"empty {partition} partition after holding out {held_out_policy}")
        fold[partition] = {key: value[mask] for key, value in source.items()}

    test_parts = []
    for partition in PARTITIONS:
        source = dataset[partition]
        mask = source["agent_policy"] == held_out_policy
        if np.any(mask):
            test_parts.append({key: value[mask] for key, value in source.items()})
    if not test_parts:
        raise ValueError(f"no test episodes for held-out policy {held_out_policy}")
    fold["test"] = {
        key: np.concatenate([part[key] for part in test_parts], axis=0)
        for key in test_parts[0]
    }

    for partition in ("train", "validation", "calibration"):
        if np.any(fold[partition]["agent_policy"] == held_out_policy):
            raise AssertionError("held-out policy leaked into a fitting/tuning partition")
    if set(fold["test"]["agent_policy"].tolist()) != {held_out_policy}:
        raise AssertionError("LOPO test partition contains a non-held-out policy")
    fold["manifest"] = {
        "protocol": "leave-one-policy-out",
        "held_out_policy": held_out_policy,
        "train_policies": sorted(set(fold["train"]["agent_policy"].tolist())),
        "validation_policies": sorted(set(fold["validation"]["agent_policy"].tolist())),
        "calibration_policies": sorted(set(fold["calibration"]["agent_policy"].tolist())),
        "test_policies": sorted(set(fold["test"]["agent_policy"].tolist())),
        "test_episodes": len(set(fold["test"]["episode_id"].tolist())),
        "test_transitions": len(fold["test"]["context"]),
        "held_out_policy_absent_from_fitting_and_tuning": True,
    }
    return fold


def save_dataset(dataset: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for partition in PARTITIONS:
        np.savez_compressed(output_dir / f"{partition}.npz", **dataset[partition])
    (output_dir / "manifest.json").write_text(
        json.dumps(dataset["manifest"], indent=2) + "\n", encoding="utf-8"
    )
    lopo_dir = output_dir / "lopo"
    lopo_dir.mkdir(exist_ok=True)
    for policy in AGENT_POLICIES:
        fold = build_lopo_fold(dataset, policy)
        fold_dir = lopo_dir / policy
        fold_dir.mkdir(exist_ok=True)
        for partition in PARTITIONS:
            np.savez_compressed(fold_dir / f"{partition}.npz", **fold[partition])
        (fold_dir / "manifest.json").write_text(
            json.dumps(fold["manifest"], indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("g8-4-8-2-dataset"))
    args = parser.parse_args()
    dataset = build_dataset(args.seed, args.episodes_per_family_policy, args.episode_length)
    save_dataset(dataset, args.out)
    print(json.dumps(dataset["manifest"], indent=2))


if __name__ == "__main__":
    main()
