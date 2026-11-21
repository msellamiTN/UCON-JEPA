"""G8.4.8.7.1 controlled within-archetype behavior-policy parameter shifts.

This generator is deliberately separate from uncertain_action_dataset.py so
G8.4.8.2 and all previously reported results remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from intervention_task_manifests import sample_task_context, task_manifest_records, validate_task_manifests
from stateful_trajectory_dataset import (
    ACTION_VOCAB, FAMILIES, _context, _initial_state, _online_gate_step,
    _policy_oracle, _state_features,
)

POLICY_VERSION = "ucon-jeepa-policy-rho-v1-g8.4.8.7.1"
POLICY_ARCHETYPES = (
    "goal_directed",
    "uniform",
    "retrying_adversary",
    "prerequisite_blind_planner",
    "denial_adaptive_explorer",
)
TRAIN_RHO = (0.20, 0.40, 0.60)
SHIFT_RHO = 0.85
REVERSE_RHO = 0.05
PARTITIONS = ("train", "validation", "calibration", "test_shifted", "test_reverse")
ACTION_DIM = len(ACTION_VOCAB)
FAMILY_ACTIONS = {
    "iam": ("grant_role", "assume_admin", "iam_read"),
    "data": ("read_sensitive", "export_sensitive", "data_list"),
    "sod": ("request", "approve", "pay"),
}
INITIAL_RISK_ACTION = {
    "iam": "assume_admin",
    "data": "export_sensitive",
    "sod": "pay",
}


def _policy_context(state: dict[str, Any], last_denied_action: str | None) -> np.ndarray:
    feedback = np.zeros(ACTION_DIM, dtype=np.float64)
    if last_denied_action is not None:
        feedback[ACTION_VOCAB.index(last_denied_action)] = 1.0
    return np.concatenate((_state_features(state), feedback))


def _goal_directed_action(state: dict[str, Any]) -> str:
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


def _parameterized_action(
    archetype: str,
    rho: float,
    state: dict[str, Any],
    last_denied_action: str | None,
    step: int,
    rng: random.Random,
) -> str:
    """Sample one action from the declared archetype/rho mixture."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    family = state["context"]["family"]
    actions = FAMILY_ACTIONS[family]

    if archetype == "goal_directed":
        return _goal_directed_action(state) if rng.random() < rho else rng.choice(actions)

    if archetype == "uniform":
        # rho mixes a uniform explorer with a deterministic first-action policy.
        return rng.choice(actions) if rng.random() < rho else actions[0]

    if archetype == "retrying_adversary":
        if last_denied_action is None:
            return INITIAL_RISK_ACTION[family]
        if rng.random() < rho:
            return last_denied_action
        alternatives = [action for action in actions if action != last_denied_action]
        return rng.choice(alternatives)

    if archetype == "prerequisite_blind_planner":
        return actions[step % len(actions)] if rng.random() < rho else rng.choice(actions)

    if archetype == "denial_adaptive_explorer":
        if last_denied_action is not None and rng.random() < rho:
            return actions[(actions.index(last_denied_action) + 1) % len(actions)]
        return rng.choice(actions)

    raise ValueError(f"unknown policy archetype: {archetype}")


def _episode(
    episode_seed: int,
    family: str,
    archetype: str,
    rho: float,
    length: int,
    episode_id: str,
    task_aware: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(episode_seed)
    ctx = (
        sample_task_context(rng, family, _context)
        if task_aware else _context(rng, family)
    )
    state = _initial_state(ctx)
    last_denied_action: str | None = None
    rows = []
    for step in range(length):
        action_name = _parameterized_action(archetype, rho, state, last_denied_action, step, rng)
        context = _policy_context(state, last_denied_action)
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[ACTION_VOCAB.index(action_name)] = 1.0

        oracle_allowed, oracle_reason = _policy_oracle(state, action_name)
        if oracle_allowed:
            oracle_reason = None
        after, gate_allowed, gate_reason = _online_gate_step(state, action_name)
        if oracle_allowed != gate_allowed or oracle_reason != gate_reason:
            raise AssertionError(
                f"oracle/gate disagreement: {family}/{archetype}/rho={rho}/step={step}"
            )
        next_denied = None if oracle_allowed else action_name
        security = np.asarray([
            float(not oracle_allowed),
            float(after["approved"]),
            float(after["sensitive_read"]),
            float(after["role_granted"]),
        ], dtype=np.float64)
        rows.append({
            "context": context,
            "action": action,
            "future": _policy_context(after, next_denied),
            "security": security,
            "episode_id": episode_id,
            "step": step,
            "family": family,
            "agent_policy": archetype,
            "rho": float(rho),
            "policy_config": f"{archetype}:rho={rho:.2f}",
            "action_name": action_name,
            "denial_reason": oracle_reason or "",
        })
        # Denial never mutates protected workflow state; feedback is visible
        # only to the behavior policy and as the current context feature.
        state = after
        last_denied_action = next_denied
    return rows


def _bucket(episode_id: str) -> int:
    return int(hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:8], 16) % 10


def _partition_for_rho(episode_id: str, rho: float) -> str:
    if np.isclose(rho, SHIFT_RHO):
        return "test_shifted"
    if np.isclose(rho, REVERSE_RHO):
        return "test_reverse"
    bucket = _bucket(episode_id)
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    return "calibration"


def _pack(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot pack an empty partition")
    return {
        "context": np.stack([r["context"] for r in rows]),
        "action": np.stack([r["action"] for r in rows]),
        "future": np.stack([r["future"] for r in rows]),
        "security": np.stack([r["security"] for r in rows]),
        "episode_id": np.asarray([r["episode_id"] for r in rows]),
        "step": np.asarray([r["step"] for r in rows], dtype=np.int64),
        "family": np.asarray([r["family"] for r in rows]),
        "agent_policy": np.asarray([r["agent_policy"] for r in rows]),
        "rho": np.asarray([r["rho"] for r in rows], dtype=np.float64),
        "policy_config": np.asarray([r["policy_config"] for r in rows]),
        "action_name": np.asarray([r["action_name"] for r in rows]),
        "denial_reason": np.asarray([r["denial_reason"] for r in rows]),
    }


def build_dataset(
    seed: int = 20261120,
    episodes_per_cell: int = 300,
    episode_length: int = 8,
    include_reverse: bool = True,
    task_aware: bool = False,
) -> dict[str, Any]:
    if episodes_per_cell < 50:
        raise ValueError("episodes_per_cell must be >= 50")
    if episode_length != 8:
        raise ValueError("G8.4.8.7 requires fixed eight-transition episodes")
    rho_cells = list(TRAIN_RHO) + [SHIFT_RHO] + ([REVERSE_RHO] if include_reverse else [])
    rows: list[dict[str, Any]] = []
    for policy_index, archetype in enumerate(POLICY_ARCHETYPES):
        for family_index, family in enumerate(FAMILIES):
            for rho_index, rho in enumerate(rho_cells):
                for episode_index in range(episodes_per_cell):
                    episode_seed = (
                        seed + policy_index * 100_000_000
                        + family_index * 10_000_000 + rho_index * 1_000_000
                        + episode_index
                    )
                    episode_id = (
                        f"rho-{seed}-{archetype}-{family}-"
                        f"{rho:.2f}-{episode_index:05d}"
                    )
                    rows.extend(_episode(
                        episode_seed, family, archetype, rho, episode_length, episode_id,
                        task_aware=task_aware,
                    ))

    partition_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in PARTITIONS}
    episode_partitions: dict[str, str] = {}
    for row in rows:
        partition = _partition_for_rho(row["episode_id"], row["rho"])
        previous = episode_partitions.setdefault(row["episode_id"], partition)
        if previous != partition:
            raise AssertionError("episode crossed partitions")
        partition_rows[partition].append(row)

    output = {name: _pack(partition_rows[name]) for name in PARTITIONS if partition_rows[name]}
    # Verify support and shifted-parameter boundaries by episode, not row.
    for name in ("train", "validation", "calibration"):
        if not np.isin(output[name]["rho"], TRAIN_RHO).all():
            raise AssertionError(f"shifted rho leaked into {name}")
        for archetype in POLICY_ARCHETYPES:
            for family in FAMILIES:
                for rho in TRAIN_RHO:
                    mask = (
                        (output[name]["agent_policy"] == archetype)
                        & (output[name]["family"] == family)
                        & np.isclose(output[name]["rho"], rho)
                    )
                    if not np.any(mask):
                        raise AssertionError(f"missing {name} cell: {archetype}/{family}/{rho}")
    if not np.allclose(output["test_shifted"]["rho"], SHIFT_RHO):
        raise AssertionError("primary test contains unexpected rho values")
    if "test_reverse" in output and not np.allclose(output["test_reverse"]["rho"], REVERSE_RHO):
        raise AssertionError("reverse test contains unexpected rho values")

    counts = {
        name: {
            "transitions": int(len(part["action"])),
            "episodes": int(len(set(part["episode_id"].tolist()))),
            "rho_values": sorted(set(float(x) for x in part["rho"].tolist())),
        }
        for name, part in output.items()
    }
    denial_rates = {}
    for name, part in output.items():
        denial_rates[name] = {}
        for archetype in POLICY_ARCHETYPES:
            for family in FAMILIES:
                for rho in sorted(set(float(x) for x in part["rho"].tolist())):
                    mask = (
                        (part["agent_policy"] == archetype)
                        & (part["family"] == family)
                        & np.isclose(part["rho"], rho)
                    )
                    if np.any(mask):
                        denial_rates[name][f"{archetype}/{family}/rho={rho:.2f}"] = float(
                            part["security"][mask, 0].mean()
                        )

    manifest = {
        "benchmark": "G8.4.8.7 controlled policy-parameter shift",
        "seed": seed,
        "task_aware_contexts": bool(task_aware),
        "task_manifests": task_manifest_records() if task_aware else None,
        "task_manifest_sha256": validate_task_manifests() if task_aware else None,
        "policy_version": POLICY_VERSION,
        "episodes_per_cell": episodes_per_cell,
        "episode_length": episode_length,
        "policy_archetypes": list(POLICY_ARCHETYPES),
        "train_rho": list(TRAIN_RHO),
        "shifted_test_rho": SHIFT_RHO,
        "reverse_test_rho": REVERSE_RHO if include_reverse else None,
        "families": list(FAMILIES),
        "action_vocabulary": list(ACTION_VOCAB),
        "context_feature_boundary": (
            "decision-time observable state plus current denial feedback only; "
            "archetype, rho, episode ID, action labels, future state, security targets, "
            "partition and oracle reason are metadata/targets only"
        ),
        "partition_counts": counts,
        "denial_rates_by_partition_archetype_family_rho": denial_rates,
    }
    return {"manifest": manifest, **output}


def write_dataset(dataset: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in PARTITIONS:
        if name not in dataset:
            continue
        part = dataset[name]
        np.savez_compressed(
            output_dir / f"{name}.npz",
            **{key: value for key, value in part.items()},
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(dataset["manifest"], indent=2) + "\n", encoding="utf-8")
    checksums = {}
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--episodes-per-cell", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("policy-parameter-dataset-output"))
    parser.add_argument("--no-reverse", action="store_true")
    parser.add_argument("--task-aware", action="store_true")
    args = parser.parse_args()
    dataset = build_dataset(
        seed=args.seed, episodes_per_cell=args.episodes_per_cell,
        include_reverse=not args.no_reverse, task_aware=args.task_aware,
    )
    write_dataset(dataset, args.out)
    print(json.dumps({
        "output": str(args.out),
        "manifest": dataset["manifest"]["partition_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
