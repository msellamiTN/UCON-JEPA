"""Fixed-length policy-conditioned UCON trajectories for G8.4.7.

Unlike G8.4.3, episodes do not stop at the first denied action. A denial leaves
protected workflow state unchanged, records the attempted action and denial
feedback, and the agent proposes another action. This removes early-stop
censoring from fixed-horizon windows while explicitly modelling repeated
attempts and post-denial adaptation.

The original G8.4.3 dataset is preserved unchanged for reproducibility.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np

from stateful_trajectory_dataset import (
    ACTION_VOCAB, CONTEXT_DIM, FAMILIES, SECURITY_DIM,
    _context, _initial_state, _online_gate_step, _policy_oracle, _state_features,
)

POLICY_VERSION = "ucon-jeepa-synthetic-v2"
AGENT_POLICIES = ("goal_directed", "uniform", "retrying_adversary")
CONTEXT_DIM_V2 = CONTEXT_DIM + len(ACTION_VOCAB)


def _policy_state_features(state: dict, last_denied_action: str | None) -> np.ndarray:
    feedback = np.zeros(len(ACTION_VOCAB), dtype=np.float64)
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
    raise ValueError(f"unknown policy family: {family}")


def _agent_action(policy: str, state: dict, last_denied_action: str | None,
                  rng: random.Random) -> str:
    family = state["context"]["family"]
    actions = {
        "iam": ("grant_role", "assume_admin", "iam_read"),
        "data": ("read_sensitive", "export_sensitive", "data_list"),
        "sod": ("request", "approve", "pay"),
    }[family]
    if policy == "goal_directed":
        return _goal_directed_action(state)
    if policy == "uniform":
        return rng.choice(actions)
    if policy == "retrying_adversary":
        if last_denied_action is not None:
            return last_denied_action
        return {"iam": "assume_admin", "data": "export_sensitive", "sod": "pay"}[family]
    raise ValueError(f"unknown agent policy: {policy}")


def _episode(seed: int, family: str, policy: str, length: int) -> list[dict]:
    rng = random.Random(seed)
    ctx = _context(rng, family)
    state = _initial_state(ctx)
    last_denied_action = None
    rows = []
    for t in range(length):
        action = _agent_action(policy, state, last_denied_action, rng)
        before = _policy_state_features(state, last_denied_action)
        action_vec = np.zeros(len(ACTION_VOCAB), dtype=np.float64)
        action_vec[ACTION_VOCAB.index(action)] = 1.0

        oracle_allowed, oracle_reason = _policy_oracle(state, action)
        if oracle_allowed:
            oracle_reason = None
        after, gate_allowed, gate_reason = _online_gate_step(state, action)
        if oracle_allowed != gate_allowed or oracle_reason != gate_reason:
            raise AssertionError(
                f"oracle/gate disagreement family={family} policy={policy} action={action} "
                f"oracle={(oracle_allowed, oracle_reason)} gate={(gate_allowed, gate_reason)}"
            )

        next_last_denied = None if oracle_allowed else action
        security = np.asarray([
            float(not oracle_allowed),
            float(after["approved"]),
            float(after["sensitive_read"]),
            float(after["role_granted"]),
        ], dtype=np.float64)
        rows.append({
            "context": before,
            "action": action_vec,
            "future": _policy_state_features(after, next_last_denied),
            "security": security,
            "episode_id": f"{policy}-{family}-{seed}",
            "step": t,
            "family": family,
            "agent_policy": policy,
            "action_name": action,
            "denial_reason": oracle_reason,
        })
        # Denied operations do not change protected workflow flags. They do
        # update attempt history and the feedback observation; the episode
        # continues because the agent may adapt, retry, or choose another tool.
        state = after
        last_denied_action = next_last_denied
    return rows


def build_policy_dataset(seed: int = 20261120, episodes_per_family_policy: int = 300,
                         episode_length: int = 8) -> dict:
    if episodes_per_family_policy < 30 or episode_length < 2:
        raise ValueError("need >=30 episodes per family/policy and episode_length >=2")
    rows = []
    for pi, policy in enumerate(AGENT_POLICIES):
        for fi, family in enumerate(FAMILIES):
            for i in range(episodes_per_family_policy):
                ep_seed = seed + pi * 10_000_000 + fi * 1_000_000 + i
                rows.extend(_episode(ep_seed, family, policy, episode_length))

    episode_partitions = {}
    for row in rows:
        digest = int(hashlib.sha256(row["episode_id"].encode()).hexdigest()[:8], 16) % 10
        row["partition"] = "test" if digest < 2 else ("validation" if digest < 4 else "train")
        old = episode_partitions.setdefault(row["episode_id"], row["partition"])
        if old != row["partition"]:
            raise AssertionError("episode leakage across partitions")

    output = {}
    for part in ("train", "validation", "test"):
        subset = [r for r in rows if r["partition"] == part]
        output[part] = {
            "context": np.stack([r["context"] for r in subset]),
            "action": np.stack([r["action"] for r in subset]),
            "future": np.stack([r["future"] for r in subset]),
            "security": np.stack([r["security"] for r in subset]),
            "episode_id": np.asarray([r["episode_id"] for r in subset]),
            "family": np.asarray([r["family"] for r in subset]),
            "agent_policy": np.asarray([r["agent_policy"] for r in subset]),
            "action_name": np.asarray([r["action_name"] for r in subset]),
            "denial_reason": np.asarray([r["denial_reason"] or "" for r in subset]),
            "step": np.asarray([r["step"] for r in subset], dtype=np.int64),
        }

    episode_policy = {}
    for row in rows:
        episode_policy[row["episode_id"]] = (row["family"], row["agent_policy"])
    policy_counts = {}
    for part in ("train", "validation", "test"):
        ids = set(output[part]["episode_id"].tolist())
        policy_counts[part] = {
            f"{policy}/{family}": sum(1 for eid in ids if episode_policy[eid] == (family, policy))
            for policy in AGENT_POLICIES for family in FAMILIES
        }
    output["manifest"] = {
        "benchmark": "G8.4.7 fixed-length policy-conditioned stateful UCON trajectories",
        "seed": seed,
        "episodes_per_family_policy": episodes_per_family_policy,
        "episode_length": episode_length,
        "policy_version": POLICY_VERSION,
        "families": list(FAMILIES),
        "agent_policies": list(AGENT_POLICIES),
        "action_vocabulary": list(ACTION_VOCAB),
        "context_dim": CONTEXT_DIM_V2,
        "action_dim": len(ACTION_VOCAB),
        "security_dim": SECURITY_DIM,
        "split_unit": "episode",
        "split_ratio_target": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "episode_counts": {p: len(set(output[p]["episode_id"].tolist()))
                           for p in ("train", "validation", "test")},
        "row_counts": {p: len(output[p]["context"]) for p in ("train", "validation", "test")},
        "episodes_by_policy_family": policy_counts,
        "fixed_length_check": "every episode has exactly episode_length rows; denied actions do not truncate episodes",
        "denial_semantics": "protected workflow state is unchanged; attempt history and denial feedback are observed; agent continues",
        "limitations": [
            "synthetic policies and hand-designed agent behavior policies",
            "goal-directed policy sees the synthetic governance context",
            "retrying_adversary intentionally repeats denied privileged actions",
            "not a model of real agent planning, language-model tool selection or production security",
        ],
    }
    return output


def save_dataset(dataset: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for part in ("train", "validation", "test"):
        np.savez_compressed(out / f"{part}.npz", **dataset[part])
    (out / "manifest.json").write_text(json.dumps(dataset["manifest"], indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--episodes-per-family-policy", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("policy-trajectories"))
    args = parser.parse_args()
    dataset = build_policy_dataset(args.seed, args.episodes_per_family_policy, args.episode_length)
    save_dataset(dataset, args.out)
    print(json.dumps(dataset["manifest"], indent=2))


if __name__ == "__main__":
    main()
