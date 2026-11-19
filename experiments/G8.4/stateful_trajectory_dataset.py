"""Randomized synthetic stateful UCON transition dataset for G8.4.

This extends the repository's IAM, sensitive-data export and SoD policy families.
Each row is a decision-time (past state, candidate action) paired with the realized
post-transition observation and target-side security labels. The episode is the
independent split unit. The data are synthetic and do not represent deployment
security or a validated real-world policy distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np

POLICY_VERSION = "ucon-jeepa-synthetic-v1"
FAMILIES = ("iam", "data", "sod")
ACTION_VOCAB = (
    "grant_role", "assume_admin", "iam_read",
    "read_sensitive", "export_sensitive", "data_list",
    "request", "approve", "pay",
)
FAMILY_ACTIONS = {
    "iam": ("grant_role", "assume_admin", "iam_read"),
    "data": ("read_sensitive", "export_sensitive", "data_list"),
    "sod": ("request", "approve", "pay"),
}
# Family indicators + eight policy-context features + four workflow flags
# + nine action-history counts + one violation flag.
CONTEXT_DIM = 3 + 8 + 4 + len(ACTION_VOCAB) + 1
ACTION_DIM = len(ACTION_VOCAB)
SECURITY_DIM = 4


def _context(rng: random.Random, family: str) -> dict:
    return {
        "family": family,
        "security_admin": rng.random() < 0.5,
        "scope_match": rng.random() < 0.8,
        "clearance": rng.randint(0, 3),
        "classification": rng.randint(1, 3),
        "export_approved": rng.random() < 0.5,
        "recipient_owner": rng.random() < 0.5,
        "approver_independent": rng.random() < 0.75,
        "payer_independent": rng.random() < 0.75,
    }


def _initial_state(ctx: dict) -> dict:
    return {
        "context": ctx, "role_granted": False, "sensitive_read": False,
        "request_seen": False, "approved": False, "violation": False,
        "history_counts": {a: 0 for a in ACTION_VOCAB},
    }


def _state_features(state: dict) -> np.ndarray:
    c = state["context"]
    fam = [float(c["family"] == f) for f in FAMILIES]
    governance = [
        float(c["security_admin"]), float(c["scope_match"]),
        c["clearance"] / 3.0, c["classification"] / 3.0,
        float(c["export_approved"]), float(c["recipient_owner"]),
        float(c["approver_independent"]), float(c["payer_independent"]),
    ]
    workflow = [
        float(state["role_granted"]), float(state["sensitive_read"]),
        float(state["request_seen"]), float(state["approved"]),
    ]
    history = [float(state["history_counts"][a]) / 8.0 for a in ACTION_VOCAB]
    return np.asarray(fam + governance + workflow + history + [float(state["violation"])], dtype=np.float64)


def _policy_oracle(state: dict, action: str) -> tuple[bool, str | None]:
    """Versioned batch policy semantics; returns admission and violation reason."""
    c = state["context"]
    family = c["family"]
    if family == "iam":
        if action == "grant_role":
            return (c["security_admin"], "grant_requires_security_admin")
        if action == "assume_admin":
            return (state["role_granted"] and c["scope_match"],
                    "admin_assumption_requires_prior_matching_grant")
    elif family == "data":
        if action == "read_sensitive":
            return (c["clearance"] >= c["classification"], "insufficient_clearance")
        if action == "export_sensitive":
            ok = state["sensitive_read"] and c["export_approved"] and c["recipient_owner"]
            return (ok, "export_requires_read_approval_and_owner_recipient")
    elif family == "sod":
        if action == "request":
            return (True, None)
        if action == "approve":
            return (state["request_seen"] and c["approver_independent"],
                    "approval_requires_request_and_independent_approver")
        if action == "pay":
            return (state["approved"] and c["payer_independent"],
                    "payment_requires_approval_and_independent_payer")
    return (True, None)


def _online_gate_step(state: dict, action: str) -> tuple[dict, bool, str | None]:
    """Online enforcement transition, intentionally separate from batch oracle."""
    c = state["context"]
    f = c["family"]
    allowed = True
    reason = None
    if f == "iam" and action == "grant_role" and not c["security_admin"]:
        allowed, reason = False, "grant_requires_security_admin"
    elif f == "iam" and action == "assume_admin" and not (state["role_granted"] and c["scope_match"]):
        allowed, reason = False, "admin_assumption_requires_prior_matching_grant"
    elif f == "data" and action == "read_sensitive" and c["clearance"] < c["classification"]:
        allowed, reason = False, "insufficient_clearance"
    elif f == "data" and action == "export_sensitive" and not (
        state["sensitive_read"] and c["export_approved"] and c["recipient_owner"]
    ):
        allowed, reason = False, "export_requires_read_approval_and_owner_recipient"
    elif f == "sod" and action == "approve" and not (
        state["request_seen"] and c["approver_independent"]
    ):
        allowed, reason = False, "approval_requires_request_and_independent_approver"
    elif f == "sod" and action == "pay" and not (
        state["approved"] and c["payer_independent"]
    ):
        allowed, reason = False, "payment_requires_approval_and_independent_payer"

    nxt = {
        "context": c,
        "role_granted": state["role_granted"],
        "sensitive_read": state["sensitive_read"],
        "request_seen": state["request_seen"],
        "approved": state["approved"],
        "violation": not allowed,
        "history_counts": dict(state["history_counts"]),
    }
    nxt["history_counts"][action] += 1
    if allowed:
        if action == "grant_role": nxt["role_granted"] = True
        elif action == "read_sensitive": nxt["sensitive_read"] = True
        elif action == "request": nxt["request_seen"] = True
        elif action == "approve": nxt["approved"] = True
    return nxt, allowed, reason


def _episode(seed: int, family: str, length: int) -> list[dict]:
    rng = random.Random(seed)
    ctx = _context(rng, family)
    state = _initial_state(ctx)
    rows = []
    for t in range(length):
        actions = FAMILY_ACTIONS[family]
        # Random behavior with mild coverage pressure for workflow transitions.
        action = rng.choice(actions)
        before = _state_features(state)
        action_vec = np.zeros(ACTION_DIM, dtype=np.float64)
        action_vec[ACTION_VOCAB.index(action)] = 1.0

        # Independent policy oracle labels the candidate transition.
        oracle_allowed, oracle_reason = _policy_oracle(state, action)
        after, gate_allowed, gate_reason = _online_gate_step(state, action)
        if oracle_allowed != gate_allowed or oracle_reason != gate_reason:
            raise AssertionError(
                f"oracle/gate disagreement family={family} action={action} "
                f"oracle={(oracle_allowed, oracle_reason)} gate={(gate_allowed, gate_reason)}"
            )
        security = np.asarray([
            float(not oracle_allowed),
            float(after["approved"]),
            float(after["sensitive_read"]),
            float(after["role_granted"]),
        ], dtype=np.float64)
        rows.append({
            "context": before, "action": action_vec,
            "future": _state_features(after), "security": security,
            "episode_id": f"{family}-{seed}", "step": t,
            "family": family, "policy_version": POLICY_VERSION,
            "oracle_reason": oracle_reason,
        })
        state = after
        if not oracle_allowed:
            break
    return rows


def build_dataset(seed: int = 20261119, episodes_per_family: int = 1200,
                  episode_length: int = 8) -> dict:
    if episodes_per_family < 30 or episode_length < 2:
        raise ValueError("need >=30 episodes/family and episode_length >=2")
    rows = []
    for fi, family in enumerate(FAMILIES):
        for i in range(episodes_per_family):
            ep_seed = seed + fi * 1_000_000 + i
            rows.extend(_episode(ep_seed, family, episode_length))

    # Stable hash split: every row of an episode receives exactly one partition.
    for row in rows:
        digest = int(hashlib.sha256(row["episode_id"].encode()).hexdigest()[:8], 16) % 10
        row["partition"] = "test" if digest < 2 else ("validation" if digest < 4 else "train")

    episode_partitions = {}
    for row in rows:
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
            "step": np.asarray([r["step"] for r in subset], dtype=np.int64),
        }
    output["manifest"] = {
        "benchmark": "G8.4 randomized stateful UCON transitions v1",
        "seed": seed, "episodes_per_family": episodes_per_family,
        "episode_length_cap": episode_length, "policy_version": POLICY_VERSION,
        "families": list(FAMILIES), "action_vocabulary": list(ACTION_VOCAB),
        "split_unit": "episode", "split_ratio_target": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "rows": {p: int(len(output[p]["context"])) for p in ("train", "validation", "test")},
        "episodes": {p: len(set(output[p]["episode_id"].tolist())) for p in ("train", "validation", "test")},
        "context_dim": CONTEXT_DIM, "action_dim": ACTION_DIM, "future_dim": CONTEXT_DIM,
        "security_dim": SECURITY_DIM,
        "label_source": "versioned stateful policy oracle differentially checked against separately coded online gate",
        "limitations": [
            "synthetic policy/context distributions; not representative of production environments",
            "oracle and gate are separately coded but share the written policy semantics",
            "episode-level random split only; compositional holdout and domain shift are later evaluation requirements",
            "candidate actions are sampled from a fixed family vocabulary; no adaptive agent policy is modelled",
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
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--episodes-per-family", type=int, default=1200)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("trajectory-data"))
    args = parser.parse_args()
    dataset = build_dataset(args.seed, args.episodes_per_family, args.episode_length)
    save_dataset(dataset, args.out)
    print(json.dumps(dataset["manifest"], indent=2))


if __name__ == "__main__":
    main()
