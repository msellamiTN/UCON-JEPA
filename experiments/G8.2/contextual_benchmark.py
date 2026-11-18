#!/usr/bin/env python3
"""Contextual, stateful UCON benchmark with labels from an independent policy oracle.

The oracle evaluates action effects against authorization/workflow state. It does
not inspect motif labels. The same action trace is instantiated in safe and
unsafe contexts to test context sensitivity directly.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

SCENARIOS = {
    "iam": {
        "trace": ["grant_role", "assume_admin"],
        "safe": {"actor_role": "security_admin", "grant_scope": "project", "requested_scope": "project"},
        "unsafe": {"actor_role": "analyst", "grant_scope": "project", "requested_scope": "project"},
    },
    "data": {
        "trace": ["read_sensitive", "export_sensitive"],
        "safe": {"clearance": 3, "classification": 3, "export_approved": True, "recipient": "owner"},
        "unsafe": {"clearance": 1, "classification": 3, "export_approved": False, "recipient": "external"},
    },
    "sod": {
        "trace": ["request", "approve", "pay"],
        "safe": {"requester": "alice", "approver": "bob", "payer": "carol"},
        "unsafe": {"requester": "alice", "approver": "alice", "payer": "carol"},
    },
}

def initial_state(family: str, context: dict) -> dict:
    return {"family": family, "context": dict(context), "role_granted": False,
            "sensitive_read": False, "request_seen": False, "approved": False,
            "violation": False, "violation_action": None, "history": []}

def policy_step(state: dict, action: str) -> tuple[bool, str | None]:
    """Pure policy decision over current state, context and proposed action."""
    c, f = state["context"], state["family"]
    if f == "iam":
        if action == "grant_role":
            return (c["actor_role"] == "security_admin", "grant_requires_security_admin")
        if action == "assume_admin":
            return (state["role_granted"] and c["grant_scope"] == c["requested_scope"],
                    "admin_assumption_requires_matching_prior_grant")
    elif f == "data":
        if action == "read_sensitive":
            return (c["clearance"] >= c["classification"], "insufficient_data_clearance")
        if action == "export_sensitive":
            return (state["sensitive_read"] and c["export_approved"] and c["recipient"] == "owner",
                    "export_requires_authorized_read_approval_and_owner_recipient")
    elif f == "sod":
        if action == "request":
            return (True, None)
        if action == "approve":
            return (state["request_seen"] and c["approver"] != c["requester"],
                    "requester_and_approver_must_differ")
        if action == "pay":
            return (state["approved"] and c["payer"] not in {c["requester"], c["approver"]},
                    "payer_must_be_independent")
    return (True, None)

def apply_action(state: dict, action: str) -> dict:
    allowed, reason = policy_step(state, action)
    nxt = dict(state)
    nxt["history"] = list(state["history"])
    nxt["history"].append(action)
    if not allowed:
        nxt["violation"] = True
        nxt["violation_action"] = action
        nxt["violation_reason"] = reason
        return nxt
    if action == "grant_role": nxt["role_granted"] = True
    if action == "read_sensitive": nxt["sensitive_read"] = True
    if action == "request": nxt["request_seen"] = True
    if action == "approve": nxt["approved"] = True
    return nxt

def evaluate_trace(family: str, trace: list[str], context: dict) -> dict:
    state = initial_state(family, context)
    for action in trace:
        state = apply_action(state, action)
        if state["violation"]:
            break
    return {"unsafe": state["violation"], "violation_action": state.get("violation_action"),
            "violation_reason": state.get("violation_reason"), "executed": state["history"]}

def reactive_ucon_gate(family: str, trace: list[str], context: dict) -> dict:
    """Authoritative reactive gate: deny the first action violating policy state."""
    state = initial_state(family, context)
    accepted, denied = [], []
    for action in trace:
        allowed, reason = policy_step(state, action)
        if not allowed:
            denied.append({"action": action, "reason": reason})
            break
        accepted.append(action)
        state = apply_action(state, action)
    return {"accepted": accepted, "denied": denied, "intervened": bool(denied)}

def paired_cases():
    cases = []
    for family, spec in SCENARIOS.items():
        for context_name in ("safe", "unsafe"):
            ctx = spec[context_name]
            oracle = evaluate_trace(family, spec["trace"], ctx)
            gate = reactive_ucon_gate(family, spec["trace"], ctx)
            cases.append({"family": family, "context": context_name,
                          "trace": spec["trace"], "context_state": ctx,
                          "label_unsafe": oracle["unsafe"], "oracle": oracle, "gate": gate})
    return cases

def run():
    cases = paired_cases()
    safe = [x for x in cases if x["context"] == "safe"]
    unsafe = [x for x in cases if x["context"] == "unsafe"]
    return {
        "benchmark": "G8.2 contextual-stateful policy benchmark v0.1",
        "label_source": "stateful policy transition oracle; no motif matching",
        "cases": cases,
        "invariants": {
            "same_trace_safe_and_unsafe_context": all(s["trace"] == u["trace"] for s,u in zip(safe,unsafe)),
            "safe_contexts_authorized": all(not x["label_unsafe"] for x in safe),
            "unsafe_contexts_violate_policy": all(x["label_unsafe"] for x in unsafe),
            "reactive_gate_blocks_unsafe": all(x["gate"]["intervened"] for x in unsafe),
            "reactive_gate_does_not_block_safe": all(not x["gate"]["intervened"] for x in safe),
        },
        "limitations": "Six deterministic paired cases validate context sensitivity and oracle/gate mechanics only; they are not a statistical efficacy study or JEPA result."
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("contextual-output"))
    args = p.parse_args()
    result = run()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "contextual_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
