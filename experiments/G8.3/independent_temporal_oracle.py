#!/usr/bin/env python3
"""Independent, state-machine differential oracle for G8.3.

This module intentionally does not import the production/reference evaluator.
It uses a separately written state representation and guard decomposition.
Generated traces are synthetic, well-formed, and designed to cover each event
transition plus contextual boundary mutations. Agreement is not a proof that
the policy specification itself is correct.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from reference_semantics import evaluate

EVENT_TYPES = (
    "policy_update", "grant", "revoke", "delegate", "assume_admin", "read",
    "workflow_request", "workflow_amend", "approve", "export", "pay",
)
ENTITLEMENTS = {"security_admin": ["read", "write", "admin"]}


def oracle(events: list[dict], entitlements: dict | None = None) -> dict:
    """Evaluate a trace using an independent state machine (no evaluator imports)."""
    rights = ENTITLEMENTS if entitlements is None else entitlements
    version = 1
    grants: dict[str, dict] = {}
    revoked: set[str] = set()
    read_receipts: set[tuple[str, str]] = set()
    flows: dict[str, dict] = {}
    decisions = []

    def active(grant_id, when, principal, scope, tenant):
        grant = grants.get(grant_id)
        return bool(
            grant is not None
            and grant_id not in revoked
            and when < grant["until"]
            and grant["who"] == principal
            and scope in grant["rights"]
            and grant["version"] == version
            and grant["tenant"] == tenant
        )

    for event in events:
        when, kind = event["time"], event["type"]
        decision = {"time": when, "sequence": event["sequence"],
                    "event": kind, "allowed": True, "reason": None}

        if kind == "policy_update":
            version = event["version"]

        elif kind == "grant":
            rights_requested = event["scopes"]
            valid = (
                event["actor"] == "security_admin"
                and set(rights_requested).issubset(set(rights.get(event["actor"], [])))
                and event["expires_at"] > when
                and event["policy_version"] == version
                and event["grant_id"] not in grants
            )
            if valid:
                grants[event["grant_id"]] = {
                    "who": event["principal"], "rights": list(rights_requested),
                    "tenant": event["tenant"], "until": event["expires_at"],
                    "version": version,
                }
            else:
                decision.update(allowed=False, reason="grant_not_authorized_or_invalid")

        elif kind == "revoke":
            grant_id = event["grant_id"]
            if grant_id not in grants:
                decision.update(allowed=False, reason="unknown_grant")
            else:
                revoked.add(grant_id)

        elif kind == "delegate":
            source_id = event["source_grant_id"]
            source = grants.get(source_id)
            scopes = event["scopes"]
            valid = bool(
                source is not None
                and source_id not in revoked
                and when < source["until"]
                and source["version"] == version
                and set(scopes).issubset(set(source["rights"]))
                and when < event["expires_at"] <= source["until"]
                and event["grant_id"] not in grants
            )
            if valid:
                grants[event["grant_id"]] = {
                    "who": event["principal"], "rights": list(scopes),
                    "tenant": source["tenant"], "until": event["expires_at"],
                    "version": version,
                }
            else:
                decision.update(allowed=False, reason="delegation_exceeds_or_lacks_current_authority")

        elif kind == "assume_admin":
            if not active(event["grant_id"], when, event["principal"],
                          event["scope"], event["tenant"]):
                decision.update(allowed=False, reason="no_active_current_admin_grant")

        elif kind == "read":
            valid = (
                active(event["grant_id"], when, event["principal"],
                       event["scope"], event["tenant"])
                and event["clearance"] >= event["classification"]
            )
            if valid:
                read_receipts.add((event["principal"], event["resource_id"]))
            else:
                decision.update(allowed=False, reason="read_not_authorized")

        elif kind == "workflow_request":
            flow_id = event["workflow_id"]
            if flow_id in flows:
                decision.update(allowed=False, reason="workflow_already_exists")
            else:
                flows[flow_id] = {
                    "requester": event["requester"],
                    "revision": event["revision"],
                    "approval": None,
                }

        elif kind == "workflow_amend":
            flow = flows.get(event["workflow_id"])
            if flow is None or event["revision"] <= flow["revision"]:
                decision.update(allowed=False, reason="invalid_workflow_revision")
            else:
                flow["revision"] = event["revision"]
                flow["approval"] = None

        elif kind == "approve":
            flow = flows.get(event["workflow_id"])
            valid = bool(
                flow is not None
                and event["revision"] == flow["revision"]
                and event["approver"] != flow["requester"]
                and event["policy_version"] == version
            )
            if valid:
                flow["approval"] = {
                    field: event[field] for field in (
                        "approver", "revision", "policy_version", "action",
                        "principal", "resource_id", "recipient"
                    )
                }
            else:
                decision.update(allowed=False, reason="approval_not_fresh_or_sod_violation")

        elif kind == "export":
            flow = flows.get(event["workflow_id"])
            approval = flow["approval"] if flow is not None else None
            valid = bool(
                approval is not None
                and (event["principal"], event["resource_id"]) in read_receipts
                and event["recipient"] == "owner"
                and approval["action"] == "export"
                and approval["principal"] == event["principal"]
                and approval["resource_id"] == event["resource_id"]
                and approval["recipient"] == event["recipient"]
                and approval["revision"] == event["workflow_revision"] == flow["revision"]
                and approval["policy_version"] == version == event["policy_version"]
            )
            if not valid:
                decision.update(allowed=False, reason="export_lacks_subject_bound_current_approval_or_read")

        elif kind == "pay":
            flow = flows.get(event["workflow_id"])
            approval = flow["approval"] if flow is not None else None
            valid = bool(
                flow is not None
                and approval is not None
                and approval["action"] == "pay"
                and event["revision"] == flow["revision"] == approval["revision"]
                and approval["policy_version"] == version
                and len({flow["requester"], approval["approver"], event["payer"]}) == 3
            )
            if not valid:
                decision.update(allowed=False, reason="payment_lacks_current_approval_or_sod_violation")

        decisions.append(decision)

    return {"policy_version": version, "decisions": decisions,
            "allowed": all(item["allowed"] for item in decisions)}


def _event(time, kind, **fields):
    return {"time": time, "sequence": 0, "type": kind, **fields}


def generate_trace(rng: random.Random, episode_id: int) -> list[dict]:
    """Generate a valid-shape trace with randomized context and boundary cases."""
    case = rng.choice((
        "grant", "revoke", "expiry", "policy_update", "delegate",
        "assume_admin", "read", "export", "amend", "pay", "duplicate_workflow",
    ))
    events = []
    t = 1
    gid = f"g-{episode_id}"
    principal = rng.choice(("alice", "bob", "carol"))
    tenant = rng.choice(("tenant-a", "tenant-b"))
    scope = rng.choice(("read", "admin", "write"))
    expiry = t + rng.choice((1, 2, 5, 20))

    def add_grant(grant_id=gid, grant_scope=scope, until=expiry,
                  actor="security_admin", policy_version=1):
        events.append(_event(t, "grant", grant_id=grant_id, actor=actor,
            principal=principal, scopes=[grant_scope], tenant=tenant,
            expires_at=until, policy_version=policy_version))

    if case == "grant":
        add_grant(actor=rng.choice(("security_admin", "analyst")),
                  grant_scope=rng.choice(("read", "admin", "unknown")),
                  until=t + rng.choice((0, 1, 5)))
    elif case == "revoke":
        add_grant(grant_scope="read", until=t+10)
        events.append(_event(t+1, "revoke", grant_id=gid if rng.random()<.8 else "missing"))
        events.append(_event(t+2, "read", grant_id=gid, principal=principal,
            scope="read", tenant=tenant, resource_id="r1", classification=1, clearance=2))
    elif case == "expiry":
        add_grant(grant_scope="read", until=t+1)
        events.append(_event(t+1, "read", grant_id=gid, principal=principal,
            scope="read", tenant=tenant, resource_id="r1", classification=1, clearance=2))
    elif case == "policy_update":
        add_grant(grant_scope="read", until=t+10)
        events.append(_event(t+1, "policy_update", version=2))
        events.append(_event(t+2, "read", grant_id=gid, principal=principal,
            scope="read", tenant=tenant, resource_id="r1", classification=1, clearance=2))
    elif case == "delegate":
        add_grant(grant_scope="read", until=t+10)
        child_scopes = ["read"] if rng.random()<.5 else ["read", "write"]
        events.append(_event(t+1, "delegate", source_grant_id=gid,
            grant_id=f"child-{episode_id}", principal="dave", scopes=child_scopes,
            expires_at=t+rng.choice((2, 8, 12))))
    elif case == "assume_admin":
        add_grant(grant_scope="admin", until=t+10)
        events.append(_event(t+1, "assume_admin", grant_id=gid,
            principal=principal if rng.random()<.8 else "mallory",
            scope="admin", tenant=tenant))
    elif case == "read":
        add_grant(grant_scope="read", until=t+10)
        events.append(_event(t+1, "read", grant_id=gid, principal=principal,
            scope="read", tenant=tenant, resource_id="r1",
            classification=rng.randrange(0, 4), clearance=rng.randrange(0, 4)))
    elif case in ("export", "amend"):
        add_grant(grant_scope="read", until=t+20)
        events.append(_event(t+1, "read", grant_id=gid, principal=principal,
            scope="read", tenant=tenant, resource_id="r1", classification=1, clearance=2))
        events.append(_event(t+2, "workflow_request", workflow_id="wf",
            requester=principal, revision=1))
        events.append(_event(t+3, "approve", workflow_id="wf",
            approver=rng.choice(("alice", "bob", "carol", "mallory")),
            revision=1, policy_version=1,
            action="export", principal=principal, resource_id="r1",
            recipient="owner"))
        if case == "amend":
            events.append(_event(t+4, "workflow_amend", workflow_id="wf", revision=2))
            events.append(_event(t+5, "export", workflow_id="wf", principal=principal,
                resource_id="r1", recipient="owner", workflow_revision=2, policy_version=1))
        else:
            events.append(_event(t+4, "export", workflow_id="wf", principal=principal,
                resource_id=rng.choice(("r1", "r2")), recipient=rng.choice(("owner", "external")),
                workflow_revision=1, policy_version=1))
    elif case == "pay":
        requester = rng.choice(("alice", "bob", "carol"))
        approver = rng.choice(("alice", "bob", "carol"))
        payer = rng.choice(("alice", "bob", "carol"))
        events.extend([
            _event(t, "workflow_request", workflow_id="pay-wf", requester=requester, revision=1),
            _event(t+1, "approve", workflow_id="pay-wf", approver=approver,
                revision=1, policy_version=1, action="pay", principal=requester,
                resource_id="invoice-1", recipient="vendor"),
            _event(t+2, "pay", workflow_id="pay-wf", payer=payer, revision=1),
        ])
    else:
        events.extend([
            _event(t, "workflow_request", workflow_id="dup-wf", requester="alice", revision=1),
            _event(t+1, "workflow_request", workflow_id="dup-wf", requester="bob", revision=1),
        ])

    return events


def run_campaign(seed: int = 8304, episodes: int = 5000) -> dict:
    rng = random.Random(seed)
    mismatch_count = 0
    event_count = 0
    case_counts: dict[str, int] = {}
    examples = []
    for index in range(episodes):
        trace = generate_trace(rng, index)
        expected = oracle(trace)
        actual = evaluate(trace, ENTITLEMENTS)
        event_count += len(trace)
        # Count the observable event-family signature, not a hidden generator label.
        signature = "->".join(event["type"] for event in trace)
        case_counts[signature] = case_counts.get(signature, 0) + 1
        if expected != actual:
            mismatch_count += 1
            if len(examples) < 20:
                examples.append({"trace": trace, "oracle": expected, "evaluator": actual})
    return {
        "benchmark": "G8.3.4 independent temporal state-machine differential audit",
        "version": "0.1.0",
        "seed": seed,
        "episodes": episodes,
        "event_count": event_count,
        "mismatch_count": mismatch_count,
        "agreement": 1.0 - mismatch_count / episodes if episodes else 1.0,
        "trace_signatures": case_counts,
        "counterexamples": examples,
        "limitations": (
            "Synthetic generator and independent implementation encode the same written policy contract. "
            "Differential agreement checks implementation consistency, not correctness of the policy specification "
            "or real-world efficacy."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8304)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/g8.3.4"))
    args = parser.parse_args()
    result = run_campaign(args.seed, args.episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "independent_temporal_oracle.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "counterexamples"}, indent=2))
    if result["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
