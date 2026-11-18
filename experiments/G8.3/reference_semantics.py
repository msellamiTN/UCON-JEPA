#!/usr/bin/env python3
"""G8.3 executable reference semantics for temporal UCON policy events.

This is a deliberately small, explicit research oracle. It is not production
authorization middleware and does not claim to cover distributed concurrency.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

class TraceError(ValueError):
    pass

@dataclass
class State:
    policy_version: int = 1
    grants: dict[str, dict[str, Any]] = field(default_factory=dict)
    revoked: set[str] = field(default_factory=set)
    reads: set[tuple[str, str]] = field(default_factory=set)
    workflows: dict[str, dict[str, Any]] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)

def _require(event: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in event]
    if missing:
        raise TraceError(f"missing required fields: {missing}")

def _active_grant(s: State, grant_id: str, at: int, principal: str,
                  scope: str, tenant: str | None = None) -> bool:
    g = s.grants.get(grant_id)
    return bool(g and grant_id not in s.revoked
                and at < g["expires_at"]
                and g["principal"] == principal
                and scope in g["scopes"]
                and g["policy_version"] == s.policy_version
                and (tenant is None or g["tenant"] == tenant))

def evaluate(events: list[dict]) -> dict:
    s = State()
    previous = (-1, -1)
    for e in events:
        if not isinstance(e, dict):
            raise TraceError("each event must be an object")
        _require(e, "time", "sequence", "type")
        key = (e["time"], e["sequence"])
        if not all(isinstance(x, int) and x >= 0 for x in key) or key <= previous:
            raise TraceError("event order must be strictly increasing nonnegative (time, sequence)")
        previous = key
        t, typ = e["time"], e["type"]
        decision = {"time": t, "sequence": e["sequence"], "event": typ, "allowed": True, "reason": None}
        if typ == "policy_update":
            _require(e, "version")
            if not isinstance(e["version"], int) or e["version"] <= s.policy_version:
                raise TraceError("policy version must increase monotonically")
            s.policy_version = e["version"]
        elif typ == "grant":
            _require(e, "grant_id", "actor", "principal", "scopes", "tenant", "expires_at", "policy_version", "grantable_scopes")
            allowed = (e["actor"] == "security_admin"
                       and set(e["scopes"]).issubset(set(e["grantable_scopes"]))
                       and e["expires_at"] > t
                       and e["policy_version"] == s.policy_version)
            if allowed:
                s.grants[e["grant_id"]] = {k:e[k] for k in ("principal","scopes","tenant","expires_at","policy_version")}
            else:
                decision.update(allowed=False, reason="grant_not_authorized_or_invalid")
        elif typ == "revoke":
            _require(e, "grant_id")
            if e["grant_id"] not in s.grants:
                decision.update(allowed=False, reason="unknown_grant")
            else:
                s.revoked.add(e["grant_id"])
        elif typ == "delegate":
            _require(e, "source_grant_id", "grant_id", "principal", "scopes", "expires_at")
            src = s.grants.get(e["source_grant_id"])
            allowed = bool(src and e["source_grant_id"] not in s.revoked
                           and t < src["expires_at"]
                           and set(e["scopes"]).issubset(set(src["scopes"]))
                           and e["expires_at"] <= src["expires_at"]
                           and e["expires_at"] > t)
            if allowed:
                s.grants[e["grant_id"]] = {"principal":e["principal"], "scopes":list(e["scopes"]),
                    "tenant":src["tenant"], "expires_at":e["expires_at"], "policy_version":s.policy_version}
            else:
                decision.update(allowed=False, reason="delegation_exceeds_or_lacks_active_authority")
        elif typ == "read":
            _require(e, "grant_id", "principal", "scope", "tenant", "resource_id", "classification", "clearance")
            allowed = (_active_grant(s,e["grant_id"],t,e["principal"],e["scope"],e["tenant"])
                       and e["clearance"] >= e["classification"])
            if allowed:
                s.reads.add((e["principal"],e["resource_id"]))
            else:
                decision.update(allowed=False, reason="read_not_authorized")
        elif typ == "export":
            _require(e, "principal", "resource_id", "recipient", "approval", "workflow_revision", "policy_version")
            a = e["approval"]
            allowed = ((e["principal"],e["resource_id"]) in s.reads
                       and e["recipient"] == "owner"
                       and a.get("approved") is True
                       and a.get("revision") == e["workflow_revision"]
                       and a.get("policy_version") == s.policy_version
                       and e["policy_version"] == s.policy_version)
            if not allowed:
                decision.update(allowed=False, reason="export_lacks_current_read_or_fresh_approval")
        elif typ == "workflow_request":
            _require(e, "workflow_id", "requester", "revision")
            s.workflows[e["workflow_id"]] = {"requester":e["requester"],"revision":e["revision"],"approval":None}
        elif typ == "workflow_amend":
            _require(e, "workflow_id", "revision")
            w=s.workflows.get(e["workflow_id"])
            if not w or e["revision"] <= w["revision"]:
                decision.update(allowed=False, reason="invalid_workflow_revision")
            else:
                w["revision"]=e["revision"]; w["approval"]=None
        elif typ == "approve":
            _require(e, "workflow_id", "approver", "revision", "policy_version")
            w=s.workflows.get(e["workflow_id"])
            allowed=bool(w and e["revision"]==w["revision"] and e["revision"]>=0
                         and e["approver"]!=w["requester"]
                         and e["policy_version"]==s.policy_version)
            if allowed:
                w["approval"]={"approver":e["approver"],"revision":e["revision"],"policy_version":s.policy_version}
            else:
                decision.update(allowed=False, reason="approval_not_fresh_or_sod_violation")
        elif typ == "pay":
            _require(e, "workflow_id", "payer", "revision")
            w=s.workflows.get(e["workflow_id"]); a=w.get("approval") if w else None
            allowed=bool(w and a and e["revision"]==w["revision"]==a["revision"]
                         and a["policy_version"]==s.policy_version
                         and len({w["requester"],a["approver"],e["payer"]})==3)
            if not allowed:
                decision.update(allowed=False, reason="payment_lacks_current_approval_or_sod_violation")
        else:
            raise TraceError(f"unknown event type: {typ}")
        s.decisions.append(decision)
    return {"policy_version":s.policy_version,"decisions":s.decisions,
            "allowed":all(x["allowed"] for x in s.decisions)}

if __name__ == "__main__":
    import json, sys
    print(json.dumps(evaluate(json.load(sys.stdin)), indent=2))
