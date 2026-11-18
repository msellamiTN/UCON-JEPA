#!/usr/bin/env python3
"""G8.3.1 temporal UCON reference semantics (research model, not production)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

EVENT_TYPES = frozenset({"policy_update","grant","revoke","delegate","assume_admin","read","workflow_request","workflow_amend","approve","export","pay"})\n\nclass TraceError(ValueError): pass

@dataclass
class State:
    policy_version: int = 1
    grants: dict[str, dict[str, Any]] = field(default_factory=dict)
    revoked: set[str] = field(default_factory=set)
    reads: set[tuple[str, str]] = field(default_factory=set)
    workflows: dict[str, dict[str, Any]] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)

def _require(e: dict, *keys: str) -> None:
    missing=[k for k in keys if k not in e]
    if missing: raise TraceError(f"missing required fields: {missing}")

def _active(s: State, gid: str, t: int, principal: str, scope: str, tenant: str|None=None) -> bool:
    g=s.grants.get(gid)
    return bool(g and gid not in s.revoked and t < g["expires_at"]
      and g["principal"]==principal and scope in g["scopes"]
      and g["policy_version"]==s.policy_version
      and (tenant is None or g["tenant"]==tenant))

def evaluate(events: list[dict], trusted_entitlements: dict[str, list[str]]|None=None) -> dict:
    """Evaluate events against caller-independent entitlement state.

    trusted_entitlements is provisioned out-of-band by the experiment harness;
    no event may assert or extend an actor's own grantable scopes.
    """
    entitlements=trusted_entitlements or {}
    s=State(); previous=(-1,-1)
    for e in events:
        if not isinstance(e,dict): raise TraceError("each event must be an object")
        _require(e,"time","sequence","type")
        key=(e["time"],e["sequence"])
        if not all(type(x) is int and x>=0 for x in key) or key<=previous:
            raise TraceError("event order must be strictly increasing nonnegative (time, sequence)")
        previous=key; t,typ=e["time"],e["type"]
        d={"time":t,"sequence":e["sequence"],"event":typ,"allowed":True,"reason":None}
        if typ=="policy_update":
            _require(e,"version")
            if type(e["version"]) is not int or e["version"]<=s.policy_version: raise TraceError("policy version must increase")
            s.policy_version=e["version"]
        elif typ=="grant":
            _require(e,"grant_id","actor","principal","scopes","tenant","expires_at","policy_version")
            scopes=e["scopes"]
            allowed=(e["actor"]=="security_admin" and isinstance(scopes,list) and bool(scopes)
              and set(scopes).issubset(set(entitlements.get(e["actor"],[])))
              and type(e["expires_at"]) is int and e["expires_at"]>t
              and e["policy_version"]==s.policy_version and e["grant_id"] not in s.grants)
            if allowed:
                s.grants[e["grant_id"]]={"principal":e["principal"],"scopes":list(scopes),"tenant":e["tenant"],
                  "expires_at":e["expires_at"],"policy_version":s.policy_version}
            else: d.update(allowed=False,reason="grant_not_authorized_or_invalid")
        elif typ=="revoke":
            _require(e,"grant_id")
            if e["grant_id"] not in s.grants: d.update(allowed=False,reason="unknown_grant")
            else: s.revoked.add(e["grant_id"])
        elif typ=="delegate":
            _require(e,"source_grant_id","grant_id","principal","scopes","expires_at")
            src=s.grants.get(e["source_grant_id"])
            allowed=bool(src and e["source_grant_id"] not in s.revoked
              and t<src["expires_at"] and src["policy_version"]==s.policy_version
              and isinstance(e["scopes"],list) and bool(e["scopes"])
              and set(e["scopes"]).issubset(set(src["scopes"]))
              and type(e["expires_at"]) is int and t<e["expires_at"]<=src["expires_at"]
              and e["grant_id"] not in s.grants)
            if allowed:
                s.grants[e["grant_id"]]={"principal":e["principal"],"scopes":list(e["scopes"]),
                  "tenant":src["tenant"],"expires_at":e["expires_at"],"policy_version":s.policy_version}
            else: d.update(allowed=False,reason="delegation_exceeds_or_lacks_current_authority")
        elif typ=="assume_admin":
            _require(e,"grant_id","principal","scope","tenant")
            if not _active(s,e["grant_id"],t,e["principal"],e["scope"],e["tenant"]):
                d.update(allowed=False,reason="no_active_current_admin_grant")
        elif typ=="read":
            _require(e,"grant_id","principal","scope","tenant","resource_id","classification","clearance")
            ok=(_active(s,e["grant_id"],t,e["principal"],e["scope"],e["tenant"])
                and type(e["classification"]) is int and type(e["clearance"]) is int
                and e["clearance"]>=e["classification"])
            if ok: s.reads.add((e["principal"],e["resource_id"]))
            else: d.update(allowed=False,reason="read_not_authorized")
        elif typ=="workflow_request":
            _require(e,"workflow_id","requester","revision")
            if e["workflow_id"] in s.workflows: d.update(allowed=False,reason="workflow_already_exists")
            else: s.workflows[e["workflow_id"]]={"requester":e["requester"],"revision":e["revision"],"approval":None}
        elif typ=="workflow_amend":
            _require(e,"workflow_id","revision")
            w=s.workflows.get(e["workflow_id"])
            if not w or type(e["revision"]) is not int or e["revision"]<=w["revision"]:
                d.update(allowed=False,reason="invalid_workflow_revision")
            else: w["revision"]=e["revision"]; w["approval"]=None
        elif typ=="approve":
            _require(e,"workflow_id","approver","revision","policy_version","action","principal","resource_id","recipient")
            w=s.workflows.get(e["workflow_id"])
            ok=bool(w and e["revision"]==w["revision"] and e["approver"]!=w["requester"]
              and e["policy_version"]==s.policy_version)
            if ok:
                w["approval"]={k:e[k] for k in ("approver","revision","policy_version","action","principal","resource_id","recipient")}
            else: d.update(allowed=False,reason="approval_not_fresh_or_sod_violation")
        elif typ=="export":
            _require(e,"workflow_id","principal","resource_id","recipient","workflow_revision","policy_version")
            w=s.workflows.get(e["workflow_id"]); a=w.get("approval") if w else None
            ok=bool(a and (e["principal"],e["resource_id"]) in s.reads and e["recipient"]=="owner"
              and a["action"]=="export" and a["principal"]==e["principal"]
              and a["resource_id"]==e["resource_id"] and a["recipient"]==e["recipient"]
              and a["revision"]==e["workflow_revision"]==w["revision"]
              and a["policy_version"]==s.policy_version==e["policy_version"])
            if not ok: d.update(allowed=False,reason="export_lacks_subject_bound_current_approval_or_read")
        elif typ=="pay":
            _require(e,"workflow_id","payer","revision")
            w=s.workflows.get(e["workflow_id"]); a=w.get("approval") if w else None
            ok=bool(w and a and a["action"]=="pay" and e["revision"]==w["revision"]==a["revision"]
              and a["policy_version"]==s.policy_version
              and len({w["requester"],a["approver"],e["payer"]})==3)
            if not ok: d.update(allowed=False,reason="payment_lacks_current_approval_or_sod_violation")
        else: raise TraceError(f"unknown event type: {typ}")
        s.decisions.append(d)
    return {"policy_version":s.policy_version,"decisions":s.decisions,"allowed":all(x["allowed"] for x in s.decisions)}

if __name__=="__main__":
    import json,sys
    payload=json.load(sys.stdin)
    print(json.dumps(evaluate(payload["events"],payload.get("trusted_entitlements")),indent=2))
