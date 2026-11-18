#!/usr/bin/env python3
"""Seeded randomized temporal UCON trace campaign for G8.3.2.

Episodes are synthetic, generated from explicit scenario templates plus random
benign padding. They test evaluator behavior and trace reproducibility; they
are not a calibrated model of operational security traffic.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from reference_semantics import evaluate

ENTITLEMENTS={"security_admin":["read","write","admin"]}
SCENARIOS=("benign_admin","revoked_grant","expired_grant","stale_policy_read",
           "stale_delegation","export_subject_substitution","sod_violation")

# A motif is a contiguous event-type pattern, not a scenario label. It is
# descriptive only: its presence does not imply that a trace is safe or unsafe.
MOTIF_EVENT_PATTERNS=(
    ("grant","assume_admin"),
    ("workflow_request","approve","export"),
    ("workflow_request","approve","pay"),
)

def has_motif_like_subsequence(events:list[dict])->bool:
    event_types=tuple(event.get("type") for event in events)
    return any(
        event_types[i:i+len(pattern)]==pattern
        for pattern in MOTIF_EVENT_PATTERNS
        for i in range(len(event_types)-len(pattern)+1)
    )

def _event(t, typ, **kw):
    return {"time":t,"sequence":0,"type":typ,**kw}

def _base_grant(t=1, gid="g", principal="alice", scope="read", expiry=100):
    return _event(t,"grant",grant_id=gid,actor="security_admin",principal=principal,
        scopes=[scope],tenant="tenant-a",expires_at=expiry,policy_version=1)

def _benign_padding(rng, start, count, prefix):
    events=[]; t=start
    for i in range(count):
        gid=f"{prefix}-pad-{i}"
        events.append(_base_grant(t,gid,principal=f"user-{i}",expiry=10000))
        t+=1
        events.append(_event(t,"read",grant_id=gid,principal=f"user-{i}",scope="read",
            tenant="tenant-a",resource_id=f"public-{i}",classification=0,clearance=0))
        t+=1
    return events,t

def make_episode(seed:int, episode_id:int, scenario:str|None=None)->dict:
    rng=random.Random(seed)
    scenario=scenario or rng.choice(SCENARIOS)
    if scenario not in SCENARIOS: raise ValueError(f"unknown scenario: {scenario}")
    events=[]; t=1
    # Variable-length harmless prefixes, including ordinary grant/read motifs.
    pad_count=rng.randint(0,5)
    pad,t=_benign_padding(rng,t,pad_count,f"ep{episode_id}")
    events.extend(pad)
    if scenario=="benign_admin":
        events.extend([_base_grant(t,"target","alice","admin",t+20),
            _event(t+1,"assume_admin",grant_id="target",principal="alice",scope="admin",tenant="tenant-a")])
    elif scenario=="revoked_grant":
        events.extend([_base_grant(t,"target","alice","admin",t+20),
            _event(t+1,"revoke",grant_id="target",actor="security_admin"),
            _event(t+2,"assume_admin",grant_id="target",principal="alice",scope="admin",tenant="tenant-a")])
    elif scenario=="expired_grant":
        events.extend([_base_grant(t,"target","alice","read",t+1),
            _event(t+1,"read",grant_id="target",principal="alice",scope="read",tenant="tenant-a",
                resource_id="sensitive-1",classification=1,clearance=2)])
        motif_like=False
    elif scenario=="stale_policy_read":
        events.extend([_base_grant(t,"target","alice","read",t+20),
            _event(t+1,"policy_update",actor="security_admin",version=2),
            _event(t+2,"read",grant_id="target",principal="alice",scope="read",tenant="tenant-a",
                resource_id="sensitive-1",classification=1,clearance=2)])
        motif_like=False
    elif scenario=="stale_delegation":
        events.extend([_base_grant(t,"target","alice","read",t+20),
            _event(t+1,"policy_update",version=2),
            _event(t+2,"delegate",source_grant_id="target",actor="alice",grant_id="child",principal="bob",
                scopes=["read"],expires_at=t+10)])
        motif_like=False
    elif scenario=="export_subject_substitution":
        events.extend([_base_grant(t,"target","alice","read",t+50),
            _event(t+1,"read",grant_id="target",principal="alice",scope="read",tenant="tenant-a",
                resource_id="resource-A",classification=1,clearance=2),
            _event(t+2,"workflow_request",workflow_id="wf",requester="alice",revision=1),
            _event(t+3,"approve",workflow_id="wf",approver="bob",revision=1,policy_version=1,
                action="export",principal="alice",resource_id="resource-A",recipient="owner"),
            _event(t+4,"export",workflow_id="wf",principal="alice",resource_id="resource-B",
                recipient="owner",workflow_revision=1,policy_version=1)])
        motif_like=True
    else: # SoD violation: requester and payer coincide.
        events.extend([_event(t,"workflow_request",workflow_id="wf",requester="alice",revision=1),
            _event(t+1,"approve",workflow_id="wf",approver="bob",revision=1,policy_version=1,
                action="pay",principal="alice",resource_id="invoice-1",recipient="vendor"),
            _event(t+2,"pay",workflow_id="wf",payer="alice",revision=1)])
        motif_like=True
    # Deterministic event ordering; times are strictly increasing.
    events.sort(key=lambda e:(e["time"],e["sequence"]))
    motif_like=has_motif_like_subsequence(events)
    result=evaluate(events,ENTITLEMENTS)
    first_bad=next((i for i,d in enumerate(result["decisions"]) if not d["allowed"]),None)
    prefix=events[:first_bad+1] if first_bad is not None else []
    payload=json.dumps(events,sort_keys=True,separators=(",",":")).encode()
    return {"episode_id":episode_id,"seed":seed,"scenario":scenario,"event_count":len(events),
        "motif_like_subsequence":motif_like,"unsafe":first_bad is not None,
        "events":events,"decisions":result["decisions"],
        "first_violation_index":first_bad,"minimal_failing_prefix":prefix,
        "trace_sha256":hashlib.sha256(payload).hexdigest()}

def campaign(seed:int=8302,episodes:int=1200)->dict:
    if episodes<1: raise ValueError("episodes must be positive")
    rng=random.Random(seed); records=[]
    for i in range(episodes):
        episode_seed=rng.getrandbits(64)
        records.append(make_episode(episode_seed,i))
    counts={}
    for x in records:
        k=x["scenario"]; counts.setdefault(k,{"episodes":0,"unsafe":0,"benign":0,"motif_like":0})
        counts[k]["episodes"]+=1
        counts[k]["unsafe" if x["unsafe"] else "benign"]+=1
        counts[k]["motif_like"]+=int(x["motif_like_subsequence"])
    return {"benchmark":"G8.3.2 randomized temporal traces","version":"0.1.0",
        "seed":seed,"episodes":episodes,"scenario_counts":counts,
        "unsafe_count":sum(x["unsafe"] for x in records),
        "benign_count":sum(not x["unsafe"] for x in records),
        "episodes_with_motif_like_subsequence":sum(x["motif_like_subsequence"] for x in records),
        "records":records,
        "limitations":"Synthetic scenario-template distribution; generator/evaluator share a policy model; no JEPA, causal or real-world security efficacy claim."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=8302)
    p.add_argument("--episodes",type=int,default=1200)
    p.add_argument("--out",default="artifacts/g8.3.2")
    a=p.parse_args()
    data=campaign(a.seed,a.episodes); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"randomized_temporal_traces.json").write_text(json.dumps(data,indent=2)+"\n")
    summary={k:v for k,v in data.items() if k!="records"}
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
