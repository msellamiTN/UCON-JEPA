#!/usr/bin/env python3
"""Deterministic synthetic UCON workflow pilot (G8.2).

This pilot deliberately implements data/label validation and a matched-feature
supervised baseline only. It is NOT a JEPA implementation or security proof.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import Counter
import numpy as np

ACTIONS = {
    "iam": ["read", "list", "grant_role", "assume_admin", "export"],
    "data": ["read_public", "read_sensitive", "mask", "export_sensitive", "share_external"],
    "sod": ["request", "approve", "pay", "change_vendor", "audit"],
}
FAMILIES = list(ACTIONS)
SEQUENCE_LEN = 12
HORIZON = 3

def episode(seed: int, family: str, n: int = SEQUENCE_LEN):
    rng = random.Random(seed)
    actions = ACTIONS[family]
    trace = [rng.choice(actions) for _ in range(n)]
    # Inject a controlled, independently defined unsafe pattern in some episodes.
    unsafe = rng.random() < 0.35
    if unsafe:
        if family == "iam":
            trace[rng.randrange(n)] = "grant_role"
            trace[rng.randrange(n)] = "assume_admin"
        elif family == "data":
            trace[rng.randrange(n)] = "read_sensitive"
            trace[rng.randrange(n)] = "export_sensitive"
        else:
            # request -> approve -> pay, with no independent approver
            i = rng.randrange(n - 2)
            trace[i:i+3] = ["request", "approve", "pay"]
    return trace

def violation(family: str, seq: list[str]) -> bool:
    if family == "iam":
        return "grant_role" in seq and "assume_admin" in seq
    if family == "data":
        return "read_sensitive" in seq and "export_sensitive" in seq
    # This synthetic rule is deliberately simplified: sequence presence is the label.
    return all(x in seq for x in ("request", "approve", "pay"))

def make_dataset(seed: int, episodes_per_family: int):
    rows = []
    for f_idx, family in enumerate(FAMILIES):
        for i in range(episodes_per_family):
            s = seed + f_idx * 100_000 + i
            trace = episode(s, family)
            for t in range(SEQUENCE_LEN - HORIZON):
                prefix = trace[:t+1]
                target = int(violation(family, trace[t+1:t+1+HORIZON]))
                rows.append({"family": family, "episode_id": f"{family}-{i:05d}",
                         "seed": s, "t": t, "prefix": prefix, "target": target})
    return rows

def features(row):
    family = row["family"]
    prefix = row["prefix"]
    vocab = [a for f in FAMILIES for a in ACTIONS[f]]
    # Family indicator + action-count vector + prefix length + last-action one-hot.
    out = [int(family == f) for f in FAMILIES]
    counts = Counter(prefix)
    out += [counts[a] for a in vocab]
    out += [len(prefix)]
    out += [int(prefix[-1] == a) for a in vocab]
    return np.asarray(out, dtype=float)

def split(rows):
    # Episode-level split, so windows from one episode never cross partitions.
    ids = sorted(set(r["episode_id"] for r in rows))
    train_ids, test_ids = set(), set()
    for eid in ids:
        # Deterministic family-balanced split via episode index encoded in ID.
        ix = int(eid.split("-")[-1])
        (test_ids if ix % 5 == 0 else train_ids).add(eid)
    return ([r for r in rows if r["episode_id"] in train_ids],
            [r for r in rows if r["episode_id"] in test_ids])

def fit_logistic(X, y, lr=0.08, epochs=350, l2=1e-3):
    # Standardize using training data only; constant columns get unit scale.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Z = (X - mean) / std
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(epochs):
        p = 1/(1+np.exp(-np.clip(Z@w+b, -30, 30)))
        w -= lr * (Z.T@(p-y)/len(y) + l2*w)
        b -= lr * float(np.mean(p-y))
    return mean, std, w, b

def predict(model, X):
    mean, std, w, b = model
    Z = (X-mean)/std
    return 1/(1+np.exp(-np.clip(Z@w+b, -30, 30)))

def metrics(y, p):
    pred = p >= 0.5
    tp=int(np.sum((y==1)&pred)); fp=int(np.sum((y==0)&pred))
    fn=int(np.sum((y==1)&(~pred))); tn=int(np.sum((y==0)&(~pred)))
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    acc=(tp+tn)/len(y) if len(y) else 0.0
    # Average precision from ranked scores (ties grouped by stable sort).
    order=np.argsort(-p, kind="stable")
    ys=y[order]; cum=np.cumsum(ys)
    ap=float(np.sum((cum/np.arange(1,len(y)+1))*ys)/max(1,int(y.sum())))
    return {"n":len(y),"positive":int(y.sum()),"prevalence":float(y.mean()),
            "accuracy":acc,"precision":precision,"recall":recall,"auprc":ap,
            "tp":tp,"fp":fp,"fn":fn,"tn":tn}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--episodes-per-family", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("pilot-output"))
    args=ap.parse_args()
    rows=make_dataset(args.seed,args.episodes_per_family)
    train,test=split(rows)
    Xtr=np.stack([features(r) for r in train]); ytr=np.array([r["target"] for r in train])
    Xte=np.stack([features(r) for r in test]); yte=np.array([r["target"] for r in test])
    model=fit_logistic(Xtr,ytr); p=predict(model,Xte)
    result={"benchmark":"G8.2 pilot v0.1","seed":args.seed,"episodes_per_family":args.episodes_per_family,
        "families":FAMILIES,"horizon":HORIZON,"split":"episode-level deterministic 80/20 (episode index divisible by 5 -> test)",
        "train_windows":len(train),"test_windows":len(test),"metrics":metrics(yte,p),
        "disclaimer":"Pilot label/pipeline and supervised baseline only; not JEPA, not a causal evaluation, not a security guarantee."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"pilot_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.out/"pilot_manifest.json").write_text(json.dumps({"seed":args.seed,"episodes_per_family":args.episodes_per_family,"python_numpy_required":"numpy","label_horizon":HORIZON,"trace_length":SEQUENCE_LEN},indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
