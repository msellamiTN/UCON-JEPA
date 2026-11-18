#!/usr/bin/env python3
"""G8.2 compositional held-out motif benchmark.

Train/validation episodes contain only the in-distribution unsafe motif. Test
episodes contain a distinct held-out action combination, while all constituent
actions occur in background traces. This is a controlled synthetic probe, not
JEPA and not a security guarantee.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import Counter
import numpy as np
from pilot_benchmark import fit_logistic, predict, metrics, FAMILIES, SEQUENCE_LEN, HORIZON

MOTIFS = {
    "iam": {"train": ["grant_role", "assume_admin"], "test": ["create_policy", "escalate_role"]},
    "data": {"train": ["read_sensitive", "export_sensitive"], "test": ["decrypt_sensitive", "share_external"]},
    "sod": {"train": ["request", "approve", "pay"], "test": ["change_vendor", "approve", "pay"]},
}
ACTIONS = {
    "iam": ["read", "list", "grant_role", "assume_admin", "export", "create_policy", "escalate_role"],
    "data": ["read_public", "read_sensitive", "mask", "export_sensitive", "share_external", "decrypt_sensitive"],
    "sod": ["request", "approve", "pay", "change_vendor", "audit"],
}

def contains_motif(seq, motif):
    return any(seq[i:i+len(motif)] == motif for i in range(len(seq)-len(motif)+1))

def generate_episode(seed, family, split, length=SEQUENCE_LEN):
    rng = random.Random(seed)
    motif = MOTIFS[family]["test" if split == "test" else "train"]
    trace = [rng.choice(ACTIONS[family]) for _ in range(length)]
    # Remove accidental appearances of the opposite split's motif from the trace.
    other = MOTIFS[family]["train" if split == "test" else "test"]
    for _ in range(20):
        if not contains_motif(trace, other): break
        trace = [rng.choice(ACTIONS[family]) for _ in range(length)]
    # Inject a guaranteed unsafe motif, fully within sequence length.
    pos = rng.randrange(length-len(motif)+1)
    trace[pos:pos+len(motif)] = motif
    # If random background created a cross-split motif, replace that occurrence
    # with a harmless action. Keep all constituent actions in the vocabulary.
    for _ in range(20):
        if not contains_motif(trace, other): break
        j=next(i for i in range(len(trace)-len(other)+1) if trace[i:i+len(other)] == other)
        trace[j]=ACTIONS[family][0]
    return trace

def motif_label(family, future):
    return int(contains_motif(future, MOTIFS[family]["train"]) or
               contains_motif(future, MOTIFS[family]["test"]))

def build(seed, episodes_per_family):
    rows=[]
    for fi,family in enumerate(FAMILIES):
        for ix in range(episodes_per_family):
            part="test" if ix%5==0 else ("validation" if ix%5==1 else "train")
            epseed=seed+fi*100000+ix
            trace=generate_episode(epseed,family,part)
            for t in range(SEQUENCE_LEN-HORIZON):
                fut=trace[t+1:t+1+HORIZON]
                rows.append({"family":family,"episode_id":f"{family}-{ix:05d}","partition":part,
                         "split_motif":part,"seed":epseed,"t":t,"prefix":trace[:t+1],
                         "target":motif_label(family,fut)})
    return rows

def feat(row):
    fam=row["family"]; prefix=row["prefix"]
    vocab=[a for f in FAMILIES for a in ACTIONS[f]]
    counts=Counter(prefix)
    return np.asarray([int(fam==f for f in FAMILIES)] + [counts[a] for a in vocab] +
                     [len(prefix)] + [int(prefix[-1]==a) for a in vocab], dtype=float)

def partition(rows,name):
    return [r for r in rows if r["partition"]==name]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--episodes-per-family",type=int,default=100)
    parser.add_argument("--out",type=Path,default=Path("pilot-output"))
    args=parser.parse_args()
    rows=build(args.seed,args.episodes_per_family)
    tr,va,te=(partition(rows,x) for x in ("train","validation","test"))
    Xtr=np.stack([feat(r) for r in tr]); ytr=np.array([r["target"] for r in tr])
    Xv=np.stack([feat(r) for r in va]); yv=np.array([r["target"] for r in va])
    Xte=np.stack([feat(r) for r in te]); yte=np.array([r["target"] for r in te])
    model=fit_logistic(Xtr,ytr); pv=predict(model,Xv); p=predict(model,Xte)
    thresholds=np.linspace(.05,.95,91)
    def f1(t):
        m=metrics(yv,pv,float(t)); return 2*m["precision"]*m["recall"]/(m["precision"]+m["recall"]+1e-12)
    threshold=float(max(thresholds,key=f1))
    res={"benchmark":"G8.2 compositional motif holdout v0.1","seed":args.seed,
         "episodes_per_family":args.episodes_per_family,"families":FAMILIES,"horizon":HORIZON,
         "split":"episode-level 60/20/20; test motif combination is held out from train/validation",
         "heldout_motifs":{f:MOTIFS[f]["test"] for f in FAMILIES},
         "train_windows":len(tr),"validation_windows":len(va),"test_windows":len(te),
         "train_positive":int(ytr.sum()),"validation_positive":int(yv.sum()),"test_positive":int(yte.sum()),
         "validation_threshold":threshold,"validation_metrics":metrics(yv,pv,threshold),
         "test_metrics":metrics(yte,p,threshold),
         "warning":"Synthetic controlled compositional stress test. Not a JEPA result, causal claim, or security guarantee."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"compositional_metrics.json").write_text(json.dumps(res,indent=2)+"\n")
    (args.out/"compositional_manifest.json").write_text(json.dumps({"seed":args.seed,"episodes_per_family":args.episodes_per_family,"split_by_episode":True,"heldout_test_motifs":{f:MOTIFS[f]["test"] for f in FAMILIES}},indent=2)+"\n")
    print(json.dumps(res,indent=2))

if __name__=="__main__": main()
