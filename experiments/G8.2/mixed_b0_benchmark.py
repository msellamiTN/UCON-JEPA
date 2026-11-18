#!/usr/bin/env python3
"""Mixed benign/unsafe episode evaluation for reactive UCON baseline B0."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from compositional_benchmark import ACTIONS, MOTIFS, contains_motif
from reactive_b0 import simulate_reactive_gate

def has_any_motif(family, trace):
    return any(contains_motif(trace, motif) for motif in MOTIFS[family].values())

def sanitize(trace, family, allowed_motif=None):
    """Remove disallowed unsafe motif occurrences while preserving trace length."""
    safe_token=ACTIONS[family][0]
    for _ in range(100):
        violations=[m for m in MOTIFS[family].values()
                   if (allowed_motif is None or m != allowed_motif) and contains_motif(trace,m)]
        if not violations:
            return trace
        motif=violations[0]
        i=next(i for i in range(len(trace)-len(motif)+1) if trace[i:i+len(motif)]==motif)
        trace[i+len(motif)-1]=safe_token
    raise RuntimeError(f"Could not sanitize family={family}")

def generate(seed, family, ix, length=12):
    rng=random.Random(seed)
    trace=[rng.choice(ACTIONS[family]) for _ in range(length)]
    unsafe=(ix//5)%2==1
    if not unsafe:
        trace=sanitize(trace,family)
        assert not has_any_motif(family,trace), "benign trace contains motif"
        return trace,False
    motif=MOTIFS[family]["test" if (ix//5)%4 in (1,3) else "train"]
    pos=rng.randrange(length-len(motif)+1)
    trace[pos:pos+len(motif)]=motif
    trace=sanitize(trace,family,allowed_motif=motif)
    assert contains_motif(trace,motif), "injected unsafe motif lost during sanitization"
    assert not any(contains_motif(trace,m) for m in MOTIFS[family].values() if m!=motif), "cross-motif contamination"
    return trace,True

def wilson(k,n,z=1.959963984540054):
    if n==0:return None
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/den
    return [max(0.0,center-half),min(1.0,center+half)]

def blank_counts():
    return {"n_test":0,"benign":0,"unsafe":0,"benign_intervened":0,
            "unsafe_intercepted":0,"total_blocked":0}

def run_one_seed(seed,episodes_per_family):
    results={}
    for family in ACTIONS:
        counts=blank_counts()
        for ix in range(episodes_per_family):
            if ix%5!=0:continue
            epseed=seed+list(ACTIONS).index(family)*100000+ix
            trace,label=generate(epseed,family,ix)
            counts["n_test"]+=1
            counts["benign"]+=int(not label)
            counts["unsafe"]+=int(label)
            gate=simulate_reactive_gate(trace,family)
            intervened=gate["blocked_count"]>0
            counts["total_blocked"]+=gate["blocked_count"]
            counts["benign_intervened"]+=int((not label) and intervened)
            counts["unsafe_intercepted"]+=int(label and intervened)
        nb,nu=counts["benign"],counts["unsafe"]
        results[family]={**counts,
          "unsafe_prevention_recall":counts["unsafe_intercepted"]/nu if nu else None,
          "false_intervention_rate":counts["benign_intervened"]/nb if nb else None}
    return results

def evaluate(seed=2026,episodes_per_family=100,replicates=1):
    aggregate={f:blank_counts() for f in ACTIONS}
    replicate_results=[]
    for rep in range(replicates):
        rep_seed=seed+rep*100003
        res=run_one_seed(rep_seed,episodes_per_family)
        replicate_results.append({"replicate":rep,"seed":rep_seed,"by_family":res})
        for family,counts in res.items():
            for k in aggregate[family]:
                aggregate[family][k]+=counts[k]
    summary={}
    pooled=blank_counts()
    for family,counts in aggregate.items():
        nb,nu=counts["benign"],counts["unsafe"]
        summary[family]={**counts,
          "unsafe_prevention_recall":counts["unsafe_intercepted"]/nu if nu else None,
          "unsafe_prevention_recall_95ci_wilson":wilson(counts["unsafe_intercepted"],nu),
          "false_intervention_rate":counts["benign_intervened"]/nb if nb else None,
          "false_intervention_rate_95ci_wilson":wilson(counts["benign_intervened"],nb),
          "mean_blocked_actions_per_episode":counts["total_blocked"]/counts["n_test"] if counts["n_test"] else None}
        for k in pooled: pooled[k]+=counts[k]
    nb,nu=pooled["benign"],pooled["unsafe"]
    pooled_summary={**pooled,
      "unsafe_prevention_recall":pooled["unsafe_intercepted"]/nu if nu else None,
      "unsafe_prevention_recall_95ci_wilson":wilson(pooled["unsafe_intercepted"],nu),
      "false_intervention_rate":pooled["benign_intervened"]/nb if nb else None,
      "false_intervention_rate_95ci_wilson":wilson(pooled["benign_intervened"],nb),
      "mean_blocked_actions_per_episode":pooled["total_blocked"]/pooled["n_test"] if pooled["n_test"] else None}
    return {"benchmark":"G8.2 mixed benign/unsafe reactive B0 cohort",
      "seed_base":seed,"replicates":replicates,"episodes_per_family_per_replicate":episodes_per_family,
      "total_test_episodes":pooled["n_test"],
      "test_episode_selection":"episode index modulo 5; balanced benign/unsafe in each replicate",
      "unsafe_motif_types":"in-distribution and held-out variants",
      "metrics_by_family":summary,"pooled_metrics":pooled_summary,"replicate_results":replicate_results,
      "limitations":"Synthetic motif labels and deterministic reactive gate only; no real policy context or external side effects.",
      "overall_warning":"Benchmark mechanics validation, not real-world security efficacy."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--episodes-per-family",type=int,default=100)
    p.add_argument("--replicates",type=int,default=1)
    p.add_argument("--out",type=Path,default=Path("pilot-output"))
    a=p.parse_args()
    result=evaluate(a.seed,a.episodes_per_family,a.replicates)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"mixed_b0_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
