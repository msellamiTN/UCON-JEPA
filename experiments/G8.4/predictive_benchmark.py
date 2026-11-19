"""Matched JEPA / supervised synthetic UCON prediction benchmark."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA
from stateful_trajectory_dataset import build_dataset

def scaler(x):
    m=x.mean(0); s=x.std(0); s[s<1e-8]=1.0
    return m,s

def transform(x,p): return (x-p[0])/p[1]

def ridge_fit(x,y,alpha=1.0):
    d=np.column_stack((np.ones(len(x)),x)); r=np.eye(d.shape[1])*alpha; r[0,0]=0
    return np.linalg.solve(d.T@d+r,d.T@y)

def ridge_predict(w,x): return np.column_stack((np.ones(len(x)),x))@w

def auc(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pos=int(y.sum()); neg=len(y)-pos
    if not pos or not neg: return None
    order=np.argsort(p,kind="stable"); ranks=np.empty(len(p)); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and p[order[j]]==p[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def ap(y,p):
    y=np.asarray(y,dtype=int)
    if not y.sum(): return None
    z=y[np.argsort(-p,kind="stable")]
    return float(((np.cumsum(z)/np.arange(1,len(z)+1))*z).sum()/z.sum())

def ece(y,p,bins=10):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),0,1); out=0.
    for lo in np.linspace(0,1,bins+1)[:-1]:
        hi=lo+1/bins; mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any(): out+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(out)

def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),0,1); q=p>=.5
    tp=int(((y==1)&q).sum()); fp=int(((y==0)&q).sum())
    fn=int(((y==1)&~q).sum()); tn=int(((y==0)&~q).sum())
    return {"n":len(y),"positive":int(y.sum()),"prevalence":float(y.mean()),
      "auroc":auc(y,p),"auprc":ap(y,p),"brier":float(np.mean((p-y)**2)),
      "ece_10":ece(y,p),"threshold":.5,"tp":tp,"fp":fp,"fn":fn,"tn":tn,
      "precision":tp/(tp+fp) if tp+fp else 0.,
      "recall":tp/(tp+fn) if tp+fn else 0.,
      "false_positive_rate":fp/(fp+tn) if fp+tn else 0.}

def run_seed(seed, episodes_per_family=200, epochs=100):
    ds=build_dataset(seed,episodes_per_family,8); tr,te=ds["train"],ds["test"]
    cp,aparam,fp=scaler(tr["context"]),scaler(tr["action"]),scaler(tr["future"])
    xc,xa=transform(tr["context"],cp),transform(tr["action"],aparam)
    yf=transform(tr["future"],fp)
    xt,at=transform(te["context"],cp),transform(te["action"],aparam)
    cfg=JEPAConfig(latent_dim=12,learning_rate=.008,epochs=epochs,batch_size=64,
        ema_tau=.99,variance_floor=.2,variance_weight=.1,security_weight=.5,seed=seed)
    model=LinearActionConditionedJEPA(xc.shape[1],xa.shape[1],yf.shape[1],
        tr["security"].shape[1],cfg).fit(xc,xa,yf,tr["security"])
    pred_j=np.clip(model.decode_security(model.predict_latent(xt,at)),0,1)
    pred_b=np.clip(ridge_predict(ridge_fit(np.column_stack((xc,xa)),tr["security"]),
        np.column_stack((xt,at))),0,1)
    pred_c=np.clip(ridge_predict(ridge_fit(xc,tr["security"]),xt),0,1)
    pred_a=np.clip(ridge_predict(ridge_fit(xa,tr["security"]),at),0,1)
    truth=te["security"]; methods={}
    for name,p in (("action_conditioned_jepa",pred_j),
                   ("supervised_context_plus_action",pred_b),
                   ("context_only",pred_c),("action_only",pred_a)):
        methods[name]={"violation":metrics(truth[:,0],p[:,0]),
                       "security_feature_mse":float(np.mean((truth-p)**2))}
    return {"seed":seed,"rows":{k:len(ds[k]["context"]) for k in ("train","validation","test")},
      "episodes":ds["manifest"]["episodes"],"methods":methods,
      "jepa":{"parameter_count":model.parameter_count(),
       "ema_target_distance":model.ema_target_distance(),
       "final_latent_loss":model.history.latent_loss[-1],
       "final_security_loss":model.history.security_loss[-1],
       "minimum_predicted_batch_std":min(model.history.predicted_std_min)},
      "policy_version":ds["manifest"]["policy_version"]}

def aggregate(runs):
    out={}
    for method in runs[0]["methods"]:
        out[method]={}
        for metric in ("auroc","auprc","brier","ece_10","precision","recall","false_positive_rate","security_feature_mse"):
            vals=[r["methods"][method]["security_feature_mse"] if metric=="security_feature_mse"
                  else r["methods"][method]["violation"][metric] for r in runs]
            vals=[float(x) for x in vals if x is not None]
            out[method][metric]={"mean":float(np.mean(vals)) if vals else None,
                "std_across_seeds":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,
                "seed_count":len(vals)}
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",type=int,nargs="+",default=[2026,2027,2028])
    p.add_argument("--episodes-per-family",type=int,default=200); p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--out",type=Path,default=Path("predictive-benchmark")); a=p.parse_args()
    runs=[run_seed(s,a.episodes_per_family,a.epochs) for s in a.seeds]
    result={"benchmark":"G8.4.4 matched synthetic stateful UCON prediction","seeds":a.seeds,
      "episodes_per_family":a.episodes_per_family,"epochs":a.epochs,"fixed_threshold":.5,
      "split":"episode-disjoint 60/20/20; scaling fitted on training rows only",
      "per_seed":runs,"aggregate":aggregate(runs),
      "limitations":"Synthetic data only. Across-seed SD is descriptive, not a population confidence interval. No production security or causal claim."}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"predictive_benchmark.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
