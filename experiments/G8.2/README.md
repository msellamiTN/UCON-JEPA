# G8.2 Synthetic Pilot v0.1

This is a **pipeline smoke pilot**, not the full G8.2 benchmark.

## What it tests
- Deterministic episode generation across IAM, sensitive-data, and approval/payment workflow families.
- Ordered, contiguous rule-motif labels for unsafe patterns in the next 3 actions.
- Episode-level deterministic split (windows from a single episode cannot leak across train/test).
- A plain supervised logistic baseline using prefix features.
- JSON metrics/manifests written to disk.


## Important limits
- Synthetic labels are simplified and imperfect proxies.
- Split is episode-level, **not yet a compositional held-out action-combination split**. Threshold is tuned on the validation partition only.
- No formal/reactive baseline, JEPA, probabilistic baseline, calibration, multi-seed CI, or intervention simulation yet.
- This smoke pilot cannot establish predictive security benefit.


## Run
```bash
python -m pip install numpy
python experiments/G8.2/pilot_benchmark.py --seed 2026 --episodes-per-family 500 --out experiments/G8.2/pilot-output
```


Expected outputs: `pilot_metrics.json`, `pilot_manifest.json`.

## Compositional held-out benchmark v0.1

Run:
```bash
python experiments/G8.2/compositional_benchmark.py --seed 2026 --episodes-per-family 100 --out experiments/G8.2/compositional-output
```

The episode-index split is 60/20/20. Training and validation episodes receive
one unsafe motif variant; test episodes receive a different held-out motif
variant. Each constituent action remains present in training background traces.
The held-out pair/triple is explicitly audited by unit tests for leakage.

Held-out motif combinations:
- IAM: `create_policy -> escalate_role`
- Sensitive data: `decrypt_sensitive -> share_external`
- SoD: `change_vendor -> approve -> pay`

This remains a controlled synthetic compositional probe: it does not model real
policy semantics, JEPA, causality, or a security guarantee. A negative or weak
result is scientifically acceptable.
