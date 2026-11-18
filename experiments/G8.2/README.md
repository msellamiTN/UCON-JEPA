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
