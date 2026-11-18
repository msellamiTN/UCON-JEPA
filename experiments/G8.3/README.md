# G8.3.2 — Randomized temporal trace campaign

This increment creates deterministic, seeded, variable-length synthetic traces over the G8.3.1 sequential reference semantics.

## Scenarios
- Benign grant/read and authorized administrative assumption (motif-like but safe).
- Revoked and expired grant use.
- Policy-version invalidation before read or delegation.
- Export approval subject substitution.
- Separation-of-duties violation.

Each episode stores its seed, full event sequence, decisions, first violation, replayable failing prefix, and SHA-256 trace digest. The campaign also records scenario counts and aggregate benign/unsafe counts.

The `motif_like_subsequence` field is computed from the event sequence, not assigned from the scenario name. It detects these contiguous event-type patterns: `grant → assume_admin`, `workflow_request → approve → export`, and `workflow_request → approve → pay`. This is a structural descriptor only; it does not classify a trace as safe or unsafe.

## Reproduction

```bash
python -m unittest discover -s experiments/G8.3 -p "test_*.py" -v
python experiments/G8.3/randomized_temporal_traces.py --seed 8302 --episodes 1200 --out artifacts/g8.3.2
```

## Scientific limits
The scenario generator and evaluator share the G8.3.1 policy model. Agreement is therefore not an independent correctness proof. The scenario distribution is hand-designed and synthetic, not an empirical workload. The failing prefix is a shortest prefix by truncation at the first denied event, not a globally minimal subsequence. These data validate replay, variability, and selected policy edge cases; they do not establish security efficacy, predictive lead time, or JEPA benefit.
