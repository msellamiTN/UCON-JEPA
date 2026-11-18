# G8.3.4 — Independent temporal state-machine differential audit

## Purpose

Add a second executable semantics for the G8.3 temporal policy model and compare its event-level decisions with the reference evaluator over randomized, well-formed traces.

## Design

- The oracle is implemented in a separate module with its own state representation and transition decomposition.
- It does not call or import the reference evaluator from its `oracle()` function.
- The campaign generates traces across grant authorization, revocation, expiry, policy-version invalidation, delegation, admin assumption, data reads, approval/export binding, workflow amendment, payment separation-of-duties, and duplicate workflow creation.
- Randomized boundary choices include authorized/unauthorized actors, valid/invalid scopes, matching/mismatching principals, expiry boundaries, scope expansion, resource/recipient substitution, stale approvals, and repeated workflow identifiers.
- The CI campaign uses seed `8304` and 5,000 episodes; unit tests use a smaller deterministic sample.

## Interpretation

A zero-mismatch result is evidence that two separately coded evaluators agree on the generated trace distribution. It does not establish that the shared written policy specification is complete, coherent, conflict-free, or appropriate for an operational organization. Both implementations can share a conceptual defect. Independent domain review and property-based invariant analysis remain separate gates.

## Dependency and integration

This increment is stacked on PR #9, which hardens the reference evaluator's input schema. The combined PR validation must run against that strict-schema branch before PR #9 is merged and PR #10 is retargeted to `main`.

## Reproduction

```bash
python -m unittest discover -s experiments/G8.3 -p "test_*.py" -v
python experiments/G8.3/independent_temporal_oracle.py --seed 8304 --episodes 5000 --out artifacts/g8.3.4
```
