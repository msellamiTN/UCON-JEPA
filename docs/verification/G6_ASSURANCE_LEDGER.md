# G6 formal runtime-assurance ledger

**Baseline date:** 18 November 2026  
**Scope:** G6 formalisation and model-checking of G4 runtime-enforcement assumptions.

## Stage ledger

| Stage | Deliverable | Status |
|---|---|---|
| G6.1 | Threat model, TCB, trust boundaries, effect/revocation/policy-update semantics | Specification committed |
| G6.2 | Explicit labelled transition system and state variables | Not started |
| G6.3 | TLA+ model, safety invariants, temporal properties | Not started |
| G6.4 | TLC model-checking configurations, state-space results, counterexample traces | Not started |
| G6.5 | Refinement map to implementation controls and executable/property-based tests | Not started |
| G6.6 | LaTeX assurance dossier, proof review, final traceability update | Not started |

## G6.1 claims and status

| Claim | Current status | Evidence needed |
|---|---|---|
| JEPA is outside authorization TCB | Architectural requirement specified | Architecture review and fault-injection tests |
| UCON evaluator/gateway mediate protected effects | Assumption specified | Complete operation inventory, credential isolation, bypass tests |
| External effect is distinct from governance commit | Semantics specified | Per-tool effect-boundary contracts |
| Revocation is effective at a defined linearization point | Semantics specified | Distributed propagation/coordination protocol and race tests |
| Policy update is serialized with effect acceptance | Requirement specified | Version publication protocol and race tests |
| Mutable state is protected against TOCTOU | Alternatives specified; not selected per tool | CAS/transaction/locking protocol and concurrency tests |
| Ambiguous remote outcomes are not treated as success | Requirement specified | Idempotency and reconciliation evidence |
| JEPA cannot override formal denial | Non-authority property specified | Architecture, access-control, and fault-injection tests |

## Threat-model baseline

The initial model includes adversarial agent inputs, prompt injection, confused-deputy attempts, replay/duplication, plan mutation, revocation and policy-update races, stale state, asynchronous tools, network partitions, timeouts, crash-stop failures, and incomplete effect evidence.

It does not initially model arbitrary kernel compromise, malicious cloud-provider control planes, cryptographic breaks, arbitrary Byzantine replicas, covert channels, or side-channel leakage.

## Formal properties reserved for G6.2–G6.4

- SEC-1 Complete mediation.
- SEC-2 Authorization soundness at the effect boundary.
- SEC-3 Revocation safety after effective revocation.
- SEC-4 Policy freshness at effect acceptance.
- SEC-5 Invocation/action/principal/certificate binding.
- SEC-6 At-most-once acceptance where idempotency is supported.
- SEC-7 Commit only after verified effect evidence.
- SEC-8 JEPA non-authority.
- SEC-9 Unknown-outcome and recovery honesty.
- SEC-10 Resource invariant preservation under the selected concurrency protocol.

Safety and liveness will be analysed separately. Any atomicity introduced in the model must be mapped to a concrete implementation protocol; otherwise the resulting theorem is conditional only on an abstraction.

## Exit criteria and freeze rules

G6 is not complete until the transition system, TLA+ specification, bounded TLC runs, counterexample traces, implementation refinement map, and LaTeX dossier are committed.

A model that checks successfully is not an implementation proof. No model-checking result is claimed in this ledger until TLC has actually been run and its exact version, configuration, state count, invariants, and outputs are recorded.

A safety property that assumes an atomic effect gate must not be claimed for an API that cannot supply an equivalent linearizable protocol. Revocation cannot undo effects accepted before its effective point. A timeout with an ambiguous remote outcome is UNKNOWN, not failure and not success.
