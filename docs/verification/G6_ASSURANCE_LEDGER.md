# G6 formal runtime-assurance ledger

**Baseline date:** 18 November 2026  
**Scope:** G6 formalisation and model-checking of G4 runtime-enforcement assumptions.

## Stage ledger

| Stage | Deliverable | Status |
|---|---|---|
| G6.1 | Threat model, TCB, trust boundaries, effect/revocation/policy-update semantics | Specification committed |
| G6.2 | Explicit labelled transition system and state variables | Specification committed |
| G6.3 | TLA+ model, safety invariants, temporal properties | Core model encoded |
| G6.4 | TLC model-checking configurations, state-space results, counterexample traces | Bounded check passed; four negative controls detected |
| G6.5 | Refinement map to implementation controls and executable/property-based tests | Abstract reference tests pass; production refinement open |
| G6.6 | LaTeX assurance dossier, proof review, final traceability update | Dossier and claim ledger committed; LaTeX CI pending |

## G6.6 proof review and assurance dossier

The full LaTeX dossier is paper/formal/g6_runtime_assurance.tex. Claim-by-claim proof status and open obligations are in docs/verification/G6.6_PROOF_REVIEW.md. The dossier records the abstract proofs, TLC evidence, negative controls, Python tests, and deployment limitations. CI compilation was added; no LaTeX pass is claimed until the current workflow finishes.

## G6.5 refinement and contract tests

The refinement matrix maps complete mediation, identity binding, certificate integrity, authoritative state, evaluator correctness, effect linearizability, revocation, policy updates, evidence, durable intent, serializability, predictor isolation, and statistical scope to implementation controls and tests. Ten abstract runtime contract tests passed in workflow run 44199165327. Full report: docs/verification/G6.5_REFINEMENT_AND_TESTS.md. The tests execute a reference harness only; no production adapter has yet been selected or validated.

## G6.4 TLC results

Workflow run 44199003346 passed. TLC version 2026.11.15.002146, Java Temurin 17.0.21+9, one worker: 13,441 states generated, 1,664 distinct states, zero remaining. All seven configured invariants passed. Mutants M1–M4 each triggered the expected invariant violation (revocation, policy freshness, UCON bypass, and commit-before-verification). Full record: docs/verification/G6.4_TLC_RESULTS.md. This is a finite abstract-model result only; implementation refinement and richer protocol modelling remain open.

## G6.3 TLA+ model status

Committed formal/tla/G6_RuntimeEnforcement.tla, the positive TLC configuration, and four mutant configurations. The module parsed and the finite positive model was checked by TLC in workflow run 44199003346: 13,441 states generated, 1,664 distinct states, zero remaining, and all seven configured invariants passed. Mutants M1–M4 detected their expected invariant violations. Full execution record: docs/verification/G6.4_TLC_RESULTS.md. The check is bounded to the committed finite abstraction.

## G6.2 transition-system status

The LTS distinguishes Request, Authorize, predictive restriction, Prepare, GateCheck, Dispatch, Accept, Observe, Verify, Commit, revocation, policy publication, state mutation, crash, and recovery. Safety and liveness are separated. Tool-specific refinement obligations and ten counterexample scenarios are recorded in docs/research/G6.2_RUNTIME_TRANSITION_SYSTEM.md. Its synchronous acceptance kernel is encoded and bounded-checked in G6.3–G6.4; richer asynchronous and resource-concurrency semantics remain open.

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
