# G4 proof ledger and validation protocol

**Baseline:** G4 proof dossier v0.1  
**Date:** 2026-11-17  
**Purpose:** central traceability from claim to assumptions, proof, counterexample and implementation obligation.

## 1. Proof classification

- **Proved:** follows from definitions and stated formal premises without an unresolved nontrivial system premise.
- **Proved conditionally:** derivation is valid, but deployment interpretation depends on explicit assumptions that must be discharged.
- **Open:** theorem or assumption is not yet established.
- **Empirical:** requires measured evidence; no result is claimed until executed.

## 2. Theorem ledger

| ID | Claim | Key assumptions | Proof location | Status |
|---|---|---|---|---|
| THM-1 | External-effect authorization soundness | Complete mediation, certificate integrity, authoritative versions, evaluator correctness, linearizable gate | LaTeX §5 | Proved conditionally |
| THM-2 | No protected effect after effective revocation | THM-1 gate premises plus linearizable revocation | LaTeX §6 | Proved conditionally |
| THM-3 | No stale-policy external effect | Linearizable policy update and invalidation/revalidation | LaTeX §7 | Proved conditionally |
| THM-4 | Lifecycle validity preserved | Valid initial state; allowed atomic transitions | LaTeX §8 | Proved under transition discipline |
| THM-5 | Verified mutation consistency | Complete/correct effect evidence and correct verifier | LaTeX §9 | Proved conditionally |
| THM-6 | Every effect has durable intent | Write-ahead intent, append-only recoverable log | LaTeX §10 | Proved conditionally |
| THM-7 | Historical replay equivalence | Implementation/reference evaluator equality on recorded samples | LaTeX §11 | Proved |
| THM-8 | State invariant preserved under concurrency | Serializable execution, complete conflict abstraction, individual invariant preservation | LaTeX §12 | Proved conditionally |
| THM-9 | Enforcement soundness independent of predictor | Predictor cannot alter/bypass formal gate | LaTeX §13 | Proved conditionally |
| THM-10 | Selective risk bound | Pointwise risk domination by estimate plus delta | LaTeX §14 | Proved conditionally; key premise open |

## 3. Assumption discharge ledger

| ID | Assumption | Required implementation evidence |
|---|---|---|
| A-GATE | Every protected external effect crosses the gate | Tool inventory, gateway routing proof, bypass tests |
| A-CERT | Authorization certificate integrity | Cryptographic/binding design, replay and substitution tests |
| A-STATE | Gate reads authoritative state | State authority design, consistency tests |
| A-SEM | Evaluator implements policy semantics | Reference semantics, differential tests |
| A-ATOMIC | Gate/effect acceptance linearizable | Tool-specific protocol, concurrency model and tests |
| A-REV | Revocation linearizable | Replica/credential invalidation protocol and race tests |
| A-POL | Policy update invalidates/revalidates old decisions | Version protocol and update-race tests |
| A-EVIDENCE | Effect evidence complete and correct | Per-tool attestation/receipt/read-after-write specification |
| A-MUT | Commit verifier is the sole mutation authority | Code architecture and negative tests |
| A-LOG | Write-ahead durable intent | Crash-injection and recovery tests |
| A-SERIAL | Execution serializable for protected dependencies | Isolation proof, conflict graph/model checking |
| A-ISO | Predictor cannot bypass formal gate | Architecture and fault-injection tests |
| A-RISK | Pointwise risk domination | Requires a separate finite-sample statistical construction; currently open |

## 4. Mandatory counterexamples

| ID | Scenario | Expected finding |
|---|---|---|
| CE-01 | Revocation after authorization but before effect | Old certificate must be rejected |
| CE-02 | Concurrent overspending | Non-serializable execution violates balance invariant |
| CE-03 | Confident predictive false negative | Formal checks remain in force; predictive miss is recorded |
| CE-04 | Policy update with pending invocation | Stale-policy certificate cannot execute |
| CE-05 | Crash after remote acceptance before local commit | Recovery must reconcile, not blindly retry |
| CE-06 | Latent aliasing of trusted/untrusted states | Explicit security attributes prevent evaluator aliasing |
| CE-07 | Agent changes plan after policy update | Fixed-action estimate is not behavioural counterfactual |

## 5. Validation gates before G5

G4 must not be described as implementation-validated until the following are completed:

1. Compile the LaTeX document with the agreed TeX toolchain.
2. Check theorem labels/references and mathematical notation.
3. Independently review every proof for hidden premises and circularity.
4. Encode lifecycle, revocation, policy update, concurrency and crash cases in TLA+.
5. Run model checking on bounded instances and retain all counterexample traces.
6. Map each formal assumption to an implementation control and test.
7. Confirm the tool-class evidence taxonomy and external-effect linearization semantics.

## 6. Known limitations and corrections

- Authorization must be checked at the external-effect boundary, not inferred from later governance commit.
- Revocation request is distinct from effective revocation.
- A policy update cannot undo an already effective external operation.
- Mutation consistency is only as strong as effect observability.
- A version check is not automatically serializability.
- An append-only audit model does not automatically survive crash windows; write-ahead intent and reconciliation are required.
- Historical replay says nothing about unseen future workflows.
- JEPA latent prediction is not a probability distribution.
- Ordinary calibration metrics do not prove the selective-risk bound.
- THM-10 is conditional on pointwise risk domination; deriving that property is still open.

## 7. Research freeze rule

G4 is a **formal specification baseline**, not a verified software implementation. G5 may develop statistical risk control in parallel with formal model checking, but the manuscript must keep all conditional assumptions and open obligations visible.
