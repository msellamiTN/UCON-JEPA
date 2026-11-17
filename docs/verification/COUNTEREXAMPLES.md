# UCON-JEPA formal counterexample catalogue

This file contains trace patterns that must be retained as adversarial test cases and, where expressible, TLA+ model-checking scenarios.

## CE-01 — stale authorization after revocation

Initial: capability active.  
Trace: request → authorize → effective revoke → reuse old certificate → external effect.  
Violation: effect occurs after effective revocation.  
Expected prevention: effect-boundary revalidation and linearizable revocation.

## CE-02 — concurrent overspending

Initial balance: 100.  
Two invocations each read 100 and request payment of 80. Both pass independent checks and both commit.  
Violation: final balance/invariant inconsistent with authorized serial semantics.  
Expected prevention: serializable execution, reservation, or atomic conditional update.

## CE-03 — JEPA false negative

Predictor estimates low risk; an unobserved external route leaks sensitive information.  
Violation: predictive assurance failed, but formal soundness is not thereby disproved.  
Expected response: retain formal gate; record false negative; improve observability/model/calibration.

## CE-04 — policy update race

Invocation authorized under policy version v. A stricter policy v+1 becomes effective before external effect. The old certificate is reused.  
Violation: stale policy.  
Expected prevention: version binding and reauthorization.

## CE-05 — external effect / crash window

External service accepts an irreversible operation. Gateway crashes before local governance commit. A blind retry duplicates effect.  
Expected prevention: durable write-ahead intent, idempotency key, reconciliation and compensation/containment.

## CE-06 — latent aliasing

Two governance states map to similar latent vectors while differing in a security-critical trust or classification attribute.  
Expected prevention: explicit authoritative attributes and security-feature prediction evaluation.

## CE-07 — behavioural adaptation

A revised policy prohibits tool A; the agent switches to tool B. Replaying the original fixed action sequence misses this adaptation.  
Expected prevention: keep fixed-action and agent-policy-induced counterfactual estimands separate.
