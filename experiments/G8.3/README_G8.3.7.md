# G8.3.7 — Authenticated evaluation pipeline

## Purpose
Make provenance verification an explicit, fail-closed precondition for UCON evaluation. The integration entry point accepts signed envelopes, verifies every envelope and replay identifier, then passes only verified event bodies to the existing policy evaluator.

## Contract
`evaluate_signed_trace(...)` returns:
- policy decisions
- accepted event count
- candidate replay-ID set for the caller to persist
- explicit provenance-verified marker

Any signature, key, replay, envelope, or event-schema failure raises an exception and returns no evaluation result.

## Transaction boundary
The returned replay-ID set is a candidate commit token. The caller must atomically persist it with the accepted evaluation/audit record. A production service must not acknowledge the event before this transaction commits. The library does not pretend that an in-memory set provides distributed exactly-once processing.

## Validation
Adversarial tests cover unsigned inputs, tampering, successful grant/read evaluation, replay, and schema failure. A denied authorization is a valid policy decision and is still recorded as a processed, authenticated event; provenance authenticity is distinct from authorization success.
