# G8.3.8 — Transactional replay ledger

## Objective
Close the gap between in-memory replay checking and durable processing. The SQLite reference processor uses `BEGIN IMMEDIATE` to serialize writers, reads the existing replay set, verifies the entire signed trace, evaluates it, and commits all event IDs plus the evaluation record in one transaction.

## Guarantees within the stated SQLite deployment model
- A previously committed event ID cannot be accepted again.
- A multi-event trace with a late provenance or policy-schema failure commits neither earlier IDs nor an evaluation record.
- An authenticated but unauthorized event is still recorded with its deny decision; authenticity and authorization are separate.
- Replay IDs and the corresponding evaluation result commit atomically.

## Concurrency regression

A two-writer race test submits the same signed event through separate SQLite connections at the same time. Under the stated single-database SQLite model, exactly one submission may commit; the second must observe the committed replay ID and fail without creating a second evaluation.

## Deployment boundary
This is a single-database transactional prototype. It does not provide distributed consensus, cross-region exactly-once delivery, or safe concurrent operation over arbitrary network filesystems. Production deployment needs a transactional database with documented isolation semantics, durable backup/recovery, retention controls, and an outbox/acknowledgement protocol. Event acknowledgement must occur only after commit.
