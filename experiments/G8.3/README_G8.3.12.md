# G8.3.12 — Multi-writer replay serialization model

## Objective

Generalize G8.3.11's single-event crash model to two concurrent submitters sharing one durable replay ledger. The abstraction models the serialization property relied upon by SQLite `BEGIN IMMEDIATE`: one writer owns the transaction boundary at a time, and a later writer must observe already committed event IDs before acceptance.

## State and transitions

The TLA+ model tracks each writer's phase, a single transaction lock owner, staged event/evaluation pair, and the set of durable pairs. It models lock acquisition, prepare, commit, acknowledgement, replay rejection, crash before commit, crash after commit but before acknowledgement, and retries following either recovery outcome.

Two configurations are checked:
- **Duplicate event ID:** both writers submit the same event ID with distinct evaluation IDs. At most one pair may be committed; the other writer must be rejected as replay.
- **Distinct event IDs:** both writers submit different event IDs. Both may commit, in either serialized order.

## Safety invariants

- `TypeOK`: state variables remain in their finite domains.
- `NoDuplicateEventCommit`: no event ID occurs in more than one durable event/evaluation pair.
- `CommitRecordWellFormed`: every durable pair corresponds to one declared writer request.
- `AcknowledgementRequiresDurability`: an acknowledged request has its durable pair.
- `ReplayRejectionRequiresPriorCommit`: replay rejection implies the event ID is already durable.
- `LockConsistency`: staged state exists only for the lock owner and matches that owner's request.

A negative-control action deliberately bypasses replay rejection and attempts a second evaluation commit for an already durable event ID. The mutant TLC configuration must produce a counterexample to `NoDuplicateEventCommit`.

## Assumptions and limitations

The model assumes a single linearizable transaction lock, authenticated request identities, unique evaluation IDs, and a crash that releases the lock while preserving already committed pairs. It abstracts away SQLite's page/WAL implementation, scheduler fairness, lock timeouts, filesystem and power-loss behavior, process coordination outside the database, and multi-host/distributed database semantics. TLC results are finite-state model-checking evidence under these assumptions, not a proof of SQLite's implementation or distributed exactly-once delivery.
