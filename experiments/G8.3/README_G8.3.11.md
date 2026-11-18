# G8.3.11 — Finite-state crash-safety model

## Purpose

G8.3.10 added process-level fault injection. This increment makes the abstract transaction/recovery protocol explicit and checks its finite-state safety invariants independently of the SQLite implementation.

## Model

`formal/tla/G8_3_11_CrashConsistency.tla` models one event ID and one evaluation ID through:

`IDLE → TXN → PREPARED → COMMITTED → ACKNOWLEDGED`

with crash branches before commit and after commit/before acknowledgement. Recovery after a pre-commit crash permits retry. Recovery after a post-commit crash rejects the event ID as replay.

The model contains a deliberately unsafe negative-control action, `UnsafeCommitEventOnly`, which commits an event ID without its evaluation record. The mutant configuration is expected to violate `AtomicEventEvaluation`.

## Invariants

- `TypeOK`: every state variable remains in its declared finite domain.
- `AtomicEventEvaluation`: the event ID and its evaluation record are either both durable or both absent.
- `AtMostOnceCommit`: at most one commit occurs for the single event ID.
- `AcknowledgementRequiresCommit`: acknowledgement implies both durable records exist.
- `ReplayRejectionRequiresPriorCommit`: replay rejection is possible only after the original commit.

## Executable evidence

`experiments/G8.3/crash_consistency_model.py` is a separate finite-state breadth-first explorer used by the existing G8.3 transactional test command. It explores all reachable states in the one-event abstraction and checks the invariants. A negative-control test enables the event-only commit mutant and requires the atomicity invariant to fail.

The TLA+ source and TLC configurations are committed for independent TLC execution. Unless a TLC run is attached to this revision, the TLA+ model itself is **not** labelled TLC-checked. The Python explorer's exhaustive result is bounded to the finite abstraction described above.

## Boundary

This model abstracts away SQLite internals, filesystem ordering, WAL page mechanics, multiple event IDs, multiple concurrent clients, key management, broker delivery, backup/restore, and distributed replicas. G8.3.10 process-level tests remain the implementation evidence. Neither artifact establishes distributed exactly-once processing or universal physical durability.
