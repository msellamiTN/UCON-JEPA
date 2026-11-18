# G8.3.13 — Multi-process SQLite refinement evidence

## Objective

Test the G8.3.12 abstract single-writer lock against the actual SQLite implementation using independent operating-system processes, each with its own `SQLiteEventProcessor` instance and database connection.

## Tests

The tests use a filesystem start gate: each child process opens its own processor and signals readiness; the parent releases all workers together.

1. **Duplicate event race:** four independent processes submit the same valid signed event ID. Exactly one may commit; the other three must be rejected as replay. The database must contain one processed event and one evaluation, and `PRAGMA integrity_check` must return `ok`.
2. **Distinct event race:** four independent processes submit four distinct valid signed event IDs. All four should commit, yielding four processed events and four evaluations with an intact database.

These tests complement the existing thread-level duplicate race and abrupt-crash/restart tests.

## Refinement interpretation

The abstract G8.3.12 lock owner corresponds to SQLite's `BEGIN IMMEDIATE` writer serialization in this concrete implementation. The tests provide implementation evidence for the tested local SQLite/WAL runtime and the tested four-process schedules. They do not prove SQLite's implementation formally, exhaust all OS scheduling interleavings, or establish multi-host/distributed exactly-once behavior.

## Deployment boundary

The evidence assumes one local SQLite database on a supported filesystem, WAL mode, `synchronous=FULL`, and a single machine. It does not validate arbitrary network filesystems, cross-host writers, storage-controller failure, backup/restore, or external message-broker acknowledgement.
