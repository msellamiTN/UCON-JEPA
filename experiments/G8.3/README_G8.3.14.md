# G8.3.14 — Crash recovery under concurrent replay contention

## Objective

Combine G8.3.10's abrupt process-death fault injection with G8.3.13's independent-process contention tests. Validate the behavior of competing retries immediately after a writer process dies at either side of the transaction commit point.

## Scenarios

1. **Crash before commit while holding the write transaction:** a child process exits with `os._exit` after staging event and evaluation rows but before `COMMIT`. Four independent retry processes then race on the same signed event. Exactly one must commit; the other three must be rejected as replay. The recovered database must contain one event/evaluation pair and pass `PRAGMA integrity_check`.
2. **Crash after commit but before acknowledgement:** a child process exits immediately after `COMMIT`. Four independent retry processes then submit the same signed event. All four must be rejected as replay; the already committed event/evaluation pair must remain the only durable record.

## Interpretation

The pre-commit case checks that OS process termination releases SQLite's writer lock and recovery discards the uncommitted transaction before a competing writer proceeds. The post-commit case checks that commit durability dominates the missing client acknowledgement: retries observe the committed replay ID and cannot create another evaluation.

This is concrete fault-injection evidence for the tested single-machine SQLite/WAL runtime and four-process schedules. It does not prove every crash point, every filesystem or storage-controller failure, distributed transactions, or cross-host exactly-once delivery.
