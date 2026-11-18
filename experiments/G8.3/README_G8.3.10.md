# G8.3.10 — Crash consistency, recovery, and safety boundary

## Objective

Validate the durable replay ledger at process-failure boundaries, not only at Python exception boundaries. The tests terminate a child interpreter with `os._exit` at two explicit points: after all SQL writes but before `COMMIT`, and after `COMMIT` but before the processor returns an acknowledgement.

## State model

For a submitted authenticated trace with event-ID set (E), persistent state is ((P, A)), where (P) is the set of committed event IDs and (A) is the set of committed evaluation records.

The processor transition is:

1. **Idle**: persistent state is ((P,A)).
2. **TransactionOpen**: acquire SQLite's `BEGIN IMMEDIATE` write transaction; read (P).
3. **Verified**: verify signatures, issuer/key binding, replay exclusion, event schema, and temporal evaluation.
4. **Prepared**: insert all IDs in (E) and one evaluation record inside the same uncommitted transaction.
5. **Committed**: execute SQLite `COMMIT`; state becomes ((P \cup E, A \cup \{a\})).
6. **Acknowledged**: return the committed result to the caller.

A failure in states 2–4 must not change persistent state. A failure after state 5 but before state 6 leaves the commit durable; retry is rejected by the replay ledger rather than executing the trace again.

## Safety properties and assumptions

Under a single SQLite database, supported local filesystem, SQLite WAL recovery, and the configured `synchronous=FULL` durability contract:

- **S1 — Atomic acceptance:** one trace's event IDs and evaluation record are committed in one transaction; no proper subset is visible after recovery.
- **S2 — At-most-once event-ID acceptance:** the primary key on `processed_events.event_id`, transaction serialization, and replay check prevent a second committed acceptance of an already committed ID.
- **S3 — Commit-before-ack:** the API returns a successful result only after `COMMIT` completes.
- **S4 — Lost-ack recovery:** if the process dies after commit and before return, the event remains committed and a retry is rejected as replay.
- **S5 — Authenticity gate:** only traces accepted by the signature/schema/evaluation pipeline reach the insert/commit sequence.
- **S6 — Denial is an evaluated outcome:** an authentic but unauthorized event may have a deny decision and still be committed to the audit/replay ledger.

These are conditional implementation-level properties. They do not prove policy completeness, coherence, conflict-freedom, correctness of the UCON specification, physical durability against every storage-controller failure, or distributed exactly-once delivery.

## Fault-injection evidence

The tests launch separate Python processes and use `os._exit` to bypass exception handlers and normal connection cleanup:

- **Crash before commit:** after inserts but before `COMMIT`; reopen database, require zero event/evaluation rows and `PRAGMA integrity_check = ok`; then process the same envelope successfully.
- **Crash after commit, before acknowledgement:** terminate immediately after `COMMIT`; reopen database, require one event and one evaluation; retry the same envelope and require replay rejection with no additional rows.
- **Restart persistence:** reopen a committed database, verify replay history and evaluation pairing, and reject duplicate delivery.
- Retain the concurrent duplicate-delivery and late-schema-failure regressions.

The private `_before_commit` and `_after_commit` methods are deterministic fault-injection seams. Production subclasses must not override them to alter transaction semantics.

## Explicit non-claims

This increment does not establish distributed consensus, cross-region exactly-once processing, safe multi-host access over arbitrary network filesystems, backup/restore correctness, retention safety, external message-broker acknowledgement, or transactional outbox delivery. Those require a separate deployment architecture and fault model.
