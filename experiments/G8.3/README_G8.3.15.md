# G8.3.15 — Transactional outbox and delivery semantics

## Objective

Extend the authenticated transactional replay ledger so that a committed UCON
decision always has a durable delivery intent, then specify and test the
dispatcher and consumer failure boundaries.

## Atomic decision + delivery intent

`SQLiteEventProcessor.process` now commits the following in one SQLite
transaction:

1. authenticated event replay identifiers;
2. the temporal evaluation result;
3. one outbox row per accepted event, keyed by
   `<event_id>:ucon-decision:v1`.

A crash before commit leaves none of these records. A crash after commit leaves
the decision and its outbox intent together. The outbox payload carries the
schema version, event ID, evaluation ID, and evaluation result.

## Dispatcher protocol

`SQLiteOutboxDispatcher` claims one pending item in a `BEGIN IMMEDIATE`
transaction and assigns a unique lease token and expiry. Expired in-flight
leases can be reclaimed. Acknowledgement and retry require the current lease
token, so a stale worker cannot overwrite a newer worker's state.

The dispatcher deliberately does **not** claim atomicity between an external
broker send and the SQLite acknowledgement. If a worker sends successfully and
crashes before `mark_delivered`, the lease expires and the message is sent
again. Therefore the dispatcher provides **at-least-once delivery attempts**,
assuming workers continue retrying and the destination eventually becomes
available. A permanently failing destination can prevent delivery.

## Consumer protocol

`SQLiteIdempotentConsumer` stores the inbox idempotency key and the
demonstration business-effect row in one local SQLite transaction. Repeated
delivery of the same key does not create a second effect. This gives an
effectively-once local database effect within this transaction boundary.

It does not make arbitrary external side effects exactly-once. If a consumer
calls a payment provider, email service, or another system outside the local
transaction, that system must itself support idempotency or participate in a
shared transactional protocol. Otherwise the same send/ack crash window exists.

## Validation scenarios

`test_transactional_outbox.py` covers:

- atomic decision + outbox insertion;
- fail-closed rollback on invalid trace;
- lease acquisition and token-bound acknowledgement;
- expired lease reclamation and stale-worker rejection;
- explicit retry after dispatch failure;
- successful send followed by dispatcher crash before acknowledgement,
  resulting in redelivery but one consumer-side effect;
- consumer crash before commit, proving inbox and effect both roll back.

## Guarantees and non-guarantees

| Boundary | Claim supported by this implementation |
|---|---|
| Event + evaluation + outbox intent | Atomic in one local SQLite transaction |
| Dispatcher claim/lease state | Serialized by SQLite writer transaction |
| Broker send + outbox acknowledgement | Not atomic; duplicate sends are possible |
| Delivery attempts | At-least-once under retry/liveness assumptions |
| Consumer inbox + local DB effect | Atomic and idempotent in one local SQLite transaction |
| Arbitrary external side effect | No exactly-once guarantee |

This is a single-host SQLite reference implementation. It is not a distributed
outbox service, broker integration, consensus protocol, or proof of
cross-host exactly-once processing. Production deployment requires a supported
transactional database, broker-specific publisher confirmation, operational
lease/timeout tuning, poison-message handling, retention, observability, and
consumer-side idempotency.
