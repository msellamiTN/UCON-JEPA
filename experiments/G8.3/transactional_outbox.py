#!/usr/bin/env python3
"""SQLite transactional-outbox dispatcher and idempotent consumer reference.

The dispatcher provides at-least-once delivery intent. It cannot atomically
commit a broker send and a local SQLite acknowledgement. The consumer example
achieves an effectively-once local business effect only because its inbox
deduplication row and effect row share one SQLite transaction.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class OutboxError(RuntimeError):
    """Raised for invalid or stale outbox state transitions."""


class SQLiteOutboxDispatcher:
    def __init__(self, database: str | Path):
        self.database = str(database)

    def claim(self, lease_seconds: float = 30.0) -> dict | None:
        """Atomically lease one ready item; expired leases are reclaimable."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time.time()
        token = str(uuid.uuid4())
        with sqlite3.connect(self.database, timeout=30, isolation_level=None) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT idempotency_key, payload_json, attempts
                   FROM outbox
                   WHERE (status='pending' AND available_at<=?)
                      OR (status='inflight' AND lease_until<=?)
                   ORDER BY available_at, idempotency_key LIMIT 1""",
                (now, now),
            ).fetchone()
            if row is None:
                db.execute("COMMIT")
                return None
            key, payload_json, attempts = row
            updated = db.execute(
                """UPDATE outbox
                   SET status='inflight', attempts=?, lease_token=?, lease_until=?
                   WHERE idempotency_key=?
                     AND ((status='pending' AND available_at<=?)
                       OR (status='inflight' AND lease_until<=?))""",
                (attempts + 1, token, now + lease_seconds, key, now, now),
            ).rowcount
            if updated != 1:
                db.execute("ROLLBACK")
                raise OutboxError("outbox lease changed during claim")
            db.execute("COMMIT")
        return {
            "idempotency_key": key,
            "payload": json.loads(payload_json),
            "attempt": attempts + 1,
            "lease_token": token,
            "lease_until": now + lease_seconds,
        }

    def mark_delivered(self, idempotency_key: str, lease_token: str) -> None:
        """Acknowledge only the current lease; stale workers cannot ack."""
        with sqlite3.connect(self.database, timeout=30, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                """UPDATE outbox
                   SET status='delivered', delivered_at=?, lease_token=NULL, lease_until=NULL
                   WHERE idempotency_key=? AND status='inflight' AND lease_token=?""",
                (time.time(), idempotency_key, lease_token),
            ).rowcount
            if changed != 1:
                db.execute("ROLLBACK")
                raise OutboxError("missing, expired, or superseded outbox lease")
            db.execute("COMMIT")

    def retry(self, idempotency_key: str, lease_token: str,
              delay_seconds: float = 0.0) -> None:
        """Return a failed attempt to pending with an explicit retry delay."""
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")
        with sqlite3.connect(self.database, timeout=30, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                """UPDATE outbox
                   SET status='pending', available_at=?, lease_token=NULL, lease_until=NULL
                   WHERE idempotency_key=? AND status='inflight' AND lease_token=?""",
                (time.time() + delay_seconds, idempotency_key, lease_token),
            ).rowcount
            if changed != 1:
                db.execute("ROLLBACK")
                raise OutboxError("missing, expired, or superseded outbox lease")
            db.execute("COMMIT")


class SQLiteIdempotentConsumer:
    def __init__(self, database: str | Path):
        self.database = str(database)
        with sqlite3.connect(self.database) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS consumer_inbox (
                    idempotency_key TEXT PRIMARY KEY,
                    received_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS business_effects (
                    idempotency_key TEXT PRIMARY KEY
                        REFERENCES consumer_inbox(idempotency_key),
                    effect_json TEXT NOT NULL,
                    applied_at REAL NOT NULL
                );
            """)

    def _before_commit(self, db: sqlite3.Connection) -> None:
        """Fault-injection seam between staged inbox/effect writes and commit."""
        return None

    def consume(self, idempotency_key: str, effect: dict[str, Any]) -> bool:
        """Apply a local effect once; return False for a duplicate delivery."""
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        with sqlite3.connect(self.database, timeout=30, isolation_level=None) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            inserted = db.execute(
                "INSERT OR IGNORE INTO consumer_inbox (idempotency_key, received_at) VALUES (?, ?)",
                (idempotency_key, time.time()),
            ).rowcount
            if inserted == 0:
                db.execute("ROLLBACK")
                return False
            db.execute(
                "INSERT INTO business_effects (idempotency_key, effect_json, applied_at) VALUES (?, ?, ?)",
                (idempotency_key, json.dumps(effect, sort_keys=True), time.time()),
            )
            self._before_commit(db)
            db.execute("COMMIT")
            return True

    def counts(self) -> dict:
        with sqlite3.connect(self.database) as db:
            inbox = db.execute("SELECT COUNT(*) FROM consumer_inbox").fetchone()[0]
            effects = db.execute("SELECT COUNT(*) FROM business_effects").fetchone()[0]
        return {"inbox": inbox, "business_effects": effects}
