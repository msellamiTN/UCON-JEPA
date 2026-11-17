"""Transactional outbox and idempotent receiver reference implementation.

The publisher is at-least-once: a crash after sending but before marking the
outbox row delivered can cause redelivery. A receiver can make its *local
database effect* idempotent by committing inbox deduplication and the effect
in one transaction. This does not make arbitrary remote side effects exactly
once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from .postgres_service import PostgresConfig, PostgresUconService


@dataclass(frozen=True)
class OutboxEvent:
    event_id: str
    event_type: str
    payload: dict
    attempts: int


class PostgresOutbox:
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.service = PostgresUconService(config)

    def initialize(self) -> None:
        self.service.initialize()

    def claim(self, *, limit: int = 20, lease_seconds: int = 30) -> list[OutboxEvent]:
        if limit < 1 or lease_seconds < 0:
            raise ValueError("limit must be positive and lease_seconds non-negative")
        with self.service._connect() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(
                    """
                    WITH candidates AS (
                        SELECT event_id FROM ucon_outbox
                        WHERE delivered_at IS NULL
                          AND available_at <= clock_timestamp()
                          AND (lease_until IS NULL OR lease_until <= clock_timestamp())
                        ORDER BY created_at, event_id
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE ucon_outbox AS o
                    SET lease_until = clock_timestamp() + (%s * INTERVAL '1 second'),
                        attempts = attempts + 1
                    FROM candidates AS c
                    WHERE o.event_id = c.event_id
                    RETURNING o.event_id, o.event_type, o.payload, o.attempts
                    """,
                    (limit, lease_seconds),
                ).fetchall()
                return [
                    OutboxEvent(
                        event_id=row["event_id"], event_type=row["event_type"],
                        payload=dict(row["payload"]), attempts=row["attempts"],
                    )
                    for row in rows
                ]

    def acknowledge(self, event_id: str, *, attempts: int) -> bool:
        """Acknowledge only the current lease generation (fencing stale workers)."""
        with self.service._connect() as conn:
            row = conn.execute(
                "UPDATE ucon_outbox SET delivered_at=clock_timestamp(),lease_until=NULL "
                "WHERE event_id=%s AND attempts=%s AND delivered_at IS NULL "
                "RETURNING event_id",
                (event_id, attempts),
            ).fetchone()
            return row is not None

    def reschedule(self, event_id: str, *, attempts: int,
                   delay_seconds: int = 5) -> bool:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        with self.service._connect() as conn:
            row = conn.execute(
                "UPDATE ucon_outbox SET lease_until=NULL, "
                "available_at=clock_timestamp() + (%s * INTERVAL '1 second') "
                "WHERE event_id=%s AND attempts=%s AND delivered_at IS NULL "
                "RETURNING event_id",
                (delay_seconds, event_id, attempts),
            ).fetchone()
            return row is not None

    def deliver_once(self, send: Callable[[OutboxEvent], Any], *,
                     limit: int = 20, lease_seconds: int = 30) -> tuple[int, int]:
        """Send claimed events, then acknowledge; failed sends remain leased."""
        events = self.claim(limit=limit, lease_seconds=lease_seconds)
        sent = failed = 0
        for event in events:
            try:
                send(event)
                self.acknowledge(event.event_id, attempts=event.attempts)
                sent += 1
            except Exception:
                self.reschedule(event.event_id, attempts=event.attempts,
                                delay_seconds=min(300, 2 ** min(event.attempts, 8)))
                failed += 1
        return sent, failed


class PostgresIdempotentReceiver:
    """Receiver-side inbox and a sample local effect in the same PostgreSQL DB."""

    def __init__(self, config: PostgresConfig):
        self.config = config
        self.service = PostgresUconService(config)

    def initialize(self) -> None:
        with self.service._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ucon_receiver_inbox (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ucon_receiver_effects (
                    event_id TEXT PRIMARY KEY REFERENCES ucon_receiver_inbox(event_id),
                    resource_id TEXT NOT NULL,
                    observed_version BIGINT NOT NULL,
                    effect_count INTEGER NOT NULL DEFAULT 1 CHECK(effect_count=1)
                )
            """)

    def process(self, event: OutboxEvent) -> bool:
        """Return True iff first delivery applied; duplicates return False."""
        payload = event.payload
        required = ("resource_id", "resource_version")
        if any(key not in payload for key in required):
            raise ValueError("event payload lacks required resource binding")
        with self.service._connect() as conn:
            with conn.cursor() as cur:
                inserted = cur.execute(
                    "INSERT INTO ucon_receiver_inbox(event_id,event_type) VALUES(%s,%s) "
                    "ON CONFLICT(event_id) DO NOTHING RETURNING event_id",
                    (event.event_id, event.event_type),
                ).fetchone()
                if inserted is None:
                    return False
                cur.execute(
                    "INSERT INTO ucon_receiver_effects(event_id,resource_id,observed_version) "
                    "VALUES(%s,%s,%s)",
                    (event.event_id, payload["resource_id"], payload["resource_version"]),
                )
                return True
