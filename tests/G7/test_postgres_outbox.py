"""PostgreSQL integration tests for transactional outbox and receiver deduplication."""
from __future__ import annotations

import os
import unittest
from uuid import uuid4

try:
    import psycopg
    from ucon_jepa.postgres_service import PostgresConfig, PostgresUconService
    from ucon_jepa.postgres_outbox import (
        PostgresOutbox, PostgresIdempotentReceiver, OutboxEvent,
    )
except ImportError:
    psycopg = None
    PostgresConfig = PostgresUconService = None
    PostgresOutbox = PostgresIdempotentReceiver = OutboxEvent = None

DSN = os.environ.get("UCON_JEPA_TEST_POSTGRES_DSN")


@unittest.skipUnless(DSN and psycopg, "PostgreSQL DSN and psycopg 3 required")
class PostgresOutboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = PostgresConfig(DSN)
        cls.service = PostgresUconService(cls.config)
        cls.service.initialize()
        cls.outbox = PostgresOutbox(cls.config)
        cls.outbox.initialize()
        cls.receiver = PostgresIdempotentReceiver(cls.config)
        cls.receiver.initialize()

    def create_accepted_event(self):
        resource = "outbox-r-" + uuid4().hex
        principal = "outbox-p-" + uuid4().hex
        self.service.create_resource(resource, b"before")
        self.service.grant(principal, "write", resource)
        snap = self.service.snapshot(resource)
        invocation = "outbox-i-" + uuid4().hex
        result = self.service.mutate(
            invocation_id=invocation, principal=principal, action="write",
            resource_id=resource, expected_resource_version=snap["resource_version"],
            expected_policy_version=snap["policy_version"],
            expected_authorization_epoch=snap["authorization_epoch"],
            payload=b"after",
        )
        self.assertEqual(result.outcome, "ACCEPTED")
        return invocation, resource

    def test_accepted_mutation_and_outbox_row_commit_together(self):
        invocation, resource = self.create_accepted_event()
        with self.service._connect() as conn:
            row = conn.execute(
                "SELECT event_id,event_type,payload FROM ucon_outbox WHERE invocation_id=%s",
                (invocation,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["event_id"], invocation)
        self.assertEqual(row["event_type"], "RESOURCE_MUTATED")
        self.assertEqual(row["payload"]["resource_id"], resource)

    def test_receiver_applies_local_effect_once_under_duplicate_delivery(self):
        invocation, _ = self.create_accepted_event()
        events = self.outbox.claim(limit=100, lease_seconds=30)
        event = next(item for item in events if item.event_id == invocation)
        self.assertTrue(self.receiver.process(event))
        self.assertFalse(self.receiver.process(event))
        with self.service._connect() as conn:
            count = conn.execute(
                "SELECT count(*) AS n FROM ucon_receiver_effects WHERE event_id=%s",
                (invocation,),
            ).fetchone()["n"]
        self.assertEqual(count, 1)

    def test_crash_after_send_before_ack_causes_redelivery_but_no_duplicate_effect(self):
        invocation, _ = self.create_accepted_event()
        first_claim = self.outbox.claim(limit=100, lease_seconds=0)
        event = next(item for item in first_claim if item.event_id == invocation)
        self.assertTrue(self.receiver.process(event))
        # Simulates publisher crash after receiver commit and before outbox ack.
        second_claim = self.outbox.claim(limit=100, lease_seconds=0)
        redelivered = next(item for item in second_claim if item.event_id == invocation)
        self.assertFalse(self.receiver.process(redelivered))
        self.assertTrue(self.outbox.acknowledge(
            invocation, attempts=redelivered.attempts
        ))
        self.assertFalse(self.outbox.acknowledge(
            invocation, attempts=event.attempts
        ))

    def test_stale_lease_holder_cannot_acknowledge_newer_claim(self):
        invocation, _ = self.create_accepted_event()
        old_claim = self.outbox.claim(limit=100, lease_seconds=0)
        old_event = next(item for item in old_claim if item.event_id == invocation)
        new_claim = self.outbox.claim(limit=100, lease_seconds=0)
        new_event = next(item for item in new_claim if item.event_id == invocation)
        self.assertGreater(new_event.attempts, old_event.attempts)
        self.assertFalse(self.outbox.acknowledge(
            invocation, attempts=old_event.attempts
        ))
        self.assertTrue(self.outbox.acknowledge(
            invocation, attempts=new_event.attempts
        ))

    def test_dispatcher_acknowledges_only_after_successful_send(self):
        invocation, _ = self.create_accepted_event()
        seen = []

        def send(event):
            if event.event_id == invocation:
                seen.append(event.event_id)
                self.receiver.process(event)

        sent, failed = self.outbox.deliver_once(send, limit=100, lease_seconds=0)
        self.assertGreaterEqual(sent, 1)
        self.assertEqual(failed, 0)
        self.assertIn(invocation, seen)
        with self.service._connect() as conn:
            row = conn.execute(
                "SELECT delivered_at FROM ucon_outbox WHERE event_id=%s",
                (invocation,),
            ).fetchone()
        self.assertIsNotNone(row["delivered_at"])

    def test_failed_send_is_not_marked_delivered(self):
        invocation, _ = self.create_accepted_event()

        def fail(_event):
            raise RuntimeError("simulated downstream outage")

        sent, failed = self.outbox.deliver_once(
            fail, limit=100, lease_seconds=0
        )
        self.assertGreaterEqual(failed, 1)
        self.assertGreaterEqual(sent, 0)
        with self.service._connect() as conn:
            row = conn.execute(
                "SELECT delivered_at FROM ucon_outbox WHERE event_id=%s",
                (invocation,),
            ).fetchone()
        self.assertIsNone(row["delivered_at"])


if __name__ == "__main__":
    unittest.main()
