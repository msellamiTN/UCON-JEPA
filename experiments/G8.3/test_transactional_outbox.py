import base64
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from durable_event_processor import SQLiteEventProcessor
from event_provenance import sign_event
from transactional_outbox import (
    OutboxError,
    SQLiteIdempotentConsumer,
    SQLiteOutboxDispatcher,
)


class TransactionalOutboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "outbox.sqlite"
        self.processor = SQLiteEventProcessor(self.database)
        self.dispatcher = SQLiteOutboxDispatcher(self.database)
        self.consumer = SQLiteIdempotentConsumer(self.database)
        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.keys = {"admin-k1": {
            "issuer": "security_admin",
            "public_key": base64.b64encode(public).decode("ascii"),
            "valid_from": 0, "valid_until": 1000, "revoked_at": None,
        }}

    def tearDown(self):
        self.temp.cleanup()

    def signed(self, event_id):
        event = {
            "time": 1, "sequence": 0, "type": "workflow_request",
            "workflow_id": f"wf-{event_id}", "requester": "alice", "revision": 1,
        }
        return sign_event(event, issuer="security_admin", key_id="admin-k1",
                          event_id=event_id, private_key=self.private)

    def process_one(self, event_id="outbox-e1"):
        return self.processor.process([self.signed(event_id)], self.keys)

    def test_decision_and_outbox_intent_commit_atomically(self):
        result = self.process_one()
        self.assertTrue(result["committed"])
        self.assertEqual(result["outbox_count"], 1)
        self.assertEqual(self.processor.counts(),
                         {"processed_events": 1, "evaluations": 1})
        self.assertEqual(self.processor.outbox_counts(),
                         {"pending": 1, "inflight": 0, "delivered": 0, "total": 1})
        with sqlite3.connect(self.database) as db:
            row = db.execute(
                "SELECT idempotency_key, event_id, evaluation_id, status FROM outbox"
            ).fetchone()
        self.assertEqual(row[0], "outbox-e1:ucon-decision:v1")
        self.assertEqual(row[1], "outbox-e1")
        self.assertEqual(row[3], "pending")

    def test_invalid_trace_rolls_back_event_evaluation_and_outbox(self):
        malformed = {
            "time": 1, "sequence": 0, "type": "grant",
            "actor": "security_admin",
        }
        envelope = sign_event(
            malformed, issuer="security_admin", key_id="admin-k1",
            event_id="invalid-e1", private_key=self.private,
        )
        with self.assertRaises(ValueError):
            self.processor.process([envelope], self.keys)
        self.assertEqual(self.processor.counts(),
                         {"processed_events": 0, "evaluations": 0})
        self.assertEqual(self.processor.outbox_counts(),
                         {"pending": 0, "inflight": 0, "delivered": 0, "total": 0})

    def test_claim_is_leased_and_delivery_ack_is_token_bound(self):
        self.process_one()
        first = self.dispatcher.claim(lease_seconds=5)
        self.assertIsNotNone(first)
        self.assertEqual(first["attempt"], 1)
        self.assertEqual(first["payload"]["event_id"], "outbox-e1")
        self.assertIsNone(self.dispatcher.claim())
        self.dispatcher.mark_delivered(first["idempotency_key"], first["lease_token"])
        self.assertEqual(self.processor.outbox_counts(),
                         {"pending": 0, "inflight": 0, "delivered": 1, "total": 1})
        with self.assertRaises(OutboxError):
            self.dispatcher.mark_delivered(first["idempotency_key"], first["lease_token"])

    def test_expired_lease_is_reclaimed_and_stale_worker_cannot_ack(self):
        self.process_one()
        first = self.dispatcher.claim(lease_seconds=0.01)
        time.sleep(0.03)
        second = self.dispatcher.claim(lease_seconds=5)
        self.assertIsNotNone(second)
        self.assertEqual(second["attempt"], 2)
        self.assertNotEqual(first["lease_token"], second["lease_token"])
        with self.assertRaises(OutboxError):
            self.dispatcher.mark_delivered(first["idempotency_key"], first["lease_token"])
        self.dispatcher.mark_delivered(second["idempotency_key"], second["lease_token"])

    def test_failed_dispatch_can_be_retried(self):
        self.process_one()
        claim = self.dispatcher.claim()
        self.dispatcher.retry(claim["idempotency_key"], claim["lease_token"])
        again = self.dispatcher.claim()
        self.assertEqual(again["attempt"], 2)
        self.dispatcher.mark_delivered(again["idempotency_key"], again["lease_token"])

    def test_send_then_crash_before_ack_causes_redelivery_but_one_consumer_effect(self):
        self.process_one()
        first = self.dispatcher.claim(lease_seconds=0.01)
        # Model successful external send followed by dispatcher crash: no
        # mark_delivered call occurs, so the lease eventually expires.
        self.assertTrue(self.consumer.consume(
            first["idempotency_key"], {"decision": "allow", "resource": "r1"}
        ))
        time.sleep(0.03)
        second = self.dispatcher.claim(lease_seconds=5)
        self.assertEqual(second["idempotency_key"], first["idempotency_key"])
        self.assertFalse(self.consumer.consume(
            second["idempotency_key"], {"decision": "allow", "resource": "r1"}
        ))
        self.dispatcher.mark_delivered(second["idempotency_key"], second["lease_token"])
        self.assertEqual(self.consumer.counts(), {"inbox": 1, "business_effects": 1})

    def test_consumer_effect_and_inbox_are_atomic_on_failure(self):
        class FailingConsumer(SQLiteIdempotentConsumer):
            def _before_commit(self, db):
                raise RuntimeError("injected pre-commit failure")

        failing = FailingConsumer(self.database)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            failing.consume("consumer-fail-e1", {"action": "charge"})
        self.assertEqual(self.consumer.counts(), {"inbox": 0, "business_effects": 0})
        self.assertTrue(self.consumer.consume(
            "consumer-fail-e1", {"action": "charge"}
        ))
        self.assertFalse(self.consumer.consume(
            "consumer-fail-e1", {"action": "charge"}
        ))
        self.assertEqual(self.consumer.counts(), {"inbox": 1, "business_effects": 1})


if __name__ == "__main__":
    unittest.main()
