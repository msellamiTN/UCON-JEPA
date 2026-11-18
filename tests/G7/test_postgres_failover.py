"""PostgreSQL writer-epoch fencing and failover-reconciliation contract tests."""
from __future__ import annotations

import os
import unittest
from uuid import uuid4

try:
    import psycopg
    from ucon_jepa.postgres_service import (
        PostgresConfig, PostgresUconService, StaleWriterEpoch,
    )
except ImportError:
    psycopg = None
    PostgresConfig = PostgresUconService = StaleWriterEpoch = None

DSN = os.environ.get("UCON_JEPA_TEST_POSTGRES_DSN")


@unittest.skipUnless(DSN and psycopg, "PostgreSQL DSN and psycopg 3 required")
class PostgresWriterFenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = PostgresUconService(PostgresConfig(DSN))
        cls.base.initialize()

    def setUp(self):
        self.service = PostgresUconService(
            PostgresConfig(DSN, require_writer_epoch=True)
        )
        self.epoch = self.service.current_writer_epoch()
        self.resource = "fence-r-" + uuid4().hex
        self.principal = "fence-p-" + uuid4().hex
        self.service.create_resource(
            self.resource, b"before", expected_writer_epoch=self.epoch
        )
        self.service.grant(
            self.principal, "write", self.resource,
            expected_writer_epoch=self.epoch,
        )

    def test_missing_writer_epoch_is_rejected_when_fencing_required(self):
        with self.assertRaises(StaleWriterEpoch):
            self.service.update_policy()

    def test_monotonic_promotion_fences_old_writer_token(self):
        promoted = self.service.advance_writer_epoch(
            expected_current_epoch=self.epoch,
            new_writer_epoch=self.epoch + 1,
        )
        self.assertEqual(promoted, self.epoch + 1)
        with self.assertRaises(StaleWriterEpoch):
            self.service.update_policy(expected_writer_epoch=self.epoch)
        with self.assertRaises(StaleWriterEpoch):
            self.service.revoke(
                self.principal, expected_writer_epoch=self.epoch
            )
        with self.assertRaises(StaleWriterEpoch):
            self.service.create_resource(
                "stale-" + uuid4().hex, expected_writer_epoch=self.epoch
            )
        self.service.update_policy(expected_writer_epoch=promoted)

    def test_mutation_with_stale_epoch_is_rejected_before_resource_change(self):
        snap = self.service.snapshot(self.resource)
        promoted = self.service.advance_writer_epoch(
            expected_current_epoch=self.epoch,
            new_writer_epoch=self.epoch + 1,
        )
        with self.assertRaises(StaleWriterEpoch):
            self.service.mutate(
                invocation_id="stale-i-" + uuid4().hex,
                principal=self.principal, action="write", resource_id=self.resource,
                expected_resource_version=snap["resource_version"],
                expected_policy_version=snap["policy_version"],
                expected_authorization_epoch=snap["authorization_epoch"],
                payload=b"should-not-apply",
                expected_writer_epoch=self.epoch,
            )
        current = self.service.snapshot(self.resource)
        self.assertEqual(current["resource_version"], 0)
        self.assertEqual(current["writer_epoch"], promoted)

    def test_current_epoch_mutation_and_invocation_reconciliation(self):
        snap = self.service.snapshot(self.resource)
        invocation = "fresh-i-" + uuid4().hex
        result = self.service.mutate(
            invocation_id=invocation, principal=self.principal, action="write",
            resource_id=self.resource,
            expected_resource_version=snap["resource_version"],
            expected_policy_version=snap["policy_version"],
            expected_authorization_epoch=snap["authorization_epoch"],
            payload=b"after", expected_writer_epoch=self.epoch,
        )
        self.assertEqual(result.outcome, "ACCEPTED")
        with self.service._connect() as conn:
            event = conn.execute(
                "SELECT writer_epoch FROM ucon_events WHERE sequence=%s",
                (result.event_sequence,),
            ).fetchone()
        self.assertEqual(event["writer_epoch"], self.epoch)
        recovered = self.service.lookup_invocation(invocation)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.receipt_id, result.receipt_id)
        self.assertIsNone(self.service.lookup_invocation("missing-" + uuid4().hex))

    def test_writer_epoch_cannot_move_backwards_or_repeat(self):
        with self.assertRaises(ValueError):
            self.service.advance_writer_epoch(
                expected_current_epoch=self.epoch,
                new_writer_epoch=self.epoch,
            )
        with self.assertRaises(ValueError):
            self.service.advance_writer_epoch(
                expected_current_epoch=self.epoch,
                new_writer_epoch=self.epoch - 1,
            )


if __name__ == "__main__":
    unittest.main()
