"""Real PostgreSQL integration tests; skipped without UCON_JEPA_TEST_POSTGRES_DSN."""
from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

try:
    import psycopg  # noqa: F401
    from ucon_jepa.postgres_service import PostgresConfig, PostgresUconService
    from ucon_jepa.transactional_service import IdempotencyConflict
except ImportError:
    psycopg = None
    PostgresConfig = PostgresUconService = IdempotencyConflict = None


DSN = os.environ.get("UCON_JEPA_TEST_POSTGRES_DSN")


@unittest.skipUnless(DSN and psycopg, "PostgreSQL DSN and psycopg 3 required")
class PostgresServiceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin = PostgresUconService(PostgresConfig(DSN))
        cls.admin.initialize()

    def setUp(self):
        self.resource = "r-" + uuid4().hex
        self.principal = "p-" + uuid4().hex
        self.admin.create_resource(self.resource, b"old")
        self.admin.grant(self.principal, "write", self.resource)

    def command(self, snapshot, invocation_id=None, payload=b"new"):
        return dict(
            invocation_id=invocation_id or "i-" + uuid4().hex,
            principal=self.principal, action="write", resource_id=self.resource,
            expected_resource_version=snapshot["resource_version"],
            expected_policy_version=snapshot["policy_version"],
            expected_authorization_epoch=snapshot["authorization_epoch"],
            payload=payload,
        )

    def test_acceptance_and_idempotent_replay(self):
        snap = self.admin.snapshot(self.resource)
        cmd = self.command(snap)
        first = self.admin.mutate(**cmd)
        second = self.admin.mutate(**cmd)
        self.assertEqual(first.outcome, "ACCEPTED")
        self.assertTrue(second.replayed)
        self.assertEqual(first.receipt_id, second.receipt_id)
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 1)

    def test_invocation_key_cannot_be_reused_for_different_command(self):
        snap = self.admin.snapshot(self.resource)
        cmd = self.command(snap)
        self.admin.mutate(**cmd)
        with self.assertRaises(IdempotencyConflict):
            self.admin.mutate(**{**cmd, "payload": b"different"})

    def test_revocation_fences_subsequent_mutation_and_audits_writer_epoch(self):
        writer_epoch = self.admin.current_writer_epoch()
        _, _, sequence = self.admin.revoke(self.principal)
        with self.admin._connect() as conn:
            event = conn.execute(
                "SELECT event_type,writer_epoch FROM ucon_events WHERE sequence=%s",
                (sequence,),
            ).fetchone()
        self.assertEqual(event["event_type"], "REVOKE")
        self.assertEqual(event["writer_epoch"], writer_epoch)
        snap = self.admin.snapshot(self.resource)
        result = self.admin.mutate(**self.command(snap))
        self.assertEqual(result.outcome, "DENIED")
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 0)

    def test_policy_update_fences_stale_authorization_snapshot(self):
        snap = self.admin.snapshot(self.resource)
        self.admin.update_policy()
        result = self.admin.mutate(**self.command(snap))
        self.assertEqual(result.outcome, "CONFLICT")
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 0)

    def test_concurrent_same_version_has_exactly_one_winner(self):
        snap = self.admin.snapshot(self.resource)
        barrier = threading.Barrier(12)

        def attempt(i):
            svc = PostgresUconService(PostgresConfig(DSN))
            cmd = self.command(snap, invocation_id=f"race-{uuid4().hex}-{i}")
            barrier.wait(timeout=10)
            return svc.mutate(**cmd).outcome

        with ThreadPoolExecutor(max_workers=12) as pool:
            outcomes = list(pool.map(attempt, range(12)))
        self.assertEqual(outcomes.count("ACCEPTED"), 1)
        self.assertEqual(outcomes.count("CONFLICT"), 11)
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 1)

    def test_precommit_failure_rolls_back_and_allows_same_command_retry(self):
        snap = self.admin.snapshot(self.resource)
        cmd = self.command(snap)

        def fail(stage):
            if stage == "before_commit":
                raise RuntimeError("injected pre-commit failure")

        faulty = PostgresUconService(PostgresConfig(DSN), fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            faulty.mutate(**cmd)
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 0)
        result = self.admin.mutate(**cmd)
        self.assertEqual(result.outcome, "ACCEPTED")

    def test_postcommit_response_loss_replays_committed_receipt(self):
        snap = self.admin.snapshot(self.resource)
        cmd = self.command(snap)

        def fail(stage):
            if stage == "after_commit":
                raise RuntimeError("injected response loss")

        faulty = PostgresUconService(PostgresConfig(DSN), fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "response loss"):
            faulty.mutate(**cmd)
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 1)
        replay = self.admin.mutate(**cmd)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.outcome, "ACCEPTED")
        self.assertEqual(self.admin.snapshot(self.resource)["resource_version"], 1)


if __name__ == "__main__":
    unittest.main()
