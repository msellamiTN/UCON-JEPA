import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path

from ucon_jepa.transactional_service import (
    IdempotencyConflict, SqliteUconService,
)


class TransactionalServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "ucon.sqlite")
        self.service = SqliteUconService(self.db)
        self.service.create_resource("doc-1", b"initial")
        self.policy_version, self.epoch = self.service.grant(
            "agent-A", "write", "doc-1"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, invocation_id="i-1", **overrides):
        args = dict(
            invocation_id=invocation_id, principal="agent-A", action="write",
            resource_id="doc-1", expected_resource_version=0,
            expected_policy_version=self.policy_version,
            expected_authorization_epoch=self.epoch, payload=b"updated",
        )
        args.update(overrides)
        return self.service.mutate(**args)

    def test_joint_authorization_and_cas_accepts_fresh_command(self):
        result = self.command()
        self.assertEqual(result.outcome, "ACCEPTED")
        self.assertIsNotNone(result.receipt_id)
        snap = self.service.snapshot("doc-1")
        self.assertEqual(snap["resource_version"], 1)
        self.assertEqual(snap["payload"], b"updated")

    def test_stale_resource_version_is_conflict_without_partial_write(self):
        result = self.command(expected_resource_version=9)
        self.assertEqual(result.outcome, "CONFLICT")
        snap = self.service.snapshot("doc-1")
        self.assertEqual(snap["resource_version"], 0)
        self.assertEqual(snap["payload"], b"initial")

    def test_stale_policy_version_is_conflict(self):
        self.service.update_policy()
        result = self.command()
        self.assertEqual(result.outcome, "CONFLICT")
        self.assertIn("policy version", result.reason)

    def test_stale_authorization_epoch_is_conflict(self):
        self.service.revoke("agent-A")
        result = self.command()
        self.assertEqual(result.outcome, "CONFLICT")
        self.assertIn("authorization epoch", result.reason)

    def test_current_epoch_but_revoked_principal_is_denied(self):
        self.policy_version, self.epoch, _ = self.service.revoke("agent-A")
        result = self.command()
        self.assertEqual(result.outcome, "DENIED")
        self.assertEqual(result.reason, "principal revoked")
        self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 0)

    def test_absent_ucon_grant_is_denied(self):
        result = self.command(principal="agent-B")
        self.assertEqual(result.outcome, "DENIED")
        self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 0)

    def test_identical_invocation_retry_returns_original_receipt(self):
        first = self.command()
        replay = self.command()
        self.assertEqual(replay.outcome, first.outcome)
        self.assertEqual(replay.receipt_id, first.receipt_id)
        self.assertEqual(replay.event_sequence, first.event_sequence)
        self.assertTrue(replay.replayed)
        self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 1)

    def test_invocation_id_cannot_be_reused_for_different_command(self):
        self.command()
        with self.assertRaises(IdempotencyConflict):
            self.command(payload=b"different")

    def test_denied_invocation_is_durably_bound(self):
        denied = self.command(principal="agent-B")
        self.assertEqual(denied.outcome, "DENIED")
        with self.assertRaises(IdempotencyConflict):
            self.command(principal="agent-A")

    def test_jePA_score_is_not_an_authorization_parameter(self):
        # The API has no score/embedding/representation field; only a current
        # explicit grant plus versions can authorize a state change.
        with self.assertRaises(TypeError):
            self.command(jePA_score=1.0)

    def test_concurrent_same_version_mutations_have_single_winner(self):
        workers = 24
        barrier = threading.Barrier(workers)

        def run(i):
            barrier.wait()
            return self.command(invocation_id=f"parallel-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, range(workers)))
        self.assertEqual(sum(r.outcome == "ACCEPTED" for r in results), 1)
        self.assertEqual(sum(r.outcome == "CONFLICT" for r in results), workers - 1)
        self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 1)

    def test_revoke_vs_mutation_linearizes_in_one_database_order(self):
        trials = 40
        accepted = 0
        revoked_first = 0
        for i in range(trials):
            with tempfile.TemporaryDirectory() as td:
                svc = SqliteUconService(str(Path(td) / "race.sqlite"))
                svc.create_resource("r", b"x")
                pv, epoch = svc.grant("p", "write", "r")
                barrier = threading.Barrier(2)

                def mutate():
                    barrier.wait()
                    return svc.mutate(
                        invocation_id=f"race-{i}", principal="p", action="write",
                        resource_id="r", expected_resource_version=0,
                        expected_policy_version=pv,
                        expected_authorization_epoch=epoch, payload=b"y",
                    )

                def revoke():
                    barrier.wait()
                    return svc.revoke("p")

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    fm = pool.submit(mutate)
                    fr = pool.submit(revoke)
                    mutation_result = fm.result()
                    _, _, revoke_sequence = fr.result()

                if mutation_result.outcome == "ACCEPTED":
                    accepted += 1
                    self.assertLess(mutation_result.event_sequence, revoke_sequence)
                    self.assertEqual(svc.snapshot("r")["resource_version"], 1)
                else:
                    revoked_first += 1
                    self.assertIn(mutation_result.outcome, {"CONFLICT", "DENIED"})
                    self.assertGreater(mutation_result.event_sequence, revoke_sequence)
                    self.assertEqual(svc.snapshot("r")["resource_version"], 0)
        self.assertEqual(accepted + revoked_first, trials)

    def test_policy_update_vs_mutation_never_accepts_old_policy_after_update(self):
        barrier = threading.Barrier(2)

        def mutate():
            barrier.wait()
            return self.command(invocation_id="policy-race")

        def update():
            barrier.wait()
            return self.service.update_policy()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fm = pool.submit(mutate)
            fu = pool.submit(update)
            mutation_result = fm.result()
            fu.result()
        if mutation_result.outcome == "ACCEPTED":
            # Mutation serialized before policy update.
            self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 1)
        else:
            self.assertEqual(mutation_result.outcome, "CONFLICT")
            self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 0)



    def test_fault_before_commit_rolls_back_everything(self):
        def fail(stage):
            if stage == "before_commit":
                raise RuntimeError("simulated process failure before commit")

        crashing = SqliteUconService(self.db, fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "before commit"):
            crashing.mutate(
                invocation_id="crash-before", principal="agent-A", action="write",
                resource_id="doc-1", expected_resource_version=0,
                expected_policy_version=self.policy_version,
                expected_authorization_epoch=self.epoch, payload=b"new",
            )
        snap = self.service.snapshot("doc-1")
        self.assertEqual(snap["resource_version"], 0)
        self.assertEqual(snap["payload"], b"initial")
        result = self.command(invocation_id="crash-before")
        self.assertEqual(result.outcome, "ACCEPTED")

    def test_lost_response_after_commit_recovers_original_receipt(self):
        def fail(stage):
            if stage == "after_commit":
                raise RuntimeError("simulated response loss after commit")

        crashing = SqliteUconService(self.db, fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "after commit"):
            crashing.mutate(
                invocation_id="crash-after", principal="agent-A", action="write",
                resource_id="doc-1", expected_resource_version=0,
                expected_policy_version=self.policy_version,
                expected_authorization_epoch=self.epoch, payload=b"new",
            )
        self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 1)
        recovered = self.service.mutate(
            invocation_id="crash-after", principal="agent-A", action="write",
            resource_id="doc-1", expected_resource_version=0,
            expected_policy_version=self.policy_version,
            expected_authorization_epoch=self.epoch, payload=b"new",
        )
        self.assertEqual(recovered.outcome, "ACCEPTED")
        self.assertTrue(recovered.replayed)
        self.assertEqual(self.service.snapshot("doc-1")["resource_version"], 1)


if __name__ == "__main__":
    unittest.main()
