import base64
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from durable_event_processor import SQLiteEventProcessor
from event_provenance import ProvenanceError, sign_event


# The child process exits without Python cleanup at the selected transaction
# boundary. This exercises SQLite/WAL recovery after abrupt process death.
_CRASH_SCRIPT = r"""
import json
import os
import sys
from durable_event_processor import SQLiteEventProcessor

mode, database, envelopes_json, keys_json = sys.argv[1:]
class CrashProcessor(SQLiteEventProcessor):
    def _before_commit(self, connection):
        if mode == "before_commit":
            os._exit(71)
    def _after_commit(self, connection):
        if mode == "after_commit":
            os._exit(72)

CrashProcessor(database).process(json.loads(envelopes_json), json.loads(keys_json))
"""


_PROCESS_RACE_SCRIPT = r"""
import json
import sys
import time
from pathlib import Path
from durable_event_processor import SQLiteEventProcessor
from event_provenance import ProvenanceError

database, envelope_json, keys_json, ready_file, start_file = sys.argv[1:]
processor = SQLiteEventProcessor(database)
Path(ready_file).touch()
deadline = time.monotonic() + 20
while not Path(start_file).exists():
    if time.monotonic() > deadline:
        raise SystemExit("timed out waiting for concurrent start gate")
    time.sleep(0.005)

try:
    result = processor.process([json.loads(envelope_json)], json.loads(keys_json))
    print(json.dumps({"committed": True, "evaluation_id": result["evaluation_id"]}))
except ProvenanceError as exc:
    print(json.dumps({"committed": False, "reason": str(exc)}))
"""


class TransactionalReplayLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "events.sqlite"
        self.processor = SQLiteEventProcessor(self.database)
        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.keys = {"admin-k1": {
            "issuer": "security_admin", "public_key": base64.b64encode(public).decode("ascii"),
            "valid_from": 0, "valid_until": 1000, "revoked_at": None,
        }}
        self.entitlements = {"security_admin": ["read", "write", "admin"]}

    def tearDown(self):
        self.temp.cleanup()

    def signed(self, event, event_id):
        return sign_event(event, issuer="security_admin", key_id="admin-k1",
                          event_id=event_id, private_key=self.private)

    def workflow_event(self, workflow_id="w1"):
        return {"time": 1, "sequence": 0, "type": "workflow_request",
                "workflow_id": workflow_id, "requester": "alice", "revision": 1}

    def crash_child(self, mode, envelope):
        completed = subprocess.run(
            [sys.executable, "-c", _CRASH_SCRIPT, mode, str(self.database),
             json.dumps([envelope]), json.dumps(self.keys)],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=20,
        )
        expected = 71 if mode == "before_commit" else 72
        self.assertEqual(completed.returncode, expected, completed.stderr)

    def assert_database_healthy(self):
        import sqlite3
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            event_ids = connection.execute(
                "SELECT event_id FROM processed_events ORDER BY event_id"
            ).fetchall()
            evaluation_count = connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        return [row[0] for row in event_ids], evaluation_count

    def test_commit_stores_decision_and_replay_id_together(self):
        result = self.processor.process([self.signed(self.workflow_event(), "e1")], self.keys)
        self.assertTrue(result["committed"])
        self.assertEqual(self.processor.counts(), {"processed_events": 1, "evaluations": 1})

    def test_replay_is_rejected_without_partial_second_commit(self):
        envelope = self.signed(self.workflow_event(), "e1")
        self.processor.process([envelope], self.keys)
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            self.processor.process([envelope], self.keys)
        self.assertEqual(self.processor.counts(), {"processed_events": 1, "evaluations": 1})

    def test_schema_failure_rolls_back_replay_id_and_evaluation(self):
        malformed = {"time": 1, "sequence": 0, "type": "grant", "actor": "security_admin"}
        with self.assertRaises(ValueError):
            self.processor.process([self.signed(malformed, "bad-e1")], self.keys,
                                   self.entitlements)
        self.assertEqual(self.processor.counts(), {"processed_events": 0, "evaluations": 0})

    def test_denied_authorization_is_still_committed_as_authenticated_event(self):
        read = {"time": 1, "sequence": 0, "type": "read", "grant_id": "missing",
                "principal": "alice", "scope": "read", "tenant": "t1",
                "resource_id": "r1", "classification": 1, "clearance": 2}
        result = self.processor.process([self.signed(read, "denied-e1")], self.keys)
        self.assertFalse(result["evaluation"]["decisions"][0]["allowed"])
        self.assertEqual(self.processor.counts(), {"processed_events": 1, "evaluations": 1})

    def test_concurrent_duplicate_delivery_commits_exactly_once(self):
        envelope = self.signed(self.workflow_event("concurrent-w1"), "concurrent-e1")
        barrier = Barrier(2)

        def submit():
            barrier.wait()
            try:
                return self.processor.process([envelope], self.keys)["committed"]
            except ProvenanceError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: submit(), range(2)))
        self.assertEqual(sorted(outcomes), [False, True])
        self.assertEqual(self.processor.counts(), {"processed_events": 1, "evaluations": 1})

    def test_multi_event_trace_is_atomic_on_late_schema_failure(self):
        good = self.workflow_event("atomic-w1")
        bad = {"time": 2, "sequence": 0, "type": "grant", "actor": "security_admin"}
        envelopes = [self.signed(good, "good-e1"), self.signed(bad, "bad-e2")]
        with self.assertRaises(ValueError):
            self.processor.process(envelopes, self.keys, self.entitlements)
        self.assertEqual(self.processor.counts(), {"processed_events": 0, "evaluations": 0})

    def test_crash_before_commit_leaves_no_partial_state_after_restart(self):
        envelope = self.signed(self.workflow_event("crash-before"), "crash-before-e1")
        self.crash_child("before_commit", envelope)
        # Reopening the processor forces SQLite to recover its WAL/journal.
        recovered = SQLiteEventProcessor(self.database)
        self.assertEqual(recovered.counts(), {"processed_events": 0, "evaluations": 0})
        self.assertEqual(self.assert_database_healthy(), ([], 0))
        # The same signed event remains processable after the aborted attempt.
        result = recovered.process([envelope], self.keys)
        self.assertTrue(result["committed"])
        self.assertEqual(recovered.counts(), {"processed_events": 1, "evaluations": 1})

    def test_commit_then_lost_ack_is_recovered_as_replay(self):
        envelope = self.signed(self.workflow_event("crash-after"), "crash-after-e1")
        self.crash_child("after_commit", envelope)
        recovered = SQLiteEventProcessor(self.database)
        self.assertEqual(recovered.counts(), {"processed_events": 1, "evaluations": 1})
        self.assertEqual(self.assert_database_healthy(), (["crash-after-e1"], 1))
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            recovered.process([envelope], self.keys)
        self.assertEqual(recovered.counts(), {"processed_events": 1, "evaluations": 1})

    def test_restart_preserves_replay_history_and_evaluation_pairing(self):
        envelope = self.signed(self.workflow_event("restart"), "restart-e1")
        self.processor.process([envelope], self.keys)
        recovered = SQLiteEventProcessor(self.database)
        self.assertEqual(recovered.counts(), {"processed_events": 1, "evaluations": 1})
        self.assertEqual(self.assert_database_healthy(), (["restart-e1"], 1))
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            recovered.process([envelope], self.keys)

    def test_finite_crash_safety_model_has_no_invariant_counterexample(self):
        from crash_consistency_model import explore
        result = explore()
        self.assertGreater(result["visited_states"], 0)
        self.assertEqual(result["counterexamples"], [])

    def test_finite_model_detects_event_only_commit_mutant(self):
        from crash_consistency_model import explore
        result = explore(include_unsafe_mutant=True)
        self.assertTrue(result["counterexamples"])
        first = result["counterexamples"][0]
        self.assertIn("AtomicEventEvaluation", first["violations"])
        self.assertIn("UnsafeCommitEventOnly", first["trace"])


    def run_process_race(self, envelopes):
        ready_dir = Path(self.temp.name) / "ready"
        ready_dir.mkdir(exist_ok=True)
        start_file = Path(self.temp.name) / "start-race"
        processes = []
        for index, envelope in enumerate(envelopes):
            ready_file = ready_dir / f"ready-{index}"
            process = subprocess.Popen(
                [sys.executable, "-c", _PROCESS_RACE_SCRIPT,
                 str(self.database), json.dumps(envelope), json.dumps(self.keys),
                 str(ready_file), str(start_file)],
                cwd=Path(__file__).resolve().parent,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append((process, ready_file))

        deadline = __import__("time").monotonic() + 30
        while not all(ready_file.exists() for _, ready_file in processes):
            if __import__("time").monotonic() > deadline:
                for process, _ in processes:
                    process.kill()
                self.fail("worker processes did not all reach the start barrier")
            __import__("time").sleep(0.01)

        start_file.touch()
        outcomes = []
        for process, _ in processes:
            stdout, stderr = process.communicate(timeout=40)
            self.assertEqual(process.returncode, 0, stderr)
            lines = [line for line in stdout.splitlines() if line.strip()]
            self.assertTrue(lines, f"worker produced no result; stderr={stderr}")
            outcomes.append(json.loads(lines[-1]))
        return outcomes

    def test_independent_processes_duplicate_delivery_commits_at_most_once(self):
        envelope = self.signed(self.workflow_event("multiprocess-duplicate"), "mp-duplicate-e1")
        outcomes = self.run_process_race([envelope] * 4)
        self.assertEqual(sum(item["committed"] for item in outcomes), 1)
        self.assertEqual(sum(not item["committed"] for item in outcomes), 3)
        self.assertEqual(self.processor.counts(), {"processed_events": 1, "evaluations": 1})
        self.assertEqual(self.assert_database_healthy(), (["mp-duplicate-e1"], 1))

    def test_independent_processes_distinct_events_all_commit(self):
        envelopes = [
            self.signed(self.workflow_event(f"multiprocess-distinct-{i}"), f"mp-distinct-e{i}")
            for i in range(4)
        ]
        outcomes = self.run_process_race(envelopes)
        self.assertEqual(sum(item["committed"] for item in outcomes), 4)
        self.assertEqual(self.processor.counts(), {"processed_events": 4, "evaluations": 4})
        self.assertEqual(
            self.assert_database_healthy(),
            ([f"mp-distinct-e{i}" for i in range(4)], 4),
        )


if __name__ == "__main__":
    unittest.main()
