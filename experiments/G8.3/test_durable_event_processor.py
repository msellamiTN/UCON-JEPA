import base64
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from durable_event_processor import SQLiteEventProcessor
from event_provenance import ProvenanceError, sign_event


class TransactionalReplayLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.processor = SQLiteEventProcessor(Path(self.temp.name) / "events.sqlite")
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

    def test_commit_stores_decision_and_replay_id_together(self):
        request = {"time": 1, "sequence": 0, "type": "workflow_request",
                   "workflow_id": "w1", "requester": "alice", "revision": 1}
        result = self.processor.process([self.signed(request, "e1")], self.keys)
        self.assertTrue(result["committed"])
        self.assertEqual(self.processor.counts(), {"processed_events": 1, "evaluations": 1})

    def test_replay_is_rejected_without_partial_second_commit(self):
        event = {"time": 1, "sequence": 0, "type": "workflow_request",
                 "workflow_id": "w1", "requester": "alice", "revision": 1}
        envelope = self.signed(event, "e1")
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

    def test_multi_event_trace_is_atomic_on_late_schema_failure(self):
        good = {"time": 1, "sequence": 0, "type": "workflow_request",
                "workflow_id": "w1", "requester": "alice", "revision": 1}
        bad = {"time": 2, "sequence": 0, "type": "grant", "actor": "security_admin"}
        envelopes = [self.signed(good, "good-e1"), self.signed(bad, "bad-e2")]
        with self.assertRaises(ValueError):
            self.processor.process(envelopes, self.keys, self.entitlements)
        self.assertEqual(self.processor.counts(), {"processed_events": 0, "evaluations": 0})


if __name__ == "__main__":
    unittest.main()
