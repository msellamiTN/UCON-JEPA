import base64
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from authenticated_evaluation import evaluate_signed_trace
from event_provenance import ProvenanceError, sign_event


class AuthenticatedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.keys = {
            "admin-k1": {
                "issuer": "security_admin",
                "public_key": base64.b64encode(public).decode("ascii"),
                "valid_from": 0, "valid_until": 1000, "revoked_at": None,
            }
        }
        self.entitlements = {"security_admin": ["read", "write", "admin"]}

    def sign(self, event, event_id):
        return sign_event(event, issuer="security_admin", key_id="admin-k1",
                          event_id=event_id, private_key=self.private)

    def test_valid_signed_grant_and_read_reach_policy_evaluator(self):
        grant = {
            "time": 1, "sequence": 0, "type": "grant", "grant_id": "g1",
            "actor": "security_admin", "principal": "alice", "scopes": ["read"],
            "tenant": "t1", "expires_at": 20, "policy_version": 1,
        }
        read = {
            "time": 2, "sequence": 0, "type": "read", "grant_id": "g1",
            "principal": "alice", "scope": "read", "tenant": "t1",
            "resource_id": "r1", "classification": 1, "clearance": 2,
        }
        result = evaluate_signed_trace(
            [self.sign(grant, "e1"), self.sign(read, "e2")],
            self.keys, self.entitlements,
        )
        self.assertTrue(result["provenance_verified"])
        self.assertEqual(result["accepted_event_count"], 2)
        self.assertTrue(result["evaluation"]["decisions"][0]["allowed"])
        self.assertTrue(result["evaluation"]["decisions"][1]["allowed"])
        self.assertEqual(result["seen_event_ids"], ["e1", "e2"])

    def test_unsigned_event_is_not_accepted_as_input(self):
        event = {"time": 1, "sequence": 0, "type": "grant", "actor": "security_admin"}
        with self.assertRaises(ProvenanceError):
            evaluate_signed_trace([event], self.keys, self.entitlements)

    def test_tampered_event_fails_before_policy_evaluation(self):
        event = {
            "time": 1, "sequence": 0, "type": "grant", "grant_id": "g1",
            "actor": "security_admin", "principal": "alice", "scopes": ["read"],
            "tenant": "t1", "expires_at": 20, "policy_version": 1,
        }
        envelope = self.sign(event, "e1")
        envelope["event"]["principal"] = "mallory"
        with self.assertRaisesRegex(ProvenanceError, "invalid Ed25519 signature"):
            evaluate_signed_trace([envelope], self.keys, self.entitlements)

    def test_replay_ledger_candidate_is_not_returned_on_failed_trace(self):
        event = {"time": 1, "sequence": 0, "type": "read", "grant_id": "missing",
                 "principal": "alice", "scope": "read", "tenant": "t1",
                 "resource_id": "r1", "classification": 1, "clearance": 2}
        envelope = self.sign(event, "e1")
        # Schema-valid but unauthorized reads are decisions, not exceptions.
        result = evaluate_signed_trace([envelope], self.keys, self.entitlements)
        self.assertFalse(result["evaluation"]["decisions"][0]["allowed"])
        self.assertIn("e1", result["seen_event_ids"])

    def test_replayed_id_is_rejected_before_second_evaluation(self):
        event = {"time": 1, "sequence": 0, "type": "workflow_request",
                 "workflow_id": "w1", "requester": "alice", "revision": 1}
        envelope = self.sign(event, "e1")
        first = evaluate_signed_trace([envelope], self.keys)
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            evaluate_signed_trace([envelope], self.keys,
                                  seen_event_ids=set(first["seen_event_ids"]))

    def test_policy_schema_failure_does_not_return_replay_commit_token(self):
        # Signature is valid, but the event is malformed for the policy evaluator.
        envelope = self.sign({"time": 1, "sequence": 0, "type": "grant", "actor": "security_admin"}, "e1")
        with self.assertRaises(ValueError):
            evaluate_signed_trace([envelope], self.keys, self.entitlements)


if __name__ == "__main__":
    unittest.main()
