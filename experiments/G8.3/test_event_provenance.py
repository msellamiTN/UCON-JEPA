import base64
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from event_provenance import ProvenanceError, sign_event, verify_signed_trace


class SignedEventProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        public_bytes = self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.registry = {
            "security-admin-k1": {
                "issuer": "security_admin",
                "public_key": base64.b64encode(public_bytes).decode("ascii"),
                "valid_from": 0,
                "valid_until": 1000,
                "revoked_at": None,
            }
        }

    def grant_event(self, time=10):
        return {
            "time": time, "sequence": 0, "type": "grant",
            "grant_id": "g1", "actor": "security_admin",
            "principal": "alice", "scopes": ["read"],
            "tenant": "tenant-a", "expires_at": 100,
            "policy_version": 1,
        }

    def sign(self, event=None, event_id="evt-1"):
        return sign_event(
            self.grant_event() if event is None else event,
            issuer="security_admin",
            key_id="security-admin-k1",
            event_id=event_id,
            private_key=self.private,
        )

    def test_valid_signature_verifies_and_returns_plain_event(self):
        envelope = self.sign()
        events, seen = verify_signed_trace([envelope], self.registry)
        self.assertEqual(events, [self.grant_event()])
        self.assertEqual(seen, {"evt-1"})

    def test_canonicalization_is_independent_of_mapping_insertion_order(self):
        event = self.grant_event()
        reversed_event = dict(reversed(list(event.items())))
        envelope = self.sign(event)
        envelope["event"] = reversed_event
        events, _ = verify_signed_trace([envelope], self.registry)
        self.assertEqual(events[0], event)

    def test_body_tampering_invalidates_signature(self):
        envelope = self.sign()
        envelope["event"]["principal"] = "mallory"
        with self.assertRaisesRegex(ProvenanceError, "invalid Ed25519 signature"):
            verify_signed_trace([envelope], self.registry)

    def test_privileged_actor_must_equal_signing_issuer(self):
        event = self.grant_event()
        event["actor"] = "mallory"
        with self.assertRaisesRegex(ProvenanceError, "actor must equal"):
            self.sign(event)

    def test_verifier_rejects_actor_substitution_even_with_valid_signature(self):
        envelope = self.sign()
        envelope["event"]["actor"] = "mallory"
        # Re-sign the altered event as the authenticated principal to test the
        # actor-binding guard independently of signature-integrity validation.
        envelope = sign_event(
            envelope["event"], issuer="security_admin",
            key_id="security-admin-k1", event_id="evt-2",
            private_key=self.private,
        )
        with self.assertRaisesRegex(ProvenanceError, "actor does not match"):
            verify_signed_trace([envelope], self.registry)

    def test_unknown_key_is_rejected(self):
        envelope = self.sign()
        with self.assertRaisesRegex(ProvenanceError, "unknown signing key"):
            verify_signed_trace([envelope], {})

    def test_key_owner_must_match_envelope_issuer(self):
        envelope = self.sign()
        envelope["issuer"] = "mallory"
        with self.assertRaisesRegex(ProvenanceError, "issuer does not match"):
            verify_signed_trace([envelope], self.registry)

    def test_key_validity_interval_is_enforced(self):
        envelope = self.sign(self.grant_event(time=1000))
        with self.assertRaisesRegex(ProvenanceError, "outside its validity"):
            verify_signed_trace([envelope], self.registry)

    def test_key_revocation_is_effective_at_revocation_time(self):
        self.registry["security-admin-k1"]["revoked_at"] = 10
        envelope = self.sign(self.grant_event(time=9))
        events, _ = verify_signed_trace([envelope], self.registry)
        self.assertEqual(len(events), 1)
        revoked_event = self.sign(self.grant_event(time=10), event_id="evt-2")
        with self.assertRaisesRegex(ProvenanceError, "was revoked"):
            verify_signed_trace([revoked_event], self.registry)

    def test_replay_is_rejected_across_calls(self):
        envelope = self.sign()
        _, seen = verify_signed_trace([envelope], self.registry)
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            verify_signed_trace([envelope], self.registry, seen_event_ids=seen)

    def test_duplicate_event_ids_within_trace_are_rejected(self):
        first = self.sign(event_id="same")
        second = self.sign(event_id="same")
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            verify_signed_trace([first, second], self.registry)

    def test_caller_replay_ledger_is_not_mutated(self):
        seen = {"previous-event"}
        with self.assertRaisesRegex(ProvenanceError, "replayed event_id"):
            verify_signed_trace([self.sign(event_id="previous-event")], self.registry, seen_event_ids=seen)
        self.assertEqual(seen, {"previous-event"})

    def test_signature_must_be_valid_base64(self):
        envelope = self.sign()
        envelope["signature"] = "not-base64!"
        with self.assertRaisesRegex(ProvenanceError, "not valid base64"):
            verify_signed_trace([envelope], self.registry)

    def test_envelope_rejects_unrecognized_fields(self):
        envelope = self.sign()
        envelope["untrusted_role"] = "admin"
        with self.assertRaisesRegex(ProvenanceError, "fields invalid"):
            verify_signed_trace([envelope], self.registry)

    def test_non_privileged_event_can_be_signed_by_its_service_issuer(self):
        event = {
            "time": 11, "sequence": 0, "type": "read",
            "grant_id": "g1", "principal": "alice", "scope": "read",
            "tenant": "tenant-a", "resource_id": "r1",
            "classification": 1, "clearance": 2,
        }
        envelope = sign_event(
            event, issuer="data-agent", key_id="security-admin-k1",
            event_id="read-1", private_key=self.private,
        )
        # A key must be registered to the authenticated issuer; this key is
        # intentionally registered to security_admin, so the issuer mismatch
        # is rejected rather than inferred from the event's subject.
        with self.assertRaisesRegex(ProvenanceError, "issuer does not match"):
            verify_signed_trace([envelope], self.registry)


if __name__ == "__main__":
    unittest.main()
