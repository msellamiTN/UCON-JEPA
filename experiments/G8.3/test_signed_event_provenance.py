import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from signed_event_provenance import EventVerifier, ProvenanceError, TrustedKey, sign_event


class SignedEventProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.verifier = EventVerifier({"key-alice-v1": TrustedKey("alice", public)})
        self.event = {"time": 1, "sequence": 0, "type": "revoke",
                      "grant_id": "g-1", "actor": "alice"}

    def envelope(self, event=None, nonce="n-1"):
        return sign_event(self.private, "key-alice-v1", nonce,
                          self.event if event is None else event)

    def test_valid_signature_returns_authenticated_principal(self):
        result = self.verifier.verify(self.envelope())
        self.assertEqual(result["authenticated_actor"], "alice")
        self.assertTrue(result["provenance"]["signature_verified"])

    def test_tampered_event_is_rejected(self):
        envelope = self.envelope()
        envelope["event"]["grant_id"] = "g-2"
        with self.assertRaises(ProvenanceError):
            self.verifier.verify(envelope)

    def test_actor_spoofing_is_rejected(self):
        forged = dict(self.event, actor="mallory")
        with self.assertRaises(ProvenanceError):
            self.verifier.verify(self.envelope(forged))

    def test_replay_nonce_is_rejected(self):
        envelope = self.envelope()
        self.verifier.verify(envelope)
        with self.assertRaisesRegex(ProvenanceError, "replayed"):
            self.verifier.verify(envelope)

    def test_same_nonce_from_different_key_is_separate_namespace(self):
        other = Ed25519PrivateKey.generate()
        public = other.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.verifier.keys["key-bob-v1"] = TrustedKey("bob", public)
        event = {"time": 2, "sequence": 0, "type": "read"}
        self.verifier.verify(self.envelope(event, "shared"))
        envelope = sign_event(other, "key-bob-v1", "shared", event)
        self.assertEqual(self.verifier.verify(envelope)["authenticated_actor"], "bob")

    def test_revoked_key_is_rejected(self):
        key = self.verifier.keys["key-alice-v1"]
        self.verifier.keys["key-alice-v1"] = TrustedKey(key.principal, key.public_key, True)
        with self.assertRaisesRegex(ProvenanceError, "revoked"):
            self.verifier.verify(self.envelope())

    def test_unknown_key_is_rejected(self):
        envelope = self.envelope()
        envelope["key_id"] = "unknown"
        with self.assertRaises(ProvenanceError):
            self.verifier.verify(envelope)

    def test_malformed_envelope_is_rejected(self):
        for value in (None, [], {}, {"key_id": "x"}):
            with self.subTest(value=value), self.assertRaises(ProvenanceError):
                self.verifier.verify(value)

    def test_noncanonical_float_payload_is_rejected(self):
        with self.assertRaises(ProvenanceError):
            EventVerifier.canonical_bytes({"x": 1.25})

    def test_signature_is_bound_to_key_id_and_nonce(self):
        envelope = self.envelope()
        envelope["nonce"] = "changed"
        with self.assertRaises(ProvenanceError):
            self.verifier.verify(envelope)


if __name__ == "__main__":
    unittest.main()
