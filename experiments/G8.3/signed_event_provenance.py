#!/usr/bin/env python3
"""G8.3.6 authenticated event envelope verification.

The verifier returns a normalized event only after checking an Ed25519
signature against an out-of-band key registry and enforcing key revocation,
principal binding, and per-key nonce uniqueness.

Canonicalization is a deliberately restricted JSON profile: UTF-8, sorted
object keys, compact separators, no floats, no non-string object keys, and
only JSON scalar/list/object values. This is NOT claimed to implement the
full RFC 8785 JCS number model.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class ProvenanceError(ValueError):
    """Raised when an event envelope cannot be authenticated."""


@dataclass(frozen=True)
class TrustedKey:
    principal: str
    public_key: bytes
    revoked: bool = False


@dataclass
class EventVerifier:
    """Stateful verifier; replay cache is process-local and must be persisted in production."""
    keys: dict[str, TrustedKey]
    seen_nonces: set[tuple[str, str]] = field(default_factory=set)

    @staticmethod
    def canonical_bytes(value: Any) -> bytes:
        def validate(node: Any) -> None:
            if node is None or isinstance(node, (str, bool)):
                return
            if type(node) is int:
                return
            if isinstance(node, float):
                raise ProvenanceError("floating-point values are forbidden in signed payloads")
            if isinstance(node, list):
                for item in node:
                    validate(item)
                return
            if isinstance(node, dict):
                if any(not isinstance(k, str) for k in node):
                    raise ProvenanceError("signed object keys must be strings")
                for item in node.values():
                    validate(item)
                return
            raise ProvenanceError(f"unsupported signed value: {type(node).__name__}")
        validate(value)
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ProvenanceError("payload cannot be canonically serialized") from exc

    @classmethod
    def signed_bytes(cls, key_id: str, nonce: str, event: dict) -> bytes:
        return cls.canonical_bytes({"key_id": key_id, "nonce": nonce, "event": event})

    def verify(self, envelope: dict) -> dict:
        if not isinstance(envelope, dict):
            raise ProvenanceError("envelope must be an object")
        required = {"key_id", "nonce", "event", "signature"}
        if set(envelope) != required:
            raise ProvenanceError("envelope must contain exactly key_id, nonce, event, signature")
        key_id, nonce, event, signature = (envelope[k] for k in
                                            ("key_id", "nonce", "event", "signature"))
        if not isinstance(key_id, str) or not key_id.strip():
            raise ProvenanceError("key_id must be a non-empty string")
        if not isinstance(nonce, str) or not nonce.strip():
            raise ProvenanceError("nonce must be a non-empty string")
        if not isinstance(event, dict):
            raise ProvenanceError("event must be an object")
        if not isinstance(signature, str):
            raise ProvenanceError("signature must be base64 text")
        key = self.keys.get(key_id)
        if key is None or key.revoked:
            raise ProvenanceError("unknown or revoked signing key")
        if not isinstance(key.principal, str) or not key.principal.strip():
            raise ProvenanceError("trusted key has invalid principal binding")
        replay_key = (key_id, nonce)
        if replay_key in self.seen_nonces:
            raise ProvenanceError("replayed event nonce")
        if "actor" in event and event["actor"] != key.principal:
            raise ProvenanceError("event actor does not match authenticated key principal")
        try:
            sig = base64.b64decode(signature, validate=True)
            public_key = Ed25519PublicKey.from_public_bytes(key.public_key)
            public_key.verify(sig, self.signed_bytes(key_id, nonce, event))
        except (ValueError, InvalidSignature) as exc:
            raise ProvenanceError("invalid event signature") from exc
        self.seen_nonces.add(replay_key)
        authenticated = dict(event)
        authenticated["authenticated_actor"] = key.principal
        authenticated["provenance"] = {"key_id": key_id, "nonce": nonce,
                                       "signature_verified": True}
        return authenticated


def sign_event(private_key, key_id: str, nonce: str, event: dict) -> dict:
    """Test/producer helper. Production private keys must remain in a KMS/HSM."""
    signature = private_key.sign(EventVerifier.signed_bytes(key_id, nonce, event))
    return {"key_id": key_id, "nonce": nonce, "event": event,
            "signature": base64.b64encode(signature).decode("ascii")}
