#!/usr/bin/env python3
"""Ed25519 provenance envelopes for G8.3 temporal UCON events.

This module authenticates event origin and detects replay within a caller-
maintained replay ledger. It is a research reference component, not a complete
production PKI, identity provider, trusted clock or durable ledger.
"""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

DOMAIN = "UCON-JEPA:event-provenance:v1"
SCHEMA_VERSION = 1
PRIVILEGED_ACTOR_EVENTS = frozenset({"policy_update", "grant", "revoke", "delegate"})
ENVELOPE_FIELDS = frozenset({"issuer", "key_id", "event_id", "event", "signature"})
KEY_FIELDS = frozenset({"issuer", "public_key", "valid_from", "valid_until", "revoked_at"})


class ProvenanceError(ValueError):
    """Raised when an event envelope fails provenance or replay validation."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProvenanceError(f"payload is not canonical-JSON serializable: {exc}") from exc


def _signed_message(issuer: str, key_id: str, event_id: str, event: dict) -> bytes:
    return _canonical_bytes({
        "domain": DOMAIN,
        "schema_version": SCHEMA_VERSION,
        "issuer": issuer,
        "key_id": key_id,
        "event_id": event_id,
        "event": event,
    })


def sign_event(
    event: Mapping[str, Any],
    *,
    issuer: str,
    key_id: str,
    event_id: str,
    private_key: Ed25519PrivateKey,
) -> dict:
    """Return a signed envelope; the event body is copied, never mutated."""
    if not isinstance(event, Mapping):
        raise ProvenanceError("event must be a mapping")
    if not all(isinstance(x, str) and x.strip() for x in (issuer, key_id, event_id)):
        raise ProvenanceError("issuer, key_id and event_id must be non-empty strings")
    body = dict(event)
    if type(body.get("time")) is not int or body["time"] < 0:
        raise ProvenanceError("event time must be a nonnegative integer")
    if body.get("type") in PRIVILEGED_ACTOR_EVENTS and body.get("actor") != issuer:
        raise ProvenanceError("privileged event actor must equal the signing issuer")
    message = _signed_message(issuer, key_id, event_id, body)
    try:
        signature = private_key.sign(message)
    except Exception as exc:
        raise ProvenanceError(f"event signing failed: {exc}") from exc
    return {
        "issuer": issuer,
        "key_id": key_id,
        "event_id": event_id,
        "event": body,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def _decode_b64(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{field} must be a non-empty base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProvenanceError(f"{field} is not valid base64") from exc


def _validate_key_record(key_id: str, record: Any) -> tuple[str, Ed25519PublicKey, int, int, int | None]:
    if not isinstance(record, Mapping):
        raise ProvenanceError(f"trusted key record {key_id!r} must be a mapping")
    missing = KEY_FIELDS.difference(record)
    if missing:
        raise ProvenanceError(f"trusted key record {key_id!r} is missing {sorted(missing)}")
    issuer = record["issuer"]
    if not isinstance(issuer, str) or not issuer.strip():
        raise ProvenanceError(f"trusted key record {key_id!r} has invalid issuer")
    valid_from, valid_until, revoked_at = (
        record["valid_from"], record["valid_until"], record["revoked_at"]
    )
    if type(valid_from) is not int or valid_from < 0:
        raise ProvenanceError(f"trusted key {key_id!r} valid_from must be a nonnegative integer")
    if type(valid_until) is not int or valid_until <= valid_from:
        raise ProvenanceError(f"trusted key {key_id!r} valid_until must exceed valid_from")
    if revoked_at is not None and (type(revoked_at) is not int or revoked_at < 0):
        raise ProvenanceError(f"trusted key {key_id!r} revoked_at must be a nonnegative integer or null")
    public_bytes = _decode_b64(record["public_key"], f"trusted key {key_id!r} public_key")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
    except ValueError as exc:
        raise ProvenanceError(f"trusted key {key_id!r} is not a valid Ed25519 public key") from exc
    return issuer, public_key, valid_from, valid_until, revoked_at


def verify_signed_trace(
    signed_events: Sequence[Mapping[str, Any]],
    trusted_keys: Mapping[str, Mapping[str, Any]],
    *,
    seen_event_ids: set[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """Verify a signed trace and return (plain events, updated replay-ID set).

    The caller must persist the returned ID set transactionally in a durable
    replay ledger. This function does not mutate the caller's input set.
    """
    if not isinstance(signed_events, (list, tuple)):
        raise ProvenanceError("signed_events must be a list or tuple")
    if not isinstance(trusted_keys, Mapping):
        raise ProvenanceError("trusted_keys must be a mapping")
    seen = set() if seen_event_ids is None else set(seen_event_ids)
    if any(not isinstance(item, str) or not item for item in seen):
        raise ProvenanceError("seen_event_ids must contain only non-empty strings")

    verified: list[dict] = []
    accepted = set(seen)

    for index, envelope in enumerate(signed_events):
        if not isinstance(envelope, Mapping):
            raise ProvenanceError(f"envelope {index} must be a mapping")
        if set(envelope) != ENVELOPE_FIELDS:
            missing = ENVELOPE_FIELDS.difference(envelope)
            extra = set(envelope).difference(ENVELOPE_FIELDS)
            raise ProvenanceError(
                f"envelope {index} fields invalid; missing={sorted(missing)}, extra={sorted(extra)}"
            )

        issuer, key_id, event_id = envelope["issuer"], envelope["key_id"], envelope["event_id"]
        if not all(isinstance(x, str) and x.strip() for x in (issuer, key_id, event_id)):
            raise ProvenanceError(f"envelope {index} issuer, key_id and event_id must be non-empty strings")
        if event_id in accepted:
            raise ProvenanceError(f"replayed event_id: {event_id}")
        event = envelope["event"]
        if not isinstance(event, Mapping):
            raise ProvenanceError(f"envelope {index} event body must be a mapping")
        body = dict(event)
        event_time = body.get("time")
        if type(event_time) is not int or event_time < 0:
            raise ProvenanceError(f"envelope {index} event time must be a nonnegative integer")

        record = trusted_keys.get(key_id)
        if record is None:
            raise ProvenanceError(f"unknown signing key: {key_id}")
        key_issuer, public_key, valid_from, valid_until, revoked_at = _validate_key_record(key_id, record)
        if issuer != key_issuer:
            raise ProvenanceError(f"issuer does not match trusted key owner for {key_id}")
        if not (valid_from <= event_time < valid_until):
            raise ProvenanceError(f"signing key {key_id} is outside its validity interval")
        if revoked_at is not None and event_time >= revoked_at:
            raise ProvenanceError(f"signing key {key_id} was revoked at event time")

        if body.get("type") in PRIVILEGED_ACTOR_EVENTS and body.get("actor") != issuer:
            raise ProvenanceError("privileged event actor does not match authenticated issuer")

        signature = _decode_b64(envelope["signature"], f"envelope {index} signature")
        try:
            public_key.verify(signature, _signed_message(issuer, key_id, event_id, body))
        except InvalidSignature as exc:
            raise ProvenanceError(f"invalid Ed25519 signature for event_id {event_id}") from exc

        verified.append(body)
        accepted.add(event_id)

    return verified, accepted
