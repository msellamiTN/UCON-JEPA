# G8.3.6 — Signed event provenance

## Objective
Replace self-asserted event identity with a verifiable principal binding. The verifier accepts an event only when its Ed25519 signature validates against an out-of-band trusted-key registry, the key is active, and the event actor (when present) agrees with the registry principal.

## Envelope
An envelope contains exactly `key_id`, `nonce`, `event`, and base64 `signature`. The signature covers the canonical serialization of `{key_id, nonce, event}`; changing the event, key identifier, or nonce invalidates it.

## Canonicalization profile
This prototype uses UTF-8 JSON with sorted object keys, compact separators, non-string object keys rejected, and floating-point values forbidden. It is a constrained deterministic JSON profile, **not** a full RFC 8785 implementation. Interoperable production deployments must adopt a formally specified canonical format and publish test vectors.

## Security properties tested
- Ed25519 signature verification and tamper detection
- Binding between signing key, trusted principal and event actor
- Unknown/revoked key rejection
- Per-key nonce replay rejection
- Signature binding to key ID and nonce

## Explicit limitations
- The replay cache is process-local. Production must use durable, atomic replay protection shared across verifier replicas.
- The key registry is an injected trusted input; secure provisioning, rotation, revocation distribution and audit are not implemented.
- Private-key custody must be delegated to KMS/HSM; the helper in this module is for tests/producers only.
- This layer authenticates event origin; it does not prove authorization-policy correctness or replace the G8.3 evaluator.
- No trusted timestamp/freshness window is enforced yet; logical event time is not wall-clock time.

## Reproduction
```bash
python -m pip install cryptography
python -m unittest discover -s experiments/G8.3 -p 'test_signed_event_provenance.py' -v
```
