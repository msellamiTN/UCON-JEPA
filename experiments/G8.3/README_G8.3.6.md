# G8.3.6 — Authenticated signed event provenance

## Objective
Ensure privileged event identity is authenticated by a cryptographic key rather than trusted from a self-asserted `actor` field.

## Implemented envelope
The existing `event_provenance.py` module signs a domain-separated canonical JSON message containing:
- domain and schema version
- issuer and key identifier
- unique event identifier
- complete event body

Ed25519 signatures are verified against an out-of-band trusted-key registry. The registry binds key IDs to issuers and defines validity intervals and revocation time.

## Security checks
- signature integrity and domain separation
- authenticated issuer-to-key binding
- privileged actor-to-issuer binding
- key validity interval and revocation
- event-ID replay detection, including across calls when the caller supplies the returned replay set
- strict envelope fields and base64 validation

## Important limits
- The caller must persist the replay-ID set transactionally in a durable shared ledger; in-memory state alone is insufficient for distributed deployment.
- Trusted-key provisioning, rotation workflows, secure registry distribution, KMS/HSM custody and trusted timestamping are external responsibilities.
- Canonicalization uses deterministic Python JSON serialization; cross-language interoperability still needs a formally specified canonicalization profile and published test vectors.
- Provenance verification authenticates origin. The policy evaluator must still enforce authorization; cryptographic authenticity does not imply permission or prove policy correctness.

## Reproduction
```bash
python -m pip install cryptography
python -m unittest discover -s experiments/G8.3 -p 'test_event_provenance.py' -v
```
