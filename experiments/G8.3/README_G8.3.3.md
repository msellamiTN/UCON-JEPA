# G8.3.3 — Strict temporal event schema

## Objective

Make the G8.3 reference evaluator reject malformed input deterministically before policy evaluation. This closes type-confusion and malformed-container gaps in the event contract; it does not change the authorization rules for well-formed events.

## Validation rules

- Event timestamps and sequence numbers are nonnegative integers; Python booleans are not accepted as integers.
- Event types are non-empty strings from the declared vocabulary.
- Identifiers and textual policy fields are non-empty strings.
- Scope collections are non-empty lists of non-empty strings.
- Versions and workflow revisions are positive integers.
- Classification and clearance are nonnegative integers.
- Out-of-band trusted entitlements must be a mapping from non-empty actor names to lists of non-empty scope strings.
- Unknown event types, missing fields, malformed values, and non-monotonic event order raise `TraceError`.

## Regression cases

The test suite includes boolean-as-integer rejection, malformed workflow revisions, unhashable scope payloads, malformed trusted entitlement mappings, non-string event types, unknown event types, and non-monotonic event ordering.

## Scientific and security boundary

This is input-contract hardening for a research reference model. It is not a production enforcement point, a proof of the policy model, or evidence of deployment security. The model's authorization semantics remain those defined in `policy_spec.json`.
