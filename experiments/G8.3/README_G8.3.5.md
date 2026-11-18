# G8.3.5 — Actor-bound privileged temporal transitions

## Integration note

This branch now targets `main` directly and includes the strict event-schema validation changes from PR #9 in its comparison with `main`. This allows the combined semantic increment to be validated as one integration candidate rather than merging a duplicate stacked branch first.

## Finding

The earlier temporal model validated grant scope, expiry and policy version but did not bind three state-changing event types to an authorized actor. A caller could attempt to delegate a grant using only its identifier; revoke and policy-update events also lacked actor authorization.

## Corrected rules

1. **Policy update:** requires `actor=security_admin` and the trusted `admin` entitlement. Unauthorized updates are denied and do not change the policy version.
2. **Revocation:** requires the recorded grant issuer, or `security_admin` with the trusted `admin` entitlement. Unauthorized revocations are denied and do not modify revocation state.
3. **Delegation:** requires `actor` to equal the principal of the active source grant. Scope-subset, expiry, revocation and policy-version checks remain in force.

The event contract now requires the actor field for these three event types. The evaluator stores the grant issuer at grant creation.

## Regression coverage

Tests cover an unauthorized delegate attempting to use another principal's grant, successful delegation by the source principal, unauthorized revoke without state mutation, unauthorized policy update without version mutation, missing actor fields, and absence of the admin entitlement.

## Boundary

This is a research reference model. It does not yet define a complete identity-proofing, authentication, signature, trusted-event-ingestion or key-management protocol. The `actor` field is a semantic principal identifier, not cryptographic proof of who emitted the event.
