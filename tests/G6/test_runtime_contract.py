"""Executable reference contract for the G6 runtime-enforcement boundary.

This is an abstract protocol test harness, not production gateway code.
The acceptance method revalidates authoritative policy and revocation state
at the external-effect linearization point.
"""
from dataclasses import dataclass
from itertools import permutations
import unittest


class Denied(RuntimeError):
    pass


@dataclass
class Invocation:
    invocation_id: str
    action: str
    principal: str
    capability: str
    ucon_allowed: bool
    status: str = "REQUESTED"
    cert_policy: int | None = None
    gate_passed: bool = False
    accepted: bool = False
    accepted_policy: int | None = None
    accepted_while_revoked: bool = False
    observed: bool = False
    verified: bool = False
    committed: bool = False
    outcome_unknown: bool = False


class ReferenceRuntime:
    """Small executable abstraction of the G6 acceptance protocol."""

    def __init__(self):
        self.policy_version = 1
        self.revoked_capabilities = set()
        self.invocations: dict[str, Invocation] = {}
        self.consumed_invocations = set()

    def request(self, invocation_id, action, principal, capability, ucon_allowed):
        if invocation_id in self.invocations:
            raise Denied("duplicate invocation identifier")
        inv = Invocation(
            invocation_id, action, principal, capability, bool(ucon_allowed)
        )
        self.invocations[invocation_id] = inv
        return inv

    def authorize(self, invocation_id):
        inv = self.invocations[invocation_id]
        if inv.status != "REQUESTED":
            raise Denied("invalid lifecycle state")
        if not inv.ucon_allowed:
            inv.status = "DENIED"
            raise Denied("UCON denied")
        inv.cert_policy = self.policy_version
        inv.status = "AUTHORIZED"

    def gate_check(self, invocation_id, *, action, principal, capability):
        inv = self.invocations[invocation_id]
        if inv.status != "AUTHORIZED":
            raise Denied("not formally authorized")
        if (action, principal, capability) != (
            inv.action, inv.principal, inv.capability
        ):
            raise Denied("certificate/action/principal binding mismatch")
        if inv.cert_policy != self.policy_version:
            raise Denied("stale policy certificate")
        if capability in self.revoked_capabilities:
            raise Denied("capability revoked")
        inv.gate_passed = True
        inv.status = "DISPATCHED"

    def dispatch(self, invocation_id):
        inv = self.invocations[invocation_id]
        if inv.status != "DISPATCHED" or not inv.gate_passed:
            raise Denied("dispatch requires successful gate check")
        # Dispatch is not external acceptance.
        return True

    def accept(self, invocation_id):
        """Linearization point: recheck authoritative state atomically."""
        inv = self.invocations[invocation_id]
        if inv.status != "DISPATCHED" or not inv.gate_passed:
            raise Denied("no live gated dispatch")
        if not inv.ucon_allowed:
            raise Denied("formal UCON denial")
        if inv.capability in self.revoked_capabilities:
            raise Denied("capability revoked before acceptance")
        if inv.cert_policy != self.policy_version:
            raise Denied("policy changed before acceptance")
        if invocation_id in self.consumed_invocations:
            raise Denied("invocation already consumed")
        self.consumed_invocations.add(invocation_id)
        inv.accepted = True
        inv.accepted_policy = self.policy_version
        inv.accepted_while_revoked = (
            inv.capability in self.revoked_capabilities
        )
        inv.status = "ACCEPTED"

    def effective_revoke(self, capability):
        self.revoked_capabilities.add(capability)

    def publish_policy(self):
        self.policy_version += 1

    def observe(self, invocation_id, *, receipt_correlated):
        inv = self.invocations[invocation_id]
        if inv.status != "ACCEPTED" or not inv.accepted:
            raise Denied("no accepted effect to observe")
        if not receipt_correlated:
            inv.status = "UNKNOWN"
            inv.outcome_unknown = True
            return
        inv.observed = True
        inv.status = "OBSERVED"

    def verify(self, invocation_id, *, mutation_matches):
        inv = self.invocations[invocation_id]
        if inv.status != "OBSERVED" or not inv.observed:
            raise Denied("observation required")
        if not mutation_matches:
            inv.status = "UNKNOWN"
            inv.outcome_unknown = True
            return
        inv.verified = True
        inv.status = "VERIFIED"

    def commit(self, invocation_id):
        inv = self.invocations[invocation_id]
        if inv.status != "VERIFIED" or not inv.verified or not inv.observed:
            raise Denied("commit requires verified effect evidence")
        inv.committed = True
        inv.status = "COMMITTED"

    def lose_acceptance_response(self, invocation_id):
        inv = self.invocations[invocation_id]
        if inv.status != "ACCEPTED" or not inv.accepted:
            raise Denied("no accepted response to lose")
        # Remote acceptance remains true; only the caller's knowledge is unknown.
        inv.status = "UNKNOWN"
        inv.outcome_unknown = True


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.rt = ReferenceRuntime()
        self.rt.request("x1", "write", "svc-agent", "cap-A", True)

    def authorize_and_gate(self):
        self.rt.authorize("x1")
        self.rt.gate_check(
            "x1", action="write", principal="svc-agent", capability="cap-A"
        )
        self.rt.dispatch("x1")

    def test_happy_path_requires_verified_effect(self):
        self.authorize_and_gate()
        self.rt.accept("x1")
        self.rt.observe("x1", receipt_correlated=True)
        self.rt.verify("x1", mutation_matches=True)
        self.rt.commit("x1")
        inv = self.rt.invocations["x1"]
        self.assertTrue(inv.accepted and inv.observed and inv.verified and inv.committed)

    def test_jepa_positive_cannot_override_ucon_denial(self):
        self.rt.invocations["x1"].ucon_allowed = False
        with self.assertRaises(Denied):
            self.rt.authorize("x1")
        # JEPA is deliberately not an input to any authorization method.
        jepa_output = "ALLOW_RECOMMENDATION"
        self.assertEqual(jepa_output, "ALLOW_RECOMMENDATION")
        self.assertEqual(self.rt.invocations["x1"].status, "DENIED")
        with self.assertRaises(Denied):
            self.rt.accept("x1")

    def test_revocation_after_authorization_before_acceptance(self):
        self.authorize_and_gate()
        self.rt.effective_revoke("cap-A")
        with self.assertRaises(Denied):
            self.rt.accept("x1")
        self.assertFalse(self.rt.invocations["x1"].accepted)

    def test_policy_update_after_authorization_before_acceptance(self):
        self.authorize_and_gate()
        self.rt.publish_policy()
        with self.assertRaises(Denied):
            self.rt.accept("x1")
        self.assertFalse(self.rt.invocations["x1"].accepted)

    def test_gate_binds_action_principal_and_capability(self):
        self.rt.authorize("x1")
        with self.assertRaises(Denied):
            self.rt.gate_check(
                "x1", action="delete", principal="svc-agent", capability="cap-A"
            )
        self.assertFalse(self.rt.invocations["x1"].gate_passed)

    def test_duplicate_invocation_identifier_is_rejected(self):
        with self.assertRaises(Denied):
            self.rt.request("x1", "delete", "svc-agent", "cap-A", True)

    def test_commit_before_verification_is_rejected(self):
        self.authorize_and_gate()
        self.rt.accept("x1")
        self.rt.observe("x1", receipt_correlated=True)
        with self.assertRaises(Denied):
            self.rt.commit("x1")
        self.assertFalse(self.rt.invocations["x1"].committed)

    def test_unmatched_receipt_never_commits(self):
        self.authorize_and_gate()
        self.rt.accept("x1")
        self.rt.observe("x1", receipt_correlated=False)
        with self.assertRaises(Denied):
            self.rt.commit("x1")
        self.assertTrue(self.rt.invocations["x1"].outcome_unknown)

    def test_lost_acceptance_response_preserves_remote_effect_but_blocks_commit(self):
        self.authorize_and_gate()
        self.rt.accept("x1")
        self.rt.lose_acceptance_response("x1")
        inv = self.rt.invocations["x1"]
        self.assertTrue(inv.accepted)
        self.assertTrue(inv.outcome_unknown)
        with self.assertRaises(Denied):
            self.rt.commit("x1")

    def test_revocation_and_policy_interleavings(self):
        # Every schedule in which either event precedes Accept must block Accept.
        for schedule in permutations(("revoke", "policy", "accept")):
            rt = ReferenceRuntime()
            rt.request("i", "write", "principal", "cap", True)
            rt.authorize("i")
            rt.gate_check("i", action="write", principal="principal", capability="cap")
            rt.dispatch("i")
            accepted = False
            for event in schedule:
                if event == "revoke":
                    rt.effective_revoke("cap")
                elif event == "policy":
                    rt.publish_policy()
                else:
                    try:
                        rt.accept("i")
                        accepted = True
                    except Denied:
                        accepted = False
            if schedule.index("accept") > schedule.index("revoke") or schedule.index("accept") > schedule.index("policy"):
                self.assertFalse(accepted, schedule)
            else:
                self.assertTrue(accepted, schedule)


if __name__ == "__main__":
    unittest.main()
