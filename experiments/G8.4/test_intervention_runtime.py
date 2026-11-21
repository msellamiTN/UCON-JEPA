import unittest

from intervention_runtime import (
    CapabilityManifest,
    CapabilityOverlay,
    ProposalOutcome,
    UCONVerdict,
)


def manifest():
    return CapabilityManifest(
        base_capabilities=frozenset({"read", "write", "export", "admin"}),
        essential_capabilities=frozenset({"read", "write"}),
        restricted_capabilities=frozenset({"export", "admin"}),
    )


class CapabilityManifestTests(unittest.TestCase):
    def test_rejects_essential_capability_in_restricted_set(self):
        with self.assertRaisesRegex(ValueError, "essential capabilities cannot be restricted"):
            CapabilityManifest(
                frozenset({"read", "export"}),
                frozenset({"read"}),
                frozenset({"read"}),
            )

    def test_rejects_restricted_capability_outside_base(self):
        with self.assertRaisesRegex(ValueError, "subset of base"):
            CapabilityManifest(
                frozenset({"read"}),
                frozenset({"read"}),
                frozenset({"export"}),
            )

    def test_rejects_empty_or_invalid_names(self):
        with self.assertRaises(ValueError):
            CapabilityManifest(frozenset({""}), frozenset(), frozenset())

    def test_rejects_mutable_capability_collections(self):
        with self.assertRaisesRegex(TypeError, "must be a frozenset"):
            CapabilityManifest({"read"}, frozenset({"read"}), frozenset())

    def test_rejects_empty_base_and_non_intervenable_manifest(self):
        with self.assertRaisesRegex(ValueError, "base_capabilities must not be empty"):
            CapabilityManifest(frozenset(), frozenset(), frozenset({"export"}))
        with self.assertRaisesRegex(ValueError, "intervention-eligible"):
            CapabilityManifest(frozenset({"read"}), frozenset({"read"}), frozenset())


class CapabilityOverlayTests(unittest.TestCase):
    def test_overlay_applies_to_exactly_two_decision_opportunities(self):
        overlay = CapabilityOverlay(manifest(), duration_decisions=2)
        self.assertTrue(overlay.trigger("JEPA risk threshold"))
        self.assertEqual(overlay.trigger_reason, "JEPA risk threshold")
        self.assertEqual(overlay.remaining_decisions, 2)

        first = overlay.begin_decision()
        self.assertEqual(first, frozenset({"read", "write"}))
        self.assertFalse(overlay.end_decision())
        self.assertEqual(overlay.remaining_decisions, 1)

        second = overlay.begin_decision()
        self.assertEqual(second, frozenset({"read", "write"}))
        self.assertTrue(overlay.end_decision())
        self.assertEqual(overlay.remaining_decisions, 0)

        third = overlay.begin_decision()
        self.assertEqual(third, frozenset({"read", "write", "export", "admin"}))
        self.assertFalse(overlay.end_decision())

    def test_duplicate_trigger_does_not_extend_overlay(self):
        overlay = CapabilityOverlay(manifest(), duration_decisions=2)
        self.assertTrue(overlay.trigger("first trigger"))
        overlay.begin_decision()
        overlay.end_decision()
        self.assertFalse(overlay.trigger("duplicate trigger"))
        self.assertEqual(overlay.remaining_decisions, 1)

    def test_overlay_never_expands_or_restricts_essential_capabilities(self):
        overlay = CapabilityOverlay(manifest(), duration_decisions=2)
        overlay.trigger("test")
        for _ in range(3):
            visible = overlay.begin_decision()
            self.assertLessEqual(visible, manifest().base_capabilities)
            self.assertTrue(manifest().essential_capabilities <= visible)
            overlay.end_decision()

    def test_ucon_deny_dominates_even_when_overlay_is_inactive(self):
        overlay = CapabilityOverlay(manifest())
        overlay.begin_decision()
        decision = overlay.evaluate_proposal("read", lambda _: UCONVerdict.DENY)
        self.assertEqual(decision.outcome, ProposalOutcome.DENIED_BY_UCON)
        self.assertFalse(decision.may_execute)
        overlay.end_decision()

    def test_overlay_restriction_never_overrides_ucon_or_skips_ucon_call(self):
        overlay = CapabilityOverlay(manifest())
        overlay.trigger("test")
        overlay.begin_decision()
        calls = []
        decision = overlay.evaluate_proposal(
            "export",
            lambda proposal: calls.append(proposal) or UCONVerdict.PERMIT,
        )
        self.assertEqual(calls, ["export"])
        self.assertEqual(decision.outcome, ProposalOutcome.SUPPRESSED_BY_OVERLAY)
        self.assertFalse(decision.may_execute)
        overlay.end_decision()

    def test_ucon_failure_fails_closed(self):
        overlay = CapabilityOverlay(manifest())
        overlay.begin_decision()

        def broken(_):
            raise RuntimeError("policy engine unavailable")

        decision = overlay.evaluate_proposal("read", broken)
        self.assertEqual(decision.outcome, ProposalOutcome.UCON_FAILURE_DENY)
        self.assertFalse(decision.may_execute)
        overlay.end_decision()

    def test_invalid_ucon_verdict_fails_closed(self):
        overlay = CapabilityOverlay(manifest())
        overlay.begin_decision()
        decision = overlay.evaluate_proposal("read", lambda _: "PERMIT")
        self.assertEqual(decision.outcome, ProposalOutcome.UCON_FAILURE_DENY)
        self.assertFalse(decision.may_execute)
        overlay.end_decision()

    def test_current_ucon_policy_is_rechecked_after_policy_change(self):
        overlay = CapabilityOverlay(manifest())
        state = {"permitted": True}
        overlay.begin_decision()
        first = overlay.evaluate_proposal(
            "read",
            lambda _: UCONVerdict.PERMIT if state["permitted"] else UCONVerdict.DENY,
        )
        self.assertEqual(first.outcome, ProposalOutcome.READY)
        overlay.end_decision()

        state["permitted"] = False
        overlay.begin_decision()
        second = overlay.evaluate_proposal(
            "read",
            lambda _: UCONVerdict.PERMIT if state["permitted"] else UCONVerdict.DENY,
        )
        self.assertEqual(second.outcome, ProposalOutcome.DENIED_BY_UCON)
        self.assertFalse(second.may_execute)
        overlay.end_decision()

    def test_unknown_or_restricted_proposal_cannot_become_executable(self):
        overlay = CapabilityOverlay(manifest())
        overlay.begin_decision()
        decision = overlay.evaluate_proposal("root-shell", lambda _: UCONVerdict.PERMIT)
        self.assertEqual(decision.outcome, ProposalOutcome.SUPPRESSED_BY_OVERLAY)
        self.assertFalse(decision.may_execute)
        overlay.end_decision()

    def test_requires_well_formed_decision_lifecycle(self):
        overlay = CapabilityOverlay(manifest())
        with self.assertRaisesRegex(RuntimeError, "begin_decision"):
            overlay.end_decision()
        overlay.begin_decision()
        with self.assertRaisesRegex(RuntimeError, "previous decision"):
            overlay.begin_decision()
        with self.assertRaisesRegex(RuntimeError, "decision boundary"):
            overlay.trigger("illegal mid-decision trigger")
        overlay.end_decision()


if __name__ == "__main__":
    unittest.main()
