import unittest

from jepa_roles import (
    Advisory,
    FutureTarget,
    JEPAExample,
    OperationalDecision,
    PredictiveAssessment,
    PredictorInput,
    compose_operational_decision,
)


class JEPARoleContractTests(unittest.TestCase):
    def test_predictor_input_contains_only_context_action_and_horizon(self):
        x = PredictorInput.from_mapping({
            "context_latent": [0.1, 0.2],
            "action_latent": [0.3, 0.4],
            "horizon": 2,
        })
        self.assertEqual(x.horizon, 2)
        self.assertEqual(x.context_latent, (0.1, 0.2))

    def test_future_observation_and_oracle_label_are_rejected_as_predictor_inputs(self):
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            PredictorInput.from_mapping({
                "context_latent": [0.1],
                "action_latent": [0.2],
                "horizon": 1,
                "future_observation": {"role": "admin"},
            })
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            PredictorInput.from_mapping({
                "context_latent": [0.1],
                "action_latent": [0.2],
                "horizon": 1,
                "violation_label": True,
            })

    def test_target_data_is_separate_from_predictor_input(self):
        example = JEPAExample(
            predictor_input=PredictorInput((0.1,), (0.2,), 1),
            target=FutureTarget(
                future_observation={"clearance": 1},
                security_labels={"policy_violation": True},
            ),
        )
        self.assertFalse(hasattr(example.predictor_input, "future_observation"))
        self.assertFalse(hasattr(example.predictor_input, "security_labels"))
        self.assertTrue(example.target.security_labels["policy_violation"])

    def test_candidate_action_changes_action_latent_interface(self):
        first = PredictorInput((0.5, 0.7), (1.0, 0.0), 1)
        second = PredictorInput((0.5, 0.7), (0.0, 1.0), 1)
        self.assertEqual(first.context_latent, second.context_latent)
        self.assertNotEqual(first.action_latent, second.action_latent)

    def test_formal_deny_dominates_every_predictive_advisory(self):
        for advisory in Advisory:
            assessment = PredictiveAssessment(advisory, 0.99, 3)
            self.assertEqual(
                compose_operational_decision(False, assessment),
                OperationalDecision.DENY,
            )

    def test_advisory_can_constrain_but_cannot_create_authorization(self):
        expected = {
            Advisory.NONE: OperationalDecision.ALLOW,
            Advisory.RESTRICT: OperationalDecision.RESTRICT,
            Advisory.ESCALATE: OperationalDecision.ESCALATE,
            Advisory.ABSTAIN: OperationalDecision.ABSTAIN,
        }
        for advisory, decision in expected.items():
            self.assertEqual(
                compose_operational_decision(
                    True, PredictiveAssessment(advisory, 0.5, 1)
                ),
                decision,
            )

    def test_invalid_latents_and_risk_are_rejected(self):
        with self.assertRaises(ValueError):
            PredictorInput((), (0.1,))
        with self.assertRaises(ValueError):
            PredictorInput((float("nan"),), (0.1,))
        with self.assertRaises(ValueError):
            PredictiveAssessment(Advisory.NONE, 1.1, 1)


if __name__ == "__main__":
    unittest.main()
