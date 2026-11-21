import unittest

import numpy as np

from randomized_intervention_simulator import (
    ARMS, assert_q0_negative_control, fit_alert_threshold,
    make_randomized_assignment, run_episode,
)


def _scores(context, horizon, seed):
    # Deterministic, context-only test double. Production tests use trained models.
    assert isinstance(context, np.ndarray)
    assert context.ndim == 1
    assert horizon in (1, 2, 4, 8)
    return 1.0


def _thresholds():
    return {
        model: {
            horizon: {
                step: {"cutoff_score": 0.0, "tie_cutoff": 0.0}
                for step in range(9 - horizon)
            }
            for horizon in (1, 2, 4, 8)
        }
        for model in ("jepa", "supervised")
    }


def _spec(**overrides):
    base = {
        "episode_id": "unit-test-episode",
        "seed": 20261120,
        "archetype": "retrying_adversary",
        "family": "iam",
        "rho": 0.85,
        "q": 0.5,
        "horizon": 1,
        "arm": "ucon_only",
    }
    return dict(base, **overrides)


class RandomizedInterventionSimulatorTests(unittest.TestCase):
    def test_assignment_is_reproducible_and_balanced_within_every_stratum(self):
        kwargs = dict(
            seeds=[11], archetypes=["goal_directed"], families=["iam"],
            rhos=[0.85], qs=[0.0, 0.5], horizons=[1, 4],
            episodes_per_stratum=40, assignment_seed=91,
        )
        first = make_randomized_assignment(**kwargs)
        second = make_randomized_assignment(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 160)
        for stratum in {r["stratum"] for r in first}:
            counts = {arm: 0 for arm in ARMS}
            for row in first:
                if row["stratum"] == stratum:
                    counts[row["arm"]] += 1
            self.assertEqual(set(counts.values()), {10})

    def test_assignment_rejects_unbalanced_arm_counts(self):
        with self.assertRaises(ValueError):
            make_randomized_assignment(
                seeds=[1], archetypes=["uniform"], families=["data"],
                episodes_per_stratum=10,
            )

    def test_calibration_threshold_has_deterministic_tie_break(self):
        scores = np.ones(100)
        ids = np.asarray([f"cal-{i}" for i in range(100)])
        steps = np.zeros(100, dtype=int)
        threshold = fit_alert_threshold(scores, ids, steps, alert_rate=0.10)
        fired = sum(
            1 for i in range(100)
            if (
                scores[i] > threshold["cutoff_score"]
                or (
                    scores[i] == threshold["cutoff_score"]
                    and __import__("randomized_intervention_simulator")._tie_uniform(
                        0, ids[i], 0, "threshold-tie"
                    ) <= threshold["tie_cutoff"]
                )
            )
        )
        self.assertEqual(fired, 10)
        self.assertEqual(threshold, fit_alert_threshold(scores, ids, steps, alert_rate=0.10))

    def test_q0_is_exact_negative_control_for_both_predictive_arms(self):
        assert_q0_negative_control(
            _spec(q=0.0), score_functions={"jepa": _scores, "supervised": _scores},
            thresholds=_thresholds(),
        )

    def test_q1_revises_every_warned_proposal_but_still_enforces_ucon(self):
        result = run_episode(
            _spec(q=1.0, arm="jepa_advisory"),
            score_functions={"jepa": _scores},
            thresholds=_thresholds(),
        )
        self.assertEqual(result["warning_count"], 8)
        self.assertEqual(result["revision_count"], 8)
        self.assertEqual(len(result["records"]), 8)
        for row in result["records"]:
            self.assertNotEqual(row["original_action"], row["proposed_action"])
        self.assertEqual(result["state_violation_count"], 0)

    def test_horizon_limits_warning_eligibility(self):
        for horizon, expected in ((1, 8), (2, 7), (4, 5), (8, 1)):
            result = run_episode(
                _spec(q=1.0, horizon=horizon, arm="jepa_advisory"),
                score_functions={"jepa": _scores},
                thresholds=_thresholds(),
            )
            self.assertEqual(result["warning_count"], expected)
            self.assertEqual(result["revision_count"], expected)

    def test_random_warning_arm_is_reproducible(self):
        spec = _spec(arm="random_advisory")
        a, b = run_episode(spec), run_episode(spec)
        self.assertEqual(a, b)

    def test_every_proposal_is_checked_and_deny_does_not_mutate_protected_state(self):
        # Repeated adversarial IAM proposals are expected to be denied; DENY
        # must not change role-grant state and is not itself a state violation.
        result = run_episode(_spec(q=0.0))
        self.assertEqual(len(result["records"]), 8)
        self.assertEqual(result["state_violation_count"], 0)
        self.assertTrue(all("ucon_allowed" in row for row in result["records"]))

    def test_predictor_callback_receives_only_current_context_and_horizon(self):
        seen = []
        def instrumented(context, horizon, seed):
            seen.append((context.copy(), horizon, seed))
            return 1.0
        run_episode(
            _spec(arm="jepa_advisory", horizon=4),
            score_functions={"jepa": instrumented},
            thresholds=_thresholds(),
        )
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(x[0].shape == seen[0][0].shape for x in seen))
        self.assertTrue(all(x[1] == 4 for x in seen))


if __name__ == "__main__":
    unittest.main()
